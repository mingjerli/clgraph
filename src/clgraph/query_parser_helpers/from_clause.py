"""FROM clause parsing helper.

Handles parsing of FROM clause sources including tables, subqueries,
and delegates to SpecialSourcesHandler for UNNEST, TVF, VALUES, and LATERAL.
"""

from typing import TYPE_CHECKING, List, Optional

from sqlglot import exp

from ..models import QueryUnitType
from .special_sources import SpecialSourcesHandler

if TYPE_CHECKING:
    from ..models import QueryUnit
    from ..query_parser import RecursiveQueryParser


class FromClauseParser:
    """Helper for parsing FROM clause sources."""

    def __init__(self, parser: "RecursiveQueryParser"):
        """Initialize with reference to parent parser.

        Args:
            parser: The parent RecursiveQueryParser instance
        """
        self._parser = parser
        # Initialize sub-helper for special sources
        self._special_sources = SpecialSourcesHandler(parser)

    def parse(self, from_node: exp.Expression, parent_unit: "QueryUnit", depth: int):
        """Parse FROM/JOIN clause sources.

        Handles:
        - Base tables
        - CTEs
        - Subqueries (derived tables)
        - UNNEST/FLATTEN/EXPLODE expressions (array expansion)
        - Table-Valued Functions (TVFs)
        - VALUES clauses
        - LATERAL subqueries
        - PIVOT/UNPIVOT operations

        Args:
            from_node: The FROM expression node
            parent_unit: The parent QueryUnit
            depth: Current recursion depth
        """
        # Get preceding tables from alias_mapping for LATERAL correlation detection
        preceding_tables: List[str] = list(parent_unit.alias_mapping.keys())
        preceding_tables.extend(parent_unit.depends_on_tables)

        # Process the main FROM source
        if isinstance(from_node, (exp.Table, exp.Subquery)):
            self._process_table_source(from_node, parent_unit, depth)
            source_name = self._get_source_name(from_node)
            if source_name and source_name not in preceding_tables:
                preceding_tables.append(source_name)
        elif isinstance(from_node, exp.Unnest):
            self._special_sources.process_unnest(from_node, parent_unit)
        elif isinstance(from_node, exp.Lateral):
            self._special_sources.process_lateral_subquery(
                from_node, parent_unit, preceding_tables, depth
            )
        elif isinstance(from_node, exp.Join):
            self._process_join_source(from_node, parent_unit, preceding_tables, depth)
        elif hasattr(from_node, "this"):
            self._process_from_this(from_node, parent_unit, preceding_tables, depth)

        # Process JOIN clauses
        if hasattr(from_node, "args") and "joins" in from_node.args:
            joins = from_node.args["joins"]
            if joins:
                for join in joins:
                    if hasattr(join, "this"):
                        join_source = join.this
                        if isinstance(join_source, exp.Unnest):
                            self._special_sources.process_unnest(join_source, parent_unit)
                        elif isinstance(join_source, exp.Lateral):
                            self._special_sources.process_lateral_subquery(
                                join_source, parent_unit, preceding_tables, depth
                            )
                        else:
                            self._process_table_source(join_source, parent_unit, depth)
                            source_name = self._get_source_name(join_source)
                            if source_name and source_name not in preceding_tables:
                                preceding_tables.append(source_name)

        # Scan for nested UNNEST and LATERAL expressions
        self._scan_for_nested_sources(from_node, parent_unit, preceding_tables, depth)

    def _process_join_source(
        self,
        from_node: exp.Join,
        parent_unit: "QueryUnit",
        preceding_tables: List[str],
        depth: int,
    ):
        """Process a JOIN source."""
        if hasattr(from_node, "this"):
            join_source = from_node.this
            if isinstance(join_source, exp.Lateral):
                self._special_sources.process_lateral_subquery(
                    join_source, parent_unit, preceding_tables, depth
                )
            elif isinstance(join_source, exp.Unnest):
                self._special_sources.process_unnest(join_source, parent_unit)
            else:
                self._process_table_source(join_source, parent_unit, depth)
                source_name = self._get_source_name(join_source)
                if source_name and source_name not in preceding_tables:
                    preceding_tables.append(source_name)

    def _process_from_this(
        self,
        from_node: exp.Expression,
        parent_unit: "QueryUnit",
        preceding_tables: List[str],
        depth: int,
    ):
        """Process FROM clause with 'this' attribute."""
        if isinstance(from_node.this, exp.Unnest):
            self._special_sources.process_unnest(from_node.this, parent_unit)
        elif isinstance(from_node.this, exp.Lateral):
            self._special_sources.process_lateral_subquery(
                from_node.this, parent_unit, preceding_tables, depth
            )
        else:
            self._process_table_source(from_node.this, parent_unit, depth)
            source_name = self._get_source_name(from_node.this)
            if source_name and source_name not in preceding_tables:
                preceding_tables.append(source_name)

    def _process_table_source(
        self, source_node: exp.Expression, parent_unit: "QueryUnit", depth: int
    ):
        """Process a single table source."""
        # Check for VALUES clause directly in FROM
        if isinstance(source_node, exp.Values):
            self._special_sources.process_values_direct(source_node, parent_unit)
            return

        if isinstance(source_node, exp.Table):
            # Check if this is a Table-Valued Function (TVF)
            if self._special_sources.is_tvf(source_node):
                self._special_sources.process_tvf(source_node, parent_unit)
                return

            # Check for PIVOT/UNPIVOT operations
            if self._process_pivot_unpivot(source_node, parent_unit, depth):
                return

            # Regular table
            self._process_regular_table(source_node, parent_unit)

        elif isinstance(source_node, exp.Subquery):
            # Check if this is a VALUES clause wrapped in Subquery
            if self._special_sources.handle_values_in_subquery(source_node, parent_unit):
                return

            # Check for PIVOT/UNPIVOT on subquery
            if self._process_pivot_unpivot(source_node, parent_unit, depth):
                return

            # Regular subquery
            self._process_subquery(source_node, parent_unit, depth)

    def _process_pivot_unpivot(
        self, source_node: exp.Expression, parent_unit: "QueryUnit", depth: int
    ) -> bool:
        """Process PIVOT/UNPIVOT operations if present.

        Returns:
            True if PIVOT/UNPIVOT was processed, False otherwise
        """
        has_pivot = (
            hasattr(source_node, "args")
            and "pivots" in source_node.args
            and source_node.args["pivots"]
        )

        if not has_pivot:
            return False

        for pivot_node in source_node.args["pivots"]:
            if isinstance(pivot_node, exp.Pivot):
                is_unpivot = pivot_node.args.get("unpivot", False)

                pivot_name = (
                    source_node.alias
                    if hasattr(source_node, "alias") and source_node.alias
                    else f"pivot_{self._parser.subquery_counter}"
                )
                self._parser.subquery_counter += 1

                if is_unpivot:
                    pivot_unit = self._parser._parse_unpivot(
                        unpivot_node=pivot_node,
                        name=pivot_name,
                        parent_unit=parent_unit,
                        depth=depth + 1,
                        table_node=source_node,
                    )
                else:
                    pivot_unit = self._parser._parse_pivot(
                        pivot_node=pivot_node,
                        name=pivot_name,
                        parent_unit=parent_unit,
                        depth=depth + 1,
                        table_node=source_node,
                    )

                if pivot_unit.unit_id not in parent_unit.depends_on_units:
                    parent_unit.depends_on_units.append(pivot_unit.unit_id)

                parent_unit.alias_mapping[pivot_name] = (pivot_name, True)

        return True

    def _process_regular_table(self, source_node: exp.Table, parent_unit: "QueryUnit"):
        """Process a regular table reference (not TVF, not PIVOT)."""
        table_name = (
            source_node.this.name if hasattr(source_node.this, "name") else source_node.name
        )

        alias = source_node.alias if hasattr(source_node, "alias") and source_node.alias else None

        # Check if this is a CTE reference
        cte_unit = self._parser.unit_graph.get_unit_by_name(table_name)
        if cte_unit:
            if cte_unit.unit_id not in parent_unit.depends_on_units:
                parent_unit.depends_on_units.append(cte_unit.unit_id)
            if alias:
                parent_unit.alias_mapping[alias] = (table_name, True)
            parent_unit.alias_mapping[table_name] = (table_name, True)
        else:
            if table_name not in parent_unit.depends_on_tables:
                parent_unit.depends_on_tables.append(table_name)
            if alias:
                parent_unit.alias_mapping[alias] = (table_name, False)
            parent_unit.alias_mapping[table_name] = (table_name, False)

    def _process_subquery(self, source_node: exp.Subquery, parent_unit: "QueryUnit", depth: int):
        """Process a subquery in FROM clause."""
        subquery_select = source_node.this
        if isinstance(subquery_select, exp.Select):
            subquery_name = (
                source_node.alias_or_name
                if hasattr(source_node, "alias") and source_node.alias_or_name
                else f"subquery_{self._parser.subquery_counter}"
            )
            self._parser.subquery_counter += 1

            subquery_unit = self._parser._parse_select_unit(
                select_node=subquery_select,
                unit_type=QueryUnitType.SUBQUERY_FROM,
                name=subquery_name,
                parent_unit=parent_unit,
                depth=depth + 1,
            )

            if subquery_unit.unit_id not in parent_unit.depends_on_units:
                parent_unit.depends_on_units.append(subquery_unit.unit_id)

            parent_unit.alias_mapping[subquery_name] = (subquery_name, True)

    def _get_source_name(self, source_node: exp.Expression) -> Optional[str]:
        """Get the name or alias of a table source."""
        if isinstance(source_node, exp.Table):
            if hasattr(source_node, "alias") and source_node.alias:
                return str(source_node.alias)
            return source_node.name
        elif isinstance(source_node, exp.Subquery):
            if hasattr(source_node, "alias") and source_node.alias:
                return str(source_node.alias)
        return None

    def _scan_for_nested_sources(
        self,
        from_node: exp.Expression,
        parent_unit: "QueryUnit",
        preceding_tables: List[str],
        depth: int,
    ):
        """Scan for UNNEST and LATERAL expressions nested in the FROM clause."""
        for node in from_node.walk():
            if isinstance(node, exp.Unnest):
                # Check if not already processed
                alias_node = node.args.get("alias")
                if alias_node:
                    alias = None
                    if hasattr(alias_node, "columns") and alias_node.columns:
                        alias = alias_node.columns[0].name
                    elif hasattr(alias_node, "this"):
                        alias = (
                            alias_node.this.name
                            if hasattr(alias_node.this, "name")
                            else str(alias_node.this)
                        )
                    if alias and alias not in parent_unit.unnest_sources:
                        self._special_sources.process_unnest(node, parent_unit)
            elif isinstance(node, exp.Lateral):
                if isinstance(node.this, exp.Explode):
                    # Snowflake LATERAL FLATTEN
                    if hasattr(node, "alias") and node.alias:
                        alias = None
                        if hasattr(node.alias, "this"):
                            alias = (
                                node.alias.this.name
                                if hasattr(node.alias.this, "name")
                                else str(node.alias.this)
                            )
                        if alias and alias not in parent_unit.unnest_sources:
                            self._special_sources.process_lateral_flatten(node, parent_unit)
                elif isinstance(node.this, exp.Subquery):
                    alias = None
                    if hasattr(node, "alias") and node.alias:
                        if hasattr(node.alias, "this"):
                            alias = (
                                node.alias.this.name
                                if hasattr(node.alias.this, "name")
                                else str(node.alias.this)
                            )
                        else:
                            alias = str(node.alias)
                    if alias and alias not in parent_unit.lateral_sources:
                        self._special_sources.process_lateral_subquery(
                            node, parent_unit, preceding_tables, depth
                        )
