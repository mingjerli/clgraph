"""Recursive CTE parsing helper.

Handles parsing of WITH RECURSIVE common table expressions.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

from sqlglot import exp

from ..models import QueryUnit, QueryUnitType, RecursiveCTEInfo

if TYPE_CHECKING:
    from ..query_parser import RecursiveQueryParser


class RecursiveCTEParser:
    """Helper for parsing recursive CTEs."""

    def __init__(self, parser: "RecursiveQueryParser"):
        """Initialize with reference to parent parser.

        Args:
            parser: The parent RecursiveQueryParser instance
        """
        self._parser = parser

    def is_recursive_cte(self, query: exp.Expression, cte_name: str) -> bool:
        """
        Check if a CTE is recursive (references itself).

        A recursive CTE:
        1. Has a UNION/UNION ALL structure
        2. The right side of the UNION references the CTE name

        Args:
            query: The CTE query expression
            cte_name: Name of the CTE

        Returns:
            True if the CTE is self-referencing
        """
        # Recursive CTEs must be UNION expressions
        if not isinstance(query, exp.Union):
            return False

        # Check if the right side (recursive part) references the CTE name
        right_side = query.expression  # Right side of UNION
        if right_side is None:
            return False

        # Look for table references to the CTE name
        for table in right_side.find_all(exp.Table):
            table_name = table.name
            if table_name and table_name.lower() == cte_name.lower():
                return True

        return False

    def parse(
        self,
        cte: exp.CTE,
        cte_name: str,
        parent_unit: Optional[QueryUnit],
        depth: int,
    ) -> QueryUnit:
        """
        Parse a recursive CTE into base and recursive components.

        A recursive CTE has the form:
            WITH RECURSIVE cte_name AS (
                <base_case>          -- Anchor/initial rows
                UNION [ALL]
                <recursive_case>     -- References cte_name
            )

        Args:
            cte: The CTE expression node
            cte_name: Name of the CTE
            parent_unit: Parent query unit
            depth: Nesting depth

        Returns:
            QueryUnit representing the recursive CTE
        """
        union_expr = cte.this  # Should be exp.Union

        # Split into base and recursive cases
        base_query = union_expr.this  # Left side (base case)
        recursive_query = union_expr.expression  # Right side (recursive case)

        # Determine union type (UNION vs UNION ALL)
        # In sqlglot, Union.args.get("distinct") is True for UNION DISTINCT
        is_distinct = union_expr.args.get("distinct", False)
        union_type = "union" if is_distinct else "union_all"

        # Parse base case
        base_unit = None
        if isinstance(base_query, exp.Select):
            base_unit = self._parser._parse_select_unit(
                select_node=base_query,
                unit_type=QueryUnitType.CTE_BASE,
                name=f"{cte_name}_base",
                parent_unit=parent_unit,
                depth=depth + 1,
            )
        elif isinstance(base_query, exp.Subquery):
            # Handle parenthesized base case
            inner = base_query.this
            if isinstance(inner, exp.Select):
                base_unit = self._parser._parse_select_unit(
                    select_node=inner,
                    unit_type=QueryUnitType.CTE_BASE,
                    name=f"{cte_name}_base",
                    parent_unit=parent_unit,
                    depth=depth + 1,
                )

        # Find self-reference info before parsing recursive case
        self_ref_info = self._find_self_reference(recursive_query, cte_name)

        # Parse recursive case
        recursive_unit = None
        if isinstance(recursive_query, exp.Select):
            recursive_unit = self._parser._parse_select_unit(
                select_node=recursive_query,
                unit_type=QueryUnitType.CTE_RECURSIVE,
                name=f"{cte_name}_recursive",
                parent_unit=parent_unit,
                depth=depth + 1,
            )
            # Mark that this unit references the recursive CTE
            recursive_unit.is_recursive_reference = True
        elif isinstance(recursive_query, exp.Subquery):
            inner = recursive_query.this
            if isinstance(inner, exp.Select):
                recursive_unit = self._parser._parse_select_unit(
                    select_node=inner,
                    unit_type=QueryUnitType.CTE_RECURSIVE,
                    name=f"{cte_name}_recursive",
                    parent_unit=parent_unit,
                    depth=depth + 1,
                )
                recursive_unit.is_recursive_reference = True

        # Extract column names from base and recursive cases
        base_columns = self._extract_select_column_names(base_query)
        recursive_columns = self._extract_select_column_names(recursive_query)

        # Create the main CTE unit
        unit_id = self._parser._generate_unit_id(QueryUnitType.CTE, cte_name)
        cte_unit = QueryUnit(
            unit_id=unit_id,
            unit_type=QueryUnitType.CTE,
            name=cte_name,
            select_node=None,  # Recursive CTEs have no single select_node
            parent_unit=parent_unit,
            depth=depth + 1,
        )

        # Store recursive CTE info
        cte_unit.recursive_cte_info = RecursiveCTEInfo(
            cte_name=cte_name,
            is_recursive=True,
            base_columns=base_columns,
            recursive_columns=recursive_columns,
            union_type=union_type,
            self_reference_alias=self_ref_info.get("alias"),
            join_condition=self_ref_info.get("join_condition"),
        )

        # Add dependencies
        if base_unit:
            cte_unit.depends_on_units.append(base_unit.unit_id)
        if recursive_unit:
            cte_unit.depends_on_units.append(recursive_unit.unit_id)

        # Add set operation info
        cte_unit.set_operation_type = union_type
        if base_unit:
            cte_unit.set_operation_branches.append(base_unit.unit_id)
        if recursive_unit:
            cte_unit.set_operation_branches.append(recursive_unit.unit_id)

        # Add to graph
        self._parser.unit_graph.add_unit(cte_unit)

        return cte_unit

    def _find_self_reference(
        self, query: exp.Expression, cte_name: str
    ) -> Dict[str, Optional[str]]:
        """
        Find where the recursive query references the CTE itself.

        Args:
            query: The recursive query expression
            cte_name: Name of the CTE

        Returns:
            Dictionary with 'alias' and 'join_condition' keys
        """
        result: Dict[str, Optional[str]] = {"alias": None, "join_condition": None}

        # Handle Subquery wrapper
        if isinstance(query, exp.Subquery):
            query = query.this

        if not isinstance(query, exp.Select):
            return result

        # Find table reference to the CTE
        for table in query.find_all(exp.Table):
            table_name = table.name
            if table_name and table_name.lower() == cte_name.lower():
                # Get alias
                alias = str(table.alias) if table.alias else cte_name
                result["alias"] = alias

                # Find join condition (look in JOIN ON clauses)
                join_condition = self._find_join_condition_for_alias(query, alias)
                result["join_condition"] = join_condition
                break

        return result

    def _find_join_condition_for_alias(self, query: exp.Select, alias: str) -> Optional[str]:
        """
        Find the JOIN condition for a given table alias.

        Args:
            query: The SELECT query
            alias: The table alias to find

        Returns:
            JOIN condition as SQL string, or None if not found
        """
        joins = query.args.get("joins", [])
        for join in joins:
            # Check if this join involves our alias
            join_table = join.this
            if isinstance(join_table, exp.Table):
                join_alias = str(join_table.alias) if join_table.alias else join_table.name
                if join_alias and join_alias.lower() == alias.lower():
                    # Found the join - extract ON condition
                    on_condition = join.args.get("on")
                    if on_condition:
                        return on_condition.sql()
        return None

    def _extract_select_column_names(self, query: exp.Expression) -> List[str]:
        """
        Extract output column names from a SELECT query.

        Args:
            query: The SELECT query expression

        Returns:
            List of column names/aliases
        """
        columns: List[str] = []

        # Handle Subquery wrapper
        if isinstance(query, exp.Subquery):
            query = query.this

        if not isinstance(query, exp.Select):
            return columns

        for expr in query.expressions:
            if isinstance(expr, exp.Alias):
                # Aliased expression: SELECT x AS y
                columns.append(expr.alias)
            elif isinstance(expr, exp.Column):
                # Column reference: SELECT x
                columns.append(expr.name)
            elif isinstance(expr, exp.Star):
                # Star: SELECT *
                columns.append("*")
            else:
                # Other expression - try to get output name
                # For literals, functions, etc. without alias, use string representation
                columns.append(str(expr)[:50])  # Truncate for safety

        return columns
