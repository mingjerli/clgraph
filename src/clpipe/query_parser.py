"""
Recursive query parser for SQL statements.

Parses SQL queries recursively to identify all query units (CTEs, subqueries, main query)
and builds a QueryUnitGraph representing the query structure.
"""

from typing import Optional

import sqlglot
from sqlglot import exp

from .models import QueryUnit, QueryUnitGraph, QueryUnitType


class RecursiveQueryParser:
    """
    Recursively parse SQL query to identify all query units.
    """

    def __init__(self, sql_query: str, dialect: str = "bigquery"):
        self.sql_query = sql_query
        self.dialect = dialect
        self.parsed = sqlglot.parse_one(sql_query, read=dialect)
        self.unit_graph = QueryUnitGraph()
        self.subquery_counter = 0

    def parse(self) -> QueryUnitGraph:
        """
        Main entry point: parse entire query and return QueryUnitGraph.
        """
        # Start with main query
        assert isinstance(self.parsed, exp.Select), "Parsed query must be a SELECT statement"
        self._parse_select_unit(
            select_node=self.parsed,
            unit_type=QueryUnitType.MAIN_QUERY,
            name="main",
            parent_unit=None,
            depth=0,
        )

        return self.unit_graph

    def _parse_select_unit(
        self,
        select_node: exp.Select,
        unit_type: QueryUnitType,
        name: str,
        parent_unit: Optional[QueryUnit],
        depth: int,
    ) -> QueryUnit:
        """
        Recursively parse a SELECT statement and all its nested queries.
        This is the core recursive method.
        """
        # Create QueryUnit for this SELECT
        unit_id = self._generate_unit_id(unit_type, name)
        unit = QueryUnit(
            unit_id=unit_id,
            unit_type=unit_type,
            name=name,
            select_node=select_node,
            parent_unit=parent_unit,
            depth=depth,
        )

        # 1. Parse CTEs first (they're available to this SELECT)
        if hasattr(select_node, "ctes") and select_node.ctes:
            for cte in select_node.ctes:
                if isinstance(cte, exp.CTE):
                    cte_name = cte.alias_or_name
                    cte_select = cte.this

                    if isinstance(cte_select, exp.Select):
                        # Recursively parse CTE
                        self._parse_select_unit(
                            select_node=cte_select,
                            unit_type=QueryUnitType.CTE,
                            name=cte_name,
                            parent_unit=unit,
                            depth=depth + 1,
                        )

        # 2. Parse FROM clause (may contain subqueries or CTEs)
        from_clause = select_node.args.get("from")
        if from_clause:
            self._parse_from_sources(from_clause, unit, depth)

        # 3. Parse JOIN clauses (may contain subqueries)
        joins = select_node.args.get("joins", [])
        for join in joins:
            self._parse_from_sources(join, unit, depth)

        # 4. Parse WHERE clause (may contain subqueries)
        where_clause = select_node.args.get("where")
        if where_clause:
            self._parse_where_subqueries(where_clause, unit, depth)

        # 5. Parse HAVING clause (may contain subqueries)
        having_clause = select_node.args.get("having")
        if having_clause:
            self._parse_having_subqueries(having_clause, unit, depth)

        # 6. Parse SELECT expressions (may contain scalar subqueries)
        for expr in select_node.expressions:
            self._parse_select_subqueries(expr, unit, depth)

        # 7. Validate star usage (after parsing FROM/JOINs so we know table count)
        self._validate_star_usage(unit, select_node)

        # Add unit to graph
        self.unit_graph.add_unit(unit)

        return unit

    def _parse_from_sources(self, from_node: exp.Expression, parent_unit: QueryUnit, depth: int):
        """
        Parse FROM/JOIN clause, which may contain:
        - Base tables
        - CTEs
        - Subqueries (derived tables)

        Note: We need to extract table sources from FROM and JOIN clauses only,
        not from the entire subtree (which would include column references).

        Also captures alias mappings for proper column reference resolution.
        """

        # Helper to process a single table source
        def process_table_source(source_node):
            if isinstance(source_node, exp.Table):
                # Get the actual table name (not alias)
                table_name = (
                    source_node.this.name if hasattr(source_node.this, "name") else source_node.name
                )

                # Get the alias (if any)
                alias = (
                    source_node.alias
                    if hasattr(source_node, "alias") and source_node.alias
                    else None
                )

                # Check if this is a CTE reference
                cte_unit = self.unit_graph.get_unit_by_name(table_name)
                if cte_unit:
                    # Reference to CTE
                    if cte_unit.unit_id not in parent_unit.depends_on_units:
                        parent_unit.depends_on_units.append(cte_unit.unit_id)

                    # Store alias mapping: alias -> (actual_name, is_unit=True)
                    if alias:
                        parent_unit.alias_mapping[alias] = (table_name, True)
                    # Also map the actual name to itself for unqualified references
                    parent_unit.alias_mapping[table_name] = (table_name, True)
                else:
                    # Base table
                    if table_name not in parent_unit.depends_on_tables:
                        parent_unit.depends_on_tables.append(table_name)

                    # Store alias mapping: alias -> (actual_name, is_unit=False)
                    if alias:
                        parent_unit.alias_mapping[alias] = (table_name, False)
                    # Also map the actual name to itself
                    parent_unit.alias_mapping[table_name] = (table_name, False)

            elif isinstance(source_node, exp.Subquery):
                # Subquery in FROM clause
                subquery_select = source_node.this
                if isinstance(subquery_select, exp.Select):
                    # Use alias if provided, otherwise generate a name like "subquery_0"
                    subquery_name = (
                        source_node.alias_or_name
                        if (hasattr(source_node, "alias") and source_node.alias_or_name)
                        else f"subquery_{self.subquery_counter}"
                    )
                    self.subquery_counter += 1

                    # Recursively parse subquery
                    subquery_unit = self._parse_select_unit(
                        select_node=subquery_select,
                        unit_type=QueryUnitType.SUBQUERY_FROM,
                        name=subquery_name,
                        parent_unit=parent_unit,
                        depth=depth + 1,
                    )

                    # Add dependency (avoid duplicates)
                    if subquery_unit.unit_id not in parent_unit.depends_on_units:
                        parent_unit.depends_on_units.append(subquery_unit.unit_id)

                    # Store alias mapping for subquery
                    parent_unit.alias_mapping[subquery_name] = (subquery_name, True)

        # Process the main FROM source
        if isinstance(from_node, (exp.Table, exp.Subquery)):
            process_table_source(from_node)
        elif hasattr(from_node, "this"):
            # FROM clause with this attribute
            process_table_source(from_node.this)

        # Process JOIN clauses
        # JOINs are stored in the 'joins' attribute
        if hasattr(from_node, "args") and "joins" in from_node.args:
            joins = from_node.args["joins"]
            if joins:
                for join in joins:
                    # Each join has a 'this' which is the table/subquery being joined
                    if hasattr(join, "this"):
                        process_table_source(join.this)

    def _parse_where_subqueries(
        self, where_node: exp.Expression, parent_unit: QueryUnit, depth: int
    ):
        """Parse subqueries in WHERE clause"""
        for node in where_node.walk():
            if isinstance(node, exp.Subquery):
                subquery_select = node.this
                if isinstance(subquery_select, exp.Select):
                    subquery_name = f"where_subq_{self.subquery_counter}"
                    self.subquery_counter += 1

                    # Recursively parse
                    subquery_unit = self._parse_select_unit(
                        select_node=subquery_select,
                        unit_type=QueryUnitType.SUBQUERY_WHERE,
                        name=subquery_name,
                        parent_unit=parent_unit,
                        depth=depth + 1,
                    )

                    parent_unit.depends_on_units.append(subquery_unit.unit_id)

    def _parse_having_subqueries(
        self, having_node: exp.Expression, parent_unit: QueryUnit, depth: int
    ):
        """Parse subqueries in HAVING clause"""
        for node in having_node.walk():
            if isinstance(node, exp.Subquery):
                subquery_select = node.this
                if isinstance(subquery_select, exp.Select):
                    subquery_name = f"having_subq_{self.subquery_counter}"
                    self.subquery_counter += 1

                    # Recursively parse
                    subquery_unit = self._parse_select_unit(
                        select_node=subquery_select,
                        unit_type=QueryUnitType.SUBQUERY_HAVING,
                        name=subquery_name,
                        parent_unit=parent_unit,
                        depth=depth + 1,
                    )

                    parent_unit.depends_on_units.append(subquery_unit.unit_id)

    def _parse_select_subqueries(self, expr: exp.Expression, parent_unit: QueryUnit, depth: int):
        """Parse scalar subqueries in SELECT clause"""
        for node in expr.walk():
            if isinstance(node, exp.Subquery):
                subquery_select = node.this
                if isinstance(subquery_select, exp.Select):
                    subquery_name = f"select_subq_{self.subquery_counter}"
                    self.subquery_counter += 1

                    # Recursively parse
                    subquery_unit = self._parse_select_unit(
                        select_node=subquery_select,
                        unit_type=QueryUnitType.SUBQUERY_SELECT,
                        name=subquery_name,
                        parent_unit=parent_unit,
                        depth=depth + 1,
                    )

                    parent_unit.depends_on_units.append(subquery_unit.unit_id)

    def _validate_star_usage(self, unit: QueryUnit, select_node: exp.Select):
        """
        Validate that star notation is used correctly.

        Rule: Unqualified SELECT * with multiple tables (JOINs) is ambiguous.
        Must use qualified stars like u.*, o.* instead.
        """
        # Check if there's an unqualified star in SELECT
        has_unqualified_star = False
        for expr in select_node.expressions:
            if isinstance(expr, exp.Star):
                has_unqualified_star = True
                break
            elif isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star):
                # Check if it's qualified (has table prefix)
                if not (hasattr(expr, "table") and expr.table):
                    has_unqualified_star = True
                    break

        if not has_unqualified_star:
            return  # No issue

        # Count total tables/units this query references
        table_count = len(unit.depends_on_tables) + len(unit.depends_on_units)

        if table_count > 1:
            # Ambiguous star usage
            context = f"{unit.unit_type.value}"
            if unit.name and unit.unit_type == QueryUnitType.CTE:
                context = f"CTE '{unit.name}'"

            raise ValueError(
                f"Ambiguous SELECT * in {context} with {table_count} tables. "
                f"Use qualified stars like 'alias.*' instead."
            )

    def _generate_unit_id(self, unit_type: QueryUnitType, name: str) -> str:
        """Generate unique unit ID"""
        if unit_type == QueryUnitType.MAIN_QUERY:
            return "main"
        elif unit_type == QueryUnitType.CTE:
            return f"cte:{name}"
        else:
            return f"subq:{name}"


__all__ = ["RecursiveQueryParser"]
