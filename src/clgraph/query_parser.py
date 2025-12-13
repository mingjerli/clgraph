"""
Recursive query parser for SQL statements.

Parses SQL queries recursively to identify all query units (CTEs, subqueries, main query)
and builds a QueryUnitGraph representing the query structure.
"""

from typing import List, Optional, Union

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

        Handles both single SELECT queries and set operations (UNION/INTERSECT/EXCEPT).
        """
        # Handle different top-level query types
        if isinstance(self.parsed, exp.Select):
            # Single SELECT query
            self._parse_select_unit(
                select_node=self.parsed,
                unit_type=QueryUnitType.MAIN_QUERY,
                name="main",
                parent_unit=None,
                depth=0,
            )
        elif isinstance(self.parsed, exp.Union):
            # UNION query
            self._parse_set_operation(
                set_node=self.parsed,
                operation_type="union",
                name="main",
                parent_unit=None,
                depth=0,
            )
        elif isinstance(self.parsed, exp.Intersect):
            # INTERSECT query
            self._parse_set_operation(
                set_node=self.parsed,
                operation_type="intersect",
                name="main",
                parent_unit=None,
                depth=0,
            )
        elif isinstance(self.parsed, exp.Except):
            # EXCEPT query
            self._parse_set_operation(
                set_node=self.parsed,
                operation_type="except",
                name="main",
                parent_unit=None,
                depth=0,
            )
        else:
            raise ValueError(
                f"Unsupported top-level query type: {type(self.parsed).__name__}. "
                f"Expected Select, Union, Intersect, or Except."
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
        # Note: sqlglot >=28.0.0 uses "from_" instead of "from" (Python keyword)
        from_clause = select_node.args.get("from_") or select_node.args.get("from")
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

    def _parse_set_operation(
        self,
        set_node: Union[exp.Union, exp.Intersect, exp.Except],
        operation_type: str,
        name: str,
        parent_unit: Optional[QueryUnit] = None,
        depth: int = 0,
    ) -> QueryUnit:
        """
        Parse UNION/INTERSECT/EXCEPT set operations.

        Set operations combine results from multiple SELECT statements.
        Each branch is parsed as a separate query unit.

        Args:
            set_node: The set operation node (Union, Intersect, or Except)
            operation_type: Type of operation ("union", "intersect", "except")
            name: Name for this set operation unit
            parent_unit: Parent query unit (if nested)
            depth: Nesting depth

        Returns:
            QueryUnit representing the set operation

        sqlglot Structure:
            - Union.this = left SELECT
            - Union.expression = right SELECT
            - Union.distinct = True if UNION (not UNION ALL)
        """
        # Determine unit type based on operation
        unit_type_map = {
            "union": QueryUnitType.UNION,
            "intersect": QueryUnitType.INTERSECT,
            "except": QueryUnitType.EXCEPT,
        }
        unit_type = unit_type_map[operation_type]

        # Determine specific operation variant (e.g., UNION vs UNION ALL)
        if operation_type == "union":
            # Check if DISTINCT is explicitly set (UNION DISTINCT vs UNION ALL)
            is_distinct = set_node.args.get("distinct", False)
            set_op_variant = "union" if is_distinct else "union_all"
        else:
            set_op_variant = operation_type

        # Create unit for the set operation itself
        unit_id = self._generate_unit_id(unit_type, name)
        unit = QueryUnit(
            unit_id=unit_id,
            unit_type=unit_type,
            name=name,
            select_node=None,  # Set operations don't have a select_node
            parent_unit=parent_unit,
            depth=depth,
            set_operation_type=set_op_variant,
            set_operation_branches=[],
        )

        # Collect all SELECT branches (handles nested set operations)
        branches = self._collect_set_operation_branches(set_node, operation_type)

        # Parse each branch as a separate query unit
        for idx, branch_select in enumerate(branches):
            branch_name = f"{name}_branch_{idx}"
            branch_unit = self._parse_select_unit(
                select_node=branch_select,
                unit_type=QueryUnitType.SUBQUERY_UNION,
                name=branch_name,
                parent_unit=unit,
                depth=depth + 1,
            )

            # Track branch in set operation
            unit.set_operation_branches.append(branch_unit.unit_id)
            unit.depends_on_units.append(branch_unit.unit_id)

        # Add to graph
        self.unit_graph.add_unit(unit)

        return unit

    def _collect_set_operation_branches(
        self,
        set_node: Union[exp.Union, exp.Intersect, exp.Except],
        operation_type: str,
    ) -> List[exp.Select]:
        """
        Recursively collect all SELECT branches from a set operation.

        Handles nested set operations by flattening them into a list.
        Example: (A UNION B) UNION C → [A, B, C]

        Args:
            set_node: The set operation node
            operation_type: Type of operation ("union", "intersect", "except")

        Returns:
            List of SELECT statements in the set operation
        """
        branches = []

        # Determine the node type we're collecting
        node_class_map = {
            "union": exp.Union,
            "intersect": exp.Intersect,
            "except": exp.Except,
        }
        target_class = node_class_map[operation_type]

        # Process left side (this)
        left_node = set_node.this
        # Handle parenthesized expressions wrapped in Subquery
        if isinstance(left_node, exp.Subquery):
            left_node = left_node.this

        if isinstance(left_node, target_class):
            # Nested set operation - recurse
            branches.extend(self._collect_set_operation_branches(left_node, operation_type))
        elif isinstance(left_node, exp.Select):
            # Base case - SELECT statement
            branches.append(left_node)
        else:
            raise ValueError(f"Unexpected node type in set operation: {type(left_node).__name__}")

        # Process right side (expression)
        right_node = set_node.expression
        # Handle parenthesized expressions wrapped in Subquery
        if isinstance(right_node, exp.Subquery):
            right_node = right_node.this

        if isinstance(right_node, target_class):
            # Nested set operation - recurse
            branches.extend(self._collect_set_operation_branches(right_node, operation_type))
        elif isinstance(right_node, exp.Select):
            # Base case - SELECT statement
            branches.append(right_node)
        else:
            raise ValueError(f"Unexpected node type in set operation: {type(right_node).__name__}")

        return branches

    def _parse_pivot(
        self,
        pivot_node: exp.Pivot,
        name: str,
        parent_unit: QueryUnit,
        depth: int,
        table_node,  # Can be exp.Table or exp.Subquery
    ) -> QueryUnit:
        """
        Parse PIVOT operation.

        PIVOT transforms rows into columns based on pivot values.
        Example: PIVOT(SUM(revenue) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4'))

        In sqlglot, PIVOT is stored as part of Table or Subquery nodes.
        """
        # Create unit for PIVOT operation
        unit_id = self._generate_unit_id(QueryUnitType.PIVOT, name)
        unit = QueryUnit(
            unit_id=unit_id,
            unit_type=QueryUnitType.PIVOT,
            name=name,
            select_node=None,
            parent_unit=parent_unit,
            depth=depth,
        )

        # Extract PIVOT configuration
        pivot_config = {}

        # Get aggregation expressions (e.g., SUM(revenue))
        if hasattr(pivot_node, "expressions") and pivot_node.expressions:
            pivot_config["aggregations"] = [str(expr) for expr in pivot_node.expressions]

        # Get pivot column (the FOR column)
        # In sqlglot, the pivot column is in 'fields' which contains an In expression
        if hasattr(pivot_node, "fields") and pivot_node.fields:
            for field in pivot_node.fields:
                if isinstance(field, exp.In):
                    # The 'this' is the column being pivoted
                    pivot_config["pivot_column"] = str(field.this)

        # Get pivot values (the IN clause values)
        # In sqlglot, columns are stored in args, not as a direct attribute
        if hasattr(pivot_node, "args") and "columns" in pivot_node.args:
            columns = pivot_node.args["columns"]
            if columns:
                pivot_config["value_columns"] = [str(col) for col in columns]

        unit.pivot_config = pivot_config

        # Parse the source
        # table_node can be either a Table or a Subquery
        if isinstance(table_node, exp.Subquery):
            # PIVOT is applied to a subquery: (SELECT ...) PIVOT(...)
            source_select = table_node.this
            if isinstance(source_select, exp.Select):
                source_name = f"{name}_source"
                source_unit = self._parse_select_unit(
                    select_node=source_select,
                    unit_type=QueryUnitType.SUBQUERY_PIVOT_SOURCE,
                    name=source_name,
                    parent_unit=unit,
                    depth=depth + 1,
                )
                unit.depends_on_units.append(source_unit.unit_id)
        elif isinstance(table_node, exp.Table):
            # PIVOT is applied to a table: table_name PIVOT(...)
            table_source = table_node.this

            # Check if it's a subquery or table reference
            if isinstance(table_source, exp.Subquery):
                # Shouldn't happen, but handle it
                source_select = table_source.this
                if isinstance(source_select, exp.Select):
                    source_name = f"{name}_source"
                    source_unit = self._parse_select_unit(
                        select_node=source_select,
                        unit_type=QueryUnitType.SUBQUERY_PIVOT_SOURCE,
                        name=source_name,
                        parent_unit=unit,
                        depth=depth + 1,
                    )
                    unit.depends_on_units.append(source_unit.unit_id)
            else:
                # Source is a base table or CTE
                table_name = (
                    table_source.name if hasattr(table_source, "name") else str(table_source)
                )

                # Check if it's a CTE reference
                cte_unit = self.unit_graph.get_unit_by_name(table_name)
                if cte_unit:
                    unit.depends_on_units.append(cte_unit.unit_id)
                else:
                    unit.depends_on_tables.append(table_name)

        # Add to graph
        self.unit_graph.add_unit(unit)

        return unit

    def _parse_unpivot(
        self,
        unpivot_node: exp.Pivot,  # Note: sqlglot uses Pivot class for both PIVOT and UNPIVOT
        name: str,
        parent_unit: QueryUnit,
        depth: int,
        table_node,  # Can be exp.Table or exp.Subquery
    ) -> QueryUnit:
        """
        Parse UNPIVOT operation.

        UNPIVOT transforms columns into rows.
        Example: UNPIVOT(revenue FOR quarter IN (q1_revenue, q2_revenue, q3_revenue, q4_revenue))

        In sqlglot, UNPIVOT is represented as a Pivot node with unpivot=True.
        """
        # Create unit for UNPIVOT operation
        unit_id = self._generate_unit_id(QueryUnitType.UNPIVOT, name)
        unit = QueryUnit(
            unit_id=unit_id,
            unit_type=QueryUnitType.UNPIVOT,
            name=name,
            select_node=None,
            parent_unit=parent_unit,
            depth=depth,
        )

        # Extract UNPIVOT configuration
        unpivot_config = {}

        # For UNPIVOT, we need to extract:
        # - value_column: The new column for unpivoted values (e.g., "revenue")
        # - name_column: The new column for column names (e.g., "quarter")
        # - unpivot_columns: The columns being unpivoted (e.g., [q1_revenue, q2_revenue, ...])

        # Get value column from expressions (e.g., revenue)
        if hasattr(unpivot_node, "expressions") and unpivot_node.expressions:
            unpivot_config["value_column"] = str(unpivot_node.expressions[0])

        # Get name column and unpivot columns from fields (the FOR ... IN clause)
        if hasattr(unpivot_node, "fields") and unpivot_node.fields:
            for field in unpivot_node.fields:
                if isinstance(field, exp.In):
                    # The 'this' is the name column (e.g., quarter)
                    unpivot_config["name_column"] = str(field.this)
                    # The 'expressions' are the columns being unpivoted
                    if hasattr(field, "expressions"):
                        unpivot_config["unpivot_columns"] = [str(col) for col in field.expressions]

        unit.unpivot_config = unpivot_config

        # Parse the source
        # table_node can be either a Table or a Subquery
        if isinstance(table_node, exp.Subquery):
            # UNPIVOT is applied to a subquery: (SELECT ...) UNPIVOT(...)
            source_select = table_node.this
            if isinstance(source_select, exp.Select):
                source_name = f"{name}_source"
                source_unit = self._parse_select_unit(
                    select_node=source_select,
                    unit_type=QueryUnitType.SUBQUERY_PIVOT_SOURCE,
                    name=source_name,
                    parent_unit=unit,
                    depth=depth + 1,
                )
                unit.depends_on_units.append(source_unit.unit_id)
        elif isinstance(table_node, exp.Table):
            # UNPIVOT is applied to a base table: table_name UNPIVOT(...)
            table_name = (
                table_node.this.name if hasattr(table_node.this, "name") else table_node.name
            )
            unit.depends_on_tables.append(table_name)

        # Add to graph
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
                # Check if this table has PIVOT/UNPIVOT operations (stored in args)
                has_pivot = (
                    hasattr(source_node, "args")
                    and "pivots" in source_node.args
                    and source_node.args["pivots"]
                )

                if has_pivot:
                    # Process PIVOT/UNPIVOT operations
                    for pivot_node in source_node.args["pivots"]:
                        if isinstance(pivot_node, exp.Pivot):
                            # Check if this is UNPIVOT (has unpivot=True attribute)
                            is_unpivot = pivot_node.args.get("unpivot", False)

                            pivot_name = (
                                source_node.alias
                                if hasattr(source_node, "alias") and source_node.alias
                                else f"pivot_{self.subquery_counter}"
                            )
                            self.subquery_counter += 1

                            # Parse UNPIVOT or PIVOT operation
                            if is_unpivot:
                                pivot_unit = self._parse_unpivot(
                                    unpivot_node=pivot_node,
                                    name=pivot_name,
                                    parent_unit=parent_unit,
                                    depth=depth + 1,
                                    table_node=source_node,
                                )
                            else:
                                pivot_unit = self._parse_pivot(
                                    pivot_node=pivot_node,
                                    name=pivot_name,
                                    parent_unit=parent_unit,
                                    depth=depth + 1,
                                    table_node=source_node,
                                )

                            # Add dependency
                            if pivot_unit.unit_id not in parent_unit.depends_on_units:
                                parent_unit.depends_on_units.append(pivot_unit.unit_id)

                            # Store alias mapping for PIVOT/UNPIVOT
                            parent_unit.alias_mapping[pivot_name] = (pivot_name, True)
                            return  # PIVOT/UNPIVOT processed, skip normal table processing

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
                # Check if this subquery has PIVOT/UNPIVOT operations
                has_pivot = (
                    hasattr(source_node, "args")
                    and "pivots" in source_node.args
                    and source_node.args["pivots"]
                )

                if has_pivot:
                    # Process PIVOT/UNPIVOT operations
                    for pivot_node in source_node.args["pivots"]:
                        if isinstance(pivot_node, exp.Pivot):
                            pivot_name = (
                                source_node.alias
                                if hasattr(source_node, "alias") and source_node.alias
                                else f"pivot_{self.subquery_counter}"
                            )
                            self.subquery_counter += 1

                            # Parse PIVOT operation (pass the subquery as table_node)
                            pivot_unit = self._parse_pivot(
                                pivot_node=pivot_node,
                                name=pivot_name,
                                parent_unit=parent_unit,
                                depth=depth + 1,
                                table_node=source_node,
                            )

                            # Add dependency
                            if pivot_unit.unit_id not in parent_unit.depends_on_units:
                                parent_unit.depends_on_units.append(pivot_unit.unit_id)

                            # Store alias mapping for PIVOT
                            parent_unit.alias_mapping[pivot_name] = (pivot_name, True)
                            return  # PIVOT processed, skip normal subquery processing

                # Subquery in FROM clause (no PIVOT)
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
            # NOTE: We now collect this as a ValidationIssue in RecursiveLineageBuilder
            # instead of raising an error, so we can continue parsing and find all issues
            pass

    def _generate_unit_id(self, unit_type: QueryUnitType, name: str) -> str:
        """Generate unique unit ID"""
        if unit_type == QueryUnitType.MAIN_QUERY:
            return "main"
        elif unit_type == QueryUnitType.CTE:
            return f"cte:{name}"
        elif unit_type in (QueryUnitType.UNION, QueryUnitType.INTERSECT, QueryUnitType.EXCEPT):
            return f"setop:{name}"
        elif unit_type == QueryUnitType.PIVOT:
            return f"pivot:{name}"
        elif unit_type == QueryUnitType.UNPIVOT:
            return f"unpivot:{name}"
        else:
            return f"subq:{name}"


__all__ = ["RecursiveQueryParser"]
