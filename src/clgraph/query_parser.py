"""
Recursive query parser for SQL statements.

Parses SQL queries recursively to identify all query units (CTEs, subqueries, main query)
and builds a QueryUnitGraph representing the query structure.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

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
        elif isinstance(self.parsed, exp.Merge):
            # MERGE INTO statement
            self._parse_merge_statement(
                merge_node=self.parsed,
                name="main",
                depth=0,
            )
        else:
            raise ValueError(
                f"Unsupported top-level query type: {type(self.parsed).__name__}. "
                f"Expected Select, Union, Intersect, Except, or Merge."
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

        # 6. Parse QUALIFY clause (extracts window function columns)
        qualify_clause = select_node.args.get("qualify")
        if qualify_clause:
            self._parse_qualify_clause(qualify_clause, unit)

        # 7. Parse GROUP BY clause for GROUPING SETS/CUBE/ROLLUP
        group_clause = select_node.args.get("group")
        if group_clause:
            self._parse_grouping_sets(group_clause, unit)

        # 8. Parse SELECT expressions (may contain scalar subqueries)
        for expr in select_node.expressions:
            self._parse_select_subqueries(expr, unit, depth)

        # 9. Validate star usage (after parsing FROM/JOINs so we know table count)
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

    def _parse_merge_statement(
        self,
        merge_node: exp.Merge,
        name: str,
        depth: int,
    ) -> QueryUnit:
        """
        Parse MERGE INTO statement.

        MERGE combines INSERT, UPDATE, and DELETE operations based on match conditions.
        Example:
            MERGE INTO target t
            USING source s ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET t.value = s.new_value
            WHEN NOT MATCHED THEN INSERT (id, value) VALUES (s.id, s.new_value)
        """
        # Create unit for MERGE operation
        unit_id = self._generate_unit_id(QueryUnitType.MERGE, name)
        unit = QueryUnit(
            unit_id=unit_id,
            unit_type=QueryUnitType.MERGE,
            name=name,
            select_node=None,
            parent_unit=None,
            depth=depth,
        )

        # Extract target table
        target_table = merge_node.this
        target_name = None
        target_alias = None
        if isinstance(target_table, exp.Table):
            target_name = target_table.name
            if hasattr(target_table, "alias") and target_table.alias:
                target_alias = str(target_table.alias)

        # Extract source table (can be table or subquery)
        source = merge_node.args.get("using")
        source_name = None
        source_alias = None
        if isinstance(source, exp.Table):
            source_name = source.name
            if hasattr(source, "alias") and source.alias:
                source_alias = str(source.alias)
            unit.depends_on_tables.append(source_name)
        elif isinstance(source, exp.Subquery):
            # Source is a subquery - parse it
            source_select = source.this
            if isinstance(source_select, exp.Select):
                source_alias = (
                    str(source.alias) if hasattr(source, "alias") and source.alias else "source"
                )
                source_unit = self._parse_select_unit(
                    select_node=source_select,
                    unit_type=QueryUnitType.MERGE_SOURCE,
                    name=source_alias,
                    parent_unit=unit,
                    depth=depth + 1,
                )
                unit.depends_on_units.append(source_unit.unit_id)
                source_name = source_alias

        # Add target to depends_on_tables (MERGE reads and modifies target)
        if target_name:
            unit.depends_on_tables.append(target_name)

        # Store alias mappings
        if target_alias and target_name:
            unit.alias_mapping[target_alias] = (target_name, False)
        if source_alias and source_name:
            unit.alias_mapping[source_alias] = (source_name, False)

        # Extract match condition
        match_condition = merge_node.args.get("on")
        match_condition_sql = match_condition.sql() if match_condition else None

        # Extract match columns from ON condition
        match_columns: List[Tuple[str, str]] = []
        if match_condition:
            for eq in match_condition.find_all(exp.EQ):
                left_col = eq.left
                right_col = eq.right
                if isinstance(left_col, exp.Column) and isinstance(right_col, exp.Column):
                    match_columns.append((left_col.name, right_col.name))

        # Parse WHEN clauses from the 'whens' arg
        whens = merge_node.args.get("whens")
        matched_actions: List[Dict[str, Any]] = []
        not_matched_actions: List[Dict[str, Any]] = []

        if whens and hasattr(whens, "expressions"):
            for when in whens.expressions:
                is_matched = when.args.get("matched", False)
                then_expr = when.args.get("then")
                condition = when.args.get("condition")
                condition_sql = condition.sql() if condition else None

                action: Dict[str, Any] = {
                    "condition": condition_sql,
                    "column_mappings": {},
                }

                if isinstance(then_expr, exp.Update):
                    action["action_type"] = "update"
                    # Extract SET clause mappings
                    for set_expr in then_expr.expressions:
                        if isinstance(set_expr, exp.EQ):
                            target_col = (
                                set_expr.left.name
                                if hasattr(set_expr.left, "name")
                                else str(set_expr.left)
                            )
                            source_expr = set_expr.right.sql()
                            action["column_mappings"][target_col] = source_expr
                    if is_matched:
                        matched_actions.append(action)
                    else:
                        not_matched_actions.append(action)

                elif isinstance(then_expr, exp.Insert):
                    action["action_type"] = "insert"
                    # Extract target columns and source values
                    target_cols = []
                    if then_expr.this and hasattr(then_expr.this, "expressions"):
                        target_cols = [col.name for col in then_expr.this.expressions]
                    source_vals = []
                    if then_expr.expression and hasattr(then_expr.expression, "expressions"):
                        source_vals = [val.sql() for val in then_expr.expression.expressions]
                    for i, target_col in enumerate(target_cols):
                        if i < len(source_vals):
                            action["column_mappings"][target_col] = source_vals[i]
                    not_matched_actions.append(action)

                elif isinstance(then_expr, exp.Delete):
                    action["action_type"] = "delete"
                    if is_matched:
                        matched_actions.append(action)

        # Store merge configuration in a custom attribute
        # Using unpivot_config as a general-purpose config storage
        unit.unpivot_config = {
            "merge_type": "merge",
            "target_table": target_name,
            "target_alias": target_alias,
            "source_table": source_name,
            "source_alias": source_alias,
            "match_condition": match_condition_sql,
            "match_columns": match_columns,
            "matched_actions": matched_actions,
            "not_matched_actions": not_matched_actions,
        }

        # Add to graph
        self.unit_graph.add_unit(unit)

        return unit

    def _parse_from_sources(self, from_node: exp.Expression, parent_unit: QueryUnit, depth: int):
        """
        Parse FROM/JOIN clause, which may contain:
        - Base tables
        - CTEs
        - Subqueries (derived tables)
        - UNNEST/FLATTEN/EXPLODE expressions (array expansion)

        Note: We need to extract table sources from FROM and JOIN clauses only,
        not from the entire subtree (which would include column references).

        Also captures alias mappings for proper column reference resolution.
        """

        # Helper to process UNNEST expression
        def process_unnest_source(unnest_node: exp.Unnest, parent_unit: QueryUnit):
            """Process UNNEST expression and store metadata in parent_unit."""
            # Extract the array column being unnested
            array_expr = None
            if unnest_node.expressions:
                array_expr = unnest_node.expressions[0]

            if not array_expr:
                return

            # Get source table and column from array expression
            source_table = None
            source_column = None
            if isinstance(array_expr, exp.Column):
                source_column = array_expr.name
                if hasattr(array_expr, "table") and array_expr.table:
                    source_table = (
                        array_expr.table.name
                        if hasattr(array_expr.table, "name")
                        else str(array_expr.table)
                    )

            # Get the alias for the unnested values
            unnest_alias = None
            alias_node = unnest_node.args.get("alias")
            if alias_node:
                # TableAlias has columns attribute for the value aliases
                if hasattr(alias_node, "columns") and alias_node.columns:
                    unnest_alias = alias_node.columns[0].name
                elif hasattr(alias_node, "this"):
                    unnest_alias = (
                        alias_node.this.name
                        if hasattr(alias_node.this, "name")
                        else str(alias_node.this)
                    )

            if not unnest_alias:
                unnest_alias = f"_unnest_{self.subquery_counter}"
                self.subquery_counter += 1

            # Get offset alias if WITH OFFSET is used
            offset_alias = None
            offset_node = unnest_node.args.get("offset")
            if offset_node:
                if hasattr(offset_node, "name"):
                    offset_alias = offset_node.name
                elif hasattr(offset_node, "this"):
                    offset_alias = (
                        offset_node.this if isinstance(offset_node.this, str) else str(offset_node)
                    )
                else:
                    offset_alias = str(offset_node)

            # Store UNNEST info in parent_unit
            parent_unit.unnest_sources[unnest_alias] = {
                "source_table": source_table,
                "source_column": source_column,
                "offset_alias": offset_alias,
                "expansion_type": "unnest",
            }

            # Also add offset alias if present
            if offset_alias:
                parent_unit.unnest_sources[offset_alias] = {
                    "source_table": source_table,
                    "source_column": source_column,
                    "is_offset": True,
                    "unnest_alias": unnest_alias,
                    "expansion_type": "unnest",
                }

        # Helper to process Snowflake LATERAL FLATTEN
        def process_lateral_flatten(lateral_node: exp.Lateral, parent_unit: QueryUnit):
            """Process Snowflake LATERAL FLATTEN and store metadata."""
            inner_expr = lateral_node.this
            if not isinstance(inner_expr, exp.Explode):
                return

            # Extract INPUT parameter from FLATTEN
            source_table = None
            source_column = None

            input_expr = inner_expr.this
            if isinstance(input_expr, exp.EQ):
                # INPUT => col format
                right = input_expr.right
                if isinstance(right, exp.Column):
                    source_column = right.name
                    if hasattr(right, "table") and right.table:
                        source_table = (
                            right.table.name if hasattr(right.table, "name") else str(right.table)
                        )
            elif isinstance(input_expr, exp.Column):
                source_column = input_expr.name
                if hasattr(input_expr, "table") and input_expr.table:
                    source_table = (
                        input_expr.table.name
                        if hasattr(input_expr.table, "name")
                        else str(input_expr.table)
                    )

            # Get alias
            flatten_alias = None
            if hasattr(lateral_node, "alias") and lateral_node.alias:
                if hasattr(lateral_node.alias, "this"):
                    flatten_alias = (
                        lateral_node.alias.this.name
                        if hasattr(lateral_node.alias.this, "name")
                        else str(lateral_node.alias.this)
                    )
                else:
                    flatten_alias = str(lateral_node.alias)

            if not flatten_alias:
                flatten_alias = f"_flatten_{self.subquery_counter}"
                self.subquery_counter += 1

            # Store FLATTEN info
            parent_unit.unnest_sources[flatten_alias] = {
                "source_table": source_table,
                "source_column": source_column,
                "offset_alias": None,  # FLATTEN uses .INDEX field instead
                "expansion_type": "flatten",
                "flatten_fields": ["VALUE", "INDEX", "KEY", "PATH", "SEQ", "THIS"],
            }

        # Helper to process general LATERAL subquery (correlated subquery)
        def process_lateral_subquery(
            lateral_node: exp.Lateral, parent_unit: QueryUnit, preceding_tables: List[str]
        ):
            """Process LATERAL subquery and identify correlated column references.

            Args:
                lateral_node: The LATERAL AST node
                parent_unit: The parent query unit
                preceding_tables: List of table names/aliases that precede this LATERAL
            """
            inner_expr = lateral_node.this

            # Skip if this is a FLATTEN (handled separately)
            if isinstance(inner_expr, exp.Explode):
                process_lateral_flatten(lateral_node, parent_unit)
                return

            # Skip if not a Subquery
            if not isinstance(inner_expr, exp.Subquery):
                return

            subquery = inner_expr.this
            if not isinstance(subquery, exp.Select):
                return

            # Get LATERAL alias
            lateral_alias = None
            if hasattr(lateral_node, "alias") and lateral_node.alias:
                if hasattr(lateral_node.alias, "this"):
                    lateral_alias = (
                        lateral_node.alias.this.name
                        if hasattr(lateral_node.alias.this, "name")
                        else str(lateral_node.alias.this)
                    )
                else:
                    lateral_alias = str(lateral_node.alias)

            if not lateral_alias:
                lateral_alias = f"_lateral_{self.subquery_counter}"
                self.subquery_counter += 1

            # Find all column references in the subquery
            correlated_columns: List[str] = []
            for col in subquery.find_all(exp.Column):
                table_ref = None
                if hasattr(col, "table") and col.table:
                    table_ref = (
                        str(col.table.name) if hasattr(col.table, "name") else str(col.table)
                    )

                # Check if this column references a preceding table (correlation)
                if table_ref and table_ref in preceding_tables:
                    correlated_columns.append(f"{table_ref}.{col.name}")

            # Store LATERAL info
            parent_unit.lateral_sources[lateral_alias] = {
                "correlated_columns": correlated_columns,
                "preceding_tables": preceding_tables.copy(),
                "subquery_sql": subquery.sql(),
            }

            # Parse the LATERAL subquery as a unit
            subquery_name = lateral_alias
            subquery_unit = self._parse_select_unit(
                select_node=subquery,
                unit_type=QueryUnitType.SUBQUERY_FROM,
                name=subquery_name,
                parent_unit=parent_unit,
                depth=depth + 1,
            )

            # Mark as LATERAL and store correlation info
            subquery_unit.is_lateral = True
            subquery_unit.correlated_columns = correlated_columns

            # Add dependency and alias mapping
            if subquery_unit.unit_id not in parent_unit.depends_on_units:
                parent_unit.depends_on_units.append(subquery_unit.unit_id)
            parent_unit.alias_mapping[lateral_alias] = (subquery_name, True)

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

        # Helper to get table name/alias from a source node
        def get_source_name(source_node) -> Optional[str]:
            """Get the name or alias of a table source."""
            if isinstance(source_node, exp.Table):
                if hasattr(source_node, "alias") and source_node.alias:
                    return str(source_node.alias)
                return source_node.name
            elif isinstance(source_node, exp.Subquery):
                if hasattr(source_node, "alias") and source_node.alias:
                    return str(source_node.alias)
            return None

        # Get preceding tables from alias_mapping for LATERAL correlation detection
        # This allows detecting correlations to tables already registered in previous
        # calls to _parse_from_sources (e.g., FROM clause before processing JOINs)
        preceding_tables: List[str] = list(parent_unit.alias_mapping.keys())
        # Also add tables from depends_on_tables (base tables)
        preceding_tables.extend(parent_unit.depends_on_tables)

        # Helper to register a table/alias to preceding tables
        def register_preceding_table(name: str):
            if name and name not in preceding_tables:
                preceding_tables.append(name)

        # Process the main FROM source
        if isinstance(from_node, (exp.Table, exp.Subquery)):
            process_table_source(from_node)
            source_name = get_source_name(from_node)
            if source_name:
                register_preceding_table(source_name)
        elif isinstance(from_node, exp.Unnest):
            # UNNEST directly in FROM clause
            process_unnest_source(from_node, parent_unit)
        elif isinstance(from_node, exp.Lateral):
            # LATERAL subquery - process with preceding tables context
            process_lateral_subquery(from_node, parent_unit, preceding_tables)
        elif isinstance(from_node, exp.Join):
            # JOIN clause with a LATERAL subquery inside
            if hasattr(from_node, "this"):
                join_source = from_node.this
                if isinstance(join_source, exp.Lateral):
                    process_lateral_subquery(join_source, parent_unit, preceding_tables)
                elif isinstance(join_source, exp.Unnest):
                    process_unnest_source(join_source, parent_unit)
                else:
                    process_table_source(join_source)
                    source_name = get_source_name(join_source)
                    if source_name:
                        register_preceding_table(source_name)
        elif hasattr(from_node, "this"):
            # FROM clause with this attribute
            if isinstance(from_node.this, exp.Unnest):
                process_unnest_source(from_node.this, parent_unit)
            elif isinstance(from_node.this, exp.Lateral):
                # LATERAL subquery - process with preceding tables context
                process_lateral_subquery(from_node.this, parent_unit, preceding_tables)
            else:
                process_table_source(from_node.this)
                source_name = get_source_name(from_node.this)
                if source_name:
                    register_preceding_table(source_name)

        # Process JOIN clauses (includes CROSS JOIN UNNEST)
        # JOINs are stored in the 'joins' attribute
        if hasattr(from_node, "args") and "joins" in from_node.args:
            joins = from_node.args["joins"]
            if joins:
                for join in joins:
                    # Each join has a 'this' which is the table/subquery being joined
                    if hasattr(join, "this"):
                        join_source = join.this
                        if isinstance(join_source, exp.Unnest):
                            process_unnest_source(join_source, parent_unit)
                        elif isinstance(join_source, exp.Lateral):
                            # LATERAL subquery in JOIN - process with preceding tables
                            process_lateral_subquery(join_source, parent_unit, preceding_tables)
                        else:
                            process_table_source(join_source)
                            # Add to preceding tables for subsequent LATERAL
                            source_name = get_source_name(join_source)
                            if source_name:
                                register_preceding_table(source_name)

        # Also scan the entire from_node for UNNEST expressions that may be nested
        for node in from_node.walk():
            if isinstance(node, exp.Unnest):
                # Check if we already processed this UNNEST (by checking unnest_sources)
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
                        process_unnest_source(node, parent_unit)
            elif isinstance(node, exp.Lateral):
                # Check if this is a FLATTEN (Explode) or general LATERAL subquery
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
                            process_lateral_flatten(node, parent_unit)
                elif isinstance(node.this, exp.Subquery):
                    # General LATERAL subquery - check if not already processed
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
                        process_lateral_subquery(node, parent_unit, preceding_tables)

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

    def _parse_qualify_clause(self, qualify_node: exp.Qualify, unit: QueryUnit):
        """
        Parse QUALIFY clause to extract window function column dependencies.

        QUALIFY filters rows based on window function results.
        Example:
            QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1

        This extracts:
        - condition: The full QUALIFY condition as SQL
        - partition_columns: Columns used in PARTITION BY
        - order_columns: Columns used in ORDER BY
        - window_functions: Names of window functions used
        """
        condition = qualify_node.this
        partition_columns: List[str] = []
        order_columns: List[str] = []
        window_functions: List[str] = []

        # Walk the condition to find window functions
        for node in condition.walk():
            if isinstance(node, exp.Window):
                # Get function name
                func = node.this
                # Try sql_name() first (works for ROW_NUMBER, RANK, etc.), fall back to type name
                if hasattr(func, "sql_name"):
                    func_name = func.sql_name()
                elif hasattr(func, "name") and func.name:
                    func_name = func.name
                else:
                    func_name = type(func).__name__
                window_functions.append(func_name.upper())

                # Get PARTITION BY columns
                partition_by = node.args.get("partition_by")
                if partition_by:
                    for partition_expr in partition_by:
                        for col in partition_expr.find_all(exp.Column):
                            table_ref = str(col.table) if col.table else None
                            col_name = col.name
                            full_name = f"{table_ref}.{col_name}" if table_ref else col_name
                            if full_name not in partition_columns:
                                partition_columns.append(full_name)

                # Get ORDER BY columns
                order_by = node.args.get("order")
                if order_by and hasattr(order_by, "expressions"):
                    for order_expr in order_by.expressions:
                        expr_node = (
                            order_expr.this if isinstance(order_expr, exp.Ordered) else order_expr
                        )
                        for col in expr_node.find_all(exp.Column):
                            table_ref = str(col.table) if col.table else None
                            col_name = col.name
                            full_name = f"{table_ref}.{col_name}" if table_ref else col_name
                            if full_name not in order_columns:
                                order_columns.append(full_name)

        # Store QUALIFY info on the unit
        unit.qualify_info = {
            "condition": condition.sql(),
            "partition_columns": partition_columns,
            "order_columns": order_columns,
            "window_functions": window_functions,
        }

    def _parse_grouping_sets(self, group_clause: exp.Group, unit: QueryUnit):
        """
        Parse GROUP BY clause for GROUPING SETS, CUBE, and ROLLUP constructs.

        These constructs generate multiple grouping levels in a single query:
        - CUBE(a, b): All combinations: (a,b), (a), (b), ()
        - ROLLUP(a, b): Hierarchical: (a,b), (a), ()
        - GROUPING SETS(...): Explicit list of grouping combinations

        Args:
            group_clause: The GROUP BY clause expression
            unit: The query unit to store grouping config
        """
        # Check for CUBE
        cube_list = group_clause.args.get("cube", [])
        if cube_list:
            for cube_node in cube_list:
                if isinstance(cube_node, exp.Cube):
                    columns = self._extract_grouping_columns(cube_node.expressions)
                    # CUBE generates all 2^n combinations
                    grouping_sets = self._expand_cube(columns)
                    unit.grouping_config = {
                        "grouping_type": "cube",
                        "grouping_columns": columns,
                        "grouping_sets": grouping_sets,
                    }
                    return

        # Check for ROLLUP
        rollup_list = group_clause.args.get("rollup", [])
        if rollup_list:
            for rollup_node in rollup_list:
                if isinstance(rollup_node, exp.Rollup):
                    columns = self._extract_grouping_columns(rollup_node.expressions)
                    # ROLLUP generates n+1 hierarchical combinations
                    grouping_sets = self._expand_rollup(columns)
                    unit.grouping_config = {
                        "grouping_type": "rollup",
                        "grouping_columns": columns,
                        "grouping_sets": grouping_sets,
                    }
                    return

        # Check for GROUPING SETS
        gs_list = group_clause.args.get("grouping_sets", [])
        if gs_list:
            for gs_node in gs_list:
                if isinstance(gs_node, exp.GroupingSets):
                    grouping_sets = []
                    columns_set: set = set()
                    for set_expr in gs_node.expressions:
                        if isinstance(set_expr, exp.Tuple):
                            # Tuple: (a, b)
                            cols = self._extract_grouping_columns(set_expr.expressions)
                            grouping_sets.append(cols)
                            columns_set.update(cols)
                        elif isinstance(set_expr, exp.Paren):
                            # Single column: (a)
                            cols = self._extract_grouping_columns([set_expr.this])
                            grouping_sets.append(cols)
                            columns_set.update(cols)
                        else:
                            # Could be empty () for grand total
                            grouping_sets.append([])
                    unit.grouping_config = {
                        "grouping_type": "grouping_sets",
                        "grouping_columns": list(columns_set),
                        "grouping_sets": grouping_sets,
                    }
                    return

    def _extract_grouping_columns(self, expressions: List[exp.Expression]) -> List[str]:
        """Extract column names from a list of expressions."""
        columns = []
        for expr in expressions:
            if isinstance(expr, exp.Column):
                table_ref = str(expr.table) if expr.table else None
                col_name = expr.name
                full_name = f"{table_ref}.{col_name}" if table_ref else col_name
                if full_name not in columns:
                    columns.append(full_name)
            else:
                # Walk nested expressions for columns
                for col in expr.find_all(exp.Column):
                    table_ref = str(col.table) if col.table else None
                    col_name = col.name
                    full_name = f"{table_ref}.{col_name}" if table_ref else col_name
                    if full_name not in columns:
                        columns.append(full_name)
        return columns

    def _expand_cube(self, columns: List[str]) -> List[List[str]]:
        """Expand CUBE into all 2^n combinations."""
        from itertools import combinations

        result = []
        n = len(columns)
        # Generate all subsets from full set to empty set
        for r in range(n, -1, -1):
            for combo in combinations(columns, r):
                result.append(list(combo))
        return result

    def _expand_rollup(self, columns: List[str]) -> List[List[str]]:
        """Expand ROLLUP into hierarchical combinations."""
        result = []
        # From full set down to empty set hierarchically
        for i in range(len(columns), -1, -1):
            result.append(columns[:i])
        return result

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
