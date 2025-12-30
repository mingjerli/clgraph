"""
Recursive query parser for SQL statements.

Parses SQL queries recursively to identify all query units (CTEs, subqueries, main query)
and builds a QueryUnitGraph representing the query structure.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import sqlglot
from sqlglot import exp

from .models import (
    QueryUnit,
    QueryUnitGraph,
    QueryUnitType,
    RecursiveCTEInfo,
    TVFInfo,
    TVFType,
    ValuesInfo,
)

# ============================================================================
# Table-Valued Functions (TVF) Registry
# ============================================================================

# Known TVF expressions mapped to their types
KNOWN_TVF_EXPRESSIONS: Dict[type, TVFType] = {
    # Generator TVFs
    exp.ExplodingGenerateSeries: TVFType.GENERATOR,
    exp.GenerateSeries: TVFType.GENERATOR,
    exp.GenerateDateArray: TVFType.GENERATOR,
    # External data TVFs
    exp.ReadCSV: TVFType.EXTERNAL,
}

# Known TVF function names (for Anonymous function calls)
KNOWN_TVF_NAMES: Dict[str, TVFType] = {
    # Generator TVFs
    "generate_series": TVFType.GENERATOR,
    "generate_date_array": TVFType.GENERATOR,
    "generate_timestamp_array": TVFType.GENERATOR,
    "sequence": TVFType.GENERATOR,
    "generator": TVFType.GENERATOR,
    "range": TVFType.GENERATOR,
    # Column-input TVFs (UNNEST/EXPLODE handled separately)
    "flatten": TVFType.COLUMN_INPUT,
    "explode": TVFType.COLUMN_INPUT,
    "posexplode": TVFType.COLUMN_INPUT,
    # External data TVFs
    "read_csv": TVFType.EXTERNAL,
    "read_parquet": TVFType.EXTERNAL,
    "read_json": TVFType.EXTERNAL,
    "read_ndjson": TVFType.EXTERNAL,
    "external_query": TVFType.EXTERNAL,
    # System TVFs
    "table": TVFType.SYSTEM,
    "result_scan": TVFType.SYSTEM,
}

# Default output column names for known TVFs
TVF_DEFAULT_COLUMNS: Dict[str, List[str]] = {
    "generate_series": ["generate_series"],
    "generate_date_array": ["date"],
    "generate_timestamp_array": ["timestamp"],
    "sequence": ["value"],
    "generator": ["seq4"],
    "range": ["range"],
    "flatten": ["value", "index", "key", "path", "this"],
    "explode": ["col"],
    "posexplode": ["pos", "col"],
}


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
        # Check if this is a WITH RECURSIVE clause
        # Note: sqlglot uses "with_" to avoid Python keyword conflict
        with_clause = select_node.args.get("with_") or select_node.args.get("with")
        is_recursive_with = False
        if with_clause:
            is_recursive_with = with_clause.args.get("recursive", False)

        if hasattr(select_node, "ctes") and select_node.ctes:
            for cte in select_node.ctes:
                if isinstance(cte, exp.CTE):
                    cte_name = cte.alias_or_name
                    cte_query = cte.this

                    # Check if this specific CTE is recursive (self-referencing)
                    if is_recursive_with and self._is_recursive_cte(cte_query, cte_name):
                        # Parse as recursive CTE
                        self._parse_recursive_cte(
                            cte=cte,
                            cte_name=cte_name,
                            parent_unit=unit,
                            depth=depth,
                        )
                    elif isinstance(cte_query, exp.Select):
                        # Regular CTE - parse as before
                        self._parse_select_unit(
                            select_node=cte_query,
                            unit_type=QueryUnitType.CTE,
                            name=cte_name,
                            parent_unit=unit,
                            depth=depth + 1,
                        )
                    elif isinstance(cte_query, exp.Union):
                        # Non-recursive UNION CTE
                        self._parse_set_operation(
                            set_node=cte_query,
                            operation_type="union",
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

        # 8. Parse window functions in SELECT (extracts PARTITION BY, ORDER BY, frame specs)
        self._parse_window_functions(select_node, unit)

        # 9. Parse SELECT expressions (may contain scalar subqueries)
        for expr in select_node.expressions:
            self._parse_select_subqueries(expr, unit, depth)

        # 10. Validate star usage (after parsing FROM/JOINs so we know table count)
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

        # Helper to detect if an expression is a TVF
        def is_tvf_expression(expr: exp.Expression) -> bool:
            """Check if expression is a Table-Valued Function."""
            # Check for known TVF expression types
            if type(expr) in KNOWN_TVF_EXPRESSIONS:
                return True

            # Check for Anonymous function calls with known TVF names
            if isinstance(expr, exp.Anonymous):
                func_name = expr.name.lower() if expr.name else ""
                return func_name in KNOWN_TVF_NAMES

            return False

        # Helper to extract TVF info from expression
        def extract_tvf_info(
            tvf_expr: exp.Expression, alias: str, column_aliases: List[str]
        ) -> TVFInfo:
            """Extract TVF information from a TVF expression."""
            # Determine function name and type
            tvf_type: TVFType = TVFType.GENERATOR  # default
            func_name: str = ""
            parameters: Dict[str, Any] = {}
            input_columns: List[str] = []
            external_source: Optional[str] = None

            # Get type from expression class
            if type(tvf_expr) in KNOWN_TVF_EXPRESSIONS:
                tvf_type = KNOWN_TVF_EXPRESSIONS[type(tvf_expr)]
                # Get function name from class name
                func_name = type(tvf_expr).__name__.lower()
                # Map to standard name
                if func_name in ("explodinggenerateseries", "generateseries"):
                    func_name = "generate_series"
                elif func_name == "generatedatearray":
                    func_name = "generate_date_array"
                elif func_name == "readcsv":
                    func_name = "read_csv"

            # Handle Anonymous function calls
            elif isinstance(tvf_expr, exp.Anonymous):
                func_name = tvf_expr.name.lower() if tvf_expr.name else "unknown"
                tvf_type = KNOWN_TVF_NAMES.get(func_name, TVFType.GENERATOR)

            # Extract parameters from expressions attribute
            if hasattr(tvf_expr, "expressions") and tvf_expr.expressions:
                for i, arg in enumerate(tvf_expr.expressions):
                    if isinstance(arg, exp.Literal):
                        # Literal value
                        value = arg.this
                        # Detect file paths for external TVFs
                        if i == 0 and tvf_type == TVFType.EXTERNAL:
                            external_source = str(value)
                        parameters[f"arg_{i}"] = value
                    elif isinstance(arg, exp.Column):
                        # Column reference - indicates COLUMN_INPUT type
                        col_ref = f"{arg.table}.{arg.name}" if arg.table else arg.name
                        input_columns.append(col_ref)
                        parameters[f"arg_{i}"] = col_ref
                    elif isinstance(arg, exp.Kwarg):
                        # Named parameter (e.g., ROWCOUNT => 100)
                        key = str(arg.this) if arg.this else f"arg_{i}"
                        value = str(arg.expression) if arg.expression else None
                        parameters[key.lower()] = value
                    else:
                        parameters[f"arg_{i}"] = str(arg)

            # Also extract parameters from args dict (for typed TVFs like ExplodingGenerateSeries)
            if hasattr(tvf_expr, "args"):
                args_dict = tvf_expr.args
                for key, value in args_dict.items():
                    if key == "expressions":
                        continue  # Already handled above
                    if isinstance(value, exp.Literal):
                        param_value = value.this
                        parameters[key] = param_value
                        # For external TVFs, extract the source path
                        if tvf_type == TVFType.EXTERNAL and key == "this":
                            external_source = str(param_value)
                    elif isinstance(value, exp.Column):
                        col_ref = f"{value.table}.{value.name}" if value.table else value.name
                        input_columns.append(col_ref)
                        parameters[key] = col_ref
                    elif value is not None and key != "this":
                        # Skip 'this' for non-external TVFs (it's often None or internal)
                        parameters[key] = str(value)

            # Get default output columns if not provided via alias
            output_columns = column_aliases if column_aliases else []
            if not output_columns:
                output_columns = TVF_DEFAULT_COLUMNS.get(func_name, ["value"])

            return TVFInfo(
                function_name=func_name,
                tvf_type=tvf_type,
                alias=alias,
                output_columns=output_columns,
                parameters=parameters,
                input_columns=input_columns,
                external_source=external_source,
            )

        # Helper to process TVF source
        def process_tvf_source(source_node: exp.Table, parent_unit: QueryUnit):
            """Process a Table-Valued Function in FROM clause."""
            inner_expr = source_node.this

            # Get alias
            alias = str(source_node.alias) if source_node.alias else None
            if not alias:
                alias = f"_tvf_{self.subquery_counter}"
                self.subquery_counter += 1

            # Extract column aliases from TableAlias (e.g., AS t(col1, col2))
            column_aliases: List[str] = []
            alias_node = source_node.args.get("alias")
            if alias_node and hasattr(alias_node, "columns") and alias_node.columns:
                column_aliases = [col.name for col in alias_node.columns if hasattr(col, "name")]

            # Extract TVF info
            tvf_info = extract_tvf_info(inner_expr, alias, column_aliases)

            # Store in parent unit
            parent_unit.tvf_sources[alias] = tvf_info

            # Also register alias mapping so columns can be resolved
            # TVFs are like virtual tables, so we map alias to itself with is_unit=False
            parent_unit.alias_mapping[alias] = (alias, False)

        # Helper to process VALUES clause
        def process_values_source(
            values_node: exp.Values, alias: str, column_aliases: List[str]
        ) -> ValuesInfo:
            """Process a VALUES clause and extract its information."""
            rows: List[List[Any]] = []

            # Parse each row (tuple)
            for row_expr in values_node.expressions:
                if isinstance(row_expr, exp.Tuple):
                    row = [extract_literal(v) for v in row_expr.expressions]
                    rows.append(row)

            # If no column aliases provided, generate defaults
            num_cols = len(rows[0]) if rows else 0
            if not column_aliases and num_cols > 0:
                column_aliases = [f"column{i + 1}" for i in range(num_cols)]

            # Infer column types
            column_types = infer_value_types(rows)

            return ValuesInfo(
                alias=alias,
                column_names=column_aliases,
                row_count=len(rows),
                column_types=column_types,
                sample_values=rows[:3],  # Keep first 3 rows as sample
            )

        def extract_literal(expr: exp.Expression) -> Any:
            """Extract literal value from expression."""
            if isinstance(expr, exp.Literal):
                if expr.is_int:
                    return int(expr.this)
                elif expr.is_number:
                    return float(expr.this)
                elif expr.is_string:
                    return expr.this
                return expr.this
            elif isinstance(expr, exp.Boolean):
                return expr.this
            elif isinstance(expr, exp.Null):
                return None
            # Complex expression - store as string
            return expr.sql()

        def infer_value_types(rows: List[List[Any]]) -> List[str]:
            """Infer column types from sample values."""
            if not rows:
                return []

            num_cols = len(rows[0])
            types: List[str] = []

            for col_idx in range(num_cols):
                col_values = [
                    row[col_idx] for row in rows if col_idx < len(row) and row[col_idx] is not None
                ]

                if not col_values:
                    types.append("unknown")
                elif all(isinstance(v, bool) for v in col_values):
                    types.append("boolean")
                elif all(isinstance(v, int) for v in col_values):
                    types.append("integer")
                elif all(isinstance(v, (int, float)) for v in col_values):
                    types.append("numeric")
                else:
                    types.append("string")

            return types

        def handle_values_in_subquery(source_node: exp.Subquery, parent_unit: QueryUnit):
            """Handle VALUES clause wrapped in a Subquery."""
            inner = source_node.this
            if not isinstance(inner, exp.Values):
                return False

            # Get alias
            alias = source_node.alias_or_name if hasattr(source_node, "alias") else None
            if not alias:
                alias = f"_values_{self.subquery_counter}"
                self.subquery_counter += 1

            # Extract column aliases from TableAlias (e.g., AS t(col1, col2))
            column_aliases: List[str] = []
            alias_node = source_node.args.get("alias")
            if alias_node and hasattr(alias_node, "columns") and alias_node.columns:
                column_aliases = [col.name for col in alias_node.columns if hasattr(col, "name")]

            # Process VALUES
            values_info = process_values_source(inner, alias, column_aliases)

            # Store in parent unit
            parent_unit.values_sources[alias] = values_info

            # Add alias mapping so columns can be resolved
            parent_unit.alias_mapping[alias] = (alias, False)

            return True

        # Helper to process a single table source
        def process_table_source(source_node):
            # Check for VALUES clause directly in FROM (not wrapped in Subquery)
            if isinstance(source_node, exp.Values):
                # Get alias
                alias = str(source_node.alias) if source_node.alias else None
                if not alias:
                    alias = f"_values_{self.subquery_counter}"
                    self.subquery_counter += 1

                # Extract column aliases from alias node
                column_aliases: List[str] = []
                alias_node = source_node.args.get("alias")
                if alias_node and hasattr(alias_node, "columns") and alias_node.columns:
                    column_aliases = [
                        col.name for col in alias_node.columns if hasattr(col, "name")
                    ]

                # Process VALUES
                values_info = process_values_source(source_node, alias, column_aliases)

                # Store in parent unit
                parent_unit.values_sources[alias] = values_info

                # Add alias mapping so columns can be resolved
                parent_unit.alias_mapping[alias] = (alias, False)
                return  # VALUES processed

            if isinstance(source_node, exp.Table):
                # Check if this is a Table-Valued Function (TVF)
                if hasattr(source_node, "this") and is_tvf_expression(source_node.this):
                    process_tvf_source(source_node, parent_unit)
                    return  # TVF processed, skip normal table processing

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
                # Check if this is a VALUES clause wrapped in Subquery
                if handle_values_in_subquery(source_node, parent_unit):
                    return  # VALUES processed, skip normal subquery processing

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

    def _parse_window_functions(self, select_node: exp.Select, unit: QueryUnit):
        """
        Parse window functions in SELECT clause to extract:
        - Function name and arguments
        - PARTITION BY columns
        - ORDER BY columns with direction
        - Frame specification (ROWS/RANGE/GROUPS)
        - Named window definitions

        Args:
            select_node: The SELECT expression
            unit: The query unit to store window info
        """
        # 1. Parse named window definitions from WINDOW clause
        window_defs = select_node.args.get("windows", [])
        for window_def in window_defs:
            if isinstance(window_def, exp.Window):
                # Named window: window alias is in 'this' as Identifier
                window_name_node = window_def.args.get("this")
                if isinstance(window_name_node, exp.Identifier):
                    window_name = window_name_node.this
                    spec = self._parse_window_spec(window_def)
                    unit.window_definitions[window_name] = spec

        # 2. Parse window functions in SELECT expressions
        windows = []
        for i, col_expr in enumerate(select_node.expressions):
            # Get output column alias
            alias = None
            if isinstance(col_expr, exp.Alias):
                alias = col_expr.alias
                search_expr = col_expr.this
            else:
                search_expr = col_expr

            # Find window functions in this expression
            for window in search_expr.find_all(exp.Window):
                window_info = self._parse_single_window(window, alias, i, unit)
                if window_info:
                    windows.append(window_info)

        if windows:
            unit.window_info = {"windows": windows}

    def _parse_single_window(
        self,
        window: exp.Window,
        output_alias: Optional[str],
        col_index: int,
        unit: QueryUnit,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a single window function expression.

        Args:
            window: The Window expression node
            output_alias: Alias of the output column (if any)
            col_index: Index of the column in SELECT
            unit: Query unit for resolving named window references

        Returns:
            Dictionary with window function details
        """
        # Get function name and arguments
        func = window.args.get("this")
        if func is None:
            return None

        func_name = ""
        func_args = []

        if hasattr(func, "sql_name"):
            func_name = func.sql_name()
        elif hasattr(func, "name"):
            func_name = func.name

        # Extract function arguments
        # Most aggregate functions have a single argument in 'this'
        if hasattr(func, "this") and func.this:
            arg_cols = self._extract_window_columns(func.this)
            func_args.extend(arg_cols)
        # Some functions (like NTILE) may have additional arguments in 'expressions'
        if hasattr(func, "expressions") and func.expressions:
            for arg in func.expressions:
                arg_cols = self._extract_window_columns(arg)
                func_args.extend(arg_cols)

        # Check for named window reference
        window_alias = window.alias
        base_spec: Dict[str, Any] = {}

        if window_alias and window_alias in unit.window_definitions:
            # Resolve named window reference
            base_spec = unit.window_definitions[window_alias].copy()

        # Parse inline window spec (may override or extend named window)
        inline_spec = self._parse_window_spec(window)

        # Merge: inline spec overrides base spec
        spec = {**base_spec, **inline_spec}

        # Build result
        output_column = output_alias if output_alias else f"_col{col_index}"

        return {
            "output_column": output_column,
            "function": func_name,
            "arguments": func_args,
            "partition_by": spec.get("partition_by", []),
            "order_by": spec.get("order_by", []),
            "frame_type": spec.get("frame_type"),
            "frame_start": spec.get("frame_start"),
            "frame_end": spec.get("frame_end"),
            "window_name": window_alias if window_alias else None,
        }

    def _parse_window_spec(self, window: exp.Window) -> Dict[str, Any]:
        """
        Parse window specification from a Window expression.

        Args:
            window: The Window expression node

        Returns:
            Dictionary with partition_by, order_by, and frame details
        """
        result: Dict[str, Any] = {}

        # Parse PARTITION BY
        partition_by = window.args.get("partition_by", [])
        if partition_by:
            partition_cols = []
            for expr in partition_by:
                cols = self._extract_window_columns(expr)
                partition_cols.extend(cols)
            result["partition_by"] = partition_cols

        # Parse ORDER BY
        order = window.args.get("order")
        if order:
            order_cols = []
            for order_expr in order.expressions:
                col_info = self._parse_order_by_column(order_expr)
                order_cols.append(col_info)
            result["order_by"] = order_cols

        # Parse frame specification
        spec = window.args.get("spec")
        if spec:
            frame_info = self._parse_frame_spec(spec)
            result.update(frame_info)

        return result

    def _parse_order_by_column(self, order_expr: exp.Expression) -> Dict[str, Any]:
        """
        Parse ORDER BY column with direction and nulls handling.

        Args:
            order_expr: The Ordered expression or column

        Returns:
            Dictionary with column, direction, and nulls info
        """
        if isinstance(order_expr, exp.Ordered):
            # Extract column(s) from the ordered expression
            cols = self._extract_window_columns(order_expr.this)
            column = cols[0] if cols else str(order_expr.this)

            # Get direction from args (desc is None for ASC, True for DESC)
            desc_val = order_expr.args.get("desc")
            if desc_val is True:
                direction = "desc"
            else:
                direction = "asc"

            # Get nulls handling from args
            nulls_first_val = order_expr.args.get("nulls_first")
            if nulls_first_val is True:
                nulls = "first"
            else:
                nulls = "last"

            return {"column": column, "direction": direction, "nulls": nulls}
        else:
            # Plain column without direction
            cols = self._extract_window_columns(order_expr)
            column = cols[0] if cols else str(order_expr)
            return {"column": column, "direction": "asc", "nulls": "last"}

    def _parse_frame_spec(self, spec: exp.WindowSpec) -> Dict[str, Any]:
        """
        Parse frame specification (ROWS/RANGE/GROUPS BETWEEN ... AND ...).

        Args:
            spec: The WindowSpec expression

        Returns:
            Dictionary with frame_type, frame_start, frame_end
        """
        result: Dict[str, Any] = {}

        # Frame type: ROWS, RANGE, or GROUPS
        kind = spec.args.get("kind")
        if kind:
            result["frame_type"] = kind.lower()

        # Frame start
        start = spec.args.get("start")
        start_side = spec.args.get("start_side")
        if start is not None:
            result["frame_start"] = self._format_frame_boundary(start, start_side)

        # Frame end
        end = spec.args.get("end")
        end_side = spec.args.get("end_side")
        if end is not None:
            result["frame_end"] = self._format_frame_boundary(end, end_side)

        return result

    def _format_frame_boundary(self, boundary: Any, side: Optional[str]) -> str:
        """
        Format a frame boundary as a string.

        Args:
            boundary: The boundary value (UNBOUNDED, number, CURRENT ROW)
            side: PRECEDING or FOLLOWING

        Returns:
            String like "unbounded preceding", "3 preceding", "current row"
        """
        if isinstance(boundary, str):
            # Already a string like "CURRENT ROW" or "UNBOUNDED"
            boundary_str = boundary.lower()
            if boundary_str == "current row":
                return "current row"
            elif boundary_str == "unbounded":
                if side:
                    return f"unbounded {side.lower()}"
                return "unbounded"
            else:
                if side:
                    return f"{boundary_str} {side.lower()}"
                return boundary_str
        elif isinstance(boundary, exp.Literal):
            # Numeric literal
            value = boundary.this
            if side:
                return f"{value} {side.lower()}"
            return str(value)
        elif hasattr(boundary, "this"):
            # Some other expression
            value = str(boundary.this)
            if side:
                return f"{value} {side.lower()}"
            return value
        else:
            return str(boundary)

    def _extract_window_columns(self, expr: exp.Expression) -> List[str]:
        """
        Extract column references from an expression.

        Args:
            expr: The expression to extract columns from

        Returns:
            List of column names (qualified if table is present)
        """
        columns = []
        if isinstance(expr, str):
            # Sometimes expr is a string (e.g., from certain window functions)
            # Just return empty - we can't extract columns from a string
            return columns
        if isinstance(expr, exp.Column):
            table_ref = str(expr.table) if expr.table else None
            col_name = expr.name
            if table_ref:
                columns.append(f"{table_ref}.{col_name}")
            else:
                columns.append(col_name)
        elif hasattr(expr, "find_all"):
            # Walk expression tree for columns
            for col in expr.find_all(exp.Column):
                table_ref = str(col.table) if col.table else None
                col_name = col.name
                if table_ref:
                    full_name = f"{table_ref}.{col_name}"
                else:
                    full_name = col_name
                if full_name not in columns:
                    columns.append(full_name)
        return columns

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

    # ============================================================================
    # Recursive CTE Parsing
    # ============================================================================

    def _is_recursive_cte(self, query: exp.Expression, cte_name: str) -> bool:
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

    def _parse_recursive_cte(
        self,
        cte: exp.CTE,
        cte_name: str,
        parent_unit: QueryUnit,
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
            base_unit = self._parse_select_unit(
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
                base_unit = self._parse_select_unit(
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
            recursive_unit = self._parse_select_unit(
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
                recursive_unit = self._parse_select_unit(
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
        unit_id = self._generate_unit_id(QueryUnitType.CTE, cte_name)
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
        self.unit_graph.add_unit(cte_unit)

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

    def _generate_unit_id(self, unit_type: QueryUnitType, name: str) -> str:
        """Generate unique unit ID"""
        if unit_type == QueryUnitType.MAIN_QUERY:
            return "main"
        elif unit_type == QueryUnitType.CTE:
            return f"cte:{name}"
        elif unit_type == QueryUnitType.CTE_BASE:
            return f"cte_base:{name}"
        elif unit_type == QueryUnitType.CTE_RECURSIVE:
            return f"cte_recursive:{name}"
        elif unit_type in (QueryUnitType.UNION, QueryUnitType.INTERSECT, QueryUnitType.EXCEPT):
            return f"setop:{name}"
        elif unit_type == QueryUnitType.PIVOT:
            return f"pivot:{name}"
        elif unit_type == QueryUnitType.UNPIVOT:
            return f"unpivot:{name}"
        else:
            return f"subq:{name}"


__all__ = ["RecursiveQueryParser"]
