"""
Recursive query parser for SQL statements.

Parses SQL queries recursively to identify all query units (CTEs, subqueries, main query)
and builds a QueryUnitGraph representing the query structure.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import sqlglot
from sqlglot import exp

from .models import (
    JoinPredicateInfo,
    QueryUnit,
    QueryUnitGraph,
    QueryUnitType,
    RecursiveCTEInfo,
    WherePredicateInfo,
)

# Import helper classes for composition-based decomposition
from .query_parser_helpers import FromClauseParser, SpecialSourcesHandler, WindowFunctionsParser

# ============================================================================
# Table-Valued Functions (TVF) Registry
# Import from tvf_registry.py and re-export for backward compatibility
# ============================================================================
from .tvf_registry import (  # noqa: F401, E402
    KNOWN_TVF_EXPRESSIONS,
    KNOWN_TVF_NAMES,
    TVF_DEFAULT_COLUMNS,
)


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

        # Initialize helper classes for composition-based decomposition
        self._from_clause = FromClauseParser(self)
        self._special_sources = SpecialSourcesHandler(self)
        self._window_functions = WindowFunctionsParser(self)

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

        # 3b. Extract JOIN ON predicate columns for lineage tracking
        for join in joins:
            on_clause = join.args.get("on")
            if on_clause:
                cols = self._extract_join_predicate_columns(on_clause)
                join_type = self._get_join_type(join)
                right_table = self._get_join_right_table(join, unit)
                unit.join_predicates.append(
                    JoinPredicateInfo(
                        condition_sql=on_clause.sql(),
                        columns=cols,
                        join_type=join_type,
                        right_table=right_table,
                    )
                )

        # 4. Parse WHERE clause (may contain subqueries)
        where_clause = select_node.args.get("where")
        if where_clause:
            self._parse_where_subqueries(where_clause, unit, depth)

        # 4b. Extract WHERE clause column refs for filter lineage
        if where_clause:
            where_cols = self._extract_where_columns(where_clause.this)
            if where_cols:
                unit.where_predicates.append(
                    WherePredicateInfo(
                        condition_sql=where_clause.this.sql(),
                        columns=where_cols,
                    )
                )

        # 4c. Promote dedup qualify info from WHERE (Gap 2)
        self._promote_dedup_qualify_if_applicable(select_node, unit)

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
        match_filter_columns: List[Tuple[str, str]] = []
        if match_condition:
            for eq in match_condition.find_all(exp.EQ):
                left_col = eq.left
                right_col = eq.right
                if isinstance(left_col, exp.Column) and isinstance(right_col, exp.Column):
                    match_columns.append((left_col.name, right_col.name))
                elif isinstance(left_col, exp.Column) and not isinstance(right_col, exp.Column):
                    match_filter_columns.append((left_col.name, right_col.sql()))
                elif isinstance(right_col, exp.Column) and not isinstance(left_col, exp.Column):
                    match_filter_columns.append((right_col.name, left_col.sql()))

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
            "match_filter_columns": match_filter_columns,
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

        Delegates to FromClauseParser for the actual implementation.
        """
        self._from_clause.parse(from_node, parent_unit, depth)

    def _extract_join_predicate_columns(
        self, on_clause: exp.Expression
    ) -> List[Tuple[Optional[str], str]]:
        """
        Extract column references from a JOIN ON clause expression.

        Walks the expression tree to find all exp.Column nodes and returns
        (table_ref, col_name) pairs. Literals are ignored (they are not columns).

        Args:
            on_clause: The ON clause expression from a JOIN

        Returns:
            List of (table_ref_or_None, column_name) tuples
        """
        columns: List[Tuple[Optional[str], str]] = []
        for node in on_clause.walk():
            if isinstance(node, exp.Column):
                table_ref = node.table if node.table else None
                col_name = node.name
                columns.append((table_ref, col_name))
        return columns

    def _get_join_type(self, join: exp.Join) -> str:
        """
        Extract the join type string from a sqlglot Join node.

        Args:
            join: The sqlglot Join expression

        Returns:
            Join type string like "inner", "left", "right", "full", "cross"
        """
        side = join.side
        kind = join.kind

        if side:
            return side.lower()
        if kind:
            return kind.lower()
        return "inner"

    def _get_join_right_table(self, join: exp.Join, unit: QueryUnit) -> Optional[str]:
        """
        Extract the right-side table name or alias from a JOIN clause.

        The join's `this` contains the table being joined (the right side).
        Uses the table's alias if present, otherwise the table name.

        Args:
            join: The sqlglot Join expression
            unit: The QueryUnit (for alias_mapping lookup)

        Returns:
            The alias or name of the right-side table, or None
        """
        table_node = join.this

        # Handle subquery case
        if isinstance(table_node, exp.Subquery):
            alias = table_node.alias
            if alias:
                return str(alias)
            return None

        # Handle table case
        if isinstance(table_node, exp.Table):
            # Prefer alias over table name
            if hasattr(table_node, "alias") and table_node.alias:
                return str(table_node.alias)
            return table_node.name

        return None

    def _extract_where_columns(self, condition: exp.Expression):
        """Extract column refs from WHERE condition, skipping exp.Subquery subtrees."""
        subquery_columns: set = set()
        for subq in condition.find_all(exp.Subquery):
            for col in subq.find_all(exp.Column):
                subquery_columns.add(id(col))

        columns = []
        for col in condition.find_all(exp.Column):
            if id(col) not in subquery_columns:
                table_ref = col.table if col.table else None
                columns.append((table_ref, col.name))
        return columns

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

    def _promote_dedup_qualify_if_applicable(self, select_node: exp.Select, unit: QueryUnit):
        """
        Promote dedup qualify info from a subquery-based WHERE pattern (Gap 2).

        Detects the common dedup pattern:
            SELECT ... FROM (SELECT *, ROW_NUMBER() OVER (...) AS rn FROM t) WHERE rn = 1
        and promotes it to qualify_info on the outer unit.

        Only ranking functions (ROW_NUMBER, RANK, DENSE_RANK, NTILE) are eligible.
        Comparison operators =, <=, < against a literal are recognized.

        Args:
            select_node: The SELECT expression
            unit: The query unit to potentially add qualify_info to
        """
        where_clause = select_node.args.get("where")
        if not where_clause or unit.qualify_info:
            return

        for dep_unit_id in unit.depends_on_units:
            dep_unit = self.unit_graph.units.get(dep_unit_id)
            if not dep_unit or not dep_unit.ranking_window_columns:
                continue

            for node in where_clause.walk():
                if isinstance(node, (exp.EQ, exp.LTE, exp.LT)):
                    left, right = node.left, node.right
                    col_name = None
                    if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                        col_name = left.name
                    elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
                        col_name = right.name

                    if col_name and col_name in dep_unit.ranking_window_columns:
                        window_meta = dep_unit.ranking_window_columns[col_name]
                        unit.qualify_info = {
                            "condition": where_clause.this.sql(),
                            "partition_columns": list(window_meta["partition_by"]),
                            "order_columns": [
                                c["column"] if isinstance(c, dict) else c
                                for c in window_meta["order_by"]
                            ],
                            "window_functions": [window_meta["function"]],
                            "promoted_from_subquery": True,
                        }
                        return

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
        Parse window functions in SELECT clause.

        Delegates to WindowFunctionsParser for the actual implementation.

        Args:
            select_node: The SELECT expression
            unit: The query unit to store window info
        """
        self._window_functions.parse(select_node, unit)

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
