"""
Recursive query parser for SQL statements.

Parses SQL queries recursively to identify all query units (CTEs, subqueries, main query)
and builds a QueryUnitGraph representing the query structure.
"""

from typing import List, Optional, Tuple, Union

import sqlglot
from sqlglot import exp

from .models import (
    JoinPredicateInfo,
    QueryUnit,
    QueryUnitGraph,
    QueryUnitType,
    WherePredicateInfo,
)

# Import helper classes for composition-based decomposition
from .query_parser_helpers import (
    FromClauseParser,
    GroupingParser,
    MergeParser,
    PivotUnpivotParser,
    RecursiveCTEParser,
    SetOperationsParser,
    SpecialSourcesHandler,
    SubqueryParser,
    WindowFunctionsParser,
)

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
        self._recursive_cte = RecursiveCTEParser(self)
        self._set_operations = SetOperationsParser(self)
        self._pivot_unpivot = PivotUnpivotParser(self)
        self._merge = MergeParser(self)
        self._grouping = GroupingParser(self)
        self._subqueries = SubqueryParser(self)

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
        """Parse UNION/INTERSECT/EXCEPT set operations. Delegates to SetOperationsParser."""
        return self._set_operations.parse(set_node, operation_type, name, parent_unit, depth)

    def _parse_pivot(
        self,
        pivot_node: exp.Pivot,
        name: str,
        parent_unit: QueryUnit,
        depth: int,
        table_node,  # Can be exp.Table or exp.Subquery
    ) -> QueryUnit:
        """Parse PIVOT operation. Delegates to PivotUnpivotParser."""
        return self._pivot_unpivot.parse_pivot(pivot_node, name, parent_unit, depth, table_node)

    def _parse_unpivot(
        self,
        unpivot_node: exp.Pivot,  # Note: sqlglot uses Pivot class for both PIVOT and UNPIVOT
        name: str,
        parent_unit: QueryUnit,
        depth: int,
        table_node,  # Can be exp.Table or exp.Subquery
    ) -> QueryUnit:
        """Parse UNPIVOT operation. Delegates to PivotUnpivotParser."""
        return self._pivot_unpivot.parse_unpivot(unpivot_node, name, parent_unit, depth, table_node)

    def _parse_merge_statement(
        self,
        merge_node: exp.Merge,
        name: str,
        depth: int,
    ) -> QueryUnit:
        """Parse MERGE INTO statement. Delegates to MergeParser."""
        return self._merge.parse(merge_node, name, depth)

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
        """Parse subqueries in WHERE clause. Delegates to SubqueryParser."""
        self._subqueries.parse_where_subqueries(where_node, parent_unit, depth)

    def _parse_having_subqueries(
        self, having_node: exp.Expression, parent_unit: QueryUnit, depth: int
    ):
        """Parse subqueries in HAVING clause. Delegates to SubqueryParser."""
        self._subqueries.parse_having_subqueries(having_node, parent_unit, depth)

    def _parse_qualify_clause(self, qualify_node: exp.Qualify, unit: QueryUnit):
        """Parse QUALIFY clause. Delegates to SubqueryParser."""
        self._subqueries.parse_qualify_clause(qualify_node, unit)

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
        """Parse GROUP BY clause for GROUPING SETS, CUBE, ROLLUP. Delegates to GroupingParser."""
        self._grouping.parse_grouping_sets(group_clause, unit)

    def _parse_window_functions(self, select_node: exp.Select, unit: QueryUnit):
        """Parse window functions in SELECT clause. Delegates to WindowFunctionsParser."""
        self._window_functions.parse(select_node, unit)

    def _parse_select_subqueries(self, expr: exp.Expression, parent_unit: QueryUnit, depth: int):
        """Parse scalar subqueries in SELECT clause. Delegates to SubqueryParser."""
        self._subqueries.parse_select_subqueries(expr, parent_unit, depth)

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
    # Recursive CTE Parsing (delegated to RecursiveCTEParser)
    # ============================================================================

    def _is_recursive_cte(self, query: exp.Expression, cte_name: str) -> bool:
        """Check if a CTE is recursive (references itself). Delegates to RecursiveCTEParser."""
        return self._recursive_cte.is_recursive_cte(query, cte_name)

    def _parse_recursive_cte(
        self,
        cte: exp.CTE,
        cte_name: str,
        parent_unit: QueryUnit,
        depth: int,
    ) -> QueryUnit:
        """Parse a recursive CTE. Delegates to RecursiveCTEParser."""
        return self._recursive_cte.parse(cte, cte_name, parent_unit, depth)

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
