"""
Recursive lineage builder for SQL column lineage.

Builds complete column lineage graphs by recursively tracing through query units.
Includes SQLColumnTracer wrapper for backward compatibility (re-exported from sql_column_tracer).
"""

from typing import Dict, List, Optional, Set, Tuple

from sqlglot import exp

from .aggregate_parser import (
    has_star_in_aggregate,
)
from .column_extractor import (
    ExtractionContext,
    extract_merge_columns,
    extract_pivot_columns,
    extract_union_columns,
    extract_unpivot_columns,
)

# ============================================================================
# Import utilities from lineage_utils.py
# Re-export for backward compatibility
# ============================================================================
from .lineage_utils import (  # noqa: F401, E402
    # Aggregate registry and functions
    AGGREGATE_REGISTRY,
    # JSON constants
    JSON_EXPRESSION_TYPES,
    JSON_FUNCTION_NAMES,
    # Type definitions
    BackwardLineageResult,
    SourceColumnRef,
    # Schema qualification functions
    _convert_to_nested_schema,
    # JSON functions
    _extract_json_path,
    # Nested access functions
    _extract_nested_path_from_expression,
    _find_json_function_ancestor,
    _find_nested_access_ancestor,
    _get_aggregate_type,
    _get_json_function_name,
    _is_complex_aggregate,
    _is_json_extract_function,
    _is_nested_access_expression,
    _normalize_json_path,
    _qualify_sql_with_schema,
)
from .metadata_parser import MetadataExtractor
from .models import (
    ColumnEdge,
    ColumnLineageGraph,
    ColumnNode,
    IssueCategory,
    IssueSeverity,
    QueryUnit,
    QueryUnitType,
    ValidationIssue,
)
from .node_factory import (
    create_column_node,
    find_or_create_table_column_node,
    get_node_key,
)
from .query_parser import RecursiveQueryParser

# ============================================================================
# Part 1: Recursive Lineage Builder
# ============================================================================


class RecursiveLineageBuilder:
    """
    Build column lineage graph by recursively tracing through query units.
    """

    def __init__(
        self,
        sql_query: str,
        external_table_columns: Optional[Dict[str, List[str]]] = None,
        dialect: str = "bigquery",
        query_id: Optional[str] = None,
    ):
        self.sql_query = sql_query
        self.external_table_columns = external_table_columns or {}
        self.dialect = dialect
        self.query_id = query_id

        # Qualify unqualified columns using schema info before parsing
        # This ensures columns like "order_date" in a JOIN get the correct table prefix
        qualified_sql = _qualify_sql_with_schema(sql_query, self.external_table_columns, dialect)

        # Parse query structure using qualified SQL
        parser = RecursiveQueryParser(qualified_sql, dialect=dialect)
        self.unit_graph = parser.parse()

        # Column lineage graph (to be built)
        self.lineage_graph = ColumnLineageGraph()

        # Cache for resolved columns per unit
        self.unit_columns_cache: Dict[str, List[ColumnNode]] = {}

        # Metadata extractor for parsing SQL comments
        self.metadata_extractor = MetadataExtractor()

    def build(self) -> ColumnLineageGraph:
        """
        Build complete column lineage graph.

        Algorithm:
        1. Process units in topological order (bottom-up)
        2. For each unit:
           a. Resolve input columns from dependencies
           b. Extract output columns
           c. Create column nodes
           d. Create edges
        3. Return final graph
        """
        # Get units in dependency order (leaves first)
        ordered_units = self.unit_graph.get_topological_order()

        for unit in ordered_units:
            self._process_unit(unit)

        return self.lineage_graph

    def _process_unit(self, unit: QueryUnit):
        """
        Process a single query unit and add its nodes/edges to lineage graph.
        This is the core recursive lineage building method.
        """
        # 1. Get output columns from this unit's SELECT
        output_cols = self._extract_output_columns(unit)

        # 2. Run validation checks on this unit
        self._run_validations(unit, output_cols)

        # 3. For each output column, trace to its sources
        for col_info in output_cols:
            # Create output node
            output_node = create_column_node(
                unit=unit,
                col_info=col_info,
                metadata_extractor=self.metadata_extractor,
                is_output=True,
            )
            self.lineage_graph.add_node(output_node)

            # 4. Trace dependencies recursively
            self._trace_column_dependencies(unit=unit, output_node=output_node, col_info=col_info)

        # Cache this unit's columns for parent units to reference
        self.unit_columns_cache[unit.unit_id] = [
            self.lineage_graph.nodes[get_node_key(unit, col_info)] for col_info in output_cols
        ]

        # 5. Create lateral correlation edges if this is a LATERAL subquery
        if unit.is_lateral and unit.correlated_columns:
            self._create_lateral_correlation_edges(unit)

        # 6. Create QUALIFY clause edges for window function columns
        if unit.qualify_info:
            self._create_qualify_edges(unit, output_cols)

        # 7. Create GROUPING SETS/CUBE/ROLLUP edges for grouping columns
        if unit.grouping_config:
            self._create_grouping_edges(unit, output_cols)

        # 8. Create window function edges for PARTITION BY, ORDER BY columns
        if unit.window_info:
            self._create_window_function_edges(unit, output_cols)

    def _create_window_function_edges(self, unit: QueryUnit, output_cols: List[Dict]):
        """
        Create edges for columns used in window functions.

        Window functions have dependencies on:
        - Function arguments (the columns being aggregated/calculated)
        - PARTITION BY columns (determine row grouping)
        - ORDER BY columns (determine row ordering within partitions)

        Args:
            unit: The query unit with window_info
            output_cols: The output columns of this unit
        """
        window_info = unit.window_info
        if not window_info:
            return

        windows = window_info.get("windows", [])

        for window_def in windows:
            output_column = window_def.get("output_column")
            func_name = window_def.get("function", "")
            arguments = window_def.get("arguments", [])
            partition_by = window_def.get("partition_by", [])
            order_by = window_def.get("order_by", [])
            frame_type = window_def.get("frame_type")
            frame_start = window_def.get("frame_start")
            frame_end = window_def.get("frame_end")

            # Find the output node for this window function
            output_node = None
            for col_info in output_cols:
                col_name = col_info.get("name") or col_info.get("alias")
                if col_name == output_column:
                    node_key = get_node_key(unit, col_info)
                    if node_key in self.lineage_graph.nodes:
                        output_node = self.lineage_graph.nodes[node_key]
                        break

            if not output_node:
                continue

            # 1. Create edges for function arguments (window_aggregate)
            for arg_col in arguments:
                source_node = self._resolve_window_column(unit, arg_col)
                if source_node:
                    edge = ColumnEdge(
                        from_node=source_node,
                        to_node=output_node,
                        edge_type="window_aggregate",
                        transformation=f"{func_name}({arg_col})",
                        context="WINDOW",
                        is_window_function=True,
                        window_role="aggregate",
                        window_function=func_name,
                        window_frame_type=frame_type,
                        window_frame_start=frame_start,
                        window_frame_end=frame_end,
                    )
                    self.lineage_graph.add_edge(edge)

            # 2. Create edges for PARTITION BY columns
            for part_col in partition_by:
                source_node = self._resolve_window_column(unit, part_col)
                if source_node:
                    edge = ColumnEdge(
                        from_node=source_node,
                        to_node=output_node,
                        edge_type="window_partition",
                        transformation=f"PARTITION BY {part_col}",
                        context="WINDOW",
                        is_window_function=True,
                        window_role="partition",
                        window_function=func_name,
                        window_frame_type=frame_type,
                        window_frame_start=frame_start,
                        window_frame_end=frame_end,
                    )
                    self.lineage_graph.add_edge(edge)

            # 3. Create edges for ORDER BY columns
            for order_col_info in order_by:
                order_col = (
                    order_col_info.get("column")
                    if isinstance(order_col_info, dict)
                    else order_col_info
                )
                direction = (
                    order_col_info.get("direction", "asc")
                    if isinstance(order_col_info, dict)
                    else "asc"
                )
                nulls = (
                    order_col_info.get("nulls", "last")
                    if isinstance(order_col_info, dict)
                    else "last"
                )

                source_node = self._resolve_window_column(unit, order_col)
                if source_node:
                    edge = ColumnEdge(
                        from_node=source_node,
                        to_node=output_node,
                        edge_type="window_order",
                        transformation=f"ORDER BY {order_col} {direction.upper()}",
                        context="WINDOW",
                        is_window_function=True,
                        window_role="order",
                        window_function=func_name,
                        window_frame_type=frame_type,
                        window_frame_start=frame_start,
                        window_frame_end=frame_end,
                        window_order_direction=direction,
                        window_order_nulls=nulls,
                    )
                    self.lineage_graph.add_edge(edge)

    def _resolve_window_column(self, unit: QueryUnit, col_ref: str) -> Optional[ColumnNode]:
        """
        Resolve a column reference from a window function to a ColumnNode.

        Args:
            unit: The query unit
            col_ref: Column reference like "amount" or "orders.amount"

        Returns:
            ColumnNode or None if not found
        """
        # Reuse the QUALIFY column resolution logic
        return self._resolve_qualify_column(unit, col_ref)

    def _create_grouping_edges(self, unit: QueryUnit, output_cols: List[Dict]):
        """
        Create edges for columns used in GROUPING SETS/CUBE/ROLLUP.

        These constructs generate multiple grouping levels in a single query.
        The columns used in grouping affect output aggregations.

        Args:
            unit: The query unit with grouping config
            output_cols: The output columns of this unit
        """
        grouping_config = unit.grouping_config
        if not grouping_config:
            return

        grouping_type = grouping_config.get("grouping_type", "")
        grouping_columns = grouping_config.get("grouping_columns", [])

        # Get the first aggregate output column as the target for grouping edges
        # (Grouping affects all aggregate columns)
        output_node = None
        for col_info in output_cols:
            if col_info.get("is_aggregate") or col_info.get("type") == "aggregate":
                node_key = get_node_key(unit, col_info)
                if node_key in self.lineage_graph.nodes:
                    output_node = self.lineage_graph.nodes[node_key]
                    break

        # If no aggregate found, use first non-star output column
        if not output_node:
            for col_info in output_cols:
                if not col_info.get("is_star"):
                    node_key = get_node_key(unit, col_info)
                    if node_key in self.lineage_graph.nodes:
                        output_node = self.lineage_graph.nodes[node_key]
                        break

        if not output_node:
            return

        # Create edges for each grouping column
        for col_ref in grouping_columns:
            source_node = self._resolve_grouping_column(unit, col_ref)
            if source_node:
                edge = ColumnEdge(
                    from_node=source_node,
                    to_node=output_node,
                    edge_type=f"grouping_{grouping_type}",
                    transformation=f"grouping_{grouping_type}",
                    context=grouping_type.upper(),
                    is_grouping_column=True,
                    grouping_type=grouping_type,
                )
                self.lineage_graph.add_edge(edge)

    def _resolve_grouping_column(self, unit: QueryUnit, col_ref: str) -> Optional[ColumnNode]:
        """
        Resolve a column reference from GROUPING to a ColumnNode.

        Args:
            unit: The query unit
            col_ref: Column reference like "region" or "sales_data.region"

        Returns:
            ColumnNode or None if not found
        """
        # Reuse the QUALIFY column resolution logic
        return self._resolve_qualify_column(unit, col_ref)

    def _create_qualify_edges(self, unit: QueryUnit, output_cols: List[Dict]):
        """
        Create edges for columns used in QUALIFY clause.

        QUALIFY filters rows based on window function results. The columns used in
        PARTITION BY and ORDER BY affect which rows are returned, so we create
        edges from these columns to the output columns.

        Args:
            unit: The query unit with QUALIFY info
            output_cols: The output columns of this unit
        """
        qualify_info = unit.qualify_info
        if not qualify_info:
            return

        partition_columns = qualify_info.get("partition_columns", [])
        order_columns = qualify_info.get("order_columns", [])
        window_functions = qualify_info.get("window_functions", [])
        func_name = window_functions[0] if window_functions else "WINDOW"

        # Get the first non-star output column as the target for qualify edges
        # (QUALIFY affects all output columns by filtering rows)
        output_node = None
        for col_info in output_cols:
            if not col_info.get("is_star"):
                node_key = get_node_key(unit, col_info)
                if node_key in self.lineage_graph.nodes:
                    output_node = self.lineage_graph.nodes[node_key]
                    break

        if not output_node:
            return

        # Create edges for PARTITION BY columns
        for col_ref in partition_columns:
            source_node = self._resolve_qualify_column(unit, col_ref)
            if source_node:
                edge = ColumnEdge(
                    from_node=source_node,
                    to_node=output_node,
                    edge_type="qualify_partition",
                    transformation="qualify_partition",
                    context="QUALIFY",
                    expression=qualify_info.get("condition"),
                    is_qualify_column=True,
                    qualify_context="partition",
                    qualify_function=func_name,
                )
                self.lineage_graph.add_edge(edge)

        # Create edges for ORDER BY columns
        for col_ref in order_columns:
            source_node = self._resolve_qualify_column(unit, col_ref)
            if source_node:
                edge = ColumnEdge(
                    from_node=source_node,
                    to_node=output_node,
                    edge_type="qualify_order",
                    transformation="qualify_order",
                    context="QUALIFY",
                    expression=qualify_info.get("condition"),
                    is_qualify_column=True,
                    qualify_context="order",
                    qualify_function=func_name,
                )
                self.lineage_graph.add_edge(edge)

    def _resolve_qualify_column(self, unit: QueryUnit, col_ref: str) -> Optional[ColumnNode]:
        """
        Resolve a column reference from QUALIFY to a ColumnNode.

        Args:
            unit: The query unit
            col_ref: Column reference like "customer_id" or "orders.customer_id"

        Returns:
            ColumnNode or None if not found
        """
        # Parse table.column format
        if "." in col_ref:
            parts = col_ref.split(".", 1)
            table_ref, col_name = parts
        else:
            table_ref = None
            col_name = col_ref

        # Try to resolve as a source unit
        source_unit = self._resolve_source_unit(unit, table_ref) if table_ref else None
        if source_unit:
            return self._find_column_in_unit(source_unit, col_name)

        # Try as base table
        base_table = self._resolve_base_table_name(unit, table_ref) if table_ref else None
        if base_table:
            return find_or_create_table_column_node(self.lineage_graph, base_table, col_name)

        # Try without table qualifier - infer from dependencies
        if not table_ref and unit.depends_on_tables:
            for table in unit.depends_on_tables:
                node = find_or_create_table_column_node(self.lineage_graph, table, col_name)
                if node:
                    return node

        # Fallback: use table_ref directly if provided
        if table_ref:
            return find_or_create_table_column_node(self.lineage_graph, table_ref, col_name)

        return None

    def _create_lateral_correlation_edges(self, unit: QueryUnit):
        """
        Create correlation edges for a LATERAL subquery.

        For each correlated column (reference to outer table), create an edge
        showing the correlation relationship.
        """
        lateral_alias = unit.name or ""

        for correlated_col in unit.correlated_columns:
            # Parse table.column format
            parts = correlated_col.split(".", 1)
            if len(parts) == 2:
                table_name, col_name = parts

                # Create source node for the correlated column
                source_node = find_or_create_table_column_node(
                    self.lineage_graph, table_name, col_name
                )

                # Create a correlation context node for the LATERAL subquery
                # This represents the fact that the LATERAL uses this column for correlation
                correlation_node = ColumnNode(
                    full_name=f"{lateral_alias}._correlation.{col_name}",
                    column_name=f"_correlation.{col_name}",
                    table_name=lateral_alias,
                    layer="correlation",
                    node_type="correlation",
                    is_star=False,
                )
                self.lineage_graph.add_node(correlation_node)

                # Create correlation edge
                edge = ColumnEdge(
                    from_node=source_node,
                    to_node=correlation_node,
                    edge_type="lateral_correlation",
                    transformation="lateral_correlation",
                    context="LATERAL",
                    expression=f"LATERAL correlation: {correlated_col}",
                    is_lateral_correlation=True,
                    lateral_alias=lateral_alias,
                )
                self.lineage_graph.add_edge(edge)

    def _extract_output_columns(self, unit: QueryUnit) -> List[Dict]:
        """
        Extract output columns from a query unit's SELECT.
        Handles star notation and expands stars when source columns are known.

        Also handles special query types: UNION, PIVOT, UNPIVOT.
        """
        # Handle special query types that don't have a select_node
        ctx = ExtractionContext(
            unit_graph=self.unit_graph,
            unit_columns_cache=self.unit_columns_cache,
            external_table_columns=self.external_table_columns,
            lineage_graph=self.lineage_graph,
        )
        if unit.unit_type in (QueryUnitType.UNION, QueryUnitType.INTERSECT, QueryUnitType.EXCEPT):
            return extract_union_columns(ctx, unit)
        elif unit.unit_type == QueryUnitType.PIVOT:
            return extract_pivot_columns(ctx, unit)
        elif unit.unit_type == QueryUnitType.UNPIVOT:
            return extract_unpivot_columns(ctx, unit)
        elif unit.unit_type == QueryUnitType.MERGE:
            return extract_merge_columns(ctx, unit)

        select_node = unit.select_node
        output_cols = []

        # Guard against None select_node or missing expressions
        if not select_node or not select_node.expressions:
            return output_cols

        for i, expr in enumerate(select_node.expressions):
            col_info = {"index": i, "ast_node": expr}

            # Check if star
            is_star = isinstance(expr, exp.Star) or (
                isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star)
            )

            if is_star:
                # Star expression
                star_alias = None
                if isinstance(expr, exp.Column) and hasattr(expr, "table") and expr.table:
                    # expr.table can be a string or an Identifier object
                    star_alias = expr.table.name if hasattr(expr.table, "name") else str(expr.table)

                # Resolve star source
                if star_alias:
                    # Qualified star - find source unit or table
                    source_unit = self._resolve_source_unit(unit, star_alias)
                    if source_unit:
                        col_info["source_unit"] = source_unit.unit_id
                        col_info["source_table"] = source_unit.name
                    else:
                        # Resolve alias to actual base table name
                        actual_table = self._resolve_base_table_name(unit, star_alias)
                        col_info["source_table"] = actual_table if actual_table else star_alias
                else:
                    # Unqualified star - get default FROM table
                    default_table = self._get_default_from_table(unit)
                    if default_table:
                        col_info["source_table"] = default_table
                        # Also try to resolve as a source unit (CTE/subquery)
                        source_unit = self._resolve_source_unit(unit, default_table)
                        if source_unit:
                            col_info["source_unit"] = source_unit.unit_id

                    # VALIDATION: Check for unqualified SELECT * with multiple tables
                    table_count = len(unit.depends_on_tables) + len(unit.depends_on_units)
                    if table_count > 1:
                        # This is ambiguous - which table does * refer to?
                        tables = [str(t) for t in unit.depends_on_tables] + [
                            str(self.unit_graph.units[uid].name)
                            for uid in unit.depends_on_units
                            if uid in self.unit_graph.units
                        ]
                        issue = ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES,
                            message=f"Unqualified SELECT * with {table_count} tables: {', '.join(tables)}. Cannot determine column sources.",
                            query_id=self.query_id,
                            location="SELECT clause",
                            suggestion=f"Use qualified star (e.g., SELECT {tables[0]}.*, {tables[1]}.* ...) or list columns explicitly",
                            context={"tables": tables, "table_count": table_count},
                        )
                        self.lineage_graph.add_issue(issue)

                col_info["is_star"] = True
                col_info["name"] = "*"
                col_info["type"] = "star"
                col_info["expression"] = expr.sql()

                # Handle EXCEPT/REPLACE
                # Note: sqlglot 28.x uses 'except_' and 'replace_' (with underscore)
                # while sqlglot 27.x uses 'except' and 'replace' (without underscore)
                star_expr = expr.this if isinstance(expr, exp.Column) else expr
                except_clause = None
                if hasattr(star_expr, "args"):
                    # Try both old and new sqlglot key names
                    except_clause = star_expr.args.get("except") or star_expr.args.get("except_")
                if except_clause:
                    col_info["except_columns"] = {col.name for col in except_clause}
                    col_info["type"] = "star_except"
                else:
                    col_info["except_columns"] = set()

                replace_clause = None
                if hasattr(star_expr, "args"):
                    # Try both old and new sqlglot key names
                    replace_clause = star_expr.args.get("replace") or star_expr.args.get("replace_")
                if replace_clause:
                    col_info["replace_columns"] = {
                        replace_expr.alias: replace_expr.sql()
                        for replace_expr in replace_clause
                        if hasattr(replace_expr, "alias")
                    }
                else:
                    col_info["replace_columns"] = {}

                # STAR EXPANSION: Try to expand star if we know the source columns
                # This only applies to the main query output (not CTEs/subqueries)
                if unit.unit_type == QueryUnitType.MAIN_QUERY:
                    expanded_cols = None

                    # Case 1: Source is a CTE or subquery (internal query unit)
                    if col_info.get("source_unit"):
                        source_unit_id = str(col_info["source_unit"])
                        expanded_cols = self._try_expand_star(unit, source_unit_id, col_info)

                    # Case 2: Source is an external table with known columns
                    elif col_info.get("source_table"):
                        source_table = col_info["source_table"]
                        # Type guard: ensure source_table is a string
                        if isinstance(source_table, str):
                            expanded_cols = self._try_expand_star_from_external_table(
                                unit, source_table, col_info
                            )

                    if expanded_cols:
                        # Replace star with expanded columns
                        output_cols.extend(expanded_cols)
                        continue  # Skip adding the star itself

            else:
                # Regular column
                col_info["is_star"] = False
                col_info["name"] = expr.alias_or_name
                col_info["type"] = self._determine_expression_type(expr)
                col_info["expression"] = expr.sql()
                col_info["source_columns"] = self._extract_source_column_refs(expr)

            output_cols.append(col_info)

        return output_cols

    def _trace_column_dependencies(self, unit: QueryUnit, output_node: ColumnNode, col_info: Dict):
        """Dispatch column dependency tracing to the appropriate strategy."""
        from .trace_strategies import (
            trace_aggregate_star,
            trace_merge_columns,
            trace_regular_columns,
            trace_set_operation,
            trace_star_passthrough,
        )

        source_columns = col_info.get("source_columns", [])

        # Branch 1: Star passthrough
        if col_info.get("is_star"):
            trace_star_passthrough(
                self.lineage_graph,
                unit,
                output_node,
                col_info,
                self.unit_graph,
            )
            return

        # Branch 2: COUNT(*) / aggregate with star
        if has_star_in_aggregate(col_info.get("ast_node")):
            trace_aggregate_star(
                self.lineage_graph,
                unit,
                output_node,
                col_info,
                self.unit_columns_cache,
                self.external_table_columns,
                self.unit_graph,
            )
            return

        # Branch 3: UNION/INTERSECT/EXCEPT
        if col_info.get("type") == "union_column" and "source_branches" in col_info:
            trace_set_operation(
                self.lineage_graph,
                unit,
                output_node,
                col_info,
                self.unit_columns_cache,
            )
            return

        # Branch 4: MERGE
        if col_info.get("type") in ("merge_match", "merge_update", "merge_insert"):
            trace_merge_columns(
                self.lineage_graph,
                unit,
                output_node,
                col_info,
                self._resolve_source_unit,
                self._resolve_base_table_name,
                self._find_column_in_unit,
            )
            return

        # Branch 5: Regular columns
        trace_regular_columns(
            self.lineage_graph,
            unit,
            output_node,
            col_info,
            source_columns,
            self._resolve_source_unit,
            self._resolve_base_table_name,
            self._find_column_in_unit,
            self._get_default_from_table,
        )

    def _resolve_source_unit(
        self, current_unit: QueryUnit, table_ref: Optional[str]
    ) -> Optional[QueryUnit]:
        """
        Resolve a table reference to a query unit (CTE or subquery).

        This checks:
        1. Alias mappings in current unit (handles aliases like "b" -> "base")
        2. Units this current_unit depends on
        3. CTEs in parent units (scope traversal)
        """
        if not table_ref:
            return None

        # First, resolve alias to actual name
        actual_name = table_ref
        if table_ref in current_unit.alias_mapping:
            mapped_name, is_unit = current_unit.alias_mapping[table_ref]
            if is_unit:
                # This is an alias to a unit (CTE or subquery), use the actual name
                actual_name = mapped_name
            else:
                # This is a base table, not a unit
                return None

        # Check direct dependencies using actual name
        for dep_unit_id in current_unit.depends_on_units:
            dep_unit = self.unit_graph.units[dep_unit_id]
            if dep_unit.name == actual_name:
                return dep_unit

        # Check CTEs in parent scope (for nested subqueries)
        if current_unit.parent_unit:
            return self._resolve_source_unit(current_unit.parent_unit, actual_name)

        return None

    def _resolve_base_table_name(
        self, current_unit: QueryUnit, table_ref: Optional[str]
    ) -> Optional[str]:
        """
        Resolve a table reference (which might be an alias) to the actual base table name.

        For example, if the query has "FROM products p", this resolves "p" -> "products".

        Args:
            current_unit: The current query unit
            table_ref: The table reference (might be an alias or actual table name)

        Returns:
            The actual base table name, or None if not found
        """
        if not table_ref:
            return None

        # Check if this is in the alias mapping
        if table_ref in current_unit.alias_mapping:
            mapped_name, is_unit = current_unit.alias_mapping[table_ref]
            if not is_unit:
                # This is a base table alias, return the actual table name
                return mapped_name
            else:
                # This is a unit (CTE/subquery), not a base table
                return None

        # If not in alias mapping, it might already be the actual table name
        # Check if it's in depends_on_tables
        if table_ref in current_unit.depends_on_tables:
            return table_ref

        return None

    def _find_column_in_unit(
        self, unit: QueryUnit, col_name: str, visited: Optional[set] = None
    ) -> Optional[ColumnNode]:
        """
        Find a column node in a processed unit's output.

        If the column is not explicitly defined but the unit has a star column
        (SELECT *), trace through to find the actual source column.
        """
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        if unit.unit_id in visited:
            return None
        visited.add(unit.unit_id)

        if unit.unit_id not in self.unit_columns_cache:
            return None

        # First, look for explicit column
        for node in self.unit_columns_cache[unit.unit_id]:
            if node.column_name == col_name:
                return node

        # Not found - check if this unit has a star column
        # If so, the column might come through the star from an upstream source
        star_node = None
        for node in self.unit_columns_cache[unit.unit_id]:
            if node.is_star:
                star_node = node
                break

        if star_node:
            # Find where the star comes from by looking at edges
            for edge in self.lineage_graph.edges:
                if edge.to_node.full_name == star_node.full_name:
                    source_node = edge.from_node
                    if source_node.is_star and source_node.table_name:
                        # The star comes from another unit - look up the column there
                        source_unit = self._get_unit_by_name(source_node.table_name)
                        if source_unit:
                            result = self._find_column_in_unit(source_unit, col_name, visited)
                            if result:
                                return result

        return None

    def _get_unit_by_name(self, name: str) -> Optional[QueryUnit]:
        """Get a QueryUnit by its name (CTE name or alias)"""
        for unit in self.unit_graph.units.values():
            if unit.name == name:
                return unit
        return None

    def _extract_source_column_refs(
        self, expr: exp.Expression
    ) -> List[
        Tuple[Optional[str], str, Optional[str], Optional[str], Optional[str], Optional[str]]
    ]:
        """
        Extract source column references from expression with JSON and nested access metadata.

        Args:
            expr: The SQL expression to analyze

        Returns:
            List of tuples: (table_ref, column_name, json_path, json_function, nested_path, access_type)
            - table_ref: Table/alias name or None for unqualified columns
            - column_name: Column name
            - json_path: Normalized JSON path (e.g., "$.address.city") or None
            - json_function: JSON function name (e.g., "JSON_EXTRACT") or None
            - nested_path: Normalized nested path (e.g., "[0].field") or None
            - access_type: "array", "map", "struct", "mixed" or None
        """
        refs: List[
            Tuple[Optional[str], str, Optional[str], Optional[str], Optional[str], Optional[str]]
        ] = []
        processed_columns: Set[int] = set()  # Track processed columns to avoid duplicates

        # First pass: process Dot and Bracket expressions to extract nested paths
        for node in expr.walk():
            if isinstance(node, (exp.Dot, exp.Bracket)):
                # Extract nested path info
                (
                    table_ref,
                    column_name,
                    nested_path,
                    access_type,
                ) = _extract_nested_path_from_expression(node)

                if column_name:
                    # Check for JSON function ancestor
                    json_path: Optional[str] = None
                    json_function: Optional[str] = None

                    # Find the base column to check for JSON ancestor
                    current = node
                    while current and not isinstance(current, exp.Column):
                        current = getattr(current, "this", None)

                    if current and isinstance(current, exp.Column):
                        processed_columns.add(id(current))
                        json_ancestor = _find_json_function_ancestor(current, expr)
                        if json_ancestor:
                            json_path = _extract_json_path(json_ancestor)
                            json_function = _get_json_function_name(json_ancestor)

                    refs.append(
                        (table_ref, column_name, json_path, json_function, nested_path, access_type)
                    )

        # Second pass: process remaining Column nodes (not part of nested access)
        for node in expr.walk():
            if isinstance(node, exp.Column):
                if id(node) in processed_columns:
                    continue  # Already processed as part of nested access

                # Check if this column is inside a Dot or Bracket we already processed
                nested_ancestor = _find_nested_access_ancestor(node, expr)
                if nested_ancestor:
                    continue  # Skip, will be handled by the nested access processing

                table_ref: Optional[str] = None
                if hasattr(node, "table") and node.table:
                    table_ref = (
                        str(node.table.name) if hasattr(node.table, "name") else str(node.table)
                    )

                # Check if this column is inside a JSON function
                json_path: Optional[str] = None
                json_function: Optional[str] = None

                json_ancestor = _find_json_function_ancestor(node, expr)
                if json_ancestor:
                    json_path = _extract_json_path(json_ancestor)
                    json_function = _get_json_function_name(json_ancestor)

                refs.append((table_ref, node.name, json_path, json_function, None, None))

        return refs

    def _determine_expression_type(self, expr: exp.Expression) -> str:
        """Classify expression type"""
        # Check if expression contains aggregates (walk the AST)
        for node in expr.walk():
            # Check for standard aggregate expressions
            if isinstance(
                node,
                (
                    exp.Count,
                    exp.Sum,
                    exp.Avg,
                    exp.Min,
                    exp.Max,
                    exp.ArrayAgg,
                    exp.GroupConcat,
                    exp.AggFunc,
                ),
            ):
                return "aggregate"
            elif isinstance(node, exp.Window):
                return "window"
            # Check for named aggregate functions
            elif isinstance(node, (exp.Anonymous, exp.Func)):
                func_name = ""
                if hasattr(node, "name") and node.name:
                    func_name = node.name
                elif hasattr(node, "sql_name"):
                    func_name = node.sql_name()
                if func_name and _get_aggregate_type(func_name):
                    return "aggregate"

        # Check top-level expression type
        if isinstance(expr, exp.Case):
            return "case"
        elif isinstance(expr, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
            return "arithmetic"
        elif isinstance(expr, exp.Column):
            return "direct_column"
        else:
            return "expression"

    def _get_default_from_table(self, unit: QueryUnit) -> Optional[str]:
        """Get the default FROM table for unqualified column references"""
        # Get the first table from depends_on_tables or depends_on_units
        if unit.depends_on_tables:
            return unit.depends_on_tables[0]
        elif unit.depends_on_units:
            # Get first unit's name
            first_unit = self.unit_graph.units[unit.depends_on_units[0]]
            return first_unit.name
        return None

    def _try_expand_star(
        self, unit: QueryUnit, source_unit_id: str, star_col_info: Dict
    ) -> Optional[List[Dict]]:
        """
        Try to expand a star into individual columns if the source columns are known.

        This is key for star expansion: when we SELECT * FROM cte3 and cte3 has
        explicit columns (id, name), we expand the star to output.id and output.name.

        Args:
            unit: The current query unit (main query)
            source_unit_id: The unit ID we're selecting * from
            star_col_info: The star column info dict

        Returns:
            List of expanded column dicts, or None if expansion not possible
        """
        # Get the source unit
        source_unit = self.unit_graph.units.get(source_unit_id)
        if not source_unit:
            return None

        # Check if source unit has already been processed (cached columns exist)
        if source_unit_id not in self.unit_columns_cache:
            return None

        source_columns = self.unit_columns_cache[source_unit_id]

        # Check if source has explicit columns (no stars)
        # If source itself has stars, we can't expand
        has_stars = any(col.is_star for col in source_columns)
        if has_stars:
            return None

        # Great! We can expand. Create individual column entries
        expanded: List[Dict] = []
        except_cols = star_col_info.get("except_columns", set())

        for source_col in source_columns:
            # Skip columns in EXCEPT clause
            if source_col.column_name in except_cols:
                continue

            # Create a new column info for this expanded column
            expanded_col = {
                "index": len(expanded),
                "ast_node": None,  # No specific AST node for expanded columns
                "is_star": False,  # This is now an explicit column
                "name": source_col.column_name,
                "type": "direct_column",  # Direct pass-through from source
                "expression": source_col.column_name,
                "source_columns": [(source_unit.name, source_col.column_name)],
                # Mark this as star-expanded so we can trace it properly
                "star_expanded": True,
                "star_source_unit": source_unit_id,
            }
            expanded.append(expanded_col)

        return expanded if expanded else None

    def _try_expand_star_from_external_table(
        self, unit: QueryUnit, source_table: str, star_col_info: Dict
    ) -> Optional[List[Dict]]:
        """
        Try to expand a star from an external table using external_table_columns.

        This handles the cross-query scenario where Query 2 does SELECT * FROM staging.orders,
        and staging.orders was created by Query 1 with known columns.

        Args:
            unit: The current query unit (main query)
            source_table: The external table name (e.g., "staging.orders")
            star_col_info: The star column info dict

        Returns:
            List of expanded column dicts, or None if expansion not possible
        """
        # Check if we have column information for this external table
        if source_table not in self.external_table_columns:
            # VALIDATION: Missing schema information for external table
            issue = ValidationIssue(
                severity=IssueSeverity.INFO,
                category=IssueCategory.STAR_WITHOUT_SCHEMA,
                message=f"SELECT * from external table '{source_table}' without known schema. Star cannot be expanded to individual columns.",
                query_id=self.query_id,
                location="SELECT clause",
                suggestion=f"Provide schema for '{source_table}' or list columns explicitly",
                context={
                    "table": source_table,
                    "available_schemas": list(self.external_table_columns.keys()),
                },
            )
            self.lineage_graph.add_issue(issue)
            return None

        column_names = self.external_table_columns[source_table]
        if not column_names:
            # VALIDATION: Empty schema information
            issue = ValidationIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.MISSING_SCHEMA_INFO,
                message=f"Table '{source_table}' has empty schema. Star cannot be expanded.",
                query_id=self.query_id,
                location="SELECT clause",
                suggestion=f"Check schema definition for '{source_table}'",
                context={"table": source_table},
            )
            self.lineage_graph.add_issue(issue)
            return None

        # Great! We can expand. Create individual column entries
        expanded: List[Dict] = []
        except_cols = star_col_info.get("except_columns", set())
        replace_cols = star_col_info.get("replace_columns", {})

        for col_name in column_names:
            # Skip columns in EXCEPT clause
            if col_name in except_cols:
                continue

            # Create a new column info for this expanded column
            expanded_col = {
                "index": len(expanded),
                "ast_node": None,  # No specific AST node for expanded columns
                "is_star": False,  # This is now an explicit column
                "name": col_name,
                "type": "direct_column",  # Direct pass-through from source
                "expression": replace_cols.get(col_name, col_name),  # Use REPLACE if specified
                "source_columns": [(source_table, col_name)],
                # Mark this as star-expanded so we can trace it properly
                "star_expanded": True,
                "star_source_table": source_table,
            }
            expanded.append(expanded_col)

        return expanded if expanded else None

    # ========================================================================
    # Validation Methods (Static Analysis)
    # ========================================================================

    def _validate_column_aliases(self, unit: QueryUnit, output_cols: List[Dict]) -> List[str]:
        """
        Validate that expressions have proper aliases.

        Rules:
        - Aggregates MUST have aliases (ERROR)
        - Window functions MUST have aliases (ERROR)
        - Other expressions SHOULD have aliases (WARNING)
        - Plain column references don't need aliases
        """
        warnings = []

        for _i, col_info in enumerate(output_cols):
            # Skip stars
            if col_info.get("is_star"):
                continue

            expr = col_info.get("ast_node")
            if not expr:
                continue

            # Check if this has an alias
            has_alias = bool(expr.alias)

            # Determine expression category
            is_plain_column = isinstance(expr, exp.Column) and not isinstance(expr.this, exp.Star)

            # Check for aggregates
            is_aggregate = False
            for node in expr.walk():
                if isinstance(node, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max, exp.AggFunc)):
                    is_aggregate = True
                    break

            # Check for window functions
            is_window = False
            for node in expr.walk():
                if isinstance(node, exp.Window):
                    is_window = True
                    break

            # Apply validation rules
            if is_aggregate and not has_alias:
                col_expr = expr.sql()
                warnings.append(
                    f"[SUGGESTION] Aggregate function in {unit.unit_id} lacks alias: '{col_expr}'. "
                    f"Add 'AS alias_name' for reliable column lineage across databases."
                )
            elif is_window and not has_alias:
                col_expr = expr.sql()
                warnings.append(
                    f"[SUGGESTION] Window function in {unit.unit_id} lacks alias: '{col_expr}'. "
                    f"Add 'AS alias_name' for reliable column lineage across databases."
                )
            elif not is_plain_column and not has_alias:
                col_expr = expr.sql()
                warnings.append(
                    f"[SUGGESTION] Expression in {unit.unit_id} lacks explicit alias: '{col_expr}'. "
                    f"Add 'AS alias_name' for better lineage clarity."
                )

        return warnings

    def _validate_unique_column_names(self, unit: QueryUnit, output_cols: List[Dict]) -> List[str]:
        """
        Validate that all output column names are unique.
        Duplicate column names make lineage ambiguous.
        """
        warnings = []
        col_names = []

        for col_info in output_cols:
            # Skip star columns (they expand to multiple columns)
            if col_info.get("is_star"):
                continue

            col_name = col_info["name"]

            # Check for duplicates
            if col_name in col_names:
                warnings.append(
                    f"[SUGGESTION] Duplicate column name '{col_name}' in {unit.unit_id}. "
                    f"Use explicit aliases to make lineage clearer."
                )
            else:
                col_names.append(col_name)

        return warnings

    def _validate_qualified_columns_in_joins(
        self, unit: QueryUnit, output_cols: List[Dict]
    ) -> List[str]:
        """
        Flag unqualified column references when query has JOINs.
        This is a readability/correctness issue - lineage should be obvious from SQL.

        When multiple tables are joined, unqualified column references can be ambiguous
        and may resolve incorrectly. This validation:
        1. Walks ALL expressions (not just top-level columns)
        2. Finds unqualified column references inside expressions
        3. Adds ValidationIssue objects for proper issue tracking
        """
        warnings = []

        # Skip validation for units without select_node (UNION, PIVOT, UNPIVOT)
        if unit.select_node is None:
            return warnings

        # Check if this query has JOINs or multiple tables
        joins = unit.select_node.args.get("joins", [])
        has_multiple_tables = (len(unit.depends_on_tables) + len(unit.depends_on_units) > 1) or (
            joins and len(joins) > 0
        )

        if not has_multiple_tables:
            return warnings

        # Collect all tables/schemas available for resolution
        available_tables = list(unit.depends_on_tables) + [
            self.unit_graph.units[uid].name for uid in unit.depends_on_units
        ]

        # Query has JOINs or multiple tables - check for unqualified columns
        for col_info in output_cols:
            # Skip stars (they're explicit about being unqualified)
            if col_info.get("is_star"):
                continue

            expr = col_info.get("ast_node")
            if not expr:
                continue

            output_col_name = col_info.get("name", "unknown")

            # Walk the ENTIRE expression to find all column references
            for node in expr.walk():
                if isinstance(node, exp.Column) and not isinstance(node.this, exp.Star):
                    # Check if column is unqualified (no table prefix)
                    table_ref = node.table if hasattr(node, "table") else None
                    col_name = node.name

                    if not table_ref:  # Unqualified!
                        # Add string warning for backward compatibility
                        warnings.append(
                            f"Unqualified column '{col_name}' in expression for '{output_col_name}' "
                            f"with multiple tables. Cannot determine source table."
                        )

                        # Also add proper ValidationIssue
                        issue = ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.UNQUALIFIED_COLUMN,
                            message=(
                                f"Unqualified column '{col_name}' in expression for '{output_col_name}'. "
                                f"With multiple tables ({', '.join(str(t) for t in available_tables)}), "
                                f"the source table is ambiguous."
                            ),
                            query_id=self.query_id,
                            location=f"SELECT clause: {output_col_name}",
                            suggestion=(
                                f"Qualify the column with table name: e.g., 'table_name.{col_name}'"
                            ),
                            context={
                                "column_name": col_name,
                                "output_column": output_col_name,
                                "available_tables": available_tables,
                                "expression": expr.sql() if expr else None,
                            },
                        )
                        self.lineage_graph.add_issue(issue)

        return warnings

    def _run_validations(self, unit: QueryUnit, output_cols: List[Dict]):
        """
        Run all validation checks on a query unit and add warnings to the lineage graph.
        """
        # Collect all warnings
        all_warnings = []

        # 1. Check column aliases
        all_warnings.extend(self._validate_column_aliases(unit, output_cols))

        # 2. Check for duplicate column names
        all_warnings.extend(self._validate_unique_column_names(unit, output_cols))

        # 3. Check for unqualified columns in JOINs
        all_warnings.extend(self._validate_qualified_columns_in_joins(unit, output_cols))

        # Add warnings to lineage graph
        for warning in all_warnings:
            self.lineage_graph.add_warning(warning)


# ============================================================================
# Part 2: SQLColumnTracer Wrapper (Backward Compatibility)
# Re-exported from sql_column_tracer.py for backward compatibility
# ============================================================================

from .sql_column_tracer import SQLColumnTracer  # noqa: F401, E402

__all__ = [
    "RecursiveLineageBuilder",
    "SQLColumnTracer",
    # Re-exported from lineage_utils for backward compatibility
    "SourceColumnRef",
    "BackwardLineageResult",
    "JSON_FUNCTION_NAMES",
    "JSON_EXPRESSION_TYPES",
    "_is_json_extract_function",
    "_get_json_function_name",
    "_extract_json_path",
    "_normalize_json_path",
    "_find_json_function_ancestor",
    "AGGREGATE_REGISTRY",
    "_get_aggregate_type",
    "_is_complex_aggregate",
    "_is_nested_access_expression",
    "_extract_nested_path_from_expression",
    "_find_nested_access_ancestor",
    "_convert_to_nested_schema",
    "_qualify_sql_with_schema",
]

# NOTE: The following is removed - SQLColumnTracer is now defined in sql_column_tracer.py
# This comment preserved for git history awareness


# End of module - SQLColumnTracer is now in sql_column_tracer.py
