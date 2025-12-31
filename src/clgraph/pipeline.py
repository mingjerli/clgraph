"""
Pipeline orchestration with integrated lineage analysis.

Contains Pipeline class for unified SQL workflow orchestration with:
- Table and column lineage
- Metadata propagation
- LLM-powered documentation
- Pipeline execution (sync/async)
- Airflow DAG generation
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sqlglot import exp

from .column import (
    PipelineLineageGraph,
    generate_description,
    propagate_metadata,
    propagate_metadata_backward,
)
from .lineage_builder import RecursiveLineageBuilder
from .models import (
    ColumnEdge,
    ColumnLineageGraph,
    ColumnNode,
    DescriptionSource,
    IssueCategory,
    IssueSeverity,
    ParsedQuery,
    ValidationIssue,
)
from .table import TableDependencyGraph


class PipelineLineageBuilder:
    """
    Builds unified lineage graph from multiple queries.
    Combines single-query lineage with cross-query connections.
    """

    def build(self, pipeline_or_graph) -> "Pipeline":
        """
        Build unified pipeline lineage graph.

        Args:
            pipeline_or_graph: Either a Pipeline instance to populate,
                              or a TableDependencyGraph (for backward compatibility)

        Returns:
            The populated Pipeline instance

        Algorithm:
        1. Topologically sort queries
        2. For each query (bottom-up):
           a. Run single-query lineage (RecursiveLineageBuilder)
           b. Add columns to pipeline graph
           c. Add within-query edges
        3. Add cross-query edges (connect tables)
        """
        # Handle backward compatibility: accept TableDependencyGraph directly
        if isinstance(pipeline_or_graph, TableDependencyGraph):
            pipeline = Pipeline._create_empty(pipeline_or_graph)
        else:
            pipeline = pipeline_or_graph

        table_graph = pipeline.table_graph

        # Step 1: Topological sort
        sorted_query_ids = table_graph.topological_sort()

        # Step 2: Process each query
        for query_id in sorted_query_ids:
            query = table_graph.queries[query_id]

            # Step 2a: Run single-query lineage
            try:
                # Extract SELECT statement from DDL/DML if needed
                sql_for_lineage = self._extract_select_from_query(query)

                if sql_for_lineage:
                    # Collect upstream table schemas from already-processed queries
                    external_table_columns = self._collect_upstream_table_schemas(
                        pipeline, query, table_graph
                    )

                    # RecursiveLineageBuilder handles parsing internally
                    # Pass external_table_columns so it can resolve * to actual columns
                    lineage_builder = RecursiveLineageBuilder(
                        sql_for_lineage,
                        external_table_columns=external_table_columns,
                        dialect=pipeline.dialect,
                        query_id=query_id,
                    )
                    query_lineage = lineage_builder.build()

                    # Store query lineage
                    pipeline.query_graphs[query_id] = query_lineage
                    query.query_lineage = query_lineage

                    # Step 2b: Add columns to pipeline graph
                    self._add_query_columns(pipeline, query, query_lineage)

                    # Step 2c: Add within-query edges
                    self._add_query_edges(pipeline, query, query_lineage)
                else:
                    # No SELECT to analyze (e.g., UPDATE without SELECT)
                    print(f"Info: Skipping lineage for {query_id} (no SELECT statement)")
            except Exception as e:
                # If lineage fails, skip this query
                print(f"Warning: Failed to build lineage for {query_id}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Step 3: Add cross-query edges
        self._add_cross_query_edges(pipeline)

        return pipeline

    def _expand_star_nodes_in_pipeline(
        self, pipeline: "Pipeline", query: ParsedQuery, nodes: list[ColumnNode]
    ) -> list[ColumnNode]:
        """
        Expand * nodes in output layer when upstream columns are known.

        For cross-query scenarios:
        - If query_1 does SELECT * EXCEPT (col1) FROM staging.table
        - And staging.table was created by query_0 with known columns
        - We should expand the * to the actual columns (minus excepted ones)

        This gives users precise column-level lineage instead of just *.
        """
        result = []

        # Find all input layer * nodes to get source table info
        input_star_nodes = {
            node.table_name: node for node in nodes if node.is_star and node.layer == "input"
        }

        for node in nodes:
            # Only expand output layer * nodes
            if not (node.is_star and node.layer == "output"):
                result.append(node)
                continue

            # Get the source table from the corresponding input * node
            # The output * has EXCEPT/REPLACE info, but we need the source table from input *
            source_table_name = None
            except_columns = node.except_columns
            replace_columns = node.replace_columns

            # Find which input table this output * is selecting from
            for input_table, input_star in input_star_nodes.items():
                # Check if the input * feeds into this output *
                # (in simple cases, there's only one input * per query)
                # Infer the fully qualified table name for the input table
                source_table_name = self._infer_table_name(input_star, query) or input_table
                break

            if not source_table_name:
                # Can't expand - keep the * node
                result.append(node)
                continue

            # Try to find upstream table columns
            upstream_columns = self._get_upstream_table_columns(pipeline, source_table_name)

            if not upstream_columns:
                # Can't expand - keep the * node
                result.append(node)
                continue

            # Expand the * to individual columns
            for upstream_col in upstream_columns:
                col_name = upstream_col.column_name

                # Skip excepted columns
                if col_name in except_columns:
                    continue

                # Create expanded column node
                # Get the properly inferred destination table name
                dest_table_name = self._infer_table_name(node, query) or node.table_name

                expanded_node = ColumnNode(
                    column_name=col_name,
                    table_name=dest_table_name,
                    full_name=f"{dest_table_name}.{col_name}",
                    unit_id=node.unit_id,
                    layer=node.layer,
                    query_id=node.query_id,
                    node_type="direct_column",
                    is_star=False,
                    # Check if this column is being replaced
                    expression=(
                        replace_columns.get(col_name, col_name)
                        if col_name in replace_columns
                        else col_name
                    ),
                    # Preserve metadata from upstream if available
                    description=upstream_col.description,
                    pii=upstream_col.pii,
                    owner=upstream_col.owner,
                    tags=upstream_col.tags.copy(),
                )
                result.append(expanded_node)

        return result

    def _collect_upstream_table_schemas(
        self,
        pipeline: "Pipeline",
        query: ParsedQuery,
        table_graph: TableDependencyGraph,
    ) -> Dict[str, List[str]]:
        """
        Collect column names from upstream tables that this query reads from.

        This is used to pass to RecursiveLineageBuilder so it can resolve * properly.

        Args:
            pipeline: Pipeline being built
            query: Current query being processed
            table_graph: Table dependency graph

        Returns:
            Dict mapping table_name -> list of column names
            Example: {"staging.orders": ["order_id", "user_id", "amount", "status", "order_date"]}
        """
        external_table_columns = {}

        # For each source table this query reads from
        for source_table in query.source_tables:
            # Get the table node
            table_node = table_graph.tables.get(source_table)
            if not table_node:
                continue

            # If this table was created by a previous query, get its output columns
            if table_node.created_by:
                creating_query_id = table_node.created_by

                # Get output columns from the creating query
                output_cols = [
                    col.column_name
                    for col in pipeline.columns.values()
                    if col.query_id == creating_query_id
                    and col.table_name == source_table
                    and col.layer == "output"
                    and not col.is_star  # Don't include * nodes
                ]

                if output_cols:
                    external_table_columns[source_table] = output_cols

        return external_table_columns

    def _get_upstream_table_columns(
        self, pipeline: "Pipeline", table_name: str
    ) -> list[ColumnNode]:
        """
        Get columns from an upstream table that was created in the pipeline.

        Returns the output columns from the query that created this table.
        """
        # Find which query created this table
        table_node = pipeline.table_graph.tables.get(table_name)
        if not table_node or not table_node.created_by:
            return []

        creating_query_id = table_node.created_by

        # Get output columns from the creating query
        upstream_cols = [
            col
            for col in pipeline.columns.values()
            if col.query_id == creating_query_id
            and col.table_name == table_name
            and col.layer == "output"
            and not col.is_star  # Don't use * nodes as source
        ]

        return upstream_cols

    def _add_query_columns(
        self,
        pipeline: "Pipeline",
        query: ParsedQuery,
        query_lineage: ColumnLineageGraph,
    ):
        """
        Add all columns from a query to the pipeline graph.

        Physical table columns (source tables, intermediate tables, output tables) use
        shared naming (table.column) so the same column appears only once in the graph.
        When a column already exists, we skip adding it to avoid duplicates.

        Internal structures (CTEs, subqueries) use query-scoped naming to avoid collisions.

        Special handling for star expansion:
        - If output layer has a * node and we know the upstream columns, expand it
        - This is crucial for cross-query lineage to show exact columns
        """
        # Check if we need to expand any * nodes in the output layer
        nodes_to_add = list(query_lineage.nodes.values())
        expanded_nodes = self._expand_star_nodes_in_pipeline(pipeline, query, nodes_to_add)

        # Add columns with table context
        for node in expanded_nodes:
            # Skip input layer star nodes ONLY when we have explicit columns for that table.
            # This filters out redundant stars (e.g., staging.raw_data.* when Query 1
            # already defined explicit columns), but keeps stars for external tables
            # with unknown schema (e.g., COUNT(*) FROM external.customers).
            if node.is_star and node.layer == "input":
                table_name = self._infer_table_name(node, query) or node.table_name
                has_explicit_cols = any(
                    col.table_name == table_name and not col.is_star
                    for col in pipeline.columns.values()
                )
                if has_explicit_cols:
                    continue

            full_name = self._make_full_name(node, query)

            # Skip if column already exists (shared physical table column)
            if full_name in pipeline.columns:
                continue

            # Extract metadata from SQL comments if available
            description = None
            description_source = None
            pii = False
            owner = None
            tags = set()
            custom_metadata = {}

            if node.sql_metadata is not None:
                metadata = node.sql_metadata
                description = metadata.description
                pii = metadata.pii or False
                owner = metadata.owner
                tags = metadata.tags
                custom_metadata = metadata.custom_metadata

                # Set description source if we have a description from SQL
                if description:
                    description_source = DescriptionSource.SOURCE

            column = ColumnNode(
                column_name=node.column_name,
                table_name=self._infer_table_name(node, query) or node.table_name,
                full_name=full_name,
                query_id=query.query_id,
                unit_id=node.unit_id,
                node_type=node.node_type,
                layer=node.layer,
                expression=node.expression,
                operation=node.node_type,  # Use node_type as operation for now
                description=description,
                description_source=description_source,
                pii=pii,
                owner=owner,
                tags=tags,
                custom_metadata=custom_metadata,
                # Star expansion fields
                is_star=node.is_star,
                except_columns=node.except_columns,
                replace_columns=node.replace_columns,
                # TVF/Synthetic column fields
                is_synthetic=getattr(node, "is_synthetic", False),
                synthetic_source=getattr(node, "synthetic_source", None),
                tvf_parameters=getattr(node, "tvf_parameters", {}),
                # VALUES/Literal column fields
                is_literal=getattr(node, "is_literal", False),
                literal_values=getattr(node, "literal_values", None),
                literal_type=getattr(node, "literal_type", None),
            )
            pipeline.add_column(column)

    def _add_query_edges(
        self,
        pipeline: "Pipeline",
        query: ParsedQuery,
        query_lineage: ColumnLineageGraph,
    ):
        """
        Add all edges from a query to the pipeline graph.

        Handles star expansion: when an edge points to an output * that was expanded,
        create edges to all expanded columns instead.
        """
        for edge in query_lineage.edges:
            from_full = self._make_full_name(edge.from_node, query)
            to_full = self._make_full_name(edge.to_node, query)

            if from_full in pipeline.columns and to_full in pipeline.columns:
                # Normal case: both nodes exist
                pipeline_edge = ColumnEdge(
                    from_node=pipeline.columns[from_full],
                    to_node=pipeline.columns[to_full],
                    edge_type=edge.edge_type if hasattr(edge, "edge_type") else edge.transformation,
                    transformation=edge.transformation,
                    context=edge.context,
                    query_id=query.query_id,
                    # Preserve JSON extraction metadata
                    json_path=getattr(edge, "json_path", None),
                    json_function=getattr(edge, "json_function", None),
                    # Preserve array expansion metadata
                    is_array_expansion=getattr(edge, "is_array_expansion", False),
                    expansion_type=getattr(edge, "expansion_type", None),
                    offset_column=getattr(edge, "offset_column", None),
                    # Preserve nested access metadata
                    nested_path=getattr(edge, "nested_path", None),
                    access_type=getattr(edge, "access_type", None),
                    # Preserve LATERAL correlation metadata
                    is_lateral_correlation=getattr(edge, "is_lateral_correlation", False),
                    lateral_alias=getattr(edge, "lateral_alias", None),
                    # Preserve MERGE operation metadata
                    is_merge_operation=getattr(edge, "is_merge_operation", False),
                    merge_action=getattr(edge, "merge_action", None),
                    merge_condition=getattr(edge, "merge_condition", None),
                    # Preserve QUALIFY clause metadata
                    is_qualify_column=getattr(edge, "is_qualify_column", False),
                    qualify_context=getattr(edge, "qualify_context", None),
                    qualify_function=getattr(edge, "qualify_function", None),
                    # Preserve GROUPING SETS/CUBE/ROLLUP metadata
                    is_grouping_column=getattr(edge, "is_grouping_column", False),
                    grouping_type=getattr(edge, "grouping_type", None),
                    # Preserve window function metadata
                    is_window_function=getattr(edge, "is_window_function", False),
                    window_role=getattr(edge, "window_role", None),
                    window_function=getattr(edge, "window_function", None),
                    window_frame_type=getattr(edge, "window_frame_type", None),
                    window_frame_start=getattr(edge, "window_frame_start", None),
                    window_frame_end=getattr(edge, "window_frame_end", None),
                    window_order_direction=getattr(edge, "window_order_direction", None),
                    window_order_nulls=getattr(edge, "window_order_nulls", None),
                    # Preserve complex aggregate metadata
                    aggregate_spec=getattr(edge, "aggregate_spec", None),
                )
                pipeline.add_edge(pipeline_edge)

            elif (
                edge.to_node.is_star
                and edge.to_node.layer == "output"
                and (from_full in pipeline.columns or edge.from_node.is_star)
            ):
                # Output * was expanded - create edges to all expanded columns
                # Note: from_full might not be in pipeline.columns if it's an input star
                # that we filtered out. The star-to-star logic handles this case.
                dest_table = query.destination_table
                expanded_outputs = [
                    col
                    for col in pipeline.columns.values()
                    if col.table_name == dest_table and col.layer == "output" and not col.is_star
                ]

                # If input is also a *, get EXCEPT columns
                except_columns = edge.to_node.except_columns or set()

                # For input *, connect to all matching output columns
                if edge.from_node.is_star:
                    # Find the source table for this input *
                    source_table = self._infer_table_name(edge.from_node, query)
                    if source_table:
                        # Get all columns from source table
                        source_columns = [
                            col
                            for col in pipeline.columns.values()
                            if col.table_name == source_table
                            and col.layer == "output"
                            and not col.is_star
                        ]

                        # Create edge from each source column to matching output column
                        for source_col in source_columns:
                            if source_col.column_name in except_columns:
                                continue

                            # Find matching output column
                            for output_col in expanded_outputs:
                                if output_col.column_name == source_col.column_name:
                                    pipeline_edge = ColumnEdge(
                                        from_node=source_col,
                                        to_node=output_col,
                                        edge_type="direct_column",
                                        transformation="direct_column",
                                        context=edge.context,
                                        query_id=query.query_id,
                                    )
                                    pipeline.add_edge(pipeline_edge)
                                    break
                else:
                    # Single input column to expanded outputs
                    for output_col in expanded_outputs:
                        pipeline_edge = ColumnEdge(
                            from_node=pipeline.columns[from_full],
                            to_node=output_col,
                            edge_type=edge.edge_type
                            if hasattr(edge, "edge_type")
                            else edge.transformation,
                            transformation=edge.transformation,
                            context=edge.context,
                            query_id=query.query_id,
                        )
                        pipeline.add_edge(pipeline_edge)

    def _add_cross_query_edges(self, pipeline: "Pipeline"):
        """
        Add edges connecting upstream columns to downstream * nodes.

        With the new unified naming for physical tables, most cross-query edges
        flow naturally through shared column nodes. For example:
        - Query 0 creates: staging.orders.customer_id
        - Query 1 reads from staging.orders
        - The single-query lineage for Query 1 has: staging.orders.customer_id -> output
        - This edge is created automatically in _add_query_edges

        However, we still need to handle * nodes (for COUNT(*), etc.):
        - When a query uses COUNT(*), we need edges from all upstream columns to *
        - These edges represent "all columns contribute to this aggregate"
        """
        for table_name, table_node in pipeline.table_graph.tables.items():
            # Find query that creates this table
            if not table_node.created_by:
                continue  # External source table

            # Find output columns from creating query for this table
            # With unified naming, these are just table_name.column_name
            table_columns = [
                col
                for col in pipeline.columns.values()
                if col.table_name == table_name and col.column_name != "*" and col.layer == "output"
            ]

            # Find queries that read this table
            for reading_query_id in table_node.read_by:
                # Check if reading query has a * column for this table
                # This represents COUNT(*) or similar aggregate usage
                star_column = None
                for col in pipeline.columns.values():
                    if (
                        col.query_id == reading_query_id
                        and col.table_name == table_name
                        and col.column_name == "*"
                    ):
                        star_column = col
                        break

                # Connect all table columns to the * node
                if star_column:
                    # Get EXCEPT columns if any
                    except_columns = star_column.except_columns or set()

                    for table_col in table_columns:
                        # Skip columns in EXCEPT clause
                        if table_col.column_name in except_columns:
                            continue

                        edge = ColumnEdge(
                            from_node=table_col,
                            to_node=star_column,
                            edge_type="cross_query",
                            context="cross_query",
                            transformation="all columns -> *",
                            query_id=None,  # Cross-query edge
                        )
                        pipeline.add_edge(edge)

    def _infer_table_name(self, node: ColumnNode, query: ParsedQuery) -> Optional[str]:
        """
        Infer which table this column belongs to.
        Maps table references (aliases) to fully qualified names.

        For queries without a destination table (plain SELECT statements),
        output columns are assigned to a virtual result table named '{query_id}_result'.
        This ensures they appear in simplified lineage views.
        """
        # For output columns, use destination table or virtual result table
        if node.layer == "output":
            if query.destination_table:
                return query.destination_table
            else:
                # Plain SELECT without destination - create virtual result table
                # Use underscore (not colon) so it's treated as physical table in simplified view
                return f"{query.query_id}_result"

        # For input columns, map table_name to fully qualified name
        if node.table_name:
            # Single-query lineage uses short table names like "orders", "users"
            # Pipeline uses fully qualified names like "raw.orders", "staging.users"

            # Try exact match first (already qualified)
            if node.table_name in query.source_tables:
                return node.table_name

            # Try to find matching source table by suffix
            for source_table in query.source_tables:
                # Check if source_table ends with ".{node.table_name}"
                if source_table.endswith(f".{node.table_name}"):
                    return source_table
                # Or if they're the same (no schema prefix)
                if source_table == node.table_name:
                    return source_table

            # If only one source table, assume it's that one
            if len(query.source_tables) == 1:
                return list(query.source_tables)[0]

        # Fallback: if only one source table, use it
        if len(query.source_tables) == 1:
            return list(query.source_tables)[0]

        # Ambiguous - can't determine table
        return None

    def _make_full_name(self, node: ColumnNode, query: ParsedQuery) -> str:
        """
        Create fully qualified column name.

        Naming convention:
        - Physical tables: {table_name}.{column_name}
          Examples: raw.orders.customer_id, staging.orders.amount
          These are shared nodes - same column appears once regardless of which query uses it

        - CTEs: {query_id}:cte:{cte_name}.{column_name}
          Examples: query_0:cte:order_totals.total
          Query-scoped to avoid collisions between CTEs with same name in different queries

        - Subqueries: {query_id}:subq:{subq_id}.{column_name}
          Examples: query_0:subq:derived.count
          Query-scoped internal structures

        - Other internal: {query_id}:{unit_id}.{column_name}
          Fallback for other query-internal structures
        """
        table_name = self._infer_table_name(node, query)
        unit_id = node.unit_id

        # Determine if this is a physical table column or internal structure
        is_physical_table = self._is_physical_table_column(node, query, table_name)

        if is_physical_table and table_name:
            # Physical table: use simple table.column naming (shared across queries)
            return f"{table_name}.{node.column_name}"

        elif unit_id and unit_id.startswith("cte:"):
            # CTE: query-scoped
            return f"{query.query_id}:{unit_id}.{node.column_name}"

        elif unit_id and unit_id.startswith("subq:"):
            # Subquery: query-scoped
            return f"{query.query_id}:{unit_id}.{node.column_name}"

        elif unit_id and unit_id != "main":
            # Other internal structure: query-scoped
            return f"{query.query_id}:{unit_id}.{node.column_name}"

        else:
            # Fallback: use table name if available
            if table_name:
                return f"{table_name}.{node.column_name}"
            else:
                return f"{query.query_id}:unknown.{node.column_name}"

    def _is_physical_table_column(
        self, node: ColumnNode, query: ParsedQuery, table_name: Optional[str]
    ) -> bool:
        """
        Determine if a column belongs to a physical table (vs CTE, subquery, etc).

        Physical table columns get shared naming (table.column) so they appear
        once in the graph regardless of how many queries use them.

        A column is from a physical table if:
        - It's an input from a source table (listed in query.source_tables)
        - It's an output to a destination table (query.destination_table)
        - It has no unit_id or unit_id is 'main' with a real table name
        """
        if not table_name:
            return False

        unit_id = node.unit_id

        # Input layer: check if table is a source table
        if node.layer == "input":
            # Source tables are physical tables
            if table_name in query.source_tables:
                return True
            # Also check if it matches any source table by suffix
            for source in query.source_tables:
                if source.endswith(f".{table_name}") or source == table_name:
                    return True

        # Output layer: check if it's the destination table
        if node.layer == "output":
            if table_name == query.destination_table:
                return True

        # No unit_id or main unit_id typically means physical table
        if unit_id is None or unit_id == "main":
            # But verify it's not from an internal structure
            if node.layer in ("input", "output"):
                return True

        return False

    def _extract_select_from_query(self, query: ParsedQuery) -> Optional[str]:
        """
        Extract SELECT statement from DDL/DML queries.
        Single-query lineage only works on SELECT statements, so we need to extract
        the SELECT from CREATE TABLE AS SELECT, INSERT INTO ... SELECT, etc.
        """
        ast = query.ast

        # CREATE TABLE/VIEW AS SELECT
        if isinstance(ast, exp.Create):
            if ast.expression and isinstance(ast.expression, exp.Select):
                return ast.expression.sql()

        # INSERT INTO ... SELECT
        elif isinstance(ast, exp.Insert):
            if ast.expression and isinstance(ast.expression, exp.Select):
                return ast.expression.sql()

        # MERGE INTO statement - pass full SQL to lineage builder
        elif isinstance(ast, exp.Merge):
            return query.sql

        # Plain SELECT
        elif isinstance(ast, exp.Select):
            return query.sql

        # UPDATE, DELETE, etc. - no SELECT to extract
        return None


class Pipeline:
    """
    Main pipeline class for SQL workflow orchestration with integrated lineage analysis.

    Provides:
    - Table and column lineage tracking
    - Metadata propagation (PII, owner, tags)
    - LLM-powered documentation generation
    - Pipeline execution (sync and async)
    - Airflow DAG generation (TaskFlow API)
    - Loading queries from SQL files or tuples

    Example:
        # Load from SQL files
        pipeline = Pipeline.from_sql_files("queries/", dialect="bigquery")

        # Or define inline
        queries = [
            ("staging", "CREATE TABLE staging AS SELECT * FROM raw"),
            ("final", "CREATE TABLE final AS SELECT * FROM staging"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Execute locally
        result = pipeline.run(executor=execute_sql)

        # Trace lineage
        sources = pipeline.trace_column_backward("final", "metric")

        # Propagate metadata
        pipeline.columns["raw.users.email"].pii = True
        pipeline.propagate_all_metadata()

        # Or generate Airflow DAG
        dag = pipeline.to_airflow_dag(executor=execute_sql, dag_id="my_pipeline")
    """

    def __init__(
        self,
        queries: List[Tuple[str, str]],
        dialect: str = "bigquery",
        template_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Create pipeline from queries.

        Prefer using factory methods for clearer intent:
        - Pipeline.from_tuples() - from [(query_id, sql), ...]
        - Pipeline.from_dict() - from {query_id: sql, ...}
        - Pipeline.from_sql_list() - from [sql, ...] with auto-generated IDs
        - Pipeline.from_sql_string() - from semicolon-separated SQL string
        - Pipeline.from_sql_files() - from directory of .sql files

        Args:
            queries: List of (query_id, sql) tuples
            dialect: SQL dialect (bigquery, snowflake, etc.)
            template_context: Optional dictionary of template variables for Jinja2/variable substitution
                Example: {"env": "prod", "project": "my_project"}

        Example:
            # With template variables
            queries = [
                ("staging", "CREATE TABLE {{env}}_staging.orders AS SELECT * FROM {{env}}_raw.orders")
            ]
            pipeline = Pipeline(queries, dialect="bigquery", template_context={"env": "prod"})
        """
        from .multi_query import MultiQueryParser

        self.dialect = dialect
        self.template_context = template_context
        self.query_mapping: Dict[str, str] = {}  # Maps auto_id -> user_id

        # Column-level lineage graph
        self.column_graph: PipelineLineageGraph = PipelineLineageGraph()
        self.query_graphs: Dict[str, ColumnLineageGraph] = {}
        self.llm: Optional[Any] = None  # LangChain BaseChatModel

        # Convert tuples to plain SQL strings for MultiQueryParser
        sql_list = []
        for user_query_id, sql in queries:
            sql_list.append(sql)
            auto_id = f"query_{len(sql_list) - 1}"
            self.query_mapping[auto_id] = user_query_id

        # Parse queries using current API with template support
        parser = MultiQueryParser(dialect=dialect)
        self.table_graph = parser.parse_queries(sql_list, template_context=template_context)

        # Remap auto-generated query IDs to user-provided IDs
        self._remap_query_ids()

        # Build lineage directly into this Pipeline instance
        builder = PipelineLineageBuilder()
        builder.build(self)

    @property
    def columns(self) -> Dict[str, ColumnNode]:
        """Access columns through column_graph for backward compatibility"""
        return self.column_graph.columns

    @property
    def edges(self) -> List[ColumnEdge]:
        """Access edges through column_graph for backward compatibility"""
        return self.column_graph.edges

    def get_column(
        self, table_name: str, column_name: str, query_id: Optional[str] = None
    ) -> Optional[ColumnNode]:
        """
        Get a column by table and column name.

        Column keys now include query_id prefix (e.g., "query_1:table.column")
        for uniqueness. This method provides convenient lookup by table/column name.

        Args:
            table_name: The table name
            column_name: The column name
            query_id: Optional query_id to filter by

        Returns:
            The ColumnNode if found, None otherwise
        """
        for col in self.columns.values():
            if col.table_name == table_name and col.column_name == column_name:
                if query_id is None or col.query_id == query_id:
                    return col
        return None

    def get_columns_by_table(self, table_name: str) -> List[ColumnNode]:
        """
        Get all columns for a given table.

        Args:
            table_name: The table name to filter by

        Returns:
            List of ColumnNodes for the table
        """
        return [col for col in self.columns.values() if col.table_name == table_name]

    def get_simplified_column_graph(self) -> "PipelineLineageGraph":
        """
        Get a simplified version of the column lineage graph.

        This removes query-internal structures (CTEs, subqueries) and creates
        direct edges between physical table columns.

        - Keeps: All physical table columns (raw.*, staging.*, analytics.*, etc.)
        - Removes: CTE columns, subquery columns
        - Edges: Traces through CTEs/subqueries to create direct table-to-table edges

        Returns:
            A new PipelineLineageGraph with only physical table columns and direct edges.

        Example:
            pipeline = Pipeline(queries, dialect="bigquery")
            simplified = pipeline.get_simplified_column_graph()

            # Full graph has CTEs
            print(f"Full: {len(pipeline.columns)} columns")

            # Simplified has only table columns
            print(f"Simplified: {len(simplified.columns)} columns")
        """
        return self.column_graph.to_simplified()

    @classmethod
    def from_tuples(
        cls,
        queries: List[Tuple[str, str]],
        dialect: str = "bigquery",
        template_context: Optional[Dict[str, Any]] = None,
    ) -> "Pipeline":
        """
        Create pipeline from list of (query_id, sql) tuples.

        Args:
            queries: List of (query_id, sql) tuples
            dialect: SQL dialect (bigquery, snowflake, etc.)
            template_context: Optional dictionary of template variables

        Returns:
            Pipeline instance

        Example:
            pipeline = Pipeline.from_tuples([
                ("staging", "CREATE TABLE staging AS SELECT * FROM raw"),
                ("final", "CREATE TABLE final AS SELECT * FROM staging"),
            ])

            # With templates
            pipeline = Pipeline.from_tuples([
                ("staging", "CREATE TABLE {{env}}_staging.orders AS SELECT * FROM {{env}}_raw.orders"),
            ], template_context={"env": "prod"})
        """
        return cls(queries, dialect=dialect, template_context=template_context)

    @classmethod
    def from_dict(
        cls,
        queries: Dict[str, str],
        dialect: str = "bigquery",
        template_context: Optional[Dict[str, Any]] = None,
    ) -> "Pipeline":
        """
        Create pipeline from dictionary of {query_id: sql}.

        Args:
            queries: Dictionary mapping query_id to SQL string
            dialect: SQL dialect (bigquery, snowflake, etc.)
            template_context: Optional dictionary of template variables

        Returns:
            Pipeline instance

        Example:
            pipeline = Pipeline.from_dict({
                "staging": "CREATE TABLE staging AS SELECT * FROM raw",
                "final": "CREATE TABLE final AS SELECT * FROM staging",
            })

            # With templates
            pipeline = Pipeline.from_dict({
                "staging": "CREATE TABLE {{env}}_staging.orders AS SELECT * FROM raw.orders"
            }, template_context={"env": "prod"})
        """
        query_list = list(queries.items())
        return cls(query_list, dialect=dialect, template_context=template_context)

    @classmethod
    def from_sql_list(
        cls,
        queries: List[str],
        dialect: str = "bigquery",
        template_context: Optional[Dict[str, Any]] = None,
    ) -> "Pipeline":
        """
        Create pipeline from list of SQL strings (auto-generates query IDs).

        Query IDs are generated as: {operation}_{table_name}
        If duplicates exist, a number suffix is added: {operation}_{table_name}_2

        Args:
            queries: List of SQL query strings
            dialect: SQL dialect (bigquery, snowflake, etc.)
            template_context: Optional dictionary of template variables

        Returns:
            Pipeline instance

        Example:
            pipeline = Pipeline.from_sql_list([
                "CREATE TABLE staging AS SELECT * FROM raw",
                "INSERT INTO staging SELECT * FROM raw2",
                "CREATE TABLE final AS SELECT * FROM staging",
            ])
            # Query IDs will be: create_staging, insert_staging, create_final

            # With templates
            pipeline = Pipeline.from_sql_list([
                "CREATE TABLE {{env}}_staging.orders AS SELECT * FROM raw.orders"
            ], template_context={"env": "prod"})
        """
        query_list = []
        id_counts: Dict[str, int] = {}

        for sql in queries:
            query_id = cls._generate_query_id(sql, dialect, id_counts)
            query_list.append((query_id, sql))

        return cls(query_list, dialect=dialect, template_context=template_context)

    @staticmethod
    def _generate_query_id(sql: str, dialect: str, id_counts: Dict[str, int]) -> str:
        """
        Generate a meaningful query ID from SQL statement.

        Format priority:
        1. {operation}_{dest_table}
        2. {operation}_{dest_table}_from_{source_table} (if duplicate)
        3. {operation}_{dest_table}_from_{source_table}_2 (if still duplicate)

        Args:
            sql: SQL query string
            dialect: SQL dialect
            id_counts: Dictionary tracking ID usage counts

        Returns:
            Generated query ID
        """
        import sqlglot
        from sqlglot import exp

        try:
            parsed = sqlglot.parse_one(sql, dialect=dialect)

            # Determine operation
            if isinstance(parsed, exp.Create):
                if parsed.kind == "VIEW":
                    operation = "create_view"
                else:
                    operation = "create"
            elif isinstance(parsed, exp.Insert):
                operation = "insert"
            elif isinstance(parsed, exp.Merge):
                operation = "merge"
            elif isinstance(parsed, exp.Update):
                operation = "update"
            elif isinstance(parsed, exp.Delete):
                operation = "delete"
            elif isinstance(parsed, exp.Select):
                operation = "select"
            else:
                operation = "query"

            # Determine destination table name
            dest_table = None
            if isinstance(parsed, exp.Create):
                table_expr = parsed.this
                if table_expr:
                    dest_table = table_expr.name
            elif isinstance(parsed, (exp.Insert, exp.Merge)):
                table_expr = parsed.this
                if table_expr:
                    dest_table = table_expr.name
            elif isinstance(parsed, (exp.Update, exp.Delete)):
                table_expr = parsed.this
                if table_expr:
                    dest_table = table_expr.name

            # Determine source tables
            source_tables = []
            for table in parsed.find_all(exp.Table):
                # Skip the destination table
                table_name = table.name
                if table_name and table_name != dest_table:
                    source_tables.append(table_name)

            # Build base ID
            if dest_table:
                base_id = f"{operation}_{dest_table}"
            else:
                base_id = operation

            # Try base_id first
            if base_id not in id_counts:
                id_counts[base_id] = 1
                return base_id

            # Try with source table
            if source_tables:
                # Use first source table
                id_with_source = f"{base_id}_from_{source_tables[0]}"
                if id_with_source not in id_counts:
                    id_counts[id_with_source] = 1
                    return id_with_source
                else:
                    # Still duplicate, use number
                    id_counts[id_with_source] += 1
                    return f"{id_with_source}_{id_counts[id_with_source]}"
            else:
                # No source table, use number
                id_counts[base_id] += 1
                return f"{base_id}_{id_counts[base_id]}"

        except Exception:
            # Fallback if parsing fails
            base_id = "query"
            if base_id not in id_counts:
                id_counts[base_id] = 1
                return base_id
            else:
                id_counts[base_id] += 1
                return f"{base_id}_{id_counts[base_id]}"

    @classmethod
    def from_sql_string(
        cls,
        sql: str,
        dialect: str = "bigquery",
        template_context: Optional[Dict[str, Any]] = None,
    ) -> "Pipeline":
        """
        Create pipeline from single SQL string with semicolon-separated queries.

        Query IDs are generated as: {operation}_{table_name}
        If duplicates exist, a number suffix is added: {operation}_{table_name}_2

        Args:
            sql: SQL string with multiple queries separated by semicolons
            dialect: SQL dialect (bigquery, snowflake, etc.)
            template_context: Optional dictionary of template variables

        Returns:
            Pipeline instance

        Example:
            pipeline = Pipeline.from_sql_string('''
                CREATE TABLE staging AS SELECT * FROM raw;
                CREATE TABLE final AS SELECT * FROM staging
            ''')
            # Query IDs will be: create_staging, create_final

            # With templates
            pipeline = Pipeline.from_sql_string(
                "CREATE TABLE {{env}}_staging.orders AS SELECT * FROM raw.orders",
                template_context={"env": "prod"}
            )
        """
        # Split by semicolon and filter empty strings
        queries = [q.strip() for q in sql.split(";") if q.strip()]
        return cls.from_sql_list(queries, dialect=dialect, template_context=template_context)

    @classmethod
    def from_json(
        cls,
        data: Dict[str, Any],
        apply_metadata: bool = True,
    ) -> "Pipeline":
        """
        Create pipeline from JSON data exported by JSONExporter.

        This enables round-trip serialization: export a pipeline to JSON,
        store/transfer it, and recreate the same pipeline later.

        Args:
            data: JSON dictionary from JSONExporter.export() or Pipeline.to_json()
            apply_metadata: Whether to apply metadata (descriptions, PII, etc.)
                from the JSON to the reconstructed pipeline

        Returns:
            Pipeline instance

        Example:
            # Save pipeline to file
            data = pipeline.to_json()
            with open("pipeline.json", "w") as f:
                json.dump(data, f)

            # Later, reload the pipeline
            with open("pipeline.json") as f:
                data = json.load(f)
            pipeline = Pipeline.from_json(data)

            # Verify round-trip
            assert len(pipeline.columns) > 0
            assert len(pipeline.edges) > 0
        """
        # Validate required fields for round-trip
        if "queries" not in data:
            raise ValueError(
                "JSON data missing 'queries' field. "
                "Ensure JSONExporter.export() was called with include_queries=True"
            )

        if "dialect" not in data:
            raise ValueError("JSON data missing 'dialect' field")

        # Extract pipeline construction data
        dialect = data["dialect"]
        template_context = data.get("template_context")

        # Reconstruct queries list
        queries = [(q["query_id"], q["sql"]) for q in data["queries"]]

        # Create pipeline from queries
        pipeline = cls.from_tuples(queries, dialect=dialect, template_context=template_context)

        # Apply metadata if requested
        if apply_metadata and "columns" in data:
            for col_data in data["columns"]:
                full_name = col_data.get("full_name")
                if full_name and full_name in pipeline.columns:
                    col = pipeline.columns[full_name]

                    # Apply metadata fields
                    if col_data.get("description"):
                        col.description = col_data["description"]
                    if col_data.get("description_source"):
                        col.description_source = DescriptionSource(col_data["description_source"])
                    if col_data.get("owner"):
                        col.owner = col_data["owner"]
                    if col_data.get("pii"):
                        col.pii = col_data["pii"]
                    if col_data.get("tags"):
                        col.tags = set(col_data["tags"])
                    if col_data.get("custom_metadata"):
                        col.custom_metadata = col_data["custom_metadata"]

        return pipeline

    @classmethod
    def from_json_file(cls, file_path: str, apply_metadata: bool = True) -> "Pipeline":
        """
        Create pipeline from JSON file exported by JSONExporter.

        Args:
            file_path: Path to JSON file
            apply_metadata: Whether to apply metadata from the JSON

        Returns:
            Pipeline instance

        Example:
            # Export pipeline
            JSONExporter.export_to_file(pipeline, "pipeline.json")

            # Later, reload it
            pipeline = Pipeline.from_json_file("pipeline.json")
        """
        import json
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        with open(path) as f:
            data = json.load(f)

        return cls.from_json(data, apply_metadata=apply_metadata)

    @classmethod
    def _create_empty(cls, table_graph: "TableDependencyGraph") -> "Pipeline":
        """
        Create an empty Pipeline with just a table_graph (for testing).

        This bypasses SQL parsing and creates a minimal Pipeline that can be
        populated manually with columns and edges.

        Args:
            table_graph: Pre-built table dependency graph

        Returns:
            Empty Pipeline instance
        """
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.dialect = "bigquery"
        instance.query_mapping = {}
        instance.column_graph = PipelineLineageGraph()
        instance.query_graphs = {}
        instance.llm = None
        instance.table_graph = table_graph
        return instance

    # === Lineage methods (from PipelineLineageGraph) ===

    def add_column(self, column: ColumnNode) -> ColumnNode:
        """Add a column node to the graph"""
        self.column_graph.add_column(column)

        # Also add to table's columns set if table exists
        if column.table_name and column.table_name in self.table_graph.tables:
            self.table_graph.tables[column.table_name].columns.add(column.column_name)

        return column

    def add_edge(self, edge: ColumnEdge):
        """Add a lineage edge"""
        self.column_graph.add_edge(edge)

    def trace_column_backward(self, table_name: str, column_name: str) -> List[ColumnNode]:
        """
        Trace a column backward to its ultimate sources.
        Returns list of source columns across all queries.

        For full lineage path with all intermediate nodes, use trace_column_backward_full().
        """
        # Find the target column(s) - there may be multiple with same table.column
        # from different queries. For output columns, we want the one with layer="output"
        target_columns = [
            col
            for col in self.columns.values()
            if col.table_name == table_name and col.column_name == column_name
        ]

        if not target_columns:
            return []

        # Prefer output layer columns as starting point for backward tracing
        output_cols = [c for c in target_columns if c.layer == "output"]
        start_columns = output_cols if output_cols else target_columns

        # BFS backward through edges
        visited = set()
        queue = list(start_columns)
        sources = []

        while queue:
            current = queue.pop(0)
            if current.full_name in visited:
                continue
            visited.add(current.full_name)

            # Find incoming edges
            incoming = [e for e in self.edges if e.to_node.full_name == current.full_name]

            if not incoming:
                # No incoming edges = source column
                sources.append(current)
            else:
                for edge in incoming:
                    queue.append(edge.from_node)

        return sources

    def trace_column_backward_full(
        self, table_name: str, column_name: str, include_ctes: bool = True
    ) -> Tuple[List[ColumnNode], List[ColumnEdge]]:
        """
        Trace a column backward with full transparency.

        Returns complete lineage path including all intermediate tables and CTEs.

        Args:
            table_name: The table containing the column to trace
            column_name: The column name to trace
            include_ctes: If True, include CTE columns; if False, only real tables

        Returns:
            Tuple of (nodes, edges) representing the complete lineage path.
            - nodes: All columns in the lineage, in BFS order from target to sources
            - edges: All edges connecting the columns

        Example:
            nodes, edges = pipeline.trace_column_backward_full("mart_customer_ltv", "lifetime_revenue")

            # Print the lineage path:
            for node in nodes:
                print(f"{node.table_name}.{node.column_name} (query={node.query_id})")

            # Print the edges:
            for edge in edges:
                print(f"{edge.from_node.table_name}.{edge.from_node.column_name} -> "
                      f"{edge.to_node.table_name}.{edge.to_node.column_name}")
        """
        # Find the target column(s)
        target_columns = [
            col
            for col in self.columns.values()
            if col.table_name == table_name and col.column_name == column_name
        ]

        if not target_columns:
            return [], []

        # Prefer output layer columns as starting point
        output_cols = [c for c in target_columns if c.layer == "output"]
        start_columns = output_cols if output_cols else target_columns

        # BFS backward through edges, collecting all nodes and edges
        visited = set()
        queue = list(start_columns)
        all_nodes = []
        all_edges = []

        while queue:
            current = queue.pop(0)
            if current.full_name in visited:
                continue
            visited.add(current.full_name)

            # Optionally skip CTE columns
            if not include_ctes and current.layer == "cte":
                # Still need to traverse through CTEs to find real tables
                incoming = [e for e in self.edges if e.to_node.full_name == current.full_name]
                for edge in incoming:
                    queue.append(edge.from_node)
                continue

            all_nodes.append(current)

            # Find incoming edges
            incoming = [e for e in self.edges if e.to_node.full_name == current.full_name]

            for edge in incoming:
                all_edges.append(edge)
                queue.append(edge.from_node)

        return all_nodes, all_edges

    def get_table_lineage_path(
        self, table_name: str, column_name: str
    ) -> List[Tuple[str, str, str]]:
        """
        Get simplified table-level lineage path for a column.

        Returns list of (table_name, column_name, query_id) tuples representing
        the lineage through real tables only (skipping CTEs).

        This provides a clear view of how data flows between tables in your pipeline.

        Example:
            path = pipeline.get_table_lineage_path("mart_customer_ltv", "lifetime_revenue")
            # Returns:
            # [
            #   ("mart_customer_ltv", "lifetime_revenue", "07_mart_customer_ltv"),
            #   ("stg_orders_enriched", "total_amount", "05_stg_orders_enriched"),
            #   ("raw_orders", "total_amount", "01_raw_orders"),
            #   ("source_orders", "total_amount", "01_raw_orders"),
            # ]
        """
        nodes, _ = self.trace_column_backward_full(table_name, column_name, include_ctes=False)

        # Deduplicate by table.column (keep first occurrence which is closest to target)
        seen = set()
        result = []
        for node in nodes:
            key = (node.table_name, node.column_name)
            if key not in seen:
                seen.add(key)
                result.append((node.table_name, node.column_name, node.query_id))

        return result

    def trace_column_forward(self, table_name: str, column_name: str) -> List[ColumnNode]:
        """
        Trace a column forward to see what depends on it.
        Returns list of final downstream columns across all queries.

        For full impact path with all intermediate nodes, use trace_column_forward_full().
        """
        # Find the source column(s) - there may be multiple with same table.column
        # from different queries. For input columns, we want the one with layer="input"
        source_columns = [
            col
            for col in self.columns.values()
            if col.table_name == table_name and col.column_name == column_name
        ]

        if not source_columns:
            return []

        # Prefer input layer columns as starting point for forward tracing
        input_cols = [c for c in source_columns if c.layer == "input"]
        start_columns = input_cols if input_cols else source_columns

        # BFS forward through edges
        visited = set()
        queue = list(start_columns)
        descendants = []

        while queue:
            current = queue.pop(0)
            if current.full_name in visited:
                continue
            visited.add(current.full_name)

            # Find outgoing edges
            outgoing = [e for e in self.edges if e.from_node.full_name == current.full_name]

            if not outgoing:
                # No outgoing edges = final column
                descendants.append(current)
            else:
                for edge in outgoing:
                    queue.append(edge.to_node)

        return descendants

    def trace_column_forward_full(
        self, table_name: str, column_name: str, include_ctes: bool = True
    ) -> Tuple[List[ColumnNode], List[ColumnEdge]]:
        """
        Trace a column forward with full transparency.

        Returns complete impact path including all intermediate tables and CTEs.

        Args:
            table_name: The table containing the column to trace
            column_name: The column name to trace
            include_ctes: If True, include CTE columns; if False, only real tables

        Returns:
            Tuple of (nodes, edges) representing the complete impact path.
            - nodes: All columns impacted, in BFS order from source to finals
            - edges: All edges connecting the columns

        Example:
            nodes, edges = pipeline.trace_column_forward_full("raw_orders", "total_amount")

            # Print the impact path:
            for node in nodes:
                print(f"{node.table_name}.{node.column_name} (query={node.query_id})")
        """
        # Find the source column(s)
        source_columns = [
            col
            for col in self.columns.values()
            if col.table_name == table_name and col.column_name == column_name
        ]

        if not source_columns:
            return [], []

        # Prefer input/output layer columns as starting point
        input_cols = [c for c in source_columns if c.layer in ("input", "output")]
        start_columns = input_cols if input_cols else source_columns

        # BFS forward through edges, collecting all nodes and edges
        visited = set()
        queue = list(start_columns)
        all_nodes = []
        all_edges = []

        while queue:
            current = queue.pop(0)
            if current.full_name in visited:
                continue
            visited.add(current.full_name)

            # Optionally skip CTE columns
            if not include_ctes and current.layer == "cte":
                # Still need to traverse through CTEs to find real tables
                outgoing = [e for e in self.edges if e.from_node.full_name == current.full_name]
                for edge in outgoing:
                    queue.append(edge.to_node)
                continue

            all_nodes.append(current)

            # Find outgoing edges
            outgoing = [e for e in self.edges if e.from_node.full_name == current.full_name]

            for edge in outgoing:
                all_edges.append(edge)
                queue.append(edge.to_node)

        return all_nodes, all_edges

    def get_table_impact_path(
        self, table_name: str, column_name: str
    ) -> List[Tuple[str, str, str]]:
        """
        Get simplified table-level impact path for a column.

        Returns list of (table_name, column_name, query_id) tuples representing
        the downstream impact through real tables only (skipping CTEs).

        This provides a clear view of how a source column impacts downstream tables.

        Example:
            path = pipeline.get_table_impact_path("raw_orders", "total_amount")
            # Returns:
            # [
            #   ("raw_orders", "total_amount", "01_raw_orders"),
            #   ("stg_orders_enriched", "total_amount", "05_stg_orders_enriched"),
            #   ("mart_customer_ltv", "lifetime_revenue", "07_mart_customer_ltv"),
            #   ...
            # ]
        """
        nodes, _ = self.trace_column_forward_full(table_name, column_name, include_ctes=False)

        # Deduplicate by table.column (keep first occurrence which is closest to source)
        seen = set()
        result = []
        for node in nodes:
            key = (node.table_name, node.column_name)
            if key not in seen:
                seen.add(key)
                result.append((node.table_name, node.column_name, node.query_id))

        return result

    def get_lineage_path(
        self, from_table: str, from_column: str, to_table: str, to_column: str
    ) -> List[ColumnEdge]:
        """
        Find the lineage path between two columns.
        Returns list of edges connecting them (if path exists).
        """
        # Find source columns by table and column name
        from_columns = [
            col
            for col in self.columns.values()
            if col.table_name == from_table and col.column_name == from_column
        ]

        to_columns = [
            col
            for col in self.columns.values()
            if col.table_name == to_table and col.column_name == to_column
        ]

        if not from_columns or not to_columns:
            return []

        # Get target full_names for matching
        to_full_names = {col.full_name for col in to_columns}

        # BFS with path tracking, starting from all matching source columns
        queue = [(col, []) for col in from_columns]
        visited = set()

        while queue:
            current, path = queue.pop(0)
            if current.full_name in visited:
                continue
            visited.add(current.full_name)

            if current.full_name in to_full_names:
                return path

            # Find outgoing edges
            for edge in self.edges:
                if edge.from_node.full_name == current.full_name:
                    queue.append((edge.to_node, path + [edge]))

        return []  # No path found

    def generate_all_descriptions(self, batch_size: int = 10, verbose: bool = True):
        """
        Generate descriptions for all columns using LLM.

        Processes columns in topological order (sources first).

        Args:
            batch_size: Number of columns per batch (currently processes sequentially)
            verbose: If True, print progress messages
        """
        if not self.llm:
            raise ValueError("LLM not configured. Set pipeline.llm before calling.")

        # Get columns in topological order
        sorted_query_ids = self.table_graph.topological_sort()

        columns_to_process = []
        for query_id in sorted_query_ids:
            query = self.table_graph.queries[query_id]
            if query.destination_table:
                for col in self.columns.values():
                    if (
                        col.table_name == query.destination_table
                        and not col.description
                        and col.is_computed()
                    ):
                        columns_to_process.append(col)

        if verbose:
            print(f"📊 Generating descriptions for {len(columns_to_process)} columns...")

        # Process columns
        for i, col in enumerate(columns_to_process):
            if verbose and (i + 1) % batch_size == 0:
                print(f"   Processed {i + 1}/{len(columns_to_process)} columns...")

            generate_description(col, self.llm, self)

        if verbose:
            print(f"✅ Done! Generated {len(columns_to_process)} descriptions")

    def propagate_all_metadata(self, verbose: bool = True):
        """
        Propagate metadata (owner, PII, tags) through lineage.

        Uses a two-pass approach:
        1. Backward pass: Propagate metadata from output columns (with SQL comment
           metadata) to their input layer sources. This ensures that if an output
           column has PII from a comment, the source column also gets PII.
        2. Forward pass: Propagate metadata from source columns to downstream
           columns in topological order.

        Args:
            verbose: If True, print progress messages
        """
        # Get columns in topological order
        sorted_query_ids = self.table_graph.topological_sort()

        # Pass 1: Backward propagation from output columns to input columns
        # This handles metadata set via SQL comments on output columns
        output_columns = [col for col in self.columns.values() if col.layer == "output"]

        if verbose:
            print(
                f"📊 Pass 1: Propagating metadata backward from "
                f"{len(output_columns)} output columns..."
            )

        for col in output_columns:
            propagate_metadata_backward(col, self)

        # Pass 2: Forward propagation through lineage
        # Process all computed columns (output columns from each query)
        columns_to_process = []
        for query_id in sorted_query_ids:
            query = self.table_graph.queries[query_id]
            # Get the table name for this query's output
            # For CREATE TABLE queries, use destination_table
            # For plain SELECTs, use query_id_result pattern
            target_table = query.destination_table or f"{query_id}_result"
            for col in self.columns.values():
                if col.table_name == target_table and col.is_computed():
                    columns_to_process.append(col)

        if verbose:
            print(
                f"📊 Pass 2: Propagating metadata forward for {len(columns_to_process)} columns..."
            )

        # Process columns
        for col in columns_to_process:
            propagate_metadata(col, self)

        if verbose:
            print(f"✅ Done! Propagated metadata for {len(columns_to_process)} columns")

    def get_pii_columns(self) -> List[ColumnNode]:
        """
        Get all columns marked as PII.

        Returns:
            List of columns where pii == True
        """
        return [col for col in self.columns.values() if col.pii]

    def get_columns_by_owner(self, owner: str) -> List[ColumnNode]:
        """
        Get all columns with a specific owner.

        Args:
            owner: Owner name to filter by

        Returns:
            List of columns with matching owner
        """
        return [col for col in self.columns.values() if col.owner == owner]

    def get_columns_by_tag(self, tag: str) -> List[ColumnNode]:
        """
        Get all columns containing a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of columns containing the tag
        """
        return [col for col in self.columns.values() if tag in col.tags]

    def diff(self, other: "Pipeline"):
        """
        Compare this pipeline with another and return differences.

        Args:
            other: The other pipeline to compare with (typically older version)

        Returns:
            PipelineDiff object containing the differences
        """
        from .diff import PipelineDiff

        return PipelineDiff(new_graph=self, old_graph=other)

    def to_json(self, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Export pipeline to JSON-serializable dictionary.

        Convenience wrapper for JSONExporter.export().

        Args:
            include_metadata: Whether to include metadata (descriptions, PII, etc.)

        Returns:
            Dictionary with columns, edges, and tables

        Example:
            data = pipeline.to_json()
            with open("lineage.json", "w") as f:
                json.dump(data, f, indent=2)
        """
        from .export import JSONExporter

        return JSONExporter.export(self, include_metadata=include_metadata)

    @classmethod
    def from_sql_files(
        cls,
        sql_dir: str,
        dialect: str = "bigquery",
        pattern: str = "*.sql",
        query_id_from: str = "filename",
        template_context: Optional[Dict[str, Any]] = None,
    ) -> "Pipeline":
        """
        Create pipeline from SQL files in a directory.

        Args:
            sql_dir: Directory containing SQL files
            dialect: SQL dialect (bigquery, snowflake, etc.)
            pattern: Glob pattern for SQL files (default: "*.sql")
            query_id_from: How to determine query ID:
                - "filename": Use filename without extension (default)
                - "comment": Extract from first line comment (-- query_id: name)
            template_context: Optional dictionary of template variables

        Returns:
            Pipeline instance

        Example:
            # Query IDs from filenames
            pipeline = Pipeline.from_sql_files("queries/", dialect="bigquery")

            # Query IDs from comments
            pipeline = Pipeline.from_sql_files(
                "queries/",
                query_id_from="comment"
            )

            # With templates
            pipeline = Pipeline.from_sql_files(
                "queries/",
                template_context={"env": "prod", "project": "my_project"}
            )
        """
        import re
        from pathlib import Path

        sql_path = Path(sql_dir)
        sql_files = sorted(sql_path.glob(pattern))

        if not sql_files:
            raise ValueError(f"No SQL files found in {sql_dir} matching {pattern}")

        queries = []
        for sql_file in sql_files:
            sql_content = sql_file.read_text()

            if query_id_from == "filename":
                query_id = sql_file.stem  # Filename without extension
            elif query_id_from == "comment":
                # Extract from first line comment: -- query_id: name
                match = re.match(r"--\s*query_id:\s*(\w+)", sql_content)
                if match:
                    query_id = match.group(1)
                else:
                    # Fallback to filename if no comment found
                    query_id = sql_file.stem
            else:
                raise ValueError(f"Invalid query_id_from: {query_id_from}")

            queries.append((query_id, sql_content))

        return cls(queries, dialect=dialect, template_context=template_context)

    def _remap_query_ids(self):
        """Remap auto-generated query IDs to user-provided IDs"""
        # Remap in queries dict
        new_queries = {}
        for auto_id, query in self.table_graph.queries.items():
            user_id = self.query_mapping.get(auto_id, auto_id)
            query.query_id = user_id
            new_queries[user_id] = query
        self.table_graph.queries = new_queries

        # Remap in table references
        for table in self.table_graph.tables.values():
            if table.created_by and table.created_by in self.query_mapping:
                table.created_by = self.query_mapping[table.created_by]
            table.read_by = [self.query_mapping.get(qid, qid) for qid in table.read_by]
            table.modified_by = [self.query_mapping.get(qid, qid) for qid in table.modified_by]

    def __repr__(self):
        """Show topologically sorted SQL statements with query units"""
        sorted_query_ids = self.table_graph.topological_sort()
        query_strs = []

        for query_id in sorted_query_ids:
            query = self.table_graph.queries[query_id]
            # Truncate SQL to first 60 chars for readability
            sql_preview = query.sql.strip().replace("\n", " ")
            if len(sql_preview) > 60:
                sql_preview = sql_preview[:57] + "..."

            query_str = f"{query_id}: {sql_preview}"

            # Add query units if lineage exists
            if query_id in self.query_graphs:
                query_lineage = self.query_graphs[query_id]
                # Extract unique unit_ids from nodes
                unit_ids = sorted({n.unit_id for n in query_lineage.nodes.values() if n.unit_id})
                if unit_ids:
                    # Format each unit on its own line with indentation
                    for unit_id in unit_ids:
                        query_str += f"\n    {unit_id}"

            query_strs.append(query_str)

        queries_display = "\n  ".join(query_strs)
        return f"Pipeline(\n  {queries_display}\n)"

    def build_subpipeline(self, target_table: str) -> "Pipeline":
        """
        Build a subpipeline containing only queries needed to build a specific table.

        This is a convenience wrapper around split() for building a single target.

        Args:
            target_table: The table to build (e.g., "analytics.revenue")

        Returns:
            A new Pipeline containing only the queries needed to build target_table

        Example:
            # Build only what's needed for analytics.revenue
            subpipeline = pipeline.build_subpipeline("analytics.revenue")

            print(f"Full pipeline: {len(pipeline.table_graph.queries)} queries")
            print(f"Subpipeline: {len(subpipeline.table_graph.queries)} queries")

            # Run just the subpipeline
            result = subpipeline.run(executor=execute_sql)
        """
        subpipelines = self.split([target_table])
        return subpipelines[0]

    def split(self, sinks: List) -> List["Pipeline"]:
        """
        Split pipeline into non-overlapping subpipelines based on target tables.

        Each subpipeline contains all queries needed to build its sink tables,
        ensuring no query appears in multiple subpipelines.

        Args:
            sinks: List of sink specifications. Each element can be:
                   - A single table name (str)
                   - A list of table names (List[str])

        Returns:
            List of Pipeline instances, one per sink group

        Examples:
            # Split into 3 subpipelines
            subpipelines = pipeline.split(
                sinks=[
                    "final_table",           # Single table
                    ["metrics", "summary"],  # Multiple tables in one subpipeline
                    "aggregated_data"        # Another single table
                ]
            )

            # Each subpipeline can be run independently
            subpipelines[0].run(executor=execute_sql)  # Builds final_table
            subpipelines[1].run(executor=execute_sql)  # Builds metrics + summary
            subpipelines[2].run(executor=execute_sql)  # Builds aggregated_data
        """
        # Normalize sinks to list of lists
        normalized_sinks: List[List[str]] = []
        for sink in sinks:
            if isinstance(sink, str):
                normalized_sinks.append([sink])
            elif isinstance(sink, list):
                normalized_sinks.append(sink)
            else:
                raise ValueError(f"Invalid sink type: {type(sink)}. Expected str or List[str]")

        # For each sink group, find all required queries
        subpipeline_queries: List[set] = []

        for sink_group in normalized_sinks:
            required_queries = set()

            # BFS backward from each sink to find all dependencies
            for sink_table in sink_group:
                if sink_table not in self.table_graph.tables:
                    raise ValueError(
                        f"Sink table '{sink_table}' not found in pipeline. "
                        f"Available tables: {list(self.table_graph.tables.keys())}"
                    )

                # Find all queries needed for this sink
                visited = set()
                queue = [sink_table]

                while queue:
                    current_table = queue.pop(0)
                    if current_table in visited:
                        continue
                    visited.add(current_table)

                    table_node = self.table_graph.tables.get(current_table)
                    if not table_node:
                        continue

                    # Add the query that creates this table
                    if table_node.created_by:
                        query_id = table_node.created_by
                        required_queries.add(query_id)

                        # Add source tables to queue
                        query = self.table_graph.queries[query_id]
                        for source_table in query.source_tables:
                            if source_table not in visited:
                                queue.append(source_table)

            subpipeline_queries.append(required_queries)

        # Ensure non-overlapping: assign each query to only one subpipeline
        # Strategy: Assign to the first subpipeline that needs it
        assigned_queries: dict = {}  # query_id -> subpipeline_index

        for idx, query_set in enumerate(subpipeline_queries):
            for query_id in query_set:
                if query_id not in assigned_queries:
                    assigned_queries[query_id] = idx

        # Build final non-overlapping query sets
        final_query_sets: List[set] = [set() for _ in normalized_sinks]
        for query_id, subpipeline_idx in assigned_queries.items():
            final_query_sets[subpipeline_idx].add(query_id)

        # Create Pipeline instances for each subpipeline
        subpipelines = []

        for query_ids in final_query_sets:
            if not query_ids:
                # Empty subpipeline - skip
                continue

            # Extract queries in order
            subpipeline_query_list = []
            for query_id in self.table_graph.topological_sort():
                if query_id in query_ids:
                    query = self.table_graph.queries[query_id]
                    subpipeline_query_list.append((query_id, query.sql))

            # Create new Pipeline instance
            subpipeline = Pipeline(subpipeline_query_list, dialect=self.dialect)
            subpipelines.append(subpipeline)

        return subpipelines

    def _get_execution_levels(self) -> List[List[str]]:
        """
        Group queries into levels for concurrent execution.

        Level 0: Queries with no dependencies
        Level 1: Queries that depend only on Level 0
        Level 2: Queries that depend on Level 0 or 1
        etc.

        Queries in the same level can run concurrently.

        Returns:
            List of levels, where each level is a list of query IDs
        """
        levels = []
        completed = set()

        while len(completed) < len(self.table_graph.queries):
            current_level = []

            for query_id, query in self.table_graph.queries.items():
                if query_id in completed:
                    continue

                # Check if all dependencies are completed
                dependencies_met = True
                for source_table in query.source_tables:
                    # Find query that creates this table
                    table_node = self.table_graph.tables.get(source_table)
                    if table_node and table_node.created_by:
                        if table_node.created_by not in completed:
                            dependencies_met = False
                            break

                if dependencies_met:
                    current_level.append(query_id)

            if not current_level:
                # No progress - circular dependency
                raise RuntimeError("Circular dependency detected in pipeline")

            levels.append(current_level)
            completed.update(current_level)

        return levels

    def to_airflow_dag(
        self,
        executor: Callable[[str], None],
        dag_id: str,
        schedule: str = "@daily",
        start_date: Optional[datetime] = None,
        default_args: Optional[dict] = None,
        **dag_kwargs,
    ):
        """
        Create Airflow DAG from this pipeline using TaskFlow API.

        Supports all Airflow DAG parameters via **dag_kwargs for complete flexibility.

        Args:
            executor: Function that executes SQL (takes sql string)
            dag_id: Airflow DAG ID
            schedule: Schedule interval (default: "@daily")
            start_date: DAG start date (default: datetime(2024, 1, 1))
            default_args: Airflow default_args (default: owner='data_team', retries=2)
            **dag_kwargs: Additional DAG parameters (catchup, tags, max_active_runs,
                         description, max_active_tasks, dagrun_timeout, etc.)
                         See Airflow DAG documentation for all available parameters.

        Returns:
            Airflow DAG instance

        Examples:
            # Basic usage
            def execute_sql(sql: str):
                from google.cloud import bigquery
                client = bigquery.Client()
                client.query(sql).result()

            dag = pipeline.to_airflow_dag(
                executor=execute_sql,
                dag_id="my_pipeline"
            )

            # Advanced usage with all DAG parameters
            dag = pipeline.to_airflow_dag(
                executor=execute_sql,
                dag_id="my_pipeline",
                schedule="0 0 * * *",  # Daily at midnight
                description="Customer analytics pipeline",
                catchup=False,
                max_active_runs=3,
                max_active_tasks=10,
                tags=["analytics", "daily"],
                default_view="graph",  # Airflow 2.x only
                orientation="LR",  # Airflow 2.x only
            )

        Note:
            Currently supports Airflow 2.x only. Airflow 3.x support is planned.
        """
        try:
            from airflow.decorators import dag, task  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "Airflow is required for DAG generation. "
                "Install it with: pip install 'apache-airflow>=2.7.0,<3.0.0'"
            ) from e

        if start_date is None:
            start_date = datetime(2024, 1, 1)

        if default_args is None:
            default_args = {
                "owner": "data_team",
                "retries": 2,
                "retry_delay": timedelta(minutes=5),
            }

        # Build DAG parameters
        dag_params = {
            "dag_id": dag_id,
            "schedule": schedule,
            "start_date": start_date,
            "default_args": default_args,
            **dag_kwargs,  # Allow user to override any parameter
        }

        # Set default values only if not provided by user
        dag_params.setdefault("catchup", False)
        dag_params.setdefault("tags", ["clgraph"])

        table_graph = self.table_graph

        @dag(**dag_params)
        def pipeline_dag():
            """Generated pipeline DAG"""

            # Create task callables for each query
            task_callables = {}

            for query_id in table_graph.topological_sort():
                query = table_graph.queries[query_id]
                sql_to_execute = query.sql

                # Create task with unique function name using closure
                def make_task(qid, sql):
                    @task(task_id=qid.replace("-", "_"))
                    def execute_query():
                        """Execute SQL query"""
                        executor(sql)
                        return f"Completed: {qid}"

                    return execute_query

                task_callables[query_id] = make_task(query_id, sql_to_execute)

            # Instantiate all tasks once before wiring dependencies
            task_instances = {qid: callable() for qid, callable in task_callables.items()}

            # Set up dependencies based on table lineage
            for _table_name, table_node in table_graph.tables.items():
                if table_node.created_by:
                    upstream_id = table_node.created_by
                    for downstream_id in table_node.read_by:
                        if upstream_id in task_instances and downstream_id in task_instances:
                            # Airflow: downstream >> upstream means upstream runs first
                            task_instances[upstream_id] >> task_instances[downstream_id]

        return pipeline_dag()

    def run(
        self,
        executor: Callable[[str], None],
        max_workers: int = 4,
        verbose: bool = True,
    ) -> dict:
        """
        Execute pipeline synchronously with concurrent execution.

        Args:
            executor: Function that executes SQL (takes sql string)
            max_workers: Max concurrent workers (default: 4)
            verbose: Print progress (default: True)

        Returns:
            dict with execution results: {
                "completed": list of completed query IDs,
                "failed": list of (query_id, error) tuples,
                "elapsed_seconds": total execution time,
                "total_queries": total number of queries
            }

        Example:
            def execute_sql(sql: str):
                import duckdb
                conn = duckdb.connect()
                conn.execute(sql)

            result = pipeline.run(executor=execute_sql, max_workers=4)
            print(f"Completed {len(result['completed'])} queries")
        """
        if verbose:
            print(f"🚀 Starting pipeline execution ({len(self.table_graph.queries)} queries)")
            print()

        # Track completed queries
        completed = set()
        failed = []
        start_time = time.time()

        # Group queries by level for concurrent execution
        levels = self._get_execution_levels()

        # Execute level by level
        for level_num, level_queries in enumerate(levels, 1):
            if verbose:
                print(f"📊 Level {level_num}: {len(level_queries)} queries")

            # Execute queries in this level concurrently
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {}

                for query_id in level_queries:
                    query = self.table_graph.queries[query_id]
                    future = pool.submit(executor, query.sql)
                    futures[future] = query_id

                # Wait for completion
                for future in as_completed(futures):
                    query_id = futures[future]

                    try:
                        future.result()
                        completed.add(query_id)

                        if verbose:
                            print(f"  ✅ {query_id}")
                    except Exception as e:
                        failed.append((query_id, str(e)))

                        if verbose:
                            print(f"  ❌ {query_id}: {e}")

            if verbose:
                print()

        elapsed = time.time() - start_time

        # Summary
        if verbose:
            print("=" * 60)
            print(f"✅ Pipeline completed in {elapsed:.2f}s")
            print(f"   Successful: {len(completed)}")
            print(f"   Failed: {len(failed)}")
            if failed:
                print("\n⚠️  Failed queries:")
                for query_id, error in failed:
                    print(f"   - {query_id}: {error}")
            print("=" * 60)

        return {
            "completed": list(completed),
            "failed": failed,
            "elapsed_seconds": elapsed,
            "total_queries": len(self.table_graph.queries),
        }

    async def async_run(
        self,
        executor: Callable[[str], Awaitable[None]],
        max_workers: int = 4,
        verbose: bool = True,
    ) -> dict:
        """
        Execute pipeline asynchronously with concurrent execution.

        Args:
            executor: Async function that executes SQL (takes sql string)
            max_workers: Max concurrent workers (controls semaphore, default: 4)
            verbose: Print progress (default: True)

        Returns:
            dict with execution results: {
                "completed": list of completed query IDs,
                "failed": list of (query_id, error) tuples,
                "elapsed_seconds": total execution time,
                "total_queries": total number of queries
            }

        Example:
            async def execute_sql(sql: str):
                # Your async database connection
                await async_conn.execute(sql)

            result = await pipeline.async_run(executor=execute_sql, max_workers=4)
            print(f"Completed {len(result['completed'])} queries")
        """
        if verbose:
            print(f"🚀 Starting async pipeline execution ({len(self.table_graph.queries)} queries)")
            print()

        # Track completed queries
        completed = set()
        failed = []
        start_time = time.time()

        # Group queries by level for concurrent execution
        levels = self._get_execution_levels()

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_workers)

        # Execute level by level
        for level_num, level_queries in enumerate(levels, 1):
            if verbose:
                print(f"📊 Level {level_num}: {len(level_queries)} queries")

            async def execute_with_semaphore(query_id: str, sql: str):
                """Execute query with semaphore for concurrency control"""
                async with semaphore:
                    try:
                        await executor(sql)
                        completed.add(query_id)
                        if verbose:
                            print(f"  ✅ {query_id}")
                    except Exception as e:
                        failed.append((query_id, str(e)))
                        if verbose:
                            print(f"  ❌ {query_id}: {e}")

            # Execute queries in this level concurrently
            tasks = []
            for query_id in level_queries:
                query = self.table_graph.queries[query_id]
                task = execute_with_semaphore(query_id, query.sql)
                tasks.append(task)

            # Wait for all tasks in this level to complete
            await asyncio.gather(*tasks)

            if verbose:
                print()

        elapsed = time.time() - start_time

        # Summary
        if verbose:
            print("=" * 60)
            print(f"✅ Pipeline completed in {elapsed:.2f}s")
            print(f"   Successful: {len(completed)}")
            print(f"   Failed: {len(failed)}")
            if failed:
                print("\n⚠️  Failed queries:")
                for query_id, error in failed:
                    print(f"   - {query_id}: {error}")
            print("=" * 60)

        return {
            "completed": list(completed),
            "failed": failed,
            "elapsed_seconds": elapsed,
            "total_queries": len(self.table_graph.queries),
        }

    # ========================================================================
    # Validation Methods
    # ========================================================================

    def get_all_issues(self) -> List["ValidationIssue"]:
        """
        Get all validation issues from all queries in the pipeline.

        Returns combined list of issues from:
        - Individual query lineage graphs
        - Pipeline-level lineage graph

        Returns:
            List of ValidationIssue objects
        """
        all_issues: List[ValidationIssue] = []

        # Collect issues from individual query lineage graphs
        for _query_id, query_lineage in self.query_graphs.items():
            all_issues.extend(query_lineage.issues)

        # Add pipeline-level issues
        all_issues.extend(self.column_graph.issues)

        return all_issues

    def get_issues(
        self,
        severity: Optional[str | IssueSeverity] = None,
        category: Optional[str | IssueCategory] = None,
        query_id: Optional[str] = None,
    ) -> List["ValidationIssue"]:
        """
        Get filtered validation issues.

        Args:
            severity: Filter by severity ('error', 'warning', 'info' or IssueSeverity enum)
            category: Filter by category (string or IssueCategory enum)
            query_id: Filter by query ID

        Returns:
            Filtered list of ValidationIssue objects

        Example:
            # Get all errors (using string)
            errors = pipeline.get_issues(severity='error')

            # Get all errors (using enum)
            errors = pipeline.get_issues(severity=IssueSeverity.ERROR)

            # Get all star-related issues
            star_issues = pipeline.get_issues(category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES)

            # Get all issues from a specific query
            query_issues = pipeline.get_issues(query_id='query_1')
        """
        issues = self.get_all_issues()

        # Filter by severity
        if severity:
            severity_enum = (
                severity if isinstance(severity, IssueSeverity) else IssueSeverity(severity)
            )
            issues = [i for i in issues if i.severity == severity_enum]

        # Filter by category
        if category:
            category_enum = (
                category if isinstance(category, IssueCategory) else IssueCategory(category)
            )
            issues = [i for i in issues if i.category == category_enum]

        # Filter by query_id
        if query_id:
            issues = [i for i in issues if i.query_id == query_id]

        return issues

    def has_errors(self) -> bool:
        """Check if pipeline has any ERROR-level issues"""
        return any(i.severity.value == "error" for i in self.get_all_issues())

    def has_warnings(self) -> bool:
        """Check if pipeline has any WARNING-level issues"""
        return any(i.severity.value == "warning" for i in self.get_all_issues())

    def print_issues(self, severity: Optional[str | IssueSeverity] = None):
        """
        Print all validation issues in a human-readable format.

        Args:
            severity: Optional filter by severity ('error', 'warning', 'info' or IssueSeverity enum)
        """
        issues = self.get_issues(severity=severity) if severity else self.get_all_issues()

        if not issues:
            print("✅ No validation issues found!")
            return

        # Group by severity
        from collections import defaultdict

        by_severity = defaultdict(list)
        for issue in issues:
            by_severity[issue.severity.value].append(issue)

        # Print by severity (errors first, then warnings, then info)
        for sev in ["error", "warning", "info"]:
            if sev not in by_severity:
                continue

            issues_list = by_severity[sev]
            icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[sev]
            print(f"\n{icon} {sev.upper()} ({len(issues_list)})")
            print("=" * 80)

            for issue in issues_list:
                print(f"\n{issue}")


__all__ = [
    "PipelineLineageBuilder",
    "Pipeline",
]
