"""
PipelineLineageBuilder: builds unified lineage from multiple queries.

Extracted from pipeline.py to keep file sizes manageable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import sqlglot.errors
from sqlglot import exp

from .lineage_builder import RecursiveLineageBuilder
from .models import (
    ColumnEdge,
    ColumnLineageGraph,
    ColumnNode,
    DescriptionSource,
    ParsedQuery,
)
from .table import TableDependencyGraph

if TYPE_CHECKING:
    from .pipeline import Pipeline

logger = logging.getLogger(__name__)


class PipelineLineageBuilder:
    """
    Builds unified lineage graph from multiple queries.
    Combines single-query lineage with cross-query connections.
    """

    def build(self, pipeline_or_graph: "Pipeline | TableDependencyGraph") -> "Pipeline":
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
        # Import locally at runtime to avoid circular imports
        from .pipeline import Pipeline

        # Handle backward compatibility: accept TableDependencyGraph directly
        if isinstance(pipeline_or_graph, TableDependencyGraph):
            pipeline = Pipeline._create_empty(pipeline_or_graph)
        else:
            pipeline = pipeline_or_graph

        table_graph = pipeline.table_graph

        # Step 1: Topological sort
        sorted_query_ids = table_graph.topological_sort()
        self.sorted_query_ids = sorted_query_ids

        # Step 2: Process each query
        for query_id in sorted_query_ids:
            query = table_graph.queries[query_id]

            # Step 2a: Run single-query lineage
            try:
                # Extract SELECT statement from DDL/DML if needed
                sql_for_lineage = self._extract_select_from_query(query, pipeline.dialect)

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
                    # Persist unit graph for cross-query edge construction
                    pipeline._unit_graphs[query_id] = lineage_builder.unit_graph

                    # Step 2b: Add columns to pipeline graph
                    self._add_query_columns(pipeline, query, query_lineage)

                    # Step 2c: Add within-query edges
                    self._add_query_edges(pipeline, query, query_lineage)
                else:
                    # No SELECT to analyze (e.g., UPDATE without SELECT)
                    logger.info("Skipping lineage for %s (no SELECT statement)", query_id)
            except (sqlglot.errors.SqlglotError, KeyError, ValueError, TypeError) as e:
                # If lineage fails due to SQL parsing or data issues, skip this query
                logger.warning("Failed to build lineage for %s: %s", query_id, e)
                logger.debug("Traceback for %s lineage failure", query_id, exc_info=True)
                continue

        # Step 3: Add cross-query edges
        self._add_cross_query_edges(pipeline)

        # Step 4: Add cross-query edges for self-read columns
        self._add_self_read_cross_query_edges(pipeline, sorted_query_ids)

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

            # Detect self-read columns for node_type override
            is_self_read = self._is_self_read_column(node, query)

            # Skip if column already exists (shared physical table column)
            if full_name in pipeline.columns:
                # Diagnostic: log when a physical-table column is dropped by
                # the dedup guard for a query with self-referenced tables.
                if getattr(query, "self_referenced_tables", set()):
                    logger.debug(
                        "Dedup guard dropped %s for query %s (self_referenced_tables=%s)",
                        full_name,
                        query.query_id,
                        query.self_referenced_tables,
                    )
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
                node_type="self_read" if is_self_read else node.node_type,
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
        # Compute statement_order from sorted_query_ids
        stmt_order = None
        if hasattr(self, "sorted_query_ids"):
            try:
                stmt_order = self.sorted_query_ids.index(query.query_id)
            except ValueError:
                pass

        for edge in query_lineage.edges:
            from_full = self._make_full_name(edge.from_node, query)
            to_full = self._make_full_name(edge.to_node, query)

            if from_full in pipeline.columns and to_full in pipeline.columns:
                # Determine edge_role for self-read edges
                edge_role = None
                if self._is_self_read_column(edge.from_node, query):
                    edge_role = "prior_state_read"

                # Normal case: both nodes exist
                pipeline_edge = ColumnEdge(
                    from_node=pipeline.columns[from_full],
                    to_node=pipeline.columns[to_full],
                    edge_type=edge.edge_type if hasattr(edge, "edge_type") else edge.transformation,
                    transformation=edge.transformation,
                    context=edge.context,
                    query_id=query.query_id,
                    statement_order=stmt_order,
                    edge_role=edge_role,
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
                    # Preserve JOIN predicate metadata
                    is_join_predicate=getattr(edge, "is_join_predicate", False),
                    join_condition=getattr(edge, "join_condition", None),
                    join_side=getattr(edge, "join_side", None),
                    # Preserve WHERE filter metadata
                    is_where_filter=getattr(edge, "is_where_filter", False),
                    where_condition=getattr(edge, "where_condition", None),
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

        # Connect physical-table columns into CTE columns that read from them.
        # This handles the case where a CTE aliases a physical table (e.g.
        # ``WITH orders AS (SELECT * FROM marts.orders)``). Without this step,
        # lineage breaks at the query boundary because the CTE columns share a
        # name with the physical table but are not linked to it.
        self._add_cte_cross_query_edges(pipeline)

    def _add_cte_cross_query_edges(self, pipeline: "Pipeline"):
        """Link physical-table columns into CTE columns that read from them."""
        from .models import QueryUnitType

        for query_id, unit_graph in pipeline._unit_graphs.items():
            # Build mapping {cte_alias: physical_source_table} for unambiguous CTEs
            cte_to_source: Dict[str, str] = {}
            for unit in unit_graph.units.values():
                if unit.unit_type != QueryUnitType.CTE:
                    continue
                if not unit.name:
                    continue
                # Only unambiguous: exactly one physical upstream, no unit upstream
                if len(unit.depends_on_tables) == 1 and not unit.depends_on_units:
                    cte_to_source[unit.name] = unit.depends_on_tables[0]

            if not cte_to_source:
                continue

            # Resolve each physical source to the fully-qualified table name used
            # in the pipeline (matching ParsedQuery.source_tables), since
            # depends_on_tables may carry short names.
            query = pipeline.table_graph.queries.get(query_id)
            source_tables = set(query.source_tables) if query else set()

            # Iterate CTE columns for this query and connect them
            for col in list(pipeline.columns.values()):
                if col.query_id != query_id:
                    continue
                if not col.unit_id or not col.unit_id.startswith("cte:"):
                    continue
                if col.is_star:
                    continue
                cte_name = col.table_name
                if cte_name not in cte_to_source:
                    continue
                physical = self._resolve_physical_table(
                    cte_to_source[cte_name], pipeline, source_tables
                )
                if not physical:
                    continue
                physical_full = f"{physical}.{col.column_name}"
                source_col = pipeline.columns.get(physical_full)
                if source_col is None:
                    continue
                # Dedup using O(1) incoming adjacency index
                incoming = pipeline._get_incoming_edges(col.full_name)
                if any(e.from_node.full_name == physical_full for e in incoming):
                    continue
                pipeline.add_edge(
                    ColumnEdge(
                        from_node=source_col,
                        to_node=col,
                        edge_type="cross_query",
                        context="cross_query",
                        transformation=f"{physical} -> CTE {cte_name}",
                        query_id=None,
                    )
                )

    def _add_self_read_cross_query_edges(self, pipeline: "Pipeline", sorted_query_ids: List[str]):
        """
        Connect prior-statement output columns to self-read input columns.

        For each query with self_referenced_tables, find self-read input nodes
        (node_type=="self_read") and connect them to the most recent prior query
        that wrote that specific column to the same table. If no prior writer
        exists, create a pre-pipeline source-state node.
        """
        # Build index: query_id -> topo sort position
        query_order = {qid: i for i, qid in enumerate(sorted_query_ids)}

        for query_id in sorted_query_ids:
            query = pipeline.table_graph.queries.get(query_id)
            if not query:
                continue
            self_ref_tables = getattr(query, "self_referenced_tables", set())
            if not self_ref_tables:
                continue

            current_order = query_order.get(query_id, 0)

            # Find all self-read input nodes for this query
            self_read_nodes = [
                col
                for col in pipeline.columns.values()
                if col.query_id == query_id and col.node_type == "self_read"
            ]

            for sr_node in self_read_nodes:
                # Extract the physical table name from the full_name pattern:
                # "{query_id}:self_read:{table}.{column}"
                parts = sr_node.full_name.split(":self_read:", 1)
                if len(parts) != 2:
                    continue
                table_col = parts[1]  # e.g., "dim_customer.id"

                # Find the most recent prior query that wrote this column
                prior_output = None
                best_order = -1
                for col in pipeline.columns.values():
                    if (
                        col.full_name == table_col
                        and col.layer == "output"
                        and col.query_id
                        and col.query_id != query_id
                    ):
                        col_order = query_order.get(col.query_id, -1)
                        if col_order < current_order and col_order > best_order:
                            best_order = col_order
                            prior_output = col

                if prior_output:
                    # Connect prior output to self-read input
                    edge = ColumnEdge(
                        from_node=prior_output,
                        to_node=sr_node,
                        edge_type="cross_query_self_ref",
                        context="cross_query",
                        transformation=f"prior state of {table_col}",
                        query_id=None,
                        edge_role="cross_query_self_ref",
                        statement_order=current_order,
                    )
                    pipeline.add_edge(edge)
                else:
                    # No prior writer found - create a pre-pipeline source-state node
                    # if it doesn't already exist
                    if table_col not in pipeline.columns:
                        # Parse table and column from table_col
                        last_dot = table_col.rfind(".")
                        if last_dot < 0:
                            continue
                        tbl = table_col[:last_dot]
                        col_name = table_col[last_dot + 1 :]
                        source_node = ColumnNode(
                            column_name=col_name,
                            table_name=tbl,
                            full_name=table_col,
                            layer="input",
                            node_type="source",
                        )
                        pipeline.add_column(source_node)

                    source_col = pipeline.columns.get(table_col)
                    if source_col:
                        edge = ColumnEdge(
                            from_node=source_col,
                            to_node=sr_node,
                            edge_type="cross_query_self_ref",
                            context="cross_query",
                            transformation=f"pre-pipeline state of {table_col}",
                            query_id=None,
                            edge_role="cross_query_self_ref",
                            statement_order=current_order,
                        )
                        pipeline.add_edge(edge)

    @staticmethod
    def _resolve_physical_table(
        tbl: str, pipeline: "Pipeline", source_tables: set
    ) -> Optional[str]:
        """Resolve a short/qualified table name to a fully-qualified pipeline table."""
        if tbl in pipeline.table_graph.tables:
            return tbl
        if tbl in source_tables:
            return tbl
        for st in source_tables:
            if st.endswith(f".{tbl}") or st == tbl:
                return st
        for tn in pipeline.table_graph.tables:
            if tn == tbl or tn.endswith(f".{tbl}"):
                return tn
        return None

    def _infer_table_name(self, node: ColumnNode, query: ParsedQuery) -> Optional[str]:
        """
        Infer which table this column belongs to.
        Maps table references (aliases) to fully qualified names.

        For queries without a destination table (plain SELECT statements),
        output columns are assigned to a virtual result table named '{query_id}_result'.
        This ensures they appear in simplified lineage views.
        """
        # CTE and subquery columns keep their internal table_name (the CTE/subquery
        # alias). Do not map them to physical source tables — they are
        # query-internal structures, not external tables.
        unit_id = node.unit_id
        if unit_id and (unit_id.startswith("cte:") or unit_id.startswith("subq:")):
            return node.table_name

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

    def _is_self_read_column(self, node: ColumnNode, query: ParsedQuery) -> bool:
        """
        Check if an input-layer column is a self-read (reads from a table
        that this query also writes to).
        """
        if node.layer != "input":
            return False
        self_ref_tables = getattr(query, "self_referenced_tables", set())
        if not self_ref_tables:
            return False
        # Resolve alias -> table name if needed
        candidate = node.table_name
        resolved = getattr(query, "self_ref_aliases", {}).get(candidate, candidate)
        if resolved in self_ref_tables:
            return True
        # Also try the inferred table name
        inferred = self._infer_table_name(node, query)
        if inferred:
            resolved_inferred = getattr(query, "self_ref_aliases", {}).get(inferred, inferred)
            if resolved_inferred in self_ref_tables:
                return True
        return False

    def _resolve_self_read_table(self, node: ColumnNode, query: ParsedQuery) -> str:
        """Resolve the physical table name for a self-read column."""
        candidate = self._infer_table_name(node, query) or node.table_name
        return getattr(query, "self_ref_aliases", {}).get(candidate, candidate)

    def _make_full_name(self, node: ColumnNode, query: ParsedQuery) -> str:
        """
        Create fully qualified column name.

        Naming convention:
        - Self-read: {query_id}:self_read:{table_name}.{column_name}
          For input columns that read from a table this query also writes to.

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
        # Check self-read BEFORE physical table check
        if self._is_self_read_column(node, query):
            resolved_table = self._resolve_self_read_table(node, query)
            return f"{query.query_id}:self_read:{resolved_table}.{node.column_name}"

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

    def _extract_select_from_query(
        self, query: ParsedQuery, dialect: str = "bigquery"
    ) -> Optional[str]:
        """
        Extract SELECT statement from DDL/DML queries.
        Single-query lineage only works on SELECT statements, so we need to extract
        the SELECT from CREATE TABLE AS SELECT, INSERT INTO ... SELECT, etc.

        Args:
            query: The parsed query to extract SELECT from
            dialect: SQL dialect for proper SQL serialization (important for functions
                    like DATE_TRUNC which have different argument orders in different dialects)

        Returns:
            The SELECT SQL string, or None if no SELECT found
        """
        ast = query.ast

        # CREATE TABLE/VIEW AS SELECT
        if isinstance(ast, exp.Create):
            if ast.expression and isinstance(ast.expression, exp.Select):
                # Use dialect to ensure proper SQL serialization
                return ast.expression.sql(dialect=dialect)

        # INSERT INTO ... SELECT
        elif isinstance(ast, exp.Insert):
            if ast.expression and isinstance(ast.expression, exp.Select):
                return ast.expression.sql(dialect=dialect)

        # MERGE INTO statement - pass full SQL to lineage builder
        elif isinstance(ast, exp.Merge):
            return query.sql

        # Plain SELECT
        elif isinstance(ast, exp.Select):
            return query.sql

        # UPDATE, DELETE, etc. - no SELECT to extract
        return None
