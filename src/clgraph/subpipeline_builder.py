"""
Subpipeline building component for Pipeline.

This module provides the SubpipelineBuilder class which contains all pipeline
splitting logic extracted from the Pipeline class.

The SubpipelineBuilder handles:
- Building subpipelines for specific target tables
- Splitting pipelines into non-overlapping subpipelines
"""

from collections import deque
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .pipeline import Pipeline


class SubpipelineBuilder:
    """
    Subpipeline building logic for Pipeline.

    This class is extracted from Pipeline to follow the Single Responsibility
    Principle. It contains all subpipeline/split methods that operate on
    the Pipeline's table dependency graph.

    The builder is lazily initialized by Pipeline when first needed.

    Example (via Pipeline - recommended):
        pipeline = Pipeline(queries, dialect="bigquery")
        subpipeline = pipeline.build_subpipeline("analytics.revenue")

    Example (direct usage - advanced):
        from clgraph.subpipeline_builder import SubpipelineBuilder

        builder = SubpipelineBuilder(pipeline)
        subpipeline = builder.build_subpipeline("analytics.revenue")
    """

    def __init__(self, pipeline: "Pipeline"):
        """
        Initialize SubpipelineBuilder with a Pipeline reference.

        Args:
            pipeline: The Pipeline instance to build subpipelines from.
        """
        self._pipeline = pipeline

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
            subpipeline = builder.build_subpipeline("analytics.revenue")
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
            subpipelines = builder.split(
                sinks=[
                    "final_table",           # Single table
                    ["metrics", "summary"],  # Multiple tables in one subpipeline
                    "aggregated_data"        # Another single table
                ]
            )
        """
        # Import Pipeline here to avoid circular import
        from .pipeline import Pipeline

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
                if sink_table not in self._pipeline.table_graph.tables:
                    raise ValueError(
                        f"Sink table '{sink_table}' not found in pipeline. "
                        f"Available tables: {list(self._pipeline.table_graph.tables.keys())}"
                    )

                # Find all queries needed for this sink
                visited = set()
                queue = deque([sink_table])

                while queue:
                    current_table = queue.popleft()
                    if current_table in visited:
                        continue
                    visited.add(current_table)

                    table_node = self._pipeline.table_graph.tables.get(current_table)
                    if not table_node:
                        continue

                    # Add the query that creates this table
                    if table_node.created_by:
                        query_id = table_node.created_by
                        required_queries.add(query_id)

                        # Add source tables to queue
                        query = self._pipeline.table_graph.queries[query_id]
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
            for query_id in self._pipeline.table_graph.topological_sort():
                if query_id in query_ids:
                    query = self._pipeline.table_graph.queries[query_id]
                    subpipeline_query_list.append((query_id, query.sql))

            # Create new Pipeline instance
            subpipeline = Pipeline(subpipeline_query_list, dialect=self._pipeline.dialect)
            subpipelines.append(subpipeline)

        return subpipelines


__all__ = ["SubpipelineBuilder"]
