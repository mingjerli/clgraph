"""
Lineage tracing algorithms for column lineage analysis.

This module provides the LineageTracer class which contains all lineage
traversal algorithms extracted from the Pipeline class.

The LineageTracer operates on a Pipeline's column graph to perform:
- Backward lineage tracing (finding sources)
- Forward lineage tracing (finding descendants/impact)
- Lineage path finding between columns
- Table-level lineage views
"""

from collections import deque
from typing import TYPE_CHECKING, List, Optional, Tuple

from .models import ColumnEdge, ColumnNode

if TYPE_CHECKING:
    from .pipeline import Pipeline


class LineageTracer:
    """
    Lineage traversal algorithms for Pipeline column graphs.

    This class is extracted from Pipeline to follow the Single Responsibility
    Principle. It contains all lineage tracing algorithms that operate on
    the Pipeline's column graph.

    The tracer is lazily initialized by Pipeline when first needed.

    Example (via Pipeline - recommended):
        pipeline = Pipeline(queries, dialect="bigquery")
        sources = pipeline.trace_column_backward("output.table", "column")

    Example (direct usage - advanced):
        from clgraph.lineage_tracer import LineageTracer

        tracer = LineageTracer(pipeline)
        sources = tracer.trace_backward("output.table", "column")
    """

    def __init__(self, pipeline: "Pipeline"):
        """
        Initialize LineageTracer with a Pipeline reference.

        Args:
            pipeline: The Pipeline instance to trace lineage in.
        """
        self._pipeline = pipeline

    def trace_backward(self, table_name: str, column_name: str) -> List[ColumnNode]:
        """
        Trace a column backward to its ultimate sources.
        Returns list of source columns across all queries.

        For full lineage path with all intermediate nodes, use trace_backward_full().

        Args:
            table_name: The table containing the column to trace
            column_name: The column name to trace

        Returns:
            List of source ColumnNodes (columns with no incoming edges)
        """
        # Find the target column(s) - there may be multiple with same table.column
        # from different queries. For output columns, we want the one with layer="output"
        target_columns = [
            col
            for col in self._pipeline.columns.values()
            if col.table_name == table_name and col.column_name == column_name
        ]

        if not target_columns:
            return []

        # Prefer output layer columns as starting point for backward tracing
        output_cols = [c for c in target_columns if c.layer == "output"]
        start_columns = output_cols if output_cols else target_columns

        # BFS backward through edges
        visited = set()
        queue = deque(start_columns)
        sources = []

        while queue:
            current = queue.popleft()
            if current.full_name in visited:
                continue
            visited.add(current.full_name)

            # Find incoming edges
            incoming = self._pipeline._get_incoming_edges(current.full_name)

            if not incoming:
                # No incoming edges = source column
                sources.append(current)
            else:
                for edge in incoming:
                    queue.append(edge.from_node)

        return sources

    def trace_backward_full(
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
        """
        # Find the target column(s)
        target_columns = [
            col
            for col in self._pipeline.columns.values()
            if col.table_name == table_name and col.column_name == column_name
        ]

        if not target_columns:
            return [], []

        # Prefer output layer columns as starting point
        output_cols = [c for c in target_columns if c.layer == "output"]
        start_columns = output_cols if output_cols else target_columns

        # BFS backward through edges, collecting all nodes and edges
        visited = set()
        queue = deque(start_columns)
        all_nodes = []
        all_edges = []

        while queue:
            current = queue.popleft()
            if current.full_name in visited:
                continue
            visited.add(current.full_name)

            # Optionally skip CTE columns
            if not include_ctes and current.layer == "cte":
                # Still need to traverse through CTEs to find real tables
                incoming = self._pipeline._get_incoming_edges(current.full_name)
                for edge in incoming:
                    queue.append(edge.from_node)
                continue

            all_nodes.append(current)

            # Find incoming edges
            incoming = self._pipeline._get_incoming_edges(current.full_name)

            for edge in incoming:
                all_edges.append(edge)
                queue.append(edge.from_node)

        return all_nodes, all_edges

    def trace_forward(self, table_name: str, column_name: str) -> List[ColumnNode]:
        """
        Trace a column forward to see what depends on it.
        Returns list of final downstream columns across all queries.

        For full impact path with all intermediate nodes, use trace_forward_full().

        Args:
            table_name: The table containing the column to trace
            column_name: The column name to trace

        Returns:
            List of final ColumnNodes (columns with no outgoing edges)
        """
        # Find the source column(s) - there may be multiple with same table.column
        # from different queries. For input columns, we want the one with layer="input"
        source_columns = [
            col
            for col in self._pipeline.columns.values()
            if col.table_name == table_name and col.column_name == column_name
        ]

        if not source_columns:
            return []

        # Prefer input layer columns as starting point for forward tracing
        input_cols = [c for c in source_columns if c.layer == "input"]
        start_columns = input_cols if input_cols else source_columns

        # BFS forward through edges
        visited = set()
        queue = deque(start_columns)
        descendants = []

        while queue:
            current = queue.popleft()
            if current.full_name in visited:
                continue
            visited.add(current.full_name)

            # Find outgoing edges
            outgoing = self._pipeline._get_outgoing_edges(current.full_name)

            if not outgoing:
                # No outgoing edges = final column
                descendants.append(current)
            else:
                for edge in outgoing:
                    queue.append(edge.to_node)

        return descendants

    def trace_forward_full(
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
        """
        # Find the source column(s)
        source_columns = [
            col
            for col in self._pipeline.columns.values()
            if col.table_name == table_name and col.column_name == column_name
        ]

        if not source_columns:
            return [], []

        # Prefer input/output layer columns as starting point
        input_cols = [c for c in source_columns if c.layer in ("input", "output")]
        start_columns = input_cols if input_cols else source_columns

        # BFS forward through edges, collecting all nodes and edges
        visited = set()
        queue = deque(start_columns)
        all_nodes = []
        all_edges = []

        while queue:
            current = queue.popleft()
            if current.full_name in visited:
                continue
            visited.add(current.full_name)

            # Optionally skip CTE columns
            if not include_ctes and current.layer == "cte":
                # Still need to traverse through CTEs to find real tables
                outgoing = self._pipeline._get_outgoing_edges(current.full_name)
                for edge in outgoing:
                    queue.append(edge.to_node)
                continue

            all_nodes.append(current)

            # Find outgoing edges
            outgoing = self._pipeline._get_outgoing_edges(current.full_name)

            for edge in outgoing:
                all_edges.append(edge)
                queue.append(edge.to_node)

        return all_nodes, all_edges

    def get_lineage_path(
        self, from_table: str, from_column: str, to_table: str, to_column: str
    ) -> List[ColumnEdge]:
        """
        Find the lineage path between two columns.
        Returns list of edges connecting them (if path exists).

        Args:
            from_table: Source table name
            from_column: Source column name
            to_table: Destination table name
            to_column: Destination column name

        Returns:
            List of ColumnEdges forming the path, or empty list if no path exists
        """
        # Find source columns by table and column name
        from_columns = [
            col
            for col in self._pipeline.columns.values()
            if col.table_name == from_table and col.column_name == from_column
        ]

        to_columns = [
            col
            for col in self._pipeline.columns.values()
            if col.table_name == to_table and col.column_name == to_column
        ]

        if not from_columns or not to_columns:
            return []

        # Get target full_names for matching
        to_full_names = {col.full_name for col in to_columns}

        # BFS with path tracking, starting from all matching source columns
        queue = deque((col, []) for col in from_columns)
        visited = set()

        while queue:
            current, path = queue.popleft()
            if current.full_name in visited:
                continue
            visited.add(current.full_name)

            if current.full_name in to_full_names:
                return path

            # Find outgoing edges
            for edge in self._pipeline._get_outgoing_edges(current.full_name):
                queue.append((edge.to_node, path + [edge]))

        return []  # No path found

    def get_table_lineage_path(
        self, table_name: str, column_name: str
    ) -> List[Tuple[str, str, Optional[str]]]:
        """
        Get simplified table-level lineage path for a column.

        Returns list of (table_name, column_name, query_id) tuples representing
        the lineage through real tables only (skipping CTEs).

        This provides a clear view of how data flows between tables in your pipeline.

        Args:
            table_name: The table containing the column to trace
            column_name: The column name to trace

        Returns:
            List of tuples: (table_name, column_name, query_id)
        """
        nodes, _ = self.trace_backward_full(table_name, column_name, include_ctes=False)

        # Deduplicate by table.column (keep first occurrence which is closest to target)
        seen = set()
        result = []
        for node in nodes:
            key = (node.table_name, node.column_name)
            if key not in seen:
                seen.add(key)
                result.append((node.table_name, node.column_name, node.query_id))

        return result

    def get_table_impact_path(
        self, table_name: str, column_name: str
    ) -> List[Tuple[str, str, Optional[str]]]:
        """
        Get simplified table-level impact path for a column.

        Returns list of (table_name, column_name, query_id) tuples representing
        the downstream impact through real tables only (skipping CTEs).

        This provides a clear view of how a source column impacts downstream tables.

        Args:
            table_name: The table containing the column to trace
            column_name: The column name to trace

        Returns:
            List of tuples: (table_name, column_name, query_id)
        """
        nodes, _ = self.trace_forward_full(table_name, column_name, include_ctes=False)

        # Deduplicate by table.column (keep first occurrence which is closest to source)
        seen = set()
        result = []
        for node in nodes:
            key = (node.table_name, node.column_name)
            if key not in seen:
                seen.add(key)
                result.append((node.table_name, node.column_name, node.query_id))

        return result


__all__ = ["LineageTracer"]
