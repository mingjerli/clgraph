"""
Lineage tracing module for column-level BFS traversals.

Provides pure functions that operate on column/edge graph data structures
without depending on the Pipeline class.  The Pipeline delegates to these
functions for all lineage/impact queries.

Public API
----------
trace_backward            - BFS backward to ultimate source columns
trace_backward_full       - BFS backward with all intermediate nodes + edges
trace_forward             - BFS forward to final descendant columns
trace_forward_full        - BFS forward with all intermediate nodes + edges
get_table_lineage_path    - Table-level backward path (real tables only)
get_table_impact_path     - Table-level forward path (real tables only)
get_lineage_path          - Shortest edge-path between two specific columns

Internal helpers
----------------
_get_incoming  - Look up incoming edges from the adjacency index
_get_outgoing  - Look up outgoing edges from the adjacency index
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

from .models import ColumnEdge, ColumnNode

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_incoming(
    incoming_index: Dict[str, List[ColumnEdge]],
    full_name: str,
) -> List[ColumnEdge]:
    """Return all incoming edges for *full_name* (empty list if none)."""
    return incoming_index.get(full_name, [])


def _get_outgoing(
    outgoing_index: Dict[str, List[ColumnEdge]],
    full_name: str,
) -> List[ColumnEdge]:
    """Return all outgoing edges for *full_name* (empty list if none)."""
    return outgoing_index.get(full_name, [])


# ---------------------------------------------------------------------------
# Backward tracing
# ---------------------------------------------------------------------------


def trace_backward(
    columns: Dict[str, ColumnNode],
    incoming_index: Dict[str, List[ColumnEdge]],
    table_name: str,
    column_name: str,
) -> List[ColumnNode]:
    """
    BFS backward from *table_name.column_name* to its ultimate source columns.

    Returns only leaf columns (those with no incoming edges).  Use
    :func:`trace_backward_full` when you need all intermediate nodes and edges.

    Args:
        columns:        Mapping of full_name -> ColumnNode for the pipeline.
        incoming_index: Adjacency dict mapping full_name -> [ColumnEdge].
        table_name:     Table containing the column to trace.
        column_name:    Column name to trace.

    Returns:
        List of source :class:`~clgraph.models.ColumnNode` objects.
    """
    target_columns = [
        col
        for col in columns.values()
        if col.table_name == table_name and col.column_name == column_name
    ]

    if not target_columns:
        return []

    # Prefer output-layer columns as the starting point
    output_cols = [c for c in target_columns if c.layer == "output"]
    start_columns = output_cols if output_cols else target_columns

    visited: set = set()
    queue: deque = deque(start_columns)
    sources: List[ColumnNode] = []

    while queue:
        current = queue.popleft()
        if current.full_name in visited:
            continue
        visited.add(current.full_name)

        incoming = _get_incoming(incoming_index, current.full_name)

        if not incoming:
            sources.append(current)
        else:
            for edge in incoming:
                queue.append(edge.from_node)

    return sources


def trace_backward_full(
    columns: Dict[str, ColumnNode],
    incoming_index: Dict[str, List[ColumnEdge]],
    table_name: str,
    column_name: str,
    include_ctes: bool = True,
) -> Tuple[List[ColumnNode], List[ColumnEdge]]:
    """
    BFS backward with full lineage transparency.

    Collects every node and edge encountered while traversing backward from
    *table_name.column_name*.

    Args:
        columns:        Mapping of full_name -> ColumnNode for the pipeline.
        incoming_index: Adjacency dict mapping full_name -> [ColumnEdge].
        table_name:     Table containing the column to trace.
        column_name:    Column name to trace.
        include_ctes:   When False, CTE nodes are skipped in the result (but
                        the traversal still passes through them).

    Returns:
        ``(nodes, edges)`` — lists of nodes and edges in BFS order.
    """
    target_columns = [
        col
        for col in columns.values()
        if col.table_name == table_name and col.column_name == column_name
    ]

    if not target_columns:
        return [], []

    output_cols = [c for c in target_columns if c.layer == "output"]
    start_columns = output_cols if output_cols else target_columns

    visited: set = set()
    queue: deque = deque(start_columns)
    all_nodes: List[ColumnNode] = []
    all_edges: List[ColumnEdge] = []

    while queue:
        current = queue.popleft()
        if current.full_name in visited:
            continue
        visited.add(current.full_name)

        if not include_ctes and current.layer == "cte":
            # Traverse through CTEs but don't include them
            for edge in _get_incoming(incoming_index, current.full_name):
                queue.append(edge.from_node)
            continue

        all_nodes.append(current)

        for edge in _get_incoming(incoming_index, current.full_name):
            all_edges.append(edge)
            queue.append(edge.from_node)

    return all_nodes, all_edges


# ---------------------------------------------------------------------------
# Forward tracing
# ---------------------------------------------------------------------------


def trace_forward(
    columns: Dict[str, ColumnNode],
    outgoing_index: Dict[str, List[ColumnEdge]],
    table_name: str,
    column_name: str,
) -> List[ColumnNode]:
    """
    BFS forward from *table_name.column_name* to its final descendant columns.

    Returns only leaf columns (those with no outgoing edges).  Use
    :func:`trace_forward_full` when you need all intermediate nodes and edges.

    Args:
        columns:        Mapping of full_name -> ColumnNode for the pipeline.
        outgoing_index: Adjacency dict mapping full_name -> [ColumnEdge].
        table_name:     Table containing the column to trace.
        column_name:    Column name to trace.

    Returns:
        List of descendant :class:`~clgraph.models.ColumnNode` objects.
    """
    source_columns = [
        col
        for col in columns.values()
        if col.table_name == table_name and col.column_name == column_name
    ]

    if not source_columns:
        return []

    # Prefer input-layer columns as the starting point
    input_cols = [c for c in source_columns if c.layer == "input"]
    start_columns = input_cols if input_cols else source_columns

    visited: set = set()
    queue: deque = deque(start_columns)
    descendants: List[ColumnNode] = []

    while queue:
        current = queue.popleft()
        if current.full_name in visited:
            continue
        visited.add(current.full_name)

        outgoing = _get_outgoing(outgoing_index, current.full_name)

        if not outgoing:
            descendants.append(current)
        else:
            has_unvisited = False
            for edge in outgoing:
                if edge.to_node.full_name not in visited:
                    has_unvisited = True
                    queue.append(edge.to_node)
            # If all outgoing targets are already visited, treat as terminal
            if not has_unvisited:
                descendants.append(current)

    return descendants


def trace_forward_full(
    columns: Dict[str, ColumnNode],
    outgoing_index: Dict[str, List[ColumnEdge]],
    table_name: str,
    column_name: str,
    include_ctes: bool = True,
) -> Tuple[List[ColumnNode], List[ColumnEdge]]:
    """
    BFS forward with full impact transparency.

    Collects every node and edge encountered while traversing forward from
    *table_name.column_name*.

    Args:
        columns:        Mapping of full_name -> ColumnNode for the pipeline.
        outgoing_index: Adjacency dict mapping full_name -> [ColumnEdge].
        table_name:     Table containing the column to trace.
        column_name:    Column name to trace.
        include_ctes:   When False, CTE nodes are skipped in the result (but
                        the traversal still passes through them).

    Returns:
        ``(nodes, edges)`` — lists of nodes and edges in BFS order.
    """
    source_columns = [
        col
        for col in columns.values()
        if col.table_name == table_name and col.column_name == column_name
    ]

    if not source_columns:
        return [], []

    # Prefer input/output layer columns as starting point
    input_cols = [c for c in source_columns if c.layer in ("input", "output")]
    start_columns = input_cols if input_cols else source_columns

    visited: set = set()
    queue: deque = deque(start_columns)
    all_nodes: List[ColumnNode] = []
    all_edges: List[ColumnEdge] = []

    while queue:
        current = queue.popleft()
        if current.full_name in visited:
            continue
        visited.add(current.full_name)

        if not include_ctes and current.layer == "cte":
            for edge in _get_outgoing(outgoing_index, current.full_name):
                queue.append(edge.to_node)
            continue

        all_nodes.append(current)

        for edge in _get_outgoing(outgoing_index, current.full_name):
            all_edges.append(edge)
            queue.append(edge.to_node)

    return all_nodes, all_edges


# ---------------------------------------------------------------------------
# Table-level path helpers
# ---------------------------------------------------------------------------


def get_table_lineage_path(
    columns: Dict[str, ColumnNode],
    incoming_index: Dict[str, List[ColumnEdge]],
    table_name: str,
    column_name: str,
    tables: Optional[Dict] = None,
) -> List[Tuple[str, str, str]]:
    """
    Simplified table-level lineage path (backward direction, real tables only).

    Internally calls :func:`trace_backward_full` with ``include_ctes=False``
    and returns deduplicated ``(table_name, column_name, query_id)`` tuples.

    Args:
        columns:        Mapping of full_name -> ColumnNode.
        incoming_index: Adjacency dict mapping full_name -> [ColumnEdge].
        table_name:     Table containing the column to trace.
        column_name:    Column name to trace.
        tables:         Optional table registry (currently reserved for future
                        filtering; pass ``pipeline.table_graph.tables``).

    Returns:
        Deduplicated list of ``(table_name, column_name, query_id)`` tuples
        ordered from target back to ultimate sources.
    """
    nodes, _ = trace_backward_full(
        columns,
        incoming_index,
        table_name,
        column_name,
        include_ctes=False,
    )

    seen: set = set()
    result: List[Tuple[str, str, str]] = []
    for node in nodes:
        key = (node.table_name, node.column_name)
        if key not in seen:
            seen.add(key)
            result.append((node.table_name, node.column_name, node.query_id))

    return result


def get_table_impact_path(
    columns: Dict[str, ColumnNode],
    outgoing_index: Dict[str, List[ColumnEdge]],
    table_name: str,
    column_name: str,
    tables: Optional[Dict] = None,
) -> List[Tuple[str, str, str]]:
    """
    Simplified table-level impact path (forward direction, real tables only).

    Internally calls :func:`trace_forward_full` with ``include_ctes=False``
    and returns deduplicated ``(table_name, column_name, query_id)`` tuples.

    Args:
        columns:        Mapping of full_name -> ColumnNode.
        outgoing_index: Adjacency dict mapping full_name -> [ColumnEdge].
        table_name:     Table containing the column to trace.
        column_name:    Column name to trace.
        tables:         Optional table registry (currently reserved for future
                        filtering; pass ``pipeline.table_graph.tables``).

    Returns:
        Deduplicated list of ``(table_name, column_name, query_id)`` tuples
        ordered from source forward to final impact.
    """
    nodes, _ = trace_forward_full(
        columns,
        outgoing_index,
        table_name,
        column_name,
        include_ctes=False,
    )

    seen: set = set()
    result: List[Tuple[str, str, str]] = []
    for node in nodes:
        key = (node.table_name, node.column_name)
        if key not in seen:
            seen.add(key)
            result.append((node.table_name, node.column_name, node.query_id))

    return result


# ---------------------------------------------------------------------------
# Point-to-point path
# ---------------------------------------------------------------------------


def get_lineage_path(
    columns: Dict[str, ColumnNode],
    outgoing_index: Dict[str, List[ColumnEdge]],
    from_table: str,
    from_column: str,
    to_table: str,
    to_column: str,
) -> List[ColumnEdge]:
    """
    BFS to find the shortest edge-path between two columns.

    Args:
        columns:        Mapping of full_name -> ColumnNode.
        outgoing_index: Adjacency dict mapping full_name -> [ColumnEdge].
        from_table:     Source table name.
        from_column:    Source column name.
        to_table:       Target table name.
        to_column:      Target column name.

    Returns:
        Ordered list of :class:`~clgraph.models.ColumnEdge` objects forming
        the path, or an empty list if no path exists.
    """
    from_columns = [
        col
        for col in columns.values()
        if col.table_name == from_table and col.column_name == from_column
    ]
    to_columns = [
        col
        for col in columns.values()
        if col.table_name == to_table and col.column_name == to_column
    ]

    if not from_columns or not to_columns:
        return []

    to_full_names = {col.full_name for col in to_columns}

    queue: deque = deque((col, []) for col in from_columns)
    visited: set = set()

    while queue:
        current, path = queue.popleft()
        if current.full_name in visited:
            continue
        visited.add(current.full_name)

        if current.full_name in to_full_names:
            return path

        for edge in _get_outgoing(outgoing_index, current.full_name):
            queue.append((edge.to_node, path + [edge]))

    return []
