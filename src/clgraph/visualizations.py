"""
Pure visualization functions for SQL lineage graphs.

These functions translate graph structures into Graphviz DOT format.
No business logic - just presentation layer.
"""

from typing import Optional

import graphviz

from clgraph import ColumnLineageGraph, QueryUnitGraph


def _sanitize_graphviz_id(node_id: str) -> str:
    """
    Sanitize a node ID for use in Graphviz.

    Graphviz interprets colons as node:port syntax, so we need to replace
    them with safe characters.

    Args:
        node_id: The original node ID (e.g., "cte:my_cte", "table:users")

    Returns:
        Sanitized ID safe for Graphviz (e.g., "cte__my_cte", "table__users")
    """
    return node_id.replace(":", "__").replace(".", "_")


def visualize_query_units(query_graph: QueryUnitGraph) -> graphviz.Digraph:
    """
    Create Graphviz visualization of QueryUnitGraph.

    Pure function: Takes QueryUnitGraph, returns Graphviz Digraph.
    No logic - just reads the graph and formats for display.

    Args:
        query_graph: The query unit graph from parser

    Returns:
        graphviz.Digraph object ready to render
    """
    dot = graphviz.Digraph(comment="Query Unit Dependencies")
    dot.attr(rankdir="LR")  # Left to right layout for better flow
    dot.attr("node", shape="box", style="rounded,filled", fontname="Arial", fontsize="12")
    dot.attr("edge", fontsize="10", color="#555555")

    # Color scheme for different unit types
    colors = {
        "main_query": "#4CAF50",  # Green
        "cte": "#2196F3",  # Blue
        "subquery_from": "#FF9800",  # Orange
        "subquery_select": "#FFC107",  # Amber
        "subquery_where": "#9C27B0",  # Purple
        "subquery_having": "#E91E63",  # Pink
        "union": "#00BCD4",  # Cyan
        "intersect": "#00BCD4",  # Cyan
        "except": "#00BCD4",  # Cyan
        "subquery_union": "#80DEEA",  # Light Cyan
        "pivot": "#8BC34A",  # Light Green
        "unpivot": "#CDDC39",  # Lime
    }

    # Icons for different unit types
    icons = {
        "main_query": "🎯",
        "cte": "📦",
        "subquery_from": "🔸",
        "subquery_select": "🔹",
        "subquery_where": "🔶",
        "subquery_having": "🔷",
        "union": "🔀",
        "intersect": "∩",
        "except": "−",
        "subquery_union": "🔹",
        "pivot": "↻",
        "unpivot": "↺",
    }

    # Track table nodes to avoid duplicates
    table_nodes_created = set()

    # Group units by depth for hierarchical layout
    units_by_depth = {}
    max_depth = 0
    for unit in query_graph.units.values():
        depth = unit.depth
        if depth not in units_by_depth:
            units_by_depth[depth] = []
        units_by_depth[depth].append(unit)
        max_depth = max(max_depth, depth)

    # Create ID mapping for all units and tables
    unit_id_map = {}  # original unit_id -> sanitized id
    table_id_map = {}  # original table_id -> sanitized id

    # Pre-compute all mappings
    for unit in query_graph.units.values():
        unit_id_map[unit.unit_id] = _sanitize_graphviz_id(unit.unit_id)
        for table_name in unit.depends_on_tables:
            table_id = f"table:{table_name}"
            table_id_map[table_id] = _sanitize_graphviz_id(table_id)

    # Add query unit nodes grouped by depth (reverse order for LR layout)
    for depth in sorted(units_by_depth.keys(), reverse=True):
        with dot.subgraph() as s:
            s.attr(rank="same")  # Place all nodes at this depth at the same rank
            for unit in units_by_depth[depth]:
                color = colors.get(unit.unit_type.value, "#9E9E9E")
                icon = icons.get(unit.unit_type.value, "❓")

                # Create multi-line label with better formatting
                label = f"{icon} {unit.name}\\n({unit.unit_type.value})\\nDepth: {unit.depth}"

                # Use sanitized ID for Graphviz node, but keep original in tooltip
                safe_id = unit_id_map[unit.unit_id]
                s.node(
                    safe_id,
                    label=label,
                    fillcolor=color,
                    fontcolor="white",
                    tooltip=f"Unit: {unit.name}, Type: {unit.unit_type.value}, Depth: {unit.depth}",
                )

    # Add table nodes (external base tables)
    for unit in query_graph.units.values():
        for table_name in unit.depends_on_tables:
            table_id = f"table:{table_name}"
            safe_table_id = table_id_map[table_id]
            if table_id not in table_nodes_created:
                dot.node(
                    safe_table_id,
                    label=f"📊 {table_name}\\n(base table)",
                    shape="cylinder",
                    fillcolor="#607D8B",
                    fontcolor="white",
                    tooltip=f"External base table: {table_name}",
                )
                table_nodes_created.add(table_id)

    # Add edges - table dependencies first (to control layout)
    for unit in query_graph.units.values():
        safe_unit_id = unit_id_map[unit.unit_id]
        for table_name in unit.depends_on_tables:
            table_id = f"table:{table_name}"
            safe_table_id = table_id_map[table_id]
            dot.edge(
                safe_table_id, safe_unit_id, label="reads from", color="#607D8B", style="dashed"
            )

    # Add edges - unit dependencies (CTEs, subqueries)
    for unit in query_graph.units.values():
        safe_unit_id = unit_id_map[unit.unit_id]
        for dep_unit_id in unit.depends_on_units:
            # Get dependency unit for better labeling
            dep_unit = query_graph.units.get(dep_unit_id)
            edge_label = "uses"
            edge_color = "#2196F3"

            if dep_unit:
                if dep_unit.unit_type.value == "cte":
                    edge_label = "uses CTE"
                elif dep_unit.unit_type.value == "subquery_union":
                    # For set operation branches, show the operation type
                    if unit.set_operation_type:
                        edge_label = f"{unit.set_operation_type.upper()} branch"
                    else:
                        edge_label = "branch"
                    edge_color = "#00BCD4"  # Cyan for set operations

            # Use sanitized IDs for edges
            safe_dep_id = unit_id_map.get(dep_unit_id, _sanitize_graphviz_id(dep_unit_id))
            dot.edge(
                safe_dep_id,
                safe_unit_id,
                label=edge_label,
                color=edge_color,
                penwidth="2.0",  # Thicker edges for unit dependencies
            )

    return dot


def visualize_query_structure_from_lineage(
    lineage_graph: ColumnLineageGraph, query_graph: QueryUnitGraph
) -> graphviz.Digraph:
    """
    Create a high-level query structure visualization by collapsing column nodes
    into QueryUnit/table nodes.

    This provides a clearer view of the overall query structure showing:
    - Base tables (input layer)
    - CTEs
    - Subqueries
    - Main query
    And the relationships between them.

    Args:
        lineage_graph: The column lineage graph (to derive table/unit relationships)
        query_graph: The query unit graph (for unit metadata)

    Returns:
        graphviz.Digraph showing high-level query structure
    """
    dot = graphviz.Digraph(comment="Query Structure")
    dot.attr(rankdir="LR")  # Left to right layout
    dot.attr("node", fontname="Arial", fontsize="12", style="rounded,filled")
    dot.attr("edge", fontsize="10")

    # Color scheme for different node types
    colors = {
        "table": "#607D8B",  # Grey for base tables
        "main_query": "#4CAF50",  # Green
        "cte": "#2196F3",  # Blue
        "subquery_from": "#FF9800",  # Orange
        "subquery_select": "#FFC107",  # Amber
        "subquery_where": "#9C27B0",  # Purple
        "subquery_having": "#E91E63",  # Pink
        "union": "#8BC34A",  # Light Green
        "intersect": "#CDDC39",  # Lime
        "except": "#FFEB3B",  # Yellow
        "subquery_union": "#FFC107",  # Amber
        "pivot": "#00BCD4",  # Cyan
        "unpivot": "#009688",  # Teal
        "subquery_pivot_source": "#FF9800",  # Orange
    }

    # Icons for different node types
    icons = {
        "table": "📊",
        "main_query": "🎯",
        "cte": "📦",
        "subquery_from": "🔸",
        "subquery_select": "🔹",
        "subquery_where": "🔶",
        "subquery_having": "🔷",
        "union": "🔀",
        "intersect": "∩",
        "except": "−",
        "subquery_union": "🔹",
        "pivot": "↻",
        "unpivot": "↺",
        "subquery_pivot_source": "🔸",
    }

    # Track which nodes and edges we've created
    created_nodes = set()
    created_edges = set()

    # Create a mapping from original IDs to sanitized IDs (to avoid Graphviz port syntax issues)
    id_mapping = {}

    # First, collect all base tables from the lineage graph
    base_tables = set()
    for node in lineage_graph.nodes.values():
        if node.layer == "input" and node.unit_id is None:
            base_tables.add(node.table_name)

    # Create nodes for base tables
    for table_name in sorted(base_tables):
        # Create sanitized node ID (replace special chars)
        node_id = f"table_{table_name.replace(':', '_').replace('.', '_')}"
        id_mapping[f"table:{table_name}"] = node_id  # Store mapping for edge creation

        # Count columns in this table
        table_columns = [
            n
            for n in lineage_graph.nodes.values()
            if n.layer == "input" and n.unit_id is None and n.table_name == table_name
        ]
        col_count = len(table_columns)

        dot.node(
            node_id,
            label=f"{icons['table']} {table_name}\\n{col_count} columns",
            shape="cylinder",
            fillcolor=colors["table"],
            fontcolor="white",
        )
        created_nodes.add(node_id)

    # Create nodes for all query units
    for unit in query_graph.units.values():
        unit_type = unit.unit_type.value
        color = colors.get(unit_type, "#9E9E9E")
        icon = icons.get(unit_type, "❓")

        # Count columns in this unit
        unit_columns = [n for n in lineage_graph.nodes.values() if n.unit_id == unit.unit_id]
        col_count = len(unit_columns)

        # Use the unit name directly (which is the CTE name, subquery name, or "output" for main query)
        # For main query, we'll just show "Main Query" with the icon
        display_name = unit.name if unit.name != "output" else "Main Query"
        label = f"{icon} {display_name}\\n{col_count} columns"

        # Create sanitized node ID (replace colons which Graphviz interprets as port syntax)
        sanitized_id = unit.unit_id.replace(":", "_").replace(".", "_")
        id_mapping[unit.unit_id] = sanitized_id

        dot.node(sanitized_id, label=label, fillcolor=color, fontcolor="white", shape="box")
        created_nodes.add(sanitized_id)

    # Now derive edges from column lineage
    # For each edge in column lineage, create a unit-level edge if it crosses units/tables
    for edge in lineage_graph.edges:
        from_node = edge.from_node
        to_node = edge.to_node

        # Determine source and target identifiers (original IDs)
        if from_node.unit_id is None:
            # Source is a base table
            source_original = f"table:{from_node.table_name}"
        else:
            # Source is a query unit
            source_original = from_node.unit_id

        if to_node.unit_id is None:
            # Target is a base table (shouldn't happen, but handle it)
            target_original = f"table:{to_node.table_name}"
        else:
            # Target is a query unit
            target_original = to_node.unit_id

        # Get sanitized IDs from mapping
        source_id = id_mapping.get(
            source_original, source_original.replace(":", "_").replace(".", "_")
        )
        target_id = id_mapping.get(
            target_original, target_original.replace(":", "_").replace(".", "_")
        )

        # Only create edge if source != target (cross-unit dependency)
        if source_id != target_id:
            edge_key = (source_id, target_id)
            if edge_key not in created_edges:
                # Determine edge style based on source type
                if source_id.startswith("table_"):
                    edge_color = "#607D8B"
                    edge_style = "dashed"
                else:
                    edge_color = "#2196F3"
                    edge_style = "solid"

                dot.edge(source_id, target_id, color=edge_color, style=edge_style, penwidth="2.0")
                created_edges.add(edge_key)

    return dot


def visualize_column_lineage(
    lineage_graph: ColumnLineageGraph,
    query_graph: QueryUnitGraph,
    max_nodes: int = 100,
    layers_to_show: Optional[list] = None,
) -> graphviz.Digraph:
    """
    Create Graphviz visualization of ColumnLineageGraph clustered by QueryUnit.

    Pure function: Takes ColumnLineageGraph and QueryUnitGraph, returns Graphviz Digraph.

    Args:
        lineage_graph: The column lineage graph from builder
        query_graph: The query unit graph (for unit metadata)
        max_nodes: Maximum nodes to display (to prevent cluttering)
        layers_to_show: List of layers to include (e.g., ['input', 'output'])
                       If None, shows all layers

    Returns:
        graphviz.Digraph object ready to render
    """
    dot = graphviz.Digraph(comment="Column Lineage")
    dot.attr(rankdir="LR")  # Left to right layout
    dot.attr("node", fontname="Arial", fontsize="10")

    # Color scheme for query unit types
    unit_type_colors = {
        "main_query": "#C8E6C9",  # Light green
        "cte": "#C5CAE9",  # Light purple
        "subquery_from": "#FFCCBC",  # Light orange
        "subquery_select": "#FFE0B2",  # Lighter orange
        "subquery_where": "#E1BEE7",  # Light purple
        "subquery_having": "#F8BBD0",  # Light pink
        "union": "#DCEDC8",  # Light lime
        "intersect": "#F0F4C3",  # Light lime yellow
        "except": "#FFF9C4",  # Light yellow
        "subquery_union": "#FFE0B2",  # Lighter orange
        "pivot": "#B2EBF2",  # Light cyan
        "unpivot": "#B2DFDB",  # Light teal
        "subquery_pivot_source": "#FFCCBC",  # Light orange
    }

    # Filter nodes by layer if specified
    if layers_to_show:
        nodes_to_show = [n for n in lineage_graph.nodes.values() if n.layer in layers_to_show]
    else:
        nodes_to_show = list(lineage_graph.nodes.values())

    # Limit number of nodes
    if len(nodes_to_show) > max_nodes:
        nodes_to_show = nodes_to_show[:max_nodes]

    # Create node IDs set for edge filtering
    node_ids = {n.full_name for n in nodes_to_show}

    # Create mapping from original full_name to sanitized Graphviz ID
    node_id_map = {n.full_name: _sanitize_graphviz_id(n.full_name) for n in nodes_to_show}

    # Group nodes by unit_id (QueryUnit)
    unit_subgraphs = {}

    for node in nodes_to_show:
        unit_id = node.unit_id

        # Handle external tables (unit_id = None)
        if unit_id is None:
            unit_key = f"external_{node.table_name}"
            # Sanitize cluster name for Graphviz
            safe_cluster_name = _sanitize_graphviz_id(unit_key)
            if unit_key not in unit_subgraphs:
                unit_subgraphs[unit_key] = graphviz.Digraph(name=f"cluster_{safe_cluster_name}")
                unit_subgraphs[unit_key].attr(
                    label=f"📊 {node.table_name}",
                    style="filled",
                    color="#B3E5FC",  # Light blue for external tables
                    fontsize="12",
                    fontname="Arial Bold",
                )
        else:
            # Get QueryUnit for metadata
            unit = query_graph.units.get(unit_id)
            # Sanitize cluster name for Graphviz
            safe_cluster_name = _sanitize_graphviz_id(unit_id)
            if unit_id not in unit_subgraphs:
                unit_subgraphs[unit_id] = graphviz.Digraph(name=f"cluster_{safe_cluster_name}")

                # Get unit metadata
                unit_type = unit.unit_type.value if unit else "unknown"
                unit_name = unit.name if unit else unit_id

                # Icon for unit type
                icons = {
                    "main_query": "🎯",
                    "cte": "📦",
                    "subquery_from": "🔸",
                    "subquery_select": "🔹",
                    "subquery_where": "🔶",
                    "subquery_having": "🔷",
                    "union": "🔀",
                    "intersect": "∩",
                    "except": "−",
                    "subquery_union": "🔹",
                    "pivot": "↻",
                    "unpivot": "↺",
                    "subquery_pivot_source": "🔸",
                }
                icon = icons.get(unit_type, "❓")

                # Color for unit type
                color = unit_type_colors.get(unit_type, "#E0E0E0")

                unit_subgraphs[unit_id].attr(
                    label=f"{icon} {unit_name}",
                    style="filled",
                    color=color,
                    fontsize="12",
                    fontname="Arial Bold",
                )

        # Determine which subgraph to add to
        subgraph_key = unit_key if unit_id is None else unit_id

        # Node appearance
        shape = "box"  # Always use box shape
        node_color = "#FFFFFF"  # White background for individual nodes

        # Label with column info
        label = node.column_name  # Will be "*" for star nodes, no special icon needed

        # Use sanitized node ID for Graphviz
        safe_node_id = node_id_map[node.full_name]
        unit_subgraphs[subgraph_key].node(
            safe_node_id,
            label=label,
            shape=shape,
            style="filled",
            fillcolor=node_color,
            tooltip=f"{node.full_name}\\n{node.node_type}",
        )

    # Add subgraphs to main graph
    for subgraph in unit_subgraphs.values():
        dot.subgraph(subgraph)

    # Add edges (only between shown nodes)
    for edge in lineage_graph.edges:
        if edge.from_node.full_name in node_ids and edge.to_node.full_name in node_ids:
            # Edge label shows transformation type
            label = edge.transformation

            # Color code by transformation type
            edge_colors = {
                "direct_column": "black",
                "aggregate": "red",
                "star_passthrough": "blue",
                "expression": "purple",
            }
            color = edge_colors.get(edge.transformation, "gray")

            # Use sanitized node IDs for edges
            safe_from = node_id_map[edge.from_node.full_name]
            safe_to = node_id_map[edge.to_node.full_name]
            dot.edge(
                safe_from,
                safe_to,
                label=label,
                color=color,
                fontsize="8",
            )

    return dot


def visualize_column_lineage_simple(
    lineage_graph: ColumnLineageGraph, max_nodes: int = 50
) -> graphviz.Digraph:
    """
    Simplified column lineage visualization without layer clustering.
    Better for smaller graphs.

    Args:
        lineage_graph: The column lineage graph from builder
        max_nodes: Maximum nodes to display

    Returns:
        graphviz.Digraph object ready to render
    """
    dot = graphviz.Digraph(comment="Column Lineage (Simple)")
    dot.attr(rankdir="LR")
    dot.attr("node", fontname="Arial", fontsize="10", shape="box")

    # Get nodes (limit count)
    nodes = list(lineage_graph.nodes.values())[:max_nodes]
    node_ids = {n.full_name for n in nodes}

    # Create mapping from original full_name to sanitized Graphviz ID
    node_id_map = {n.full_name: _sanitize_graphviz_id(n.full_name) for n in nodes}

    # Color by layer
    layer_colors = {
        "input": "#90CAF9",
        "cte": "#B39DDB",
        "subquery": "#FFAB91",
        "output": "#A5D6A7",
    }

    # Add nodes
    for node in nodes:
        color = layer_colors.get(node.layer, "#BDBDBD")
        label = f"{node.column_name}\\n[{node.layer}]"

        # Use sanitized node ID for Graphviz
        safe_node_id = node_id_map[node.full_name]
        dot.node(safe_node_id, label=label, style="filled", fillcolor=color, tooltip=node.full_name)

    # Add edges
    for edge in lineage_graph.edges:
        if edge.from_node.full_name in node_ids and edge.to_node.full_name in node_ids:
            # Use sanitized node IDs for edges
            safe_from = node_id_map[edge.from_node.full_name]
            safe_to = node_id_map[edge.to_node.full_name]
            dot.edge(
                safe_from,
                safe_to,
                label=edge.transformation,
                fontsize="8",
            )

    return dot


def visualize_column_path(
    lineage_graph: ColumnLineageGraph, target_column: str
) -> graphviz.Digraph:
    """
    Visualize the lineage path for a specific column.
    Shows only nodes and edges involved in this column's lineage.

    Args:
        lineage_graph: The column lineage graph
        target_column: Full name of the column to trace (e.g., "output.total_sales")

    Returns:
        graphviz.Digraph showing only the relevant path
    """
    dot = graphviz.Digraph(comment=f"Lineage Path: {target_column}")
    dot.attr(rankdir="LR")
    dot.attr("node", fontname="Arial", fontsize="10")

    # Find target node
    if target_column not in lineage_graph.nodes:
        # Empty graph with error message
        dot.node("error", f"Column not found: {target_column}", shape="box", color="red")
        return dot

    target_node = lineage_graph.nodes[target_column]

    # Traverse backward to find all dependencies
    visited = set()
    to_visit = [target_node]
    relevant_nodes = []

    while to_visit:
        node = to_visit.pop(0)
        if node.full_name in visited:
            continue

        visited.add(node.full_name)
        relevant_nodes.append(node)

        # Get incoming edges
        incoming = lineage_graph.get_edges_to(node)
        for edge in incoming:
            if edge.from_node.full_name not in visited:
                to_visit.append(edge.from_node)

    # Create mapping from original full_name to sanitized Graphviz ID
    node_id_map = {n.full_name: _sanitize_graphviz_id(n.full_name) for n in relevant_nodes}

    # Add nodes
    for node in relevant_nodes:
        # Highlight target node
        if node.full_name == target_column:
            color = "#4CAF50"
            style = "filled,bold"
        elif node.layer == "input":
            color = "#90CAF9"
            style = "filled"
        else:
            color = "#E0E0E0"
            style = "filled"

        label = f"{node.column_name}\\n[{node.layer}]"

        # Use sanitized node ID for Graphviz
        safe_node_id = node_id_map[node.full_name]
        dot.node(
            safe_node_id,
            label=label,
            style=style,
            fillcolor=color,
            shape="box",
            tooltip=node.full_name,
        )

    # Add edges (only between relevant nodes)
    relevant_node_ids = {n.full_name for n in relevant_nodes}
    for edge in lineage_graph.edges:
        if (
            edge.from_node.full_name in relevant_node_ids
            and edge.to_node.full_name in relevant_node_ids
        ):
            # Use sanitized node IDs for edges
            safe_from = node_id_map[edge.from_node.full_name]
            safe_to = node_id_map[edge.to_node.full_name]
            dot.edge(
                safe_from,
                safe_to,
                label=edge.transformation,
                fontsize="8",
            )

    return dot
