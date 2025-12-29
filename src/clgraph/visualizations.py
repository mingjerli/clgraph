"""
Pure visualization functions for SQL lineage graphs.

These functions translate graph structures into Graphviz DOT format.
No business logic - just presentation layer.
"""

from typing import TYPE_CHECKING, List, Tuple, Union

import graphviz

from clgraph import ColumnLineageGraph, QueryUnitGraph

if TYPE_CHECKING:
    from clgraph import ColumnEdge, ColumnNode, PipelineLineageGraph
    from clgraph.pipeline import Pipeline
    from clgraph.table import TableDependencyGraph


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
    max_nodes: int = 100,
) -> graphviz.Digraph:
    """
    Create Graphviz visualization of ColumnLineageGraph clustered by QueryUnit.

    Visualizes column-level lineage within a single query, grouping columns by
    their containing query unit (CTEs, subqueries, main query) and table.

    Args:
        lineage_graph: The column lineage graph from builder
        max_nodes: Maximum nodes to display (to prevent cluttering)

    Returns:
        graphviz.Digraph object ready to render

    Note:
        For simplified views (input → output only), call lineage_graph.to_simplified()
        before passing to this function.
    """
    dot = graphviz.Digraph(comment="Query Column Lineage")
    dot.attr(rankdir="LR")
    dot.attr("node", fontname="Arial", fontsize="10")
    dot.attr("edge", fontsize="8")

    # Create safe node IDs to avoid Graphviz node:port interpretation
    node_id_map = {}
    for node in lineage_graph.nodes.values():
        safe_id = _sanitize_graphviz_id(node.full_name)
        node_id_map[node.full_name] = safe_id

    # Group columns by unit_id and table_name
    unit_groups: dict = {}

    for node in lineage_graph.nodes.values():
        unit_id = node.unit_id or "external"
        table_name = node.table_name or "unknown"

        # Create a unique key for each QueryUnit
        key = f"{unit_id}:{table_name}"

        if key not in unit_groups:
            unit_groups[key] = {
                "unit_id": unit_id,
                "table_name": table_name,
                "layer": node.layer,
                "columns": [],
            }
        unit_groups[key]["columns"].append(node)

    # Create subgraphs for each QueryUnit
    for key, group in sorted(unit_groups.items()):
        unit_id = group["unit_id"]
        table_name = group["table_name"]
        layer = group["layer"]
        columns = group["columns"]

        # Limit columns if needed
        if len(columns) > max_nodes:
            columns = columns[:max_nodes]

        # Create cluster name
        cluster_name = f"cluster_{_sanitize_graphviz_id(key)}"

        with dot.subgraph(name=cluster_name) as sub:
            # Determine cluster label and color based on unit type
            if unit_id == "external" or layer == "input":
                # External/input tables
                cluster_label = f"📊 {table_name}\\n(input)"
                cluster_color = "#E3F2FD"  # Light blue
            elif unit_id == "main":
                # Main query output
                cluster_label = f"🎯 main query\\n{table_name}"
                cluster_color = "#E8F5E9"  # Light green
            elif unit_id.startswith("cte:"):
                # CTE
                cte_name = unit_id.split(":", 1)[1] if ":" in unit_id else unit_id
                cluster_label = f"📦 CTE: {cte_name}"
                cluster_color = "#EDE7F6"  # Light purple
            elif unit_id.startswith("subq:"):
                # Subquery
                subq_name = unit_id.split(":", 1)[1] if ":" in unit_id else unit_id
                cluster_label = f"🔸 Subquery: {subq_name}"
                cluster_color = "#FFF3E0"  # Light orange
            else:
                # Other unit types
                cluster_label = f"❓ {unit_id}\\n{table_name}"
                cluster_color = "#F5F5F5"  # Light gray

            sub.attr(
                label=cluster_label,
                style="filled",
                color=cluster_color,
                fontsize="12",
                fontname="Arial Bold",
            )

            # Add column nodes within the cluster
            for col in columns:
                # Determine node color based on node type
                if col.node_type == "source":
                    fillcolor = "#90CAF9"
                    fontcolor = "#0D47A1"
                elif col.node_type in ["aggregate", "computed", "derived"]:
                    fillcolor = "#FFB74D"
                    fontcolor = "#E65100"
                elif col.is_star:
                    fillcolor = "#FFF59D"
                    fontcolor = "#F57F17"
                else:
                    fillcolor = "#A5D6A7"
                    fontcolor = "#1B5E20"

                # Create label
                label = col.column_name
                if col.is_star:
                    label = "*"

                # Use safe node ID
                safe_node_id = node_id_map[col.full_name]
                expr_preview = col.expression[:50] if col.expression else ""
                sub.node(
                    safe_node_id,
                    label=label,
                    shape="box",
                    style="filled,rounded",
                    fillcolor=fillcolor,
                    fontcolor=fontcolor,
                    tooltip=f"{col.full_name}\\n{col.node_type}\\n{expr_preview}",
                )

    # Add edges using safe node IDs
    edge_count = 0
    max_edges = 200

    for edge in lineage_graph.edges:
        if edge_count >= max_edges:
            break

        # Only show label for non-direct transformations
        label = ""
        if edge.transformation and edge.transformation not in ["direct", "direct_column"]:
            label = edge.transformation[:15]  # Truncate long labels

        # Use safe node IDs
        safe_from = node_id_map[edge.from_node.full_name]
        safe_to = node_id_map[edge.to_node.full_name]

        dot.edge(
            safe_from,
            safe_to,
            label=label,
            color="#666666",
            tooltip=f"{edge.transformation or 'direct'}",
        )
        edge_count += 1

    if len(lineage_graph.nodes) > max_nodes:
        dot.node(
            "note",
            label=f"Showing {max_nodes} of {len(lineage_graph.nodes)} columns",
            shape="note",
            fillcolor="#FFFFCC",
            fontcolor="#333333",
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
    graph: Union[ColumnLineageGraph, "PipelineLineageGraph"],
    target_column: str,
) -> graphviz.Digraph:
    """
    Visualize the lineage path for a specific column.
    Shows only nodes and edges involved in this column's lineage.

    Works with both single-query ColumnLineageGraph and multi-query PipelineLineageGraph.

    Args:
        graph: The column lineage graph (ColumnLineageGraph or PipelineLineageGraph)
        target_column: Full name of the column to trace (e.g., "output.total_sales")

    Returns:
        graphviz.Digraph showing only the relevant path
    """
    dot = graphviz.Digraph(comment=f"Lineage Path: {target_column}")
    dot.attr(rankdir="LR")
    dot.attr("node", fontname="Arial", fontsize="10")

    # Determine graph type and get appropriate accessors
    # PipelineLineageGraph uses .columns, ColumnLineageGraph uses .nodes
    if hasattr(graph, "columns"):
        # PipelineLineageGraph
        nodes_dict = graph.columns
    else:
        # ColumnLineageGraph
        nodes_dict = graph.nodes

    # Find target node
    if target_column not in nodes_dict:
        # Empty graph with error message
        dot.node("error", f"Column not found: {target_column}", shape="box", color="red")
        return dot

    target_node = nodes_dict[target_column]

    # Traverse backward to find all dependencies using BFS
    visited = set()
    to_visit = [target_node]
    relevant_nodes = []

    while to_visit:
        node = to_visit.pop(0)
        if node.full_name in visited:
            continue

        visited.add(node.full_name)
        relevant_nodes.append(node)

        # Get incoming edges - different methods for different graph types
        if hasattr(graph, "get_edges_to"):
            # ColumnLineageGraph has get_edges_to method
            incoming = graph.get_edges_to(node)
        else:
            # PipelineLineageGraph - filter edges manually
            incoming = [e for e in graph.edges if e.to_node.full_name == node.full_name]

        for edge in incoming:
            if edge.from_node.full_name not in visited:
                to_visit.append(edge.from_node)

    # Create mapping from original full_name to sanitized Graphviz ID
    node_id_map = {n.full_name: _sanitize_graphviz_id(n.full_name) for n in relevant_nodes}

    # Add nodes with enhanced styling
    for node in relevant_nodes:
        # Highlight target node
        if node.full_name == target_column:
            fillcolor = "#4CAF50"
            fontcolor = "white"
            style = "filled,bold"
        elif node.node_type == "source":
            # Source columns - light blue
            fillcolor = "#90CAF9"
            fontcolor = "#0D47A1"
            style = "filled"
        elif node.node_type in ["derived", "computed", "aggregate"]:
            # Intermediate columns - orange
            fillcolor = "#FFB74D"
            fontcolor = "#E65100"
            style = "filled"
        else:
            fillcolor = "#A5D6A7"
            fontcolor = "#1B5E20"
            style = "filled"

        label = f"{node.column_name}\\n({node.table_name})"

        # Use sanitized node ID for Graphviz
        safe_node_id = node_id_map[node.full_name]
        dot.node(
            safe_node_id,
            label=label,
            style=style,
            fillcolor=fillcolor,
            fontcolor=fontcolor,
            shape="ellipse",
            tooltip=f"{node.full_name} ({node.node_type})",
        )

    # Add edges (only between relevant nodes)
    relevant_node_ids = {n.full_name for n in relevant_nodes}
    for edge in graph.edges:
        if (
            edge.from_node.full_name in relevant_node_ids
            and edge.to_node.full_name in relevant_node_ids
        ):
            # Only show label for non-direct transformations
            label = ""
            if edge.transformation and edge.transformation not in [
                "direct",
                "direct_column",
            ]:
                label = edge.transformation

            # Use sanitized node IDs for edges
            safe_from = node_id_map[edge.from_node.full_name]
            safe_to = node_id_map[edge.to_node.full_name]
            dot.edge(
                safe_from,
                safe_to,
                label=label,
                tooltip=f"{edge.transformation or 'direct'}",
            )

    return dot


def visualize_table_dependencies(
    table_graph: "TableDependencyGraph",
) -> graphviz.Digraph:
    """
    Create Graphviz visualization of table-level dependencies.

    Shows tables as nodes with edges representing data flow between queries.
    Color-coded by table type: source (blue), final (green), intermediate (orange).

    Args:
        table_graph: The table dependency graph from Pipeline.table_graph

    Returns:
        graphviz.Digraph object ready to render
    """
    dot = graphviz.Digraph(comment="Table Dependencies")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Arial", fontsize="12")
    dot.attr("edge", fontsize="10", color="#555555")

    # Get source and final tables
    source_tables = {t.table_name for t in table_graph.get_source_tables()}
    final_tables = {t.table_name for t in table_graph.get_final_tables()}

    # Add nodes with colors based on type
    for table_name in table_graph.tables.keys():
        if table_name in source_tables:
            # Source tables - blue cylinder
            dot.node(
                _sanitize_graphviz_id(table_name),
                label=f"📊 {table_name}\\n(source)",
                fillcolor="#90CAF9",
                fontcolor="#0D47A1",
                shape="cylinder",
            )
        elif table_name in final_tables:
            # Final tables - green
            dot.node(
                _sanitize_graphviz_id(table_name),
                label=f"🎯 {table_name}\\n(final)",
                fillcolor="#4CAF50",
                fontcolor="white",
            )
        else:
            # Intermediate tables - orange
            dot.node(
                _sanitize_graphviz_id(table_name),
                label=f"🔸 {table_name}\\n(intermediate)",
                fillcolor="#FFB74D",
                fontcolor="#E65100",
            )

    # Add edges
    for query_id, query in table_graph.queries.items():
        if query.destination_table:
            for source_table in query.source_tables:
                if source_table in table_graph.tables:
                    dot.edge(
                        _sanitize_graphviz_id(source_table),
                        _sanitize_graphviz_id(query.destination_table),
                        label=query.operation.value.split()[0],
                        tooltip=f"{query_id}: {query.operation.value}",
                    )

    return dot


def visualize_table_dependencies_with_levels(
    table_graph: "TableDependencyGraph",
    pipeline: "Pipeline",
) -> graphviz.Digraph:
    """
    Create Graphviz visualization of table dependencies with execution level annotations.

    Shows tables with their execution level, useful for understanding which tables
    can be processed in parallel.

    Args:
        table_graph: The table dependency graph from Pipeline.table_graph
        pipeline: The Pipeline object to get execution levels from

    Returns:
        graphviz.Digraph object ready to render
    """
    dot = graphviz.Digraph(comment="Table Dependencies with Execution Levels")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Arial", fontsize="12")
    dot.attr("edge", fontsize="10", color="#555555")

    # Get execution levels to show parallelizable queries
    try:
        levels = pipeline._get_execution_levels()
        query_to_level = {}
        for level_num, level_queries in enumerate(levels, 1):
            for qid in level_queries:
                query_to_level[qid] = level_num
    except Exception:
        query_to_level = {}

    # Get source and final tables
    source_tables = {t.table_name for t in table_graph.get_source_tables()}
    final_tables = {t.table_name for t in table_graph.get_final_tables()}

    # Add nodes with colors based on type
    for table_name in table_graph.tables.keys():
        table_node = table_graph.tables[table_name]

        # Determine label with execution level if available
        if table_node.created_by and table_node.created_by in query_to_level:
            level = query_to_level[table_node.created_by]
            level_info = f"\\n[Level {level}]"
        else:
            level_info = ""

        if table_name in source_tables:
            # Source tables - blue cylinder
            dot.node(
                _sanitize_graphviz_id(table_name),
                label=f"📊 {table_name}\\n(source){level_info}",
                fillcolor="#90CAF9",
                fontcolor="#0D47A1",
                shape="cylinder",
            )
        elif table_name in final_tables:
            # Final tables - green
            dot.node(
                _sanitize_graphviz_id(table_name),
                label=f"🎯 {table_name}\\n(final){level_info}",
                fillcolor="#4CAF50",
                fontcolor="white",
            )
        else:
            # Intermediate tables - orange
            dot.node(
                _sanitize_graphviz_id(table_name),
                label=f"🔸 {table_name}\\n(intermediate){level_info}",
                fillcolor="#FFB74D",
                fontcolor="#E65100",
            )

    # Add edges
    for query_id, query in table_graph.queries.items():
        if query.destination_table:
            for source_table in query.source_tables:
                if source_table in table_graph.tables:
                    dot.edge(
                        _sanitize_graphviz_id(source_table),
                        _sanitize_graphviz_id(query.destination_table),
                        label=query.operation.value.split()[0],
                        tooltip=f"{query_id}: {query.operation.value}",
                    )

    return dot


def visualize_pipeline_lineage(
    graph: "PipelineLineageGraph",
    max_columns: int = 300,
    return_debug_info: bool = False,
) -> Union[graphviz.Digraph, Tuple[graphviz.Digraph, dict]]:
    """
    Create Graphviz visualization of column-level lineage across a multi-query pipeline.

    Groups columns by query_id and unit_id, showing cross-query edges with
    distinctive styling. Intelligently limits the number of displayed columns,
    prioritizing output and input layers.

    Args:
        graph: The pipeline column lineage graph from Pipeline.column_graph
        max_columns: Maximum number of columns to display (default: 300)
        return_debug_info: If True, returns (dot, debug_info) tuple with metadata

    Returns:
        If return_debug_info is False: graphviz.Digraph object
        If return_debug_info is True: Tuple of (graphviz.Digraph, dict) where
            dict contains: columns_displayed, edges_displayed, nodes_in_graph,
            nodes_by_layer, edges_added, node_id_map, edges_skipped
    """
    dot = graphviz.Digraph(comment="Pipeline Column Lineage")
    dot.attr(rankdir="LR")
    dot.attr("node", fontname="Arial", fontsize="10")
    dot.attr("edge", fontsize="8")

    # Initialize debug info
    debug_info: dict = {
        "columns_displayed": 0,
        "edges_displayed": 0,
        "nodes_in_graph": set(),
        "nodes_by_layer": {},
        "edges_added": [],
        "node_id_map": {},
        "edges_skipped": 0,
    }

    # Get all columns
    all_columns = list(graph.columns.values())

    # If too many columns, prioritize: output layer > input layer > others
    if len(all_columns) > max_columns:
        output_cols = [c for c in all_columns if c.layer == "output"]
        input_cols = [c for c in all_columns if c.layer == "input"]
        other_cols = [c for c in all_columns if c.layer not in ("output", "input")]

        max_output = min(len(output_cols), max_columns // 3)
        max_input = min(len(input_cols), max_columns // 3)
        max_other = max_columns - max_output - max_input

        columns_to_show = output_cols[:max_output] + input_cols[:max_input] + other_cols[:max_other]
    else:
        columns_to_show = all_columns

    # Group columns by query_id and unit_id
    query_unit_groups: dict = {}

    for col in columns_to_show:
        query_id = col.query_id or "source"
        unit_id = col.unit_id
        table_name = col.table_name or "unknown"

        key = f"{query_id}:{unit_id or 'external'}:{table_name}"

        if key not in query_unit_groups:
            query_unit_groups[key] = {
                "query_id": query_id,
                "unit_id": unit_id,
                "table_name": table_name,
                "columns": [],
            }
        query_unit_groups[key]["columns"].append(col)

    # Create safe node ID mapping
    node_id_map = {}
    for col in columns_to_show:
        safe_id = _sanitize_graphviz_id(col.full_name)
        node_id_map[col.full_name] = safe_id
    debug_info["node_id_map"] = node_id_map

    # Create subgraphs for each query-unit group
    for key, group in sorted(query_unit_groups.items()):
        query_id = group["query_id"]
        unit_id = group["unit_id"]
        table_name = group["table_name"]
        cols = group["columns"]

        cluster_name = f"cluster_{_sanitize_graphviz_id(key)}"

        with dot.subgraph(name=cluster_name) as sub:
            # Determine cluster label and color based on unit type
            if unit_id is None:
                cluster_label = f"📊 {table_name}"
                cluster_color = "#B3E5FC"  # Light blue
            elif unit_id == "main":
                cluster_label = f"🎯 [{query_id}] {table_name}"
                cluster_color = "#C8E6C9"  # Light green
            elif unit_id.startswith("cte:"):
                cte_name = unit_id.split(":", 1)[1]
                cluster_label = f"📦 {cte_name}"
                cluster_color = "#C5CAE9"  # Light purple
            elif unit_id.startswith("subq:"):
                cluster_label = f"🔸 {unit_id}"
                cluster_color = "#FFCCBC"  # Light orange
            else:
                cluster_label = f"❓ {unit_id}"
                cluster_color = "#E0E0E0"  # Gray

            sub.attr(
                label=cluster_label,
                style="filled",
                color=cluster_color,
                fontsize="12",
                fontname="Arial Bold",
            )

            for col in cols:
                safe_node_id = node_id_map[col.full_name]
                sub.node(
                    safe_node_id,
                    label=col.column_name,
                    shape="box",
                    style="filled",
                    fillcolor="#FFFFFF",
                    tooltip=f"{col.full_name}\\n{col.node_type}",
                )
                debug_info["nodes_in_graph"].add(col.full_name)
                debug_info["nodes_by_layer"][col.full_name] = col.layer

    debug_info["columns_displayed"] = len(columns_to_show)

    # Build set of nodes for edge validation
    nodes_in_graph = {col.full_name for col in columns_to_show}

    edge_count = 0
    max_edges = 500
    skipped_edges = 0

    for edge in graph.edges:
        if edge_count >= max_edges:
            break

        from_full_name = edge.from_node.full_name
        to_full_name = edge.to_node.full_name

        if from_full_name in nodes_in_graph and to_full_name in nodes_in_graph:
            label = ""
            if edge.transformation and edge.transformation not in [
                "direct",
                "direct_column",
            ]:
                label = edge.transformation[:20]

            edge_type_str = edge.edge_type or "within_query"
            safe_from = node_id_map[from_full_name]
            safe_to = node_id_map[to_full_name]

            if edge.edge_type == "cross_query":
                dot.edge(
                    safe_from,
                    safe_to,
                    label=label,
                    tooltip=f"{edge.transformation or 'cross_query'}",
                    color="#E65100",
                    penwidth="2.0",
                )
            else:
                dot.edge(
                    safe_from,
                    safe_to,
                    label=label,
                    tooltip=f"{edge.transformation or 'direct'}",
                    color="#666666",
                )
            edge_count += 1
            debug_info["edges_added"].append((from_full_name, to_full_name, edge_type_str))
        else:
            skipped_edges += 1

    debug_info["edges_displayed"] = edge_count
    debug_info["edges_skipped"] = skipped_edges

    # Add visual indicator if columns were limited
    if len(all_columns) > max_columns:
        dot.node(
            "limitation_note",
            label=f"⚠️ Displaying {len(columns_to_show)} of {len(all_columns)} columns\\n"
            f"({edge_count} edges shown)\\n"
            f"Prioritizing output/input layers\\n"
            f"Use Column Tracing for complete lineage",
            shape="note",
            fillcolor="#FFF9C4",
            fontcolor="#F57F17",
            fontsize="11",
            style="filled",
        )

    if return_debug_info:
        return dot, debug_info
    return dot


def visualize_lineage_path(
    nodes: List["ColumnNode"],
    edges: List["ColumnEdge"],
    is_backward: bool = True,
) -> graphviz.Digraph:
    """
    Create Graphviz visualization of a traced lineage path.

    Visualizes pre-traced lineage results from methods like:
    - pipeline.trace_column_backward_full()
    - pipeline.trace_column_forward_full()

    Groups nodes by query_id and layer for clear visual structure.

    Args:
        nodes: List of ColumnNode objects in the lineage path
        edges: List of ColumnEdge objects connecting the nodes
        is_backward: If True, labels the diagram as "Backward Lineage Path",
                    otherwise "Forward Lineage Path"

    Returns:
        graphviz.Digraph object ready to render
    """
    direction = "Backward" if is_backward else "Forward"
    dot = graphviz.Digraph(comment=f"{direction} Lineage Path")
    dot.attr(rankdir="LR")
    dot.attr("node", fontname="Arial", fontsize="10")
    dot.attr("edge", fontsize="8")

    # Create safe node IDs to avoid Graphviz node:port interpretation
    node_id_map = {}
    for node in nodes:
        safe_id = _sanitize_graphviz_id(node.full_name)
        node_id_map[node.full_name] = safe_id

    # Group nodes by query_id and layer for better layout
    query_groups: dict = {}
    for node in nodes:
        query_id = node.query_id or "source"
        layer = node.layer

        key = f"{query_id}:{layer}"
        if key not in query_groups:
            query_groups[key] = []
        query_groups[key].append(node)

    # Create subgraphs for each query-layer combination
    for key, group_nodes in sorted(query_groups.items()):
        query_id, layer = key.split(":", 1)

        cluster_name = f"cluster_{_sanitize_graphviz_id(key)}"

        with dot.subgraph(name=cluster_name) as sub:
            # Determine cluster label and color
            if layer == "input":
                cluster_label = f"📊 Input ({query_id})"
                cluster_color = "#E3F2FD"  # Very light blue
            elif layer == "cte":
                cluster_label = f"📦 CTE ({query_id})"
                cluster_color = "#EDE7F6"  # Very light purple
            elif layer == "subquery":
                cluster_label = f"🔸 Subquery ({query_id})"
                cluster_color = "#FFF3E0"  # Very light orange
            else:  # output
                cluster_label = f"🎯 Output ({query_id})"
                cluster_color = "#E8F5E9"  # Very light green

            sub.attr(
                label=cluster_label,
                style="filled",
                color=cluster_color,
                fontsize="11",
                fontname="Arial Bold",
            )

            # Add column nodes
            for node in group_nodes:
                # Determine node color based on type
                if node.node_type == "source":
                    fillcolor = "#90CAF9"
                    fontcolor = "#0D47A1"
                elif node.node_type in ["derived", "computed", "aggregate"]:
                    fillcolor = "#FFB74D"
                    fontcolor = "#E65100"
                else:
                    fillcolor = "#A5D6A7"
                    fontcolor = "#1B5E20"

                # Use safe node ID
                safe_node_id = node_id_map[node.full_name]
                expr_preview = node.expression[:50] if node.expression else ""
                sub.node(
                    safe_node_id,
                    label=f"{node.column_name}",
                    shape="box",
                    style="filled,rounded",
                    fillcolor=fillcolor,
                    fontcolor=fontcolor,
                    tooltip=f"{node.full_name}\\n{node.node_type}\\n{expr_preview}",
                )

    # Add edges using safe node IDs
    for edge in edges:
        # Only show label for non-direct transformations
        label = ""
        if edge.transformation and edge.transformation not in ["direct", "direct_column"]:
            label = edge.transformation[:20]  # Truncate long transformations

        # Use safe node IDs for both endpoints
        safe_from = node_id_map[edge.from_node.full_name]
        safe_to = node_id_map[edge.to_node.full_name]

        # Different edge style for cross-query edges
        if edge.edge_type == "cross_query":
            dot.edge(
                safe_from,
                safe_to,
                label=label,
                color="#E65100",
                penwidth="2.0",
                tooltip=f"Cross-query: {edge.transformation or 'direct'}",
            )
        else:
            dot.edge(
                safe_from,
                safe_to,
                label=label,
                color="#666666",
                tooltip=f"{edge.transformation or 'direct'}",
            )

    return dot
