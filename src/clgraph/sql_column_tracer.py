"""
SQLColumnTracer - High-level wrapper for column lineage analysis.

Provides backward compatibility with existing code while using
RecursiveLineageBuilder internally.

Extracted from lineage_builder.py to improve module organization.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot

from .lineage_utils import BackwardLineageResult
from .models import ColumnLineageGraph, QueryUnitGraph


class SQLColumnTracer:
    """
    High-level wrapper that provides backward compatibility with existing code.
    Uses RecursiveLineageBuilder internally.
    """

    def __init__(
        self,
        sql_query: str,
        external_table_columns: Optional[Dict[str, List[str]]] = None,
        dialect: str = "bigquery",
    ):
        self.sql_query = sql_query
        self.external_table_columns = external_table_columns or {}
        self.dialect = dialect
        self.parsed = sqlglot.parse_one(sql_query, read=dialect)

        # Import here to avoid circular import
        from .lineage_builder import RecursiveLineageBuilder

        # Build lineage
        self.builder = RecursiveLineageBuilder(sql_query, external_table_columns, dialect=dialect)
        self.lineage_graph = None
        self._select_columns_cache = None

    def get_column_names(self) -> List[str]:
        """Get list of output column names"""
        # Build graph if not already built
        if self.lineage_graph is None:
            self.lineage_graph = self.builder.build()

        # Get output nodes
        output_nodes = self.lineage_graph.get_output_nodes()
        return [node.column_name for node in output_nodes]

    def build_column_lineage_graph(self) -> ColumnLineageGraph:
        """Build and return the complete lineage graph"""
        if self.lineage_graph is None:
            self.lineage_graph = self.builder.build()
        return self.lineage_graph

    def get_forward_lineage(self, input_columns: List[str]) -> Dict[str, Any]:
        """
        Get forward lineage (impact analysis) for given input columns.

        Args:
            input_columns: List of input column names (e.g., ["users.id", "orders.total"])

        Returns:
            Dict with:
                - impacted_outputs: List of output column names affected
                - impacted_ctes: List of CTE names in the path
                - paths: List of path dicts with input, intermediate, output, transformations
        """
        if self.lineage_graph is None:
            self.lineage_graph = self.builder.build()

        result = {"impacted_outputs": [], "impacted_ctes": [], "paths": []}

        impacted_outputs = set()
        impacted_ctes = set()

        for input_col in input_columns:
            # Find matching input nodes
            start_nodes = []
            for node in self.lineage_graph.nodes.values():
                # Match by full_name or table.column pattern
                if node.full_name == input_col:
                    start_nodes.append(node)
                elif node.layer == "input":
                    # Try matching table.column pattern
                    if f"{node.table_name}.{node.column_name}" == input_col:
                        start_nodes.append(node)
                    # Try matching just column name for star patterns
                    elif input_col.endswith(".*") and node.is_star:
                        if node.table_name == input_col.replace(".*", ""):
                            start_nodes.append(node)

            # BFS forward from each start node
            for start_node in start_nodes:
                visited = set()
                queue = deque([(start_node, [start_node.full_name], [])])

                while queue:
                    current, path, transformations = queue.popleft()

                    if current.full_name in visited:
                        continue
                    visited.add(current.full_name)

                    # Track CTEs
                    if current.layer == "cte" or current.layer.startswith("cte_"):
                        cte_name = current.table_name
                        impacted_ctes.add(cte_name)

                    # Get outgoing edges
                    outgoing = self.lineage_graph.get_edges_from(current)

                    if not outgoing:
                        # Reached end - check if output
                        if current.layer == "output":
                            impacted_outputs.add(current.column_name)
                            result["paths"].append(
                                {
                                    "input": input_col,
                                    "intermediate": path[1:-1] if len(path) > 2 else [],
                                    "output": current.column_name,
                                    "transformations": list(set(transformations)),
                                }
                            )
                    else:
                        for edge in outgoing:
                            new_path = path + [edge.to_node.full_name]
                            new_transforms = transformations + [edge.transformation]
                            queue.append((edge.to_node, new_path, new_transforms))

        result["impacted_outputs"] = list(impacted_outputs)
        result["impacted_ctes"] = list(impacted_ctes)

        return result

    def get_backward_lineage(self, output_columns: List[str]) -> BackwardLineageResult:
        """
        Get backward lineage (source tracing) for given output columns.

        Args:
            output_columns: List of output column names (e.g., ["id", "total_amount"])

        Returns:
            Dict with:
                - required_inputs: Dict[table_name, List[column_names]]
                - required_ctes: List of CTE names in the path
                - paths: List of path dicts
        """
        if self.lineage_graph is None:
            self.lineage_graph = self.builder.build()

        result: BackwardLineageResult = {"required_inputs": {}, "required_ctes": [], "paths": []}

        required_ctes = set()

        for output_col in output_columns:
            # Find matching output nodes
            start_nodes = []
            for node in self.lineage_graph.nodes.values():
                if node.layer == "output":
                    if node.column_name == output_col or node.full_name == output_col:
                        start_nodes.append(node)

            # BFS backward from each start node
            for start_node in start_nodes:
                visited = set()
                queue = deque([(start_node, [start_node.full_name], [])])

                while queue:
                    current, path, transformations = queue.popleft()

                    if current.full_name in visited:
                        continue
                    visited.add(current.full_name)

                    # Track CTEs
                    if current.layer == "cte" or current.layer.startswith("cte_"):
                        cte_name = current.table_name
                        required_ctes.add(cte_name)

                    # Get incoming edges
                    incoming = self.lineage_graph.get_edges_to(current)

                    if not incoming:
                        # Reached source - should be input layer
                        if current.layer == "input" and current.table_name:
                            table = current.table_name
                            col = current.column_name

                            if table not in result["required_inputs"]:
                                result["required_inputs"][table] = []
                            if col not in result["required_inputs"][table]:
                                result["required_inputs"][table].append(col)

                            result["paths"].append(
                                {
                                    "output": output_col,
                                    "intermediate": list(reversed(path[1:-1]))
                                    if len(path) > 2
                                    else [],
                                    "input": f"{table}.{col}",
                                    "transformations": list(set(transformations)),
                                }
                            )
                    else:
                        for edge in incoming:
                            new_path = path + [edge.from_node.full_name]
                            new_transforms = transformations + [edge.transformation]
                            queue.append((edge.from_node, new_path, new_transforms))

        result["required_ctes"] = list(required_ctes)

        return result

    def get_query_structure(self) -> QueryUnitGraph:
        """Get the query structure graph"""
        return self.builder.unit_graph

    def trace_column_dependencies(self, column_name: str) -> Set[Tuple[int, int]]:
        """
        Trace column dependencies and return SQL positions (for backward compatibility).

        NOTE: This is a stub implementation that returns empty set.
        The new design focuses on graph-based lineage, not position-based highlighting.
        """
        # For now, return empty set - position tracking is not part of the new design
        return set()

    def get_highlighted_sql(self, column_name: str) -> str:
        """
        Return SQL with highlighted sections (for backward compatibility).

        NOTE: Returns un-highlighted SQL for now.
        Position-based highlighting is not part of the new recursive design.
        """
        return self.sql_query

    def get_syntax_tree(self, column_name: Optional[str] = None) -> str:
        """
        Return a string representation of the syntax tree.
        """
        if self.lineage_graph is None:
            self.lineage_graph = self.builder.build()

        # Build a simple tree view of the query structure
        result = ["Query Structure:", ""]

        for unit in self.builder.unit_graph.get_topological_order():
            indent = "  " * unit.depth
            deps = unit.depends_on_units + unit.depends_on_tables
            deps_str = f" <- {', '.join(deps)}" if deps else ""
            result.append(f"{indent}{unit.unit_id} ({unit.unit_type.value}){deps_str}")

        result.append("")
        result.append("Column Lineage Graph:")
        result.append(f"  Nodes: {len(self.lineage_graph.nodes)}")
        result.append(f"  Edges: {len(self.lineage_graph.edges)}")

        # Show nodes by layer
        for layer in ["input", "cte", "subquery", "output"]:
            layer_nodes = [n for n in self.lineage_graph.nodes.values() if n.layer == layer]
            if layer_nodes:
                result.append(f"\n  {layer.upper()} Layer ({len(layer_nodes)} nodes):")
                for node in sorted(layer_nodes, key=lambda n: n.full_name)[:10]:  # Show first 10
                    star_indicator = " *" if node.is_star else ""
                    result.append(f"    - {node.full_name}{star_indicator}")
                if len(layer_nodes) > 10:
                    result.append(f"    ... and {len(layer_nodes) - 10} more")

        return "\n".join(result)

    @property
    def select_columns(self) -> List[Dict]:
        """
        Get select columns info for backward compatibility with app.
        Returns list of dicts with 'alias', 'sql', 'index' keys.
        """
        if self._select_columns_cache is None:
            if self.lineage_graph is None:
                self.lineage_graph = self.builder.build()

            # Get output nodes and format them
            output_nodes = self.lineage_graph.get_output_nodes()
            self._select_columns_cache = [
                {"alias": node.column_name, "sql": node.expression, "index": i}
                for i, node in enumerate(output_nodes)
            ]

        return self._select_columns_cache


__all__ = ["SQLColumnTracer"]
