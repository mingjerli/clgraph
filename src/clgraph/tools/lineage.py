"""
Lineage tracing tools.

Tools for tracing column lineage backward (to sources) and forward (to impacts).
"""

from typing import Dict

from .base import BaseTool, ParameterSpec, ParameterType, ToolResult


class TraceBackwardTool(BaseTool):
    """
    Trace a column backward to find its sources.

    Given a table and column, finds all source columns that
    contribute to it across the entire pipeline.

    Example:
        tool = TraceBackwardTool(pipeline)
        result = tool.run(table="analytics.revenue", column="total_amount")
        # Returns sources like raw.orders.amount, staging.orders.amount
    """

    name = "trace_backward"
    description = "Find the source columns that a column is derived from"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "table": ParameterSpec(
                name="table",
                type=ParameterType.STRING,
                description="The table name containing the column to trace",
                required=True,
            ),
            "column": ParameterSpec(
                name="column",
                type=ParameterType.STRING,
                description="The column name to trace backward",
                required=True,
            ),
            "include_intermediate": ParameterSpec(
                name="include_intermediate",
                type=ParameterType.BOOLEAN,
                description="Whether to include intermediate tables (not just ultimate sources)",
                required=False,
                default=False,
            ),
        }

    def run(self, table: str, column: str, include_intermediate: bool = False) -> ToolResult:
        # Validate table exists
        if table not in self.pipeline.table_graph.tables:
            return ToolResult.error_result(
                f"Table '{table}' not found. Available tables: {list(self.pipeline.table_graph.tables.keys())[:10]}"
            )

        # Check column exists
        col_node = self.pipeline.get_column(table, column)
        if col_node is None:
            available_cols = [c.column_name for c in self.pipeline.get_columns_by_table(table)]
            return ToolResult.error_result(
                f"Column '{column}' not found in table '{table}'. Available columns: {available_cols[:10]}"
            )

        if include_intermediate:
            # Get full path including intermediate nodes
            nodes, edges = self.pipeline.trace_column_backward_full(table, column)
            sources = [
                {
                    "table": n.table_name,
                    "column": n.column_name,
                    "description": n.description,
                    "is_source": n.layer == "input",
                    "query_id": n.query_id,
                }
                for n in nodes
            ]
        else:
            # Get only ultimate sources
            source_nodes = self.pipeline.trace_column_backward(table, column)
            sources = [
                {
                    "table": n.table_name,
                    "column": n.column_name,
                    "description": n.description,
                }
                for n in source_nodes
            ]

        if not sources:
            return ToolResult.success_result(
                data=[],
                message=f"Column {table}.{column} has no upstream sources (it may be a source column itself)",
            )

        # Build human-readable message
        source_names = [f"{s['table']}.{s['column']}" for s in sources[:5]]
        msg = f"Column {table}.{column} is derived from: {', '.join(source_names)}"
        if len(sources) > 5:
            msg += f" (+{len(sources) - 5} more)"

        return ToolResult.success_result(data=sources, message=msg)


class TraceForwardTool(BaseTool):
    """
    Trace a column forward to find its downstream impact.

    Given a table and column, finds all columns that depend on it
    across the entire pipeline.

    Example:
        tool = TraceForwardTool(pipeline)
        result = tool.run(table="raw.orders", column="amount")
        # Returns impacted columns like staging.orders.amount, analytics.revenue.total
    """

    name = "trace_forward"
    description = "Find all columns that depend on (are impacted by) a column"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "table": ParameterSpec(
                name="table",
                type=ParameterType.STRING,
                description="The table name containing the column to trace",
                required=True,
            ),
            "column": ParameterSpec(
                name="column",
                type=ParameterType.STRING,
                description="The column name to trace forward",
                required=True,
            ),
            "include_intermediate": ParameterSpec(
                name="include_intermediate",
                type=ParameterType.BOOLEAN,
                description="Whether to include intermediate tables (not just final outputs)",
                required=False,
                default=False,
            ),
        }

    def run(self, table: str, column: str, include_intermediate: bool = False) -> ToolResult:
        # Validate table exists
        if table not in self.pipeline.table_graph.tables:
            return ToolResult.error_result(
                f"Table '{table}' not found. Available tables: {list(self.pipeline.table_graph.tables.keys())[:10]}"
            )

        # Check column exists
        col_node = self.pipeline.get_column(table, column)
        if col_node is None:
            available_cols = [c.column_name for c in self.pipeline.get_columns_by_table(table)]
            return ToolResult.error_result(
                f"Column '{column}' not found in table '{table}'. Available columns: {available_cols[:10]}"
            )

        if include_intermediate:
            # Get full path including intermediate nodes
            nodes, edges = self.pipeline.trace_column_forward_full(table, column)
            impacts = [
                {
                    "table": n.table_name,
                    "column": n.column_name,
                    "description": n.description,
                    "is_final": n.layer == "output",
                    "query_id": n.query_id,
                }
                for n in nodes
            ]
        else:
            # Get only final downstream columns
            downstream_nodes = self.pipeline.trace_column_forward(table, column)
            impacts = [
                {
                    "table": n.table_name,
                    "column": n.column_name,
                    "description": n.description,
                }
                for n in downstream_nodes
            ]

        if not impacts:
            return ToolResult.success_result(
                data=[],
                message=f"Column {table}.{column} has no downstream dependencies (it may be a final output)",
            )

        # Build human-readable message
        impact_names = [f"{i['table']}.{i['column']}" for i in impacts[:5]]
        msg = f"Column {table}.{column} impacts: {', '.join(impact_names)}"
        if len(impacts) > 5:
            msg += f" (+{len(impacts) - 5} more)"

        return ToolResult.success_result(data=impacts, message=msg)


class GetLineagePathTool(BaseTool):
    """
    Find the lineage path between two columns.

    Given source and target columns, returns the path of
    transformations connecting them.

    Example:
        tool = GetLineagePathTool(pipeline)
        result = tool.run(
            from_table="raw.orders", from_column="amount",
            to_table="analytics.revenue", to_column="total_amount"
        )
    """

    name = "get_lineage_path"
    description = "Find the path of transformations between two columns"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "from_table": ParameterSpec(
                name="from_table",
                type=ParameterType.STRING,
                description="Source table name",
                required=True,
            ),
            "from_column": ParameterSpec(
                name="from_column",
                type=ParameterType.STRING,
                description="Source column name",
                required=True,
            ),
            "to_table": ParameterSpec(
                name="to_table",
                type=ParameterType.STRING,
                description="Target table name",
                required=True,
            ),
            "to_column": ParameterSpec(
                name="to_column",
                type=ParameterType.STRING,
                description="Target column name",
                required=True,
            ),
        }

    def run(self, from_table: str, from_column: str, to_table: str, to_column: str) -> ToolResult:
        # Validate tables exist
        for tbl in [from_table, to_table]:
            if tbl not in self.pipeline.table_graph.tables:
                return ToolResult.error_result(f"Table '{tbl}' not found")

        # Get lineage path
        edges = self.pipeline.get_lineage_path(from_table, from_column, to_table, to_column)

        if not edges:
            return ToolResult.success_result(
                data=[],
                message=f"No lineage path found between {from_table}.{from_column} and {to_table}.{to_column}",
            )

        # Build path representation
        path = []
        for edge in edges:
            path.append(
                {
                    "from": f"{edge.from_node.table_name}.{edge.from_node.column_name}",
                    "to": f"{edge.to_node.table_name}.{edge.to_node.column_name}",
                    "transformation": edge.transformation,
                    "query_id": edge.query_id,
                }
            )

        # Build message
        path_str = " -> ".join([f"{from_table}.{from_column}"] + [p["to"] for p in path])
        msg = f"Lineage path: {path_str}"

        return ToolResult.success_result(data=path, message=msg)


class GetTableLineageTool(BaseTool):
    """
    Get table-level lineage path for a column.

    Returns a simplified view showing only table hops,
    skipping internal CTEs and subqueries.

    Example:
        tool = GetTableLineageTool(pipeline)
        result = tool.run(table="mart.revenue", column="total")
        # Returns: [("mart.revenue", "total"), ("staging.orders", "amount"), ...]
    """

    name = "get_table_lineage"
    description = "Get simplified table-level lineage path for a column"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "table": ParameterSpec(
                name="table",
                type=ParameterType.STRING,
                description="Table name",
                required=True,
            ),
            "column": ParameterSpec(
                name="column",
                type=ParameterType.STRING,
                description="Column name",
                required=True,
            ),
            "direction": ParameterSpec(
                name="direction",
                type=ParameterType.STRING,
                description="Direction: 'backward' (find sources) or 'forward' (find impacts)",
                required=False,
                default="backward",
                enum=["backward", "forward"],
            ),
        }

    def run(self, table: str, column: str, direction: str = "backward") -> ToolResult:
        # Validate table exists
        if table not in self.pipeline.table_graph.tables:
            return ToolResult.error_result(f"Table '{table}' not found")

        if direction == "backward":
            path = self.pipeline.get_table_lineage_path(table, column)
        else:
            path = self.pipeline.get_table_impact_path(table, column)

        if not path:
            return ToolResult.success_result(
                data=[],
                message=f"No table-level lineage found for {table}.{column}",
            )

        # Format path
        formatted_path = [{"table": tbl, "column": col, "query_id": qid} for tbl, col, qid in path]

        # Build message
        if direction == "backward":
            path_str = " <- ".join(f"{p['table']}.{p['column']}" for p in formatted_path)
            msg = f"Source lineage: {path_str}"
        else:
            path_str = " -> ".join(f"{p['table']}.{p['column']}" for p in formatted_path)
            msg = f"Impact path: {path_str}"

        return ToolResult.success_result(data=formatted_path, message=msg)


__all__ = [
    "TraceBackwardTool",
    "TraceForwardTool",
    "GetLineagePathTool",
    "GetTableLineageTool",
]
