"""
Schema discovery tools.

Tools for exploring tables, columns, and relationships in the pipeline.
"""

from typing import Dict, Optional

from .base import BaseTool, ParameterSpec, ParameterType, ToolResult
from .context import ContextBuilder


class ListTablesTool(BaseTool):
    """
    List all tables in the pipeline.

    Returns table names with optional filtering and metadata.

    Example:
        tool = ListTablesTool(pipeline)
        result = tool.run()  # All tables
        result = tool.run(include_sources=False)  # Only derived tables
    """

    name = "list_tables"
    description = "List all tables in the pipeline"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "include_sources": ParameterSpec(
                name="include_sources",
                type=ParameterType.BOOLEAN,
                description="Whether to include source tables",
                required=False,
                default=True,
            ),
            "include_descriptions": ParameterSpec(
                name="include_descriptions",
                type=ParameterType.BOOLEAN,
                description="Whether to include table descriptions",
                required=False,
                default=True,
            ),
        }

    def run(self, include_sources: bool = True, include_descriptions: bool = True) -> ToolResult:
        tables = []

        for name, node in self.pipeline.table_graph.tables.items():
            if not include_sources and node.is_source:
                continue

            table_info = {
                "name": name,
                "is_source": node.is_source,
                "column_count": len(node.columns),
            }

            if include_descriptions and node.description:
                table_info["description"] = node.description

            if node.created_by:
                table_info["created_by"] = node.created_by

            tables.append(table_info)

        # Sort by name
        tables.sort(key=lambda t: t["name"])

        # Build message
        source_count = sum(1 for t in tables if t.get("is_source"))
        derived_count = len(tables) - source_count
        msg = f"Found {len(tables)} tables ({source_count} source, {derived_count} derived)"

        return ToolResult.success_result(data=tables, message=msg)


class GetTableSchemaTool(BaseTool):
    """
    Get detailed schema for a specific table.

    Returns columns, descriptions, types, and metadata.

    Example:
        tool = GetTableSchemaTool(pipeline)
        result = tool.run(table="analytics.revenue")
    """

    name = "get_table_schema"
    description = "Get detailed schema information for a table"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "table": ParameterSpec(
                name="table",
                type=ParameterType.STRING,
                description="Table name to get schema for",
                required=True,
            ),
            "include_pii_info": ParameterSpec(
                name="include_pii_info",
                type=ParameterType.BOOLEAN,
                description="Whether to include PII flags",
                required=False,
                default=True,
            ),
        }

    def run(self, table: str, include_pii_info: bool = True) -> ToolResult:
        # Validate table exists
        if table not in self.pipeline.table_graph.tables:
            available = list(self.pipeline.table_graph.tables.keys())[:10]
            return ToolResult.error_result(
                f"Table '{table}' not found. Available tables: {available}"
            )

        builder = ContextBuilder(self.pipeline)
        table_info = builder.get_table_info(table)

        if table_info is None:
            return ToolResult.error_result(f"Could not get info for table '{table}'")

        # Build column list
        columns = []
        for col_name in table_info.columns:
            col_info = {
                "name": col_name,
            }

            if col_name in table_info.column_descriptions:
                col_info["description"] = table_info.column_descriptions[col_name]

            if include_pii_info and col_name in table_info.column_pii:
                col_info["pii"] = True

            if col_name in table_info.column_owners:
                col_info["owner"] = table_info.column_owners[col_name]

            columns.append(col_info)

        result = {
            "table_name": table,
            "description": table_info.description,
            "is_source": table_info.is_source,
            "columns": columns,
            "source_tables": table_info.source_tables,
            "created_by": table_info.created_by,
        }

        msg = f"Table {table} has {len(columns)} columns"
        if table_info.is_source:
            msg += " (source table)"
        elif table_info.source_tables:
            msg += f", derived from {', '.join(table_info.source_tables[:3])}"

        return ToolResult.success_result(data=result, message=msg)


class GetRelationshipsTool(BaseTool):
    """
    Get table relationships (dependencies).

    Returns how tables are connected through lineage.

    Example:
        tool = GetRelationshipsTool(pipeline)
        result = tool.run()  # All relationships
        result = tool.run(table="analytics.revenue")  # For specific table
    """

    name = "get_relationships"
    description = "Get table dependency relationships"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "table": ParameterSpec(
                name="table",
                type=ParameterType.STRING,
                description="Optional: filter relationships for this table",
                required=False,
            ),
            "direction": ParameterSpec(
                name="direction",
                type=ParameterType.STRING,
                description="Direction: 'both', 'upstream', or 'downstream'",
                required=False,
                default="both",
                enum=["both", "upstream", "downstream"],
            ),
        }

    def run(self, table: Optional[str] = None, direction: str = "both") -> ToolResult:
        if table and table not in self.pipeline.table_graph.tables:
            return ToolResult.error_result(f"Table '{table}' not found")

        relationships = []

        for tbl_name, tbl_node in self.pipeline.table_graph.tables.items():
            if tbl_node.created_by:
                query = self.pipeline.table_graph.queries.get(tbl_node.created_by)
                if query:
                    for source_table in query.source_tables:
                        rel = {
                            "source": source_table,
                            "target": tbl_name,
                            "query_id": tbl_node.created_by,
                        }

                        # Apply table filter
                        if table:
                            if direction == "upstream" and tbl_name != table:
                                continue
                            elif direction == "downstream" and source_table != table:
                                continue
                            elif direction == "both":
                                if tbl_name != table and source_table != table:
                                    continue

                        relationships.append(rel)

        # Sort by target table
        relationships.sort(key=lambda r: (r["target"], r["source"]))

        if table:
            msg = f"Found {len(relationships)} relationships involving {table}"
        else:
            msg = f"Found {len(relationships)} table relationships"

        return ToolResult.success_result(data=relationships, message=msg)


class SearchColumnsTool(BaseTool):
    """
    Search for columns by name pattern.

    Finds columns across all tables matching a search pattern.

    Example:
        tool = SearchColumnsTool(pipeline)
        result = tool.run(pattern="amount")  # Find all columns containing "amount"
    """

    name = "search_columns"
    description = "Search for columns by name pattern across all tables"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "pattern": ParameterSpec(
                name="pattern",
                type=ParameterType.STRING,
                description="Search pattern (case-insensitive substring match)",
                required=True,
            ),
            "max_results": ParameterSpec(
                name="max_results",
                type=ParameterType.INTEGER,
                description="Maximum number of results to return",
                required=False,
                default=50,
            ),
        }

    def run(self, pattern: str, max_results: int = 50) -> ToolResult:
        pattern_lower = pattern.lower()
        matches = []

        for col in self.pipeline.columns.values():
            if pattern_lower in col.column_name.lower():
                matches.append(
                    {
                        "table": col.table_name,
                        "column": col.column_name,
                        "description": col.description,
                        "pii": col.pii,
                    }
                )

            if len(matches) >= max_results:
                break

        # Deduplicate by table.column
        seen = set()
        unique_matches = []
        for m in matches:
            key = f"{m['table']}.{m['column']}"
            if key not in seen:
                seen.add(key)
                unique_matches.append(m)

        msg = f"Found {len(unique_matches)} columns matching '{pattern}'"
        if len(matches) >= max_results:
            msg += f" (limited to {max_results})"

        return ToolResult.success_result(data=unique_matches, message=msg)


class GetExecutionOrderTool(BaseTool):
    """
    Get the topological execution order of queries.

    Returns queries in the order they should be executed
    to satisfy dependencies.

    Example:
        tool = GetExecutionOrderTool(pipeline)
        result = tool.run()
    """

    name = "get_execution_order"
    description = "Get the order in which queries should be executed"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "include_sql": ParameterSpec(
                name="include_sql",
                type=ParameterType.BOOLEAN,
                description="Whether to include SQL in output",
                required=False,
                default=False,
            ),
        }

    def run(self, include_sql: bool = False) -> ToolResult:
        order = self.pipeline.table_graph.topological_sort()

        queries = []
        for query_id in order:
            query = self.pipeline.table_graph.queries.get(query_id)
            if query:
                query_info = {
                    "query_id": query_id,
                    "destination_table": query.destination_table,
                    "source_tables": list(query.source_tables),
                }
                if include_sql:
                    query_info["sql"] = query.sql

                queries.append(query_info)

        msg = f"Pipeline has {len(queries)} queries in execution order"

        return ToolResult.success_result(data=queries, message=msg)


__all__ = [
    "ListTablesTool",
    "GetTableSchemaTool",
    "GetRelationshipsTool",
    "SearchColumnsTool",
    "GetExecutionOrderTool",
]
