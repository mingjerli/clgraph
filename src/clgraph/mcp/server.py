"""
MCP Server implementation for clgraph lineage tools.

Exposes lineage tools via the Model Context Protocol (MCP),
allowing external LLMs and agents to interact with pipeline lineage.
"""

import argparse
import asyncio
import json
from typing import Any, Dict, List

from ..pipeline import Pipeline
from ..tools import BASIC_TOOLS, ToolRegistry, create_tool_registry
from ..tools.base import ParameterType

# MCP SDK imports - these are optional dependencies
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Resource, TextContent, Tool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None  # type: ignore


def _convert_param_type(param_type: ParameterType) -> str:
    """Convert ParameterType to JSON Schema type."""
    mapping = {
        ParameterType.STRING: "string",
        ParameterType.INTEGER: "integer",
        ParameterType.BOOLEAN: "boolean",
        ParameterType.ARRAY: "array",
        ParameterType.OBJECT: "object",
    }
    return mapping.get(param_type, "string")


def _build_json_schema(tool) -> Dict[str, Any]:
    """Build JSON Schema for a tool's parameters."""
    properties = {}
    required = []

    for name, param in tool.parameters.items():
        prop: Dict[str, Any] = {
            "type": _convert_param_type(param.type),
            "description": param.description,
        }

        if param.enum:
            prop["enum"] = param.enum

        if param.default is not None:
            prop["default"] = param.default

        properties[name] = prop

        if param.required:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def create_mcp_server(
    pipeline: Pipeline,
    llm=None,
    include_llm_tools: bool = True,
) -> "Server":
    """
    Create an MCP server exposing lineage tools.

    Args:
        pipeline: The clgraph Pipeline to expose
        llm: Optional LLM for SQL generation tools
        include_llm_tools: Whether to include tools that require LLM

    Returns:
        MCP Server instance

    Raises:
        ImportError: If mcp package is not installed

    Example:
        from clgraph import Pipeline
        from clgraph.mcp import create_mcp_server

        pipeline = Pipeline.from_sql_files("queries/")
        server = create_mcp_server(pipeline)
    """
    if not MCP_AVAILABLE:
        raise ImportError("MCP SDK not installed. Install with: pip install mcp")

    # Create registry with appropriate tools
    if include_llm_tools and llm is not None:
        registry = create_tool_registry(pipeline, llm)
    else:
        registry = ToolRegistry(pipeline, llm)
        registry.register_all(BASIC_TOOLS)

    # Create server
    server = Server("clgraph-lineage")

    # Register tool list handler
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """Return list of available tools."""
        tools = []
        for tool in registry.all_tools():
            tools.append(
                Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=_build_json_schema(tool),
                )
            )
        return tools

    # Register tool call handler
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Execute a tool and return results."""
        try:
            result = registry.run(name, **arguments)

            if result.success:
                # Format successful result
                content = {
                    "success": True,
                    "message": result.message,
                    "data": result.data,
                }
                if result.metadata:
                    content["metadata"] = result.metadata
            else:
                content = {
                    "success": False,
                    "error": result.error,
                }

            return [TextContent(type="text", text=json.dumps(content, indent=2, default=str))]

        except Exception as e:
            error_content = {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
            }
            return [TextContent(type="text", text=json.dumps(error_content, indent=2))]

    # Register resources
    @server.list_resources()
    async def list_resources() -> List[Resource]:
        """Return list of available resources."""
        resources = [
            Resource(
                uri="pipeline://schema",
                name="Pipeline Schema",
                description="Full schema of all tables and columns in the pipeline",
                mimeType="application/json",
            ),
            Resource(
                uri="pipeline://tables",
                name="Table List",
                description="List of all tables in the pipeline",
                mimeType="application/json",
            ),
        ]

        # Add resource for each table
        for table_name in pipeline.table_graph.tables:
            resources.append(
                Resource(
                    uri=f"pipeline://tables/{table_name}",
                    name=f"Table: {table_name}",
                    description=f"Schema and metadata for table {table_name}",
                    mimeType="application/json",
                )
            )

        return resources

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """Read a resource by URI."""
        if uri == "pipeline://schema":
            return _get_full_schema(pipeline)
        elif uri == "pipeline://tables":
            return _get_table_list(pipeline)
        elif uri.startswith("pipeline://tables/"):
            table_name = uri.replace("pipeline://tables/", "")
            return _get_table_info(pipeline, table_name)
        else:
            return json.dumps({"error": f"Unknown resource: {uri}"})

    return server


def _get_full_schema(pipeline: Pipeline) -> str:
    """Get full pipeline schema as JSON."""
    schema = {
        "dialect": pipeline.dialect,
        "tables": {},
    }

    for table_name, table_node in pipeline.table_graph.tables.items():
        columns = []
        for col in pipeline.get_columns_by_table(table_name):
            columns.append(
                {
                    "name": col.column_name,
                    "description": col.description,
                    "pii": col.pii,
                    "owner": col.owner,
                    "tags": list(col.tags) if col.tags else [],
                }
            )

        # Determine if table is final (created by a query but not read by any)
        is_final = table_node.created_by is not None and len(table_node.read_by) == 0

        # Get upstream/downstream tables from the graph
        upstream = [t.table_name for t in pipeline.table_graph.get_dependencies(table_name)]
        downstream = [t.table_name for t in pipeline.table_graph.get_downstream(table_name)]

        schema["tables"][table_name] = {
            "description": table_node.description,
            "is_source": table_node.is_source,
            "is_final": is_final,
            "columns": columns,
            "upstream": upstream,
            "downstream": downstream,
        }

    return json.dumps(schema, indent=2, default=str)


def _get_table_list(pipeline: Pipeline) -> str:
    """Get list of tables as JSON."""
    tables = []
    for table_name, table_node in pipeline.table_graph.tables.items():
        # Determine if table is final (created by a query but not read by any)
        is_final = table_node.created_by is not None and len(table_node.read_by) == 0

        tables.append(
            {
                "name": table_name,
                "description": table_node.description,
                "is_source": table_node.is_source,
                "is_final": is_final,
                "column_count": len(list(pipeline.get_columns_by_table(table_name))),
            }
        )

    return json.dumps({"tables": tables}, indent=2)


def _get_table_info(pipeline: Pipeline, table_name: str) -> str:
    """Get detailed info for a specific table."""
    if table_name not in pipeline.table_graph.tables:
        return json.dumps({"error": f"Table '{table_name}' not found"})

    table_node = pipeline.table_graph.tables[table_name]

    columns = []
    for col in pipeline.get_columns_by_table(table_name):
        columns.append(
            {
                "name": col.column_name,
                "description": col.description,
                "pii": col.pii,
                "owner": col.owner,
                "tags": list(col.tags) if col.tags else [],
            }
        )

    # Determine if table is final (created by a query but not read by any)
    is_final = table_node.created_by is not None and len(table_node.read_by) == 0

    # Get upstream/downstream tables from the graph
    upstream = [t.table_name for t in pipeline.table_graph.get_dependencies(table_name)]
    downstream = [t.table_name for t in pipeline.table_graph.get_downstream(table_name)]

    info = {
        "name": table_name,
        "description": table_node.description,
        "is_source": table_node.is_source,
        "is_final": is_final,
        "columns": columns,
        "upstream_tables": upstream,
        "downstream_tables": downstream,
    }

    return json.dumps(info, indent=2, default=str)


async def run_mcp_server_async(
    pipeline: Pipeline,
    llm=None,
    include_llm_tools: bool = True,
) -> None:
    """
    Run the MCP server asynchronously.

    Args:
        pipeline: The clgraph Pipeline to expose
        llm: Optional LLM for SQL generation tools
        include_llm_tools: Whether to include tools that require LLM
    """
    if not MCP_AVAILABLE:
        raise ImportError("MCP SDK not installed. Install with: pip install mcp")

    server = create_mcp_server(pipeline, llm, include_llm_tools)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def run_mcp_server(
    pipeline: Pipeline,
    llm=None,
    include_llm_tools: bool = True,
) -> None:
    """
    Run the MCP server (blocking).

    Args:
        pipeline: The clgraph Pipeline to expose
        llm: Optional LLM for SQL generation tools
        include_llm_tools: Whether to include tools that require LLM

    Example:
        from clgraph import Pipeline
        from clgraph.mcp import run_mcp_server

        pipeline = Pipeline.from_sql_files("queries/")
        run_mcp_server(pipeline)  # Blocks until terminated
    """
    asyncio.run(run_mcp_server_async(pipeline, llm, include_llm_tools))


def main():
    """Command-line entry point for MCP server."""
    parser = argparse.ArgumentParser(description="Run clgraph MCP server for lineage tools")
    parser.add_argument(
        "--pipeline",
        "-p",
        required=True,
        help="Path to SQL files directory or JSON pipeline file",
    )
    parser.add_argument(
        "--dialect",
        "-d",
        default="bigquery",
        help="SQL dialect (default: bigquery)",
    )
    parser.add_argument(
        "--no-llm-tools",
        action="store_true",
        help="Exclude tools that require LLM",
    )

    args = parser.parse_args()

    # Load pipeline
    pipeline_path = args.pipeline

    if pipeline_path.endswith(".json"):
        # Load from JSON
        pipeline = Pipeline.from_json(pipeline_path)
    else:
        # Load from SQL files
        pipeline = Pipeline.from_sql_files(pipeline_path, dialect=args.dialect)

    # Run server
    run_mcp_server(
        pipeline,
        llm=None,  # No LLM in CLI mode for now
        include_llm_tools=not args.no_llm_tools,
    )


if __name__ == "__main__":
    main()
