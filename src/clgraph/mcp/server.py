"""
MCP Server implementation for clgraph lineage tools.

Uses FastMCP 3.x for automatic schema generation, multi-transport
support, and streamlined tool registration.
"""

import argparse
import inspect
import json
import logging
from typing import Any, Dict, Optional

from ..pipeline import Pipeline
from ..tools import BASIC_TOOLS, ToolRegistry, create_tool_registry
from ..tools.base import ParameterType

# FastMCP imports - optional dependency
try:
    from fastmcp import FastMCP

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCP = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_TYPE_MAP = {
    ParameterType.STRING: str,
    ParameterType.INTEGER: int,
    ParameterType.BOOLEAN: bool,
    ParameterType.ARRAY: list,
    ParameterType.OBJECT: dict,
}


def create_mcp_server(
    pipeline: Pipeline,
    llm=None,
    include_llm_tools: bool = True,
) -> "FastMCP":
    """
    Create a FastMCP server exposing lineage tools.

    Args:
        pipeline: The clgraph Pipeline to expose
        llm: Optional LLM for SQL generation tools
        include_llm_tools: Whether to include tools that require LLM

    Returns:
        FastMCP server instance

    Raises:
        ImportError: If fastmcp package is not installed

    Example:
        from clgraph import Pipeline
        from clgraph.mcp import create_mcp_server

        pipeline = Pipeline.from_sql_files("queries/")
        server = create_mcp_server(pipeline)
        server.run()  # stdio transport (for Claude Desktop)
    """
    if not FASTMCP_AVAILABLE:
        raise ImportError("FastMCP not installed. Install with: pip install 'clgraph[mcp]'")

    # Create tool registry
    if include_llm_tools and llm is not None:
        registry = create_tool_registry(pipeline, llm)
    else:
        registry = ToolRegistry(pipeline, llm)
        registry.register_all(BASIC_TOOLS)

    # Create FastMCP server
    mcp = FastMCP("clgraph-lineage")

    # Register each tool from the registry
    for tool in registry.all_tools():
        _register_tool(mcp, tool)

    # Register resources
    @mcp.resource("pipeline://schema")
    def pipeline_schema() -> str:
        """Full schema of all tables and columns in the pipeline."""
        return _get_full_schema(pipeline)

    @mcp.resource("pipeline://tables")
    def table_list() -> str:
        """List of all tables in the pipeline."""
        return _get_table_list(pipeline)

    for table_name in pipeline.table_graph.tables:
        _register_table_resource(mcp, pipeline, table_name)

    return mcp


def _register_tool(mcp: "FastMCP", tool) -> None:
    """Register a single BaseTool instance with FastMCP.

    FastMCP does NOT support **kwargs handlers — it needs an explicit
    function signature to generate the JSON schema. We build a proper
    signature using inspect.Parameter and set it on the handler via
    __signature__. FastMCP reads this to produce the MCP inputSchema.

    Enum constraints from ParameterSpec are NOT included in the schema
    (FastMCP auto-schema doesn't support them). Runtime validation still
    happens via BaseTool.validate_params() in __call__.
    """
    # Build an explicit inspect.Signature from ParameterSpec
    sig_params = []
    annotations = {}
    for param_name, param_spec in tool.parameters.items():
        python_type = _TYPE_MAP.get(param_spec.type, str)
        if param_spec.required:
            sig_params.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=python_type,
                )
            )
            annotations[param_name] = python_type
        else:
            sig_params.append(
                inspect.Parameter(
                    param_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=param_spec.default,
                    annotation=Optional[python_type],
                )
            )
            annotations[param_name] = Optional[python_type]

    annotations["return"] = str
    sig = inspect.Signature(sig_params, return_annotation=str)

    def make_handler(t, signature):
        def handler(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            try:
                result = t(**bound.arguments)
                return json.dumps(result.to_dict(), indent=2, default=str)
            except Exception as e:
                logger.error("Tool '%s' execution failed: %s", t.name, e, exc_info=True)
                return json.dumps(
                    {"success": False, "error": f"Tool execution failed: {str(e)}"},
                    indent=2,
                )

        handler.__signature__ = signature
        handler.__annotations__ = annotations
        return handler

    mcp.tool(name=tool.name, description=tool.description)(make_handler(tool, sig))


def _register_table_resource(mcp: "FastMCP", pipeline: Pipeline, table_name: str) -> None:
    """Register a per-table resource with FastMCP."""
    uri = f"pipeline://tables/{table_name}"
    func_name = f"table_{table_name.replace('.', '_')}"
    description = f"Schema and metadata for table {table_name}"

    # Pass name/description to decorator directly — setting __name__/__doc__
    # after @mcp.resource() is too late since FastMCP reads them at decoration
    # time.
    @mcp.resource(uri, name=func_name, description=description)
    def table_info() -> str:
        return _get_table_info(pipeline, table_name)


# =============================================================================
# Resource data helpers (unchanged from original)
# =============================================================================


def _get_full_schema(pipeline: Pipeline) -> str:
    """Get full pipeline schema as JSON."""
    schema: Dict[str, Any] = {
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

        is_final = table_node.created_by is not None and len(table_node.read_by) == 0
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

    is_final = table_node.created_by is not None and len(table_node.read_by) == 0
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


# =============================================================================
# Server runners
# =============================================================================


def run_mcp_server(
    pipeline: Pipeline,
    llm=None,
    include_llm_tools: bool = True,
    transport: str = "stdio",
) -> None:
    """
    Run the MCP server (blocking).

    Args:
        pipeline: The clgraph Pipeline to expose
        llm: Optional LLM for SQL generation tools
        include_llm_tools: Whether to include tools that require LLM
        transport: Transport type ("stdio", "http", "sse", or "streamable-http")
    """
    server = create_mcp_server(pipeline, llm, include_llm_tools)
    server.run(transport=transport)


# =============================================================================
# CLI entry point
# =============================================================================


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
    parser.add_argument(
        "--transport",
        "-t",
        default="stdio",
        choices=["stdio", "http", "sse", "streamable-http"],
        help="Transport type (default: stdio)",
    )

    args = parser.parse_args()

    # Load pipeline
    pipeline_path = args.pipeline

    if pipeline_path.endswith(".json"):
        pipeline = Pipeline.from_json_file(pipeline_path)
    else:
        pipeline = Pipeline.from_sql_files(pipeline_path, dialect=args.dialect)

    # Run server
    run_mcp_server(
        pipeline,
        llm=None,
        include_llm_tools=not args.no_llm_tools,
        transport=args.transport,
    )
