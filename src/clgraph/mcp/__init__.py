"""
MCP Server for clgraph lineage tools.

Uses FastMCP 3.x for automatic tool registration and multi-transport support.

Usage:
    # As a module (stdio for Claude Desktop)
    python -m clgraph.mcp --pipeline path/to/queries/

    # HTTP transport (for remote access)
    python -m clgraph.mcp --pipeline path/to/queries/ --transport http

    # In Claude Desktop config
    {
        "mcpServers": {
            "clgraph": {
                "command": "python",
                "args": ["-m", "clgraph.mcp", "--pipeline", "path/to/queries/"]
            }
        }
    }
"""

from .server import create_mcp_server, run_mcp_server

__all__ = [
    "create_mcp_server",
    "run_mcp_server",
]
