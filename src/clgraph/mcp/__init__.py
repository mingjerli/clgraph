"""
MCP Server for clgraph lineage tools.

Exposes lineage tools via the Model Context Protocol (MCP),
allowing external LLMs and agents to interact with pipeline lineage.

Usage:
    # As a module
    python -m clgraph.mcp --pipeline path/to/queries/

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
