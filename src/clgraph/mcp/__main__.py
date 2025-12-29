"""
Entry point for running clgraph MCP server as a module.

Usage:
    python -m clgraph.mcp --pipeline path/to/queries/
    python -m clgraph.mcp --pipeline pipeline.json
"""

from .server import main

if __name__ == "__main__":
    main()
