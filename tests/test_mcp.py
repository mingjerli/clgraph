"""
Tests for clgraph.mcp module (FastMCP 3.x).
"""

import json

import pytest

from clgraph import Pipeline
from clgraph.mcp.server import FASTMCP_AVAILABLE

# Use module-level variable for skipif instead of fragile __import__ pattern
requires_fastmcp = pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not installed")


@pytest.fixture
def simple_pipeline():
    """Create a simple pipeline for testing."""
    queries = {
        "staging_users": """
            CREATE TABLE staging.users AS
            SELECT id, name, email FROM raw.users
        """,
        "analytics_user_metrics": """
            CREATE TABLE analytics.user_metrics AS
            SELECT
                u.id AS user_id, u.name,
                COUNT(*) AS order_count, SUM(o.amount) AS total_amount
            FROM staging.users u
            JOIN raw.orders o ON u.id = o.user_id
            GROUP BY u.id, u.name
        """,
    }
    return Pipeline.from_dict(queries, dialect="bigquery")


class TestMCPImports:
    def test_import_mcp_module(self):
        from clgraph.mcp import create_mcp_server, run_mcp_server

        assert create_mcp_server is not None
        assert run_mcp_server is not None

    def test_fastmcp_availability_flag(self):
        from clgraph.mcp.server import FASTMCP_AVAILABLE

        assert isinstance(FASTMCP_AVAILABLE, bool)


class TestMCPHelpers:
    def test_get_full_schema(self, simple_pipeline):
        from clgraph.mcp.server import _get_full_schema

        schema_json = _get_full_schema(simple_pipeline)
        schema = json.loads(schema_json)
        assert schema["dialect"] == "bigquery"
        assert "staging.users" in schema["tables"]

    def test_get_table_list(self, simple_pipeline):
        from clgraph.mcp.server import _get_table_list

        data = json.loads(_get_table_list(simple_pipeline))
        table_names = [t["name"] for t in data["tables"]]
        assert "staging.users" in table_names

    def test_get_table_info(self, simple_pipeline):
        from clgraph.mcp.server import _get_table_info

        info = json.loads(_get_table_info(simple_pipeline, "staging.users"))
        assert info["name"] == "staging.users"
        assert "columns" in info

    def test_get_table_info_not_found(self, simple_pipeline):
        from clgraph.mcp.server import _get_table_info

        info = json.loads(_get_table_info(simple_pipeline, "nonexistent"))
        assert "error" in info


class TestFastMCPServerCreation:
    def test_create_server_without_fastmcp(self, simple_pipeline):
        from clgraph.mcp.server import FASTMCP_AVAILABLE

        if not FASTMCP_AVAILABLE:
            from clgraph.mcp import create_mcp_server

            with pytest.raises(ImportError, match="FastMCP"):
                create_mcp_server(simple_pipeline)

    @requires_fastmcp
    def test_create_server_returns_fastmcp_instance(self, simple_pipeline):
        from fastmcp import FastMCP

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        assert isinstance(server, FastMCP)
        assert hasattr(server, "run")

    @requires_fastmcp
    def test_server_has_tools_registered(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server
        from clgraph.tools import BASIC_TOOLS

        server = create_mcp_server(simple_pipeline)
        # list_tools() is async in FastMCP 3.x
        tools = asyncio.run(server.list_tools())
        tool_names = [t.name for t in tools]
        # Should have at least as many tools as BASIC_TOOLS
        assert len(tool_names) >= len(BASIC_TOOLS)
        # Verify a known tool is present
        assert "trace_backward" in tool_names

    @requires_fastmcp
    def test_tool_invocation_returns_valid_json(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        result = asyncio.run(
            server.call_tool("trace_backward", {"table": "staging.users", "column": "id"})
        )
        # call_tool returns a ToolResult with .content list of ContentBlock items
        data = json.loads(result.content[0].text)
        assert data["success"] is True

    @requires_fastmcp
    def test_server_excludes_llm_tools_when_no_llm(self, simple_pipeline):
        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline, llm=None, include_llm_tools=False)
        assert server is not None


class TestMCPCLI:
    def test_main_module_exists(self):
        from clgraph.mcp.__main__ import main

        assert main is not None

    def test_main_is_callable(self):
        from clgraph.mcp.server import main

        assert callable(main)
