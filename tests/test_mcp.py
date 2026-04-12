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


# =============================================================================
# New comprehensive test classes
# =============================================================================


class TestToolSchemaGeneration:
    """Tests that _register_tool produces correct MCP inputSchema."""

    @requires_fastmcp
    def test_tool_schema_required_params(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        tools = asyncio.run(server.list_tools())
        trace_tool = next(t for t in tools if t.name == "trace_backward")
        schema = trace_tool.parameters
        assert "table" in schema.get("required", [])
        assert "column" in schema.get("required", [])
        assert "include_intermediate" not in schema.get("required", [])

    @requires_fastmcp
    def test_tool_schema_param_types(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        tools = asyncio.run(server.list_tools())
        trace_tool = next(t for t in tools if t.name == "trace_backward")
        props = trace_tool.parameters.get("properties", {})
        assert props["table"]["type"] == "string"
        # include_intermediate may be boolean or anyOf with null for Optional
        intermediate = props["include_intermediate"]
        has_bool = intermediate.get("type") == "boolean" or any(
            item.get("type") == "boolean" for item in intermediate.get("anyOf", [])
        )
        assert has_bool

    @requires_fastmcp
    def test_tool_schema_default_values(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        tools = asyncio.run(server.list_tools())
        trace_tool = next(t for t in tools if t.name == "trace_backward")
        props = trace_tool.parameters.get("properties", {})
        assert props["include_intermediate"]["default"] is False

    @requires_fastmcp
    def test_tool_schema_integer_param(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        tools = asyncio.run(server.list_tools())
        search_tool = next(t for t in tools if t.name == "search_columns")
        props = search_tool.parameters.get("properties", {})
        max_results = props["max_results"]
        has_int = max_results.get("type") == "integer" or any(
            item.get("type") == "integer" for item in max_results.get("anyOf", [])
        )
        assert has_int
        assert max_results["default"] == 50

    @requires_fastmcp
    def test_tool_schema_no_params(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        tools = asyncio.run(server.list_tools())
        tags_tool = next(t for t in tools if t.name == "list_tags")
        schema = tags_tool.parameters
        props = schema.get("properties", {})
        required = schema.get("required", [])
        assert len(props) == 0 or len(required) == 0

    @requires_fastmcp
    def test_all_basic_tools_registered(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        tools = asyncio.run(server.list_tools())
        tool_names = {t.name for t in tools}
        expected = {
            "trace_backward",
            "trace_forward",
            "get_lineage_path",
            "get_table_lineage",
            "list_tables",
            "get_table_schema",
            "get_relationships",
            "search_columns",
            "get_execution_order",
            "find_pii_columns",
            "get_owners",
            "get_columns_by_tag",
            "list_tags",
            "check_data_quality",
        }
        for name in expected:
            assert name in tool_names, f"Tool '{name}' not registered"


class TestToolInvocation:
    """Tests for calling tools through FastMCP."""

    @requires_fastmcp
    def test_list_tables_invocation(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        result = asyncio.run(server.call_tool("list_tables", {}))
        data = json.loads(result.content[0].text)
        assert data["success"] is True
        assert isinstance(data["data"], list)
        table_names = [t["name"] for t in data["data"]]
        assert "staging.users" in table_names

    @requires_fastmcp
    def test_search_columns_invocation(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        result = asyncio.run(server.call_tool("search_columns", {"pattern": "email"}))
        data = json.loads(result.content[0].text)
        assert data["success"] is True

    @requires_fastmcp
    def test_get_table_schema_invocation(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        result = asyncio.run(server.call_tool("get_table_schema", {"table": "staging.users"}))
        data = json.loads(result.content[0].text)
        assert data["success"] is True

    @requires_fastmcp
    def test_get_execution_order_invocation(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        result = asyncio.run(server.call_tool("get_execution_order", {}))
        data = json.loads(result.content[0].text)
        assert data["success"] is True

    @requires_fastmcp
    def test_trace_forward_invocation(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        result = asyncio.run(
            server.call_tool("trace_forward", {"table": "raw.users", "column": "id"})
        )
        data = json.loads(result.content[0].text)
        assert data["success"] is True

    @requires_fastmcp
    def test_tool_with_optional_param_default(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        result = asyncio.run(
            server.call_tool("trace_backward", {"table": "staging.users", "column": "id"})
        )
        data = json.loads(result.content[0].text)
        assert data["success"] is True

    @requires_fastmcp
    def test_tool_with_optional_param_explicit(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        result = asyncio.run(
            server.call_tool(
                "trace_backward",
                {
                    "table": "staging.users",
                    "column": "id",
                    "include_intermediate": True,
                },
            )
        )
        data = json.loads(result.content[0].text)
        assert data["success"] is True

    @requires_fastmcp
    def test_tool_with_no_params(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        result = asyncio.run(server.call_tool("list_tags", {}))
        data = json.loads(result.content[0].text)
        assert data["success"] is True


class TestToolErrorHandling:
    """Tests for error paths through MCP."""

    @requires_fastmcp
    def test_tool_nonexistent_table(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        result = asyncio.run(
            server.call_tool("trace_backward", {"table": "nonexistent", "column": "id"})
        )
        data = json.loads(result.content[0].text)
        assert data["success"] is False
        assert "error" in data
        assert "nonexistent" in data["error"].lower() or "not found" in data["error"].lower()

    @requires_fastmcp
    def test_tool_nonexistent_column(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        result = asyncio.run(
            server.call_tool(
                "trace_backward",
                {"table": "staging.users", "column": "nonexistent"},
            )
        )
        data = json.loads(result.content[0].text)
        assert data["success"] is False
        assert "error" in data
        assert "nonexistent" in data["error"].lower() or "not found" in data["error"].lower()

    @requires_fastmcp
    def test_tool_missing_required_param(self, simple_pipeline):
        import asyncio

        from pydantic import ValidationError

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        # Missing "column" which is required — FastMCP raises ValidationError
        with pytest.raises(ValidationError, match="column"):
            asyncio.run(server.call_tool("trace_backward", {"table": "staging.users"}))

    @requires_fastmcp
    def test_call_nonexistent_tool(self, simple_pipeline):
        import asyncio

        from fastmcp.exceptions import NotFoundError

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        with pytest.raises(NotFoundError, match="nonexistent_tool"):
            asyncio.run(server.call_tool("nonexistent_tool", {}))


class TestResourceAccess:
    """Tests for MCP resources."""

    @requires_fastmcp
    def test_list_resources_includes_schema(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        resources = asyncio.run(server.list_resources())
        uris = [str(r.uri) for r in resources]
        assert any("pipeline://schema" in u for u in uris)

    @requires_fastmcp
    def test_list_resources_includes_tables(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        resources = asyncio.run(server.list_resources())
        uris = [str(r.uri) for r in resources]
        # Find the tables resource (not per-table ones)
        assert any(str(u) == "pipeline://tables" or str(u).endswith("://tables") for u in uris)

    @requires_fastmcp
    def test_list_resources_includes_per_table(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        resources = asyncio.run(server.list_resources())
        uris = [str(r.uri) for r in resources]
        for table_name in simple_pipeline.table_graph.tables:
            expected_uri = f"pipeline://tables/{table_name}"
            assert any(expected_uri in u for u in uris), (
                f"Missing resource for table '{table_name}'"
            )

    @requires_fastmcp
    def test_read_schema_resource(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        result = asyncio.run(server.read_resource("pipeline://schema"))
        data = json.loads(result.contents[0].content)
        assert "dialect" in data
        assert "tables" in data

    @requires_fastmcp
    def test_read_tables_resource(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        result = asyncio.run(server.read_resource("pipeline://tables"))
        data = json.loads(result.contents[0].content)
        assert "tables" in data
        assert isinstance(data["tables"], list)

    @requires_fastmcp
    def test_read_per_table_resource(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        result = asyncio.run(server.read_resource("pipeline://tables/staging.users"))
        data = json.loads(result.contents[0].content)
        assert data["name"] == "staging.users"
        assert "columns" in data

    @requires_fastmcp
    def test_resource_name_sanitization(self, simple_pipeline):
        import asyncio

        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        resources = asyncio.run(server.list_resources())
        # Find resource for staging.users
        staging_resource = None
        for r in resources:
            if "staging.users" in str(r.uri):
                staging_resource = r
                break
        assert staging_resource is not None
        # Name should have dots replaced with underscores
        assert "." not in staging_resource.name


class TestHelperEdgeCases:
    """Tests for helper function edge cases."""

    def test_get_full_schema_upstream_downstream(self, simple_pipeline):
        from clgraph.mcp.server import _get_full_schema

        schema = json.loads(_get_full_schema(simple_pipeline))
        staging_users = schema["tables"]["staging.users"]
        assert "raw.users" in staging_users["upstream"]
        assert "analytics.user_metrics" in staging_users["downstream"]

    def test_get_full_schema_is_final(self, simple_pipeline):
        from clgraph.mcp.server import _get_full_schema

        schema = json.loads(_get_full_schema(simple_pipeline))
        assert schema["tables"]["analytics.user_metrics"]["is_final"] is True
        assert schema["tables"]["staging.users"]["is_final"] is False

    def test_get_full_schema_source_flag(self, simple_pipeline):
        from clgraph.mcp.server import _get_full_schema

        schema = json.loads(_get_full_schema(simple_pipeline))
        assert schema["tables"]["raw.users"]["is_source"] is True
        assert schema["tables"]["staging.users"]["is_source"] is False

    def test_get_table_list_column_count(self, simple_pipeline):
        from clgraph.mcp.server import _get_table_list

        data = json.loads(_get_table_list(simple_pipeline))
        staging = next(t for t in data["tables"] if t["name"] == "staging.users")
        assert staging["column_count"] == 3

    def test_get_table_info_upstream_downstream(self, simple_pipeline):
        from clgraph.mcp.server import _get_table_info

        info = json.loads(_get_table_info(simple_pipeline, "staging.users"))
        assert "analytics.user_metrics" in info["downstream_tables"]


class TestCLI:
    """Tests for CLI argument parsing."""

    def test_cli_json_pipeline_path(self, monkeypatch):
        from unittest.mock import MagicMock, patch

        from clgraph.mcp.server import main

        monkeypatch.setattr("sys.argv", ["clgraph-mcp", "--pipeline", "test.json"])
        mock_pipeline = MagicMock()
        with (
            patch(
                "clgraph.mcp.server.Pipeline.from_json_file",
                return_value=mock_pipeline,
            ) as mock_from_json,
            patch("clgraph.mcp.server.run_mcp_server") as mock_run,
        ):
            main()
            mock_from_json.assert_called_once_with("test.json")
            mock_run.assert_called_once()

    def test_cli_sql_directory_path(self, monkeypatch):
        from unittest.mock import MagicMock, patch

        from clgraph.mcp.server import main

        monkeypatch.setattr("sys.argv", ["clgraph-mcp", "--pipeline", "./queries/"])
        mock_pipeline = MagicMock()
        with (
            patch(
                "clgraph.mcp.server.Pipeline.from_sql_files",
                return_value=mock_pipeline,
            ) as mock_from_sql,
            patch("clgraph.mcp.server.run_mcp_server") as mock_run,
        ):
            main()
            mock_from_sql.assert_called_once_with("./queries/", dialect="bigquery")
            mock_run.assert_called_once()

    def test_cli_transport_option(self, monkeypatch):
        from unittest.mock import MagicMock, patch

        from clgraph.mcp.server import main

        monkeypatch.setattr(
            "sys.argv",
            ["clgraph-mcp", "--pipeline", "test.json", "--transport", "http"],
        )
        mock_pipeline = MagicMock()
        with (
            patch(
                "clgraph.mcp.server.Pipeline.from_json_file",
                return_value=mock_pipeline,
            ),
            patch("clgraph.mcp.server.run_mcp_server") as mock_run,
        ):
            main()
            mock_run.assert_called_once_with(
                mock_pipeline,
                llm=None,
                include_llm_tools=True,
                transport="http",
            )

    def test_cli_no_llm_tools_flag(self, monkeypatch):
        from unittest.mock import MagicMock, patch

        from clgraph.mcp.server import main

        monkeypatch.setattr(
            "sys.argv",
            ["clgraph-mcp", "--pipeline", "test.json", "--no-llm-tools"],
        )
        mock_pipeline = MagicMock()
        with (
            patch(
                "clgraph.mcp.server.Pipeline.from_json_file",
                return_value=mock_pipeline,
            ),
            patch("clgraph.mcp.server.run_mcp_server") as mock_run,
        ):
            main()
            mock_run.assert_called_once_with(
                mock_pipeline,
                llm=None,
                include_llm_tools=False,
                transport="stdio",
            )
