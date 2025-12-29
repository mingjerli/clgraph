"""
Tests for clgraph.mcp module.

Tests cover:
- MCP server creation (when mcp is installed)
- Resource generation
- JSON schema generation for tools
- Helper functions that don't require MCP
"""

import json

import pytest

from clgraph import Pipeline

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_pipeline():
    """Create a simple pipeline for testing."""
    queries = {
        "staging_users": """
            CREATE TABLE staging.users AS
            SELECT
                id,
                name,
                email
            FROM raw.users
        """,
        "analytics_user_metrics": """
            CREATE TABLE analytics.user_metrics AS
            SELECT
                u.id AS user_id,
                u.name,
                COUNT(*) AS order_count,
                SUM(o.amount) AS total_amount
            FROM staging.users u
            JOIN raw.orders o ON u.id = o.user_id
            GROUP BY u.id, u.name
        """,
    }
    return Pipeline.from_dict(queries, dialect="bigquery")


# =============================================================================
# Import Tests
# =============================================================================


class TestMCPImports:
    """Test MCP module imports."""

    def test_import_mcp_module(self):
        """Test importing the MCP module."""
        from clgraph.mcp import create_mcp_server, run_mcp_server

        assert create_mcp_server is not None
        assert run_mcp_server is not None

    def test_mcp_availability_flag(self):
        """Test MCP availability detection."""
        from clgraph.mcp.server import MCP_AVAILABLE

        # This will be True if mcp is installed, False otherwise
        assert isinstance(MCP_AVAILABLE, bool)


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestMCPHelpers:
    """Test MCP helper functions that don't require MCP SDK."""

    def test_get_full_schema(self, simple_pipeline):
        """Test generating full schema JSON."""
        from clgraph.mcp.server import _get_full_schema

        schema_json = _get_full_schema(simple_pipeline)
        schema = json.loads(schema_json)

        assert "dialect" in schema
        assert schema["dialect"] == "bigquery"
        assert "tables" in schema
        assert "staging.users" in schema["tables"]
        assert "analytics.user_metrics" in schema["tables"]

    def test_get_table_list(self, simple_pipeline):
        """Test generating table list JSON."""
        from clgraph.mcp.server import _get_table_list

        list_json = _get_table_list(simple_pipeline)
        data = json.loads(list_json)

        assert "tables" in data
        table_names = [t["name"] for t in data["tables"]]
        assert "staging.users" in table_names
        assert "analytics.user_metrics" in table_names

    def test_get_table_info(self, simple_pipeline):
        """Test generating table info JSON."""
        from clgraph.mcp.server import _get_table_info

        info_json = _get_table_info(simple_pipeline, "staging.users")
        info = json.loads(info_json)

        assert info["name"] == "staging.users"
        assert "columns" in info
        assert "upstream_tables" in info
        assert "downstream_tables" in info

    def test_get_table_info_not_found(self, simple_pipeline):
        """Test table info for non-existent table."""
        from clgraph.mcp.server import _get_table_info

        info_json = _get_table_info(simple_pipeline, "nonexistent.table")
        info = json.loads(info_json)

        assert "error" in info

    def test_convert_param_type(self):
        """Test parameter type conversion."""
        from clgraph.mcp.server import _convert_param_type
        from clgraph.tools.base import ParameterType

        assert _convert_param_type(ParameterType.STRING) == "string"
        assert _convert_param_type(ParameterType.INTEGER) == "integer"
        assert _convert_param_type(ParameterType.BOOLEAN) == "boolean"
        assert _convert_param_type(ParameterType.ARRAY) == "array"
        assert _convert_param_type(ParameterType.OBJECT) == "object"

    def test_build_json_schema(self, simple_pipeline):
        """Test building JSON schema for a tool."""
        from clgraph.mcp.server import _build_json_schema
        from clgraph.tools import TraceBackwardTool

        tool = TraceBackwardTool(simple_pipeline)
        schema = _build_json_schema(tool)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "table" in schema["properties"]
        assert "column" in schema["properties"]
        assert "table" in schema["required"]
        assert "column" in schema["required"]


# =============================================================================
# Server Creation Tests (when MCP is available)
# =============================================================================


class TestMCPServerCreation:
    """Test MCP server creation (skipped if MCP not installed)."""

    def test_create_server_without_mcp(self, simple_pipeline):
        """Test that create_mcp_server raises ImportError without MCP."""
        from clgraph.mcp.server import MCP_AVAILABLE

        if not MCP_AVAILABLE:
            from clgraph.mcp import create_mcp_server

            with pytest.raises(ImportError) as exc_info:
                create_mcp_server(simple_pipeline)
            assert "MCP SDK not installed" in str(exc_info.value)

    @pytest.mark.skipif(
        not __import__("clgraph.mcp.server", fromlist=["MCP_AVAILABLE"]).MCP_AVAILABLE,
        reason="MCP SDK not installed",
    )
    def test_create_server_with_mcp(self, simple_pipeline):
        """Test creating MCP server when MCP is available."""
        from clgraph.mcp import create_mcp_server

        server = create_mcp_server(simple_pipeline)
        assert server is not None
        # Server should have the correct name
        assert server.name == "clgraph-lineage"


# =============================================================================
# CLI Tests
# =============================================================================


class TestMCPCLI:
    """Test MCP CLI entry point."""

    def test_main_module_exists(self):
        """Test that __main__ module exists."""
        from clgraph.mcp.__main__ import main

        assert main is not None

    def test_argparse_setup(self):
        """Test that argparse is configured correctly."""

        from clgraph.mcp.server import main

        # We can't actually run main() without files, but we can verify
        # the function exists and is callable
        assert callable(main)
