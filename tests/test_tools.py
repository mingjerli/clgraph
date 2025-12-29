"""
Tests for clgraph.tools module.

Tests cover:
- Base tool classes (BaseTool, ToolResult, ToolRegistry)
- Lineage tools (TraceBackward, TraceForward, GetLineagePath)
- Schema tools (ListTables, GetTableSchema, GetRelationships)
- Governance tools (FindPIIColumns, GetOwners, GetColumnsByTag)
- Context builder utilities
"""

import pytest

from clgraph import Pipeline
from clgraph.tools import (
    BASIC_TOOLS,
    CheckDataQualityTool,
    ContextBuilder,
    ContextConfig,
    FindPIIColumnsTool,
    GetColumnsByTagTool,
    GetExecutionOrderTool,
    GetLineagePathTool,
    GetOwnersTool,
    GetRelationshipsTool,
    GetTableLineageTool,
    GetTableSchemaTool,
    ListTablesTool,
    ListTagsTool,
    SearchColumnsTool,
    ToolRegistry,
    ToolResult,
    TraceBackwardTool,
    TraceForwardTool,
    create_tool_registry,
)

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


@pytest.fixture
def pipeline_with_metadata(simple_pipeline):
    """Pipeline with metadata set."""
    pipeline = simple_pipeline

    # Set PII flags
    for col_key in pipeline.columns:
        if "email" in col_key:
            pipeline.columns[col_key].pii = True
        if "name" in col_key:
            pipeline.columns[col_key].pii = True

    # Set descriptions
    for _col_key, col in pipeline.columns.items():
        if col.table_name == "staging.users":
            if col.column_name == "id":
                col.description = "User ID"
            elif col.column_name == "email":
                col.description = "User email address"

    # Set owners
    for _col_key, col in pipeline.columns.items():
        if "raw" in col.table_name:
            col.owner = "data-platform"
        else:
            col.owner = "analytics-team"

    # Set tags
    for _col_key, col in pipeline.columns.items():
        if "amount" in col.column_name or "count" in col.column_name:
            col.tags.add("financial")
        if col.pii:
            col.tags.add("sensitive")

    return pipeline


# =============================================================================
# ToolResult Tests
# =============================================================================


class TestToolResult:
    """Tests for ToolResult class."""

    def test_success_result(self):
        """Test creating a success result."""
        result = ToolResult.success_result(
            data={"key": "value"},
            message="Operation successful",
        )
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.message == "Operation successful"
        assert result.error is None

    def test_error_result(self):
        """Test creating an error result."""
        result = ToolResult.error_result("Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.data is None

    def test_to_dict(self):
        """Test converting result to dict."""
        result = ToolResult.success_result(
            data=[1, 2, 3],
            message="Got data",
            count=3,  # Metadata is passed as **kwargs
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["data"] == [1, 2, 3]
        assert d["message"] == "Got data"
        assert d["metadata"] == {"count": 3}


# =============================================================================
# ToolRegistry Tests
# =============================================================================


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_create_tool_registry(self, simple_pipeline):
        """Test creating a tool registry."""
        registry = create_tool_registry(simple_pipeline)
        assert registry is not None

        # Should have registered basic tools
        tools = registry.all_tools()
        tool_names = [t.name for t in tools]
        assert "trace_backward" in tool_names
        assert "trace_forward" in tool_names
        assert "list_tables" in tool_names

    def test_registry_run(self, simple_pipeline):
        """Test running a tool through the registry."""
        registry = create_tool_registry(simple_pipeline)
        result = registry.run("list_tables")
        assert result.success is True
        assert "data" in result.to_dict()

    def test_registry_run_unknown_tool(self, simple_pipeline):
        """Test running an unknown tool."""
        registry = create_tool_registry(simple_pipeline)
        result = registry.run("unknown_tool")
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_registry_get_tool(self, simple_pipeline):
        """Test getting a tool by name."""
        registry = create_tool_registry(simple_pipeline)
        tool = registry.get("trace_backward")
        assert tool is not None
        assert tool.name == "trace_backward"

    def test_registry_register_all(self, simple_pipeline):
        """Test registering all basic tools."""
        registry = ToolRegistry(simple_pipeline, llm=None)
        registry.register_all(BASIC_TOOLS)
        assert len(registry.all_tools()) == len(BASIC_TOOLS)


# =============================================================================
# Lineage Tools Tests
# =============================================================================


class TestTraceBackwardTool:
    """Tests for TraceBackwardTool."""

    def test_trace_backward_basic(self, simple_pipeline):
        """Test basic backward tracing."""
        tool = TraceBackwardTool(simple_pipeline)
        result = tool.run(table="analytics.user_metrics", column="user_id")
        assert result.success is True
        assert len(result.data) > 0

        # Should trace back to source
        source_tables = {r["table"] for r in result.data}
        assert "raw.users" in source_tables or "staging.users" in source_tables

    def test_trace_backward_not_found(self, simple_pipeline):
        """Test tracing non-existent column."""
        tool = TraceBackwardTool(simple_pipeline)
        result = tool.run(table="analytics.user_metrics", column="nonexistent")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_trace_backward_invalid_table(self, simple_pipeline):
        """Test tracing from non-existent table."""
        tool = TraceBackwardTool(simple_pipeline)
        result = tool.run(table="nonexistent.table", column="col")
        assert result.success is False


class TestTraceForwardTool:
    """Tests for TraceForwardTool."""

    def test_trace_forward_basic(self, simple_pipeline):
        """Test basic forward tracing."""
        tool = TraceForwardTool(simple_pipeline)
        result = tool.run(table="raw.users", column="id")
        assert result.success is True
        assert len(result.data) > 0

        # Should trace forward to downstream
        downstream_tables = {r["table"] for r in result.data}
        assert "staging.users" in downstream_tables or "analytics.user_metrics" in downstream_tables


class TestGetLineagePathTool:
    """Tests for GetLineagePathTool."""

    def test_get_lineage_path_exists(self, simple_pipeline):
        """Test finding a lineage path."""
        tool = GetLineagePathTool(simple_pipeline)
        result = tool.run(
            from_table="raw.users",
            from_column="id",
            to_table="analytics.user_metrics",
            to_column="user_id",
        )
        assert result.success is True
        # Data is a list of path edges
        assert isinstance(result.data, list)
        assert len(result.data) > 0

    def test_get_lineage_path_not_exists(self, simple_pipeline):
        """Test when no path exists."""
        tool = GetLineagePathTool(simple_pipeline)
        result = tool.run(
            from_table="analytics.user_metrics",
            from_column="total_amount",
            to_table="raw.users",
            to_column="id",
        )
        assert result.success is True
        # Empty list means no path
        assert result.data == []


class TestGetTableLineageTool:
    """Tests for GetTableLineageTool."""

    def test_get_table_lineage(self, simple_pipeline):
        """Test getting table-level lineage."""
        tool = GetTableLineageTool(simple_pipeline)
        # GetTableLineageTool requires both table and column
        result = tool.run(table="staging.users", column="id")
        assert result.success is True
        # Returns a list of (table, column) tuples as dicts
        assert isinstance(result.data, list)


# =============================================================================
# Schema Tools Tests
# =============================================================================


class TestListTablesTool:
    """Tests for ListTablesTool."""

    def test_list_tables(self, simple_pipeline):
        """Test listing all tables."""
        tool = ListTablesTool(simple_pipeline)
        result = tool.run()
        assert result.success is True
        assert len(result.data) > 0

        table_names = [t["name"] for t in result.data]
        assert "staging.users" in table_names
        assert "analytics.user_metrics" in table_names

    def test_list_tables_include_sources(self, simple_pipeline):
        """Test listing tables with sources."""
        tool = ListTablesTool(simple_pipeline)
        result = tool.run(include_sources=True)
        assert result.success is True

        # Should include source tables
        table_names = [t["name"] for t in result.data]
        assert "raw.users" in table_names


class TestGetTableSchemaTool:
    """Tests for GetTableSchemaTool."""

    def test_get_table_schema(self, simple_pipeline):
        """Test getting table schema."""
        tool = GetTableSchemaTool(simple_pipeline)
        result = tool.run(table="staging.users")
        assert result.success is True
        assert "columns" in result.data

        col_names = [c["name"] for c in result.data["columns"]]
        assert "id" in col_names
        assert "name" in col_names
        assert "email" in col_names

    def test_get_table_schema_not_found(self, simple_pipeline):
        """Test getting schema of non-existent table."""
        tool = GetTableSchemaTool(simple_pipeline)
        result = tool.run(table="nonexistent.table")
        assert result.success is False


class TestGetRelationshipsTool:
    """Tests for GetRelationshipsTool."""

    def test_get_relationships(self, simple_pipeline):
        """Test getting table relationships."""
        tool = GetRelationshipsTool(simple_pipeline)
        result = tool.run()
        assert result.success is True
        assert len(result.data) > 0


class TestSearchColumnsTool:
    """Tests for SearchColumnsTool."""

    def test_search_columns(self, simple_pipeline):
        """Test searching for columns."""
        tool = SearchColumnsTool(simple_pipeline)
        result = tool.run(pattern="id")
        assert result.success is True
        assert len(result.data) > 0

        # All results should contain 'id'
        for col in result.data:
            assert "id" in col["column"].lower()


class TestGetExecutionOrderTool:
    """Tests for GetExecutionOrderTool."""

    def test_get_execution_order(self, simple_pipeline):
        """Test getting execution order."""
        tool = GetExecutionOrderTool(simple_pipeline)
        result = tool.run()
        assert result.success is True
        # Data is a list of query info dicts with query_id, destination_table, source_tables
        assert isinstance(result.data, list)
        assert len(result.data) == 2

        # staging_users should come before analytics_user_metrics
        query_ids = [q["query_id"] for q in result.data]
        assert query_ids.index("staging_users") < query_ids.index("analytics_user_metrics")


# =============================================================================
# Governance Tools Tests
# =============================================================================


class TestFindPIIColumnsTool:
    """Tests for FindPIIColumnsTool."""

    def test_find_pii_columns(self, pipeline_with_metadata):
        """Test finding PII columns."""
        tool = FindPIIColumnsTool(pipeline_with_metadata)
        result = tool.run()
        assert result.success is True
        assert len(result.data) > 0

        # Should find email and name columns
        pii_cols = [f"{c['table']}.{c['column']}" for c in result.data]
        assert any("email" in col for col in pii_cols)
        assert any("name" in col for col in pii_cols)

    def test_find_pii_columns_by_table(self, pipeline_with_metadata):
        """Test finding PII columns for specific table."""
        tool = FindPIIColumnsTool(pipeline_with_metadata)
        result = tool.run(table="staging.users")
        assert result.success is True

        # All results should be from staging.users
        for col in result.data:
            assert col["table"] == "staging.users"


class TestGetOwnersTool:
    """Tests for GetOwnersTool."""

    def test_get_owners(self, pipeline_with_metadata):
        """Test getting ownership info."""
        tool = GetOwnersTool(pipeline_with_metadata)
        result = tool.run()
        assert result.success is True
        assert len(result.data) > 0

    def test_get_owners_by_owner(self, pipeline_with_metadata):
        """Test filtering by owner."""
        tool = GetOwnersTool(pipeline_with_metadata)
        result = tool.run(owner="data-platform")
        assert result.success is True

        for col in result.data:
            assert col["owner"] == "data-platform"


class TestGetColumnsByTagTool:
    """Tests for GetColumnsByTagTool."""

    def test_get_columns_by_tag(self, pipeline_with_metadata):
        """Test finding columns by tag."""
        tool = GetColumnsByTagTool(pipeline_with_metadata)
        result = tool.run(tag="financial")
        assert result.success is True
        assert len(result.data) > 0

        # All results should have the tag
        for col in result.data:
            assert "financial" in col.get("tags", [])


class TestListTagsTool:
    """Tests for ListTagsTool."""

    def test_list_tags(self, pipeline_with_metadata):
        """Test listing all tags."""
        tool = ListTagsTool(pipeline_with_metadata)
        result = tool.run()
        assert result.success is True
        assert len(result.data) > 0

        tag_names = [t["tag"] for t in result.data]
        assert "financial" in tag_names
        assert "sensitive" in tag_names


class TestCheckDataQualityTool:
    """Tests for CheckDataQualityTool."""

    def test_check_data_quality(self, pipeline_with_metadata):
        """Test checking data quality."""
        tool = CheckDataQualityTool(pipeline_with_metadata)
        result = tool.run()
        assert result.success is True
        assert "total_columns" in result.data
        assert "description_coverage_pct" in result.data
        assert "ownership_coverage_pct" in result.data


# =============================================================================
# ContextBuilder Tests
# =============================================================================


class TestContextBuilder:
    """Tests for ContextBuilder class."""

    def test_build_schema_context(self, simple_pipeline):
        """Test building schema context."""
        builder = ContextBuilder(simple_pipeline)
        context = builder.build_schema_context()
        assert isinstance(context, str)
        assert "staging.users" in context
        assert "analytics.user_metrics" in context

    def test_build_context_for_tables(self, simple_pipeline):
        """Test building context for specific tables."""
        builder = ContextBuilder(simple_pipeline)
        context = builder.build_context_for_tables(["staging.users"])
        assert isinstance(context, str)
        assert "staging.users" in context

    def test_expand_with_lineage(self, simple_pipeline):
        """Test expanding table list with lineage."""
        builder = ContextBuilder(simple_pipeline)
        expanded = builder.expand_with_lineage(["staging.users"])
        assert "staging.users" in expanded
        # Should also include related tables
        assert len(expanded) >= 1

    def test_select_tables_by_keywords(self, simple_pipeline):
        """Test selecting tables by keywords."""
        builder = ContextBuilder(simple_pipeline)
        selected = builder.select_tables_by_keywords("user metrics")
        assert len(selected) > 0

    def test_get_pii_columns(self, pipeline_with_metadata):
        """Test getting PII columns."""
        builder = ContextBuilder(pipeline_with_metadata)
        pii_cols = builder.get_pii_columns()
        assert len(pii_cols) > 0

    def test_context_config(self, simple_pipeline):
        """Test context config options."""
        config = ContextConfig(
            include_descriptions=True,
            include_pii_flags=True,
            include_lineage=True,
            max_columns_per_table=5,
        )
        builder = ContextBuilder(simple_pipeline, config)
        context = builder.build_schema_context()
        assert isinstance(context, str)
