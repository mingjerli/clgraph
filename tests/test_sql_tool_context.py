# tests/test_sql_tool_context.py
"""T1: direct mode gets column lineage; one capped table set feeds every section."""

import pytest

from clgraph import Pipeline
from clgraph.tools import ContextBuilder, ContextConfig
from clgraph.tools.sql import GenerateSQLTool


def _fake_llm_capturing(prompts):
    def llm(prompt):
        prompts.append(prompt)
        return "```sql\nSELECT 1\n```"

    return llm


@pytest.fixture
def pipeline():
    return Pipeline.from_dict(
        {
            "staging_users": """
                CREATE TABLE staging.users AS
                SELECT id, email FROM raw.users
            """,
            "mart_users": """
                CREATE TABLE mart.users AS
                SELECT id, UPPER(email) AS email_norm FROM staging.users
            """,
        },
        dialect="bigquery",
    )


def test_direct_mode_prompt_contains_column_lineage(pipeline):
    prompts = []
    tool = GenerateSQLTool(pipeline, _fake_llm_capturing(prompts))
    result = tool.run(question="how many users?", include_explanation=False)
    assert result.success
    assert "## Column Lineage" in prompts[-1]
    assert "mart.users.email_norm" in prompts[-1]


def test_resolve_context_tables_is_capped_and_ordered(pipeline):
    builder = ContextBuilder(pipeline, ContextConfig(max_tables=2))
    tables = builder.resolve_context_tables()
    # role priority: final > intermediate > source (Task 6)
    assert tables == ["mart.users", "staging.users"]


def test_sections_only_reference_in_schema_tables(pipeline):
    prompts = []
    config = ContextConfig(max_tables=2)
    tool = GenerateSQLTool(pipeline, _fake_llm_capturing(prompts))
    tool_builder = ContextBuilder(pipeline, config)
    resolved = tool_builder.resolve_context_tables()

    graph_ctx = tool._build_graph_context(tool_builder, resolved)
    dropped = set(pipeline.table_graph.tables) - set(resolved)
    for table in dropped:
        assert table not in graph_ctx


def test_lineage_caps_are_configurable(pipeline):
    builder = ContextBuilder(pipeline, ContextConfig(max_lineage_lines=1))
    ctx = builder.build_lineage_context(list(pipeline.table_graph.tables))
    body = [ln for ln in ctx.splitlines() if ln.startswith("- ")]
    assert len(body) <= 1
