# tests/test_table_role_steering.py
"""T4: source/intermediate/final labels, prompt instruction, truncation priority."""

import pytest

from clgraph import Pipeline
from clgraph.tools import ContextBuilder, ContextConfig
from clgraph.tools.sql import GenerateSQLTool


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


def test_table_roles(pipeline):
    builder = ContextBuilder(pipeline)
    assert builder.table_role("raw.users") == "source"
    assert builder.table_role("staging.users") == "intermediate"
    assert builder.table_role("mart.users") == "final"


def test_roles_annotated_in_schema_context(pipeline):
    ctx = ContextBuilder(pipeline).build_schema_context()
    assert "(Source table)" in ctx
    assert "(Intermediate table)" in ctx
    assert "(Final table)" in ctx


def test_annotation_can_be_disabled(pipeline):
    ctx = ContextBuilder(pipeline, ContextConfig(annotate_table_roles=False)).build_schema_context()
    assert "(Final table)" not in ctx
    assert "(Intermediate table)" not in ctx


def test_truncation_keeps_final_over_source(pipeline):
    builder = ContextBuilder(pipeline, ContextConfig(max_tables=2))
    tables = builder.resolve_context_tables()
    assert tables == ["mart.users", "staging.users"]


def test_truncation_priority_discriminates_roles():
    # a_staging (intermediate) precedes z_mart (final) alphabetically, so the
    # old derived-first rule and plain alphabetical order would both keep
    # a_staging; role priority must keep the final table instead.
    pipeline = Pipeline.from_dict(
        {
            "build_staging": "CREATE TABLE a_staging.events AS SELECT id FROM raw.events",
            "build_mart": "CREATE TABLE z_mart.events AS SELECT id FROM a_staging.events",
        },
        dialect="bigquery",
    )
    builder = ContextBuilder(pipeline, ContextConfig(max_tables=1))
    assert builder.resolve_context_tables() == ["z_mart.events"]


def test_prompt_instruction_prefers_final_tables(pipeline):
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return "```sql\nSELECT 1\n```"

    GenerateSQLTool(pipeline, llm).run(question="emails?", include_explanation=False)
    assert "Prefer final tables" in prompts[-1]
