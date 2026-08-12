"""T3b: candidates only via identity-preserving lineage; aggregates disqualify."""

import pytest

from clgraph import Pipeline
from clgraph.tools import ContextBuilder, ContextConfig


@pytest.fixture
def two_marts_pipeline():
    return Pipeline.from_dict(
        {
            "mart_ids": "CREATE TABLE mart.user_ids AS SELECT id AS user_id FROM raw.users",
            "mart_emails": "CREATE TABLE mart.user_emails AS SELECT id AS uid, email FROM raw.users",
            "mart_counts": """
                CREATE TABLE mart.user_counts AS
                SELECT COUNT(id) AS user_count FROM raw.users
            """,
        },
        dialect="bigquery",
    )


def test_passthrough_shared_source_yields_candidate(two_marts_pipeline):
    # raw.users is included so the shared-source explanation is allowed to
    # name it (see test_candidate_hides_out_of_context_source for the case
    # where the source is NOT in the given table set).
    ctx = ContextBuilder(two_marts_pipeline).build_join_context(
        ["mart.user_ids", "mart.user_emails", "raw.users"]
    )
    assert "candidate:" in ctx
    # table names are sorted alphabetically in candidate pairs
    assert "mart.user_emails.uid = mart.user_ids.user_id" in ctx
    assert "raw.users.id" in ctx  # shared-source explanation


def test_aggregate_path_is_disqualified(two_marts_pipeline):
    ctx = ContextBuilder(two_marts_pipeline).build_join_context(
        ["mart.user_ids", "mart.user_counts"]
    )
    assert "user_count" not in ctx  # COUNT() path must never produce a candidate


def test_observed_joins_take_priority_under_cap():
    pipeline = Pipeline.from_dict(
        {
            "q": """
                CREATE TABLE mart.uo AS
                SELECT u.id FROM raw.users u JOIN raw.orders o ON u.id = o.user_id
            """,
            "mart_ids": "CREATE TABLE mart.user_ids AS SELECT id AS user_id FROM raw.users",
        },
        dialect="bigquery",
    )
    builder = ContextBuilder(pipeline, ContextConfig(max_join_hints=1))
    ctx = builder.build_join_context(list(pipeline.table_graph.tables))
    assert "(observed in q)" in ctx
    assert "candidate:" not in ctx  # cap consumed by the observed join


def test_ancestor_descendant_pair_not_proposed_as_candidate():
    """A table and its own lineage ancestor/descendant are never proposed as a
    candidate pair: that relationship is already visible via table lineage
    (derives_from), so it isn't a "hidden" join, and treating it as one would
    let the join-hints section reference tables outside the tables given to
    build_join_context (see tests/test_sql_tool_context.py, T1's invariant that
    every prompt section only references in-schema tables)."""
    pipeline = Pipeline.from_dict(
        {
            "staging": "CREATE TABLE staging.users AS SELECT id, email FROM raw.users",
            "mart": "CREATE TABLE mart.users AS SELECT id, email FROM staging.users",
        },
        dialect="bigquery",
    )
    ctx = ContextBuilder(pipeline).build_join_context(["mart.users", "staging.users"])
    assert "candidate:" not in ctx
    assert "raw.users" not in ctx


def test_candidate_hides_out_of_context_source(two_marts_pipeline):
    """When the shared ultimate source isn't part of the given table set, the
    candidate line still fires (mart.user_ids and mart.user_emails are true
    siblings, not ancestor/descendant) but must not name raw.users — doing so
    would violate the "sections only reference in-schema tables" invariant
    (tests/test_sql_tool_context.py)."""
    ctx = ContextBuilder(two_marts_pipeline).build_join_context(
        ["mart.user_ids", "mart.user_emails"]
    )
    assert "candidate:" in ctx
    assert "mart.user_emails.uid = mart.user_ids.user_id" in ctx
    assert "raw.users" not in ctx
    assert "(shared upstream key)" in ctx


def test_aliased_transform_is_disqualified():
    """UPPER(email) AS email_up is NOT identity-preserving even though its
    edge_type is "expression" (the same bucket a harmless rename like
    `id AS user_id` falls into) — the destination node's own expression must
    reduce to a bare column reference after stripping the alias."""
    pipeline = Pipeline.from_dict(
        {
            "mart_a": "CREATE TABLE mart.a AS SELECT UPPER(email) AS email_up FROM raw.users",
            "mart_b": "CREATE TABLE mart.b AS SELECT email FROM raw.users",
        },
        dialect="bigquery",
    )
    ctx = ContextBuilder(pipeline).build_join_context(["mart.a", "mart.b"])
    assert "email_up" not in ctx
