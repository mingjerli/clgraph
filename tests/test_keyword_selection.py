"""T5: lexical scoring boosted by graph neighborhood; deterministic padding."""

import pytest

from clgraph import Pipeline
from clgraph.tools import ContextBuilder


@pytest.fixture
def pipeline():
    return Pipeline.from_dict(
        {
            "mart_revenue": """
                CREATE TABLE mart.revenue AS
                SELECT o.amount FROM raw.orders o
            """,
            "unrelated": "CREATE TABLE misc.zzz_audit AS SELECT ts FROM raw.logs",
        },
        dialect="bigquery",
    )


def test_parent_of_matched_table_is_boosted(pipeline):
    selected = ContextBuilder(pipeline).select_tables_by_keywords(
        "total revenue", min_tables=2, max_tables=2
    )
    assert selected[0] == "mart.revenue"
    # raw.orders has zero lexical overlap with "total revenue" but is the
    # parent of the matched mart — diffusion must rank it above raw.logs/misc.
    assert selected[1] == "raw.orders"


def test_padding_is_deterministic(pipeline):
    builder = ContextBuilder(pipeline)
    first = builder.select_tables_by_keywords("nothing matches this", min_tables=3)
    second = builder.select_tables_by_keywords("nothing matches this", min_tables=3)
    assert first == second


def test_single_table_graph_unchanged():
    pipeline = Pipeline.from_dict(
        {"q": "CREATE TABLE only.table AS SELECT 1 AS x"}, dialect="bigquery"
    )
    selected = ContextBuilder(pipeline).select_tables_by_keywords("x", min_tables=1)
    assert selected == ["only.table"]


def test_diffusion_discriminates_from_insertion_order():
    # The unrelated query is defined FIRST, so its tables enter the table
    # graph before raw.orders. Old insertion-order padding would therefore
    # pick misc.request_log second; score diffusion must rank raw.orders
    # (parent of the lexically matched mart) second instead.
    pipeline = Pipeline.from_dict(
        {
            "unrelated": "CREATE TABLE misc.request_log AS SELECT ts FROM raw.logs",
            "mart_revenue": "CREATE TABLE mart.revenue AS SELECT o.amount FROM raw.orders o",
        },
        dialect="bigquery",
    )
    selected = ContextBuilder(pipeline).select_tables_by_keywords(
        "total revenue", min_tables=2, max_tables=2
    )
    assert selected == ["mart.revenue", "raw.orders"]


def test_padding_prefers_final_tables_over_insertion_order():
    # No lexical match at all: padding must take final tables (alphabetical)
    # before sources, not dict-insertion order (which would lead with
    # misc.request_log's earlier-parsed table set).
    pipeline = Pipeline.from_dict(
        {
            "unrelated": "CREATE TABLE misc.request_log AS SELECT ts FROM raw.logs",
            "mart_revenue": "CREATE TABLE mart.revenue AS SELECT o.amount FROM raw.orders o",
        },
        dialect="bigquery",
    )
    selected = ContextBuilder(pipeline).select_tables_by_keywords(
        "nothing matches this", min_tables=3
    )
    assert selected == ["mart.revenue", "misc.request_log", "raw.logs"]
