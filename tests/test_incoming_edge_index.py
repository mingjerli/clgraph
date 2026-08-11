"""Equivalence tests: public index lookup == linear edge scan."""

import pytest

from clgraph import Pipeline


@pytest.fixture
def pipeline():
    return Pipeline.from_dict(
        {
            "staging_users": """
                CREATE TABLE staging.users AS
                SELECT id, name, email FROM raw.users
            """,
            "user_metrics": """
                CREATE TABLE analytics.user_metrics AS
                SELECT u.id AS user_id, COUNT(*) AS order_count
                FROM staging.users u
                JOIN raw.orders o ON u.id = o.user_id
                GROUP BY u.id
            """,
        },
        dialect="bigquery",
    )


def test_public_accessor_exists(pipeline):
    assert callable(pipeline.get_incoming_edges)


def test_index_matches_linear_scan_for_every_column(pipeline):
    for col in pipeline.columns.values():
        scanned = {
            (e.from_node.full_name, e.to_node.full_name, e.edge_type)
            for e in pipeline.edges
            if e.to_node == col
        }
        indexed = {
            (e.from_node.full_name, e.to_node.full_name, e.edge_type)
            for e in pipeline.get_incoming_edges(col.full_name)
        }
        assert indexed == scanned, f"index mismatch for {col.full_name}"


def test_unknown_column_returns_empty_list(pipeline):
    assert pipeline.get_incoming_edges("no.such:column") == []
