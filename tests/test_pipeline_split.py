"""
Tests for Pipeline.split() functionality.
"""

import pytest

from clgraph.pipeline import Pipeline

# Check if DuckDB is available
try:
    import duckdb  # noqa: F401

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


def test_pipeline_split_single_sinks():
    """Test splitting pipeline with single table sinks."""
    queries = [
        # Level 0
        ("source1", "CREATE TABLE source1 AS SELECT 1 as id"),
        ("source2", "CREATE TABLE source2 AS SELECT 2 as id"),
        # Level 1
        ("derived1", "CREATE TABLE derived1 AS SELECT * FROM source1"),
        ("derived2", "CREATE TABLE derived2 AS SELECT * FROM source2"),
        # Level 2
        ("final1", "CREATE TABLE final1 AS SELECT * FROM derived1"),
        ("final2", "CREATE TABLE final2 AS SELECT * FROM derived2"),
    ]

    pipeline = Pipeline(queries, dialect="bigquery")

    # Split into 2 subpipelines based on final tables
    subpipelines = pipeline.split(sinks=["final1", "final2"])

    # Should have 2 subpipelines
    assert len(subpipelines) == 2

    # Each subpipeline should have the right queries
    # Subpipeline 0: source1 -> derived1 -> final1
    sub1_query_ids = list(subpipelines[0].table_graph.queries.keys())
    assert set(sub1_query_ids) == {"source1", "derived1", "final1"}

    # Subpipeline 1: source2 -> derived2 -> final2
    sub2_query_ids = list(subpipelines[1].table_graph.queries.keys())
    assert set(sub2_query_ids) == {"source2", "derived2", "final2"}

    # Verify no overlapping queries
    assert set(sub1_query_ids).isdisjoint(set(sub2_query_ids))


def test_pipeline_split_with_shared_dependencies():
    """Test splitting pipeline where subpipelines share a dependency."""
    queries = [
        # Shared source
        ("shared_source", "CREATE TABLE shared AS SELECT 1 as id"),
        # Branch 1
        ("branch1", "CREATE TABLE branch1 AS SELECT * FROM shared"),
        ("final1", "CREATE TABLE final1 AS SELECT * FROM branch1"),
        # Branch 2
        ("branch2", "CREATE TABLE branch2 AS SELECT * FROM shared"),
        ("final2", "CREATE TABLE final2 AS SELECT * FROM branch2"),
    ]

    pipeline = Pipeline(queries, dialect="bigquery")

    # Split into 2 subpipelines
    subpipelines = pipeline.split(sinks=["final1", "final2"])

    assert len(subpipelines) == 2

    # First subpipeline should get the shared source
    # (since it's encountered first)
    sub1_queries = set(subpipelines[0].table_graph.queries.keys())
    sub2_queries = set(subpipelines[1].table_graph.queries.keys())

    # Verify shared_source is in first subpipeline
    assert "shared_source" in sub1_queries
    assert "branch1" in sub1_queries
    assert "final1" in sub1_queries

    # Second subpipeline has only its specific queries
    assert "branch2" in sub2_queries
    assert "final2" in sub2_queries
    assert "shared_source" not in sub2_queries  # No overlap

    # Verify non-overlapping
    assert sub1_queries.isdisjoint(sub2_queries)


def test_pipeline_split_with_grouped_sinks():
    """Test splitting with grouped sinks (multiple tables in one subpipeline)."""
    queries = [
        ("source", "CREATE TABLE source AS SELECT 1 as id"),
        ("derived1", "CREATE TABLE derived1 AS SELECT * FROM source"),
        ("derived2", "CREATE TABLE derived2 AS SELECT * FROM source"),
        ("final", "CREATE TABLE final AS SELECT * FROM source"),
    ]

    pipeline = Pipeline(queries, dialect="bigquery")

    # Group derived1 and derived2 together
    subpipelines = pipeline.split(sinks=[["derived1", "derived2"], "final"])

    assert len(subpipelines) == 2

    # First subpipeline has source, derived1, derived2
    sub1_queries = set(subpipelines[0].table_graph.queries.keys())
    assert "source" in sub1_queries
    assert "derived1" in sub1_queries
    assert "derived2" in sub1_queries

    # Second subpipeline has only final
    sub2_queries = set(subpipelines[1].table_graph.queries.keys())
    assert sub2_queries == {"final"}


def test_pipeline_split_error_on_invalid_sink():
    """Test that split raises error for non-existent sink table."""
    queries = [("query1", "CREATE TABLE table1 AS SELECT 1 as id")]
    pipeline = Pipeline(queries, dialect="bigquery")

    with pytest.raises(ValueError) as exc_info:
        pipeline.split(sinks=["non_existent_table"])

    assert "not found in pipeline" in str(exc_info.value)
    assert "Available tables" in str(exc_info.value)


@pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not installed")
def test_pipeline_split_integration_with_duckdb():
    """Test split subpipelines can be executed independently with DuckDB."""
    import duckdb

    queries = [
        # Pipeline A: orders -> customer_totals
        (
            "raw_orders",
            "CREATE TABLE orders AS SELECT 1 as id, 100.0 as amount UNION ALL SELECT 2, 200.0",
        ),
        (
            "customer_totals",
            "CREATE TABLE customer_totals AS SELECT SUM(amount) as total FROM orders",
        ),
        # Pipeline B: products -> product_summary
        (
            "raw_products",
            "CREATE TABLE products AS SELECT 1 as pid, 'Widget' as name",
        ),
        (
            "product_summary",
            "CREATE TABLE product_summary AS SELECT COUNT(*) as count FROM products",
        ),
    ]

    pipeline = Pipeline(queries, dialect="duckdb")

    # Split into 2 independent subpipelines
    subpipelines = pipeline.split(sinks=["customer_totals", "product_summary"])

    assert len(subpipelines) == 2

    # Execute first subpipeline
    conn1 = duckdb.connect(":memory:")

    def execute_sql1(sql: str):
        conn1.execute(sql)

    result1 = subpipelines[0].run(executor=execute_sql1, verbose=False)
    assert len(result1["completed"]) == 2  # raw_orders + customer_totals
    assert len(result1["failed"]) == 0

    # Verify customer_totals was created
    total = conn1.execute("SELECT total FROM customer_totals").fetchone()[0]
    assert total == 300.0

    # Execute second subpipeline in separate database
    conn2 = duckdb.connect(":memory:")

    def execute_sql2(sql: str):
        conn2.execute(sql)

    result2 = subpipelines[1].run(executor=execute_sql2, verbose=False)
    assert len(result2["completed"]) == 2  # raw_products + product_summary
    assert len(result2["failed"]) == 0

    # Verify product_summary was created
    count = conn2.execute("SELECT count FROM product_summary").fetchone()[0]
    assert count == 1

    conn1.close()
    conn2.close()


def test_pipeline_split_complex_dag():
    """Test split on a complex DAG with multiple levels."""
    queries = [
        # Level 0
        ("source1", "CREATE TABLE s1 AS SELECT 1 as id"),
        ("source2", "CREATE TABLE s2 AS SELECT 2 as id"),
        ("source3", "CREATE TABLE s3 AS SELECT 3 as id"),
        # Level 1
        ("derived1", "CREATE TABLE d1 AS SELECT * FROM s1, s2"),
        ("derived2", "CREATE TABLE d2 AS SELECT * FROM s2, s3"),
        # Level 2
        ("final1", "CREATE TABLE f1 AS SELECT * FROM d1"),
        ("final2", "CREATE TABLE f2 AS SELECT * FROM d2"),
    ]

    pipeline = Pipeline(queries, dialect="bigquery")

    # Split by final tables
    subpipelines = pipeline.split(sinks=["f1", "f2"])

    assert len(subpipelines) == 2

    # First subpipeline: s1, s2, d1, f1
    sub1_queries = set(subpipelines[0].table_graph.queries.keys())
    assert "source1" in sub1_queries
    assert "source2" in sub1_queries  # s2 is shared
    assert "derived1" in sub1_queries
    assert "final1" in sub1_queries

    # Second subpipeline: s3, d2, f2 (s2 already assigned to first)
    sub2_queries = set(subpipelines[1].table_graph.queries.keys())
    assert "source3" in sub2_queries
    assert "derived2" in sub2_queries
    assert "final2" in sub2_queries
    assert "source2" not in sub2_queries  # Already in first subpipeline

    # Verify non-overlapping
    assert sub1_queries.isdisjoint(sub2_queries)
