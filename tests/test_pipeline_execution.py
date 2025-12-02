"""
Tests for pipeline execution adapters (run, async_run, to_airflow_dag).
"""

import asyncio
import time

import pytest

from clgraph.pipeline import Pipeline

# Check if DuckDB is available
try:
    import duckdb  # noqa: F401

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# Check if Airflow is available
try:
    import airflow  # noqa: F401

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


def test_pipeline_run_simple():
    """Test synchronous pipeline execution with simple queries."""
    # Define simple queries with dependencies
    queries = [
        ("staging", "CREATE TABLE staging AS SELECT 1 as id, 'Alice' as name"),
        (
            "analytics",
            "CREATE TABLE analytics AS SELECT id, name FROM staging WHERE id = 1",
        ),
    ]

    # Create pipeline
    pipeline = Pipeline(queries, dialect="bigquery")

    # Track executed queries
    executed_queries = []

    def mock_executor(sql: str):
        """Mock executor that tracks what was executed"""
        executed_queries.append(sql)
        time.sleep(0.1)  # Simulate query execution time

    # Run pipeline
    result = pipeline.run(executor=mock_executor, max_workers=2, verbose=False)

    # Verify results
    assert len(result["completed"]) == 2
    assert len(result["failed"]) == 0
    assert result["total_queries"] == 2
    assert result["elapsed_seconds"] > 0

    # Verify execution order (staging must execute before analytics)
    assert len(executed_queries) == 2
    staging_idx = next(i for i, sql in enumerate(executed_queries) if "staging" in sql.lower())
    analytics_idx = next(i for i, sql in enumerate(executed_queries) if "analytics" in sql.lower())
    assert staging_idx < analytics_idx, "staging should execute before analytics"


def test_pipeline_run_concurrent_execution():
    """Test that independent queries run concurrently."""
    # Define queries with no dependencies (can run in parallel)
    queries = [
        ("table1", "CREATE TABLE table1 AS SELECT 1 as id"),
        ("table2", "CREATE TABLE table2 AS SELECT 2 as id"),
        ("table3", "CREATE TABLE table3 AS SELECT 3 as id"),
    ]

    pipeline = Pipeline(queries, dialect="bigquery")

    execution_times = {}

    def mock_executor(sql: str):
        """Mock executor that tracks execution times"""
        start = time.time()
        time.sleep(0.2)  # Simulate query execution
        execution_times[sql] = (start, time.time())

    result = pipeline.run(executor=mock_executor, max_workers=3, verbose=False)

    # Verify all queries completed
    assert len(result["completed"]) == 3
    assert len(result["failed"]) == 0

    # Verify concurrent execution: total time should be < sum of individual times
    # If sequential: 3 * 0.2 = 0.6s
    # If concurrent: ~0.2s (with some overhead)
    assert result["elapsed_seconds"] < 0.5, "Queries should execute concurrently, not sequentially"


def test_pipeline_run_error_handling():
    """Test that errors are properly captured and reported."""
    queries = [
        ("query1", "CREATE TABLE table1 AS SELECT 1 as id"),
        ("query2", "CREATE TABLE table2 AS SELECT * FROM table1"),
    ]

    pipeline = Pipeline(queries, dialect="bigquery")

    def failing_executor(sql: str):
        """Executor that fails on second query"""
        if "table2" in sql:
            raise RuntimeError("Simulated query failure")

    result = pipeline.run(executor=failing_executor, max_workers=1, verbose=False)

    # Verify results
    assert len(result["completed"]) == 1
    assert len(result["failed"]) == 1
    assert result["failed"][0][0] == "query2"
    assert "Simulated query failure" in result["failed"][0][1]


def test_pipeline_async_run():
    """Test asynchronous pipeline execution."""

    async def run_test():
        queries = [
            ("staging", "CREATE TABLE staging AS SELECT 1 as id"),
            ("analytics", "CREATE TABLE analytics AS SELECT id FROM staging"),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        executed_queries = []

        async def async_executor(sql: str):
            """Async mock executor"""
            executed_queries.append(sql)
            await asyncio.sleep(0.1)  # Simulate async query execution

        result = await pipeline.async_run(executor=async_executor, max_workers=2, verbose=False)

        # Verify results
        assert len(result["completed"]) == 2
        assert len(result["failed"]) == 0
        assert len(executed_queries) == 2

    # Run the async test
    asyncio.run(run_test())


def test_pipeline_async_run_concurrent():
    """Test async concurrent execution."""

    async def run_test():
        queries = [
            ("table1", "CREATE TABLE table1 AS SELECT 1 as id"),
            ("table2", "CREATE TABLE table2 AS SELECT 2 as id"),
            ("table3", "CREATE TABLE table3 AS SELECT 3 as id"),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        async def async_executor(sql: str):
            """Async executor with delay"""
            await asyncio.sleep(0.2)

        result = await pipeline.async_run(executor=async_executor, max_workers=3, verbose=False)

        # Verify concurrent execution
        assert len(result["completed"]) == 3
        assert result["elapsed_seconds"] < 0.5, "Async queries should execute concurrently"

    # Run the async test
    asyncio.run(run_test())


def test_pipeline_execution_levels():
    """Test that execution levels are correctly computed."""
    queries = [
        # Level 0: No dependencies
        ("source1", "CREATE TABLE source1 AS SELECT 1 as id"),
        ("source2", "CREATE TABLE source2 AS SELECT 2 as id"),
        # Level 1: Depends on source1
        ("derived1", "CREATE TABLE derived1 AS SELECT * FROM source1"),
        # Level 2: Depends on derived1
        ("final", "CREATE TABLE final AS SELECT * FROM derived1"),
    ]

    pipeline = Pipeline(queries, dialect="bigquery")

    levels = pipeline._get_execution_levels()

    # Verify level structure
    assert len(levels) == 3

    # Level 0 should have source1 and source2 (no dependencies)
    assert set(levels[0]) == {"source1", "source2"}

    # Level 1 should have derived1 (depends on source1)
    assert levels[1] == ["derived1"]

    # Level 2 should have final (depends on derived1)
    assert levels[2] == ["final"]


def test_to_airflow_dag_requires_airflow():
    """Test that to_airflow_dag raises error when Airflow is not installed."""
    queries = [("query1", "CREATE TABLE table1 AS SELECT 1 as id")]
    pipeline = Pipeline(queries, dialect="bigquery")

    def mock_executor(sql: str):
        pass

    # This test will only fail if airflow is not installed
    # If airflow IS installed, we'll just verify the DAG is created
    try:
        dag = pipeline.to_airflow_dag(executor=mock_executor, dag_id="test_dag")
        # If we get here, airflow is installed
        assert dag is not None
        assert dag.dag_id == "test_dag"
    except ImportError as e:
        # Expected if airflow not installed
        assert "Airflow is required" in str(e)


@pytest.mark.skipif(not AIRFLOW_AVAILABLE, reason="Airflow not installed")
def test_to_airflow_dag_with_all_parameters():
    """Test that to_airflow_dag supports all DAG parameters via **kwargs."""
    queries = [("query1", "CREATE TABLE table1 AS SELECT 1 as id")]
    pipeline = Pipeline(queries, dialect="bigquery")

    def mock_executor(sql: str):
        pass

    # Test with comprehensive DAG parameters
    dag = pipeline.to_airflow_dag(
        executor=mock_executor,
        dag_id="test_dag_advanced",
        schedule="0 0 * * *",  # Cron expression
        description="Test pipeline with advanced configuration",
        catchup=True,  # Override default
        max_active_runs=5,
        max_active_tasks=20,
        tags=["test", "analytics", "daily"],  # Override default tags
        default_view="graph",
        orientation="LR",
    )

    # Verify DAG was created with custom parameters
    assert dag is not None
    assert dag.dag_id == "test_dag_advanced"
    assert dag.description == "Test pipeline with advanced configuration"
    assert dag.catchup is True
    assert dag.max_active_runs == 5
    assert dag.max_active_tasks == 20
    assert set(dag.tags) == {"test", "analytics", "daily"}
    assert dag.default_view == "graph"
    assert dag.orientation == "LR"
    assert dag.schedule_interval == "0 0 * * *"


def test_pipeline_run_with_verbose():
    """Test that verbose output doesn't break execution."""
    queries = [("query1", "CREATE TABLE table1 AS SELECT 1 as id")]
    pipeline = Pipeline(queries, dialect="bigquery")

    def mock_executor(sql: str):
        pass

    # Run with verbose=True (should print to stdout)
    result = pipeline.run(executor=mock_executor, verbose=True)

    assert len(result["completed"]) == 1
    assert len(result["failed"]) == 0


# ============================================================================
# DuckDB Integration Tests
# ============================================================================


@pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not installed")
def test_pipeline_run_with_duckdb():
    """Test pipeline execution with real DuckDB database."""
    import duckdb

    # Create a simple pipeline with dependencies
    queries = [
        (
            "source_data",
            """
            CREATE TABLE raw_orders AS
            SELECT
                1 as order_id,
                'Alice' as customer_name,
                100.0 as amount
            UNION ALL
            SELECT 2, 'Bob', 200.0
            UNION ALL
            SELECT 3, 'Alice', 150.0
        """,
        ),
        (
            "customer_totals",
            """
            CREATE TABLE customer_totals AS
            SELECT
                customer_name,
                COUNT(*) as order_count,
                SUM(amount) as total_amount
            FROM raw_orders
            GROUP BY customer_name
        """,
        ),
    ]

    pipeline = Pipeline(queries, dialect="duckdb")

    # Create DuckDB connection
    conn = duckdb.connect(":memory:")

    def execute_sql(sql: str):
        """Execute SQL on DuckDB"""
        conn.execute(sql)

    # Run pipeline
    result = pipeline.run(executor=execute_sql, max_workers=2, verbose=False)

    # Verify execution
    assert len(result["completed"]) == 2
    assert len(result["failed"]) == 0

    # Verify data was created correctly
    raw_orders = conn.execute("SELECT COUNT(*) FROM raw_orders").fetchone()[0]
    assert raw_orders == 3

    # Verify aggregation worked
    customer_data = conn.execute(
        "SELECT customer_name, order_count, total_amount FROM customer_totals ORDER BY customer_name"
    ).fetchall()

    assert len(customer_data) == 2
    # Alice: 2 orders, 250.0 total
    assert customer_data[0] == ("Alice", 2, 250.0)
    # Bob: 1 order, 200.0 total
    assert customer_data[1] == ("Bob", 1, 200.0)

    conn.close()


@pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not installed")
def test_pipeline_async_run_with_duckdb():
    """Test async pipeline execution with DuckDB."""
    import duckdb

    async def run_test():
        queries = [
            (
                "products",
                """
                CREATE TABLE products AS
                SELECT 1 as product_id, 'Widget' as name, 10.0 as price
                UNION ALL
                SELECT 2, 'Gadget', 20.0
            """,
            ),
            (
                "expensive_products",
                """
                CREATE TABLE expensive_products AS
                SELECT * FROM products WHERE price > 15.0
            """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="duckdb")

        # Create DuckDB connection
        conn = duckdb.connect(":memory:")

        async def async_execute_sql(sql: str):
            """Async wrapper for DuckDB execution"""
            await asyncio.sleep(0.01)  # Simulate async operation
            conn.execute(sql)

        # Run async
        result = await pipeline.async_run(executor=async_execute_sql, max_workers=2, verbose=False)

        # Verify execution
        assert len(result["completed"]) == 2
        assert len(result["failed"]) == 0

        # Verify filtering worked
        expensive_count = conn.execute("SELECT COUNT(*) FROM expensive_products").fetchone()[0]
        assert expensive_count == 1

        expensive_product = conn.execute("SELECT name FROM expensive_products").fetchone()[0]
        assert expensive_product == "Gadget"

        conn.close()

    asyncio.run(run_test())


@pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not installed")
def test_pipeline_complex_duckdb_workflow():
    """Test a more complex pipeline with multiple levels and joins."""
    import duckdb

    queries = [
        # Level 0: Create source tables
        (
            "raw_orders",
            """
            CREATE TABLE raw_orders AS
            SELECT 1 as order_id, 101 as customer_id, '2024-01-01' as order_date, 100.0 as amount
            UNION ALL
            SELECT 2, 102, '2024-01-02', 200.0
            UNION ALL
            SELECT 3, 101, '2024-01-03', 150.0
        """,
        ),
        (
            "raw_customers",
            """
            CREATE TABLE raw_customers AS
            SELECT 101 as customer_id, 'Alice' as name, 'Premium' as tier
            UNION ALL
            SELECT 102, 'Bob', 'Standard'
        """,
        ),
        # Level 1: Join orders with customers
        (
            "enriched_orders",
            """
            CREATE TABLE enriched_orders AS
            SELECT
                o.order_id,
                o.customer_id,
                c.name as customer_name,
                c.tier as customer_tier,
                o.order_date,
                o.amount
            FROM raw_orders o
            JOIN raw_customers c ON o.customer_id = c.customer_id
        """,
        ),
        # Level 2: Aggregate by customer
        (
            "customer_summary",
            """
            CREATE TABLE customer_summary AS
            SELECT
                customer_id,
                customer_name,
                customer_tier,
                COUNT(*) as total_orders,
                SUM(amount) as total_spent
            FROM enriched_orders
            GROUP BY customer_id, customer_name, customer_tier
        """,
        ),
        # Level 2: Aggregate by tier (independent of customer_summary)
        (
            "tier_summary",
            """
            CREATE TABLE tier_summary AS
            SELECT
                customer_tier,
                COUNT(DISTINCT customer_id) as customer_count,
                COUNT(*) as total_orders,
                SUM(amount) as total_revenue
            FROM enriched_orders
            GROUP BY customer_tier
        """,
        ),
    ]

    pipeline = Pipeline(queries, dialect="duckdb")

    # Verify execution levels are correct
    levels = pipeline._get_execution_levels()
    assert len(levels) == 3
    # Level 0: raw_orders and raw_customers (can run in parallel)
    assert set(levels[0]) == {"raw_orders", "raw_customers"}
    # Level 1: enriched_orders (depends on both sources)
    assert levels[1] == ["enriched_orders"]
    # Level 2: customer_summary and tier_summary (can run in parallel)
    assert set(levels[2]) == {"customer_summary", "tier_summary"}

    # Execute pipeline
    conn = duckdb.connect(":memory:")

    def execute_sql(sql: str):
        conn.execute(sql)

    result = pipeline.run(executor=execute_sql, max_workers=4, verbose=False)

    # Verify all queries completed
    assert len(result["completed"]) == 5
    assert len(result["failed"]) == 0

    # Verify customer summary
    customer_summary = conn.execute(
        "SELECT customer_name, total_orders, total_spent FROM customer_summary ORDER BY customer_name"
    ).fetchall()

    assert len(customer_summary) == 2
    assert customer_summary[0] == ("Alice", 2, 250.0)  # 2 orders, 100 + 150
    assert customer_summary[1] == ("Bob", 1, 200.0)  # 1 order, 200

    # Verify tier summary
    tier_summary = conn.execute(
        "SELECT customer_tier, customer_count, total_orders, total_revenue FROM tier_summary ORDER BY customer_tier"
    ).fetchall()

    assert len(tier_summary) == 2
    assert tier_summary[0] == ("Premium", 1, 2, 250.0)  # Alice (Premium)
    assert tier_summary[1] == ("Standard", 1, 1, 200.0)  # Bob (Standard)

    conn.close()


@pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not installed")
def test_pipeline_duckdb_with_error_recovery():
    """Test that pipeline handles SQL errors gracefully with DuckDB."""
    import duckdb

    queries = [
        ("valid_table", "CREATE TABLE valid_table AS SELECT 1 as id"),
        (
            "invalid_table",
            "CREATE TABLE invalid_table AS SELECT * FROM non_existent_table",
        ),
        (
            "depends_on_valid",
            "CREATE TABLE depends_on_valid AS SELECT * FROM valid_table",
        ),
    ]

    pipeline = Pipeline(queries, dialect="duckdb")
    conn = duckdb.connect(":memory:")

    def execute_sql(sql: str):
        conn.execute(sql)

    result = pipeline.run(executor=execute_sql, max_workers=2, verbose=False)

    # Should have 2 completed (valid_table and depends_on_valid)
    # and 1 failed (invalid_table)
    assert len(result["completed"]) == 2
    assert len(result["failed"]) == 1
    assert result["failed"][0][0] == "invalid_table"

    # Verify valid tables were created
    assert conn.execute("SELECT COUNT(*) FROM valid_table").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM depends_on_valid").fetchone()[0] == 1

    conn.close()
