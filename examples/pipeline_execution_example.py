"""
Pipeline Execution Example

This example demonstrates the three execution modes for SQL pipelines:
1. Synchronous native execution with pipeline.run()
2. Asynchronous native execution with pipeline.async_run()
3. Airflow DAG generation with pipeline.to_airflow_dag()
"""

import asyncio
import time
from datetime import datetime

from clgraph.pipeline import Pipeline


# ============================================================================
# Example 1: Synchronous Native Execution
# ============================================================================
def demo_sync_execution():
    """Demonstrate synchronous pipeline execution."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Synchronous Native Execution (pipeline.run())")
    print("=" * 70 + "\n")

    # Define a simple pipeline
    queries = [
        (
            "staging_orders",
            """
            CREATE TABLE staging.orders AS
            SELECT
                order_id,
                customer_id,
                order_date,
                amount
            FROM raw.orders
            WHERE order_date >= CURRENT_DATE() - 7
        """,
        ),
        (
            "daily_revenue",
            """
            CREATE TABLE analytics.daily_revenue AS
            SELECT
                DATE(order_date) as date,
                COUNT(*) as order_count,
                SUM(amount) as total_revenue
            FROM staging.orders
            GROUP BY DATE(order_date)
        """,
        ),
        (
            "customer_metrics",
            """
            CREATE TABLE analytics.customer_metrics AS
            SELECT
                customer_id,
                COUNT(*) as order_count,
                SUM(amount) as total_spend
            FROM staging.orders
            GROUP BY customer_id
        """,
        ),
    ]

    # Create pipeline
    pipeline = Pipeline(queries, dialect="bigquery")

    # Mock executor for demo (in real use, this would execute SQL on your database)
    def execute_sql(sql: str):
        """Mock executor - simulates SQL execution"""
        # Extract table name from CREATE TABLE statement
        if "CREATE TABLE" in sql:
            table_start = sql.find("CREATE TABLE") + 13
            table_end = sql.find("AS", table_start)
            table_name = sql[table_start:table_end].strip()
            print(f"      [Executing] {table_name}")
            time.sleep(0.1)  # Simulate query execution time

    # Run pipeline
    result = pipeline.run(executor=execute_sql, max_workers=4, verbose=True)

    print("\nResult Summary:")
    print(f"  Completed: {len(result['completed'])} queries")
    print(f"  Failed: {len(result['failed'])} queries")
    print(f"  Total time: {result['elapsed_seconds']:.2f}s")


# ============================================================================
# Example 2: Asynchronous Native Execution
# ============================================================================
async def demo_async_execution():
    """Demonstrate asynchronous pipeline execution."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Asynchronous Native Execution (pipeline.async_run())")
    print("=" * 70 + "\n")

    # Same queries as before
    queries = [
        (
            "staging_orders",
            "CREATE TABLE staging.orders AS SELECT * FROM raw.orders",
        ),
        (
            "daily_revenue",
            "CREATE TABLE analytics.daily_revenue AS SELECT DATE(order_date) as date, SUM(amount) FROM staging.orders GROUP BY 1",
        ),
        (
            "customer_metrics",
            "CREATE TABLE analytics.customer_metrics AS SELECT customer_id, COUNT(*) FROM staging.orders GROUP BY 1",
        ),
    ]

    pipeline = Pipeline(queries, dialect="bigquery")

    # Async executor
    async def async_execute_sql(sql: str):
        """Async mock executor"""
        if "CREATE TABLE" in sql:
            table_start = sql.find("CREATE TABLE") + 13
            table_end = sql.find("AS", table_start)
            table_name = sql[table_start:table_end].strip()
            print(f"      [Async Executing] {table_name}")
            await asyncio.sleep(0.1)  # Simulate async query execution

    # Run async
    result = await pipeline.async_run(executor=async_execute_sql, max_workers=4, verbose=True)

    print("\nAsync Result Summary:")
    print(f"  Completed: {len(result['completed'])} queries")
    print(f"  Failed: {len(result['failed'])} queries")
    print(f"  Total time: {result['elapsed_seconds']:.2f}s")


# ============================================================================
# Example 3: Airflow DAG Generation
# ============================================================================
def demo_airflow_dag():
    """Demonstrate Airflow DAG generation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Airflow DAG Generation (pipeline.to_airflow_dag())")
    print("=" * 70 + "\n")

    # Same queries
    queries = [
        (
            "staging_orders",
            "CREATE TABLE staging.orders AS SELECT * FROM raw.orders",
        ),
        (
            "daily_revenue",
            "CREATE TABLE analytics.daily_revenue AS SELECT DATE(order_date) as date, SUM(amount) FROM staging.orders GROUP BY 1",
        ),
    ]

    pipeline = Pipeline(queries, dialect="bigquery")

    # Executor function (would execute SQL in production)
    def execute_sql(sql: str):
        """Executor for Airflow tasks"""
        # In production, this would connect to BigQuery/Snowflake/etc
        # from google.cloud import bigquery
        # client = bigquery.Client()
        # client.query(sql).result()
        print(f"Would execute: {sql[:50]}...")

    # Generate Airflow DAG with advanced parameters
    try:
        dag = pipeline.to_airflow_dag(
            executor=execute_sql,
            dag_id="revenue_pipeline",
            schedule="@daily",
            start_date=datetime(2024, 1, 1),
            # Advanced DAG parameters (all optional)
            description="Daily revenue analytics pipeline",
            catchup=False,
            max_active_runs=3,
            max_active_tasks=10,
            tags=["analytics", "revenue", "daily"],
            default_view="graph",
            orientation="LR",
        )

        print("✅ Airflow DAG generated successfully!")
        print(f"   DAG ID: {dag.dag_id}")
        print(f"   Description: {dag.description}")
        print("   Schedule: @daily")
        print(f"   Max Active Runs: {dag.max_active_runs}")
        print(f"   Tags: {dag.tags}")
        print(f"   Tasks: {len(dag.task_dict)} tasks")
        print(f"   Task IDs: {list(dag.task_dict.keys())}")
        print("\nTo use this DAG:")
        print("1. Save this to your Airflow dags/ folder")
        print("2. Airflow will automatically detect and schedule it")
        print("3. View in Airflow UI: http://localhost:8080/dags/revenue_pipeline")
        print("\nNote: to_airflow_dag() supports ALL Airflow DAG parameters via **kwargs")

    except ImportError:
        print("⚠️  Airflow not installed")
        print("   Install with: pip install apache-airflow")
        print("   This is optional - you can still use run() and async_run()")


# ============================================================================
# Main Demo Runner
# ============================================================================
def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("PHASE 4 EXECUTION ADAPTERS DEMO")
    print("Write Once, Run Anywhere: Airflow or Native Python")
    print("=" * 70)

    # Demo 1: Sync execution
    demo_sync_execution()

    # Demo 2: Async execution
    asyncio.run(demo_async_execution())

    # Demo 3: Airflow DAG
    demo_airflow_dag()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
