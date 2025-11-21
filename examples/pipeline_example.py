"""
Example demonstrating multi-query pipeline lineage analysis
"""

from clpipe import Pipeline

# Example: Data pipeline with multiple dependent queries
queries = [
    # Stage 1: Raw data ingestion
    (
        "raw_events",
        """
        CREATE TABLE raw_events AS
        SELECT
            user_id,
            event_type,
            event_timestamp,
            session_id,
            page_url
        FROM source_events
        WHERE event_timestamp >= '2024-01-01'
    """,
    ),
    # Stage 2: Daily aggregation
    (
        "daily_active_users",
        """
        CREATE TABLE daily_active_users AS
        SELECT
            user_id,
            DATE(event_timestamp) as activity_date,
            COUNT(DISTINCT session_id) as session_count,
            COUNT(*) as event_count
        FROM raw_events
        GROUP BY user_id, DATE(event_timestamp)
    """,
    ),
    # Stage 3: User enrichment
    (
        "user_activity_summary",
        """
        CREATE TABLE user_activity_summary AS
        SELECT
            u.name,
            u.email,
            u.signup_date,
            dau.activity_date,
            dau.session_count,
            dau.event_count,
            DATE_DIFF(dau.activity_date, u.signup_date, DAY) as days_since_signup
        FROM users u
        JOIN daily_active_users dau ON u.id = dau.user_id
    """,
    ),
    # Stage 4: Analytics table
    (
        "user_engagement_metrics",
        """
        CREATE TABLE user_engagement_metrics AS
        SELECT
            name,
            email,
            COUNT(*) as active_days,
            SUM(session_count) as total_sessions,
            SUM(event_count) as total_events,
            AVG(session_count) as avg_daily_sessions
        FROM user_activity_summary
        GROUP BY name, email
        HAVING COUNT(*) >= 7
    """,
    ),
]


def main():
    print("=" * 80)
    print("SQL Pipeline Lineage Example")
    print("=" * 80)
    print()

    # Parse the multi-query pipeline using Pipeline class
    print("Building pipeline...")
    pipeline = Pipeline(queries, dialect="bigquery")

    print(f"✓ Built pipeline with {len(pipeline.table_graph.queries)} queries")
    print(f"✓ Found {len(pipeline.columns)} columns")
    print()

    # Example 1: Table execution order
    print("1. TABLE EXECUTION ORDER")
    print("-" * 80)
    execution_order = pipeline.table_graph.get_execution_order()
    for i, table in enumerate(execution_order, 1):
        table_node = pipeline.table_graph.tables[str(table)]
        if table_node.is_source:
            print(f"{i}. {table} (source table)")
        else:
            # Find what this table depends on
            query = pipeline.table_graph.queries.get(table_node.created_by)
            if query and query.source_tables:
                print(f"{i}. {table}")
                print(f"   Depends on: {', '.join(sorted(query.source_tables))}")
            else:
                print(f"{i}. {table}")
    print()

    # Example 2: Trace a column through the pipeline
    print("2. BACKWARD LINEAGE (End-to-end column tracing)")
    print("-" * 80)
    target_col = "user_engagement_metrics.total_events"
    print(f"Column: {target_col}")
    sources = pipeline.trace_column_backward("user_engagement_metrics", "total_events")
    print("Traces back to:")
    for source in sources:
        print(f"  → {source.table_name}.{source.column_name}")
    print()

    # Example 3: Forward lineage / Impact analysis
    print("3. FORWARD LINEAGE (Impact analysis)")
    print("-" * 80)
    print("Column: source_events.event_timestamp")
    impacts = pipeline.trace_column_forward("source_events", "event_timestamp")
    print("Used in:")
    for impact in impacts:
        print(f"  → {impact.table_name}.{impact.column_name}")
    print()

    # Example 4: Show all columns in final table
    print("4. FINAL TABLE COLUMNS")
    print("-" * 80)
    print("Table: user_engagement_metrics")
    final_cols = [
        col for col in pipeline.columns.values() if col.table_name == "user_engagement_metrics"
    ]
    for col in sorted(final_cols, key=lambda c: c.column_name):
        print(f"  • {col.column_name}: {col.expression}")
    print()

    # Example 5: Find all PII columns
    print("5. COLUMN METADATA EXAMPLE")
    print("-" * 80)
    # Mark email as PII
    if "user_engagement_metrics.email" in pipeline.columns:
        pipeline.columns["user_engagement_metrics.email"].pii = True
        pipeline.columns["user_engagement_metrics.email"].owner = "privacy_team"

    # Propagate metadata
    pipeline.propagate_all_metadata()

    # Find all PII columns
    pii_cols = pipeline.get_pii_columns()
    print("PII columns in pipeline:")
    for col in pii_cols:
        print(f"  ⚠️  {col}")
    print()

    print("=" * 80)
    print("Pipeline analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
