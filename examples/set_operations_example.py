"""
Example demonstrating SQL column lineage analysis with set operations.

This example shows how clpipe analyzes UNION, INTERSECT, and EXCEPT operations,
tracking column lineage across multiple query branches.
"""

from clpipe import SQLColumnTracer


def union_all_example():
    """Example: UNION ALL combining active and archived users"""
    print("=" * 80)
    print("Example 1: UNION ALL - Combining Active and Archived Users")
    print("=" * 80)
    print()

    sql = """
    SELECT user_id, name, email, 'active' as status
    FROM active_users
    WHERE last_login > '2024-01-01'

    UNION ALL

    SELECT user_id, name, email, 'archived' as status
    FROM archived_users
    WHERE archived_date > '2023-01-01'
    """

    print("SQL Query:")
    print(sql)
    print()

    tracer = SQLColumnTracer(sql, dialect="bigquery")
    lineage = tracer.build_column_lineage_graph()

    print("Analysis Results:")
    print("-" * 80)
    print(f"Total nodes: {len(lineage.nodes)}")
    print(f"Total edges: {len(lineage.edges)}")
    print()

    print("Output Columns:")
    for col in lineage.get_output_nodes():
        print(f"  • {col.column_name}")
    print()

    print("Source Tables:")
    input_nodes = lineage.get_input_nodes()
    source_tables = {node.table_name for node in input_nodes if node.table_name}
    for source in sorted(source_tables):
        print(f"  • {source}")
    print()


def union_distinct_example():
    """Example: UNION DISTINCT for deduplication"""
    print("=" * 80)
    print("Example 2: UNION DISTINCT - Deduplicating User IDs")
    print("=" * 80)
    print()

    sql = """
    SELECT user_id FROM orders WHERE order_date >= '2024-01-01'

    UNION DISTINCT

    SELECT user_id FROM subscriptions WHERE active = true
    """

    print("SQL Query:")
    print(sql)
    print()

    tracer = SQLColumnTracer(sql, dialect="bigquery")
    lineage = tracer.build_column_lineage_graph()

    print("Analysis Results:")
    print("-" * 80)
    print(f"Total nodes: {len(lineage.nodes)}")
    print(f"Total edges: {len(lineage.edges)}")
    print()


def three_way_union_example():
    """Example: Three-way UNION with different sources"""
    print("=" * 80)
    print("Example 3: Three-way UNION - Multiple Data Sources")
    print("=" * 80)
    print()

    sql = """
    SELECT
        transaction_id,
        user_id,
        amount,
        'online' as channel
    FROM online_transactions

    UNION ALL

    SELECT
        transaction_id,
        user_id,
        amount,
        'mobile' as channel
    FROM mobile_transactions

    UNION ALL

    SELECT
        transaction_id,
        user_id,
        amount,
        'in_store' as channel
    FROM in_store_transactions
    """

    print("SQL Query:")
    print(sql)
    print()

    tracer = SQLColumnTracer(sql, dialect="bigquery")
    lineage = tracer.build_column_lineage_graph()

    print("Analysis Results:")
    print("-" * 80)
    print(f"Total nodes: {len(lineage.nodes)}")
    print(f"Total edges: {len(lineage.edges)}")
    print()

    print("Output Columns:")
    for col in lineage.get_output_nodes():
        print(f"  • {col.column_name}")
    print()

    print("Source Tables (3 branches):")
    input_nodes = lineage.get_input_nodes()
    source_tables = {node.table_name for node in input_nodes if node.table_name}
    for source in sorted(source_tables):
        print(f"  • {source}")
    print()


def intersect_example():
    """Example: INTERSECT to find common elements"""
    print("=" * 80)
    print("Example 4: INTERSECT - Finding Users in Both Sets")
    print("=" * 80)
    print()

    sql = """
    SELECT user_id, name
    FROM premium_subscribers

    INTERSECT DISTINCT

    SELECT user_id, name
    FROM active_forum_members
    """

    print("SQL Query:")
    print(sql)
    print()

    tracer = SQLColumnTracer(sql, dialect="bigquery")
    lineage = tracer.build_column_lineage_graph()

    print("Analysis Results:")
    print("-" * 80)
    print(f"Total nodes: {len(lineage.nodes)}")
    print(f"Total edges: {len(lineage.edges)}")
    print()


def except_example():
    """Example: EXCEPT to find differences"""
    print("=" * 80)
    print("Example 5: EXCEPT - Finding Users Not in Second Set")
    print("=" * 80)
    print()

    sql = """
    SELECT user_id, email
    FROM all_registered_users

    EXCEPT DISTINCT

    SELECT user_id, email
    FROM unsubscribed_users
    """

    print("SQL Query:")
    print(sql)
    print()

    tracer = SQLColumnTracer(sql, dialect="bigquery")
    lineage = tracer.build_column_lineage_graph()

    print("Analysis Results:")
    print("-" * 80)
    print(f"Total nodes: {len(lineage.nodes)}")
    print(f"Total edges: {len(lineage.edges)}")
    print()


def union_with_cte_example():
    """Example: UNION with CTEs for complex aggregations"""
    print("=" * 80)
    print("Example 6: UNION with CTEs - Advanced Pattern")
    print("=" * 80)
    print()

    sql = """
    WITH monthly_revenue AS (
        SELECT
            DATE_TRUNC(order_date, MONTH) as month,
            SUM(amount) as revenue,
            'orders' as source
        FROM orders
        GROUP BY 1
    ),
    monthly_refunds AS (
        SELECT
            DATE_TRUNC(refund_date, MONTH) as month,
            SUM(amount) as revenue,
            'refunds' as source
        FROM refunds
        GROUP BY 1
    )
    SELECT month, revenue, source FROM monthly_revenue

    UNION ALL

    SELECT month, revenue, source FROM monthly_refunds

    ORDER BY month, source
    """

    print("SQL Query:")
    print(sql)
    print()

    tracer = SQLColumnTracer(sql, dialect="bigquery")
    lineage = tracer.build_column_lineage_graph()

    print("Analysis Results:")
    print("-" * 80)
    print(f"Total nodes: {len(lineage.nodes)}")
    print(f"Total edges: {len(lineage.edges)}")
    print()

    print("Output Columns:")
    for col in lineage.get_output_nodes():
        print(f"  • {col.column_name}")
    print()


def union_with_subquery_example():
    """Example: UNION with subqueries in branches"""
    print("=" * 80)
    print("Example 7: UNION with Subqueries - Complex Nested Pattern")
    print("=" * 80)
    print()

    sql = """
    SELECT
        user_id,
        total_spent,
        tier
    FROM (
        SELECT
            user_id,
            SUM(amount) as total_spent,
            'high' as tier
        FROM orders
        WHERE amount > 1000
        GROUP BY user_id
    )

    UNION ALL

    SELECT
        user_id,
        total_spent,
        tier
    FROM (
        SELECT
            user_id,
            SUM(amount) as total_spent,
            'low' as tier
        FROM orders
        WHERE amount <= 1000
        GROUP BY user_id
    )
    """

    print("SQL Query:")
    print(sql)
    print()

    tracer = SQLColumnTracer(sql, dialect="bigquery")
    lineage = tracer.build_column_lineage_graph()

    print("Analysis Results:")
    print("-" * 80)
    print(f"Total nodes: {len(lineage.nodes)}")
    print(f"Total edges: {len(lineage.edges)}")
    print()


def main():
    """Run all set operation examples"""
    print("\n")
    print("*" * 80)
    print("SQL COLUMN LINEAGE - SET OPERATIONS EXAMPLES")
    print("*" * 80)
    print("\n")

    # Run all examples
    union_all_example()
    print("\n")

    union_distinct_example()
    print("\n")

    three_way_union_example()
    print("\n")

    intersect_example()
    print("\n")

    except_example()
    print("\n")

    union_with_cte_example()
    print("\n")

    union_with_subquery_example()
    print("\n")

    print("*" * 80)
    print("All set operations examples completed!")
    print("*" * 80)


if __name__ == "__main__":
    main()
