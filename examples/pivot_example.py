"""
Example demonstrating SQL column lineage analysis with PIVOT operations.

This example shows how clgraph analyzes PIVOT operations, which transform
rows into columns for easier analysis and reporting.
"""

from clgraph import SQLColumnTracer


def basic_pivot_example():
    """Example: Basic PIVOT to transform quarterly data"""
    print("=" * 80)
    print("Example 1: Basic PIVOT - Quarterly Sales by Product")
    print("=" * 80)
    print()

    sql = """
    SELECT * FROM (
        SELECT
            product_name,
            quarter,
            sales_amount
        FROM quarterly_sales
        WHERE year = 2024
    )
    PIVOT(
        SUM(sales_amount) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
    )
    """

    print("SQL Query:")
    print(sql)
    print()

    print("What PIVOT does:")
    print("  Before PIVOT:")
    print("    product_name | quarter | sales_amount")
    print("    -------------|---------|-------------")
    print("    Widget A     | Q1      | 1000")
    print("    Widget A     | Q2      | 1500")
    print()
    print("  After PIVOT:")
    print("    product_name | Q1   | Q2   | Q3   | Q4")
    print("    -------------|------|------|------|------")
    print("    Widget A     | 1000 | 1500 | 2000 | 1800")
    print()

    tracer = SQLColumnTracer(sql, dialect="bigquery")
    lineage = tracer.build_column_lineage_graph()

    print("Analysis Results:")
    print("-" * 80)
    print(f"Total nodes: {len(lineage.nodes)}")
    print(f"Total edges: {len(lineage.edges)}")
    print()

    print("Output Columns (after PIVOT):")
    for col in lineage.get_output_nodes():
        print(f"  • {col.column_name}")
    print()


def pivot_from_base_table_example():
    """Example: PIVOT directly from a base table"""
    print("=" * 80)
    print("Example 2: PIVOT from Base Table - Regional Revenue")
    print("=" * 80)
    print()

    sql = """
    SELECT * FROM sales_data
    PIVOT(
        SUM(revenue) FOR region IN ('North', 'South', 'East', 'West')
    )
    """

    print("SQL Query:")
    print(sql)
    print()

    print("What PIVOT does:")
    print("  Transforms region column values (North, South, East, West)")
    print("  into separate columns with aggregated revenue values")
    print()

    tracer = SQLColumnTracer(sql, dialect="bigquery")
    lineage = tracer.build_column_lineage_graph()

    print("Analysis Results:")
    print("-" * 80)
    print(f"Total nodes: {len(lineage.nodes)}")
    print(f"Total edges: {len(lineage.edges)}")
    print()


def multi_aggregation_pivot_example():
    """Example: PIVOT with multiple aggregation functions"""
    print("=" * 80)
    print("Example 3: PIVOT with Multiple Aggregations")
    print("=" * 80)
    print()

    sql = """
    SELECT * FROM (
        SELECT
            product_category,
            month,
            sales_amount,
            order_count
        FROM monthly_metrics
    )
    PIVOT(
        SUM(sales_amount) AS total_sales,
        AVG(order_count) AS avg_orders
        FOR month IN ('Jan', 'Feb', 'Mar')
    )
    """

    print("SQL Query:")
    print(sql)
    print()

    print("What PIVOT does:")
    print("  Creates separate columns for each combination of:")
    print("    - Aggregation function (SUM, AVG)")
    print("    - Pivot value (Jan, Feb, Mar)")
    print()
    print("  Result columns: Jan_total_sales, Jan_avg_orders,")
    print("                  Feb_total_sales, Feb_avg_orders, etc.")
    print()

    tracer = SQLColumnTracer(sql, dialect="bigquery")
    lineage = tracer.build_column_lineage_graph()

    print("Analysis Results:")
    print("-" * 80)
    print(f"Total nodes: {len(lineage.nodes)}")
    print(f"Total edges: {len(lineage.edges)}")
    print()


def pivot_with_cte_example():
    """Example: PIVOT with CTE as source"""
    print("=" * 80)
    print("Example 4: PIVOT with CTE - User Activity Analysis")
    print("=" * 80)
    print()

    sql = """
    WITH user_activity AS (
        SELECT
            user_id,
            activity_type,
            COUNT(*) as activity_count
        FROM user_events
        WHERE event_date >= '2024-01-01'
        GROUP BY user_id, activity_type
    )
    SELECT * FROM user_activity
    PIVOT(
        SUM(activity_count) FOR activity_type IN ('login', 'purchase', 'review', 'share')
    )
    """

    print("SQL Query:")
    print(sql)
    print()

    print("What PIVOT does:")
    print("  Before PIVOT (CTE output):")
    print("    user_id | activity_type | activity_count")
    print("    --------|---------------|---------------")
    print("    123     | login         | 45")
    print("    123     | purchase      | 3")
    print()
    print("  After PIVOT:")
    print("    user_id | login | purchase | review | share")
    print("    --------|-------|----------|--------|-------")
    print("    123     | 45    | 3        | 1      | 2")
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


def pivot_with_filters_example():
    """Example: PIVOT with WHERE clause and complex source"""
    print("=" * 80)
    print("Example 5: PIVOT with Filters - Product Performance Dashboard")
    print("=" * 80)
    print()

    sql = """
    SELECT * FROM (
        SELECT
            p.product_name,
            p.category,
            s.store_location,
            s.quantity_sold
        FROM products p
        JOIN sales s ON p.product_id = s.product_id
        WHERE s.sale_date BETWEEN '2024-01-01' AND '2024-12-31'
    )
    PIVOT(
        SUM(quantity_sold) FOR store_location IN ('NYC', 'LA', 'Chicago', 'Houston')
    )
    """

    print("SQL Query:")
    print(sql)
    print()

    print("What PIVOT does:")
    print("  Transforms store location data from rows to columns")
    print("  Shows total quantity sold per product in each city")
    print()

    tracer = SQLColumnTracer(sql, dialect="bigquery")
    lineage = tracer.build_column_lineage_graph()

    print("Analysis Results:")
    print("-" * 80)
    print(f"Total nodes: {len(lineage.nodes)}")
    print(f"Total edges: {len(lineage.edges)}")
    print()


def pivot_real_world_example():
    """Example: Real-world PIVOT use case - Financial reporting"""
    print("=" * 80)
    print("Example 6: Real-world PIVOT - Monthly Financial Report")
    print("=" * 80)
    print()

    sql = """
    WITH monthly_data AS (
        SELECT
            department,
            EXTRACT(MONTH FROM transaction_date) as month,
            SUM(CASE WHEN type = 'revenue' THEN amount ELSE 0 END) as revenue,
            SUM(CASE WHEN type = 'expense' THEN amount ELSE 0 END) as expense
        FROM financial_transactions
        WHERE EXTRACT(YEAR FROM transaction_date) = 2024
        GROUP BY department, month
    )
    SELECT
        department,
        Jan_revenue - Jan_expense as Jan_profit,
        Feb_revenue - Feb_expense as Feb_profit,
        Mar_revenue - Mar_expense as Mar_profit
    FROM monthly_data
    PIVOT(
        SUM(revenue) as revenue,
        SUM(expense) as expense
        FOR month IN (1 as Jan, 2 as Feb, 3 as Mar)
    )
    """

    print("SQL Query:")
    print(sql)
    print()

    print("What this query does:")
    print("  1. Aggregates revenue and expenses by department and month")
    print("  2. Uses PIVOT to transform months into columns")
    print("  3. Calculates monthly profit for each department")
    print()

    tracer = SQLColumnTracer(sql, dialect="bigquery")
    lineage = tracer.build_column_lineage_graph()

    print("Analysis Results:")
    print("-" * 80)
    print(f"Total nodes: {len(lineage.nodes)}")
    print(f"Total edges: {len(lineage.edges)}")
    print()


def main():
    """Run all PIVOT examples"""
    print("\n")
    print("*" * 80)
    print("SQL COLUMN LINEAGE - PIVOT OPERATIONS EXAMPLES")
    print("*" * 80)
    print("\n")

    # Run all examples
    basic_pivot_example()
    print("\n")

    pivot_from_base_table_example()
    print("\n")

    multi_aggregation_pivot_example()
    print("\n")

    pivot_with_cte_example()
    print("\n")

    pivot_with_filters_example()
    print("\n")

    pivot_real_world_example()
    print("\n")

    print("*" * 80)
    print("All PIVOT examples completed!")
    print("*" * 80)
    print()
    print("Key Takeaways:")
    print("  • PIVOT transforms row values into column headers")
    print("  • Useful for creating cross-tabulations and reports")
    print("  • Requires an aggregate function (SUM, COUNT, AVG, etc.)")
    print("  • Can handle multiple aggregations in a single PIVOT")
    print("  • Works with CTEs, subqueries, and base tables")


if __name__ == "__main__":
    main()
