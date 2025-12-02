"""
Example demonstrating SQL column lineage analysis with UNPIVOT operations.

This example shows how clgraph analyzes UNPIVOT operations, which transform
columns into rows for normalization and easier analysis.
"""

from clgraph import SQLColumnTracer


def basic_unpivot_example():
    """Example: Basic UNPIVOT to normalize quarterly data"""
    print("=" * 80)
    print("Example 1: Basic UNPIVOT - Quarterly Revenue to Rows")
    print("=" * 80)
    print()

    sql = """
    SELECT
        product_id,
        product_name,
        quarter,
        revenue
    FROM quarterly_revenue
    UNPIVOT(
        revenue FOR quarter IN (Q1, Q2, Q3, Q4)
    )
    """

    print("SQL Query:")
    print(sql)
    print()

    print("What UNPIVOT does:")
    print("  Before UNPIVOT (wide format):")
    print("    product_id | product_name | Q1   | Q2   | Q3   | Q4")
    print("    -----------|--------------|------|------|------|------")
    print("    1          | Widget A     | 1000 | 1500 | 2000 | 1800")
    print()
    print("  After UNPIVOT (long format):")
    print("    product_id | product_name | quarter | revenue")
    print("    -----------|--------------|---------|--------")
    print("    1          | Widget A     | Q1      | 1000")
    print("    1          | Widget A     | Q2      | 1500")
    print("    1          | Widget A     | Q3      | 2000")
    print("    1          | Widget A     | Q4      | 1800")
    print()

    try:
        tracer = SQLColumnTracer(sql, dialect="bigquery")
        lineage = tracer.build_column_lineage_graph()

        print("Analysis Results:")
        print("-" * 80)
        print(f"Total nodes: {len(lineage.nodes)}")
        print(f"Total edges: {len(lineage.edges)}")
        print()

        print("Output Columns (after UNPIVOT):")
        for col in lineage.get_output_nodes():
            print(f"  • {col.column_name}")
        print()
    except Exception as e:
        print(f"Note: UNPIVOT parsing may have limitations in sqlglot: {e}")
        print()


def unpivot_with_multiple_columns_example():
    """Example: UNPIVOT multiple measure columns"""
    print("=" * 80)
    print("Example 2: UNPIVOT Multiple Measures - Sales and Costs")
    print("=" * 80)
    print()

    sql = """
    SELECT
        product_id,
        metric_name,
        metric_value
    FROM product_metrics
    UNPIVOT(
        metric_value FOR metric_name IN (
            sales_q1, sales_q2, sales_q3, sales_q4,
            cost_q1, cost_q2, cost_q3, cost_q4
        )
    )
    """

    print("SQL Query:")
    print(sql)
    print()

    print("What UNPIVOT does:")
    print("  Transforms 8 columns (sales_q1...sales_q4, cost_q1...cost_q4)")
    print("  into 2 columns (metric_name, metric_value)")
    print()

    try:
        tracer = SQLColumnTracer(sql, dialect="bigquery")
        lineage = tracer.build_column_lineage_graph()

        print("Analysis Results:")
        print("-" * 80)
        print(f"Total nodes: {len(lineage.nodes)}")
        print(f"Total edges: {len(lineage.edges)}")
        print()
    except Exception as e:
        print(f"Note: UNPIVOT parsing may have limitations: {e}")
        print()


def unpivot_include_nulls_example():
    """Example: UNPIVOT with NULL handling"""
    print("=" * 80)
    print("Example 3: UNPIVOT with NULL Handling")
    print("=" * 80)
    print()

    sql = """
    SELECT
        user_id,
        month,
        activity_count
    FROM user_activity
    UNPIVOT INCLUDE NULLS (
        activity_count FOR month IN (jan, feb, mar, apr, may, jun)
    )
    """

    print("SQL Query:")
    print(sql)
    print()

    print("What INCLUDE NULLS does:")
    print("  By default, UNPIVOT excludes rows where the value is NULL")
    print("  INCLUDE NULLS keeps those rows in the result")
    print()
    print("  Example:")
    print("    user_id | jan | feb | mar")
    print("    --------|-----|-----|-----")
    print("    1       | 10  | NULL| 15")
    print()
    print("  With INCLUDE NULLS:")
    print("    user_id | month | activity_count")
    print("    --------|-------|---------------")
    print("    1       | jan   | 10")
    print("    1       | feb   | NULL           <- Kept!")
    print("    1       | mar   | 15")
    print()

    try:
        tracer = SQLColumnTracer(sql, dialect="bigquery")
        lineage = tracer.build_column_lineage_graph()

        print("Analysis Results:")
        print("-" * 80)
        print(f"Total nodes: {len(lineage.nodes)}")
        print(f"Total edges: {len(lineage.edges)}")
        print()
    except Exception as e:
        print(f"Note: UNPIVOT parsing may have limitations: {e}")
        print()


def unpivot_with_cte_example():
    """Example: UNPIVOT with CTE preprocessing"""
    print("=" * 80)
    print("Example 4: UNPIVOT with CTE - Normalized Reporting")
    print("=" * 80)
    print()

    sql = """
    WITH wide_format AS (
        SELECT
            department,
            SUM(CASE WHEN month = 1 THEN revenue ELSE 0 END) as jan_revenue,
            SUM(CASE WHEN month = 2 THEN revenue ELSE 0 END) as feb_revenue,
            SUM(CASE WHEN month = 3 THEN revenue ELSE 0 END) as mar_revenue,
            SUM(CASE WHEN month = 4 THEN revenue ELSE 0 END) as apr_revenue
        FROM monthly_revenue
        GROUP BY department
    )
    SELECT
        department,
        month,
        revenue
    FROM wide_format
    UNPIVOT(
        revenue FOR month IN (
            jan_revenue as 'January',
            feb_revenue as 'February',
            mar_revenue as 'March',
            apr_revenue as 'April'
        )
    )
    """

    print("SQL Query:")
    print(sql)
    print()

    print("What this query does:")
    print("  1. CTE aggregates revenue by department and creates wide format")
    print("  2. UNPIVOT transforms back to long format with readable month names")
    print("  3. Result: department, month, revenue")
    print()

    try:
        tracer = SQLColumnTracer(sql, dialect="bigquery")
        lineage = tracer.build_column_lineage_graph()

        print("Analysis Results:")
        print("-" * 80)
        print(f"Total nodes: {len(lineage.nodes)}")
        print(f"Total edges: {len(lineage.edges)}")
        print()
    except Exception as e:
        print(f"Note: UNPIVOT parsing may have limitations: {e}")
        print()


def unpivot_real_world_example():
    """Example: Real-world UNPIVOT - Survey data normalization"""
    print("=" * 80)
    print("Example 5: Real-world UNPIVOT - Survey Response Analysis")
    print("=" * 80)
    print()

    sql = """
    WITH survey_responses AS (
        SELECT
            respondent_id,
            question_1_rating,
            question_2_rating,
            question_3_rating,
            question_4_rating,
            question_5_rating
        FROM customer_survey
        WHERE survey_date >= '2024-01-01'
    )
    SELECT
        respondent_id,
        question_number,
        rating
    FROM survey_responses
    UNPIVOT(
        rating FOR question_number IN (
            question_1_rating as 'Q1',
            question_2_rating as 'Q2',
            question_3_rating as 'Q3',
            question_4_rating as 'Q4',
            question_5_rating as 'Q5'
        )
    )
    WHERE rating IS NOT NULL
    """

    print("SQL Query:")
    print(sql)
    print()

    print("What this query does:")
    print("  1. Retrieves survey responses in wide format (one row per respondent)")
    print("  2. Uses UNPIVOT to normalize into long format (one row per response)")
    print("  3. Filters out NULL responses")
    print("  4. Makes it easier to analyze question performance")
    print()

    try:
        tracer = SQLColumnTracer(sql, dialect="bigquery")
        lineage = tracer.build_column_lineage_graph()

        print("Analysis Results:")
        print("-" * 80)
        print(f"Total nodes: {len(lineage.nodes)}")
        print(f"Total edges: {len(lineage.edges)}")
        print()
    except Exception as e:
        print(f"Note: UNPIVOT parsing may have limitations: {e}")
        print()


def pivot_unpivot_comparison_example():
    """Example: Comparing PIVOT and UNPIVOT transformations"""
    print("=" * 80)
    print("Example 6: PIVOT vs UNPIVOT - Inverse Operations")
    print("=" * 80)
    print()

    print("PIVOT (Rows → Columns):")
    print("-" * 80)
    sql_pivot = """
    SELECT * FROM sales_data
    PIVOT(
        SUM(amount) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
    )
    """
    print(sql_pivot)
    print()

    print("UNPIVOT (Columns → Rows):")
    print("-" * 80)
    sql_unpivot = """
    SELECT * FROM quarterly_summary
    UNPIVOT(
        amount FOR quarter IN (Q1, Q2, Q3, Q4)
    )
    """
    print(sql_unpivot)
    print()

    print("Key Differences:")
    print("  • PIVOT: Transforms unique row values into column headers")
    print("  • UNPIVOT: Transforms column headers into row values")
    print()
    print("  • PIVOT: Reduces row count, increases column count")
    print("  • UNPIVOT: Increases row count, reduces column count")
    print()
    print("  • PIVOT: Requires aggregate function (SUM, AVG, COUNT, etc.)")
    print("  • UNPIVOT: No aggregate needed, just column selection")
    print()


def main():
    """Run all UNPIVOT examples"""
    print("\n")
    print("*" * 80)
    print("SQL COLUMN LINEAGE - UNPIVOT OPERATIONS EXAMPLES")
    print("*" * 80)
    print("\n")

    # Run all examples
    basic_unpivot_example()
    print("\n")

    unpivot_with_multiple_columns_example()
    print("\n")

    unpivot_include_nulls_example()
    print("\n")

    unpivot_with_cte_example()
    print("\n")

    unpivot_real_world_example()
    print("\n")

    pivot_unpivot_comparison_example()
    print("\n")

    print("*" * 80)
    print("All UNPIVOT examples completed!")
    print("*" * 80)
    print()
    print("Key Takeaways:")
    print("  • UNPIVOT transforms column headers into row values")
    print("  • Useful for normalizing wide-format data")
    print("  • Creates two new columns: name column and value column")
    print("  • Can handle NULL values with INCLUDE NULLS option")
    print("  • Inverse operation of PIVOT")
    print()
    print("Note: UNPIVOT support in sqlglot may be limited.")
    print("      Some queries may not parse correctly.")


if __name__ == "__main__":
    main()
