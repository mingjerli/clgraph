"""
Example: Window Frame Analysis for Column Lineage

This example demonstrates how clgraph tracks column lineage through
window functions, including PARTITION BY, ORDER BY, and frame specifications.

Key features demonstrated:
1. Window function detection and parsing
2. PARTITION BY column dependencies
3. ORDER BY column dependencies with direction
4. Frame specification capture (ROWS/RANGE)
5. Named window definitions
6. Export format with window metadata
"""

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder
from clgraph.query_parser import RecursiveQueryParser


def example_basic_window():
    """
    Example: Basic window function with PARTITION BY and ORDER BY.
    """
    print("=" * 60)
    print("Example 1: Basic Window Function")
    print("=" * 60)

    sql = """
    SELECT
        customer_id,
        order_date,
        amount,
        SUM(amount) OVER (
            PARTITION BY customer_id
            ORDER BY order_date
        ) AS running_total
    FROM orders
    """

    # Parse to see window configuration
    parser = RecursiveQueryParser(sql, dialect="bigquery")
    graph = parser.parse()

    print(f"\nQuery:\n{sql}")
    print("\nWindow Function Info:")
    for window in graph.units["main"].window_info.get("windows", []):
        print(f"  Output column: {window['output_column']}")
        print(f"  Function: {window['function']}")
        print(f"  Arguments: {window['arguments']}")
        print(f"  PARTITION BY: {window['partition_by']}")
        print(f"  ORDER BY: {window['order_by']}")

    # Build lineage
    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    lineage_graph = builder.build()

    print("\nLineage Edges:")
    for edge in lineage_graph.edges:
        if edge.is_window_function:
            line = f"  {edge.from_node.full_name} -> {edge.to_node.full_name}"
            line += f" [{edge.edge_type}]"
            print(line)


def example_frame_specification():
    """
    Example: Window function with frame specification.
    """
    print("\n" + "=" * 60)
    print("Example 2: Frame Specification (Rolling Window)")
    print("=" * 60)

    sql = """
    SELECT
        order_date,
        amount,
        SUM(amount) OVER (
            ORDER BY order_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7day_sum
    FROM daily_sales
    """

    parser = RecursiveQueryParser(sql, dialect="bigquery")
    graph = parser.parse()

    print(f"\nQuery:\n{sql}")
    print("\nFrame Specification:")
    for window in graph.units["main"].window_info.get("windows", []):
        print(f"  Frame type: {window['frame_type']}")
        print(f"  Frame start: {window['frame_start']}")
        print(f"  Frame end: {window['frame_end']}")

    # Build lineage
    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    lineage_graph = builder.build()

    print("\nWindow Edges with Frame Info:")
    for edge in lineage_graph.edges:
        if edge.is_window_function:
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    Role: {edge.window_role}")
            print(
                f"    Frame: {edge.window_frame_type} {edge.window_frame_start} to {edge.window_frame_end}"
            )


def example_ranking_functions():
    """
    Example: Ranking window functions.
    """
    print("\n" + "=" * 60)
    print("Example 3: Ranking Functions")
    print("=" * 60)

    sql = """
    SELECT
        product_id,
        category,
        sales,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) AS rank,
        DENSE_RANK() OVER (PARTITION BY category ORDER BY sales DESC) AS dense_rank
    FROM product_sales
    """

    parser = RecursiveQueryParser(sql, dialect="bigquery")
    graph = parser.parse()

    print(f"\nQuery:\n{sql}")
    print("\nRanking Functions Detected:")
    for window in graph.units["main"].window_info.get("windows", []):
        print(f"  {window['function']}() -> {window['output_column']}")
        print(f"    PARTITION BY: {window['partition_by']}")
        order = window["order_by"]
        if order:
            print(f"    ORDER BY: {order[0]['column']} {order[0]['direction'].upper()}")


def example_named_window():
    """
    Example: Named window definition.
    """
    print("\n" + "=" * 60)
    print("Example 4: Named Window Definition")
    print("=" * 60)

    sql = """
    SELECT
        SUM(amount) OVER w AS total,
        AVG(amount) OVER w AS average,
        COUNT(*) OVER w AS cnt
    FROM orders
    WINDOW w AS (PARTITION BY customer_id ORDER BY order_date)
    """

    parser = RecursiveQueryParser(sql, dialect="bigquery")
    graph = parser.parse()

    print(f"\nQuery:\n{sql}")
    print("\nNamed Window Definitions:")
    for name, spec in graph.units["main"].window_definitions.items():
        print(f"  {name}: PARTITION BY {spec.get('partition_by')}, ORDER BY {spec.get('order_by')}")

    print("\nWindow Functions Using Named Window:")
    for window in graph.units["main"].window_info.get("windows", []):
        print(f"  {window['function']}() OVER {window['window_name']} -> {window['output_column']}")


def example_pipeline():
    """
    Example: Window functions through Pipeline API.
    """
    print("\n" + "=" * 60)
    print("Example 5: Window Functions in Pipeline")
    print("=" * 60)

    sql = """
    SELECT
        customer_id,
        order_date,
        amount,
        SUM(amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS running_total,
        LAG(amount, 1) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_amount
    FROM orders
    """

    pipeline = Pipeline([("window_query", sql)], dialect="bigquery")

    print(f"\nQuery:\n{sql}")
    print("\nWindow Edges in Pipeline:")

    for edge in pipeline.column_graph.edges:
        if getattr(edge, "is_window_function", False):
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    Function: {edge.window_function}")
            print(f"    Role: {edge.window_role}")


def example_export():
    """
    Example: Export window lineage to JSON.
    """
    print("\n" + "=" * 60)
    print("Example 6: Export Window Lineage")
    print("=" * 60)

    sql = """
    SELECT
        SUM(amount) OVER (
            PARTITION BY customer_id
            ORDER BY order_date DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_sum
    FROM orders
    """

    pipeline = Pipeline([("window_query", sql)], dialect="bigquery")

    exporter = JSONExporter()
    export_data = exporter.export(pipeline)

    print(f"\nQuery:\n{sql}")
    print("\nExported edges with window metadata:")

    import json

    for edge in export_data.get("edges", []):
        if edge.get("is_window_function"):
            print(json.dumps(edge, indent=2))


if __name__ == "__main__":
    example_basic_window()
    example_frame_specification()
    example_ranking_functions()
    example_named_window()
    example_pipeline()
    example_export()

    print("\n" + "=" * 60)
    print("Window Frame Analysis Examples Complete!")
    print("=" * 60)
