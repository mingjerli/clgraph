"""
Example: QUALIFY Clause Lineage Tracking

This example demonstrates how clgraph tracks column lineage through
QUALIFY clauses that filter rows based on window function results.

Key features demonstrated:
1. QUALIFY clause detection
2. PARTITION BY column tracking
3. ORDER BY column tracking
4. Window function identification
5. Export format with QUALIFY metadata
"""

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder


def example_simple_qualify():
    """
    Basic example: Simple QUALIFY with ROW_NUMBER for deduplication.
    """
    print("=" * 60)
    print("Example 1: Simple QUALIFY for Deduplication")
    print("=" * 60)

    sql = """
    SELECT customer_id, order_date, amount
    FROM orders
    QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
    """

    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nLineage Edges:")

    for edge in graph.edges:
        line = f"  {edge.from_node.full_name} -> {edge.to_node.full_name}"
        if edge.is_qualify_column:
            line += f" [QUALIFY {edge.qualify_context} - {edge.qualify_function}]"
        print(line)


def example_multiple_partition_columns():
    """
    Example: QUALIFY with multiple PARTITION BY columns.
    """
    print("\n" + "=" * 60)
    print("Example 2: Multiple PARTITION BY Columns")
    print("=" * 60)

    sql = """
    SELECT *
    FROM products
    QUALIFY ROW_NUMBER() OVER (PARTITION BY category, brand ORDER BY price) = 1
    """

    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nQUALIFY Edges:")

    for edge in graph.edges:
        if edge.is_qualify_column:
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    Context: {edge.qualify_context}, Function: {edge.qualify_function}")


def example_rank_function():
    """
    Example: QUALIFY with RANK function.
    """
    print("\n" + "=" * 60)
    print("Example 3: QUALIFY with RANK Function")
    print("=" * 60)

    sql = """
    SELECT employee_id, department, salary
    FROM employees
    QUALIFY RANK() OVER (PARTITION BY department ORDER BY salary DESC) <= 3
    """

    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nQUALIFY Edges:")

    for edge in graph.edges:
        if edge.is_qualify_column:
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    Context: {edge.qualify_context}, Function: {edge.qualify_function}")


def example_qualify_pipeline():
    """
    Example: QUALIFY lineage through Pipeline API.
    """
    print("\n" + "=" * 60)
    print("Example 4: QUALIFY in Pipeline")
    print("=" * 60)

    sql = """
    SELECT customer_id, order_date
    FROM orders
    QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
    """

    pipeline = Pipeline([("dedupe_query", sql)], dialect="bigquery")

    print(f"\nQuery:\n{sql}")
    print("\nQUALIFY edges in pipeline:")

    for edge in pipeline.column_graph.edges:
        if getattr(edge, "is_qualify_column", False):
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    Context: {edge.qualify_context}, Function: {edge.qualify_function}")


def example_qualify_export():
    """
    Example: Export QUALIFY lineage to JSON.
    """
    print("\n" + "=" * 60)
    print("Example 5: Export QUALIFY Lineage")
    print("=" * 60)

    sql = """
    SELECT customer_id, order_date
    FROM orders
    QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
    """

    pipeline = Pipeline([("dedupe_query", sql)], dialect="bigquery")

    # Export to JSON
    exporter = JSONExporter()
    export_data = exporter.export(pipeline)

    print(f"\nQuery:\n{sql}")
    print("\nExported edges with QUALIFY metadata:")

    import json

    for edge in export_data.get("edges", []):
        if edge.get("is_qualify_column"):
            print(json.dumps(edge, indent=2))


if __name__ == "__main__":
    example_simple_qualify()
    example_multiple_partition_columns()
    example_rank_function()
    example_qualify_pipeline()
    example_qualify_export()

    print("\n" + "=" * 60)
    print("QUALIFY Clause Lineage Examples Complete!")
    print("=" * 60)
