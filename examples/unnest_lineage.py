"""
Example: UNNEST/Array Expansion Lineage Tracking

This example demonstrates how clgraph tracks column lineage through
UNNEST and similar array expansion operations.

Key features demonstrated:
1. UNNEST detection in FROM clause
2. Array expansion edge creation with metadata
3. Lineage from array column to expanded scalar values
4. Support for BigQuery and PostgreSQL dialects
"""

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder


def example_simple_unnest():
    """
    Basic example: Simple UNNEST operation.
    """
    print("=" * 60)
    print("Example 1: Simple UNNEST Lineage")
    print("=" * 60)

    sql = """
    SELECT
        order_id,
        item
    FROM orders, UNNEST(items) AS item
    """

    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nLineage Edges:")

    for edge in graph.edges:
        line = f"  {edge.from_node.full_name} -> {edge.to_node.full_name}"
        if edge.is_array_expansion:
            line += f" [UNNEST: {edge.expansion_type}]"
        print(line)


def example_unnest_with_struct():
    """
    Example: UNNEST with struct array access.
    """
    print("\n" + "=" * 60)
    print("Example 2: UNNEST with Struct Access")
    print("=" * 60)

    sql = """
    SELECT
        order_id,
        item.product_id,
        item.quantity
    FROM orders, UNNEST(items) AS item
    """

    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nLineage Edges:")

    for edge in graph.edges:
        line = f"  {edge.from_node.full_name} -> {edge.to_node.full_name}"
        if edge.is_array_expansion:
            line += f" [UNNEST: {edge.expansion_type}]"
        print(line)


def example_unnest_qualified_source():
    """
    Example: UNNEST with qualified table.column reference.
    """
    print("\n" + "=" * 60)
    print("Example 3: UNNEST with Qualified Source")
    print("=" * 60)

    sql = """
    SELECT
        o.order_id,
        o.customer_name,
        item
    FROM orders o, UNNEST(o.items) AS item
    """

    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nLineage Edges:")

    for edge in graph.edges:
        line = f"  {edge.from_node.full_name} -> {edge.to_node.full_name}"
        if edge.is_array_expansion:
            line += f" [UNNEST: {edge.expansion_type}]"
        print(line)


def example_unnest_pipeline():
    """
    Example: UNNEST lineage through Pipeline API.
    """
    print("\n" + "=" * 60)
    print("Example 4: UNNEST in Pipeline")
    print("=" * 60)

    sql = """
    SELECT
        order_id,
        item
    FROM orders, UNNEST(items) AS item
    """

    pipeline = Pipeline([("query", sql)], dialect="bigquery")

    print(f"\nQuery:\n{sql}")
    print("\nArray expansion edges:")

    for edge in pipeline.column_graph.edges:
        if edge.is_array_expansion:
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    Expansion type: {edge.expansion_type}")


def example_unnest_export():
    """
    Example: Export UNNEST lineage to JSON.
    """
    print("\n" + "=" * 60)
    print("Example 5: Export UNNEST Lineage")
    print("=" * 60)

    sql = """
    SELECT item FROM orders, UNNEST(items) AS item
    """

    pipeline = Pipeline([("query", sql)], dialect="bigquery")

    # Export to JSON
    exporter = JSONExporter()
    export_data = exporter.export(pipeline)

    print(f"\nQuery:\n{sql}")
    print("\nExported edges with array expansion metadata:")

    import json

    for edge in export_data.get("edges", []):
        if edge.get("is_array_expansion"):
            print(json.dumps(edge, indent=2))


def example_postgresql_unnest():
    """
    Example: PostgreSQL UNNEST syntax.
    """
    print("\n" + "=" * 60)
    print("Example 6: PostgreSQL UNNEST")
    print("=" * 60)

    sql = """
    SELECT
        order_id,
        item
    FROM orders, UNNEST(items) AS item
    """

    builder = RecursiveLineageBuilder(sql, dialect="postgres")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nLineage Edges:")

    for edge in graph.edges:
        line = f"  {edge.from_node.full_name} -> {edge.to_node.full_name}"
        if edge.is_array_expansion:
            line += f" [UNNEST: {edge.expansion_type}]"
        print(line)


if __name__ == "__main__":
    example_simple_unnest()
    example_unnest_with_struct()
    example_unnest_qualified_source()
    example_unnest_pipeline()
    example_unnest_export()
    example_postgresql_unnest()

    print("\n" + "=" * 60)
    print("UNNEST Lineage Examples Complete!")
    print("=" * 60)
