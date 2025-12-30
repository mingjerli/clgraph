"""
Example: LATERAL Join Lineage Tracking

This example demonstrates how clgraph tracks column lineage through
LATERAL subqueries and their correlated column references.

Key features demonstrated:
1. LATERAL subquery detection
2. Correlated column identification
3. Lateral correlation edge creation
4. Export format with LATERAL metadata
"""

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder


def example_simple_lateral():
    """
    Basic example: Simple LATERAL subquery with aggregation.
    """
    print("=" * 60)
    print("Example 1: Simple LATERAL Subquery")
    print("=" * 60)

    sql = """
    SELECT o.order_id, t.total
    FROM orders o,
    LATERAL (
        SELECT SUM(amount) as total
        FROM items i
        WHERE i.order_id = o.order_id
    ) t
    """

    builder = RecursiveLineageBuilder(sql, dialect="postgres")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nLineage Edges:")

    for edge in graph.edges:
        line = f"  {edge.from_node.full_name} -> {edge.to_node.full_name}"
        if edge.is_lateral_correlation:
            line += " [LATERAL correlation]"
        print(line)


def example_lateral_with_multiple_correlations():
    """
    Example: LATERAL with multiple correlated columns.
    """
    print("\n" + "=" * 60)
    print("Example 2: LATERAL with Multiple Correlated Columns")
    print("=" * 60)

    sql = """
    SELECT c.customer_id, s.total
    FROM customers c,
    LATERAL (
        SELECT SUM(amount) as total
        FROM orders o
        WHERE o.customer_id = c.customer_id
        AND o.region = c.region
    ) s
    """

    builder = RecursiveLineageBuilder(sql, dialect="postgres")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nLATERAL Correlation Edges:")

    for edge in graph.edges:
        if edge.is_lateral_correlation:
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    Lateral alias: {edge.lateral_alias}")


def example_lateral_pipeline():
    """
    Example: LATERAL lineage through Pipeline API.
    """
    print("\n" + "=" * 60)
    print("Example 3: LATERAL in Pipeline")
    print("=" * 60)

    sql = """
    SELECT c.customer_id, recent.last_order
    FROM customers c,
    LATERAL (
        SELECT MAX(order_date) as last_order
        FROM orders o
        WHERE o.customer_id = c.customer_id
    ) recent
    """

    pipeline = Pipeline([("query", sql)], dialect="postgres")

    print(f"\nQuery:\n{sql}")
    print("\nLATERAL correlation edges:")

    for edge in pipeline.column_graph.edges:
        if getattr(edge, "is_lateral_correlation", False):
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    Lateral alias: {edge.lateral_alias}")


def example_lateral_export():
    """
    Example: Export LATERAL lineage to JSON.
    """
    print("\n" + "=" * 60)
    print("Example 4: Export LATERAL Lineage")
    print("=" * 60)

    sql = """
    SELECT o.order_id, t.total
    FROM orders o,
    LATERAL (
        SELECT SUM(amount) as total
        FROM items i
        WHERE i.order_id = o.order_id
    ) t
    """

    pipeline = Pipeline([("query", sql)], dialect="postgres")

    # Export to JSON
    exporter = JSONExporter()
    export_data = exporter.export(pipeline)

    print(f"\nQuery:\n{sql}")
    print("\nExported edges with LATERAL metadata:")

    import json

    for edge in export_data.get("edges", []):
        if edge.get("is_lateral_correlation"):
            print(json.dumps(edge, indent=2))


if __name__ == "__main__":
    example_simple_lateral()
    example_lateral_with_multiple_correlations()
    example_lateral_pipeline()
    example_lateral_export()

    print("\n" + "=" * 60)
    print("LATERAL Join Lineage Examples Complete!")
    print("=" * 60)
