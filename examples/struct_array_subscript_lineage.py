"""
Example: Struct/Array Subscript Lineage Tracking

This example demonstrates how clgraph tracks column lineage through
array subscript and struct field access operations.

Key features demonstrated:
1. Array subscript access (items[0], data[1])
2. Map/dictionary key access (metadata['key'])
3. Struct field access after array subscript (items[0].product_id)
4. Mixed nested access patterns
5. Export format with nested path metadata
"""

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder


def example_array_subscript():
    """
    Basic example: Array subscript access.
    """
    print("=" * 60)
    print("Example 1: Array Subscript Access")
    print("=" * 60)

    sql = """
    SELECT
        order_id,
        items[0] AS first_item,
        items[1] AS second_item
    FROM orders
    """

    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nLineage Edges:")

    for edge in graph.edges:
        line = f"  {edge.from_node.full_name} -> {edge.to_node.full_name}"
        if edge.nested_path:
            line += f" [nested: {edge.nested_path}, type: {edge.access_type}]"
        print(line)


def example_map_access():
    """
    Example: Map/dictionary key access.
    """
    print("\n" + "=" * 60)
    print("Example 2: Map/Dictionary Key Access")
    print("=" * 60)

    sql = """
    SELECT
        order_id,
        metadata['status'] AS order_status,
        metadata['priority'] AS priority_level
    FROM orders
    """

    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nLineage Edges:")

    for edge in graph.edges:
        line = f"  {edge.from_node.full_name} -> {edge.to_node.full_name}"
        if edge.nested_path:
            line += f" [nested: {edge.nested_path}, type: {edge.access_type}]"
        print(line)


def example_struct_after_array():
    """
    Example: Struct field access after array subscript.
    """
    print("\n" + "=" * 60)
    print("Example 3: Struct Field Access After Array")
    print("=" * 60)

    sql = """
    SELECT
        order_id,
        items[0].product_id AS first_product_id,
        items[0].quantity AS first_quantity,
        items[1].product_id AS second_product_id
    FROM orders
    """

    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nLineage Edges:")

    for edge in graph.edges:
        line = f"  {edge.from_node.full_name} -> {edge.to_node.full_name}"
        if edge.nested_path:
            line += f" [nested: {edge.nested_path}, type: {edge.access_type}]"
        print(line)


def example_mixed_patterns():
    """
    Example: Mixed regular and nested columns.
    """
    print("\n" + "=" * 60)
    print("Example 4: Mixed Regular and Nested Columns")
    print("=" * 60)

    sql = """
    SELECT
        order_id,
        customer_name,
        items[0].product_id AS first_product,
        metadata['status'] AS order_status
    FROM orders
    """

    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nLineage Edges:")

    for edge in graph.edges:
        line = f"  {edge.from_node.full_name} -> {edge.to_node.full_name}"
        if edge.nested_path:
            line += f" [nested: {edge.nested_path}, type: {edge.access_type}]"
        print(line)


def example_pipeline():
    """
    Example: Nested access lineage through Pipeline API.
    """
    print("\n" + "=" * 60)
    print("Example 5: Nested Access in Pipeline")
    print("=" * 60)

    sql = """
    SELECT
        order_id,
        items[0].product_id AS first_product
    FROM orders
    """

    pipeline = Pipeline([("query", sql)], dialect="bigquery")

    print(f"\nQuery:\n{sql}")
    print("\nNested access edges:")

    for edge in pipeline.column_graph.edges:
        if edge.nested_path:
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    Nested path: {edge.nested_path}")
            print(f"    Access type: {edge.access_type}")


def example_export():
    """
    Example: Export nested access lineage to JSON.
    """
    print("\n" + "=" * 60)
    print("Example 6: Export Nested Access Lineage")
    print("=" * 60)

    sql = """
    SELECT items[0].product_id AS first_product FROM orders
    """

    pipeline = Pipeline([("query", sql)], dialect="bigquery")

    # Export to JSON
    exporter = JSONExporter()
    export_data = exporter.export(pipeline)

    print(f"\nQuery:\n{sql}")
    print("\nExported edges with nested access metadata:")

    import json

    for edge in export_data.get("edges", []):
        if edge.get("nested_path"):
            print(json.dumps(edge, indent=2))


if __name__ == "__main__":
    example_array_subscript()
    example_map_access()
    example_struct_after_array()
    example_mixed_patterns()
    example_pipeline()
    example_export()

    print("\n" + "=" * 60)
    print("Struct/Array Subscript Lineage Examples Complete!")
    print("=" * 60)
