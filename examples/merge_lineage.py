"""
Example: MERGE Statement Lineage Tracking

This example demonstrates how clgraph tracks column lineage through
MERGE INTO statements (UPSERT operations).

Key features demonstrated:
1. MERGE INTO parsing
2. Match condition column tracking
3. UPDATE action lineage
4. INSERT action lineage
5. Export format with MERGE metadata
"""

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder


def example_simple_merge():
    """
    Basic example: Simple MERGE with UPDATE and INSERT.
    """
    print("=" * 60)
    print("Example 1: Simple MERGE Statement")
    print("=" * 60)

    sql = """
    MERGE INTO customers c
    USING updates u ON c.customer_id = u.customer_id
    WHEN MATCHED THEN UPDATE SET c.name = u.name, c.email = u.email
    WHEN NOT MATCHED THEN INSERT (customer_id, name, email) VALUES (u.customer_id, u.name, u.email)
    """

    builder = RecursiveLineageBuilder(sql, dialect="postgres")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nLineage Edges:")

    for edge in graph.edges:
        line = f"  {edge.from_node.full_name} -> {edge.to_node.full_name}"
        if edge.is_merge_operation:
            line += f" [MERGE {edge.merge_action}]"
        print(line)


def example_merge_with_conditions():
    """
    Example: MERGE with conditional UPDATE.
    """
    print("\n" + "=" * 60)
    print("Example 2: MERGE with Conditional UPDATE")
    print("=" * 60)

    sql = """
    MERGE INTO inventory i
    USING shipments s ON i.product_id = s.product_id
    WHEN MATCHED AND s.quantity > 0 THEN UPDATE SET i.stock = i.stock + s.quantity
    WHEN NOT MATCHED THEN INSERT (product_id, stock) VALUES (s.product_id, s.quantity)
    """

    builder = RecursiveLineageBuilder(sql, dialect="postgres")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nMERGE Operation Edges:")

    for edge in graph.edges:
        if edge.is_merge_operation:
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    Action: {edge.merge_action}")
            if edge.merge_condition:
                print(f"    Condition: {edge.merge_condition}")


def example_merge_pipeline():
    """
    Example: MERGE lineage through Pipeline API.
    """
    print("\n" + "=" * 60)
    print("Example 3: MERGE in Pipeline")
    print("=" * 60)

    sql = """
    MERGE INTO target t
    USING source s ON t.id = s.id
    WHEN MATCHED THEN UPDATE SET t.value = s.new_value
    WHEN NOT MATCHED THEN INSERT (id, value) VALUES (s.id, s.new_value)
    """

    pipeline = Pipeline([("merge_op", sql)], dialect="postgres")

    print(f"\nQuery:\n{sql}")
    print("\nMERGE operation edges in pipeline:")

    for edge in pipeline.column_graph.edges:
        if getattr(edge, "is_merge_operation", False):
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    Action: {edge.merge_action}")


def example_merge_export():
    """
    Example: Export MERGE lineage to JSON.
    """
    print("\n" + "=" * 60)
    print("Example 4: Export MERGE Lineage")
    print("=" * 60)

    sql = """
    MERGE INTO target t
    USING source s ON t.id = s.id
    WHEN MATCHED THEN UPDATE SET t.value = s.new_value
    """

    pipeline = Pipeline([("merge_op", sql)], dialect="postgres")

    # Export to JSON
    exporter = JSONExporter()
    export_data = exporter.export(pipeline)

    print(f"\nQuery:\n{sql}")
    print("\nExported edges with MERGE metadata:")

    import json

    for edge in export_data.get("edges", []):
        if edge.get("is_merge_operation"):
            print(json.dumps(edge, indent=2))


if __name__ == "__main__":
    example_simple_merge()
    example_merge_with_conditions()
    example_merge_pipeline()
    example_merge_export()

    print("\n" + "=" * 60)
    print("MERGE Statement Lineage Examples Complete!")
    print("=" * 60)
