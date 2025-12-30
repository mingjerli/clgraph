#!/usr/bin/env python3
"""
Example: Complex Aggregates Support for Column Lineage

Demonstrates how clgraph tracks column lineage through complex aggregate functions:
- Array aggregates (ARRAY_AGG, COLLECT_LIST)
- String aggregates (STRING_AGG, LISTAGG, GROUP_CONCAT)
- Object aggregates (OBJECT_AGG, JSON_AGG)
- Aggregate modifiers (DISTINCT, ORDER BY)
"""

from clgraph import JSONExporter, Pipeline


def main():
    # Example 1: Array aggregates with ORDER BY
    print("=" * 70)
    print("Example 1: Array Aggregate with ORDER BY")
    print("=" * 70)

    sql_array_agg = """
    SELECT
        customer_id,
        ARRAY_AGG(product_id ORDER BY purchase_date DESC) AS products_ordered,
        COUNT(*) AS purchase_count
    FROM purchases
    GROUP BY customer_id
    """

    pipeline = Pipeline([("array_agg_query", sql_array_agg)], dialect="bigquery")

    print("\nSQL:")
    print(sql_array_agg)

    print("\nColumn Lineage Edges:")
    for edge in pipeline.column_graph.edges:
        agg_spec = getattr(edge, "aggregate_spec", None)
        if agg_spec:
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    Function: {agg_spec.function_name}")
            print(f"    Type: {agg_spec.aggregate_type.value}")
            print(f"    Return type: {agg_spec.return_type}")
            if agg_spec.order_by:
                order_str = ", ".join(f"{o.column} {o.direction}" for o in agg_spec.order_by)
                print(f"    ORDER BY: {order_str}")
        else:
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name} [{edge.edge_type}]")

    # Example 2: String aggregates with DISTINCT
    print("\n" + "=" * 70)
    print("Example 2: String Aggregate with DISTINCT")
    print("=" * 70)

    sql_string_agg = """
    SELECT
        department,
        STRING_AGG(DISTINCT skill, ', ') AS unique_skills
    FROM employees
    GROUP BY department
    """

    pipeline2 = Pipeline([("string_agg_query", sql_string_agg)], dialect="postgres")

    print("\nSQL:")
    print(sql_string_agg)

    print("\nColumn Lineage Edges:")
    for edge in pipeline2.column_graph.edges:
        agg_spec = getattr(edge, "aggregate_spec", None)
        if agg_spec:
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    Function: {agg_spec.function_name}")
            print(f"    DISTINCT: {agg_spec.distinct}")
            print(f"    Type: {agg_spec.aggregate_type.value}")

    # Example 3: Multiple aggregates in one query
    print("\n" + "=" * 70)
    print("Example 3: Multiple Aggregate Functions")
    print("=" * 70)

    sql_multi = """
    SELECT
        category,
        SUM(amount) AS total_amount,
        AVG(amount) AS avg_amount,
        MIN(amount) AS min_amount,
        MAX(amount) AS max_amount,
        COUNT(DISTINCT customer_id) AS unique_customers
    FROM orders
    GROUP BY category
    """

    pipeline3 = Pipeline([("multi_agg_query", sql_multi)], dialect="bigquery")

    print("\nSQL:")
    print(sql_multi)

    print("\nAggregate Functions Detected:")
    agg_edges = [
        e for e in pipeline3.column_graph.edges if getattr(e, "aggregate_spec", None) is not None
    ]
    for edge in agg_edges:
        spec = edge.aggregate_spec
        print(
            f"  {spec.function_name}({'DISTINCT ' if spec.distinct else ''}"
            f"{', '.join(spec.value_columns)}) -> {edge.to_node.column_name}"
        )

    # Example 4: JSON Export of Aggregate Metadata
    print("\n" + "=" * 70)
    print("Example 4: JSON Export with Aggregate Metadata")
    print("=" * 70)

    exporter = JSONExporter()
    export_data = exporter.export(pipeline)

    import json

    print("\nExported Edge with Aggregate Spec:")
    for edge in export_data.get("edges", []):
        if edge.get("aggregate_spec"):
            print(json.dumps(edge, indent=2))
            break

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Complex aggregate support captures:
- Function name (ARRAY_AGG, STRING_AGG, SUM, etc.)
- Aggregate type (array, string, scalar, object, statistical)
- Return type inference (array, string, integer, float, etc.)
- DISTINCT modifier
- ORDER BY within aggregate
- Separator for STRING_AGG/LISTAGG
- Value and key columns

This metadata is preserved through:
- RecursiveLineageBuilder (single query analysis)
- Pipeline (multi-query analysis)
- JSON export
""")


if __name__ == "__main__":
    main()
