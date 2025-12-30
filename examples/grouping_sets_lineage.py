"""
Example: GROUPING SETS/CUBE/ROLLUP Lineage Tracking

This example demonstrates how clgraph tracks column lineage through
GROUPING SETS, CUBE, and ROLLUP constructs for multi-level aggregations.

Key features demonstrated:
1. CUBE clause parsing and expansion
2. ROLLUP clause parsing and expansion
3. GROUPING SETS explicit definition
4. Column lineage for grouping constructs
5. Export format with grouping metadata
"""

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder
from clgraph.query_parser import RecursiveQueryParser


def example_cube():
    """
    Example: CUBE generates all combinations of grouping columns.
    """
    print("=" * 60)
    print("Example 1: CUBE for All Combinations")
    print("=" * 60)

    sql = """
    SELECT region, product, SUM(sales) as total_sales
    FROM sales_data
    GROUP BY CUBE(region, product)
    """

    # Parse to see grouping configuration
    parser = RecursiveQueryParser(sql, dialect="bigquery")
    graph = parser.parse()

    print(f"\nQuery:\n{sql}")
    print("\nGrouping Configuration:")
    config = graph.units["main"].grouping_config
    print(f"  Type: {config.get('grouping_type')}")
    print(f"  Columns: {config.get('grouping_columns')}")
    print(f"  Expanded Sets: {config.get('grouping_sets')}")

    # Build lineage
    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    lineage_graph = builder.build()

    print("\nLineage Edges:")
    for edge in lineage_graph.edges:
        line = f"  {edge.from_node.full_name} -> {edge.to_node.full_name}"
        if edge.is_grouping_column:
            line += f" [GROUPING {edge.grouping_type}]"
        print(line)


def example_rollup():
    """
    Example: ROLLUP generates hierarchical groupings.
    """
    print("\n" + "=" * 60)
    print("Example 2: ROLLUP for Hierarchical Totals")
    print("=" * 60)

    sql = """
    SELECT year, quarter, month, SUM(revenue) as total_revenue
    FROM revenue_data
    GROUP BY ROLLUP(year, quarter, month)
    """

    parser = RecursiveQueryParser(sql, dialect="bigquery")
    graph = parser.parse()

    print(f"\nQuery:\n{sql}")
    print("\nGrouping Configuration:")
    config = graph.units["main"].grouping_config
    print(f"  Type: {config.get('grouping_type')}")
    print(f"  Columns: {config.get('grouping_columns')}")
    print("  Expanded Sets:")
    for s in config.get("grouping_sets", []):
        print(f"    {s}")

    # Build lineage
    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    lineage_graph = builder.build()

    print("\nGrouping Edges:")
    for edge in lineage_graph.edges:
        if edge.is_grouping_column:
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    grouping_type: {edge.grouping_type}")


def example_grouping_sets():
    """
    Example: GROUPING SETS with explicit combinations.
    """
    print("\n" + "=" * 60)
    print("Example 3: Explicit GROUPING SETS")
    print("=" * 60)

    sql = """
    SELECT region, category, SUM(amount)
    FROM transactions
    GROUP BY GROUPING SETS (
        (region, category),
        (region),
        ()
    )
    """

    parser = RecursiveQueryParser(sql, dialect="postgres")
    graph = parser.parse()

    print(f"\nQuery:\n{sql}")
    print("\nGrouping Configuration:")
    config = graph.units["main"].grouping_config
    print(f"  Type: {config.get('grouping_type')}")
    print(f"  Sets: {config.get('grouping_sets')}")


def example_pipeline():
    """
    Example: CUBE lineage through Pipeline API.
    """
    print("\n" + "=" * 60)
    print("Example 4: CUBE in Pipeline")
    print("=" * 60)

    sql = """
    SELECT region, product, SUM(sales) as total_sales
    FROM sales_data
    GROUP BY CUBE(region, product)
    """

    pipeline = Pipeline([("cube_agg", sql)], dialect="bigquery")

    print(f"\nQuery:\n{sql}")
    print("\nGrouping edges in pipeline:")

    for edge in pipeline.column_graph.edges:
        if getattr(edge, "is_grouping_column", False):
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    grouping_type: {edge.grouping_type}")


def example_export():
    """
    Example: Export grouping lineage to JSON.
    """
    print("\n" + "=" * 60)
    print("Example 5: Export Grouping Lineage")
    print("=" * 60)

    sql = """
    SELECT region, SUM(sales) as total
    FROM sales_data
    GROUP BY CUBE(region)
    """

    pipeline = Pipeline([("cube_query", sql)], dialect="bigquery")

    exporter = JSONExporter()
    export_data = exporter.export(pipeline)

    print(f"\nQuery:\n{sql}")
    print("\nExported edges with grouping metadata:")

    import json

    for edge in export_data.get("edges", []):
        if edge.get("is_grouping_column"):
            print(json.dumps(edge, indent=2))


if __name__ == "__main__":
    example_cube()
    example_rollup()
    example_grouping_sets()
    example_pipeline()
    example_export()

    print("\n" + "=" * 60)
    print("GROUPING SETS/CUBE/ROLLUP Lineage Examples Complete!")
    print("=" * 60)
