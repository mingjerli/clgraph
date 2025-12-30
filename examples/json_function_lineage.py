"""
Example: JSON Function Lineage Tracking

This example demonstrates how clgraph tracks column lineage through
JSON extraction functions like JSON_EXTRACT, JSON_VALUE, and JSON_QUERY.

Key features demonstrated:
1. JSON function detection across dialects (BigQuery, PostgreSQL, Snowflake)
2. JSON path extraction and normalization
3. Lineage edges with JSON metadata (json_path, json_function)
4. Multi-query pipelines with JSON extraction chains
"""

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder


def example_simple_json_lineage():
    """
    Basic example: Single query with JSON extraction.
    """
    print("=" * 60)
    print("Example 1: Simple JSON Extraction Lineage")
    print("=" * 60)

    sql = """
    SELECT
        id,
        JSON_EXTRACT(user_data, '$.address.city') AS city,
        JSON_EXTRACT(user_data, '$.address.zip') AS zip_code,
        JSON_VALUE(user_data, '$.email') AS email
    FROM users
    """

    # Use RecursiveLineageBuilder for single-query analysis
    builder = RecursiveLineageBuilder(sql, dialect="bigquery")
    graph = builder.build()

    print(f"\nQuery:\n{sql}")
    print("\nLineage Edges:")

    for edge in graph.edges:
        line = f"  {edge.from_node.full_name} -> {edge.to_node.full_name}"
        if edge.json_path:
            line += f" [JSON: {edge.json_function}, path: {edge.json_path}]"
        print(line)


def example_json_in_cte():
    """
    Example: JSON extraction in Common Table Expression (CTE).
    """
    print("\n" + "=" * 60)
    print("Example 2: JSON Extraction in CTE")
    print("=" * 60)

    sql = """
    WITH extracted AS (
        SELECT
            id,
            JSON_EXTRACT(profile, '$.name') AS name,
            JSON_EXTRACT(profile, '$.settings.theme') AS theme
        FROM raw_users
    )
    SELECT id, name, theme
    FROM extracted
    WHERE name IS NOT NULL
    """

    pipeline = Pipeline([("query", sql)], dialect="bigquery")

    print(f"\nQuery:\n{sql}")
    print("\nJSON-annotated edges:")

    for edge in pipeline.column_graph.edges:
        if edge.json_path:
            print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name}")
            print(f"    JSON function: {edge.json_function}")
            print(f"    JSON path: {edge.json_path}")


def example_multi_query_json_chain():
    """
    Example: JSON extraction across multiple queries in a pipeline.

    Shows how JSON metadata is tracked through query chains.
    """
    print("\n" + "=" * 60)
    print("Example 3: Multi-Query JSON Extraction Chain")
    print("=" * 60)

    queries = [
        (
            "staging",
            """
            CREATE TABLE staging AS
            SELECT
                event_id,
                event_time,
                JSON_EXTRACT(payload, '$.user') AS user_json
            FROM raw_events
            """,
        ),
        (
            "final",
            """
            CREATE TABLE final AS
            SELECT
                event_id,
                event_time,
                JSON_EXTRACT(user_json, '$.name') AS user_name,
                JSON_EXTRACT(user_json, '$.email') AS user_email
            FROM staging
            """,
        ),
    ]

    pipeline = Pipeline(queries, dialect="bigquery")

    print("\nQueries:")
    for query_id, sql in queries:
        print(f"\n{query_id}:")
        print(f"{sql}")

    print("\n\nFull lineage chain with JSON metadata:")

    # Group edges by destination table
    edges_by_table = {}
    for edge in pipeline.column_graph.edges:
        table = edge.to_node.table_name
        if table not in edges_by_table:
            edges_by_table[table] = []
        edges_by_table[table].append(edge)

    for table, edges in edges_by_table.items():
        print(f"\n  Table: {table}")
        for edge in edges:
            line = f"    {edge.from_node.full_name} -> {edge.to_node.full_name}"
            if edge.json_path:
                line += f"\n      [JSON: {edge.json_function} at {edge.json_path}]"
            print(line)


def example_json_export():
    """
    Example: Export JSON function lineage to JSON format.
    """
    print("\n" + "=" * 60)
    print("Example 4: Export JSON Lineage")
    print("=" * 60)

    sql = """
    SELECT
        JSON_EXTRACT(data, '$.user.name') AS user_name,
        JSON_EXTRACT(data, '$.user.email') AS user_email
    FROM events
    """

    pipeline = Pipeline([("query", sql)], dialect="bigquery")

    # Export to JSON
    exporter = JSONExporter()
    export_data = exporter.export(pipeline)

    print(f"\nQuery:\n{sql}")
    print("\nExported edges with JSON metadata:")

    import json

    for edge in export_data.get("edges", []):
        if edge.get("json_path"):
            print(json.dumps(edge, indent=2))


def example_various_json_functions():
    """
    Example: Various JSON functions and how they're tracked.
    """
    print("\n" + "=" * 60)
    print("Example 5: Various JSON Functions")
    print("=" * 60)

    # Different JSON function syntaxes
    examples = [
        ("BigQuery JSON_EXTRACT", "SELECT JSON_EXTRACT(col, '$.path') AS val FROM t"),
        ("BigQuery JSON_VALUE", "SELECT JSON_VALUE(col, '$.path') AS val FROM t"),
        ("BigQuery JSON_QUERY", "SELECT JSON_QUERY(col, '$.nested') AS val FROM t"),
    ]

    for name, sql in examples:
        print(f"\n{name}:")
        print(f"  SQL: {sql}")

        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        for edge in graph.edges:
            if edge.json_path:
                print(f"  Detected: {edge.json_function} with path {edge.json_path}")


if __name__ == "__main__":
    example_simple_json_lineage()
    example_json_in_cte()
    example_multi_query_json_chain()
    example_json_export()
    example_various_json_functions()

    print("\n" + "=" * 60)
    print("JSON Function Lineage Examples Complete!")
    print("=" * 60)
