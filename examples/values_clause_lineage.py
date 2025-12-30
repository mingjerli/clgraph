#!/usr/bin/env python3
"""
Example: VALUES Clause (Inline Table Literal) Support for Column Lineage

Demonstrates how clgraph tracks column lineage through VALUES clauses:
- Inline literal data detection
- Column alias extraction
- Type inference from sample values
- Lineage through literal columns
"""

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder
from clgraph.query_parser import RecursiveQueryParser


def main():
    # Example 1: Simple VALUES clause
    print("=" * 70)
    print("Example 1: Simple VALUES Clause")
    print("=" * 70)

    sql_simple = """
    SELECT id, name
    FROM (VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')) AS t(id, name)
    """

    parser = RecursiveQueryParser(sql_simple, dialect="postgres")
    unit_graph = parser.parse()

    print("\nSQL:")
    print(sql_simple)

    unit = unit_graph.units["main"]
    print("\nVALUES Sources:")
    for alias, values_info in unit.values_sources.items():
        print(f"  {alias}:")
        print(f"    Columns: {values_info.column_names}")
        print(f"    Types: {values_info.column_types}")
        print(f"    Rows: {values_info.row_count}")
        print(f"    Sample: {values_info.sample_values}")

    # Example 2: Column lineage
    print("\n" + "=" * 70)
    print("Example 2: Column Lineage Through VALUES")
    print("=" * 70)

    sql_lineage = """
    SELECT id, name FROM (VALUES (1, 'Alice'), (2, 'Bob')) AS t(id, name)
    """

    builder = RecursiveLineageBuilder(sql_lineage, dialect="postgres")
    graph = builder.build()

    print("\nSQL:")
    print(sql_lineage)

    print("\nColumn Nodes:")
    for name, node in graph.nodes.items():
        literal_marker = " [LITERAL]" if node.is_literal else ""
        if node.is_literal:
            print(f"  {name}{literal_marker}")
            print(f"    Type: {node.literal_type}")
            print(f"    Values: {node.literal_values}")
        else:
            print(f"  {name}")

    print("\nEdges:")
    for edge in graph.edges:
        print(f"  {edge.from_node.full_name} -> {edge.to_node.full_name} [{edge.edge_type}]")

    # Example 3: VALUES with JOIN
    print("\n" + "=" * 70)
    print("Example 3: VALUES with JOIN")
    print("=" * 70)

    sql_join = """
    SELECT u.id, u.name, l.label
    FROM users u
    JOIN (VALUES (1, 'admin'), (2, 'user')) AS l(id, label)
    ON u.role_id = l.id
    """

    parser2 = RecursiveQueryParser(sql_join, dialect="postgres")
    unit_graph2 = parser2.parse()

    print("\nSQL:")
    print(sql_join)

    unit2 = unit_graph2.units["main"]
    print("\nTables and VALUES sources:")
    print(f"  Tables: {unit2.depends_on_tables}")
    print(f"  VALUES sources: {list(unit2.values_sources.keys())}")

    # Example 4: VALUES in Pipeline
    print("\n" + "=" * 70)
    print("Example 4: VALUES in Pipeline")
    print("=" * 70)

    sql_pipeline = """
    CREATE TABLE enriched AS
    SELECT id, name FROM (VALUES (1, 'A'), (2, 'B')) AS t(id, name)
    """

    pipeline = Pipeline([("create_lookup", sql_pipeline)], dialect="postgres")

    print("\nSQL:")
    print(sql_pipeline)

    print("\nPipeline columns:")
    for name, col in pipeline.column_graph.columns.items():
        if col.is_literal:
            print(f"  {name} [LITERAL, type={col.literal_type}]")
        else:
            print(f"  {name}")

    # Example 5: Type inference
    print("\n" + "=" * 70)
    print("Example 5: Type Inference")
    print("=" * 70)

    sql_types = """
    SELECT *
    FROM (VALUES
        (1, 'text', 3.14, true),
        (2, 'more', 2.71, false)
    ) AS t(int_col, str_col, float_col, bool_col)
    """

    parser3 = RecursiveQueryParser(sql_types, dialect="postgres")
    unit_graph3 = parser3.parse()

    print("\nSQL:")
    print(sql_types)

    values_info = list(unit_graph3.units["main"].values_sources.values())[0]
    print("\nInferred Types:")
    for col_name, col_type in zip(values_info.column_names, values_info.column_types, strict=False):
        print(f"  {col_name}: {col_type}")

    # Example 6: JSON Export
    print("\n" + "=" * 70)
    print("Example 6: JSON Export with Literal Metadata")
    print("=" * 70)

    sql_export = """
    SELECT id, name FROM (VALUES (1, 'Alice')) AS t(id, name)
    """

    export_pipeline = Pipeline([("values_query", sql_export)], dialect="postgres")
    exporter = JSONExporter()
    export_data = exporter.export(export_pipeline)

    print("\nSQL:")
    print(sql_export)

    import json

    print("\nExported literal columns:")
    for col in export_data.get("columns", []):
        if col.get("is_literal"):
            print(json.dumps(col, indent=2))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(
        """
VALUES clause support captures:
- Inline literal data detection (table literals)
- Column aliases from AS t(col1, col2) syntax
- Type inference from values (integer, string, numeric, boolean)
- Sample values stored for reference
- Literal column nodes in lineage graph

This metadata is preserved through:
- RecursiveQueryParser (query structure analysis)
- RecursiveLineageBuilder (column lineage analysis)
- Pipeline (multi-query analysis)
- JSON export

VALUES clauses are commonly used for:
- Test data and examples
- Lookup/mapping tables
- Static configuration data
- Small inline reference tables
"""
    )


if __name__ == "__main__":
    main()
