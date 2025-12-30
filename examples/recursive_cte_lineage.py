#!/usr/bin/env python3
"""
Example: Recursive CTE (Common Table Expression) Support for Column Lineage

Demonstrates how clgraph tracks column lineage through recursive CTEs:
- Hierarchical data (org charts, category trees)
- Graph traversal (path finding)
- Sequence generation
- Running totals
"""

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder
from clgraph.query_parser import RecursiveQueryParser


def main():
    # Example 1: Simple Recursive Sequence
    print("=" * 70)
    print("Example 1: Simple Recursive Sequence")
    print("=" * 70)

    sql_sequence = """
    WITH RECURSIVE nums AS (
        SELECT 1 AS n
        UNION ALL
        SELECT n + 1 FROM nums WHERE n < 10
    )
    SELECT n FROM nums
    """

    parser = RecursiveQueryParser(sql_sequence, dialect="postgres")
    unit_graph = parser.parse()

    print("\nSQL:")
    print(sql_sequence)

    print("\nQuery Units:")
    for unit_id, unit in unit_graph.units.items():
        print(f"  {unit_id}: {unit.unit_type.value}")
        if unit.recursive_cte_info:
            info = unit.recursive_cte_info
            print(f"    - Is Recursive: {info.is_recursive}")
            print(f"    - Union Type: {info.union_type}")
            print(f"    - Base Columns: {info.base_columns}")
            print(f"    - Self Reference Alias: {info.self_reference_alias}")

    # Example 2: Organization Hierarchy
    print("\n" + "=" * 70)
    print("Example 2: Organization Hierarchy")
    print("=" * 70)

    sql_hierarchy = """
    WITH RECURSIVE org_hierarchy AS (
        -- Base case: Top-level employees (no manager)
        SELECT id, name, manager_id, 1 AS level
        FROM employees
        WHERE manager_id IS NULL

        UNION ALL

        -- Recursive case: Employees with managers
        SELECT e.id, e.name, e.manager_id, h.level + 1
        FROM employees e
        JOIN org_hierarchy h ON e.manager_id = h.id
    )
    SELECT id, name, level FROM org_hierarchy
    """

    parser2 = RecursiveQueryParser(sql_hierarchy, dialect="postgres")
    unit_graph2 = parser2.parse()

    print("\nSQL:")
    print(sql_hierarchy)

    # Find the recursive CTE
    cte_unit = unit_graph2.get_unit_by_name("org_hierarchy")
    if cte_unit and cte_unit.recursive_cte_info:
        info = cte_unit.recursive_cte_info
        print("\nRecursive CTE Analysis:")
        print(f"  CTE Name: {info.cte_name}")
        print(f"  Self Reference Alias: {info.self_reference_alias}")
        print(f"  Join Condition: {info.join_condition}")
        print(f"  Base Columns: {info.base_columns}")
        print(f"  Recursive Columns: {info.recursive_columns}")

    # Example 3: Graph Path Traversal
    print("\n" + "=" * 70)
    print("Example 3: Graph Path Traversal")
    print("=" * 70)

    sql_paths = """
    WITH RECURSIVE reachable_nodes AS (
        -- Start from node 1
        SELECT target_id AS node_id, 1 AS hops
        FROM edges
        WHERE source_id = 1

        UNION ALL

        -- Follow edges from reachable nodes
        SELECT e.target_id, r.hops + 1
        FROM edges e
        JOIN reachable_nodes r ON e.source_id = r.node_id
        WHERE r.hops < 5
    )
    SELECT DISTINCT node_id, MIN(hops) AS min_hops
    FROM reachable_nodes
    GROUP BY node_id
    """

    parser3 = RecursiveQueryParser(sql_paths, dialect="postgres")
    unit_graph3 = parser3.parse()

    print("\nSQL:")
    print(sql_paths)

    print("\nQuery Structure:")
    for unit_id, unit in unit_graph3.units.items():
        unit_type_str = unit.unit_type.value
        if unit.recursive_cte_info:
            unit_type_str += " [RECURSIVE]"
        print(f"  {unit_id}: {unit_type_str}")

    # Example 4: Path String Accumulation
    print("\n" + "=" * 70)
    print("Example 4: Path String Accumulation")
    print("=" * 70)

    sql_path_string = """
    WITH RECURSIVE category_path AS (
        -- Root categories
        SELECT id, name, CAST(name AS VARCHAR(1000)) AS full_path
        FROM categories
        WHERE parent_id IS NULL

        UNION ALL

        -- Child categories
        SELECT c.id, c.name, CONCAT(p.full_path, ' > ', c.name)
        FROM categories c
        JOIN category_path p ON c.parent_id = p.id
    )
    SELECT id, name, full_path FROM category_path
    """

    builder = RecursiveLineageBuilder(sql_path_string, dialect="postgres")
    graph = builder.build()

    print("\nSQL:")
    print(sql_path_string)

    print("\nColumn Lineage:")
    for name, node in graph.nodes.items():
        if node.layer == "output":
            print(f"  Output: {name}")

    # Example 5: Pipeline with Recursive CTE
    print("\n" + "=" * 70)
    print("Example 5: Pipeline with Recursive CTE")
    print("=" * 70)

    queries = [
        (
            "staging",
            "CREATE TABLE staging AS SELECT id, parent_id, name FROM raw_data",
        ),
        (
            "hierarchy",
            """
            CREATE TABLE hierarchy AS
            WITH RECURSIVE tree AS (
                SELECT id, parent_id, name, 1 AS depth
                FROM staging
                WHERE parent_id IS NULL

                UNION ALL

                SELECT s.id, s.parent_id, s.name, t.depth + 1
                FROM staging s
                JOIN tree t ON s.parent_id = t.id
            )
            SELECT * FROM tree
        """,
        ),
    ]

    pipeline = Pipeline(queries, dialect="postgres")

    print("\nPipeline Queries:")
    for query_id, q in pipeline.table_graph.queries.items():
        print(f"  {query_id}: {q.operation.value}")
        if q.destination_table:
            print(f"    -> {q.destination_table}")

    print("\nTable Dependencies:")
    for table_name in pipeline.table_graph.tables:
        deps = pipeline.table_graph.get_dependencies(table_name)
        if deps:
            dep_names = [d.table_name for d in deps]
            print(f"  {table_name} <- {dep_names}")

    # Example 6: Export with Recursive CTE
    print("\n" + "=" * 70)
    print("Example 6: JSON Export with Recursive CTE")
    print("=" * 70)

    sql_export = """
    WITH RECURSIVE nums AS (
        SELECT 1 AS n
        UNION ALL
        SELECT n + 1 FROM nums WHERE n < 5
    )
    SELECT n * 2 AS doubled FROM nums
    """

    export_pipeline = Pipeline([("numbers", sql_export)], dialect="postgres")
    exporter = JSONExporter()
    export_data = exporter.export(export_pipeline)

    print("\nSQL:")
    print(sql_export)

    print("\nExported Columns:")
    for col in export_data.get("columns", []):
        print(f"  {col['table_name']}.{col['column_name']}: {col['node_type']}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(
        """
Recursive CTE support captures:
- Detection of WITH RECURSIVE keyword
- Separation of base case and recursive case
- Self-reference alias (e.g., "h" in "JOIN cte h ON ...")
- Join condition for self-reference
- Column names from both base and recursive cases
- Union type (UNION vs UNION ALL)

This metadata is available through:
- RecursiveQueryParser (query structure analysis)
- RecursiveLineageBuilder (column lineage analysis)
- Pipeline (multi-query analysis)
- JSON export

Recursive CTEs are commonly used for:
- Hierarchical data (org charts, category trees, BOMs)
- Graph traversal (finding paths, connected components)
- Sequence generation (date ranges, numbers)
- Running calculations (running totals, cumulative sums)
"""
    )


if __name__ == "__main__":
    main()
