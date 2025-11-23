"""
E-Commerce Pipeline Lineage Analysis

This script demonstrates how to use clpipe to analyze the SQL pipeline
in this example directory.
"""

from pathlib import Path

from clpipe import Pipeline


def load_sql_queries(sql_dir: Path) -> list[tuple[str, str]]:
    """Load all SQL files from a directory in sorted order."""
    queries = []
    for sql_file in sorted(sql_dir.glob("*.sql")):
        with open(sql_file) as f:
            sql = f.read()
        query_name = sql_file.stem
        queries.append((query_name, sql))
        print(f"  Loaded: {query_name}")
    return queries


def main():
    print("=" * 80)
    print("E-Commerce Pipeline Lineage Analysis")
    print("=" * 80)
    print()

    # Load SQL files from current directory
    sql_dir = Path(__file__).parent
    queries = load_sql_queries(sql_dir)

    print(f"\nLoaded {len(queries)} SQL files")
    print()

    # Build the pipeline
    print("Building pipeline...")
    pipeline = Pipeline(queries, dialect="duckdb")
    print(f"✓ Built pipeline with {len(pipeline.table_graph.queries)} queries")
    print(f"✓ Found {len(pipeline.columns)} columns")
    print()

    # -------------------------------------------------------------------------
    # 1. Query Execution Order (Topologically Sorted)
    # -------------------------------------------------------------------------
    print("1. QUERY EXECUTION ORDER (Topologically Sorted)")
    print("-" * 80)
    print("  Queries are sorted so dependencies come before dependents.\n")
    query_order = pipeline.table_graph.topological_sort()
    for i, query_id in enumerate(query_order, 1):
        query = pipeline.table_graph.queries[query_id]
        dest = query.destination_table or "(no destination)"
        print(f"  {i}. {query_id} → {dest}")
    print()

    # -------------------------------------------------------------------------
    # 2. Query Dependencies (graphlib format)
    # -------------------------------------------------------------------------
    print("2. QUERY DEPENDENCIES (graphlib format)")
    print("-" * 80)
    print("  Format: {query_id: {dependencies}} - used by graphlib.TopologicalSorter\n")

    # Helper function to extract CTE names from SQL
    def get_cte_names(sql: str) -> set:
        """Extract CTE names from a SQL query using sqlglot."""
        import sqlglot
        try:
            parsed = sqlglot.parse_one(sql, dialect="duckdb")
            cte_names = set()
            for cte in parsed.find_all(sqlglot.exp.CTE):
                if cte.alias:
                    cte_names.add(cte.alias)
            return cte_names
        except Exception:
            return set()

    # Build graphlib-style dependency dict
    query_deps = {}
    query_details = {}  # Store details for later display

    for query_id in query_order:
        query = pipeline.table_graph.queries[query_id]
        cte_names = get_cte_names(query.sql)

        # Separate source tables into categories
        ctes = []
        external_sources = []
        dep_query_ids = set()

        for table_name in sorted(query.source_tables):
            if table_name in cte_names:
                ctes.append(table_name)
            else:
                table_node = pipeline.table_graph.tables.get(table_name)
                if table_node:
                    if table_node.created_by and table_node.created_by != query_id:
                        dep_query_ids.add(table_node.created_by)
                    elif table_node.is_source and not table_node.created_by:
                        external_sources.append(table_name)

        query_deps[query_id] = dep_query_ids
        query_details[query_id] = {
            "destination": query.destination_table,
            "ctes": ctes,
            "external_sources": external_sources,
            "dep_query_ids": dep_query_ids,
        }

    # Print graphlib format
    print("  dependency_graph = {")
    for query_id in query_order:
        deps = query_deps[query_id]
        if deps:
            deps_str = "{" + ", ".join(f'"{d}"' for d in sorted(deps)) + "}"
        else:
            deps_str = "set()"
        print(f'      "{query_id}": {deps_str},')
    print("  }")
    print()

    # -------------------------------------------------------------------------
    # 3. Query Details (CTEs and sources breakdown)
    # -------------------------------------------------------------------------
    print("3. QUERY DETAILS")
    print("-" * 80)
    print("  Breakdown of each query's internal structure and dependencies.\n")

    for query_id in query_order:
        details = query_details[query_id]

        print(f"  {query_id}")
        print(f"    Creates: {details['destination'] or '(none)'}")
        if details['ctes']:
            print(f"    CTEs: {', '.join(details['ctes'])}")
        if details['external_sources']:
            print(f"    External sources: {', '.join(details['external_sources'])}")
        if details['dep_query_ids']:
            print(f"    Depends on queries: {', '.join(sorted(details['dep_query_ids']))}")
        if not details['ctes'] and not details['external_sources'] and not details['dep_query_ids']:
            print(f"    Sources: (none)")
        print()

    # -------------------------------------------------------------------------
    # 4. Backward Lineage Examples
    # -------------------------------------------------------------------------
    print("4. BACKWARD LINEAGE (Where does data come from?)")
    print("-" * 80)

    # Example: Trace customer lifetime revenue
    # Use actual destination table names from the SQL
    examples = [
        ("mart_customer_ltv", "lifetime_revenue"),
        ("int_daily_metrics", "gross_revenue"),
        ("mart_product_performance", "total_margin"),
    ]

    for table, column in examples:
        print(f"\n  Column: {table}.{column}")
        try:
            sources = pipeline.trace_column_backward(table, column)
            if sources:
                print("  Sources:")
                for source in sources[:10]:  # Limit output
                    print(f"    ← {source.table_name}.{source.column_name}")
                if len(sources) > 10:
                    print(f"    ... and {len(sources) - 10} more")
            else:
                print("    (no sources found)")
        except Exception as e:
            print(f"    Error: {e}")
    print()

    # -------------------------------------------------------------------------
    # 5. Forward Lineage (Impact Analysis)
    # -------------------------------------------------------------------------
    print("5. FORWARD LINEAGE (Impact Analysis)")
    print("-" * 80)

    # Use actual table names
    impact_examples = [
        ("raw_orders", "total_amount"),
        ("raw_customers", "customer_id"),
        ("raw_products", "unit_cost"),
    ]

    for table, column in impact_examples:
        print(f"\n  Column: {table}.{column}")
        try:
            impacts = pipeline.trace_column_forward(table, column)
            if impacts:
                print("  Impacts:")
                for impact in impacts[:10]:  # Limit output
                    print(f"    → {impact.table_name}.{impact.column_name}")
                if len(impacts) > 10:
                    print(f"    ... and {len(impacts) - 10} more")
            else:
                print("    (no downstream impacts)")
        except Exception as e:
            print(f"    Error: {e}")
    print()

    # -------------------------------------------------------------------------
    # 6. PII Tracking
    # -------------------------------------------------------------------------
    print("6. PII COLUMN TRACKING")
    print("-" * 80)

    # Mark known PII columns (use actual table names)
    pii_columns = [
        ("raw_customers", "email"),
        ("raw_customers", "phone_number"),
        ("raw_orders", "ip_address"),
        ("raw_orders", "shipping_address"),
    ]

    marked_count = 0
    for table, column in pii_columns:
        key = f"{table}.{column}"
        if key in pipeline.columns:
            pipeline.columns[key].pii = True
            marked_count += 1
    print(f"  Marked {marked_count} columns as PII")

    # Propagate PII metadata through lineage
    pipeline.propagate_all_metadata()

    # Find all PII columns
    all_pii = list(pipeline.get_pii_columns())
    print(f"  PII columns in pipeline ({len(all_pii)}):")
    for col in sorted(all_pii, key=lambda c: f"{c.table_name}.{c.column_name}"):
        print(f"    ⚠️  {col.table_name}.{col.column_name}")
    print()

    # -------------------------------------------------------------------------
    # 7. Single Query Analysis
    # -------------------------------------------------------------------------
    print("7. SINGLE QUERY DEEP DIVE")
    print("-" * 80)

    # Show columns in the final mart tables
    print("  Columns in mart_customer_ltv:")
    ltv_cols = [
        col for col in pipeline.columns.values()
        if col.table_name == "mart_customer_ltv"
    ]
    for col in sorted(ltv_cols, key=lambda c: c.column_name)[:15]:
        print(f"    • {col.column_name}")
    if len(ltv_cols) > 15:
        print(f"    ... and {len(ltv_cols) - 15} more")
    print()

    # -------------------------------------------------------------------------
    # 8. Table Graph Methods
    # -------------------------------------------------------------------------
    print("8. TABLE GRAPH METHODS (pipeline.table_graph)")
    print("-" * 80)
    print("  Demonstrates available methods on the TableDependencyGraph.\n")

    # 8a. Get source tables (external tables not created by any query)
    print("  8a. Source Tables (external inputs):")
    source_tables = pipeline.table_graph.get_source_tables()
    for table in source_tables[:10]:
        print(f"      📥 {table.table_name}")
    if len(source_tables) > 10:
        print(f"      ... and {len(source_tables) - 10} more")
    print()

    # 8b. Get final tables (not read by any downstream query)
    print("  8b. Final Tables (pipeline outputs):")
    final_tables = pipeline.table_graph.get_final_tables()
    for table in final_tables[:10]:
        print(f"      📤 {table.table_name}")
    if len(final_tables) > 10:
        print(f"      ... and {len(final_tables) - 10} more")
    print()

    # 8c. Get execution order (tables in dependency order)
    print("  8c. Table Execution Order:")
    execution_order = pipeline.table_graph.get_execution_order()
    for i, table in enumerate(execution_order[:15], 1):
        table_type = "source" if table.is_source else "created"
        print(f"      {i}. {table.table_name} ({table_type})")
    if len(execution_order) > 15:
        print(f"      ... and {len(execution_order) - 15} more")
    print()

    # 8d. Get dependencies for a specific table
    print("  8d. Table Dependencies (upstream tables):")
    example_tables = ["mart_customer_ltv", "int_daily_metrics"]
    for table_name in example_tables:
        if table_name in pipeline.table_graph.tables:
            deps = pipeline.table_graph.get_dependencies(table_name)
            print(f"      {table_name} depends on:")
            if deps:
                for dep in deps[:5]:
                    print(f"        ← {dep.table_name}")
                if len(deps) > 5:
                    print(f"        ... and {len(deps) - 5} more")
            else:
                print("        (no dependencies)")
    print()

    # 8e. Get downstream tables
    print("  8e. Table Downstream (impact analysis):")
    example_sources = ["raw_orders", "raw_customers"]
    for table_name in example_sources:
        if table_name in pipeline.table_graph.tables:
            downstream = pipeline.table_graph.get_downstream(table_name)
            print(f"      {table_name} is used by:")
            if downstream:
                for ds in downstream[:5]:
                    print(f"        → {ds.table_name}")
                if len(downstream) > 5:
                    print(f"        ... and {len(downstream) - 5} more")
            else:
                print("        (no downstream tables)")
    print()

    # 8f. Access table and query objects directly
    print("  8f. Direct Access to Tables and Queries:")
    print(f"      Total tables: {len(pipeline.table_graph.tables)}")
    print(f"      Total queries: {len(pipeline.table_graph.queries)}")
    print()
    print("      Example table access:")
    if "raw_orders" in pipeline.table_graph.tables:
        table = pipeline.table_graph.tables["raw_orders"]
        print("        table = pipeline.table_graph.tables['raw_orders']")
        print(f"        table.is_source = {table.is_source}")
        print(f"        table.read_by = {table.read_by[:3]}{'...' if len(table.read_by) > 3 else ''}")
    print()
    print("      Example query access:")
    first_query_id = list(pipeline.table_graph.queries.keys())[0]
    first_query = pipeline.table_graph.queries[first_query_id]
    print(f"        query = pipeline.table_graph.queries['{first_query_id}']")
    print(f"        query.destination_table = {first_query.destination_table}")
    print(f"        query.source_tables = {list(first_query.source_tables)[:3]}{'...' if len(first_query.source_tables) > 3 else ''}")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"  Total queries:  {len(queries)}")
    print(f"  Total tables:   {len(pipeline.table_graph.tables)}")
    print(f"  Total columns:  {len(pipeline.columns)}")
    print(f"  PII columns:    {len(list(all_pii))}")
    print()
    print("Analysis complete!")


if __name__ == "__main__":
    main()
