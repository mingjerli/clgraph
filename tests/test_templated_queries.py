"""
Tests for templated query parsing and lineage building.

Ensures that template variables like {{project}} are properly handled
and don't break table name resolution in column lineage.
"""

from clgraph.multi_query import MultiQueryParser
from clgraph.pipeline import PipelineLineageBuilder


def test_templated_query_with_joins():
    """
    Test that templated queries with joins properly resolve table names.

    This test verifies the fix for the bug where SQL containing templates
    like {{project}}.orders was tokenized to __TMPL_0__.orders but never
    restored before being passed to RecursiveLineageBuilder, causing
    table_name=None and breaking metadata propagation.
    """
    queries = [
        # Query with templated table names in JOIN
        """
        CREATE OR REPLACE TABLE {{project}}.gold.customer_orders AS
        SELECT
            c.customer_id,
            c.customer_name,
            o.order_id,
            o.order_date,
            o.amount
        FROM {{project}}.raw.customers c
        JOIN {{project}}.raw.orders o
        ON c.customer_id = o.customer_id
        """
    ]

    # Parse with template resolution
    parser = MultiQueryParser(dialect="bigquery")
    context = {"project": "prod"}
    table_graph = parser.parse_queries(queries, template_context=context)

    # Verify table names are resolved
    assert "prod.gold.customer_orders" in table_graph.tables
    assert "prod.raw.customers" in table_graph.tables
    assert "prod.raw.orders" in table_graph.tables

    # Build column lineage
    builder = PipelineLineageBuilder()
    lineage_graph = builder.build(table_graph)

    # MAIN VERIFICATION: No tokenized table names (this was the bug!)
    for col in lineage_graph.columns.values():
        if col.table_name:
            assert "__TMPL_" not in col.table_name, (
                f"Column {col.full_name} has tokenized table name: {col.table_name}"
            )

    # Verify columns exist and have resolved table names
    assert len(lineage_graph.columns) > 0, "No columns found in lineage graph"

    # Verify we have columns from all three tables
    all_table_names = {col.table_name for col in lineage_graph.columns.values() if col.table_name}
    assert "prod.raw.customers" in all_table_names
    assert "prod.raw.orders" in all_table_names
    assert "prod.gold.customer_orders" in all_table_names


def test_templated_query_without_context():
    """
    Test that templated queries work even without context resolution.
    Templates should still be preserved in their original form.
    """
    queries = [
        """
        CREATE OR REPLACE TABLE {{project}}.analytics.summary AS
        SELECT
            customer_id,
            COUNT(*) as order_count
        FROM {{project}}.raw.orders
        GROUP BY customer_id
        """
    ]

    # Parse WITHOUT template resolution
    parser = MultiQueryParser(dialect="bigquery")
    table_graph = parser.parse_queries(queries)

    # Verify template syntax is preserved in table names
    assert any("{{project}}" in table for table in table_graph.tables)

    # Build column lineage
    builder = PipelineLineageBuilder()
    lineage_graph = builder.build(table_graph)

    # Verify columns have table names with template syntax, NOT tokens
    for col in lineage_graph.columns.values():
        if col.table_name:
            # Should NOT have tokenized names
            assert "__TMPL_" not in col.table_name, (
                f"Column {col.full_name} has tokenized table name"
            )
            # Should have original template syntax
            if col.node_type != "intermediate":
                assert "{{project}}" in col.table_name or col.table_name is None


def test_multi_query_templated_pipeline():
    """
    Test multiple queries with templates to ensure cross-query edges work.
    """
    queries = [
        # Step 1: Create staging table
        """
        CREATE OR REPLACE TABLE {{env}}_staging.orders AS
        SELECT order_id, customer_id, amount
        FROM {{env}}_raw.orders
        WHERE amount > 0
        """,
        # Step 2: Join with staging table
        """
        CREATE OR REPLACE TABLE {{env}}_analytics.customer_totals AS
        SELECT
            customer_id,
            SUM(amount) as total_amount
        FROM {{env}}_staging.orders
        GROUP BY customer_id
        """,
    ]

    # Parse with context
    parser = MultiQueryParser(dialect="bigquery")
    context = {"env": "dev"}
    table_graph = parser.parse_queries(queries, template_context=context)

    # Build lineage
    builder = PipelineLineageBuilder()
    lineage_graph = builder.build(table_graph)

    # MAIN VERIFICATION: No tokenized table names (this was the bug!)
    for col in lineage_graph.columns.values():
        if col.table_name:
            assert "__TMPL_" not in col.table_name, (
                f"Column {col.full_name} has tokenized table name: {col.table_name}"
            )

    # Verify columns exist and have resolved environment prefix
    assert len(lineage_graph.columns) > 0, "No columns found in lineage graph"
    all_table_names = {col.table_name for col in lineage_graph.columns.values() if col.table_name}

    # All tables should have "dev" prefix
    for table_name in all_table_names:
        assert "dev" in table_name, (
            f"Table name '{table_name}' doesn't contain resolved environment 'dev'"
        )


def test_templated_f_string_style():
    """
    Test f-string style templates: {var} instead of {{var}}
    Templates should be preserved (not necessarily resolved to context values).
    """
    queries = [
        """
        CREATE OR REPLACE TABLE {project}.results AS
        SELECT * FROM {project}.source_data
        """
    ]

    parser = MultiQueryParser(dialect="bigquery")
    table_graph = parser.parse_queries(queries)

    # Verify templates are preserved (f-string syntax)
    assert any("{project}" in table for table in table_graph.tables)

    # Build lineage
    builder = PipelineLineageBuilder()
    lineage_graph = builder.build(table_graph)

    # Verify no tokenized names (main goal of the fix)
    for col in lineage_graph.columns.values():
        if col.table_name:
            assert "__TMPL_" not in col.table_name, (
                f"Column {col.full_name} has tokenized table name"
            )
