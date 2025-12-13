"""
Tests for Pipeline class template variable support.

Verifies that all Pipeline factory methods properly accept and process template_context.
"""

from clgraph import Pipeline


def test_pipeline_from_tuples_with_templates():
    """Test Pipeline.from_tuples() with template_context parameter"""
    queries = [
        ("staging", "CREATE TABLE {{env}}_staging.orders AS SELECT * FROM {{env}}_raw.orders"),
        (
            "analytics",
            "CREATE TABLE {{env}}_analytics.summary AS SELECT customer_id, SUM(amount) as total FROM {{env}}_staging.orders GROUP BY customer_id",
        ),
    ]

    pipeline = Pipeline.from_tuples(queries, dialect="bigquery", template_context={"env": "prod"})

    # Verify templates were resolved
    assert "prod_staging.orders" in pipeline.table_graph.tables
    assert "prod_analytics.summary" in pipeline.table_graph.tables
    assert "prod_raw.orders" in pipeline.table_graph.tables

    # Verify no tokenized names in lineage
    for col in pipeline.columns.values():
        if col.table_name:
            assert "__TMPL_" not in col.table_name


def test_pipeline_from_dict_with_templates():
    """Test Pipeline.from_dict() with template_context parameter"""
    queries = {
        "staging": "CREATE TABLE {{project}}.staging.orders AS SELECT * FROM {{project}}.raw.orders",
        "analytics": "CREATE TABLE {{project}}.analytics.metrics AS SELECT * FROM {{project}}.staging.orders",
    }

    pipeline = Pipeline.from_dict(
        queries, dialect="bigquery", template_context={"project": "my_project"}
    )

    # Verify templates were resolved
    assert "my_project.staging.orders" in pipeline.table_graph.tables
    assert "my_project.analytics.metrics" in pipeline.table_graph.tables
    assert "my_project.raw.orders" in pipeline.table_graph.tables


def test_pipeline_from_sql_list_with_templates():
    """Test Pipeline.from_sql_list() with template_context parameter"""
    queries = [
        "CREATE TABLE {{env}}_staging.orders AS SELECT order_id, amount FROM raw.orders",
        "CREATE TABLE {{env}}_analytics.summary AS SELECT SUM(amount) as total FROM {{env}}_staging.orders",
    ]

    pipeline = Pipeline.from_sql_list(queries, dialect="bigquery", template_context={"env": "test"})

    # Verify templates were resolved
    assert "test_staging.orders" in pipeline.table_graph.tables
    assert "test_analytics.summary" in pipeline.table_graph.tables


def test_pipeline_from_sql_string_with_templates():
    """Test Pipeline.from_sql_string() with template_context parameter"""
    sql = """
    CREATE TABLE {{env}}_staging.orders AS SELECT * FROM raw.orders;
    CREATE TABLE {{env}}_analytics.summary AS SELECT COUNT(*) as cnt FROM {{env}}_staging.orders
    """

    pipeline = Pipeline.from_sql_string(sql, dialect="bigquery", template_context={"env": "dev"})

    # Verify templates were resolved
    assert "dev_staging.orders" in pipeline.table_graph.tables
    assert "dev_analytics.summary" in pipeline.table_graph.tables


def test_pipeline_multiple_template_variables():
    """Test Pipeline with multiple template variables"""
    queries = [
        (
            "staging",
            "CREATE TABLE {{project}}.{{env}}_staging.orders AS SELECT * FROM {{project}}.raw_{{region}}.orders",
        ),
    ]

    pipeline = Pipeline.from_tuples(
        queries,
        dialect="bigquery",
        template_context={"project": "myproject", "env": "prod", "region": "us"},
    )

    # Verify all templates were resolved
    assert "myproject.prod_staging.orders" in pipeline.table_graph.tables
    assert "myproject.raw_us.orders" in pipeline.table_graph.tables


def test_pipeline_without_template_context():
    """Test that Pipeline works without template_context (preserves template syntax)"""
    queries = [
        ("staging", "CREATE TABLE {{env}}_staging.orders AS SELECT * FROM {{env}}_raw.orders"),
    ]

    pipeline = Pipeline.from_tuples(queries, dialect="bigquery")

    # Verify template syntax is preserved
    assert any("{{env}}" in table for table in pipeline.table_graph.tables)

    # Verify no tokenized names leaked
    for col in pipeline.columns.values():
        if col.table_name:
            assert "__TMPL_" not in col.table_name


def test_pipeline_template_context_stored():
    """Test that template_context is stored in Pipeline instance"""
    context = {"env": "prod", "project": "test"}

    pipeline = Pipeline.from_dict({"q1": "SELECT 1"}, dialect="bigquery", template_context=context)

    assert pipeline.template_context == context


def test_pipeline_cross_query_lineage_with_templates():
    """Test that column lineage works across templated queries"""
    queries = [
        """
        CREATE TABLE {{env}}_staging.orders AS
        SELECT
            order_id,
            customer_id,
            amount
        FROM raw.orders
        """,
        """
        CREATE TABLE {{env}}_analytics.customer_totals AS
        SELECT
            customer_id,
            SUM(amount) as total_amount
        FROM {{env}}_staging.orders
        GROUP BY customer_id
        """,
    ]

    pipeline = Pipeline.from_sql_list(queries, dialect="bigquery", template_context={"env": "prod"})

    # Verify tables exist
    assert "prod_staging.orders" in pipeline.table_graph.tables
    assert "prod_analytics.customer_totals" in pipeline.table_graph.tables

    # Verify column lineage exists
    assert len(pipeline.columns) > 0

    # Verify we can trace lineage across queries
    total_amount_col = pipeline.get_column("prod_analytics.customer_totals", "total_amount")
    assert total_amount_col is not None

    # Trace backward
    sources = pipeline.trace_column_backward("prod_analytics.customer_totals", "total_amount")
    assert len(sources) > 0

    # Should trace back to raw.orders.amount
    source_tables = {s.table_name for s in sources}
    assert "raw.orders" in source_tables
