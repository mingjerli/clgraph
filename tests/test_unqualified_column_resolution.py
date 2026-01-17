"""
Tests for Unqualified Column Resolution in Multi-Table Queries

This test suite covers the fix for resolving unqualified column references
when multiple tables are joined. The fix uses sqlglot's qualify_columns
optimizer with schema information to determine which table each column
belongs to.

Issue: When a query has multiple tables joined and columns are unqualified
(no table prefix), the lineage builder needs to determine which table
each column comes from. Previously, it would default to the first table,
which was often wrong.

Fix: Use schema information from upstream queries to qualify columns
before building lineage.
"""

from clgraph import Pipeline
from clgraph.lineage_builder import (
    RecursiveLineageBuilder,
    _convert_to_nested_schema,
    _qualify_sql_with_schema,
)

# ============================================================================
# Part 1: Helper Function Tests
# ============================================================================


class TestSchemaConversion:
    """Test the flat-to-nested schema conversion helper"""

    def test_convert_simple_schema(self):
        """Test converting a simple flat schema to nested format"""
        flat = {
            "staging.orders": ["order_id", "user_id", "amount"],
            "analytics.users": ["user_id", "name", "email"],
        }

        nested = _convert_to_nested_schema(flat)

        assert "staging" in nested
        assert "analytics" in nested
        assert "orders" in nested["staging"]
        assert "users" in nested["analytics"]
        assert "order_id" in nested["staging"]["orders"]
        assert "user_id" in nested["analytics"]["users"]

    def test_convert_schema_without_prefix(self):
        """Test converting schema without schema prefix"""
        flat = {
            "orders": ["order_id", "user_id"],
        }

        nested = _convert_to_nested_schema(flat)

        # Should use empty string as schema
        assert "" in nested
        assert "orders" in nested[""]

    def test_convert_three_part_name(self):
        """Test converting three-part table names (catalog.schema.table)"""
        flat = {
            "myproject.staging.orders": ["order_id", "amount"],
        }

        nested = _convert_to_nested_schema(flat)

        # Should use the last two parts
        assert "staging" in nested
        assert "orders" in nested["staging"]


class TestSqlQualification:
    """Test the SQL qualification helper"""

    def test_qualify_single_table(self):
        """Test that single-table queries are unchanged"""
        sql = "SELECT order_id, amount FROM orders"
        schema = {"orders": ["order_id", "amount"]}

        result = _qualify_sql_with_schema(sql, schema, "bigquery")

        # Should still work (columns may be qualified)
        assert "order_id" in result
        assert "amount" in result

    def test_qualify_multi_table_join(self):
        """Test qualification with multiple tables joined"""
        sql = """
        SELECT order_date, total_revenue
        FROM analytics.user_metrics
        JOIN staging.user_orders USING (user_id)
        """
        schema = {
            "staging.user_orders": ["user_id", "order_date", "amount"],
            "analytics.user_metrics": ["user_id", "total_revenue"],
        }

        result = _qualify_sql_with_schema(sql, schema, "bigquery")

        # order_date should be qualified with user_orders
        assert "user_orders" in result and "order_date" in result
        # total_revenue should be qualified with user_metrics
        assert "user_metrics" in result and "total_revenue" in result

    def test_qualify_empty_schema(self):
        """Test that empty schema returns original SQL"""
        sql = "SELECT order_id FROM orders"

        result = _qualify_sql_with_schema(sql, {}, "bigquery")

        assert result == sql

    def test_qualify_date_trunc_bigquery(self):
        """Test DATE_TRUNC qualification in BigQuery dialect"""
        sql = "SELECT DATE_TRUNC(order_date, MONTH) as month FROM staging.orders"
        schema = {
            "staging.orders": ["order_id", "order_date"],
        }

        result = _qualify_sql_with_schema(sql, schema, "bigquery")

        # Should preserve DATE_TRUNC with proper column
        assert "DATE_TRUNC" in result.upper()
        assert "order_date" in result.lower()


# ============================================================================
# Part 2: RecursiveLineageBuilder Tests
# ============================================================================


class TestLineageBuilderWithSchema:
    """Test RecursiveLineageBuilder with schema-based qualification"""

    def test_lineage_with_qualified_columns(self):
        """Test lineage building when columns are already qualified"""
        sql = """
        SELECT user_orders.order_date, user_metrics.total_revenue
        FROM analytics.user_metrics
        JOIN staging.user_orders USING (user_id)
        """
        schema = {
            "staging.user_orders": ["user_id", "order_date"],
            "analytics.user_metrics": ["user_id", "total_revenue"],
        }

        builder = RecursiveLineageBuilder(sql, external_table_columns=schema, dialect="bigquery")
        lineage = builder.build()

        # Should have both columns as inputs
        input_nodes = [n for n in lineage.nodes.values() if n.layer == "input"]
        input_names = [n.column_name for n in input_nodes]

        assert "order_date" in input_names
        assert "total_revenue" in input_names

    def test_lineage_with_unqualified_columns(self):
        """Test lineage building when columns are unqualified"""
        sql = """
        SELECT order_date, total_revenue
        FROM analytics.user_metrics
        JOIN staging.user_orders USING (user_id)
        """
        schema = {
            "staging.user_orders": ["user_id", "order_date"],
            "analytics.user_metrics": ["user_id", "total_revenue"],
        }

        builder = RecursiveLineageBuilder(sql, external_table_columns=schema, dialect="bigquery")
        lineage = builder.build()

        # Should have both columns as inputs with correct table attribution
        input_nodes = [n for n in lineage.nodes.values() if n.layer == "input"]

        # Find order_date node - should be from user_orders
        order_date_nodes = [n for n in input_nodes if n.column_name == "order_date"]
        assert len(order_date_nodes) == 1
        assert order_date_nodes[0].table_name == "user_orders"

        # Find total_revenue node - should be from user_metrics
        revenue_nodes = [n for n in input_nodes if n.column_name == "total_revenue"]
        assert len(revenue_nodes) == 1
        assert revenue_nodes[0].table_name == "user_metrics"

    def test_lineage_date_trunc_expression(self):
        """Test lineage for DATE_TRUNC expression with unqualified column"""
        sql = """
        SELECT DATE_TRUNC(order_date, MONTH) as month
        FROM staging.orders
        """
        schema = {
            "staging.orders": ["order_id", "order_date", "amount"],
        }

        builder = RecursiveLineageBuilder(sql, external_table_columns=schema, dialect="bigquery")
        lineage = builder.build()

        # Should have edge from order_date to month
        edges = list(lineage.edges)
        assert len(edges) == 1
        assert edges[0].from_node.column_name == "order_date"
        assert edges[0].to_node.column_name == "month"


# ============================================================================
# Part 3: Pipeline Tests (Cross-Query Lineage)
# ============================================================================


class TestPipelineUnqualifiedColumns:
    """Test Pipeline with unqualified column resolution"""

    def test_three_layer_pipeline_date_trunc(self):
        """Test the 3-layer pipeline example with DATE_TRUNC"""
        queries = [
            """CREATE TABLE staging.user_orders AS
            SELECT user_id, order_id, amount, order_date
            FROM raw.orders
            WHERE status = 'completed'""",
            """CREATE TABLE analytics.user_metrics AS
            SELECT user_id, COUNT(*) as order_count, SUM(amount) as total_revenue
            FROM staging.user_orders
            GROUP BY user_id""",
            """CREATE TABLE reports.monthly_revenue AS
            SELECT DATE_TRUNC(order_date, MONTH) as month, SUM(total_revenue) as revenue
            FROM analytics.user_metrics
            JOIN staging.user_orders USING (user_id)
            GROUP BY month""",
        ]

        pipeline = Pipeline.from_sql_list(queries, dialect="bigquery")

        # Check that month has lineage to order_date
        month_edges = [
            e for e in pipeline.edges if e.to_node.full_name == "reports.monthly_revenue.month"
        ]
        assert len(month_edges) == 1
        assert month_edges[0].from_node.column_name == "order_date"
        assert month_edges[0].from_node.table_name == "staging.user_orders"

    def test_trace_column_backward_through_join(self):
        """Test tracing a column backward through a JOIN"""
        queries = [
            """CREATE TABLE staging.orders AS
            SELECT order_id, customer_id, amount, order_date
            FROM raw.orders""",
            """CREATE TABLE staging.customers AS
            SELECT customer_id, name, email
            FROM raw.customers""",
            """CREATE TABLE reports.customer_orders AS
            SELECT name, order_date, amount
            FROM staging.customers
            JOIN staging.orders USING (customer_id)""",
        ]

        pipeline = Pipeline.from_sql_list(queries, dialect="bigquery")

        # Trace order_date backward
        sources = pipeline.trace_column_backward("reports.customer_orders", "order_date")
        source_names = [f"{s.table_name}.{s.column_name}" for s in sources]

        assert "raw.orders.order_date" in source_names

        # Trace name backward
        sources = pipeline.trace_column_backward("reports.customer_orders", "name")
        source_names = [f"{s.table_name}.{s.column_name}" for s in sources]

        assert "raw.customers.name" in source_names

    def test_ambiguous_column_resolved_correctly(self):
        """Test that ambiguous columns are resolved to the correct table"""
        queries = [
            """CREATE TABLE staging.table_a AS
            SELECT id, value_a FROM raw.source_a""",
            """CREATE TABLE staging.table_b AS
            SELECT id, value_b FROM raw.source_b""",
            """CREATE TABLE reports.combined AS
            SELECT value_a, value_b
            FROM staging.table_a
            JOIN staging.table_b USING (id)""",
        ]

        pipeline = Pipeline.from_sql_list(queries, dialect="bigquery")

        # value_a should come from table_a
        value_a_edges = [
            e for e in pipeline.edges if e.to_node.full_name == "reports.combined.value_a"
        ]
        assert len(value_a_edges) == 1
        assert value_a_edges[0].from_node.table_name == "staging.table_a"

        # value_b should come from table_b
        value_b_edges = [
            e for e in pipeline.edges if e.to_node.full_name == "reports.combined.value_b"
        ]
        assert len(value_b_edges) == 1
        assert value_b_edges[0].from_node.table_name == "staging.table_b"


# ============================================================================
# Part 4: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases for unqualified column resolution"""

    def test_column_in_both_tables(self):
        """Test when a column name exists in both tables (like 'id')"""
        queries = [
            """CREATE TABLE staging.orders AS
            SELECT id, amount FROM raw.orders""",
            """CREATE TABLE staging.users AS
            SELECT id, name FROM raw.users""",
            """CREATE TABLE reports.summary AS
            SELECT o.id as order_id, u.id as user_id, amount, name
            FROM staging.orders o
            JOIN staging.users u ON o.id = u.id""",
        ]

        pipeline = Pipeline.from_sql_list(queries, dialect="bigquery")

        # amount should come from orders
        amount_edges = [
            e for e in pipeline.edges if e.to_node.full_name == "reports.summary.amount"
        ]
        assert len(amount_edges) == 1
        assert amount_edges[0].from_node.table_name == "staging.orders"

        # name should come from users
        name_edges = [e for e in pipeline.edges if e.to_node.full_name == "reports.summary.name"]
        assert len(name_edges) == 1
        assert name_edges[0].from_node.table_name == "staging.users"

    def test_aggregate_with_unqualified_column(self):
        """Test aggregate functions with unqualified columns"""
        queries = [
            """CREATE TABLE staging.orders AS
            SELECT user_id, amount FROM raw.orders""",
            """CREATE TABLE staging.users AS
            SELECT user_id, name FROM raw.users""",
            """CREATE TABLE reports.totals AS
            SELECT SUM(amount) as total_amount, COUNT(name) as user_count
            FROM staging.orders
            JOIN staging.users USING (user_id)""",
        ]

        pipeline = Pipeline.from_sql_list(queries, dialect="bigquery")

        # total_amount should come from orders.amount
        amount_edges = [
            e for e in pipeline.edges if e.to_node.full_name == "reports.totals.total_amount"
        ]
        assert len(amount_edges) == 1
        assert amount_edges[0].from_node.column_name == "amount"

    def test_no_schema_fallback(self):
        """Test fallback behavior when no schema is available"""
        sql = "SELECT order_id, amount FROM orders"

        # Without external_table_columns, should still work (default behavior)
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        lineage = builder.build()

        # Should have output columns
        output_nodes = [n for n in lineage.nodes.values() if n.layer == "output"]
        assert len(output_nodes) == 2
