"""
Tests for Phase 2: Multi-Query Lineage

This test suite covers:
1. Table Dependency Parsing (DDL/DML/DQL operations)
2. Cross-Query Lineage (connecting multiple queries)
3. Template SQL Support (Jinja2, dbt, Airflow)
4. Edge Cases
5. Real-World Scenarios

Author: Generated following Phase2_MULTI_QUERY_LINEAGE_DESIGN.md
"""

import pytest

from clgraph import (
    MultiQueryParser,
    PipelineLineageBuilder,
    SQLOperation,
    TemplateTokenizer,
)

# ============================================================================
# Part 1: Table Dependency Parsing Tests (DDL/DML/DQL)
# ============================================================================


class TestDDLOperations:
    """Test parsing of DDL operations (CREATE TABLE, CREATE VIEW)"""

    def test_parse_create_table(self):
        """Test parsing CREATE TABLE statement"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT * FROM raw.orders
            """
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        assert len(graph.queries) == 1
        query = graph.queries["query_0"]
        assert query.operation == SQLOperation.CREATE_TABLE
        assert query.destination_table == "staging.orders"
        assert "raw.orders" in query.source_tables
        assert query.is_ddl()
        assert not query.is_dml()

    def test_parse_create_or_replace_table(self):
        """Test parsing CREATE OR REPLACE TABLE statement"""
        queries = [
            """
            CREATE OR REPLACE TABLE analytics.user_metrics AS
            SELECT user_id, COUNT(*) as order_count
            FROM staging.orders
            GROUP BY user_id
            """
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        query = graph.queries["query_0"]
        assert query.operation == SQLOperation.CREATE_OR_REPLACE_TABLE
        assert query.destination_table == "analytics.user_metrics"
        assert "staging.orders" in query.source_tables

    def test_parse_create_view(self):
        """Test parsing CREATE VIEW statement"""
        queries = [
            """
            CREATE VIEW analytics.active_users AS
            SELECT * FROM staging.users WHERE is_active = TRUE
            """
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        query = graph.queries["query_0"]
        assert query.operation == SQLOperation.CREATE_VIEW
        assert query.destination_table == "analytics.active_users"
        assert "staging.users" in query.source_tables

    def test_parse_create_or_replace_view(self):
        """Test parsing CREATE OR REPLACE VIEW statement"""
        queries = [
            """
            CREATE OR REPLACE VIEW reports.monthly_revenue AS
            SELECT DATE_TRUNC(order_date, MONTH) as month, SUM(amount) as revenue
            FROM analytics.orders
            GROUP BY month
            """
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        query = graph.queries["query_0"]
        assert query.operation == SQLOperation.CREATE_OR_REPLACE_VIEW
        assert query.destination_table == "reports.monthly_revenue"
        assert "analytics.orders" in query.source_tables


class TestDMLOperations:
    """Test parsing of DML operations (INSERT, MERGE, UPDATE)"""

    def test_parse_insert(self):
        """Test parsing INSERT statement"""
        queries = [
            """
            INSERT INTO staging.orders
            SELECT * FROM raw.orders
            WHERE order_date = CURRENT_DATE()
            """
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        query = graph.queries["query_0"]
        assert query.operation == SQLOperation.INSERT
        assert query.destination_table == "staging.orders"
        assert "raw.orders" in query.source_tables
        assert query.is_dml()
        assert not query.is_ddl()

    def test_parse_merge(self):
        """Test parsing MERGE statement"""
        queries = [
            """
            MERGE INTO analytics.user_metrics t
            USING staging.orders s
            ON t.user_id = s.user_id
            WHEN MATCHED THEN UPDATE SET t.order_count = t.order_count + 1
            WHEN NOT MATCHED THEN INSERT (user_id, order_count) VALUES (s.user_id, 1)
            """
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        query = graph.queries["query_0"]
        assert query.operation == SQLOperation.MERGE
        assert query.destination_table == "analytics.user_metrics"
        assert "staging.orders" in query.source_tables

    def test_parse_update(self):
        """Test parsing UPDATE statement"""
        queries = [
            """
            UPDATE staging.orders
            SET status = 'processed'
            WHERE order_date < CURRENT_DATE()
            """
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        query = graph.queries["query_0"]
        assert query.operation == SQLOperation.UPDATE
        assert query.destination_table == "staging.orders"


class TestDQLOperations:
    """Test parsing of DQL operations (SELECT only)"""

    def test_parse_select_only(self):
        """Test parsing SELECT statement (no table creation)"""
        queries = [
            """
            SELECT user_id, SUM(amount) as total_revenue
            FROM analytics.orders
            GROUP BY user_id
            """
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        query = graph.queries["query_0"]
        assert query.operation == SQLOperation.SELECT
        assert query.destination_table is None
        assert "analytics.orders" in query.source_tables
        assert query.is_dql()
        assert not query.has_destination()


class TestTableDependencies:
    """Test table dependency extraction and topological sorting"""

    def test_extract_source_tables_from_join(self):
        """Test extracting source tables from JOIN"""
        queries = [
            """
            CREATE TABLE analytics.user_orders AS
            SELECT u.user_id, u.name, o.order_id, o.amount
            FROM staging.users u
            JOIN staging.orders o ON u.user_id = o.user_id
            """
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        query = graph.queries["query_0"]
        assert "staging.users" in query.source_tables
        assert "staging.orders" in query.source_tables
        assert len(query.source_tables) == 2

    def test_extract_source_tables_from_subquery(self):
        """Test extracting source tables from subqueries"""
        queries = [
            """
            CREATE TABLE analytics.top_users AS
            SELECT user_id, total_orders
            FROM (
                SELECT user_id, COUNT(*) as total_orders
                FROM staging.orders
                GROUP BY user_id
            )
            WHERE total_orders > 10
            """
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        query = graph.queries["query_0"]
        assert "staging.orders" in query.source_tables

    def test_topological_sort_two_queries(self):
        """Test topological sorting with two dependent queries"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT * FROM raw.orders
            """,
            """
            CREATE TABLE analytics.user_metrics AS
            SELECT user_id, COUNT(*) as order_count
            FROM staging.orders
            GROUP BY user_id
            """,
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        sorted_ids = graph.topological_sort()
        assert sorted_ids == ["query_0", "query_1"]

    def test_topological_sort_three_queries_linear(self):
        """Test topological sorting with three queries in linear dependency"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT * FROM raw.orders
            """,
            """
            CREATE TABLE analytics.user_metrics AS
            SELECT user_id, SUM(amount) as total_revenue
            FROM staging.orders
            GROUP BY user_id
            """,
            """
            CREATE TABLE reports.monthly_revenue AS
            SELECT DATE_TRUNC(order_date, MONTH) as month, SUM(total_revenue) as revenue
            FROM analytics.user_metrics
            JOIN staging.orders USING (user_id)
            GROUP BY month
            """,
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        sorted_ids = graph.topological_sort()
        # query_0 must come before query_1 and query_2
        # query_1 must come before query_2
        assert sorted_ids.index("query_0") < sorted_ids.index("query_1")
        assert sorted_ids.index("query_1") < sorted_ids.index("query_2")

    def test_get_source_tables(self):
        """Test getting external source tables"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT * FROM raw.orders
            """,
            """
            CREATE TABLE analytics.user_metrics AS
            SELECT user_id, COUNT(*) as order_count
            FROM staging.orders
            GROUP BY user_id
            """,
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        source_tables = graph.get_source_tables()
        assert len(source_tables) == 1
        assert source_tables[0].table_name == "raw.orders"
        assert source_tables[0].is_source
        assert source_tables[0].created_by is None

    def test_get_final_tables(self):
        """Test getting final tables (not read by any query)"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT * FROM raw.orders
            """,
            """
            CREATE TABLE analytics.user_metrics AS
            SELECT user_id, COUNT(*) as order_count
            FROM staging.orders
            GROUP BY user_id
            """,
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        final_tables = graph.get_final_tables()
        assert len(final_tables) == 1
        assert final_tables[0].table_name == "analytics.user_metrics"

    def test_get_dependencies(self):
        """Test getting upstream dependencies for a table"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT * FROM raw.orders
            """,
            """
            CREATE TABLE analytics.user_metrics AS
            SELECT user_id, COUNT(*) as order_count
            FROM staging.orders
            GROUP BY user_id
            """,
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        # analytics.user_metrics depends on staging.orders
        deps = graph.get_dependencies("analytics.user_metrics")
        assert len(deps) == 1
        assert deps[0].table_name == "staging.orders"

        # staging.orders depends on raw.orders
        deps = graph.get_dependencies("staging.orders")
        assert len(deps) == 1
        assert deps[0].table_name == "raw.orders"

        # raw.orders is a source table, no dependencies
        deps = graph.get_dependencies("raw.orders")
        assert len(deps) == 0

        # Non-existent table returns empty list
        deps = graph.get_dependencies("nonexistent.table")
        assert len(deps) == 0

    def test_get_downstream(self):
        """Test getting downstream tables that depend on a table"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT * FROM raw.orders
            """,
            """
            CREATE TABLE analytics.user_metrics AS
            SELECT user_id, COUNT(*) as order_count
            FROM staging.orders
            GROUP BY user_id
            """,
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        # raw.orders feeds into staging.orders
        downstream = graph.get_downstream("raw.orders")
        assert len(downstream) == 1
        assert downstream[0].table_name == "staging.orders"

        # staging.orders feeds into analytics.user_metrics
        downstream = graph.get_downstream("staging.orders")
        assert len(downstream) == 1
        assert downstream[0].table_name == "analytics.user_metrics"

        # analytics.user_metrics is a final table, no downstream
        downstream = graph.get_downstream("analytics.user_metrics")
        assert len(downstream) == 0

        # Non-existent table returns empty list
        downstream = graph.get_downstream("nonexistent.table")
        assert len(downstream) == 0

    def test_get_dependencies_multiple_sources(self):
        """Test getting dependencies when a table has multiple sources"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT * FROM raw.orders
            """,
            """
            CREATE TABLE staging.users AS
            SELECT * FROM raw.users
            """,
            """
            CREATE TABLE analytics.user_orders AS
            SELECT u.user_id, o.order_id, o.amount
            FROM staging.users u
            JOIN staging.orders o ON u.user_id = o.user_id
            """,
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        # analytics.user_orders depends on both staging.users and staging.orders
        deps = graph.get_dependencies("analytics.user_orders")
        assert len(deps) == 2
        dep_names = {d.table_name for d in deps}
        assert "staging.users" in dep_names
        assert "staging.orders" in dep_names

    def test_get_downstream_multiple_dependents(self):
        """Test getting downstream when a table feeds multiple queries"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT * FROM raw.orders
            """,
            """
            CREATE TABLE analytics.user_metrics AS
            SELECT user_id, COUNT(*) as order_count
            FROM staging.orders
            GROUP BY user_id
            """,
            """
            CREATE TABLE analytics.product_metrics AS
            SELECT product_id, COUNT(*) as order_count
            FROM staging.orders
            GROUP BY product_id
            """,
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        # staging.orders feeds into both analytics tables
        downstream = graph.get_downstream("staging.orders")
        assert len(downstream) == 2
        downstream_names = {d.table_name for d in downstream}
        assert "analytics.user_metrics" in downstream_names
        assert "analytics.product_metrics" in downstream_names

    def test_get_dependencies_with_dml(self):
        """Test getting dependencies for tables populated via DML (INSERT)"""
        queries = [
            """
            CREATE TABLE staging.daily_orders AS
            SELECT * FROM raw.orders WHERE order_date = CURRENT_DATE()
            """,
            """
            INSERT INTO analytics.orders_history
            SELECT * FROM staging.daily_orders
            """,
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        # analytics.orders_history is populated via INSERT, not CREATE
        # It should still show staging.daily_orders as a dependency
        deps = graph.get_dependencies("analytics.orders_history")
        assert len(deps) == 1
        assert deps[0].table_name == "staging.daily_orders"

    def test_get_dependencies_with_multiple_dml(self):
        """Test getting dependencies for tables with multiple DML operations"""
        queries = [
            """
            CREATE TABLE staging.orders_a AS
            SELECT * FROM raw.orders_a
            """,
            """
            CREATE TABLE staging.orders_b AS
            SELECT * FROM raw.orders_b
            """,
            """
            INSERT INTO analytics.all_orders
            SELECT * FROM staging.orders_a
            """,
            """
            INSERT INTO analytics.all_orders
            SELECT * FROM staging.orders_b
            """,
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        # analytics.all_orders has two INSERT operations from different sources
        deps = graph.get_dependencies("analytics.all_orders")
        assert len(deps) == 2
        dep_names = {d.table_name for d in deps}
        assert "staging.orders_a" in dep_names
        assert "staging.orders_b" in dep_names

    def test_get_downstream_with_dml(self):
        """Test getting downstream tables when destination uses DML (INSERT)"""
        queries = [
            """
            CREATE TABLE staging.daily_orders AS
            SELECT * FROM raw.orders WHERE order_date = CURRENT_DATE()
            """,
            """
            INSERT INTO analytics.orders_history
            SELECT * FROM staging.daily_orders
            """,
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        # raw.orders should show staging.daily_orders as downstream
        downstream = graph.get_downstream("raw.orders")
        assert len(downstream) == 1
        assert downstream[0].table_name == "staging.daily_orders"

        # staging.daily_orders should show analytics.orders_history as downstream
        # even though it's populated via INSERT
        downstream = graph.get_downstream("staging.daily_orders")
        assert len(downstream) == 1
        assert downstream[0].table_name == "analytics.orders_history"


# ============================================================================
# Part 2: Template SQL Support Tests
# ============================================================================


class TestTemplateTokenizer:
    """Test template tokenization and resolution"""

    def test_tokenize_jinja2_variable(self):
        """Test tokenizing Jinja2 variables"""
        tokenizer = TemplateTokenizer()
        sql = "CREATE TABLE {{ project }}.analytics.users AS SELECT * FROM raw.users"
        tokenized = tokenizer.tokenize_sql(sql)

        # Should replace template with token
        assert "{{ project }}" not in tokenized
        assert "__TMPL_" in tokenized

        # Should restore template
        restored = tokenizer.restore_templates(tokenized)
        assert "{{ project }}" in restored

    def test_tokenize_jinja2_function(self):
        """Test tokenizing Jinja2 function calls"""
        tokenizer = TemplateTokenizer()
        sql = "CREATE TABLE analytics.users AS SELECT * FROM {{ ref('staging_users') }}"
        tokenized = tokenizer.tokenize_sql(sql)

        # Should replace template with token
        assert "{{ ref('staging_users') }}" not in tokenized
        assert "__TMPL_" in tokenized

        # Should restore template
        restored = tokenizer.restore_templates(tokenized)
        assert "{{ ref('staging_users') }}" in restored

    def test_tokenize_nested_access(self):
        """Test tokenizing nested access (e.g., config.project)"""
        tokenizer = TemplateTokenizer()
        sql = "CREATE TABLE {{ config.project }}.analytics.users AS SELECT * FROM raw.users"
        tokenized = tokenizer.tokenize_sql(sql)

        # Should restore template
        restored = tokenizer.restore_templates(tokenized)
        assert "{{ config.project }}" in restored

    def test_resolve_with_context(self):
        """Test resolving templates with context"""
        tokenizer = TemplateTokenizer()
        sql = "CREATE TABLE {{ project }}.analytics.users AS SELECT * FROM raw.users"
        tokenized = tokenizer.tokenize_sql(sql)

        # Resolve with context
        context = {"project": "production"}
        tokenizer.resolve(context)

        # Should resolve to actual value
        restored = tokenizer.restore_templates(tokenized)
        assert "production" in restored
        assert "{{ project }}" not in restored


class TestTemplateSQL:
    """Test parsing SQL with templates"""

    def test_jinja2_template_resolution(self):
        """Test Jinja2 template resolution"""
        queries = [
            """
            CREATE TABLE {{ project }}.{{ dataset }}.user_metrics AS
            SELECT * FROM {{ source_table }}
            """
        ]

        context = {"project": "production", "dataset": "analytics", "source_table": "raw.orders"}

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries, template_context=context)

        query = graph.queries["query_0"]
        assert query.destination_table == "production.analytics.user_metrics"
        assert "raw.orders" in query.source_tables
        assert query.is_templated

    def test_dbt_style_ref_macro(self):
        """Test dbt-style ref() macro"""
        queries = [
            """
            CREATE TABLE {{ target.schema }}.user_metrics AS
            SELECT * FROM {{ ref('staging_orders') }}
            """
        ]

        context = {"target": {"schema": "analytics"}, "ref": lambda t: f"staging.{t}"}

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries, template_context=context)

        query = graph.queries["query_0"]
        assert "analytics" in query.destination_table
        assert "staging.staging_orders" in query.source_tables

    def test_airflow_ds_macro(self):
        """Test Airflow date macros"""
        queries = [
            """
            CREATE TABLE analytics.user_metrics_{{ ds_nodash }} AS
            SELECT * FROM raw.orders WHERE date = '{{ ds }}'
            """
        ]

        context = {"ds": "2025-11-02", "ds_nodash": "20251102"}

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries, template_context=context)

        query = graph.queries["query_0"]
        assert "20251102" in query.destination_table

    def test_multiple_environments(self):
        """Test same SQL template resolved for different environments"""
        sql_template = """
        CREATE TABLE {{ env }}.analytics.user_metrics AS
        SELECT * FROM {{ env }}.raw.orders
        """

        # Production
        parser = MultiQueryParser()
        prod_graph = parser.parse_queries([sql_template], template_context={"env": "production"})
        assert "production.analytics.user_metrics" in prod_graph.tables

        # Development
        dev_graph = parser.parse_queries([sql_template], template_context={"env": "development"})
        assert "development.analytics.user_metrics" in dev_graph.tables


# ============================================================================
# Part 3: Cross-Query Lineage Tests
# ============================================================================


class TestCrossQueryLineage:
    """Test column lineage across multiple queries"""

    def test_connect_two_queries(self):
        """Test connecting lineage across two queries"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT
                order_id,
                amount,
                user_id
            FROM raw.orders
            """,
            """
            CREATE TABLE analytics.user_metrics AS
            SELECT
                user_id,
                SUM(amount) as total_revenue
            FROM staging.orders
            GROUP BY user_id
            """,
        ]

        parser = MultiQueryParser()
        table_graph = parser.parse_queries(queries)

        builder = PipelineLineageBuilder()
        pipeline = builder.build(table_graph)

        # Verify columns are in the graph (use get_column for table.column lookup)
        assert pipeline.get_column("staging.orders", "amount") is not None
        assert pipeline.get_column("analytics.user_metrics", "total_revenue") is not None

        # Verify backward lineage
        sources = pipeline.trace_column_backward("analytics.user_metrics", "total_revenue")
        # Should trace back to raw.orders.amount
        assert any(s.table_name == "raw.orders" and s.column_name == "amount" for s in sources)

    def test_connect_three_queries_linear(self):
        """Test lineage through three queries in linear pipeline"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT
                order_id,
                amount as order_amount,
                user_id
            FROM raw.orders
            """,
            """
            CREATE TABLE analytics.user_totals AS
            SELECT
                user_id,
                SUM(order_amount) as total_spent
            FROM staging.orders
            GROUP BY user_id
            """,
            """
            CREATE TABLE reports.high_value_users AS
            SELECT
                user_id,
                total_spent
            FROM analytics.user_totals
            WHERE total_spent > 1000
            """,
        ]

        parser = MultiQueryParser()
        table_graph = parser.parse_queries(queries)

        builder = PipelineLineageBuilder()
        pipeline = builder.build(table_graph)

        # Verify backward lineage traces through all three queries
        sources = pipeline.trace_column_backward("reports.high_value_users", "total_spent")
        # Should trace back to raw.orders.amount
        assert any(s.table_name == "raw.orders" and s.column_name == "amount" for s in sources)

    def test_forward_lineage_across_queries(self):
        """Test forward lineage (impact analysis) across queries"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT
                order_id,
                amount,
                user_id
            FROM raw.orders
            """,
            """
            CREATE TABLE analytics.user_metrics AS
            SELECT
                user_id,
                SUM(amount) as total_revenue
            FROM staging.orders
            GROUP BY user_id
            """,
        ]

        parser = MultiQueryParser()
        table_graph = parser.parse_queries(queries)

        builder = PipelineLineageBuilder()
        pipeline = builder.build(table_graph)

        # Verify forward lineage from raw.orders.amount
        descendants = pipeline.trace_column_forward("raw.orders", "amount")
        # Should reach analytics.user_metrics.total_revenue
        assert any(
            d.table_name == "analytics.user_metrics" and d.column_name == "total_revenue"
            for d in descendants
        )

    def test_branching_pipeline(self):
        """Test pipeline where one table feeds multiple queries"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT order_id, amount, user_id, product_id
            FROM raw.orders
            """,
            """
            CREATE TABLE analytics.user_metrics AS
            SELECT user_id, SUM(amount) as total_revenue
            FROM staging.orders
            GROUP BY user_id
            """,
            """
            CREATE TABLE analytics.product_metrics AS
            SELECT product_id, SUM(amount) as product_revenue
            FROM staging.orders
            GROUP BY product_id
            """,
        ]

        parser = MultiQueryParser()
        table_graph = parser.parse_queries(queries)

        builder = PipelineLineageBuilder()
        pipeline = builder.build(table_graph)

        # Verify staging.orders is read by two queries
        staging_table = table_graph.tables["staging.orders"]
        assert len(staging_table.read_by) == 2

        # Verify forward lineage reaches both branches
        descendants = pipeline.trace_column_forward("raw.orders", "amount")
        assert any(d.table_name == "analytics.user_metrics" for d in descendants)
        assert any(d.table_name == "analytics.product_metrics" for d in descendants)

    def test_merging_pipeline(self):
        """Test pipeline where multiple tables feed one query"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT order_id, amount, user_id
            FROM raw.orders
            """,
            """
            CREATE TABLE staging.users AS
            SELECT user_id, name, email
            FROM raw.users
            """,
            """
            CREATE TABLE analytics.user_orders AS
            SELECT
                u.name,
                u.email,
                o.order_id,
                o.amount
            FROM staging.users u
            JOIN staging.orders o ON u.user_id = o.user_id
            """,
        ]

        parser = MultiQueryParser()
        table_graph = parser.parse_queries(queries)

        builder = PipelineLineageBuilder()
        pipeline = builder.build(table_graph)

        # Verify two source tables
        source_tables = table_graph.get_source_tables()
        assert len(source_tables) == 2

        # Verify backward lineage from analytics.user_orders.amount
        sources = pipeline.trace_column_backward("analytics.user_orders", "amount")
        assert any(s.table_name == "raw.orders" and s.column_name == "amount" for s in sources)

    def test_get_lineage_path(self):
        """Test finding specific lineage path between two columns"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT
                order_id,
                amount,
                user_id
            FROM raw.orders
            """,
            """
            CREATE TABLE analytics.user_metrics AS
            SELECT
                user_id,
                SUM(amount) as total_revenue
            FROM staging.orders
            GROUP BY user_id
            """,
        ]

        parser = MultiQueryParser()
        table_graph = parser.parse_queries(queries)

        builder = PipelineLineageBuilder()
        pipeline = builder.build(table_graph)

        # Get path from raw.orders.amount to analytics.user_metrics.total_revenue
        path = pipeline.get_lineage_path(
            "raw.orders", "amount", "analytics.user_metrics", "total_revenue"
        )

        # Should have edges connecting them
        assert len(path) > 0

    def test_star_expansion_in_cross_query_edges(self):
        """
        Test that * is properly resolved in cross-query lineage when schema is known.

        When upstream query creates a table with known columns, and downstream
        query uses both COUNT(*) and specific columns (for SUM(amount)),
        COUNT(*) should resolve to individual columns (no * node):
        1. All upstream columns -> aggregate output columns (for COUNT(*))
        2. Specific column matches (for individual column references)
        """
        queries = [
            """
            CREATE TABLE staging.user_orders AS
            SELECT
                user_id,
                order_id,
                amount,
                order_date
            FROM raw.orders
            WHERE status = 'completed'
            """,
            """
            CREATE TABLE analytics.user_metrics AS
            SELECT
                user_id,
                COUNT(*) as order_count,
                SUM(amount) as total_revenue,
                AVG(amount) as avg_order_value
            FROM staging.user_orders
            GROUP BY user_id
            """,
        ]

        parser = MultiQueryParser()
        table_graph = parser.parse_queries(queries)

        builder = PipelineLineageBuilder()
        pipeline = builder.build(table_graph)

        # When schema is known from upstream, there should be NO * node
        # (COUNT(*) resolves to individual columns)
        star_nodes = [
            col
            for col in pipeline.columns.values()
            if col.is_star and col.table_name == "staging.user_orders"
        ]
        assert len(star_nodes) == 0, (
            f"Expected no * node when schema is known, got {len(star_nodes)}"
        )

        # COUNT(*) should create edges from all source columns to order_count
        order_count_edges = [
            e
            for e in pipeline.edges
            if e.to_node.column_name == "order_count"
            and e.to_node.table_name == "analytics.user_metrics"
            and e.from_node.table_name == "staging.user_orders"
        ]
        source_columns = {e.from_node.column_name for e in order_count_edges}
        assert source_columns == {"user_id", "order_id", "amount", "order_date"}, (
            f"Expected edges from all 4 columns for COUNT(*), got {source_columns}"
        )

        # Verify edges exist from intermediate table columns to final output
        # for specifically referenced columns (user_id, amount)
        edges_to_output = [
            e
            for e in pipeline.edges
            if e.from_node.table_name == "staging.user_orders"
            and e.from_node.column_name in ("user_id", "amount")
            and e.to_node.table_name == "analytics.user_metrics"
        ]
        # user_id and amount should each have edges to output (plus COUNT(*) edges)
        assert len(edges_to_output) >= 2, f"Expected >= 2 edges, got {len(edges_to_output)}"

        # Verify backward lineage traces all the way to source
        sources = pipeline.trace_column_backward("analytics.user_metrics", "total_revenue")
        assert any(s.table_name == "raw.orders" and s.column_name == "amount" for s in sources)

    def test_star_except_in_cross_query_edges(self):
        """
        Test that SELECT * EXCEPT properly expands and excludes columns.

        When downstream query uses SELECT * EXCEPT (col1, col2):
        - The * should be expanded to individual columns (not kept as *)
        - The excepted columns should NOT appear in the output
        """
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT
                order_id,
                user_id,
                amount,
                sensitive_data,
                order_date
            FROM raw.orders
            """,
            """
            CREATE TABLE analytics.clean_orders AS
            SELECT * EXCEPT (sensitive_data)
            FROM staging.orders
            """,
        ]

        parser = MultiQueryParser()
        table_graph = parser.parse_queries(queries)

        builder = PipelineLineageBuilder()
        pipeline = builder.build(table_graph)

        # Find the query ID for the second query
        second_query_id = list(table_graph.queries.keys())[1]

        # Verify that * was expanded to individual columns in the output
        output_columns = [
            col
            for col in pipeline.columns.values()
            if col.query_id == second_query_id
            and col.layer == "output"
            and col.table_name == "analytics.clean_orders"
        ]

        output_col_names = {col.column_name for col in output_columns}

        # Should have 4 columns (5 total minus 1 excepted)
        assert len(output_columns) == 4

        # Should have these columns
        assert "order_id" in output_col_names
        assert "user_id" in output_col_names
        assert "amount" in output_col_names
        assert "order_date" in output_col_names

        # Should NOT have the excepted column
        assert "sensitive_data" not in output_col_names

        # Should NOT have a * column
        assert "*" not in output_col_names

    def test_star_replace_in_cross_query_edges(self):
        """
        Test that SELECT * REPLACE expands columns with transformations.

        When downstream query uses SELECT * REPLACE:
        - The * should be expanded to individual columns (not kept as *)
        - All columns should be present (REPLACE transforms, doesn't remove)
        - Replaced columns should show the transformation expression
        """
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT
                order_id,
                user_id,
                amount,
                status,
                order_date
            FROM raw.orders
            """,
            """
            CREATE TABLE analytics.orders_normalized AS
            SELECT * REPLACE (UPPER(status) as status)
            FROM staging.orders
            """,
        ]

        parser = MultiQueryParser()
        table_graph = parser.parse_queries(queries)

        builder = PipelineLineageBuilder()
        pipeline = builder.build(table_graph)

        # Find the query ID for the second query
        second_query_id = list(table_graph.queries.keys())[1]

        # Verify that * was expanded to individual columns in the output
        output_columns = [
            col
            for col in pipeline.columns.values()
            if col.query_id == second_query_id
            and col.layer == "output"
            and col.table_name == "analytics.orders_normalized"
        ]

        output_col_names = {col.column_name for col in output_columns}

        # Should have all 5 columns (REPLACE doesn't remove columns)
        assert len(output_columns) == 5

        # Should have ALL columns including the replaced one
        assert "order_id" in output_col_names
        assert "user_id" in output_col_names
        assert "amount" in output_col_names
        assert "status" in output_col_names  # Replaced column
        assert "order_date" in output_col_names

        # Should NOT have a * column
        assert "*" not in output_col_names

        # Verify the replaced column has the transformation expression
        status_col = next(col for col in output_columns if col.column_name == "status")
        assert "UPPER" in status_col.expression


# ============================================================================
# Part 4: Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_query_with_no_dependencies(self):
        """Test query that only reads from external source"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT * FROM raw.orders
            """
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        query = graph.queries["query_0"]
        assert len(query.source_tables) == 1
        assert "raw.orders" in query.source_tables

        # raw.orders should be marked as source table
        source_tables = graph.get_source_tables()
        assert len(source_tables) == 1
        assert source_tables[0].table_name == "raw.orders"

    def test_query_with_no_dependents(self):
        """Test query that produces final table"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT * FROM raw.orders
            """,
            """
            CREATE TABLE analytics.final_report AS
            SELECT * FROM staging.orders
            """,
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        final_tables = graph.get_final_tables()
        assert len(final_tables) == 1
        assert final_tables[0].table_name == "analytics.final_report"

    def test_same_table_read_by_multiple_queries(self):
        """Test table that is read by multiple downstream queries"""
        queries = [
            """
            CREATE TABLE staging.orders AS
            SELECT * FROM raw.orders
            """,
            """
            CREATE TABLE analytics.user_metrics AS
            SELECT user_id, COUNT(*) as order_count
            FROM staging.orders
            GROUP BY user_id
            """,
            """
            CREATE TABLE analytics.product_metrics AS
            SELECT product_id, COUNT(*) as order_count
            FROM staging.orders
            GROUP BY product_id
            """,
        ]

        parser = MultiQueryParser()
        graph = parser.parse_queries(queries)

        staging_orders = graph.tables["staging.orders"]
        assert len(staging_orders.read_by) == 2
        assert "query_1" in staging_orders.read_by
        assert "query_2" in staging_orders.read_by


# ============================================================================
# Part 5: Real-World Scenario Tests
# ============================================================================


class TestRealWorldScenarios:
    """Test real-world pipeline scenarios"""

    def test_staging_analytics_reporting_pipeline(self):
        """Test typical 3-layer pipeline: staging → analytics → reporting"""
        queries = [
            # Layer 1: Staging
            """
            CREATE TABLE staging.user_orders AS
            SELECT
                user_id,
                order_id,
                amount,
                order_date
            FROM raw.orders
            WHERE status = 'completed'
            """,
            # Layer 2: Analytics
            """
            CREATE TABLE analytics.user_metrics AS
            SELECT
                user_id,
                COUNT(*) as order_count,
                SUM(amount) as total_revenue,
                AVG(amount) as avg_order_value
            FROM staging.user_orders
            GROUP BY user_id
            """,
            # Layer 3: Reporting
            """
            CREATE TABLE reports.monthly_revenue AS
            SELECT
                DATE_TRUNC(order_date, MONTH) as month,
                SUM(total_revenue) as revenue
            FROM analytics.user_metrics
            JOIN staging.user_orders USING (user_id)
            GROUP BY month
            """,
        ]

        parser = MultiQueryParser()
        table_graph = parser.parse_queries(queries)

        # Verify table dependencies
        assert "staging.user_orders" in table_graph.tables
        assert "analytics.user_metrics" in table_graph.tables
        assert "reports.monthly_revenue" in table_graph.tables

        # Verify topological order
        sorted_ids = table_graph.topological_sort()
        assert sorted_ids.index("query_0") < sorted_ids.index("query_1")
        assert sorted_ids.index("query_1") < sorted_ids.index("query_2")

        # Build pipeline lineage
        builder = PipelineLineageBuilder()
        pipeline = builder.build(table_graph)

        # Verify end-to-end lineage
        sources = pipeline.trace_column_backward("reports.monthly_revenue", "revenue")
        assert any(s.table_name == "raw.orders" and s.column_name == "amount" for s in sources)

    def test_incremental_pipeline(self):
        """Test incremental pipeline (INSERT instead of CREATE OR REPLACE)"""
        queries = [
            """
            CREATE TABLE staging.daily_orders AS
            SELECT * FROM raw.orders WHERE order_date = CURRENT_DATE()
            """,
            """
            INSERT INTO analytics.orders_history
            SELECT * FROM staging.daily_orders
            """,
        ]

        parser = MultiQueryParser()
        table_graph = parser.parse_queries(queries)

        # Verify INSERT operation
        insert_query = table_graph.queries["query_1"]
        assert insert_query.operation == SQLOperation.INSERT
        assert insert_query.is_dml()

        # Verify table dependencies
        orders_history = table_graph.tables["analytics.orders_history"]
        assert "query_1" in orders_history.modified_by

    def test_star_schema_pipeline(self):
        """Test star schema (fact + dimensions)"""
        queries = [
            # Dimension tables
            """
            CREATE TABLE dim.users AS
            SELECT user_id, name, email, signup_date
            FROM raw.users
            """,
            """
            CREATE TABLE dim.products AS
            SELECT product_id, name, category, price
            FROM raw.products
            """,
            # Fact table
            """
            CREATE TABLE fact.orders AS
            SELECT
                o.order_id,
                o.user_id,
                o.product_id,
                o.amount,
                o.order_date
            FROM raw.orders o
            """,
            # Aggregated report
            """
            CREATE TABLE reports.sales_by_category AS
            SELECT
                p.category,
                SUM(f.amount) as total_sales
            FROM fact.orders f
            JOIN dim.products p ON f.product_id = p.product_id
            GROUP BY p.category
            """,
        ]

        parser = MultiQueryParser()
        table_graph = parser.parse_queries(queries)

        # Verify all tables created
        assert "dim.users" in table_graph.tables
        assert "dim.products" in table_graph.tables
        assert "fact.orders" in table_graph.tables
        assert "reports.sales_by_category" in table_graph.tables

        # Verify multiple source tables
        source_tables = table_graph.get_source_tables()
        assert len(source_tables) == 3  # raw.users, raw.products, raw.orders


# ============================================================================
# Part 7: Simplified Multi-Query Graph Tests
# ============================================================================


class TestSimplifiedMultiQueryGraph:
    """Test the simplified multi-query column lineage graph"""

    def test_simplified_keeps_all_tables(self):
        """Test that simplified graph keeps ALL physical tables (not just source/final)"""
        from clgraph import Pipeline

        queries = [
            (
                "extract",
                """
                CREATE TABLE staging.orders AS
                SELECT order_id, customer_id, amount FROM raw.orders
                """,
            ),
            (
                "transform",
                """
                CREATE TABLE staging.order_summary AS
                SELECT customer_id, SUM(amount) as total FROM staging.orders GROUP BY customer_id
                """,
            ),
            (
                "load",
                """
                CREATE TABLE analytics.customer_metrics AS
                SELECT customer_id, total FROM staging.order_summary
                """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")
        simplified = pipeline.get_simplified_column_graph()

        # Should have ALL physical tables (source, intermediate, and final)
        simplified_tables = {col.table_name for col in simplified.columns.values()}
        assert "raw.orders" in simplified_tables
        assert "staging.orders" in simplified_tables
        assert "staging.order_summary" in simplified_tables
        assert "analytics.customer_metrics" in simplified_tables

        # Edges should flow through all tables
        edges = simplified.edges
        edge_pairs = {(e.from_node.full_name, e.to_node.full_name) for e in edges}

        # raw.orders -> staging.orders
        assert ("raw.orders.customer_id", "staging.orders.customer_id") in edge_pairs
        # staging.orders -> staging.order_summary
        assert ("staging.orders.customer_id", "staging.order_summary.customer_id") in edge_pairs
        # staging.order_summary -> analytics.customer_metrics
        assert (
            "staging.order_summary.customer_id",
            "analytics.customer_metrics.customer_id",
        ) in edge_pairs

    def test_simplified_removes_ctes(self):
        """Test that simplified graph removes CTE columns but keeps all tables"""
        from clgraph import Pipeline

        queries = [
            (
                "with_cte",
                """
                CREATE TABLE staging.orders AS
                WITH filtered AS (
                    SELECT order_id, customer_id, amount
                    FROM raw.orders
                    WHERE status = 'completed'
                )
                SELECT order_id, customer_id, amount FROM filtered
                """,
            ),
            (
                "downstream",
                """
                CREATE TABLE analytics.summary AS
                SELECT customer_id, SUM(amount) as total
                FROM staging.orders
                GROUP BY customer_id
                """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # Original should have CTE columns
        original = pipeline.column_graph
        has_cte = any(":cte:" in name for name in original.columns.keys())
        assert has_cte, "Original graph should have CTE columns"

        # Simplified should NOT have CTE columns
        simplified = pipeline.get_simplified_column_graph()
        has_cte_simplified = any(":cte:" in name for name in simplified.columns.keys())
        assert not has_cte_simplified, "Simplified graph should not have CTE columns"

        # But should have all physical tables
        simplified_tables = {col.table_name for col in simplified.columns.values()}
        assert "raw.orders" in simplified_tables
        assert "staging.orders" in simplified_tables
        assert "analytics.summary" in simplified_tables

        # Edges should trace through CTEs to connect tables directly
        edge_pairs = {(e.from_node.full_name, e.to_node.full_name) for e in simplified.edges}
        # raw.orders -> staging.orders (directly, no CTE in between)
        assert ("raw.orders.customer_id", "staging.orders.customer_id") in edge_pairs

    def test_simplified_preserves_issues(self):
        """Test that simplified graph preserves validation issues"""
        from clgraph import Pipeline

        # Query with potential issue (star from unknown external table)
        queries = [
            (
                "copy_all",
                "CREATE TABLE staging.all_data AS SELECT * FROM external.unknown_table",
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")
        simplified = pipeline.get_simplified_column_graph()

        # Issues should be preserved
        assert len(simplified.issues) == len(pipeline.column_graph.issues)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
