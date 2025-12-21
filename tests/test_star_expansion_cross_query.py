"""
Tests for star expansion across multi-query pipelines.

These tests verify that SELECT * is properly resolved to individual columns
when upstream table schemas are known, while COUNT(*) and other star usages
are preserved correctly.
"""

from clgraph import Pipeline


class TestSelectStarExpansion:
    """Test that SELECT * is expanded to individual columns in multi-query pipelines."""

    def test_simple_select_star_expansion(self):
        """
        Test basic SELECT * expansion.

        Query 1 creates table with known columns.
        Query 2 does SELECT * FROM that table.
        Expected: Query 2 should have individual columns, not *.
        """
        queries = [
            (
                "create_staging",
                """
            CREATE TABLE staging.orders AS
            SELECT
                order_id,
                user_id,
                amount,
                status
            FROM raw.orders
            """,
            ),
            (
                "select_all",
                """
            CREATE TABLE analytics.orders_copy AS
            SELECT *
            FROM staging.orders
            """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # Verify Query 2 output has individual columns, not *
        query2_output = [
            col
            for col in pipeline.columns.values()
            if col.query_id == "select_all" and col.layer == "output"
        ]

        output_col_names = {col.column_name for col in query2_output}
        assert output_col_names == {"order_id", "user_id", "amount", "status"}
        assert all(not col.is_star for col in query2_output)

        # With unified column naming, edges from intermediate table columns to
        # output columns flow naturally through shared nodes.
        #
        # Verify edges connect intermediate columns to final output
        edges_to_output = [
            e
            for e in pipeline.edges
            if e.from_node.table_name == "staging.orders"
            and e.to_node.table_name == "analytics.orders_copy"
            and not e.to_node.is_star
        ]
        assert len(edges_to_output) == 4, f"Expected 4 edges, got {len(edges_to_output)}"

        # Should NOT have edges to * node in output layer
        star_edges = [
            e for e in pipeline.edges if e.to_node.is_star and e.to_node.layer == "output"
        ]
        assert len(star_edges) == 0

    def test_select_star_replace_expansion(self):
        """
        Test SELECT * REPLACE expansion.

        Query 1 creates table with known columns.
        Query 2 does SELECT * REPLACE (...) FROM that table.
        Expected: Query 2 should have individual columns with transformations.
        """
        queries = [
            (
                "create_staging",
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
            ),
            (
                "normalize",
                """
            CREATE TABLE analytics.orders_normalized AS
            SELECT * REPLACE (
                UPPER(status) as status,
                ROUND(amount, 2) as amount
            )
            FROM staging.orders
            """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # Verify Query 2 output has 5 individual columns
        query2_output = [
            col
            for col in pipeline.columns.values()
            if col.query_id == "normalize" and col.layer == "output"
        ]

        output_col_names = {col.column_name for col in query2_output}
        assert output_col_names == {"order_id", "user_id", "amount", "status", "order_date"}
        assert all(not col.is_star for col in query2_output)

        # With unified column naming, edges from intermediate table columns to
        # output columns flow naturally through shared nodes.
        #
        # Verify edges connect intermediate columns to final output
        output_edges = [
            e
            for e in pipeline.edges
            if e.from_node.table_name == "staging.orders"
            and e.to_node.table_name == "analytics.orders_normalized"
            and not e.to_node.is_star
        ]
        assert len(output_edges) == 5, f"Expected 5 edges, got {len(output_edges)}"

        # Verify specific column lineage
        assert any(
            e.from_node.column_name == "amount"
            and e.to_node.column_name == "amount"
            and e.to_node.table_name == "analytics.orders_normalized"
            for e in output_edges
        )

    def test_select_star_except_expansion(self):
        """
        Test SELECT * EXCEPT expansion.

        Query 1 creates table with known columns.
        Query 2 does SELECT * EXCEPT (...) FROM that table.
        Expected: Query 2 should have individual columns, excluding excepted ones.
        """
        queries = [
            (
                "create_staging",
                """
            CREATE TABLE staging.orders AS
            SELECT
                order_id,
                user_id,
                amount,
                customer_email,
                customer_ssn,
                order_date
            FROM raw.orders
            """,
            ),
            (
                "remove_pii",
                """
            CREATE TABLE analytics.clean_orders AS
            SELECT * EXCEPT (customer_email, customer_ssn)
            FROM staging.orders
            """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # Verify Query 2 output has 4 columns (6 minus 2 excepted)
        query2_output = [
            col
            for col in pipeline.columns.values()
            if col.query_id == "remove_pii" and col.layer == "output"
        ]

        output_col_names = {col.column_name for col in query2_output}
        assert output_col_names == {"order_id", "user_id", "amount", "order_date"}
        assert "customer_email" not in output_col_names
        assert "customer_ssn" not in output_col_names
        assert all(not col.is_star for col in query2_output)

    def test_three_query_star_chain(self):
        """
        Test star expansion through 3-query chain.

        Query 1 creates table with columns.
        Query 2 does SELECT * FROM query 1's table.
        Query 3 does SELECT * FROM query 2's table.
        Expected: All queries should have individual columns.
        """
        queries = [
            (
                "raw_to_staging",
                """
            CREATE TABLE staging.orders AS
            SELECT order_id, amount, status
            FROM raw.orders
            """,
            ),
            (
                "staging_to_intermediate",
                """
            CREATE TABLE intermediate.orders AS
            SELECT * FROM staging.orders
            """,
            ),
            (
                "intermediate_to_analytics",
                """
            CREATE TABLE analytics.orders AS
            SELECT * FROM intermediate.orders
            """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # All three queries should have the same 3 columns
        for query_id in ["raw_to_staging", "staging_to_intermediate", "intermediate_to_analytics"]:
            output = [
                col
                for col in pipeline.columns.values()
                if col.query_id == query_id and col.layer == "output"
            ]
            output_col_names = {col.column_name for col in output}
            assert output_col_names == {"order_id", "amount", "status"}
            assert all(not col.is_star for col in output)


class TestCountStarPreservation:
    """Test that COUNT(*) and other star usages are resolved when schema is known."""

    def test_count_star_with_explicit_columns(self):
        """
        Test that COUNT(*) creates edges to individual columns when schema is known.

        Query 1 creates table with known columns.
        Query 2 uses COUNT(*) and explicit columns.
        Expected: COUNT(*) resolves to individual columns from upstream, no * node.
        """
        queries = [
            (
                "create_staging",
                """
            CREATE TABLE staging.user_orders AS
            SELECT user_id, order_id, amount, order_date
            FROM raw.orders
            WHERE status = 'completed'
            """,
            ),
            (
                "aggregate",
                """
            CREATE TABLE analytics.user_metrics AS
            SELECT
                user_id,
                COUNT(*) as order_count,
                SUM(amount) as total_revenue
            FROM staging.user_orders
            GROUP BY user_id
            """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # When schema is known from upstream, there should be NO * node
        query2_star = [
            col
            for col in pipeline.columns.values()
            if col.query_id == "aggregate" and col.is_star and col.layer == "input"
        ]
        assert len(query2_star) == 0, (
            f"Expected no * node when schema is known, but found {len(query2_star)}"
        )

        # COUNT(*) should create edges to individual columns, not * node
        # Check edges from staging.user_orders columns to order_count output
        order_count_edges = [
            e
            for e in pipeline.edges
            if e.to_node.column_name == "order_count"
            and e.to_node.table_name == "analytics.user_metrics"
        ]
        # Should have edges from all 4 source columns (user_id, order_id, amount, order_date)
        source_columns = {e.from_node.column_name for e in order_count_edges}
        assert source_columns == {"user_id", "order_id", "amount", "order_date"}, (
            f"Expected edges from all 4 columns, got {source_columns}"
        )

        # Should ALSO have edges for explicitly referenced columns (user_id, amount)
        # These flow naturally through shared column nodes
        specific_edges = [
            e
            for e in pipeline.edges
            if e.from_node.table_name == "staging.user_orders"
            and e.to_node.table_name == "analytics.user_metrics"
            and not e.from_node.is_star
        ]
        # user_id and amount should have edges to output (plus COUNT(*) edges)
        assert len(specific_edges) >= 2, (
            f"Expected at least 2 specific edges, got {len(specific_edges)}"
        )

    def test_count_star_only(self):
        """
        Test COUNT(*) without other column references.

        Query 2 only uses COUNT(*), no other columns.
        Expected: Edges to individual columns from upstream schema.
        """
        queries = [
            (
                "create_staging",
                """
            CREATE TABLE staging.events AS
            SELECT event_id, event_type, timestamp
            FROM raw.events
            """,
            ),
            (
                "count_all",
                """
            CREATE TABLE analytics.event_count AS
            SELECT COUNT(*) as total_events
            FROM staging.events
            """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # With known schema, COUNT(*) should resolve to individual columns
        # There should be NO * node in the input layer
        star_nodes = [
            col for col in pipeline.columns.values() if col.query_id == "count_all" and col.is_star
        ]
        assert len(star_nodes) == 0, (
            f"Expected no * node when schema is known, got {len(star_nodes)}"
        )

        # COUNT(*) should have edges from all 3 source columns
        total_events_edges = [
            e
            for e in pipeline.edges
            if e.to_node.column_name == "total_events"
            and e.to_node.table_name == "analytics.event_count"
        ]
        source_columns = {e.from_node.column_name for e in total_events_edges}
        assert source_columns == {"event_id", "event_type", "timestamp"}, (
            f"Expected edges from all 3 columns, got {source_columns}"
        )

    def test_count_star_unknown_schema(self):
        """
        Test COUNT(*) when schema is NOT known (single query from external table).

        When we don't have upstream schema info, COUNT(*) should fall back to * node.
        """
        queries = [
            (
                "count_external",
                """
            CREATE TABLE analytics.customer_count AS
            SELECT COUNT(*) as total_customers
            FROM external.customers
            """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # When schema is unknown, we should have a * node
        star_nodes = [
            col
            for col in pipeline.columns.values()
            if col.query_id == "count_external" and col.is_star
        ]
        assert len(star_nodes) == 1, (
            f"Expected 1 * node when schema is unknown, got {len(star_nodes)}"
        )

        # The * node should be from external.customers
        assert star_nodes[0].table_name == "external.customers"

        # Should have edge from * to total_customers
        star_edges = [
            e
            for e in pipeline.edges
            if e.from_node.is_star and e.to_node.column_name == "total_customers"
        ]
        assert len(star_edges) == 1


class TestMixedStarUsage:
    """Test queries that mix SELECT * with other patterns."""

    def test_select_star_with_additional_columns(self):
        """
        Test SELECT *, additional_expr pattern.

        Query 2 does SELECT *, CURRENT_TIMESTAMP() as loaded_at.
        Expected: All original columns plus new computed column.
        """
        queries = [
            (
                "create_staging",
                """
            CREATE TABLE staging.orders AS
            SELECT order_id, amount, status
            FROM raw.orders
            """,
            ),
            (
                "add_timestamp",
                """
            CREATE TABLE analytics.orders_with_timestamp AS
            SELECT *, CURRENT_TIMESTAMP() as loaded_at
            FROM staging.orders
            """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # Query 2 should have 4 columns (3 from * + 1 new)
        query2_output = [
            col
            for col in pipeline.columns.values()
            if col.query_id == "add_timestamp" and col.layer == "output"
        ]

        output_col_names = {col.column_name for col in query2_output}
        assert "order_id" in output_col_names
        assert "amount" in output_col_names
        assert "status" in output_col_names
        assert "loaded_at" in output_col_names
        assert len(output_col_names) == 4

    def test_select_star_except_with_replace(self):
        """
        Test combining EXCEPT and REPLACE.

        Query 2 does SELECT * EXCEPT (col1) REPLACE (expr as col2).
        Expected: All columns except col1, with col2 transformed.
        """
        queries = [
            (
                "create_staging",
                """
            CREATE TABLE staging.data AS
            SELECT id, name, email, status, amount
            FROM raw.data
            """,
            ),
            (
                "clean_and_transform",
                """
            CREATE TABLE analytics.processed_data AS
            SELECT * EXCEPT (email) REPLACE (
                UPPER(status) as status,
                ROUND(amount, 2) as amount
            )
            FROM staging.data
            """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # Query 2 should have 4 columns (5 minus email)
        query2_output = [
            col
            for col in pipeline.columns.values()
            if col.query_id == "clean_and_transform" and col.layer == "output"
        ]

        output_col_names = {col.column_name for col in query2_output}
        assert output_col_names == {"id", "name", "status", "amount"}
        assert "email" not in output_col_names


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_select_star_from_unknown_table(self):
        """
        Test SELECT * when upstream table is external (not created by pipeline).

        Query only references an external table.
        Expected: * should not be expanded (we don't know the schema).
        """
        queries = [
            (
                "query_external",
                """
            CREATE TABLE analytics.all_users AS
            SELECT * FROM external.users
            """,
            )
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # Should have * node since we don't know external.users schema
        query_output = [
            col
            for col in pipeline.columns.values()
            if col.query_id == "query_external" and col.layer == "output"
        ]

        # Output should have a * node (not expanded)
        has_star = any(col.is_star for col in query_output)
        assert has_star

    def test_select_star_with_partial_column_overlap(self):
        """
        Test that we don't expand * if only partial columns are selected.

        Query 1 has 5 columns.
        Query 2 explicitly selects 2 columns (not SELECT *).
        Expected: No star expansion, only explicit column edges.
        """
        queries = [
            (
                "create_full",
                """
            CREATE TABLE staging.orders AS
            SELECT order_id, user_id, amount, status, date
            FROM raw.orders
            """,
            ),
            (
                "select_partial",
                """
            CREATE TABLE analytics.summary AS
            SELECT user_id, SUM(amount) as total
            FROM staging.orders
            GROUP BY user_id
            """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # Query 2 output should only have 2 columns
        query2_output = [
            col
            for col in pipeline.columns.values()
            if col.query_id == "select_partial" and col.layer == "output"
        ]

        output_col_names = {col.column_name for col in query2_output}
        assert len(output_col_names) == 2
        assert "user_id" in output_col_names
        assert "total" in output_col_names

    def test_multiple_queries_reading_same_table(self):
        """
        Test multiple downstream queries reading from same upstream table.

        Query 1 creates table.
        Query 2a does SELECT *.
        Query 2b does SELECT specific columns.
        Expected: Both should work correctly.
        """
        queries = [
            (
                "create_base",
                """
            CREATE TABLE staging.orders AS
            SELECT order_id, amount, status
            FROM raw.orders
            """,
            ),
            (
                "copy_all",
                """
            CREATE TABLE analytics.orders_full AS
            SELECT * FROM staging.orders
            """,
            ),
            (
                "copy_partial",
                """
            CREATE TABLE analytics.orders_summary AS
            SELECT order_id, amount FROM staging.orders
            """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # Query copy_all should have 3 columns (expanded)
        copy_all_output = [
            col
            for col in pipeline.columns.values()
            if col.query_id == "copy_all" and col.layer == "output"
        ]
        assert len(copy_all_output) == 3
        assert all(not col.is_star for col in copy_all_output)

        # Query copy_partial should have 2 columns
        copy_partial_output = [
            col
            for col in pipeline.columns.values()
            if col.query_id == "copy_partial" and col.layer == "output"
        ]
        assert len(copy_partial_output) == 2


class TestBackwardCompatibility:
    """Test that existing functionality still works with star expansion."""

    def test_lineage_tracing_through_expanded_star(self):
        """
        Test backward lineage tracing works through expanded SELECT *.

        Trace from final column back through SELECT * to source.
        Expected: Should find original source column.
        """
        queries = [
            (
                "raw_to_staging",
                """
            CREATE TABLE staging.orders AS
            SELECT order_id, amount FROM raw.orders
            """,
            ),
            (
                "staging_to_analytics",
                """
            CREATE TABLE analytics.orders AS
            SELECT * FROM staging.orders
            """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # Trace amount column backward
        sources = pipeline.trace_column_backward("analytics.orders", "amount")

        # Should trace back to raw.orders.amount
        source_names = {(s.table_name, s.column_name) for s in sources}
        assert ("raw.orders", "amount") in source_names

    def test_forward_lineage_through_expanded_star(self):
        """
        Test forward lineage tracing works through expanded SELECT *.

        Trace from source column forward through SELECT * to final.
        Expected: Should find final output column.
        """
        queries = [
            (
                "raw_to_staging",
                """
            CREATE TABLE staging.orders AS
            SELECT order_id, amount FROM raw.orders
            """,
            ),
            (
                "staging_to_analytics",
                """
            CREATE TABLE analytics.orders AS
            SELECT * FROM staging.orders
            """,
            ),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # Trace amount column forward
        descendants = pipeline.trace_column_forward("staging.orders", "amount")

        # Should trace forward to analytics.orders.amount
        desc_names = {(d.table_name, d.column_name) for d in descendants}
        assert ("analytics.orders", "amount") in desc_names
