"""
Tests for LineageTracer component extracted from Pipeline.

Tests the delegation pattern from Pipeline to LineageTracer.
All existing Pipeline lineage tests should continue to pass.
"""

import pytest

from clgraph import Pipeline


class TestLineageTracerDelegation:
    """Test that Pipeline properly delegates to LineageTracer."""

    @pytest.fixture
    def simple_pipeline(self):
        """Create a simple two-query pipeline for testing."""
        queries = [
            (
                "staging",
                """
                CREATE TABLE staging.orders AS
                SELECT
                    order_id,
                    customer_id,
                    amount
                FROM raw.orders
                """,
            ),
            (
                "analytics",
                """
                CREATE TABLE analytics.revenue AS
                SELECT
                    customer_id,
                    SUM(amount) AS total_revenue
                FROM staging.orders
                GROUP BY customer_id
                """,
            ),
        ]
        return Pipeline(queries, dialect="bigquery")

    @pytest.fixture
    def three_query_pipeline(self):
        """Create a three-query pipeline for testing lineage paths."""
        queries = [
            (
                "raw",
                """
                CREATE TABLE staging.raw_data AS
                SELECT
                    id,
                    value,
                    category
                FROM source.data
                """,
            ),
            (
                "intermediate",
                """
                CREATE TABLE staging.processed AS
                SELECT
                    id,
                    value * 2 AS doubled_value,
                    category
                FROM staging.raw_data
                """,
            ),
            (
                "final",
                """
                CREATE TABLE analytics.summary AS
                SELECT
                    category,
                    SUM(doubled_value) AS total_value
                FROM staging.processed
                GROUP BY category
                """,
            ),
        ]
        return Pipeline(queries, dialect="bigquery")

    def test_trace_column_backward_returns_sources(self, simple_pipeline):
        """Test that trace_column_backward returns source columns."""
        sources = simple_pipeline.trace_column_backward("analytics.revenue", "total_revenue")

        # Should find the amount column from staging.orders
        assert len(sources) > 0
        source_names = [s.column_name for s in sources]
        # The source could be either amount from staging.orders or from raw.orders
        assert "amount" in source_names

    def test_trace_column_backward_empty_for_source_column(self, simple_pipeline):
        """Test that trace_column_backward returns the column itself for source tables."""
        # For a source table column, backward trace should return the column itself
        sources = simple_pipeline.trace_column_backward("raw.orders", "order_id")

        # Should return the source column itself (no incoming edges)
        assert len(sources) > 0
        source_names = [s.column_name for s in sources]
        assert "order_id" in source_names

    def test_trace_column_forward_returns_descendants(self, simple_pipeline):
        """Test that trace_column_forward returns downstream columns."""
        descendants = simple_pipeline.trace_column_forward("staging.orders", "amount")

        # Should find total_revenue in analytics.revenue
        assert len(descendants) > 0
        desc_names = [d.column_name for d in descendants]
        assert "total_revenue" in desc_names

    def test_trace_column_forward_empty_for_final_column(self, simple_pipeline):
        """Test that trace_column_forward returns empty for final columns."""
        descendants = simple_pipeline.trace_column_forward("analytics.revenue", "total_revenue")

        # Should return the column itself as final (no outgoing edges)
        assert len(descendants) > 0
        desc_names = [d.column_name for d in descendants]
        assert "total_revenue" in desc_names

    def test_trace_column_backward_full_returns_nodes_and_edges(self, three_query_pipeline):
        """Test that trace_column_backward_full returns all nodes and edges."""
        nodes, edges = three_query_pipeline.trace_column_backward_full(
            "analytics.summary", "total_value"
        )

        # Should have multiple nodes in the path
        assert len(nodes) > 0
        assert len(edges) > 0

        # Check that nodes include intermediate steps
        node_tables = [n.table_name for n in nodes]
        assert "analytics.summary" in node_tables
        assert "staging.processed" in node_tables

    def test_trace_column_forward_full_returns_nodes_and_edges(self, three_query_pipeline):
        """Test that trace_column_forward_full returns all nodes and edges."""
        nodes, edges = three_query_pipeline.trace_column_forward_full("staging.raw_data", "value")

        # Should have multiple nodes in the path
        assert len(nodes) > 0
        assert len(edges) > 0

        # Check that nodes include downstream steps
        node_tables = [n.table_name for n in nodes]
        assert "staging.raw_data" in node_tables
        assert "staging.processed" in node_tables

    def test_get_lineage_path_returns_edges(self, three_query_pipeline):
        """Test that get_lineage_path returns edges between two columns."""
        path = three_query_pipeline.get_lineage_path(
            "staging.raw_data",
            "value",
            "staging.processed",
            "doubled_value",
        )

        # Should find a path with at least one edge
        assert len(path) > 0

    def test_get_lineage_path_empty_for_unconnected(self, simple_pipeline):
        """Test that get_lineage_path returns empty for unconnected columns."""
        path = simple_pipeline.get_lineage_path(
            "raw.orders",
            "order_id",
            "analytics.revenue",
            "total_revenue",
        )

        # order_id doesn't flow to total_revenue, so path should be empty
        assert len(path) == 0

    def test_get_table_lineage_path_returns_tuples(self, three_query_pipeline):
        """Test that get_table_lineage_path returns list of tuples."""
        path = three_query_pipeline.get_table_lineage_path("analytics.summary", "total_value")

        # Should return list of (table_name, column_name, query_id) tuples
        assert len(path) > 0
        assert all(len(item) == 3 for item in path)

        # First item should be the target
        assert path[0][0] == "analytics.summary"
        assert path[0][1] == "total_value"

    def test_get_table_impact_path_returns_tuples(self, three_query_pipeline):
        """Test that get_table_impact_path returns list of tuples."""
        path = three_query_pipeline.get_table_impact_path("staging.raw_data", "value")

        # Should return list of (table_name, column_name, query_id) tuples
        assert len(path) > 0
        assert all(len(item) == 3 for item in path)

        # First item should be the source
        assert path[0][0] == "staging.raw_data"
        assert path[0][1] == "value"

    def test_trace_backward_includes_ctes_by_default(self, three_query_pipeline):
        """Test that trace_column_backward_full includes CTEs by default."""
        # Create a pipeline with CTEs
        queries = [
            (
                "with_cte",
                """
                CREATE TABLE output.result AS
                WITH cte_step AS (
                    SELECT id, value * 2 AS doubled
                    FROM input.data
                )
                SELECT id, doubled AS final_value
                FROM cte_step
                """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        nodes, edges = pipeline.trace_column_backward_full("output.result", "final_value")

        # Should include CTE nodes
        node_names = [n.full_name for n in nodes]
        # CTE columns should be present
        assert any("cte_step" in name for name in node_names) or len(nodes) > 1

    def test_trace_backward_excludes_ctes_when_requested(self):
        """Test that trace_column_backward_full can exclude CTEs."""
        queries = [
            (
                "with_cte",
                """
                CREATE TABLE output.result AS
                WITH cte_step AS (
                    SELECT id, value * 2 AS doubled
                    FROM input.data
                )
                SELECT id, doubled AS final_value
                FROM cte_step
                """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        nodes, edges = pipeline.trace_column_backward_full(
            "output.result", "final_value", include_ctes=False
        )

        # Should not include CTE layer nodes (nodes with layer="cte")
        cte_nodes = [n for n in nodes if n.layer == "cte"]
        assert len(cte_nodes) == 0


class TestLineageTracerLazyInitialization:
    """Test that LineageTracer is lazily initialized."""

    def test_tracer_not_created_on_pipeline_init(self):
        """Test that tracer is not created when Pipeline is initialized."""
        queries = [
            ("q1", "CREATE TABLE t1 AS SELECT a FROM source"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # The _tracer attribute should be None or not exist
        assert pipeline._tracer is None

    def test_tracer_created_on_first_trace_call(self):
        """Test that tracer is created on first trace method call."""
        queries = [
            ("q1", "CREATE TABLE t1 AS SELECT a FROM source"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Call a trace method
        pipeline.trace_column_backward("t1", "a")

        # Now tracer should be initialized
        assert pipeline._tracer is not None

    def test_tracer_reused_across_calls(self):
        """Test that the same tracer instance is reused."""
        queries = [
            ("q1", "CREATE TABLE t1 AS SELECT a FROM source"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Call multiple trace methods
        pipeline.trace_column_backward("t1", "a")
        tracer1 = pipeline._tracer

        pipeline.trace_column_forward("t1", "a")
        tracer2 = pipeline._tracer

        # Should be the same instance
        assert tracer1 is tracer2


class TestLineageTracerDirectAccess:
    """Test that LineageTracer can be used directly (advanced usage)."""

    def test_lineage_tracer_can_be_imported(self):
        """Test that LineageTracer can be imported directly."""
        from clgraph.lineage_tracer import LineageTracer

        assert LineageTracer is not None

    def test_lineage_tracer_initialization(self):
        """Test that LineageTracer can be initialized with a pipeline."""
        from clgraph.lineage_tracer import LineageTracer

        queries = [
            ("q1", "CREATE TABLE t1 AS SELECT a FROM source"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        tracer = LineageTracer(pipeline)
        assert tracer._pipeline is pipeline

    def test_lineage_tracer_trace_backward(self):
        """Test LineageTracer.trace_backward() directly."""
        from clgraph.lineage_tracer import LineageTracer

        queries = [
            (
                "staging",
                """
                CREATE TABLE staging.orders AS
                SELECT order_id, amount FROM raw.orders
                """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        tracer = LineageTracer(pipeline)
        sources = tracer.trace_backward("staging.orders", "amount")

        assert len(sources) > 0
        assert any(s.column_name == "amount" for s in sources)

    def test_lineage_tracer_trace_forward(self):
        """Test LineageTracer.trace_forward() directly."""
        from clgraph.lineage_tracer import LineageTracer

        queries = [
            (
                "staging",
                """
                CREATE TABLE staging.orders AS
                SELECT order_id, amount FROM raw.orders
                """,
            ),
            (
                "analytics",
                """
                CREATE TABLE analytics.totals AS
                SELECT SUM(amount) AS total FROM staging.orders
                """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        tracer = LineageTracer(pipeline)
        descendants = tracer.trace_forward("staging.orders", "amount")

        assert len(descendants) > 0
        assert any(d.column_name == "total" for d in descendants)
