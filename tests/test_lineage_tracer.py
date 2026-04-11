"""
Tests for the lineage_tracer module.

Verifies that:
- Module-level functions in clgraph.lineage_tracer work correctly
- Backward tracing returns source columns
- Forward tracing returns dependent columns
- Full variants return nodes + edges
- Backward compatibility: Pipeline tracing methods still work via delegation
"""

from clgraph import Pipeline

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _simple_pipeline():
    """Two-query pipeline: raw.orders -> staging.orders -> analytics.metrics"""
    queries = [
        (
            "q1",
            """
            CREATE TABLE staging.orders AS
            SELECT
                order_id,
                amount,
                user_id
            FROM raw.orders
            """,
        ),
        (
            "q2",
            """
            CREATE TABLE analytics.metrics AS
            SELECT
                user_id,
                SUM(amount) AS total_revenue
            FROM staging.orders
            GROUP BY user_id
            """,
        ),
    ]
    return Pipeline(queries, dialect="bigquery")


def _three_hop_pipeline():
    """Three-query linear pipeline with CTEs."""
    queries = [
        (
            "q1",
            """
            CREATE TABLE raw.sales AS
            SELECT id, revenue FROM source.transactions
            """,
        ),
        (
            "q2",
            """
            CREATE TABLE staging.sales AS
            SELECT id, revenue FROM raw.sales
            """,
        ),
        (
            "q3",
            """
            CREATE TABLE mart.totals AS
            SELECT SUM(revenue) AS total FROM staging.sales
            """,
        ),
    ]
    return Pipeline(queries, dialect="bigquery")


# ---------------------------------------------------------------------------
# Import the module-level functions
# ---------------------------------------------------------------------------


class TestLinageTracerImport:
    """Smoke-tests: the module must be importable with the expected symbols."""

    def test_import_trace_backward(self):
        from clgraph.lineage_tracer import trace_backward  # noqa: F401

    def test_import_trace_backward_full(self):
        from clgraph.lineage_tracer import trace_backward_full  # noqa: F401

    def test_import_trace_forward(self):
        from clgraph.lineage_tracer import trace_forward  # noqa: F401

    def test_import_trace_forward_full(self):
        from clgraph.lineage_tracer import trace_forward_full  # noqa: F401

    def test_import_get_table_lineage_path(self):
        from clgraph.lineage_tracer import get_table_lineage_path  # noqa: F401

    def test_import_get_table_impact_path(self):
        from clgraph.lineage_tracer import get_table_impact_path  # noqa: F401

    def test_import_get_lineage_path(self):
        from clgraph.lineage_tracer import get_lineage_path  # noqa: F401

    def test_import_get_incoming(self):
        from clgraph.lineage_tracer import _get_incoming  # noqa: F401

    def test_import_get_outgoing(self):
        from clgraph.lineage_tracer import _get_outgoing  # noqa: F401


# ---------------------------------------------------------------------------
# trace_backward
# ---------------------------------------------------------------------------


class TestTraceBackward:
    """Tests for trace_backward()."""

    def test_returns_source_columns(self):
        from clgraph.lineage_tracer import trace_backward

        pipeline = _simple_pipeline()
        sources = trace_backward(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "analytics.metrics",
            "total_revenue",
        )
        assert len(sources) > 0
        # The ultimate source must be in raw.orders
        table_names = [s.table_name for s in sources]
        assert "raw.orders" in table_names

    def test_returns_list_of_column_nodes(self):
        from clgraph.lineage_tracer import trace_backward
        from clgraph.models import ColumnNode

        pipeline = _simple_pipeline()
        sources = trace_backward(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "staging.orders",
            "amount",
        )
        for s in sources:
            assert isinstance(s, ColumnNode)

    def test_unknown_column_returns_empty(self):
        from clgraph.lineage_tracer import trace_backward

        pipeline = _simple_pipeline()
        sources = trace_backward(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "nonexistent.table",
            "nonexistent_col",
        )
        assert sources == []

    def test_source_column_returns_itself(self):
        """A column that is itself a source has no incoming edges → returns itself."""
        from clgraph.lineage_tracer import trace_backward

        pipeline = _simple_pipeline()
        # raw.orders.order_id is an ultimate source
        sources = trace_backward(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "raw.orders",
            "order_id",
        )
        assert any(s.column_name == "order_id" for s in sources)

    def test_three_hop_traces_all_the_way_back(self):
        from clgraph.lineage_tracer import trace_backward

        pipeline = _three_hop_pipeline()
        sources = trace_backward(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "mart.totals",
            "total",
        )
        table_names = [s.table_name for s in sources]
        # Should reach source.transactions
        assert "source.transactions" in table_names


# ---------------------------------------------------------------------------
# trace_backward_full
# ---------------------------------------------------------------------------


class TestTraceBackwardFull:
    """Tests for trace_backward_full()."""

    def test_returns_tuple_of_nodes_and_edges(self):
        from clgraph.lineage_tracer import trace_backward_full

        pipeline = _simple_pipeline()
        result = trace_backward_full(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "analytics.metrics",
            "total_revenue",
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_non_empty_nodes_and_edges(self):
        from clgraph.lineage_tracer import trace_backward_full

        pipeline = _simple_pipeline()
        nodes, edges = trace_backward_full(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "analytics.metrics",
            "total_revenue",
        )
        assert len(nodes) > 0
        assert len(edges) > 0

    def test_nodes_include_target(self):
        from clgraph.lineage_tracer import trace_backward_full

        pipeline = _simple_pipeline()
        nodes, _ = trace_backward_full(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "analytics.metrics",
            "total_revenue",
        )
        target_tables = {n.table_name for n in nodes}
        assert "analytics.metrics" in target_tables

    def test_edges_are_column_edges(self):
        from clgraph.lineage_tracer import trace_backward_full
        from clgraph.models import ColumnEdge

        pipeline = _simple_pipeline()
        _, edges = trace_backward_full(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "analytics.metrics",
            "total_revenue",
        )
        for e in edges:
            assert isinstance(e, ColumnEdge)

    def test_include_ctes_false_skips_cte_nodes(self):
        """When include_ctes=False, CTE-layer nodes should not appear in result."""
        from clgraph.lineage_tracer import trace_backward_full

        # Build a pipeline with a CTE
        queries = [
            (
                "q1",
                """
                CREATE TABLE mart.output AS
                WITH cte AS (
                    SELECT amount FROM raw.data
                )
                SELECT amount FROM cte
                """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")
        nodes, _ = trace_backward_full(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "mart.output",
            "amount",
            include_ctes=False,
        )
        layers = {n.layer for n in nodes}
        assert "cte" not in layers

    def test_unknown_column_returns_empty(self):
        from clgraph.lineage_tracer import trace_backward_full

        pipeline = _simple_pipeline()
        nodes, edges = trace_backward_full(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "no.table",
            "no_col",
        )
        assert nodes == []
        assert edges == []


# ---------------------------------------------------------------------------
# trace_forward
# ---------------------------------------------------------------------------


class TestTraceForward:
    """Tests for trace_forward()."""

    def test_returns_dependent_columns(self):
        from clgraph.lineage_tracer import trace_forward

        pipeline = _simple_pipeline()
        descendants = trace_forward(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "raw.orders",
            "amount",
        )
        assert len(descendants) > 0
        table_names = [d.table_name for d in descendants]
        assert "analytics.metrics" in table_names

    def test_returns_list_of_column_nodes(self):
        from clgraph.lineage_tracer import trace_forward
        from clgraph.models import ColumnNode

        pipeline = _simple_pipeline()
        descendants = trace_forward(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "raw.orders",
            "amount",
        )
        for d in descendants:
            assert isinstance(d, ColumnNode)

    def test_unknown_column_returns_empty(self):
        from clgraph.lineage_tracer import trace_forward

        pipeline = _simple_pipeline()
        result = trace_forward(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "nonexistent.table",
            "nonexistent_col",
        )
        assert result == []

    def test_three_hop_reaches_final_table(self):
        from clgraph.lineage_tracer import trace_forward

        pipeline = _three_hop_pipeline()
        descendants = trace_forward(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "source.transactions",
            "revenue",
        )
        table_names = [d.table_name for d in descendants]
        assert "mart.totals" in table_names


# ---------------------------------------------------------------------------
# trace_forward_full
# ---------------------------------------------------------------------------


class TestTraceForwardFull:
    """Tests for trace_forward_full()."""

    def test_returns_tuple_of_nodes_and_edges(self):
        from clgraph.lineage_tracer import trace_forward_full

        pipeline = _simple_pipeline()
        result = trace_forward_full(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "raw.orders",
            "amount",
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_non_empty_nodes_and_edges(self):
        from clgraph.lineage_tracer import trace_forward_full

        pipeline = _simple_pipeline()
        nodes, edges = trace_forward_full(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "raw.orders",
            "amount",
        )
        assert len(nodes) > 0
        assert len(edges) > 0

    def test_nodes_include_source(self):
        from clgraph.lineage_tracer import trace_forward_full

        pipeline = _simple_pipeline()
        nodes, _ = trace_forward_full(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "raw.orders",
            "amount",
        )
        source_tables = {n.table_name for n in nodes}
        assert "raw.orders" in source_tables

    def test_unknown_column_returns_empty(self):
        from clgraph.lineage_tracer import trace_forward_full

        pipeline = _simple_pipeline()
        nodes, edges = trace_forward_full(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "no.table",
            "no_col",
        )
        assert nodes == []
        assert edges == []

    def test_include_ctes_false_skips_cte_nodes(self):
        """When include_ctes=False, CTE-layer nodes should not appear in result."""
        from clgraph.lineage_tracer import trace_forward_full

        queries = [
            (
                "q1",
                """
                CREATE TABLE mart.output AS
                WITH cte AS (
                    SELECT amount FROM raw.data
                )
                SELECT amount FROM cte
                """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")
        nodes, _ = trace_forward_full(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "raw.data",
            "amount",
            include_ctes=False,
        )
        layers = {n.layer for n in nodes}
        assert "cte" not in layers


# ---------------------------------------------------------------------------
# get_table_lineage_path
# ---------------------------------------------------------------------------


class TestGetTableLineagePath:
    """Tests for get_table_lineage_path()."""

    def test_returns_list_of_tuples(self):
        from clgraph.lineage_tracer import get_table_lineage_path

        pipeline = _simple_pipeline()
        path = get_table_lineage_path(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "analytics.metrics",
            "total_revenue",
            pipeline.table_graph.tables,
        )
        assert isinstance(path, list)
        for item in path:
            assert isinstance(item, tuple)
            assert len(item) == 3  # (table_name, column_name, query_id)

    def test_path_includes_target_table(self):
        from clgraph.lineage_tracer import get_table_lineage_path

        pipeline = _simple_pipeline()
        path = get_table_lineage_path(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "analytics.metrics",
            "total_revenue",
            pipeline.table_graph.tables,
        )
        table_names = [p[0] for p in path]
        assert "analytics.metrics" in table_names

    def test_path_no_duplicates(self):
        from clgraph.lineage_tracer import get_table_lineage_path

        pipeline = _simple_pipeline()
        path = get_table_lineage_path(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "analytics.metrics",
            "total_revenue",
            pipeline.table_graph.tables,
        )
        keys = [(p[0], p[1]) for p in path]
        assert len(keys) == len(set(keys)), "Duplicate table.column found in path"

    def test_unknown_column_returns_empty(self):
        from clgraph.lineage_tracer import get_table_lineage_path

        pipeline = _simple_pipeline()
        path = get_table_lineage_path(
            pipeline.columns,
            pipeline.column_graph._incoming_index,
            "no.table",
            "no_col",
            pipeline.table_graph.tables,
        )
        assert path == []


# ---------------------------------------------------------------------------
# get_table_impact_path
# ---------------------------------------------------------------------------


class TestGetTableImpactPath:
    """Tests for get_table_impact_path()."""

    def test_returns_list_of_tuples(self):
        from clgraph.lineage_tracer import get_table_impact_path

        pipeline = _simple_pipeline()
        path = get_table_impact_path(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "raw.orders",
            "amount",
            pipeline.table_graph.tables,
        )
        assert isinstance(path, list)
        for item in path:
            assert isinstance(item, tuple)
            assert len(item) == 3  # (table_name, column_name, query_id)

    def test_path_includes_final_table(self):
        from clgraph.lineage_tracer import get_table_impact_path

        pipeline = _simple_pipeline()
        path = get_table_impact_path(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "raw.orders",
            "amount",
            pipeline.table_graph.tables,
        )
        table_names = [p[0] for p in path]
        assert "analytics.metrics" in table_names

    def test_path_no_duplicates(self):
        from clgraph.lineage_tracer import get_table_impact_path

        pipeline = _simple_pipeline()
        path = get_table_impact_path(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "raw.orders",
            "amount",
            pipeline.table_graph.tables,
        )
        keys = [(p[0], p[1]) for p in path]
        assert len(keys) == len(set(keys)), "Duplicate table.column found in impact path"

    def test_unknown_column_returns_empty(self):
        from clgraph.lineage_tracer import get_table_impact_path

        pipeline = _simple_pipeline()
        path = get_table_impact_path(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "no.table",
            "no_col",
            pipeline.table_graph.tables,
        )
        assert path == []


# ---------------------------------------------------------------------------
# get_lineage_path
# ---------------------------------------------------------------------------


class TestGetLineagePath:
    """Tests for get_lineage_path()."""

    def test_returns_list_of_edges_when_path_exists(self):
        from clgraph.lineage_tracer import get_lineage_path
        from clgraph.models import ColumnEdge

        pipeline = _simple_pipeline()
        path = get_lineage_path(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "raw.orders",
            "amount",
            "analytics.metrics",
            "total_revenue",
        )
        assert isinstance(path, list)
        assert len(path) > 0
        for e in path:
            assert isinstance(e, ColumnEdge)

    def test_returns_empty_when_no_path(self):
        from clgraph.lineage_tracer import get_lineage_path

        pipeline = _simple_pipeline()
        path = get_lineage_path(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "analytics.metrics",
            "total_revenue",
            "raw.orders",
            "amount",
        )
        assert path == []

    def test_returns_empty_when_from_column_not_found(self):
        from clgraph.lineage_tracer import get_lineage_path

        pipeline = _simple_pipeline()
        path = get_lineage_path(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "no.table",
            "no_col",
            "analytics.metrics",
            "total_revenue",
        )
        assert path == []

    def test_returns_empty_when_to_column_not_found(self):
        from clgraph.lineage_tracer import get_lineage_path

        pipeline = _simple_pipeline()
        path = get_lineage_path(
            pipeline.columns,
            pipeline.column_graph._outgoing_index,
            "raw.orders",
            "amount",
            "no.table",
            "no_col",
        )
        assert path == []


# ---------------------------------------------------------------------------
# Backward-compatibility: Pipeline methods still work
# ---------------------------------------------------------------------------


class TestPipelineBackwardCompatibility:
    """Pipeline.trace_* methods must still work after delegation."""

    def test_trace_column_backward_still_works(self):
        pipeline = _simple_pipeline()
        sources = pipeline.trace_column_backward("analytics.metrics", "total_revenue")
        assert len(sources) > 0
        assert any(s.table_name == "raw.orders" for s in sources)

    def test_trace_column_backward_full_still_works(self):
        pipeline = _simple_pipeline()
        nodes, edges = pipeline.trace_column_backward_full("analytics.metrics", "total_revenue")
        assert len(nodes) > 0
        assert len(edges) > 0

    def test_trace_column_forward_still_works(self):
        pipeline = _simple_pipeline()
        descendants = pipeline.trace_column_forward("raw.orders", "amount")
        assert len(descendants) > 0
        assert any(d.table_name == "analytics.metrics" for d in descendants)

    def test_trace_column_forward_full_still_works(self):
        pipeline = _simple_pipeline()
        nodes, edges = pipeline.trace_column_forward_full("raw.orders", "amount")
        assert len(nodes) > 0
        assert len(edges) > 0

    def test_get_table_lineage_path_still_works(self):
        pipeline = _simple_pipeline()
        path = pipeline.get_table_lineage_path("analytics.metrics", "total_revenue")
        assert isinstance(path, list)
        assert len(path) > 0

    def test_get_table_impact_path_still_works(self):
        pipeline = _simple_pipeline()
        path = pipeline.get_table_impact_path("raw.orders", "amount")
        assert isinstance(path, list)
        assert len(path) > 0

    def test_get_lineage_path_still_works(self):
        pipeline = _simple_pipeline()
        path = pipeline.get_lineage_path(
            "raw.orders", "amount", "analytics.metrics", "total_revenue"
        )
        assert isinstance(path, list)
        assert len(path) > 0

    def test_get_incoming_edges_still_works(self):
        """_get_incoming_edges must remain on Pipeline."""
        pipeline = _simple_pipeline()
        # Pick any column that has an incoming edge
        for full_name, _col in pipeline.columns.items():
            incoming = pipeline._get_incoming_edges(full_name)
            assert isinstance(incoming, list)
            break  # Just test one

    def test_get_outgoing_edges_still_works(self):
        """_get_outgoing_edges must remain on Pipeline."""
        pipeline = _simple_pipeline()
        for full_name, _col in pipeline.columns.items():
            outgoing = pipeline._get_outgoing_edges(full_name)
            assert isinstance(outgoing, list)
            break
