"""
Test suite for Gap 7: JOIN ON Predicate Columns in Column Lineage.

Tests cover:
- CDC/SCD2 point-in-time join (BETWEEN)
- Band join (non-CDC)
- Function-based join (UPPER)
- Multi-join chain
- Existing equi-join tests still pass
- Impact analysis opt-in/opt-out
- Dialect consistency
- Self-referencing query with JOIN predicates (Gap 4 interaction)
- Multi-statement SCD2 pipeline with JOIN predicates (Gap 4 + Gap 7)
- Unqualified predicate column handling

Total: 10 test cases
"""

import pytest

from clgraph import Pipeline, RecursiveLineageBuilder, SQLColumnTracer

# ============================================================================
# Helpers
# ============================================================================


def _edges_dict(graph):
    """Build a dict keyed by (from_full_name, to_full_name) -> edge."""
    return {(e.from_node.full_name, e.to_node.full_name): e for e in graph.edges}


def _predicate_edges(graph):
    """Return only edges with is_join_predicate=True."""
    return [e for e in graph.edges if e.is_join_predicate]


def _predicate_edges_to(graph, target_full_name):
    """Return predicate edges targeting a specific output column."""
    return [
        e for e in graph.edges if e.is_join_predicate and e.to_node.full_name == target_full_name
    ]


def _predicate_sources_to(graph, target_full_name):
    """Return set of from_node.full_name for predicate edges to a target."""
    return {e.from_node.full_name for e in _predicate_edges_to(graph, target_full_name)}


# ============================================================================
# Test 1: CDC/SCD2 point-in-time join
# ============================================================================


class TestCDCSCD2PointInTimeJoin:
    """Test 1: CDC/SCD2 BETWEEN join produces predicate edges."""

    SQL = """
    SELECT o.order_id, o.customer_id, o.order_ts, o.amount,
           d.city AS customer_city_at_order
    FROM raw_orders o
    LEFT JOIN dim_customer d
      ON o.customer_id = d.id
     AND o.order_ts BETWEEN d.start_time AND d.end_time
    """

    def test_predicate_edges_from_dim_customer_start_time(self):
        """dim_customer.start_time has predicate edge to output.customer_city_at_order."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        sources = _predicate_sources_to(graph, "output.customer_city_at_order")
        assert "dim_customer.start_time" in sources

    def test_predicate_edges_from_dim_customer_end_time(self):
        """dim_customer.end_time has predicate edge to output.customer_city_at_order."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        sources = _predicate_sources_to(graph, "output.customer_city_at_order")
        assert "dim_customer.end_time" in sources

    def test_predicate_edges_from_raw_orders_order_ts(self):
        """raw_orders.order_ts has predicate edge to output.customer_city_at_order."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        sources = _predicate_sources_to(graph, "output.customer_city_at_order")
        assert "raw_orders.order_ts" in sources

    def test_predicate_edges_from_raw_orders_customer_id(self):
        """raw_orders.customer_id has predicate edge to output.customer_city_at_order."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        sources = _predicate_sources_to(graph, "output.customer_city_at_order")
        assert "raw_orders.customer_id" in sources

    def test_predicate_edges_from_dim_customer_id(self):
        """dim_customer.id has predicate edge to output.customer_city_at_order."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        sources = _predicate_sources_to(graph, "output.customer_city_at_order")
        assert "dim_customer.id" in sources

    def test_value_edge_not_marked_as_predicate(self):
        """dim_customer.city -> output.customer_city_at_order is NOT is_join_predicate."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        edges = _edges_dict(graph)
        value_edge = edges.get(("dim_customer.city", "output.customer_city_at_order"))
        assert value_edge is not None, "Value edge dim_customer.city -> output should exist"
        assert not value_edge.is_join_predicate, "Value edge should not be marked as join predicate"

    def test_all_predicate_edges_have_join_condition(self):
        """All predicate edges carry join_condition metadata."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        pred_edges = _predicate_edges_to(graph, "output.customer_city_at_order")
        assert len(pred_edges) >= 5, f"Expected at least 5 predicate edges, got {len(pred_edges)}"
        for edge in pred_edges:
            assert edge.join_condition is not None, (
                f"Predicate edge from {edge.from_node.full_name} missing join_condition"
            )
            assert edge.edge_type == "join_predicate", (
                f"Predicate edge should have edge_type='join_predicate', got '{edge.edge_type}'"
            )

    def test_predicate_edges_have_join_side(self):
        """Predicate edges carry join_side ('left' or 'right')."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        pred_edges = _predicate_edges_to(graph, "output.customer_city_at_order")
        sides = {e.from_node.full_name: e.join_side for e in pred_edges}

        # Right-side columns (dim_customer)
        assert sides.get("dim_customer.start_time") == "right"
        assert sides.get("dim_customer.end_time") == "right"
        assert sides.get("dim_customer.id") == "right"

        # Left-side columns (raw_orders)
        assert sides.get("raw_orders.customer_id") == "left"
        assert sides.get("raw_orders.order_ts") == "left"


# ============================================================================
# Test 2: Band join (non-CDC)
# ============================================================================


class TestBandJoin:
    """Test 2: Non-CDC band join with BETWEEN and equi-join produces predicate edges."""

    SQL = """
    SELECT e.event_id, e.event_ts, s.sensor_id, s.reading
    FROM events e
    INNER JOIN sensor_data s
      ON e.event_ts BETWEEN s.reading_ts - INTERVAL '5' MINUTE AND s.reading_ts + INTERVAL '5' MINUTE
     AND e.location_id = s.location_id
    """

    def test_predicate_edges_to_right_side_columns(self):
        """Predicate edges target right-side projected columns (sensor_data outputs)."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        # Check predicate edges to output.sensor_id
        sensor_id_sources = _predicate_sources_to(graph, "output.sensor_id")
        reading_sources = _predicate_sources_to(graph, "output.reading")

        # sensor_data.reading_ts should have predicate edges to right-side outputs
        assert (
            "sensor_data.reading_ts" in sensor_id_sources
            or "sensor_data.reading_ts" in reading_sources
        ), "sensor_data.reading_ts should have predicate edge to a right-side output"

        # events.event_ts should have predicate edges to right-side outputs
        assert "events.event_ts" in sensor_id_sources or "events.event_ts" in reading_sources, (
            "events.event_ts should have predicate edge to a right-side output"
        )

    def test_location_id_predicate_edges(self):
        """Both sides of equi-join in ON clause have predicate edges."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        sensor_id_sources = _predicate_sources_to(graph, "output.sensor_id")
        reading_sources = _predicate_sources_to(graph, "output.reading")

        all_pred_sources = sensor_id_sources | reading_sources

        assert "events.location_id" in all_pred_sources, (
            "events.location_id should have predicate edge to right-side output"
        )
        assert "sensor_data.location_id" in all_pred_sources, (
            "sensor_data.location_id should have predicate edge to right-side output"
        )


# ============================================================================
# Test 3: Function-based join
# ============================================================================


class TestFunctionBasedJoin:
    """Test 3: JOIN with UPPER() function wrapping produces predicate edges."""

    SQL = """
    SELECT a.id, b.name
    FROM table_a a
    INNER JOIN table_b b ON UPPER(a.key) = UPPER(b.key)
    """

    def test_function_wrapped_columns_produce_predicate_edges(self):
        """table_a.key and table_b.key have predicate edges to output.name."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        name_sources = _predicate_sources_to(graph, "output.name")

        assert "table_a.key" in name_sources, (
            "table_a.key should have predicate edge to output.name"
        )
        assert "table_b.key" in name_sources, (
            "table_b.key should have predicate edge to output.name"
        )

    def test_no_predicate_edges_to_left_side_output(self):
        """Predicate edges should not target left-side output (output.id)."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        id_pred_sources = _predicate_sources_to(graph, "output.id")
        assert len(id_pred_sources) == 0, (
            f"No predicate edges expected to output.id, got sources: {id_pred_sources}"
        )


# ============================================================================
# Test 4: Multi-join chain
# ============================================================================


class TestMultiJoinChain:
    """Test 4: Multi-join chain has scoped predicate edges per JOIN."""

    SQL = """
    SELECT a.id, b.val, c.label
    FROM table_a a
    INNER JOIN table_b b ON a.id = b.a_id
    INNER JOIN table_c c ON b.id = c.b_id AND b.category = c.category
    """

    def test_first_join_predicate_edges_to_output_val(self):
        """First join: a.id, b.a_id predicate edges to output.val."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        val_sources = _predicate_sources_to(graph, "output.val")

        assert "table_a.id" in val_sources, (
            "table_a.id should have predicate edge to output.val (first join)"
        )
        assert "table_b.a_id" in val_sources, (
            "table_b.a_id should have predicate edge to output.val (first join)"
        )

    def test_second_join_predicate_edges_to_output_label(self):
        """Second join: b.id, c.b_id, b.category, c.category predicate edges to output.label."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        label_sources = _predicate_sources_to(graph, "output.label")

        assert "table_b.id" in label_sources, (
            "table_b.id should have predicate edge to output.label (second join)"
        )
        assert "table_c.b_id" in label_sources, (
            "table_c.b_id should have predicate edge to output.label (second join)"
        )
        assert "table_b.category" in label_sources, (
            "table_b.category should have predicate edge to output.label (second join)"
        )
        assert "table_c.category" in label_sources, (
            "table_c.category should have predicate edge to output.label (second join)"
        )

    def test_first_join_predicates_do_not_target_output_label(self):
        """First join's predicates (a.id, b.a_id) should NOT have edges to output.label."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        label_sources = _predicate_sources_to(graph, "output.label")

        assert "table_a.id" not in label_sources, (
            "table_a.id (first join) should NOT have predicate edge to output.label"
        )
        assert "table_b.a_id" not in label_sources, (
            "table_b.a_id (first join) should NOT have predicate edge to output.label"
        )


# ============================================================================
# Test 5: Existing equi-join tests still pass
# ============================================================================


class TestExistingEquiJoinTestsPass:
    """Test 5: Verify existing join tests are not broken by predicate edge additions."""

    def test_existing_join_types_tests_pass(self):
        """
        Verify that existing test_join_types.py tests still pass.

        This is a meta-test: the actual verification is done by running
        `uv run pytest tests/test_join_types.py -q` during CI.
        Here we verify a representative case: simple INNER JOIN value edges
        are unchanged after Gap 7 predicate edges are added.
        """
        sql = """
        SELECT u.id, u.name, o.order_id, o.amount
        FROM users u
        INNER JOIN orders o ON u.id = o.user_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        edges = _edges_dict(graph)

        # Existing value edges are unchanged
        assert ("users.id", "output.id") in edges
        assert ("users.name", "output.name") in edges
        assert ("orders.order_id", "output.order_id") in edges
        assert ("orders.amount", "output.amount") in edges

        # Value edges are NOT marked as join predicates
        assert not edges[("users.id", "output.id")].is_join_predicate
        assert not edges[("users.name", "output.name")].is_join_predicate


# ============================================================================
# Test 6: Impact analysis opt-in/opt-out
# ============================================================================


class TestImpactAnalysisOptInOut:
    """Test 6: Forward lineage includes predicate columns; flag is accessible."""

    SQL = """
    SELECT o.order_id, o.customer_id, o.order_ts, o.amount,
           d.city AS customer_city_at_order
    FROM raw_orders o
    LEFT JOIN dim_customer d
      ON o.customer_id = d.id
     AND o.order_ts BETWEEN d.start_time AND d.end_time
    """

    def test_forward_lineage_includes_predicate_column(self):
        """Forward lineage from dim_customer.start_time includes customer_city_at_order."""
        tracer = SQLColumnTracer(self.SQL, dialect="bigquery")
        forward = tracer.get_forward_lineage(["dim_customer.start_time"])

        assert "customer_city_at_order" in forward["impacted_outputs"], (
            f"customer_city_at_order should be impacted by dim_customer.start_time; "
            f"got {forward['impacted_outputs']}"
        )

    def test_is_join_predicate_flag_accessible(self):
        """The is_join_predicate flag is accessible on edges from the graph."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        pred_edges = _predicate_edges(graph)
        assert len(pred_edges) > 0, "Should have predicate edges"

        # Verify we can filter for value-only edges
        value_edges = [e for e in graph.edges if not e.is_join_predicate]
        assert len(value_edges) > 0, "Should have value edges"
        assert len(value_edges) < len(graph.edges), (
            "Value edges should be fewer than total edges (some are predicate)"
        )


# ============================================================================
# Test 7: Dialect consistency
# ============================================================================


class TestDialectConsistency:
    """Test 7: CDC BETWEEN join produces identical predicate edges across dialects."""

    SQL = """
    SELECT o.order_id, o.customer_id, o.order_ts, o.amount,
           d.city AS customer_city_at_order
    FROM raw_orders o
    LEFT JOIN dim_customer d
      ON o.customer_id = d.id
     AND o.order_ts BETWEEN d.start_time AND d.end_time
    """

    DIALECTS = ["bigquery", "postgres", "snowflake", "databricks"]

    @pytest.mark.parametrize("dialect", DIALECTS)
    def test_predicate_edges_consistent_across_dialects(self, dialect):
        """Predicate edges from CDC BETWEEN join are consistent across dialects."""
        builder = RecursiveLineageBuilder(self.SQL, dialect=dialect)
        graph = builder.build()

        sources = _predicate_sources_to(graph, "output.customer_city_at_order")

        assert "dim_customer.start_time" in sources, (
            f"[{dialect}] dim_customer.start_time should have predicate edge"
        )
        assert "dim_customer.end_time" in sources, (
            f"[{dialect}] dim_customer.end_time should have predicate edge"
        )
        assert "raw_orders.order_ts" in sources, (
            f"[{dialect}] raw_orders.order_ts should have predicate edge"
        )
        assert "raw_orders.customer_id" in sources, (
            f"[{dialect}] raw_orders.customer_id should have predicate edge"
        )
        assert "dim_customer.id" in sources, (
            f"[{dialect}] dim_customer.id should have predicate edge"
        )

        # Value edge exists and is not a predicate
        edges = _edges_dict(graph)
        value_edge = edges.get(("dim_customer.city", "output.customer_city_at_order"))
        assert value_edge is not None, f"[{dialect}] Value edge should exist"
        assert not value_edge.is_join_predicate, f"[{dialect}] Value edge should not be predicate"


# ============================================================================
# Test 8: Self-referencing query with JOIN predicates (Gap 4 interaction)
# ============================================================================


class TestSelfRefWithJoinPredicates:
    """Test 8: Single-query self-referencing INSERT with JOIN predicates."""

    SQL = """\
INSERT INTO dim_customer
SELECT s.id, s.name, s.city, s.email,
       COALESCE(t.is_active, 'Y') AS is_active
FROM staging s
LEFT JOIN dim_customer t
  ON s.id = t.id AND t.is_active = 'Y'
WHERE t.id IS NULL OR (t.name <> s.name OR t.city <> s.city)
"""

    def test_self_read_nodes_exist(self):
        """Gap 4: self-read nodes should exist for dim_customer."""
        pipeline = Pipeline(
            queries=[("q0", self.SQL)],
            dialect="bigquery",
        )

        self_read_nodes = [
            col for col in pipeline.columns.values() if ":self_read:dim_customer." in col.full_name
        ]
        assert len(self_read_nodes) > 0, "Self-read nodes should exist for dim_customer"

    def test_predicate_edges_from_self_read_nodes(self):
        """Gap 7: predicate edges should originate from self-read nodes, not physical nodes."""
        pipeline = Pipeline(
            queries=[("q0", self.SQL)],
            dialect="bigquery",
        )

        pred_edges = [e for e in pipeline.edges if e.is_join_predicate]

        # There should be some predicate edges
        assert len(pred_edges) > 0, "Should have predicate edges from JOIN ON clause"

        # Predicate edges from the dim_customer side should reference self-read nodes
        dim_pred_edges = [e for e in pred_edges if "dim_customer" in e.from_node.full_name]
        for edge in dim_pred_edges:
            assert (
                ":self_read:" in edge.from_node.full_name or edge.from_node.node_type == "self_read"
            ), (
                f"Predicate edge from dim_customer should originate from self-read node, "
                f"got {edge.from_node.full_name} (node_type={edge.from_node.node_type})"
            )

    def test_no_predicate_edge_from_where_only_columns(self):
        """WHERE-only columns (t.name, t.city in WHERE but not ON) should NOT have predicate edges."""
        pipeline = Pipeline(
            queries=[("q0", self.SQL)],
            dialect="bigquery",
        )

        pred_edges = [e for e in pipeline.edges if e.is_join_predicate]
        {e.from_node.column_name for e in pred_edges}

        # t.name and t.city appear in WHERE but not in ON clause
        # They should NOT have join_predicate edges
        # (Note: they may have other edge types, but not is_join_predicate)
        on_clause_col_names = {"id", "is_active"}  # columns in the ON clause from dim_customer
        for edge in pred_edges:
            if "dim_customer" in edge.from_node.full_name:
                assert edge.from_node.column_name in on_clause_col_names, (
                    f"Predicate edge from dim_customer column '{edge.from_node.column_name}' "
                    f"should only come from ON-clause columns {on_clause_col_names}"
                )


# ============================================================================
# Test 9: Multi-statement SCD2 pipeline with JOIN predicates (Gap 4 + Gap 7)
# ============================================================================


SCD2_MERGE_SQL = """\
MERGE INTO dim_customer t
USING staging_customer_latest s ON t.id = s.id AND t.is_active = 'Y'
WHEN MATCHED AND (t.name <> s.name OR t.city <> s.city) THEN
  UPDATE SET t.end_time = current_timestamp(), t.is_active = 'N'
"""

SCD2_INSERT_SQL = """\
INSERT INTO dim_customer
SELECT s.id, s.name, s.city, s.email,
       current_timestamp() AS start_time,
       TIMESTAMP '9999-12-31 00:00:00' AS end_time,
       COALESCE(t.is_active, 'Y') AS is_active
FROM staging_customer_latest s
LEFT JOIN dim_customer t
  ON s.id = t.id AND t.is_active = 'Y'
WHERE t.id IS NULL OR (t.name <> s.name OR t.city <> s.city)
"""


class TestMultiStatementSCD2Pipeline:
    """Test 9: Two-step SCD2 pipeline (MERGE + INSERT) with Gap 4 + Gap 7 interaction."""

    @pytest.fixture
    def scd2_pipeline(self):
        return Pipeline(
            queries=[
                ("step1_merge", SCD2_MERGE_SQL),
                ("step2_insert", SCD2_INSERT_SQL),
            ],
            dialect="bigquery",
        )

    def test_step2_on_clause_predicate_edges_exist(self, scd2_pipeline):
        """Step 2's ON-clause predicate columns produce predicate edges."""
        pred_edges = [e for e in scd2_pipeline.edges if e.is_join_predicate]
        assert len(pred_edges) > 0, "Step 2 JOIN should produce predicate edges"

    def test_step2_predicate_edges_from_self_read_nodes(self, scd2_pipeline):
        """Step 2's predicate edges from dim_customer should use self-read nodes."""
        pred_edges = [e for e in scd2_pipeline.edges if e.is_join_predicate]

        dim_pred_edges = [e for e in pred_edges if "dim_customer" in e.from_node.full_name]
        for edge in dim_pred_edges:
            assert (
                ":self_read:" in edge.from_node.full_name or edge.from_node.node_type == "self_read"
            ), (
                f"Step 2 predicate edge from dim_customer should be from self-read node, "
                f"got {edge.from_node.full_name}"
            )

    def test_cross_query_edges_exist(self, scd2_pipeline):
        """Gap 4: cross-query edges from Step 1 output to Step 2 self-read exist."""
        cross_query_edges = [
            e for e in scd2_pipeline.edges if e.edge_role == "cross_query_self_ref"
        ]
        assert len(cross_query_edges) > 0, "Cross-query self-ref edges should exist"

    def test_self_read_nodes_exist(self, scd2_pipeline):
        """Gap 4: self-read nodes for dim_customer should exist."""
        self_read_nodes = [
            col
            for col in scd2_pipeline.columns.values()
            if ":self_read:dim_customer." in col.full_name
        ]
        assert len(self_read_nodes) > 0, "Self-read nodes for dim_customer should exist"


# ============================================================================
# Test 10: Unqualified predicate column emits warning
# ============================================================================


class TestUnqualifiedPredicateColumn:
    """Test 10: Unqualified output column with qualified ON-clause columns."""

    SQL = """
    SELECT a.id, name
    FROM table_a a
    INNER JOIN table_b b ON a.id = b.id
    """

    def test_unqualified_output_prevents_predicate_edge_targeting(self):
        """Unqualified output 'name' cannot be traced to a specific table side.

        When the only right-side projected column is unqualified, the implementation
        cannot determine it is sourced from the right table (table_b), so no
        predicate edges are emitted. This is expected: ambiguous columns do not
        produce predicate edges. The validation system emits a warning instead.
        """
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        pred_edges = _predicate_edges(graph)

        # No predicate edges are expected because the only right-side output
        # column ('name') is unqualified and cannot be attributed to table_b.
        assert len(pred_edges) == 0, (
            f"No predicate edges expected when right-side output is unqualified; "
            f"got {len(pred_edges)} edges"
        )

    def test_unqualified_name_column_still_resolves(self):
        """The unqualified 'name' column should still produce a value edge."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        # output.name should exist regardless of qualification ambiguity
        assert "output.name" in graph.nodes, "output.name should exist in graph"

        # It should have at least one value (non-predicate) edge
        value_edges_to_name = [
            e
            for e in graph.edges
            if e.to_node.full_name == "output.name" and not e.is_join_predicate
        ]
        # The unqualified 'name' may resolve to table_a.name or table_b.name
        # depending on the implementation. Either way, a value edge should exist.
        assert len(value_edges_to_name) >= 1, (
            "Unqualified 'name' should have at least one value edge"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
