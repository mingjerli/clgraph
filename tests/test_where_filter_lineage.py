"""
Test suite for Gap 8: WHERE Filter Lineage.

Tests cover:
1. Simple WHERE col = 'value' produces is_where_filter=True edges to all non-star output columns
2. Compound WHERE with multiple column refs produces filter edges for all column refs
3. WHERE x IN (SELECT y FROM other) — subquery columns are NOT in outer where_predicates
4. SELECT * FROM t WHERE t.id = 1 produces NO where_filter edges (no non-star output columns)
5. where_condition attribute contains the clause SQL without the WHERE keyword
6. ColumnEdge has is_where_filter and where_condition as actual dataclass fields

Total: 6 test cases
"""

import dataclasses

import pytest

from clgraph import RecursiveLineageBuilder
from clgraph.models import ColumnEdge

# ============================================================================
# Helpers
# ============================================================================


def _edges_dict(graph):
    """Build a dict keyed by (from_full_name, to_full_name) -> edge."""
    return {(e.from_node.full_name, e.to_node.full_name): e for e in graph.edges}


def _where_filter_edges(graph):
    """Return only edges with is_where_filter=True."""
    return [e for e in graph.edges if e.is_where_filter]


def _where_filter_edges_to(graph, target_full_name):
    """Return where_filter edges targeting a specific output column."""
    return [e for e in graph.edges if e.is_where_filter and e.to_node.full_name == target_full_name]


def _where_filter_sources_to(graph, target_full_name):
    """Return set of from_node.full_name for where_filter edges to a target."""
    return {e.from_node.full_name for e in _where_filter_edges_to(graph, target_full_name)}


# ============================================================================
# Test 1: Simple WHERE col = 'value'
# ============================================================================


class TestSimpleWhereFilter:
    """Simple WHERE col = 'value' produces is_where_filter=True edges to all non-star outputs."""

    SQL = """
    SELECT t.id, t.name, t.city
    FROM my_table t
    WHERE t.status = 'active'
    """

    def test_where_filter_edges_exist(self):
        """WHERE t.status = 'active' produces filter edges to all output columns."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        filter_edges = _where_filter_edges(graph)
        assert len(filter_edges) > 0, "Should have where_filter edges"

    def test_where_filter_edge_to_each_output(self):
        """Each non-star output column gets a where_filter edge from t.status."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        for output_col in ["output.id", "output.name", "output.city"]:
            sources = _where_filter_sources_to(graph, output_col)
            assert "my_table.status" in sources, (
                f"my_table.status should have where_filter edge to {output_col}"
            )

    def test_where_filter_edge_is_not_join_predicate(self):
        """WHERE filter edges should NOT be marked as join predicates."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        for edge in _where_filter_edges(graph):
            assert not edge.is_join_predicate, (
                f"WHERE filter edge should not be is_join_predicate: {edge}"
            )

    def test_where_filter_edge_type(self):
        """WHERE filter edges have edge_type='where_filter'."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        for edge in _where_filter_edges(graph):
            assert edge.edge_type == "where_filter", (
                f"Expected edge_type='where_filter', got '{edge.edge_type}'"
            )


# ============================================================================
# Test 2: Compound WHERE with multiple column refs
# ============================================================================


class TestCompoundWhereFilter:
    """Compound WHERE with OR/AND produces filter edges for all column refs."""

    SQL = """
    SELECT s.id, s.name, s.city
    FROM staging s
    LEFT JOIN dim_customer t ON s.id = t.id
    WHERE t.id IS NULL OR (t.name <> s.name OR t.city <> s.city)
    """

    def test_all_where_columns_produce_filter_edges(self):
        """All columns referenced in the WHERE clause produce filter edges."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        # Collect all from_node column names from where_filter edges
        filter_edge_sources = {e.from_node.column_name for e in _where_filter_edges(graph)}

        # t.id, t.name, t.city, s.name, s.city should all appear as filter sources
        expected_cols = {"id", "name", "city"}
        assert expected_cols.issubset(filter_edge_sources), (
            f"Expected WHERE columns {expected_cols} in filter sources, got {filter_edge_sources}"
        )

    def test_filter_edges_target_non_star_outputs(self):
        """Filter edges target all non-star output columns."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        filter_edges = _where_filter_edges(graph)
        target_names = {e.to_node.full_name for e in filter_edges}

        for expected in ["output.id", "output.name", "output.city"]:
            assert expected in target_names, (
                f"Expected filter edge targeting {expected}, got targets: {target_names}"
            )


# ============================================================================
# Test 3: WHERE with subquery — subquery columns excluded
# ============================================================================


class TestWhereSubqueryExclusion:
    """WHERE x IN (SELECT y FROM other) — subquery columns are NOT in outer where_predicates."""

    SQL = """
    SELECT t.id, t.name
    FROM my_table t
    WHERE t.status = 'active' AND t.id IN (SELECT o.id FROM other_table o)
    """

    def test_outer_where_column_produces_filter_edges(self):
        """t.status from the outer WHERE produces filter edges."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        sources = _where_filter_sources_to(graph, "output.id")
        assert "my_table.status" in sources, (
            "my_table.status should have where_filter edge to output.id"
        )

    def test_subquery_columns_not_in_outer_filter_edges(self):
        """o.id from the subquery should NOT appear as a where_filter source in the outer query."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        filter_edges = _where_filter_edges(graph)
        # Filter edges from the outer query should not reference other_table.id
        outer_filter_sources = {e.from_node.full_name for e in filter_edges}
        assert "other_table.id" not in outer_filter_sources, (
            "Subquery column other_table.id should NOT appear as outer where_filter source"
        )

    def test_outer_where_also_includes_t_id(self):
        """t.id from the outer WHERE (in the IN clause) also produces filter edges."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        sources = _where_filter_sources_to(graph, "output.name")
        assert "my_table.id" in sources, (
            "my_table.id (from IN clause) should have where_filter edge to output.name"
        )


# ============================================================================
# Test 4: SELECT * — no where_filter edges
# ============================================================================


class TestSelectStarNoFilterEdges:
    """SELECT * FROM t WHERE t.id = 1 produces NO where_filter edges."""

    SQL = """
    SELECT * FROM my_table t WHERE t.id = 1
    """

    def test_no_where_filter_edges_for_star(self):
        """Star outputs do not receive where_filter edges."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        filter_edges = _where_filter_edges(graph)
        assert len(filter_edges) == 0, (
            f"SELECT * should produce no where_filter edges, got {len(filter_edges)}"
        )


# ============================================================================
# Test 5: where_condition contains clause SQL without WHERE keyword
# ============================================================================


class TestWhereConditionContent:
    """where_condition attribute contains the clause SQL without the WHERE keyword."""

    SQL = """
    SELECT t.id, t.name
    FROM my_table t
    WHERE t.status = 'active'
    """

    def test_where_condition_does_not_contain_where_keyword(self):
        """where_condition should not start with 'WHERE'."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        filter_edges = _where_filter_edges(graph)
        assert len(filter_edges) > 0, "Should have filter edges"
        for edge in filter_edges:
            assert edge.where_condition is not None, "where_condition should not be None"
            assert not edge.where_condition.strip().upper().startswith("WHERE"), (
                f"where_condition should not start with WHERE keyword: {edge.where_condition}"
            )

    def test_where_condition_contains_status_reference(self):
        """where_condition should contain a reference to the status column."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        filter_edges = _where_filter_edges(graph)
        assert len(filter_edges) > 0, "Should have filter edges"
        # At least one edge should reference 'status' in its condition
        conditions = {edge.where_condition for edge in filter_edges}
        assert any("status" in c.lower() for c in conditions if c), (
            f"where_condition should reference 'status': {conditions}"
        )


# ============================================================================
# Test 6: ColumnEdge has is_where_filter and where_condition as dataclass fields
# ============================================================================


class TestColumnEdgeDataclassFields:
    """ColumnEdge has is_where_filter and where_condition as actual dataclass fields."""

    def test_is_where_filter_is_dataclass_field(self):
        """is_where_filter should be a proper dataclass field on ColumnEdge."""
        field_names = {f.name for f in dataclasses.fields(ColumnEdge)}
        assert "is_where_filter" in field_names, (
            f"is_where_filter should be a dataclass field, found: {field_names}"
        )

    def test_where_condition_is_dataclass_field(self):
        """where_condition should be a proper dataclass field on ColumnEdge."""
        field_names = {f.name for f in dataclasses.fields(ColumnEdge)}
        assert "where_condition" in field_names, (
            f"where_condition should be a dataclass field, found: {field_names}"
        )

    def test_default_values(self):
        """Default values: is_where_filter=False, where_condition=None."""
        field_map = {f.name: f for f in dataclasses.fields(ColumnEdge)}

        is_where_filter_field = field_map.get("is_where_filter")
        assert is_where_filter_field is not None
        assert is_where_filter_field.default is False, (
            f"is_where_filter default should be False, got {is_where_filter_field.default}"
        )

        where_condition_field = field_map.get("where_condition")
        assert where_condition_field is not None
        assert where_condition_field.default is None, (
            f"where_condition default should be None, got {where_condition_field.default}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
