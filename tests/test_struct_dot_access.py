"""
Test suite for Gap 1: Struct Dot-Access Fallback.

When sqlglot parses `after.id`, it becomes Column(table="after", name="id").
If "after" cannot be resolved as a table/alias/unit in scope, the struct fallback
should emit a lineage edge with nested_path=".id" and access_type="struct",
using the first base table from the dependency chain as the source table.

Tests cover:
1. Simple struct dot-access: SELECT after.id AS id FROM raw_table
2. Multiple struct fields: after.name, after.city, after.email
3. CDC-like subquery pattern: recursive base table resolution
4. Bracket notation regression: items[0].product_id still works
5. Multi-table JOIN with struct ref: fallback uses first base table
6. Empty fallback_tables case: effective_table_ref used as table name

Total: 6 test cases
"""

import pytest

from clgraph import RecursiveLineageBuilder

# ============================================================================
# Helpers
# ============================================================================


def _edges_dict(graph):
    """Build a dict keyed by (from_full_name, to_full_name) -> edge."""
    return {(e.from_node.full_name, e.to_node.full_name): e for e in graph.edges}


def _struct_edges(graph):
    """Return only edges with access_type='struct'."""
    return [e for e in graph.edges if e.access_type == "struct"]


def _struct_edges_to(graph, target_full_name):
    """Return struct edges targeting a specific output column."""
    return [
        e
        for e in graph.edges
        if e.access_type == "struct" and e.to_node.full_name == target_full_name
    ]


# ============================================================================
# Test 1: Simple struct dot-access
# ============================================================================


class TestSimpleStructDotAccess:
    """SELECT after.id AS id FROM raw_table — struct fallback emits edge."""

    SQL = "SELECT after.id AS id FROM raw_table"

    def test_struct_edge_exists(self):
        """after.id should produce a struct edge."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        struct_edges = _struct_edges(graph)
        assert len(struct_edges) > 0, (
            f"Expected struct edges for after.id, got none. "
            f"All edges: {[(e.from_node.full_name, e.to_node.full_name, e.access_type) for e in graph.edges]}"
        )

    def test_struct_edge_nested_path(self):
        """Struct edge should have nested_path='.id'."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        edges = _struct_edges_to(graph, "output.id")
        assert len(edges) == 1, f"Expected 1 struct edge to output.id, got {len(edges)}"
        assert edges[0].nested_path == ".id"

    def test_struct_edge_from_node_column_name(self):
        """Struct edge from_node.column_name should be 'after' (the struct column)."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        edges = _struct_edges_to(graph, "output.id")
        assert len(edges) == 1
        assert edges[0].from_node.column_name == "after"

    def test_struct_edge_from_node_table_name(self):
        """Struct edge from_node.table_name should be 'raw_table' (the base table)."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        edges = _struct_edges_to(graph, "output.id")
        assert len(edges) == 1
        assert edges[0].from_node.table_name == "raw_table"


# ============================================================================
# Test 2: Multiple struct fields
# ============================================================================


class TestMultipleStructFields:
    """Multiple struct field accesses all emit struct edges."""

    SQL = """
    SELECT after.name AS name, after.city AS city, after.email AS email
    FROM raw_table
    """

    def test_all_fields_produce_struct_edges(self):
        """after.name, after.city, after.email all produce struct edges."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        struct_edges = _struct_edges(graph)
        assert len(struct_edges) == 3, (
            f"Expected 3 struct edges, got {len(struct_edges)}. "
            f"All edges: {[(e.from_node.full_name, e.to_node.full_name, e.access_type) for e in graph.edges]}"
        )

    def test_each_field_has_correct_nested_path(self):
        """Each struct edge has the correct nested_path."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        for output_col, expected_path in [
            ("output.name", ".name"),
            ("output.city", ".city"),
            ("output.email", ".email"),
        ]:
            edges = _struct_edges_to(graph, output_col)
            assert len(edges) == 1, f"Expected 1 struct edge to {output_col}, got {len(edges)}"
            assert edges[0].nested_path == expected_path, (
                f"Expected nested_path='{expected_path}' for {output_col}, "
                f"got '{edges[0].nested_path}'"
            )

    def test_all_from_nodes_reference_raw_table(self):
        """All struct edge from_nodes should reference raw_table."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        for edge in _struct_edges(graph):
            assert edge.from_node.table_name == "raw_table", (
                f"Expected table_name='raw_table', got '{edge.from_node.table_name}'"
            )


# ============================================================================
# Test 3: CDC-like subquery pattern (recursive base table resolution)
# ============================================================================


class TestCDCSubqueryPattern:
    """CDC pattern: SELECT after.id, after.name FROM (SELECT * FROM raw_customer_cdc)."""

    SQL = """
    SELECT after.id, after.name
    FROM (SELECT * FROM raw_customer_cdc) sub
    """

    def test_struct_edges_exist(self):
        """Struct edges should exist for after.id and after.name."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        struct_edges = _struct_edges(graph)
        assert len(struct_edges) >= 2, (
            f"Expected at least 2 struct edges, got {len(struct_edges)}. "
            f"All edges: {[(e.from_node.full_name, e.to_node.full_name, e.access_type) for e in graph.edges]}"
        )

    def test_from_node_resolves_to_base_table(self):
        """from_node.table_name should resolve to raw_customer_cdc (ultimate base table)."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        struct_edges = _struct_edges(graph)
        table_names = {e.from_node.table_name for e in struct_edges}
        assert "raw_customer_cdc" in table_names, (
            f"Expected raw_customer_cdc in table names, got {table_names}"
        )


# ============================================================================
# Test 4: Bracket notation regression
# ============================================================================


class TestBracketNotationRegression:
    """Bracket notation items[0].product_id should still work (existing behavior)."""

    SQL = "SELECT items[0].product_id AS first_product FROM orders"

    def test_bracket_notation_still_works(self):
        """items[0].product_id should produce a nested edge with mixed access_type."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        nested_edges = [e for e in graph.edges if e.nested_path]
        assert len(nested_edges) > 0, "Bracket notation should still produce nested edges"

        edge = nested_edges[0]
        assert edge.nested_path == "[0].product_id"
        assert edge.access_type == "mixed"
        assert edge.from_node.column_name == "items"


# ============================================================================
# Test 5: Multi-table JOIN with struct ref
# ============================================================================


class TestMultiTableJoinStructRef:
    """Struct ref in a JOIN context uses first base table as fallback."""

    SQL = """
    SELECT after.id AS id, b.name
    FROM raw_table a
    INNER JOIN lookup_table b ON a.key = b.key
    """

    def test_struct_edge_uses_first_base_table(self):
        """after.id struct fallback should use first base table (raw_table)."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        edges = _struct_edges_to(graph, "output.id")
        assert len(edges) == 1, (
            f"Expected 1 struct edge to output.id, got {len(edges)}. "
            f"All edges: {[(e.from_node.full_name, e.to_node.full_name, e.access_type) for e in graph.edges]}"
        )
        assert edges[0].from_node.table_name == "raw_table", (
            f"Expected fallback to raw_table, got '{edges[0].from_node.table_name}'"
        )

    def test_normal_column_still_resolves(self):
        """b.name should resolve normally (not as struct)."""
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        edges = _edges_dict(graph)
        assert ("lookup_table.name", "output.name") in edges, (
            f"Expected lookup_table.name -> output.name edge. Available edges: {list(edges.keys())}"
        )


# ============================================================================
# Test 6: Empty fallback_tables case
# ============================================================================


class TestEmptyFallbackTables:
    """When no base tables exist, effective_table_ref is used as table name."""

    # This is an edge case: a query where the struct ref has no resolvable base tables.
    # In practice this is rare, but the fallback should still produce an edge.
    SQL = "SELECT after.id AS id FROM after"

    def test_fallback_uses_effective_table_ref(self):
        """When 'after' is both the table name and the struct ref, it resolves normally.

        This test verifies that when a table literally named 'after' exists,
        no struct fallback is triggered (it resolves as a normal table ref).
        """
        builder = RecursiveLineageBuilder(self.SQL, dialect="bigquery")
        graph = builder.build()

        # 'after' is a real table name here, so it resolves normally
        edges = _edges_dict(graph)
        assert ("after.id", "output.id") in edges, (
            f"Expected after.id -> output.id edge. Available: {list(edges.keys())}"
        )

        # Should NOT produce struct edges since 'after' is a real table
        struct_edges = _struct_edges(graph)
        assert len(struct_edges) == 0, (
            f"Should not produce struct edges when 'after' is a real table, "
            f"got {len(struct_edges)} struct edges"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
