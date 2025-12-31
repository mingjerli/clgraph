"""
Test suite for Struct/Array Subscript support in column lineage tracking.

Tests cover:
- Array subscript access (items[0], data[1])
- Map/Dictionary key access (metadata['key'])
- Struct field access after array subscript (items[0].product_id)
- Mixed nested access patterns
- Export format validation
"""

import pytest

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder

# ============================================================================
# Test Group 1: Array Subscript Detection
# ============================================================================


class TestArraySubscriptDetection:
    """Test array index access detection."""

    def test_simple_array_access(self):
        """Test detection of simple array[index] access."""
        sql = "SELECT items[0] AS first_item FROM orders"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        # Find edges with nested path
        nested_edges = [e for e in graph.edges if e.nested_path]
        assert len(nested_edges) > 0

        edge = nested_edges[0]
        assert edge.nested_path == "[0]"
        assert edge.access_type == "array"
        assert edge.from_node.column_name == "items"

    def test_multiple_array_indices(self):
        """Test multiple array index accesses."""
        sql = "SELECT items[0] AS first_item, items[1] AS second_item FROM orders"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        nested_edges = [e for e in graph.edges if e.nested_path]
        assert len(nested_edges) >= 2

        paths = {e.nested_path for e in nested_edges}
        assert "[0]" in paths
        assert "[1]" in paths


# ============================================================================
# Test Group 2: Map/Dictionary Access Detection
# ============================================================================


class TestMapAccessDetection:
    """Test map/dictionary key access detection."""

    def test_simple_map_access(self):
        """Test detection of map['key'] access."""
        sql = "SELECT metadata['status'] AS order_status FROM orders"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        nested_edges = [e for e in graph.edges if e.nested_path]
        assert len(nested_edges) > 0

        edge = nested_edges[0]
        assert edge.nested_path == "['status']"
        assert edge.access_type == "map"
        assert edge.from_node.column_name == "metadata"

    def test_multiple_map_keys(self):
        """Test multiple map key accesses."""
        sql = """
        SELECT
            metadata['status'] AS order_status,
            metadata['priority'] AS priority_level
        FROM orders
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        nested_edges = [e for e in graph.edges if e.nested_path]
        assert len(nested_edges) >= 2


# ============================================================================
# Test Group 3: Struct Field Access After Array
# ============================================================================


class TestStructAfterArrayAccess:
    """Test struct field access after array subscript (items[0].field)."""

    def test_array_then_struct_access(self):
        """Test items[0].product_id pattern."""
        sql = "SELECT items[0].product_id AS first_product FROM orders"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        nested_edges = [e for e in graph.edges if e.nested_path]
        assert len(nested_edges) > 0

        edge = nested_edges[0]
        assert edge.nested_path == "[0].product_id"
        assert edge.access_type == "mixed"  # Both array and struct access
        assert edge.from_node.column_name == "items"

    def test_array_then_multiple_struct_fields(self):
        """Test multiple struct field accesses after array."""
        sql = """
        SELECT
            items[0].product_id AS first_product_id,
            items[0].quantity AS first_quantity,
            items[1].product_id AS second_product_id
        FROM orders
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        nested_edges = [e for e in graph.edges if e.nested_path]
        assert len(nested_edges) >= 3

        paths = {e.nested_path for e in nested_edges}
        assert "[0].product_id" in paths
        assert "[0].quantity" in paths
        assert "[1].product_id" in paths


# ============================================================================
# Test Group 4: Mixed Access Patterns
# ============================================================================


class TestMixedAccessPatterns:
    """Test mixed array, map, and struct access patterns."""

    def test_regular_and_nested_columns(self):
        """Test query with both regular and nested columns."""
        sql = """
        SELECT
            order_id,
            customer_name,
            items[0].product_id AS first_product
        FROM orders
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        # Regular columns should not have nested path
        regular_edges = [
            e for e in graph.edges if e.to_node.column_name in ("order_id", "customer_name")
        ]
        for edge in regular_edges:
            assert edge.nested_path is None

        # Nested column should have path
        nested_edges = [e for e in graph.edges if e.to_node.column_name == "first_product"]
        assert len(nested_edges) > 0
        assert nested_edges[0].nested_path is not None


# ============================================================================
# Test Group 5: Pipeline API
# ============================================================================


class TestNestedAccessPipeline:
    """Test nested access lineage through Pipeline API."""

    def test_nested_access_in_pipeline(self):
        """Test nested access lineage in Pipeline."""
        sql = """
        SELECT
            order_id,
            items[0].product_id AS first_product
        FROM orders
        """
        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        # Check that edges have nested path metadata
        nested_edges = [e for e in pipeline.column_graph.edges if getattr(e, "nested_path", None)]
        assert len(nested_edges) > 0
        assert nested_edges[0].access_type in ("array", "mixed")


# ============================================================================
# Test Group 6: Export Format
# ============================================================================


class TestNestedAccessExport:
    """Test that nested access metadata is included in exports."""

    def test_nested_path_in_export(self):
        """Test that nested path metadata appears in JSON export."""
        sql = "SELECT items[0].product_id AS first_product FROM orders"
        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        exporter = JSONExporter()
        export_data = exporter.export(pipeline)

        # Check that edges in export contain nested access metadata
        edges = export_data.get("edges", [])
        nested_edges = [e for e in edges if e.get("nested_path")]

        assert len(nested_edges) > 0
        assert nested_edges[0]["access_type"] in ("array", "mixed")


# ============================================================================
# Test Group 7: Dialect Support
# ============================================================================


class TestDialectSupport:
    """Test nested access support across dialects."""

    def test_bigquery_array_access(self):
        """Test BigQuery array access syntax."""
        sql = "SELECT items[0] AS first_item FROM orders"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        nested_edges = [e for e in graph.edges if e.nested_path]
        assert len(nested_edges) > 0

    def test_postgres_array_access(self):
        """Test PostgreSQL array access syntax."""
        sql = "SELECT items[1] AS first_item FROM orders"  # PostgreSQL is 1-indexed
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        nested_edges = [e for e in graph.edges if e.nested_path]
        assert len(nested_edges) > 0


# ============================================================================
# Test Group 8: Edge Cases
# ============================================================================


class TestNestedAccessEdgeCases:
    """Test edge cases for nested access support."""

    def test_dynamic_index(self):
        """Test array access with dynamic index."""
        sql = "SELECT items[idx] AS item FROM orders"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        # Should still detect nested access with wildcard
        nested_edges = [e for e in graph.edges if e.nested_path]
        # Dynamic index may use [*] notation
        if nested_edges:
            assert nested_edges[0].nested_path == "[*]"

    def test_no_nested_for_regular_columns(self):
        """Test that regular columns don't have nested access metadata."""
        sql = "SELECT id, name, email FROM users"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        for edge in graph.edges:
            assert edge.nested_path is None
            assert edge.access_type is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
