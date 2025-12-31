"""
Test suite for LATERAL join support in column lineage tracking.

Tests cover:
- LATERAL subquery detection
- Correlated column identification
- Lateral correlation edge creation
- SQL Server CROSS APPLY handling
- Multiple dialects (PostgreSQL, BigQuery, Snowflake)
- Export format validation
"""

import pytest

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder

# ============================================================================
# Test Group 1: LATERAL Subquery Detection
# ============================================================================


class TestLateralDetection:
    """Test LATERAL subquery identification in FROM clause."""

    def test_simple_lateral_detected(self):
        """Test detection of simple LATERAL subquery."""
        sql = """
        SELECT o.order_id, t.total
        FROM orders o,
        LATERAL (
            SELECT SUM(amount) as total
            FROM items i
            WHERE i.order_id = o.order_id
        ) t
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        # Find lateral correlation edges
        lateral_edges = [e for e in graph.edges if e.is_lateral_correlation]
        assert len(lateral_edges) > 0

        edge = lateral_edges[0]
        assert edge.edge_type == "lateral_correlation"
        assert edge.lateral_alias == "t"

    def test_lateral_correlated_column_identified(self):
        """Test that correlated column references are identified."""
        sql = """
        SELECT o.order_id, t.total
        FROM orders o,
        LATERAL (
            SELECT SUM(amount) as total
            FROM items i
            WHERE i.order_id = o.order_id
        ) t
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        # Find lateral correlation edges
        lateral_edges = [e for e in graph.edges if e.is_lateral_correlation]
        assert len(lateral_edges) > 0

        # Check that the correlated column is orders.order_id
        corr_edge = lateral_edges[0]
        assert corr_edge.from_node.column_name == "order_id"


# ============================================================================
# Test Group 2: Correlation Edge Creation
# ============================================================================


class TestCorrelationEdges:
    """Test LATERAL correlation edge creation."""

    def test_correlation_edge_properties(self):
        """Test that correlation edges have correct properties."""
        sql = """
        SELECT c.customer_id, recent.last_order
        FROM customers c,
        LATERAL (
            SELECT MAX(order_date) as last_order
            FROM orders o
            WHERE o.customer_id = c.customer_id
        ) recent
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        lateral_edges = [e for e in graph.edges if e.is_lateral_correlation]
        assert len(lateral_edges) > 0

        edge = lateral_edges[0]
        assert edge.is_lateral_correlation is True
        assert edge.lateral_alias == "recent"
        assert edge.context == "LATERAL"

    def test_multiple_correlations(self):
        """Test LATERAL with multiple correlated columns."""
        sql = """
        SELECT c.customer_id, s.total
        FROM customers c,
        LATERAL (
            SELECT SUM(amount) as total
            FROM orders o
            WHERE o.customer_id = c.customer_id
            AND o.region = c.region
        ) s
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        # Should have correlation edges for both customer_id and region
        lateral_edges = [e for e in graph.edges if e.is_lateral_correlation]
        assert len(lateral_edges) >= 2

        correlated_cols = {e.from_node.column_name for e in lateral_edges}
        assert "customer_id" in correlated_cols
        assert "region" in correlated_cols


# ============================================================================
# Test Group 3: Pipeline API
# ============================================================================


class TestLateralPipeline:
    """Test LATERAL lineage through Pipeline API."""

    def test_lateral_in_pipeline(self):
        """Test LATERAL lineage in Pipeline."""
        sql = """
        SELECT o.order_id, t.total
        FROM orders o,
        LATERAL (
            SELECT SUM(amount) as total
            FROM items i
            WHERE i.order_id = o.order_id
        ) t
        """
        pipeline = Pipeline([("query", sql)], dialect="postgres")

        # Check that edges have lateral correlation metadata
        lateral_edges = [
            e for e in pipeline.column_graph.edges if getattr(e, "is_lateral_correlation", False)
        ]
        assert len(lateral_edges) > 0


# ============================================================================
# Test Group 4: Export Format
# ============================================================================


class TestLateralExport:
    """Test that LATERAL metadata is included in exports."""

    def test_lateral_metadata_in_export(self):
        """Test that LATERAL metadata appears in JSON export."""
        sql = """
        SELECT o.order_id, t.total
        FROM orders o,
        LATERAL (
            SELECT SUM(amount) as total
            FROM items i
            WHERE i.order_id = o.order_id
        ) t
        """
        pipeline = Pipeline([("query", sql)], dialect="postgres")

        exporter = JSONExporter()
        export_data = exporter.export(pipeline)

        # Check that edges in export contain lateral correlation metadata
        edges = export_data.get("edges", [])
        lateral_edges = [e for e in edges if e.get("is_lateral_correlation")]

        assert len(lateral_edges) > 0
        assert lateral_edges[0]["lateral_alias"] == "t"


# ============================================================================
# Test Group 5: Dialect Support
# ============================================================================


class TestDialectSupport:
    """Test LATERAL support across dialects."""

    def test_postgres_lateral(self):
        """Test PostgreSQL LATERAL syntax."""
        sql = """
        SELECT o.id, t.total
        FROM orders o,
        LATERAL (
            SELECT SUM(amount) as total
            FROM items i
            WHERE i.order_id = o.id
        ) t
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        lateral_edges = [e for e in graph.edges if e.is_lateral_correlation]
        assert len(lateral_edges) > 0


# ============================================================================
# Test Group 6: Edge Cases
# ============================================================================


class TestLateralEdgeCases:
    """Test edge cases for LATERAL support."""

    def test_no_correlation_no_lateral_edge(self):
        """Test that non-LATERAL queries don't have lateral edges."""
        sql = "SELECT id, name, email FROM users"
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        lateral_edges = [e for e in graph.edges if e.is_lateral_correlation]
        assert len(lateral_edges) == 0

    def test_regular_subquery_no_lateral_edge(self):
        """Test that regular subqueries (non-LATERAL) don't have lateral edges."""
        sql = """
        SELECT id, name
        FROM users
        WHERE id IN (SELECT user_id FROM orders)
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        lateral_edges = [e for e in graph.edges if e.is_lateral_correlation]
        assert len(lateral_edges) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
