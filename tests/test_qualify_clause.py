"""
Test suite for QUALIFY clause support in column lineage tracking.

Tests cover:
- QUALIFY clause detection and parsing
- PARTITION BY column extraction
- ORDER BY column extraction
- Window function identification
- Pipeline API integration
- Export format validation
"""

import pytest

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder
from clgraph.query_parser import RecursiveQueryParser

# ============================================================================
# Test Group 1: QUALIFY Clause Parsing
# ============================================================================


class TestQualifyParsing:
    """Test QUALIFY clause parsing."""

    def test_simple_qualify_detected(self):
        """Test detection of simple QUALIFY clause."""
        sql = """
        SELECT customer_id, order_date, amount
        FROM orders
        QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        assert main_unit.qualify_info is not None
        assert "ROW_NUMBER" in str(main_unit.qualify_info.get("condition", ""))

    def test_qualify_partition_columns(self):
        """Test extraction of PARTITION BY columns."""
        sql = """
        SELECT customer_id, order_date
        FROM orders
        QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        partition_cols = main_unit.qualify_info.get("partition_columns", [])
        assert "customer_id" in partition_cols

    def test_qualify_order_columns(self):
        """Test extraction of ORDER BY columns."""
        sql = """
        SELECT customer_id, order_date
        FROM orders
        QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        order_cols = main_unit.qualify_info.get("order_columns", [])
        assert "order_date" in order_cols

    def test_qualify_window_function(self):
        """Test extraction of window function name."""
        sql = """
        SELECT customer_id, order_date
        FROM orders
        QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        window_funcs = main_unit.qualify_info.get("window_functions", [])
        assert "ROW_NUMBER" in window_funcs


# ============================================================================
# Test Group 2: Multiple Window Functions
# ============================================================================


class TestMultipleWindowFunctions:
    """Test QUALIFY with multiple window functions."""

    def test_multiple_partition_columns(self):
        """Test QUALIFY with multiple PARTITION BY columns."""
        sql = """
        SELECT *
        FROM products
        QUALIFY ROW_NUMBER() OVER (PARTITION BY category, brand ORDER BY price) = 1
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        partition_cols = main_unit.qualify_info.get("partition_columns", [])
        assert "category" in partition_cols
        assert "brand" in partition_cols

    def test_rank_function(self):
        """Test QUALIFY with RANK function."""
        sql = """
        SELECT *
        FROM employees
        QUALIFY RANK() OVER (PARTITION BY department ORDER BY salary DESC) <= 3
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        window_funcs = main_unit.qualify_info.get("window_functions", [])
        assert "RANK" in window_funcs


# ============================================================================
# Test Group 3: QUALIFY Lineage Building
# ============================================================================


class TestQualifyLineage:
    """Test column lineage tracking for QUALIFY clause."""

    def test_qualify_creates_partition_edges(self):
        """Test that QUALIFY creates edges for PARTITION BY columns."""
        sql = """
        SELECT customer_id, order_date, amount
        FROM orders
        QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        qualify_edges = [e for e in graph.edges if e.is_qualify_column]
        partition_edges = [e for e in qualify_edges if e.qualify_context == "partition"]
        assert len(partition_edges) > 0

        # Check edge properties
        edge = partition_edges[0]
        assert "customer_id" in edge.from_node.full_name

    def test_qualify_creates_order_edges(self):
        """Test that QUALIFY creates edges for ORDER BY columns."""
        sql = """
        SELECT customer_id, order_date, amount
        FROM orders
        QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        qualify_edges = [e for e in graph.edges if e.is_qualify_column]
        order_edges = [e for e in qualify_edges if e.qualify_context == "order"]
        assert len(order_edges) > 0

        # Check edge properties
        edge = order_edges[0]
        assert "order_date" in edge.from_node.full_name

    def test_qualify_edge_has_function_name(self):
        """Test that QUALIFY edges have window function name."""
        sql = """
        SELECT customer_id, order_date
        FROM orders
        QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        qualify_edges = [e for e in graph.edges if e.is_qualify_column]
        assert len(qualify_edges) > 0

        for edge in qualify_edges:
            assert edge.qualify_function == "ROW_NUMBER"


# ============================================================================
# Test Group 4: Pipeline API
# ============================================================================


class TestQualifyPipeline:
    """Test QUALIFY lineage through Pipeline API."""

    def test_qualify_in_pipeline(self):
        """Test QUALIFY lineage in Pipeline."""
        sql = """
        SELECT customer_id, order_date
        FROM orders
        QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
        """
        pipeline = Pipeline([("dedupe_query", sql)], dialect="bigquery")

        # Check that edges have qualify metadata
        qualify_edges = [
            e for e in pipeline.column_graph.edges if getattr(e, "is_qualify_column", False)
        ]
        assert len(qualify_edges) > 0


# ============================================================================
# Test Group 5: Export Format
# ============================================================================


class TestQualifyExport:
    """Test that QUALIFY metadata is included in exports."""

    def test_qualify_metadata_in_export(self):
        """Test that QUALIFY metadata appears in JSON export."""
        sql = """
        SELECT customer_id, order_date
        FROM orders
        QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
        """
        pipeline = Pipeline([("dedupe_query", sql)], dialect="bigquery")

        exporter = JSONExporter()
        export_data = exporter.export(pipeline)

        # Check that edges in export contain qualify metadata
        edges = export_data.get("edges", [])
        qualify_edges = [e for e in edges if e.get("is_qualify_column")]

        assert len(qualify_edges) > 0
        assert qualify_edges[0].get("qualify_context") in ["partition", "order"]
        assert qualify_edges[0].get("qualify_function") == "ROW_NUMBER"


# ============================================================================
# Test Group 6: Edge Cases
# ============================================================================


class TestQualifyEdgeCases:
    """Test edge cases for QUALIFY support."""

    def test_non_qualify_no_qualify_edges(self):
        """Test that non-QUALIFY queries don't have qualify edges."""
        sql = "SELECT id, name, email FROM users"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        qualify_edges = [e for e in graph.edges if e.is_qualify_column]
        assert len(qualify_edges) == 0

    def test_qualify_with_dense_rank(self):
        """Test QUALIFY with DENSE_RANK function."""
        sql = """
        SELECT *
        FROM sales
        QUALIFY DENSE_RANK() OVER (PARTITION BY region ORDER BY amount DESC) = 1
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        window_funcs = main_unit.qualify_info.get("window_functions", [])
        assert "DENSE_RANK" in window_funcs


# ============================================================================
# Test Group 7: Dialect Support
# ============================================================================


class TestQualifyDialects:
    """Test QUALIFY support across dialects."""

    def test_snowflake_qualify(self):
        """Test Snowflake QUALIFY syntax."""
        sql = """
        SELECT customer_id, order_date
        FROM orders
        QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        qualify_edges = [e for e in graph.edges if e.is_qualify_column]
        assert len(qualify_edges) > 0

    def test_bigquery_qualify(self):
        """Test BigQuery QUALIFY syntax."""
        sql = """
        SELECT customer_id, order_date
        FROM orders
        QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        qualify_edges = [e for e in graph.edges if e.is_qualify_column]
        assert len(qualify_edges) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
