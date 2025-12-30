"""
Test suite for window frame analysis support in column lineage tracking.

Tests cover:
- Window function parsing (PARTITION BY, ORDER BY, frame specs)
- Named window definitions and references
- Window lineage edge creation
- Pipeline API integration
- Export format validation
"""

import pytest

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder
from clgraph.query_parser import RecursiveQueryParser

# ============================================================================
# Test Group 1: Window Function Parsing
# ============================================================================


class TestWindowFunctionParsing:
    """Test window function parsing."""

    def test_partition_by_detected(self):
        """Test detection of PARTITION BY columns."""
        sql = """
        SELECT customer_id, SUM(amount) OVER (PARTITION BY customer_id)
        FROM orders
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        assert main_unit.window_info is not None
        windows = main_unit.window_info.get("windows", [])
        assert len(windows) == 1
        assert "customer_id" in windows[0]["partition_by"]

    def test_order_by_detected(self):
        """Test detection of ORDER BY columns with direction."""
        sql = """
        SELECT ROW_NUMBER() OVER (ORDER BY created_at DESC) AS rn
        FROM events
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        windows = main_unit.window_info["windows"]
        assert len(windows) == 1
        order_by = windows[0]["order_by"]
        assert len(order_by) == 1
        assert order_by[0]["column"] == "created_at"
        assert order_by[0]["direction"] == "desc"

    def test_frame_spec_rows(self):
        """Test ROWS frame specification parsing."""
        sql = """
        SELECT SUM(amount) OVER (
            ORDER BY date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_sum
        FROM sales
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        windows = main_unit.window_info["windows"]
        assert windows[0]["frame_type"] == "rows"
        assert windows[0]["frame_start"] == "6 preceding"
        assert windows[0]["frame_end"] == "current row"

    def test_frame_spec_range(self):
        """Test RANGE frame specification parsing."""
        sql = """
        SELECT AVG(price) OVER (
            ORDER BY date
            RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_avg
        FROM prices
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        windows = main_unit.window_info["windows"]
        assert windows[0]["frame_type"] == "range"
        assert "unbounded" in windows[0]["frame_start"].lower()
        assert "current row" in windows[0]["frame_end"].lower()

    def test_function_arguments(self):
        """Test extraction of function arguments."""
        sql = """
        SELECT SUM(amount) OVER (PARTITION BY customer_id) AS total
        FROM orders
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        windows = main_unit.window_info["windows"]
        assert "amount" in windows[0]["arguments"]


# ============================================================================
# Test Group 2: Named Windows
# ============================================================================


class TestNamedWindows:
    """Test named window definitions and references."""

    def test_named_window_definition(self):
        """Test parsing of named window definitions."""
        sql = """
        SELECT
            SUM(a) OVER w,
            AVG(b) OVER w
        FROM t
        WINDOW w AS (PARTITION BY c ORDER BY d)
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        assert "w" in main_unit.window_definitions
        assert "c" in main_unit.window_definitions["w"]["partition_by"]

    def test_named_window_reference(self):
        """Test that named window references are resolved."""
        sql = """
        SELECT
            SUM(a) OVER w AS sum_a,
            AVG(b) OVER w AS avg_b
        FROM t
        WINDOW w AS (PARTITION BY c ORDER BY d)
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        windows = main_unit.window_info["windows"]

        # Both windows should have the same partition_by from named window
        for w in windows:
            assert w["window_name"] == "w"
            assert "c" in w["partition_by"]


# ============================================================================
# Test Group 3: Lineage Building
# ============================================================================


class TestWindowLineage:
    """Test window function lineage edge creation."""

    def test_aggregate_edges_created(self):
        """Test that window aggregate edges are created."""
        sql = """
        SELECT SUM(amount) OVER (PARTITION BY customer_id) AS total
        FROM orders
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.edge_type == "window_aggregate"]
        assert len(agg_edges) > 0
        assert any("amount" in e.from_node.full_name for e in agg_edges)

    def test_partition_edges_created(self):
        """Test that window partition edges are created."""
        sql = """
        SELECT SUM(amount) OVER (PARTITION BY customer_id) AS total
        FROM orders
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        partition_edges = [e for e in graph.edges if e.edge_type == "window_partition"]
        assert len(partition_edges) > 0
        assert any("customer_id" in e.from_node.full_name for e in partition_edges)

    def test_order_edges_created(self):
        """Test that window order edges are created."""
        sql = """
        SELECT ROW_NUMBER() OVER (ORDER BY created_at DESC) AS rn
        FROM events
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        order_edges = [e for e in graph.edges if e.edge_type == "window_order"]
        assert len(order_edges) > 0
        assert any("created_at" in e.from_node.full_name for e in order_edges)

    def test_order_direction_captured(self):
        """Test that ORDER BY direction is captured in edge metadata."""
        sql = """
        SELECT ROW_NUMBER() OVER (ORDER BY created_at DESC) AS rn
        FROM events
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        order_edges = [e for e in graph.edges if e.edge_type == "window_order"]
        assert len(order_edges) > 0
        assert order_edges[0].window_order_direction == "desc"

    def test_frame_spec_in_edge(self):
        """Test that frame specification is captured in edges."""
        sql = """
        SELECT SUM(amount) OVER (
            ORDER BY date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_sum
        FROM sales
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        window_edges = [e for e in graph.edges if e.is_window_function]
        assert len(window_edges) > 0
        agg_edge = [e for e in window_edges if e.edge_type == "window_aggregate"][0]
        assert agg_edge.window_frame_type == "rows"
        assert agg_edge.window_frame_start == "6 preceding"

    def test_window_function_name_captured(self):
        """Test that window function name is captured in edges."""
        sql = """
        SELECT SUM(amount) OVER (PARTITION BY customer_id) AS total
        FROM orders
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        window_edges = [e for e in graph.edges if e.is_window_function]
        for edge in window_edges:
            assert edge.window_function == "SUM"


# ============================================================================
# Test Group 4: Pipeline API
# ============================================================================


class TestWindowPipeline:
    """Test window function lineage through Pipeline API."""

    def test_window_in_pipeline(self):
        """Test that window function edges are preserved in Pipeline."""
        sql = """
        SELECT customer_id, SUM(amount) OVER (PARTITION BY customer_id) AS total
        FROM orders
        """
        pipeline = Pipeline([("window_query", sql)], dialect="bigquery")

        window_edges = [
            e for e in pipeline.column_graph.edges if getattr(e, "is_window_function", False)
        ]
        assert len(window_edges) > 0

    def test_multiple_windows_in_pipeline(self):
        """Test multiple window functions in same query."""
        sql = """
        SELECT
            customer_id,
            SUM(amount) OVER (PARTITION BY customer_id) AS total,
            AVG(amount) OVER (PARTITION BY customer_id) AS average
        FROM orders
        """
        pipeline = Pipeline([("multi_window", sql)], dialect="bigquery")

        window_edges = [
            e for e in pipeline.column_graph.edges if getattr(e, "is_window_function", False)
        ]
        # Should have edges for both window functions
        assert len(window_edges) >= 4  # 2 aggregate + 2 partition edges


# ============================================================================
# Test Group 5: Export Format
# ============================================================================


class TestWindowExport:
    """Test that window function metadata is included in exports."""

    def test_window_metadata_in_export(self):
        """Test that window metadata appears in JSON export."""
        sql = """
        SELECT SUM(amount) OVER (
            PARTITION BY customer_id
            ORDER BY order_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_sum
        FROM orders
        """
        pipeline = Pipeline([("window_query", sql)], dialect="bigquery")

        exporter = JSONExporter()
        export_data = exporter.export(pipeline)

        edges = export_data.get("edges", [])
        window_edges = [e for e in edges if e.get("is_window_function")]

        assert len(window_edges) > 0

        # Check aggregate edge has full metadata
        agg_edge = [e for e in window_edges if e.get("window_role") == "aggregate"][0]
        assert agg_edge.get("window_function") == "SUM"
        assert agg_edge.get("window_frame_type") == "rows"
        assert agg_edge.get("window_frame_start") == "6 preceding"

    def test_order_direction_in_export(self):
        """Test that ORDER BY direction is in export."""
        sql = """
        SELECT ROW_NUMBER() OVER (ORDER BY created_at DESC) AS rn
        FROM events
        """
        pipeline = Pipeline([("window_query", sql)], dialect="bigquery")

        exporter = JSONExporter()
        export_data = exporter.export(pipeline)

        edges = export_data.get("edges", [])
        order_edges = [e for e in edges if e.get("window_role") == "order"]

        assert len(order_edges) > 0
        assert order_edges[0].get("window_order_direction") == "desc"


# ============================================================================
# Test Group 6: Ranking Functions
# ============================================================================


class TestRankingFunctions:
    """Test ranking window functions."""

    def test_row_number(self):
        """Test ROW_NUMBER() function."""
        sql = """
        SELECT ROW_NUMBER() OVER (PARTITION BY category ORDER BY created_at) AS rn
        FROM products
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        window_edges = [e for e in graph.edges if e.is_window_function]
        assert len(window_edges) > 0
        assert any(e.window_function == "ROW_NUMBER" for e in window_edges)

    def test_rank(self):
        """Test RANK() function."""
        sql = """
        SELECT RANK() OVER (PARTITION BY category ORDER BY score DESC) AS rank
        FROM products
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        window_edges = [e for e in graph.edges if e.is_window_function]
        assert any(e.window_function == "RANK" for e in window_edges)

    def test_dense_rank(self):
        """Test DENSE_RANK() function."""
        sql = """
        SELECT DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
        FROM employees
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        window_edges = [e for e in graph.edges if e.is_window_function]
        assert any(e.window_function == "DENSE_RANK" for e in window_edges)


# ============================================================================
# Test Group 7: Offset Functions
# ============================================================================


class TestOffsetFunctions:
    """Test LAG/LEAD offset functions."""

    def test_lag_function(self):
        """Test LAG() function."""
        sql = """
        SELECT LAG(amount, 1) OVER (PARTITION BY customer ORDER BY order_date) AS prev_amount
        FROM orders
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        window_edges = [e for e in graph.edges if e.is_window_function]
        assert len(window_edges) > 0
        assert any(e.window_function == "LAG" for e in window_edges)

    def test_lead_function(self):
        """Test LEAD() function."""
        sql = """
        SELECT LEAD(amount, 1) OVER (PARTITION BY customer ORDER BY order_date) AS next_amount
        FROM orders
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        window_edges = [e for e in graph.edges if e.is_window_function]
        assert any(e.window_function == "LEAD" for e in window_edges)


# ============================================================================
# Test Group 8: Edge Cases
# ============================================================================


class TestWindowEdgeCases:
    """Test edge cases for window function support."""

    def test_empty_over_clause(self):
        """Test window with empty OVER() (whole-table window)."""
        sql = """
        SELECT SUM(amount) OVER () AS total
        FROM orders
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        # Should still create aggregate edge
        agg_edges = [e for e in graph.edges if e.edge_type == "window_aggregate"]
        assert len(agg_edges) > 0

    def test_multiple_windows_same_select(self):
        """Test multiple window functions in same SELECT."""
        sql = """
        SELECT
            SUM(amount) OVER (PARTITION BY customer_id) AS total,
            AVG(amount) OVER (PARTITION BY customer_id) AS average,
            COUNT(*) OVER (PARTITION BY customer_id) AS cnt
        FROM orders
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        windows = main_unit.window_info["windows"]
        assert len(windows) == 3

    def test_non_window_query_no_window_edges(self):
        """Test that non-window queries don't have window edges."""
        sql = "SELECT id, name FROM users"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        window_edges = [e for e in graph.edges if e.is_window_function]
        assert len(window_edges) == 0


# ============================================================================
# Test Group 9: Dialect Support
# ============================================================================


class TestWindowDialects:
    """Test window function support across dialects."""

    def test_snowflake_window(self):
        """Test Snowflake window function syntax."""
        sql = """
        SELECT SUM(amount) OVER (PARTITION BY region ORDER BY date) AS total
        FROM sales
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        window_edges = [e for e in graph.edges if e.is_window_function]
        assert len(window_edges) > 0

    def test_postgres_window(self):
        """Test PostgreSQL window function syntax."""
        sql = """
        SELECT SUM(amount) OVER (PARTITION BY region ORDER BY date) AS total
        FROM sales
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        window_edges = [e for e in graph.edges if e.is_window_function]
        assert len(window_edges) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
