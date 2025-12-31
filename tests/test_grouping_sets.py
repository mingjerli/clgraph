"""
Test suite for GROUPING SETS/CUBE/ROLLUP support in column lineage tracking.

Tests cover:
- CUBE clause parsing and expansion
- ROLLUP clause parsing and expansion
- GROUPING SETS clause parsing
- Column lineage tracking for grouping constructs
- Pipeline API integration
- Export format validation
"""

import pytest

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder
from clgraph.query_parser import RecursiveQueryParser

# ============================================================================
# Test Group 1: CUBE Parsing
# ============================================================================


class TestCubeParsing:
    """Test CUBE clause parsing."""

    def test_simple_cube_detected(self):
        """Test detection of simple CUBE clause."""
        sql = """
        SELECT region, product, SUM(sales) as total_sales
        FROM sales_data
        GROUP BY CUBE(region, product)
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        assert main_unit.grouping_config is not None
        assert main_unit.grouping_config.get("grouping_type") == "cube"

    def test_cube_columns_extracted(self):
        """Test extraction of CUBE columns."""
        sql = """
        SELECT region, product, SUM(sales)
        FROM sales_data
        GROUP BY CUBE(region, product)
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        columns = main_unit.grouping_config.get("grouping_columns", [])
        assert "region" in columns
        assert "product" in columns

    def test_cube_expansion(self):
        """Test CUBE expansion to all combinations."""
        sql = """
        SELECT a, b, SUM(c)
        FROM t
        GROUP BY CUBE(a, b)
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        sets = main_unit.grouping_config.get("grouping_sets", [])
        # CUBE(a, b) should produce: [a,b], [a], [b], []
        assert len(sets) == 4
        assert ["a", "b"] in sets
        assert ["a"] in sets
        assert ["b"] in sets
        assert [] in sets


# ============================================================================
# Test Group 2: ROLLUP Parsing
# ============================================================================


class TestRollupParsing:
    """Test ROLLUP clause parsing."""

    def test_simple_rollup_detected(self):
        """Test detection of simple ROLLUP clause."""
        sql = """
        SELECT year, quarter, SUM(sales)
        FROM sales_data
        GROUP BY ROLLUP(year, quarter)
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        assert main_unit.grouping_config is not None
        assert main_unit.grouping_config.get("grouping_type") == "rollup"

    def test_rollup_expansion(self):
        """Test ROLLUP expansion to hierarchical combinations."""
        sql = """
        SELECT a, b, SUM(c)
        FROM t
        GROUP BY ROLLUP(a, b)
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        sets = main_unit.grouping_config.get("grouping_sets", [])
        # ROLLUP(a, b) should produce: [a,b], [a], []
        assert len(sets) == 3
        assert ["a", "b"] in sets
        assert ["a"] in sets
        assert [] in sets


# ============================================================================
# Test Group 3: GROUPING SETS Parsing
# ============================================================================


class TestGroupingSetsParsing:
    """Test GROUPING SETS clause parsing."""

    def test_simple_grouping_sets_detected(self):
        """Test detection of explicit GROUPING SETS."""
        sql = """
        SELECT region, product, SUM(sales)
        FROM sales_data
        GROUP BY GROUPING SETS ((region, product), (region), ())
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        graph = parser.parse()

        main_unit = graph.units["main"]
        assert main_unit.grouping_config is not None
        assert main_unit.grouping_config.get("grouping_type") == "grouping_sets"

    def test_grouping_sets_columns(self):
        """Test extraction of GROUPING SETS columns."""
        sql = """
        SELECT region, product, SUM(sales)
        FROM sales_data
        GROUP BY GROUPING SETS ((region, product), (region), ())
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        graph = parser.parse()

        main_unit = graph.units["main"]
        sets = main_unit.grouping_config.get("grouping_sets", [])
        assert len(sets) == 3


# ============================================================================
# Test Group 4: Lineage Building
# ============================================================================


class TestGroupingLineage:
    """Test column lineage tracking for grouping constructs."""

    def test_cube_creates_grouping_edges(self):
        """Test that CUBE creates grouping edges."""
        sql = """
        SELECT region, product, SUM(sales) as total_sales
        FROM sales_data
        GROUP BY CUBE(region, product)
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        grouping_edges = [e for e in graph.edges if e.is_grouping_column]
        assert len(grouping_edges) > 0

        for edge in grouping_edges:
            assert edge.grouping_type == "cube"

    def test_rollup_creates_grouping_edges(self):
        """Test that ROLLUP creates grouping edges."""
        sql = """
        SELECT year, quarter, SUM(sales)
        FROM sales_data
        GROUP BY ROLLUP(year, quarter)
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        grouping_edges = [e for e in graph.edges if e.is_grouping_column]
        assert len(grouping_edges) > 0

        for edge in grouping_edges:
            assert edge.grouping_type == "rollup"

    def test_grouping_edge_targets_aggregate(self):
        """Test that grouping edges target aggregate output."""
        sql = """
        SELECT region, SUM(sales) as total
        FROM sales_data
        GROUP BY CUBE(region)
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        grouping_edges = [e for e in graph.edges if e.is_grouping_column]
        assert len(grouping_edges) > 0

        # Grouping edges should target the aggregate column
        for edge in grouping_edges:
            assert "total" in edge.to_node.full_name


# ============================================================================
# Test Group 5: Pipeline API
# ============================================================================


class TestGroupingPipeline:
    """Test grouping lineage through Pipeline API."""

    def test_cube_in_pipeline(self):
        """Test CUBE lineage in Pipeline."""
        sql = """
        SELECT region, product, SUM(sales) as total_sales
        FROM sales_data
        GROUP BY CUBE(region, product)
        """
        pipeline = Pipeline([("agg_query", sql)], dialect="bigquery")

        grouping_edges = [
            e for e in pipeline.column_graph.edges if getattr(e, "is_grouping_column", False)
        ]
        assert len(grouping_edges) > 0


# ============================================================================
# Test Group 6: Export Format
# ============================================================================


class TestGroupingExport:
    """Test that grouping metadata is included in exports."""

    def test_grouping_metadata_in_export(self):
        """Test that grouping metadata appears in JSON export."""
        sql = """
        SELECT region, SUM(sales) as total
        FROM sales_data
        GROUP BY CUBE(region)
        """
        pipeline = Pipeline([("agg_query", sql)], dialect="bigquery")

        exporter = JSONExporter()
        export_data = exporter.export(pipeline)

        edges = export_data.get("edges", [])
        grouping_edges = [e for e in edges if e.get("is_grouping_column")]

        assert len(grouping_edges) > 0
        assert grouping_edges[0].get("grouping_type") == "cube"


# ============================================================================
# Test Group 7: Edge Cases
# ============================================================================


class TestGroupingEdgeCases:
    """Test edge cases for grouping support."""

    def test_non_grouping_no_grouping_edges(self):
        """Test that non-grouping queries don't have grouping edges."""
        sql = "SELECT id, name FROM users"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        grouping_edges = [e for e in graph.edges if e.is_grouping_column]
        assert len(grouping_edges) == 0

    def test_simple_group_by_no_grouping_config(self):
        """Test that simple GROUP BY doesn't create grouping config."""
        sql = """
        SELECT region, SUM(sales)
        FROM sales_data
        GROUP BY region
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        # Simple GROUP BY should not create grouping_config
        assert main_unit.grouping_config is None


# ============================================================================
# Test Group 8: Dialect Support
# ============================================================================


class TestGroupingDialects:
    """Test grouping support across dialects."""

    def test_snowflake_cube(self):
        """Test Snowflake CUBE syntax."""
        sql = """
        SELECT region, SUM(sales)
        FROM sales_data
        GROUP BY CUBE(region)
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        grouping_edges = [e for e in graph.edges if e.is_grouping_column]
        assert len(grouping_edges) > 0

    def test_postgres_grouping_sets(self):
        """Test PostgreSQL GROUPING SETS syntax."""
        sql = """
        SELECT region, SUM(sales)
        FROM sales_data
        GROUP BY GROUPING SETS ((region), ())
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        graph = parser.parse()

        main_unit = graph.units["main"]
        assert main_unit.grouping_config is not None
        assert main_unit.grouping_config.get("grouping_type") == "grouping_sets"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
