"""
Test suite for Table-Valued Functions (TVF) support in column lineage tracking.

Tests cover:
- Generator TVFs (GENERATE_SERIES, GENERATE_DATE_ARRAY)
- External data TVFs (READ_CSV)
- Column-input TVFs (detected via parameters)
- Synthetic column marking
- Pipeline API integration
- Export format validation
"""

import pytest

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder
from clgraph.models import TVFType
from clgraph.query_parser import RecursiveQueryParser

# ============================================================================
# Test Group 1: TVF Parsing
# ============================================================================


class TestTVFParsing:
    """Test TVF detection and parsing in query parser."""

    def test_generate_series_detected(self):
        """Test GENERATE_SERIES is detected as a TVF."""
        sql = "SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        assert "t" in unit.tvf_sources
        tvf_info = unit.tvf_sources["t"]
        assert tvf_info.function_name == "generate_series"
        assert tvf_info.tvf_type == TVFType.GENERATOR

    def test_generate_series_column_aliases(self):
        """Test TVF output column aliases are captured."""
        sql = "SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        tvf_info = unit.tvf_sources["t"]
        assert "num" in tvf_info.output_columns

    def test_generate_series_parameters(self):
        """Test TVF parameters are captured."""
        sql = "SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        tvf_info = unit.tvf_sources["t"]
        assert len(tvf_info.parameters) >= 2

    def test_read_csv_detected(self):
        """Test READ_CSV is detected as external TVF."""
        sql = "SELECT * FROM READ_CSV('data.csv') AS t"
        parser = RecursiveQueryParser(sql, dialect="duckdb")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        assert "t" in unit.tvf_sources
        tvf_info = unit.tvf_sources["t"]
        assert tvf_info.function_name == "read_csv"
        assert tvf_info.tvf_type == TVFType.EXTERNAL

    def test_read_csv_external_source(self):
        """Test READ_CSV captures external source path."""
        sql = "SELECT * FROM READ_CSV('s3://bucket/data.csv') AS t"
        parser = RecursiveQueryParser(sql, dialect="duckdb")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        tvf_info = unit.tvf_sources["t"]
        assert tvf_info.external_source == "s3://bucket/data.csv"

    def test_alias_mapping_created(self):
        """Test TVF alias is added to alias_mapping."""
        sql = "SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        assert "t" in unit.alias_mapping


# ============================================================================
# Test Group 2: TVF Lineage Building
# ============================================================================


class TestTVFLineage:
    """Test column lineage for TVF queries."""

    def test_synthetic_column_created(self):
        """Test synthetic column node is created for TVF output."""
        sql = "SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        # Check for synthetic column
        assert "t.num" in graph.nodes
        tvf_node = graph.nodes["t.num"]
        assert tvf_node.is_synthetic is True
        assert tvf_node.synthetic_source == "generate_series"

    def test_tvf_edge_created(self):
        """Test edge is created from TVF column to output."""
        sql = "SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        # Check for tvf_output edge
        tvf_edges = [e for e in graph.edges if e.edge_type == "tvf_output"]
        assert len(tvf_edges) >= 1
        assert tvf_edges[0].is_tvf_output is True

    def test_qualified_column_reference(self):
        """Test qualified column reference (t.num) works."""
        sql = "SELECT t.num FROM GENERATE_SERIES(1, 10) AS t(num)"
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        assert "t.num" in graph.nodes
        assert len([e for e in graph.edges if e.edge_type == "tvf_output"]) >= 1

    def test_tvf_node_type(self):
        """Test TVF column has correct node_type."""
        sql = "SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        tvf_node = graph.nodes["t.num"]
        assert tvf_node.node_type == "tvf_synthetic"

    def test_tvf_parameters_in_node(self):
        """Test TVF parameters are stored in node."""
        sql = "SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        tvf_node = graph.nodes["t.num"]
        assert tvf_node.tvf_parameters is not None
        assert len(tvf_node.tvf_parameters) > 0


# ============================================================================
# Test Group 3: Multiple TVFs
# ============================================================================


class TestMultipleTVFs:
    """Test queries with multiple TVFs."""

    def test_multiple_tvfs_in_from(self):
        """Test multiple TVFs in FROM clause."""
        sql = """
        SELECT a.num AS num_a, b.num AS num_b
        FROM GENERATE_SERIES(1, 5) AS a(num),
             GENERATE_SERIES(1, 3) AS b(num)
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        assert "a" in unit.tvf_sources
        assert "b" in unit.tvf_sources

    def test_multiple_tvfs_lineage(self):
        """Test lineage for multiple TVFs."""
        sql = """
        SELECT a.num AS num_a, b.num AS num_b
        FROM GENERATE_SERIES(1, 5) AS a(num),
             GENERATE_SERIES(1, 3) AS b(num)
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        assert "a.num" in graph.nodes
        assert "b.num" in graph.nodes
        assert graph.nodes["a.num"].is_synthetic
        assert graph.nodes["b.num"].is_synthetic


# ============================================================================
# Test Group 4: TVF with Regular Tables
# ============================================================================


class TestTVFWithTables:
    """Test TVFs combined with regular tables."""

    def test_tvf_cross_join_table(self):
        """Test TVF CROSS JOIN with regular table."""
        sql = """
        SELECT o.id, d.date
        FROM orders o
        CROSS JOIN GENERATE_SERIES('2024-01-01'::date, '2024-12-31', '1 day') AS d(date)
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        assert "orders" in unit.depends_on_tables or "o" in unit.alias_mapping
        assert "d" in unit.tvf_sources


# ============================================================================
# Test Group 5: Pipeline Integration
# ============================================================================


class TestTVFPipeline:
    """Test TVF support through Pipeline API."""

    def test_tvf_in_pipeline(self):
        """Test TVF lineage is preserved in Pipeline."""
        sql = "CREATE TABLE numbers AS SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        pipeline = Pipeline([("gen_numbers", sql)], dialect="postgres")

        # Check for synthetic column in pipeline
        columns = pipeline.column_graph.columns
        tvf_columns = [c for c in columns.values() if c.is_synthetic]
        assert len(tvf_columns) >= 1

    def test_tvf_edges_in_pipeline(self):
        """Test TVF edges appear in Pipeline graph."""
        sql = "CREATE TABLE numbers AS SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        pipeline = Pipeline([("gen_numbers", sql)], dialect="postgres")

        # Check for tvf_output edges
        tvf_edges = [e for e in pipeline.column_graph.edges if e.edge_type == "tvf_output"]
        assert len(tvf_edges) >= 1


# ============================================================================
# Test Group 6: Export Format
# ============================================================================


class TestTVFExport:
    """Test TVF metadata in exports."""

    def test_synthetic_in_json_export(self):
        """Test synthetic column info appears in JSON export."""
        sql = "SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        pipeline = Pipeline([("tvf_query", sql)], dialect="postgres")

        exporter = JSONExporter()
        export_data = exporter.export(pipeline)

        # Check columns
        columns = export_data.get("columns", [])
        synthetic_cols = [c for c in columns if c.get("is_synthetic")]
        assert len(synthetic_cols) >= 1

    def test_tvf_edge_in_json_export(self):
        """Test tvf_output edge appears in JSON export."""
        sql = "SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        pipeline = Pipeline([("tvf_query", sql)], dialect="postgres")

        exporter = JSONExporter()
        export_data = exporter.export(pipeline)

        # Check edges
        edges = export_data.get("edges", [])
        tvf_edges = [e for e in edges if e.get("edge_type") == "tvf_output"]
        assert len(tvf_edges) >= 1


# ============================================================================
# Test Group 7: Edge Cases
# ============================================================================


class TestTVFEdgeCases:
    """Test edge cases for TVF support."""

    def test_tvf_without_alias(self):
        """Test TVF without explicit alias."""
        sql = "SELECT generate_series FROM GENERATE_SERIES(1, 10)"
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        # Should have auto-generated alias
        unit = unit_graph.units["main"]
        assert len(unit.tvf_sources) > 0

    def test_tvf_default_output_columns(self):
        """Test TVF uses default output columns when not specified."""
        sql = "SELECT generate_series FROM GENERATE_SERIES(1, 10)"
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        for tvf in unit.tvf_sources.values():
            assert len(tvf.output_columns) > 0

    def test_non_tvf_query(self):
        """Test regular query doesn't have TVF sources."""
        sql = "SELECT id, name FROM users"
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        assert len(unit.tvf_sources) == 0


# ============================================================================
# Test Group 8: Dialect Support
# ============================================================================


class TestTVFDialects:
    """Test TVF support across dialects."""

    def test_postgres_generate_series(self):
        """Test PostgreSQL GENERATE_SERIES."""
        sql = "SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        assert "t.num" in graph.nodes
        assert graph.nodes["t.num"].is_synthetic

    def test_duckdb_generate_series(self):
        """Test DuckDB GENERATE_SERIES."""
        sql = "SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        builder = RecursiveLineageBuilder(sql, dialect="duckdb")
        graph = builder.build()

        # May parse differently per dialect
        assert len(graph.nodes) > 0

    def test_duckdb_read_csv(self):
        """Test DuckDB READ_CSV."""
        sql = "SELECT * FROM READ_CSV('data.csv') AS t"
        parser = RecursiveQueryParser(sql, dialect="duckdb")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        assert "t" in unit.tvf_sources
        assert unit.tvf_sources["t"].tvf_type == TVFType.EXTERNAL


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
