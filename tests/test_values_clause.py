"""
Test suite for VALUES clause (inline table literal) support in column lineage tracking.

Tests cover:
- VALUES clause detection and parsing
- Column alias extraction
- Type inference
- Lineage building
- Pipeline API integration
- Export format validation
"""

import pytest

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder
from clgraph.models import ValuesInfo
from clgraph.query_parser import RecursiveQueryParser

# ============================================================================
# Test Group 1: VALUES Parsing
# ============================================================================


class TestValuesParsing:
    """Test VALUES clause detection and parsing."""

    def test_simple_values_detected(self):
        """Test that a simple VALUES clause is detected."""
        sql = """
        SELECT id, name
        FROM (VALUES (1, 'Alice'), (2, 'Bob')) AS t(id, name)
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        assert "t" in unit.values_sources
        values_info = unit.values_sources["t"]
        assert values_info.alias == "t"
        assert values_info.row_count == 2

    def test_column_aliases_captured(self):
        """Test that column aliases are captured from VALUES."""
        sql = """
        SELECT * FROM (VALUES (1, 'Alice'), (2, 'Bob')) AS t(id, name)
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        values_info = unit.values_sources["t"]
        assert "id" in values_info.column_names
        assert "name" in values_info.column_names

    def test_sample_values_stored(self):
        """Test that sample values are stored."""
        sql = """
        SELECT * FROM (VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')) AS t(id, name)
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        values_info = unit.values_sources["t"]
        assert len(values_info.sample_values) >= 2
        assert values_info.sample_values[0] == [1, "Alice"]

    def test_alias_mapping_created(self):
        """Test that VALUES alias is added to alias_mapping."""
        sql = """
        SELECT id FROM (VALUES (1), (2)) AS t(id)
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        assert "t" in unit.alias_mapping


# ============================================================================
# Test Group 2: Type Inference
# ============================================================================


class TestValuesTypeInference:
    """Test type inference for VALUES columns."""

    def test_integer_type_inferred(self):
        """Test integer type is inferred."""
        sql = """
        SELECT * FROM (VALUES (1), (2), (3)) AS t(num)
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        values_info = unit_graph.units["main"].values_sources["t"]
        assert "integer" in values_info.column_types

    def test_string_type_inferred(self):
        """Test string type is inferred."""
        sql = """
        SELECT * FROM (VALUES ('a'), ('b'), ('c')) AS t(letter)
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        values_info = unit_graph.units["main"].values_sources["t"]
        assert "string" in values_info.column_types

    def test_mixed_types(self):
        """Test multiple column types."""
        sql = """
        SELECT * FROM (VALUES (1, 'Alice', 3.14)) AS t(id, name, score)
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        values_info = unit_graph.units["main"].values_sources["t"]
        assert len(values_info.column_types) == 3


# ============================================================================
# Test Group 3: VALUES Lineage Building
# ============================================================================


class TestValuesLineage:
    """Test lineage building for VALUES queries."""

    def test_literal_column_created(self):
        """Test literal column node is created for VALUES output."""
        sql = """
        SELECT id, name FROM (VALUES (1, 'Alice'), (2, 'Bob')) AS t(id, name)
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        # Check for literal column nodes
        assert "t.id" in graph.nodes
        assert "t.name" in graph.nodes
        assert graph.nodes["t.id"].is_literal is True
        assert graph.nodes["t.name"].is_literal is True

    def test_literal_edge_created(self):
        """Test edge is created from VALUES column to output."""
        sql = """
        SELECT id FROM (VALUES (1), (2)) AS t(id)
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        # Check for literal_source edge
        literal_edges = [e for e in graph.edges if e.edge_type == "literal_source"]
        assert len(literal_edges) >= 1

    def test_literal_type_in_node(self):
        """Test literal type is stored in node."""
        sql = """
        SELECT num FROM (VALUES (1), (2)) AS t(num)
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        literal_node = graph.nodes["t.num"]
        assert literal_node.literal_type == "integer"

    def test_sample_values_in_node(self):
        """Test sample values are stored in node."""
        sql = """
        SELECT name FROM (VALUES ('Alice'), ('Bob')) AS t(name)
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        literal_node = graph.nodes["t.name"]
        assert literal_node.literal_values is not None
        assert "Alice" in literal_node.literal_values


# ============================================================================
# Test Group 4: Pipeline Integration
# ============================================================================


class TestValuesPipeline:
    """Test VALUES support through Pipeline API."""

    def test_values_in_pipeline(self):
        """Test VALUES works in Pipeline."""
        sql = """
        CREATE TABLE lookup AS
        SELECT id, name FROM (VALUES (1, 'A'), (2, 'B')) AS t(id, name)
        """
        pipeline = Pipeline([("create_lookup", sql)], dialect="postgres")

        # Check that columns are in pipeline
        columns = pipeline.column_graph.columns
        literal_columns = [c for c in columns.values() if c.is_literal]
        assert len(literal_columns) >= 2

    def test_values_with_other_tables(self):
        """Test VALUES combined with regular tables."""
        queries = [
            ("staging", "CREATE TABLE staging AS SELECT id FROM raw_data"),
            (
                "enriched",
                """
                CREATE TABLE enriched AS
                SELECT s.id, l.name
                FROM staging s
                JOIN (VALUES (1, 'A'), (2, 'B')) AS l(id, name)
                ON s.id = l.id
            """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="postgres")

        # Should have both table dependencies
        assert len(pipeline.table_graph.tables) >= 2


# ============================================================================
# Test Group 5: Export Format
# ============================================================================


class TestValuesExport:
    """Test VALUES metadata in exports."""

    def test_literal_in_json_export(self):
        """Test literal column info appears in JSON export."""
        sql = """
        SELECT id, name FROM (VALUES (1, 'Alice')) AS t(id, name)
        """
        pipeline = Pipeline([("values_query", sql)], dialect="postgres")

        exporter = JSONExporter()
        export_data = exporter.export(pipeline)

        # Check columns
        columns = export_data.get("columns", [])
        literal_cols = [c for c in columns if c.get("is_literal")]
        assert len(literal_cols) >= 2


# ============================================================================
# Test Group 6: Edge Cases
# ============================================================================


class TestValuesEdgeCases:
    """Test edge cases for VALUES support."""

    def test_values_without_column_aliases(self):
        """Test VALUES without explicit column aliases."""
        sql = """
        SELECT * FROM (VALUES (1, 'Alice'), (2, 'Bob')) AS t
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        values_info = unit.values_sources.get("t")
        if values_info:
            # Should generate default column names
            assert len(values_info.column_names) > 0

    def test_values_with_null(self):
        """Test VALUES with NULL values."""
        sql = """
        SELECT * FROM (VALUES (1, NULL), (2, 'Bob')) AS t(id, name)
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        values_info = unit_graph.units["main"].values_sources["t"]
        assert values_info.row_count == 2

    def test_single_column_values(self):
        """Test VALUES with single column."""
        sql = """
        SELECT num FROM (VALUES (1), (2), (3)) AS t(num)
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        assert "t.num" in graph.nodes
        assert graph.nodes["t.num"].is_literal


# ============================================================================
# Test Group 7: Dialect Support
# ============================================================================


class TestValuesDialects:
    """Test VALUES support across dialects."""

    def test_postgres_values(self):
        """Test PostgreSQL VALUES."""
        sql = """
        SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS t(id, name)
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        assert "t" in unit_graph.units["main"].values_sources

    def test_snowflake_values(self):
        """Test Snowflake VALUES."""
        sql = """
        SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS t(id, name)
        """
        parser = RecursiveQueryParser(sql, dialect="snowflake")
        unit_graph = parser.parse()

        assert "t" in unit_graph.units["main"].values_sources

    def test_duckdb_values(self):
        """Test DuckDB VALUES."""
        sql = """
        SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS t(id, name)
        """
        parser = RecursiveQueryParser(sql, dialect="duckdb")
        unit_graph = parser.parse()

        assert "t" in unit_graph.units["main"].values_sources


# ============================================================================
# Test Group 8: VALUES in CTEs
# ============================================================================


class TestValuesCTE:
    """Test VALUES in CTE context."""

    def test_values_in_cte(self):
        """Test VALUES used within a CTE."""
        sql = """
        WITH lookup AS (
            SELECT * FROM (VALUES (1, 'A'), (2, 'B')) AS t(code, label)
        )
        SELECT m.id, l.label
        FROM main_table m
        JOIN lookup l ON m.code = l.code
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        # The CTE should contain the VALUES
        cte_unit = unit_graph.get_unit_by_name("lookup")
        assert cte_unit is not None


# ============================================================================
# Test Group 9: ValuesInfo Model
# ============================================================================


class TestValuesInfoModel:
    """Test ValuesInfo dataclass."""

    def test_values_info_creation(self):
        """Test creating ValuesInfo instance."""
        info = ValuesInfo(
            alias="t",
            column_names=["id", "name"],
            row_count=3,
            column_types=["integer", "string"],
            sample_values=[[1, "Alice"], [2, "Bob"]],
        )

        assert info.alias == "t"
        assert len(info.column_names) == 2
        assert info.row_count == 3

    def test_values_info_repr(self):
        """Test ValuesInfo string representation."""
        info = ValuesInfo(
            alias="lookup",
            column_names=["id", "name"],
            row_count=5,
        )

        repr_str = repr(info)
        assert "lookup" in repr_str
        assert "id" in repr_str
        assert "5 rows" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
