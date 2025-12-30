"""
Test suite for JSON function support in column lineage tracking.

Tests cover:
- JSON function detection
- JSON path extraction and normalization
- JSON lineage edge creation with metadata
- Multiple dialects (BigQuery, PostgreSQL, Snowflake)
"""

import pytest

from clgraph import Pipeline, RecursiveLineageBuilder

# ============================================================================
# Test Group 1: JSON Path Normalization
# ============================================================================


class TestJSONPathNormalization:
    """Test JSON path normalization to consistent format."""

    def test_dollar_dot_notation_unchanged(self):
        """Test that $.field.nested format is preserved."""
        from clgraph.lineage_builder import _normalize_json_path

        assert _normalize_json_path("$.address.city") == "$.address.city"
        assert _normalize_json_path("$.user.profile.name") == "$.user.profile.name"

    def test_bracket_notation_converted(self):
        """Test that bracket notation is converted to dot notation."""
        from clgraph.lineage_builder import _normalize_json_path

        assert _normalize_json_path('$["address"]["city"]') == "$.address.city"
        assert _normalize_json_path("$['user']['name']") == "$.user.name"

    def test_snowflake_notation_converted(self):
        """Test that Snowflake format (without $) is normalized."""
        from clgraph.lineage_builder import _normalize_json_path

        assert _normalize_json_path("address.city") == "$.address.city"
        assert _normalize_json_path("user.profile.name") == "$.user.profile.name"

    def test_postgresql_array_format_converted(self):
        """Test that PostgreSQL {field,nested} format is converted."""
        from clgraph.lineage_builder import _normalize_json_path

        assert _normalize_json_path("{address,city}") == "$.address.city"
        assert _normalize_json_path("{user,profile,name}") == "$.user.profile.name"

    def test_quoted_paths_handled(self):
        """Test that surrounding quotes are stripped."""
        from clgraph.lineage_builder import _normalize_json_path

        assert _normalize_json_path("'$.address.city'") == "$.address.city"
        assert _normalize_json_path('"address.city"') == "$.address.city"

    def test_array_index_notation(self):
        """Test that array indices are normalized."""
        from clgraph.lineage_builder import _normalize_json_path

        assert _normalize_json_path("$.items[0].name") == "$.items.0.name"
        assert _normalize_json_path("items[0]") == "$.items.0"


# ============================================================================
# Test Group 2: JSON Function Detection
# ============================================================================


class TestJSONFunctionDetection:
    """Test JSON function identification across dialects."""

    def test_bigquery_json_extract_detected(self):
        """Test detection of BigQuery JSON_EXTRACT function."""
        sql = "SELECT JSON_EXTRACT(data, '$.user.name') AS user_name FROM users"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        # Find edge to user_name output
        user_name_edges = [e for e in graph.edges if e.to_node.column_name == "user_name"]
        assert len(user_name_edges) > 0

        edge = user_name_edges[0]
        assert edge.json_function == "JSON_EXTRACT"
        assert edge.json_path == "$.user.name"

    def test_bigquery_json_value_detected(self):
        """Test detection of BigQuery JSON_VALUE function.

        Note: sqlglot normalizes JSON_VALUE to JSON_EXTRACT_SCALAR expression type.
        """
        sql = "SELECT JSON_VALUE(data, '$.email') AS email FROM users"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        email_edges = [e for e in graph.edges if e.to_node.column_name == "email"]
        assert len(email_edges) > 0

        edge = email_edges[0]
        # sqlglot normalizes JSON_VALUE to JSON_EXTRACT_SCALAR
        assert edge.json_function in ("JSON_VALUE", "JSON_EXTRACT_SCALAR")
        assert edge.json_path == "$.email"

    def test_bigquery_json_query_detected(self):
        """Test detection of BigQuery JSON_QUERY function.

        Note: sqlglot normalizes JSON_QUERY to JSON_EXTRACT expression type.
        """
        sql = "SELECT JSON_QUERY(data, '$.address') AS address FROM users"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        address_edges = [e for e in graph.edges if e.to_node.column_name == "address"]
        assert len(address_edges) > 0

        edge = address_edges[0]
        # sqlglot normalizes JSON_QUERY to JSON_EXTRACT
        assert edge.json_function in ("JSON_QUERY", "JSON_EXTRACT")
        assert edge.json_path == "$.address"


# ============================================================================
# Test Group 3: JSON Lineage with Pipeline API
# ============================================================================


class TestJSONLineagePipeline:
    """Test JSON function lineage through Pipeline API."""

    def test_simple_json_extract_lineage(self):
        """Test end-to-end JSON extraction lineage."""
        sql = """
        SELECT
            id,
            JSON_EXTRACT(user_data, '$.address.city') AS city,
            JSON_EXTRACT(user_data, '$.address.zip') AS zip_code
        FROM users
        """
        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        # Check that edges have JSON metadata
        city_edges = [
            e
            for e in pipeline.column_graph.edges
            if e.to_node.column_name == "city" and e.json_path is not None
        ]
        assert len(city_edges) > 0
        assert city_edges[0].json_path == "$.address.city"
        assert city_edges[0].json_function == "JSON_EXTRACT"

        zip_edges = [
            e
            for e in pipeline.column_graph.edges
            if e.to_node.column_name == "zip_code" and e.json_path is not None
        ]
        assert len(zip_edges) > 0
        assert zip_edges[0].json_path == "$.address.zip"

    def test_json_in_cte_lineage(self):
        """Test JSON extraction in CTE propagates metadata."""
        sql = """
        WITH extracted AS (
            SELECT
                id,
                JSON_EXTRACT(profile, '$.name') AS name
            FROM raw_users
        )
        SELECT id, name FROM extracted
        """
        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        # The JSON metadata should be on the edge from raw_users.profile to extracted.name
        json_edges = [e for e in pipeline.column_graph.edges if e.json_path is not None]
        assert len(json_edges) > 0

        # Find the edge that has the JSON extraction
        json_extract_edge = next((e for e in json_edges if e.json_path == "$.name"), None)
        assert json_extract_edge is not None
        assert json_extract_edge.json_function == "JSON_EXTRACT"

    def test_nested_json_extraction(self):
        """Test deeply nested JSON path extraction."""
        sql = """
        SELECT
            JSON_EXTRACT(data, '$.user.profile.preferences.theme') AS theme
        FROM settings
        """
        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        theme_edges = [
            e
            for e in pipeline.column_graph.edges
            if e.to_node.column_name == "theme" and e.json_path is not None
        ]
        assert len(theme_edges) > 0
        assert theme_edges[0].json_path == "$.user.profile.preferences.theme"

    def test_json_with_expression(self):
        """Test JSON extraction combined with other expressions."""
        sql = """
        SELECT
            UPPER(JSON_EXTRACT(data, '$.name')) AS upper_name
        FROM users
        """
        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        # The JSON metadata should still be captured
        name_edges = [
            e
            for e in pipeline.column_graph.edges
            if e.to_node.column_name == "upper_name" and e.json_path is not None
        ]
        assert len(name_edges) > 0
        assert name_edges[0].json_path == "$.name"


# ============================================================================
# Test Group 4: Multi-Query Pipeline with JSON
# ============================================================================


class TestMultiQueryJSONLineage:
    """Test JSON lineage across multiple queries in a pipeline."""

    def test_json_extraction_chain(self):
        """Test JSON extraction flowing through multiple queries."""
        queries = [
            (
                "stage",
                """
                CREATE TABLE stage AS
                SELECT
                    id,
                    JSON_EXTRACT(raw_data, '$.user') AS user_json
                FROM raw_events
                """,
            ),
            (
                "final",
                """
                CREATE TABLE final AS
                SELECT
                    id,
                    JSON_EXTRACT(user_json, '$.email') AS email
                FROM stage
                """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # First query: raw_events.raw_data[$.user] -> stage.user_json
        stage_edges = [
            e
            for e in pipeline.column_graph.edges
            if e.to_node.table_name == "stage"
            and e.to_node.column_name == "user_json"
            and e.json_path is not None
        ]
        assert len(stage_edges) > 0
        assert stage_edges[0].json_path == "$.user"

        # Second query: stage.user_json[$.email] -> final.email
        final_edges = [
            e
            for e in pipeline.column_graph.edges
            if e.to_node.table_name == "final"
            and e.to_node.column_name == "email"
            and e.json_path is not None
        ]
        assert len(final_edges) > 0
        assert final_edges[0].json_path == "$.email"


# ============================================================================
# Test Group 5: Edge Cases
# ============================================================================


class TestJSONEdgeCases:
    """Test edge cases for JSON function support."""

    def test_non_json_columns_no_metadata(self):
        """Test that regular columns don't have JSON metadata."""
        sql = "SELECT id, name, email FROM users"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        # All edges should have None for json_path and json_function
        for edge in graph.edges:
            assert edge.json_path is None
            assert edge.json_function is None

    def test_mixed_json_and_regular_columns(self):
        """Test query with both JSON and regular columns."""
        sql = """
        SELECT
            id,
            name,
            JSON_EXTRACT(data, '$.city') AS city
        FROM users
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        # id and name edges should not have JSON metadata
        id_edges = [e for e in graph.edges if e.to_node.column_name == "id"]
        name_edges = [e for e in graph.edges if e.to_node.column_name == "name"]
        city_edges = [e for e in graph.edges if e.to_node.column_name == "city"]

        for edge in id_edges + name_edges:
            assert edge.json_path is None

        # city edge should have JSON metadata
        assert len(city_edges) > 0
        assert city_edges[0].json_path == "$.city"
        assert city_edges[0].json_function == "JSON_EXTRACT"

    def test_json_in_case_expression(self):
        """Test JSON extraction inside CASE expression."""
        sql = """
        SELECT
            CASE
                WHEN type = 'A' THEN JSON_EXTRACT(data, '$.a_value')
                ELSE JSON_EXTRACT(data, '$.b_value')
            END AS value
        FROM t
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        # Should detect JSON extraction (at least one path)
        value_edges = [
            e for e in graph.edges if e.to_node.column_name == "value" and e.json_path is not None
        ]
        # With current implementation, we capture JSON metadata for columns inside JSON functions
        # The CASE expression will have edges from data column with JSON paths
        assert len(value_edges) >= 1

    def test_json_with_aggregate(self):
        """Test JSON extraction with aggregation."""
        sql = """
        SELECT
            customer_id,
            COUNT(JSON_EXTRACT(data, '$.item')) AS item_count
        FROM orders
        GROUP BY customer_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        # The aggregate should still capture JSON metadata
        count_edges = [
            e
            for e in graph.edges
            if e.to_node.column_name == "item_count" and e.json_path is not None
        ]
        assert len(count_edges) > 0
        assert count_edges[0].json_path == "$.item"


# ============================================================================
# Test Group 6: JSON Export Format
# ============================================================================


class TestJSONExport:
    """Test that JSON metadata is included in exports."""

    def test_json_metadata_in_export(self):
        """Test that JSON metadata appears in JSON export."""
        from clgraph import JSONExporter

        sql = "SELECT JSON_EXTRACT(data, '$.user.name') AS user_name FROM users"
        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        exporter = JSONExporter()
        export_data = exporter.export(pipeline)

        # Check that edges in export contain JSON metadata
        edges = export_data.get("edges", [])
        json_edges = [e for e in edges if e.get("json_path") is not None]

        assert len(json_edges) > 0
        assert json_edges[0]["json_path"] == "$.user.name"
        assert json_edges[0]["json_function"] == "JSON_EXTRACT"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
