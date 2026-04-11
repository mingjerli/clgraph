"""
Tests for pipeline_factory module (extracted factory functions).

Tests both:
1. Direct imports from clgraph.pipeline_factory
2. Backward compatibility via Pipeline class methods
"""

import json
import tempfile
from pathlib import Path

import pytest

from clgraph.pipeline import Pipeline
from clgraph.pipeline_factory import (
    create_empty,
    create_from_dict,
    create_from_json,
    create_from_json_file,
    create_from_sql_files,
    create_from_sql_list,
    create_from_sql_string,
    create_from_tuples,
    generate_query_id,
)


class TestCreateFromTuples:
    """Tests for create_from_tuples factory function."""

    def test_basic(self):
        queries = [
            ("q1", "CREATE TABLE t1 AS SELECT a FROM raw"),
            ("q2", "CREATE TABLE t2 AS SELECT a FROM t1"),
        ]
        pipeline = create_from_tuples(queries)

        assert len(pipeline.table_graph.queries) == 2
        assert "q1" in pipeline.table_graph.queries
        assert "q2" in pipeline.table_graph.queries

    def test_with_dialect(self):
        queries = [("q1", "CREATE TABLE t1 AS SELECT a FROM raw")]
        pipeline = create_from_tuples(queries, dialect="snowflake")

        assert pipeline.dialect == "snowflake"

    def test_with_template_context(self):
        queries = [
            ("q1", "CREATE TABLE {{env}}_t1 AS SELECT a FROM raw"),
        ]
        pipeline = create_from_tuples(queries, template_context={"env": "prod"})

        assert pipeline is not None
        assert len(pipeline.table_graph.queries) == 1

    def test_returns_pipeline_instance(self):
        queries = [("q1", "CREATE TABLE t1 AS SELECT a FROM raw")]
        pipeline = create_from_tuples(queries)

        assert isinstance(pipeline, Pipeline)


class TestCreateFromDict:
    """Tests for create_from_dict factory function."""

    def test_basic(self):
        queries = {
            "staging": "CREATE TABLE staging AS SELECT a FROM raw",
            "final": "CREATE TABLE final AS SELECT a FROM staging",
        }
        pipeline = create_from_dict(queries)

        assert len(pipeline.table_graph.queries) == 2
        assert "staging" in pipeline.table_graph.queries
        assert "final" in pipeline.table_graph.queries

    def test_preserves_query_ids(self):
        queries = {
            "my_custom_id": "CREATE TABLE t1 AS SELECT a FROM raw",
            "another_id": "CREATE TABLE t2 AS SELECT a FROM t1",
        }
        pipeline = create_from_dict(queries)

        assert "my_custom_id" in pipeline.table_graph.queries
        assert "another_id" in pipeline.table_graph.queries

    def test_with_dialect(self):
        queries = {"q1": "CREATE TABLE t1 AS SELECT a FROM raw"}
        pipeline = create_from_dict(queries, dialect="snowflake")

        assert pipeline.dialect == "snowflake"

    def test_returns_pipeline_instance(self):
        queries = {"q1": "CREATE TABLE t1 AS SELECT a FROM raw"}
        pipeline = create_from_dict(queries)

        assert isinstance(pipeline, Pipeline)


class TestCreateFromSqlList:
    """Tests for create_from_sql_list factory function."""

    def test_basic(self):
        queries = [
            "CREATE TABLE staging AS SELECT a FROM raw",
            "CREATE TABLE final AS SELECT a FROM staging",
        ]
        pipeline = create_from_sql_list(queries)

        assert len(pipeline.table_graph.queries) == 2

    def test_auto_generates_query_ids(self):
        queries = [
            "CREATE TABLE staging AS SELECT a FROM raw",
            "CREATE TABLE final AS SELECT a FROM staging",
        ]
        pipeline = create_from_sql_list(queries)

        query_ids = list(pipeline.table_graph.queries.keys())
        assert "create_staging" in query_ids
        assert "create_final" in query_ids

    def test_with_dialect(self):
        queries = ["CREATE TABLE t1 AS SELECT a FROM raw"]
        pipeline = create_from_sql_list(queries, dialect="snowflake")

        assert pipeline.dialect == "snowflake"

    def test_returns_pipeline_instance(self):
        queries = ["CREATE TABLE t1 AS SELECT a FROM raw"]
        pipeline = create_from_sql_list(queries)

        assert isinstance(pipeline, Pipeline)


class TestGenerateQueryId:
    """Tests for generate_query_id pure function."""

    def test_create_table(self):
        id_counts: dict = {}
        result = generate_query_id(
            "CREATE TABLE staging AS SELECT a FROM raw", "bigquery", id_counts
        )
        assert result == "create_staging"

    def test_insert_into(self):
        id_counts: dict = {}
        result = generate_query_id(
            "INSERT INTO staging SELECT a FROM raw", "bigquery", id_counts
        )
        assert result == "insert_staging"

    def test_duplicate_gets_suffix(self):
        id_counts: dict = {}
        generate_query_id("CREATE TABLE staging AS SELECT a FROM raw", "bigquery", id_counts)
        result = generate_query_id(
            "INSERT INTO staging SELECT a FROM raw2", "bigquery", id_counts
        )
        # Second reference to staging table should get disambiguated
        assert "staging" in result

    def test_mutates_id_counts(self):
        id_counts: dict = {}
        generate_query_id("CREATE TABLE staging AS SELECT a FROM raw", "bigquery", id_counts)

        assert len(id_counts) > 0

    def test_fallback_on_invalid_sql(self):
        id_counts: dict = {}
        result = generate_query_id("NOT VALID SQL !!!!", "bigquery", id_counts)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_create_view(self):
        id_counts: dict = {}
        result = generate_query_id("CREATE VIEW v1 AS SELECT a FROM raw", "bigquery", id_counts)
        assert "view" in result.lower() or "v1" in result.lower()

    def test_insert_into_select(self):
        id_counts: dict = {}
        result = generate_query_id(
            "INSERT INTO target SELECT a FROM source", "bigquery", id_counts
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_merge_statement(self):
        id_counts: dict = {}
        result = generate_query_id(
            "MERGE INTO target USING source ON target.id = source.id "
            "WHEN MATCHED THEN UPDATE SET target.a = source.a",
            "bigquery",
            id_counts,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_bare_select(self):
        id_counts: dict = {}
        result = generate_query_id("SELECT a, b FROM raw_table", "bigquery", id_counts)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_update_statement(self):
        id_counts: dict = {}
        result = generate_query_id("UPDATE users SET name = 'x' WHERE id = 1", "bigquery", id_counts)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_delete_statement(self):
        id_counts: dict = {}
        result = generate_query_id("DELETE FROM users WHERE id = 1", "bigquery", id_counts)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_duplicate_ids_get_suffix(self):
        id_counts: dict = {}
        id1 = generate_query_id("CREATE TABLE t1 AS SELECT a FROM raw", "bigquery", id_counts)
        id2 = generate_query_id("CREATE TABLE t1 AS SELECT b FROM raw", "bigquery", id_counts)
        assert id1 != id2


class TestCreateFromSqlString:
    """Tests for create_from_sql_string factory function."""

    def test_basic(self):
        sql = """
            CREATE TABLE staging AS SELECT a FROM raw;
            CREATE TABLE final AS SELECT a FROM staging
        """
        pipeline = create_from_sql_string(sql)

        assert len(pipeline.table_graph.queries) == 2

    def test_splits_on_semicolon(self):
        sql = "CREATE TABLE staging AS SELECT a FROM raw; CREATE TABLE final AS SELECT a FROM staging"
        pipeline = create_from_sql_string(sql)

        assert "create_staging" in pipeline.table_graph.queries
        assert "create_final" in pipeline.table_graph.queries

    def test_single_query(self):
        sql = "CREATE TABLE t1 AS SELECT a FROM raw"
        pipeline = create_from_sql_string(sql)

        assert len(pipeline.table_graph.queries) == 1

    def test_with_dialect(self):
        sql = "CREATE TABLE t1 AS SELECT a FROM raw"
        pipeline = create_from_sql_string(sql, dialect="snowflake")

        assert pipeline.dialect == "snowflake"

    def test_returns_pipeline_instance(self):
        sql = "CREATE TABLE t1 AS SELECT a FROM raw"
        pipeline = create_from_sql_string(sql)

        assert isinstance(pipeline, Pipeline)


class TestCreateFromJson:
    """Tests for create_from_json factory function."""

    def _get_json_data(self):
        """Helper to create valid JSON pipeline data."""
        pipeline = create_from_tuples([
            ("q1", "CREATE TABLE t1 AS SELECT a, b FROM raw"),
            ("q2", "CREATE TABLE t2 AS SELECT a FROM t1"),
        ])
        return pipeline.to_json()

    def test_basic(self):
        data = self._get_json_data()
        pipeline = create_from_json(data)

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.table_graph.queries) == 2

    def test_preserves_query_ids(self):
        data = self._get_json_data()
        pipeline = create_from_json(data)

        assert "q1" in pipeline.table_graph.queries
        assert "q2" in pipeline.table_graph.queries

    def test_missing_queries_field_raises(self):
        with pytest.raises(ValueError, match="queries"):
            create_from_json({"dialect": "bigquery"})

    def test_missing_dialect_field_raises(self):
        with pytest.raises(ValueError, match="dialect"):
            create_from_json({"queries": []})

    def test_returns_pipeline_instance(self):
        data = self._get_json_data()
        pipeline = create_from_json(data)

        assert isinstance(pipeline, Pipeline)

    def test_apply_metadata_restores_descriptions(self):
        """Metadata fields (description, owner, pii, tags) are restored from JSON."""
        pipeline_orig = create_from_tuples([
            ("q1", "CREATE TABLE t1 AS SELECT a FROM raw"),
        ])
        # Manually set metadata on a column
        for col in pipeline_orig.columns.values():
            if col.column_name == "a" and "t1" in col.full_name:
                col.description = "Test description"
                col.owner = "data-team"
                col.pii = True
                col.tags = {"sensitive", "test"}
                break

        data = pipeline_orig.to_json()
        pipeline = create_from_json(data, apply_metadata=True)

        # Find the restored column
        restored = None
        for col in pipeline.columns.values():
            if col.column_name == "a" and "t1" in col.full_name:
                restored = col
                break
        assert restored is not None
        assert restored.description == "Test description"
        assert restored.owner == "data-team"
        assert restored.pii is True
        assert "sensitive" in restored.tags


class TestCreateFromJsonFile:
    """Tests for create_from_json_file factory function."""

    def test_basic(self):
        pipeline_orig = create_from_tuples([
            ("q1", "CREATE TABLE t1 AS SELECT a FROM raw"),
        ])
        data = pipeline_orig.to_json()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = f.name

        pipeline = create_from_json_file(tmp_path)

        assert isinstance(pipeline, Pipeline)
        assert "q1" in pipeline.table_graph.queries

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            create_from_json_file("/nonexistent/path/pipeline.json")


class TestCreateFromSqlFiles:
    """Tests for create_from_sql_files factory function."""

    def test_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "staging.sql").write_text(
                "CREATE TABLE staging AS SELECT a FROM raw"
            )
            (Path(tmpdir) / "final.sql").write_text(
                "CREATE TABLE final AS SELECT a FROM staging"
            )

            pipeline = create_from_sql_files(tmpdir)

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.table_graph.queries) == 2

    def test_query_ids_from_filenames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "my_query.sql").write_text(
                "CREATE TABLE t1 AS SELECT a FROM raw"
            )

            pipeline = create_from_sql_files(tmpdir)

        assert "my_query" in pipeline.table_graph.queries

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No SQL files"):
                create_from_sql_files(tmpdir)

    def test_query_id_from_comment(self):
        """query_id_from='comment' extracts ID from '-- query_id: name' comment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "q1.sql").write_text(
                "-- query_id: my_custom_id\nCREATE TABLE t1 AS SELECT a FROM raw"
            )
            pipeline = create_from_sql_files(tmpdir, query_id_from="comment")

        assert "my_custom_id" in pipeline.table_graph.queries

    def test_query_id_from_comment_fallback_to_filename(self):
        """Falls back to filename when no query_id comment found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "fallback_name.sql").write_text(
                "-- just a regular comment\nCREATE TABLE t1 AS SELECT a FROM raw"
            )
            pipeline = create_from_sql_files(tmpdir, query_id_from="comment")

        assert "fallback_name" in pipeline.table_graph.queries

    def test_invalid_query_id_from_raises(self):
        """Invalid query_id_from value raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "q1.sql").write_text(
                "CREATE TABLE t1 AS SELECT a FROM raw"
            )
            with pytest.raises(ValueError, match="query_id_from"):
                create_from_sql_files(tmpdir, query_id_from="invalid_option")


class TestCreateEmpty:
    """Tests for create_empty factory function."""

    def test_basic(self):
        from clgraph.table import TableDependencyGraph

        table_graph = TableDependencyGraph()
        pipeline = create_empty(table_graph)

        assert isinstance(pipeline, Pipeline)
        assert pipeline.table_graph is table_graph

    def test_has_expected_defaults(self):
        from clgraph.table import TableDependencyGraph

        table_graph = TableDependencyGraph()
        pipeline = create_empty(table_graph)

        assert pipeline.dialect == "bigquery"
        assert pipeline.query_mapping == {}
        assert pipeline.llm is None


class TestBackwardCompatibility:
    """Tests that Pipeline class methods still work (backward compatibility)."""

    def test_from_tuples_still_works(self):
        queries = [("q1", "CREATE TABLE t1 AS SELECT a FROM raw")]
        pipeline = Pipeline.from_tuples(queries)

        assert isinstance(pipeline, Pipeline)
        assert "q1" in pipeline.table_graph.queries

    def test_from_dict_still_works(self):
        queries = {"q1": "CREATE TABLE t1 AS SELECT a FROM raw"}
        pipeline = Pipeline.from_dict(queries)

        assert isinstance(pipeline, Pipeline)
        assert "q1" in pipeline.table_graph.queries

    def test_from_sql_string_still_works(self):
        sql = "CREATE TABLE t1 AS SELECT a FROM raw"
        pipeline = Pipeline.from_sql_string(sql)

        assert isinstance(pipeline, Pipeline)

    def test_from_sql_list_still_works(self):
        queries = ["CREATE TABLE t1 AS SELECT a FROM raw"]
        pipeline = Pipeline.from_sql_list(queries)

        assert isinstance(pipeline, Pipeline)

    def test_from_json_still_works(self):
        pipeline_orig = Pipeline.from_tuples([
            ("q1", "CREATE TABLE t1 AS SELECT a FROM raw"),
        ])
        data = pipeline_orig.to_json()
        pipeline = Pipeline.from_json(data)

        assert isinstance(pipeline, Pipeline)
        assert "q1" in pipeline.table_graph.queries

    def test_from_sql_files_still_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "q1.sql").write_text(
                "CREATE TABLE t1 AS SELECT a FROM raw"
            )
            pipeline = Pipeline.from_sql_files(tmpdir)

        assert isinstance(pipeline, Pipeline)
        assert "q1" in pipeline.table_graph.queries
