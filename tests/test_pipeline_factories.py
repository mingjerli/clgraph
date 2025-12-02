"""
Tests for Pipeline factory methods.
"""

import pytest

from clgraph.pipeline import Pipeline


class TestPipelineFromTuples:
    """Tests for Pipeline.from_tuples()"""

    def test_basic(self):
        """Test basic from_tuples usage."""
        queries = [
            ("q1", "CREATE TABLE t1 AS SELECT a FROM raw"),
            ("q2", "CREATE TABLE t2 AS SELECT a FROM t1"),
        ]
        pipeline = Pipeline.from_tuples(queries)

        assert len(pipeline.table_graph.queries) == 2
        assert "q1" in pipeline.table_graph.queries
        assert "q2" in pipeline.table_graph.queries

    def test_with_dialect(self):
        """Test from_tuples with custom dialect."""
        queries = [
            ("q1", "CREATE TABLE t1 AS SELECT a FROM raw"),
        ]
        pipeline = Pipeline.from_tuples(queries, dialect="snowflake")

        assert pipeline.dialect == "snowflake"


class TestPipelineFromDict:
    """Tests for Pipeline.from_dict()"""

    def test_basic(self):
        """Test basic from_dict usage."""
        queries = {
            "staging": "CREATE TABLE staging AS SELECT a FROM raw",
            "final": "CREATE TABLE final AS SELECT a FROM staging",
        }
        pipeline = Pipeline.from_dict(queries)

        assert len(pipeline.table_graph.queries) == 2
        assert "staging" in pipeline.table_graph.queries
        assert "final" in pipeline.table_graph.queries

    def test_preserves_query_ids(self):
        """Test that from_dict preserves the dictionary keys as query IDs."""
        queries = {
            "my_custom_id": "CREATE TABLE t1 AS SELECT a FROM raw",
            "another_id": "CREATE TABLE t2 AS SELECT a FROM t1",
        }
        pipeline = Pipeline.from_dict(queries)

        assert "my_custom_id" in pipeline.table_graph.queries
        assert "another_id" in pipeline.table_graph.queries


class TestPipelineFromSqlList:
    """Tests for Pipeline.from_sql_list()"""

    def test_basic_auto_ids(self):
        """Test that from_sql_list generates meaningful IDs."""
        queries = [
            "CREATE TABLE staging AS SELECT a FROM raw",
            "CREATE TABLE final AS SELECT a FROM staging",
        ]
        pipeline = Pipeline.from_sql_list(queries)

        assert len(pipeline.table_graph.queries) == 2
        assert "create_staging" in pipeline.table_graph.queries
        assert "create_final" in pipeline.table_graph.queries

    def test_duplicate_ids_with_different_sources(self):
        """Test that duplicates use source table in ID."""
        queries = [
            "CREATE TABLE staging AS SELECT a FROM raw1",
            "CREATE TABLE staging AS SELECT a FROM raw2",
        ]
        pipeline = Pipeline.from_sql_list(queries)

        query_ids = list(pipeline.table_graph.queries.keys())
        assert "create_staging" in query_ids
        assert "create_staging_from_raw2" in query_ids

    def test_duplicate_ids_with_same_source(self):
        """Test that duplicates with same source use numbers."""
        queries = [
            "CREATE TABLE staging AS SELECT a FROM raw",
            "CREATE TABLE staging AS SELECT b FROM raw",
            "CREATE TABLE staging AS SELECT c FROM raw",  # third duplicate
        ]
        pipeline = Pipeline.from_sql_list(queries)

        query_ids = list(pipeline.table_graph.queries.keys())
        assert "create_staging" in query_ids
        assert "create_staging_from_raw" in query_ids
        assert "create_staging_from_raw_2" in query_ids

    def test_various_operations(self):
        """Test ID generation for various SQL operations."""
        queries = [
            "CREATE TABLE t1 AS SELECT a FROM raw",
            "CREATE VIEW v1 AS SELECT a FROM t1",
            "INSERT INTO t1 SELECT b FROM raw2",
        ]
        pipeline = Pipeline.from_sql_list(queries)

        query_ids = list(pipeline.table_graph.queries.keys())
        assert "create_t1" in query_ids
        assert "create_view_v1" in query_ids
        assert "insert_t1" in query_ids

    def test_complex_duplication_scenario(self):
        """Test complex scenario with multiple duplicates."""
        queries = [
            "CREATE TABLE staging AS SELECT a FROM raw",
            "INSERT INTO staging SELECT a FROM raw2",
            "CREATE TABLE staging AS SELECT b FROM raw3",
            "CREATE TABLE staging AS SELECT c FROM raw3",  # same source as above
            "CREATE TABLE final AS SELECT a FROM staging",
        ]
        pipeline = Pipeline.from_sql_list(queries)

        query_ids = list(pipeline.table_graph.queries.keys())
        assert "create_staging" in query_ids
        assert "insert_staging" in query_ids
        assert "create_staging_from_raw3" in query_ids
        assert "create_staging_from_raw3_2" in query_ids
        assert "create_final" in query_ids


class TestPipelineFromSqlString:
    """Tests for Pipeline.from_sql_string()"""

    def test_basic(self):
        """Test basic from_sql_string usage."""
        sql = """
            CREATE TABLE staging AS SELECT a FROM raw;
            CREATE TABLE final AS SELECT a FROM staging
        """
        pipeline = Pipeline.from_sql_string(sql)

        assert len(pipeline.table_graph.queries) == 2
        assert "create_staging" in pipeline.table_graph.queries
        assert "create_final" in pipeline.table_graph.queries

    def test_trailing_semicolon(self):
        """Test that trailing semicolons are handled."""
        sql = "CREATE TABLE t1 AS SELECT a FROM raw;"
        pipeline = Pipeline.from_sql_string(sql)

        assert len(pipeline.table_graph.queries) == 1
        assert "create_t1" in pipeline.table_graph.queries

    def test_multiple_semicolons(self):
        """Test multiple queries with semicolons."""
        sql = """
            CREATE TABLE t1 AS SELECT a FROM raw;
            CREATE TABLE t2 AS SELECT a FROM t1;
            CREATE TABLE t3 AS SELECT a FROM t2;
        """
        pipeline = Pipeline.from_sql_string(sql)

        assert len(pipeline.table_graph.queries) == 3

    def test_empty_statements_ignored(self):
        """Test that empty statements between semicolons are ignored."""
        sql = "CREATE TABLE t1 AS SELECT a FROM raw; ; ; CREATE TABLE t2 AS SELECT a FROM t1"
        pipeline = Pipeline.from_sql_string(sql)

        assert len(pipeline.table_graph.queries) == 2


class TestPipelineFromSqlFiles:
    """Tests for Pipeline.from_sql_files()"""

    def test_no_files_raises_error(self, tmp_path):
        """Test that empty directory raises error."""
        with pytest.raises(ValueError, match="No SQL files found"):
            Pipeline.from_sql_files(str(tmp_path))

    def test_load_from_directory(self, tmp_path):
        """Test loading SQL files from directory."""
        # Create test SQL files
        (tmp_path / "01_staging.sql").write_text("CREATE TABLE staging AS SELECT a FROM raw")
        (tmp_path / "02_final.sql").write_text("CREATE TABLE final AS SELECT a FROM staging")

        pipeline = Pipeline.from_sql_files(str(tmp_path))

        assert len(pipeline.table_graph.queries) == 2
        # Query IDs from filenames (without extension)
        assert "01_staging" in pipeline.table_graph.queries
        assert "02_final" in pipeline.table_graph.queries

    def test_query_id_from_comment(self, tmp_path):
        """Test extracting query ID from comment."""
        (tmp_path / "query.sql").write_text(
            "-- query_id: my_custom_id\nCREATE TABLE t1 AS SELECT a FROM raw"
        )

        pipeline = Pipeline.from_sql_files(str(tmp_path), query_id_from="comment")

        assert "my_custom_id" in pipeline.table_graph.queries

    def test_custom_pattern(self, tmp_path):
        """Test loading with custom glob pattern."""
        (tmp_path / "query.sql").write_text("CREATE TABLE t1 AS SELECT a FROM raw")
        (tmp_path / "query.txt").write_text("CREATE TABLE t2 AS SELECT a FROM t1")

        # Only load .sql files
        pipeline = Pipeline.from_sql_files(str(tmp_path), pattern="*.sql")
        assert len(pipeline.table_graph.queries) == 1

        # Load .txt files
        pipeline = Pipeline.from_sql_files(str(tmp_path), pattern="*.txt")
        assert len(pipeline.table_graph.queries) == 1


class TestQueryIdGeneration:
    """Tests for the _generate_query_id static method."""

    def test_create_table(self):
        """Test ID generation for CREATE TABLE."""
        id_counts = {}
        query_id = Pipeline._generate_query_id(
            "CREATE TABLE my_table AS SELECT a FROM raw", "bigquery", id_counts
        )
        assert query_id == "create_my_table"

    def test_create_view(self):
        """Test ID generation for CREATE VIEW."""
        id_counts = {}
        query_id = Pipeline._generate_query_id(
            "CREATE VIEW my_view AS SELECT a FROM raw", "bigquery", id_counts
        )
        assert query_id == "create_view_my_view"

    def test_insert(self):
        """Test ID generation for INSERT."""
        id_counts = {}
        query_id = Pipeline._generate_query_id(
            "INSERT INTO my_table SELECT a FROM raw", "bigquery", id_counts
        )
        assert query_id == "insert_my_table"

    def test_merge(self):
        """Test ID generation for MERGE."""
        id_counts = {}
        query_id = Pipeline._generate_query_id(
            "MERGE INTO target USING source ON target.id = source.id "
            "WHEN MATCHED THEN UPDATE SET a = source.a",
            "bigquery",
            id_counts,
        )
        assert query_id == "merge_target"

    def test_fallback_on_parse_error(self):
        """Test fallback to 'query' on parse error."""
        id_counts = {}
        query_id = Pipeline._generate_query_id("INVALID SQL SYNTAX HERE", "bigquery", id_counts)
        assert query_id == "query"

    def test_duplicate_handling_priority(self):
        """Test that source table is tried before numbering."""
        id_counts = {}

        # First query
        id1 = Pipeline._generate_query_id(
            "CREATE TABLE t1 AS SELECT a FROM raw1", "bigquery", id_counts
        )
        assert id1 == "create_t1"

        # Second query with same dest but different source
        id2 = Pipeline._generate_query_id(
            "CREATE TABLE t1 AS SELECT a FROM raw2", "bigquery", id_counts
        )
        assert id2 == "create_t1_from_raw2"

        # Third query with same dest and same source as second
        id3 = Pipeline._generate_query_id(
            "CREATE TABLE t1 AS SELECT b FROM raw2", "bigquery", id_counts
        )
        assert id3 == "create_t1_from_raw2_2"
