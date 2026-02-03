"""
Tests for Mage orchestrator integration.

Tests the to_mage_pipeline() method and MageOrchestrator class.
"""

import pytest

from clgraph import Pipeline


class TestToMagePipelineBasic:
    """Basic tests for to_mage_pipeline method."""

    def test_basic_pipeline_generation(self):
        """Test basic Mage pipeline generation."""
        pipeline = Pipeline(
            [
                ("staging", "CREATE TABLE staging AS SELECT 1 as id"),
                ("analytics", "CREATE TABLE analytics AS SELECT * FROM staging"),
            ]
        )

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        assert "metadata.yaml" in result
        assert "blocks" in result

    def test_metadata_has_required_fields(self):
        """Test metadata.yaml has required fields."""
        pipeline = Pipeline(
            [
                ("staging", "CREATE TABLE staging AS SELECT 1 as id"),
                ("analytics", "CREATE TABLE analytics AS SELECT * FROM staging"),
            ]
        )

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        metadata = result["metadata.yaml"]
        assert "name" in metadata
        assert "uuid" in metadata
        assert "description" in metadata
        assert "type" in metadata
        assert "blocks" in metadata

    def test_block_count_matches_queries(self):
        """Test that block count matches number of queries."""
        pipeline = Pipeline(
            [
                ("q1", "CREATE TABLE t1 AS SELECT 1"),
                ("q2", "CREATE TABLE t2 AS SELECT * FROM t1"),
                ("q3", "CREATE TABLE t3 AS SELECT * FROM t2"),
            ]
        )

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        assert len(result["metadata.yaml"]["blocks"]) == 3
        assert len(result["blocks"]) == 3

    def test_custom_description(self):
        """Test pipeline with custom description."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
            description="Custom description for testing",
        )

        assert result["metadata.yaml"]["description"] == "Custom description for testing"

    def test_auto_generated_description(self):
        """Test auto-generated description."""
        pipeline = Pipeline(
            [
                ("q1", "CREATE TABLE t1 AS SELECT 1"),
                ("q2", "CREATE TABLE t2 AS SELECT * FROM t1"),
            ]
        )

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        description = result["metadata.yaml"]["description"]
        assert "2 queries" in description
        assert "clgraph" in description

    def test_default_pipeline_type(self):
        """Test default pipeline type is python."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        assert result["metadata.yaml"]["type"] == "python"

    def test_blocks_are_python_code_strings(self):
        """Test that block values are Python code strings."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        for block_code in result["blocks"].values():
            assert isinstance(block_code, str)
            assert "def " in block_code


class TestMageBlockTypes:
    """Test block type assignment."""

    def test_source_query_is_data_loader(self):
        """Test that source query (no upstream) gets data_loader type."""
        pipeline = Pipeline(
            [
                ("source", "CREATE TABLE source AS SELECT 1 as id"),
                ("derived", "CREATE TABLE derived AS SELECT * FROM source"),
            ]
        )

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        blocks = result["metadata.yaml"]["blocks"]
        source_block = next(b for b in blocks if b["name"] == "source")
        assert source_block["type"] == "data_loader"

    def test_dependent_query_is_transformer(self):
        """Test that dependent query gets transformer type."""
        pipeline = Pipeline(
            [
                ("source", "CREATE TABLE source AS SELECT 1 as id"),
                ("derived", "CREATE TABLE derived AS SELECT * FROM source"),
            ]
        )

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        blocks = result["metadata.yaml"]["blocks"]
        derived_block = next(b for b in blocks if b["name"] == "derived")
        assert derived_block["type"] == "transformer"

    def test_data_loader_decorator_in_code(self):
        """Test that data_loader block code contains correct decorator."""
        pipeline = Pipeline(
            [
                ("source", "CREATE TABLE source AS SELECT 1 as id"),
                ("derived", "CREATE TABLE derived AS SELECT * FROM source"),
            ]
        )

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        source_code = result["blocks"]["source"]
        assert "@data_loader" in source_code
        assert "def load_data" in source_code

    def test_transformer_decorator_in_code(self):
        """Test that transformer block code contains correct decorator."""
        pipeline = Pipeline(
            [
                ("source", "CREATE TABLE source AS SELECT 1 as id"),
                ("derived", "CREATE TABLE derived AS SELECT * FROM source"),
            ]
        )

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        derived_code = result["blocks"]["derived"]
        assert "@transformer" in derived_code
        assert "def transform" in derived_code


class TestMageDependencies:
    """Test upstream/downstream block wiring."""

    def test_linear_chain_upstream(self):
        """Test linear chain has correct upstream_blocks."""
        pipeline = Pipeline(
            [
                ("step1", "CREATE TABLE step1 AS SELECT 1"),
                ("step2", "CREATE TABLE step2 AS SELECT * FROM step1"),
                ("step3", "CREATE TABLE step3 AS SELECT * FROM step2"),
            ]
        )

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        blocks = result["metadata.yaml"]["blocks"]
        block_map = {b["name"]: b for b in blocks}

        assert block_map["step1"]["upstream_blocks"] == []
        assert block_map["step2"]["upstream_blocks"] == ["step1"]
        assert block_map["step3"]["upstream_blocks"] == ["step2"]

    def test_linear_chain_downstream(self):
        """Test linear chain has correct downstream_blocks."""
        pipeline = Pipeline(
            [
                ("step1", "CREATE TABLE step1 AS SELECT 1"),
                ("step2", "CREATE TABLE step2 AS SELECT * FROM step1"),
                ("step3", "CREATE TABLE step3 AS SELECT * FROM step2"),
            ]
        )

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        blocks = result["metadata.yaml"]["blocks"]
        block_map = {b["name"]: b for b in blocks}

        assert block_map["step1"]["downstream_blocks"] == ["step2"]
        assert block_map["step2"]["downstream_blocks"] == ["step3"]
        assert block_map["step3"]["downstream_blocks"] == []

    def test_diamond_pattern_dependencies(self):
        """Test diamond pattern wiring for all 4 blocks."""
        pipeline = Pipeline(
            [
                ("source", "CREATE TABLE source AS SELECT 1 as id"),
                ("left_branch", "CREATE TABLE left_branch AS SELECT * FROM source"),
                ("right_branch", "CREATE TABLE right_branch AS SELECT * FROM source"),
                (
                    "final",
                    "CREATE TABLE final AS SELECT * FROM left_branch, right_branch",
                ),
            ]
        )

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        blocks = result["metadata.yaml"]["blocks"]
        block_map = {b["name"]: b for b in blocks}

        # Source has no upstream
        assert block_map["source"]["upstream_blocks"] == []
        # Branches depend on source
        assert block_map["left_branch"]["upstream_blocks"] == ["source"]
        assert block_map["right_branch"]["upstream_blocks"] == ["source"]
        # Final depends on both branches
        assert sorted(block_map["final"]["upstream_blocks"]) == [
            "left_branch",
            "right_branch",
        ]

        # Source has both branches as downstream
        assert sorted(block_map["source"]["downstream_blocks"]) == [
            "left_branch",
            "right_branch",
        ]
        # Final has no downstream
        assert block_map["final"]["downstream_blocks"] == []

    def test_source_has_empty_upstream(self):
        """Test that source blocks have empty upstream_blocks."""
        pipeline = Pipeline(
            [
                ("source1", "CREATE TABLE source1 AS SELECT 1"),
                ("source2", "CREATE TABLE source2 AS SELECT 2"),
                (
                    "combined",
                    "CREATE TABLE combined AS SELECT * FROM source1, source2",
                ),
            ]
        )

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        blocks = result["metadata.yaml"]["blocks"]
        block_map = {b["name"]: b for b in blocks}

        assert block_map["source1"]["upstream_blocks"] == []
        assert block_map["source2"]["upstream_blocks"] == []

    def test_leaf_has_empty_downstream(self):
        """Test that leaf blocks have empty downstream_blocks."""
        pipeline = Pipeline(
            [
                ("source", "CREATE TABLE source AS SELECT 1"),
                ("leaf", "CREATE TABLE leaf AS SELECT * FROM source"),
            ]
        )

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        blocks = result["metadata.yaml"]["blocks"]
        block_map = {b["name"]: b for b in blocks}

        assert block_map["leaf"]["downstream_blocks"] == []


class TestMageConfiguration:
    """Test connection and configuration handling."""

    def test_default_connection_name_in_code(self):
        """Test default connection name clickhouse_default in block code."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        block_code = list(result["blocks"].values())[0]
        assert "clickhouse_default" in block_code

    def test_custom_connection_name_in_code(self):
        """Test custom connection name embedded in block code."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
            connection_name="my_custom_conn",
        )

        block_code = list(result["blocks"].values())[0]
        assert "my_custom_conn" in block_code

    def test_sql_embedded_in_block_code(self):
        """Test that SQL is embedded in block code."""
        sql = "CREATE TABLE test_table AS SELECT 42 as value"
        pipeline = Pipeline([("q1", sql)])

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        block_code = list(result["blocks"].values())[0]
        assert sql in block_code

    def test_clickhouse_import_in_block_code(self):
        """Test that ClickHouse import is present in block code by default."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        block_code = list(result["blocks"].values())[0]
        assert "from mage_ai.io.clickhouse import ClickHouse" in block_code


class TestMageDbConnector:
    """Test configurable database connector."""

    def test_postgres_connector(self):
        """Test postgres connector generates correct import and class."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
            db_connector="postgres",
            connection_name="pg_default",
        )

        block_code = list(result["blocks"].values())[0]
        assert "from mage_ai.io.postgres import Postgres" in block_code
        assert "Postgres.with_config" in block_code
        assert "pg_default" in block_code

    def test_bigquery_connector(self):
        """Test bigquery connector generates correct import and class."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
            db_connector="bigquery",
            connection_name="bq_default",
        )

        block_code = list(result["blocks"].values())[0]
        assert "from mage_ai.io.bigquery import BigQuery" in block_code
        assert "BigQuery.with_config" in block_code
        assert "bq_default" in block_code

    def test_snowflake_connector(self):
        """Test snowflake connector generates correct import and class."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
            db_connector="snowflake",
            connection_name="sf_default",
        )

        block_code = list(result["blocks"].values())[0]
        assert "from mage_ai.io.snowflake import Snowflake" in block_code
        assert "Snowflake.with_config" in block_code
        assert "sf_default" in block_code

    def test_default_connector_is_clickhouse(self):
        """Test that default db_connector is clickhouse."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        result = pipeline.to_mage_pipeline(
            pipeline_name="test_pipeline",
        )

        block_code = list(result["blocks"].values())[0]
        assert "ClickHouse.with_config" in block_code

    def test_unsupported_connector_raises_error(self):
        """Test that unsupported db_connector raises ValueError."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        with pytest.raises(ValueError, match="Unsupported db_connector"):
            pipeline.to_mage_pipeline(
                pipeline_name="test_pipeline",
                db_connector="mysql",
            )


class TestMageConnectionNameValidation:
    """Test connection_name validation to prevent code injection."""

    def test_valid_connection_names(self):
        """Test that valid connection names are accepted."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        for name in ["clickhouse_default", "my-conn", "conn123", "a_b-c"]:
            result = pipeline.to_mage_pipeline(
                pipeline_name="test_pipeline",
                connection_name=name,
            )
            block_code = list(result["blocks"].values())[0]
            assert name in block_code

    def test_invalid_connection_name_raises_error(self):
        """Test that invalid connection_name raises ValueError."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        with pytest.raises(ValueError, match="Invalid connection_name"):
            pipeline.to_mage_pipeline(
                pipeline_name="test_pipeline",
                connection_name='"; import os; os.system("rm -rf /")',
            )

    def test_connection_name_with_spaces_raises_error(self):
        """Test that connection_name with spaces raises ValueError."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        with pytest.raises(ValueError, match="Invalid connection_name"):
            pipeline.to_mage_pipeline(
                pipeline_name="test_pipeline",
                connection_name="my connection",
            )

    def test_connection_name_with_dots_raises_error(self):
        """Test that connection_name with dots raises ValueError."""
        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])

        with pytest.raises(ValueError, match="Invalid connection_name"):
            pipeline.to_mage_pipeline(
                pipeline_name="test_pipeline",
                connection_name="my.connection",
            )


class TestMageOrchestrator:
    """Test MageOrchestrator class directly."""

    def test_orchestrator_initialization(self):
        """Test MageOrchestrator initialization."""
        from clgraph.orchestrators import MageOrchestrator

        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])
        orchestrator = MageOrchestrator(pipeline)

        assert orchestrator.pipeline == pipeline
        assert orchestrator.table_graph == pipeline.table_graph

    def test_to_pipeline_config_returns_dict(self):
        """Test to_pipeline_config returns dictionary."""
        from clgraph.orchestrators import MageOrchestrator

        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])
        orchestrator = MageOrchestrator(pipeline)

        result = orchestrator.to_pipeline_config(pipeline_name="test")

        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["uuid"] == "test"

    def test_to_blocks_returns_dict(self):
        """Test to_blocks returns dictionary."""
        from clgraph.orchestrators import MageOrchestrator

        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])
        orchestrator = MageOrchestrator(pipeline)

        result = orchestrator.to_blocks()

        assert isinstance(result, dict)
        assert len(result) == 1

    def test_to_pipeline_files_returns_combined_dict(self):
        """Test to_pipeline_files returns combined dictionary."""
        from clgraph.orchestrators import MageOrchestrator

        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])
        orchestrator = MageOrchestrator(pipeline)

        result = orchestrator.to_pipeline_files(
            pipeline_name="test",
        )

        assert "metadata.yaml" in result
        assert "blocks" in result
        assert isinstance(result["metadata.yaml"], dict)
        assert isinstance(result["blocks"], dict)

    def test_sanitize_name(self):
        """Test _sanitize_name works correctly."""
        from clgraph.orchestrators import MageOrchestrator

        pipeline = Pipeline([("q1", "CREATE TABLE t1 AS SELECT 1")])
        orchestrator = MageOrchestrator(pipeline)

        assert orchestrator._sanitize_name("my.table") == "my_table"
        assert orchestrator._sanitize_name("my-table") == "my_table"
        assert orchestrator._sanitize_name("my_table") == "my_table"


class TestMageComplexPipeline:
    """Test Mage pipeline generation with complex pipelines."""

    def test_enterprise_like_pipeline(self):
        """Test with a pipeline similar to enterprise demo."""
        pipeline = Pipeline(
            [
                (
                    "raw_sales",
                    """
                    CREATE TABLE raw_sales AS
                    SELECT
                        toDate('2024-01-01') + number as date,
                        number % 100 as product_id,
                        number % 10 as region_id,
                        rand() % 1000 as amount
                    FROM numbers(1000)
                """,
                ),
                (
                    "raw_products",
                    """
                    CREATE TABLE raw_products AS
                    SELECT
                        number as product_id,
                        concat('Product ', toString(number)) as product_name,
                        rand() % 5 as category_id
                    FROM numbers(100)
                """,
                ),
                (
                    "sales_with_products",
                    """
                    CREATE TABLE sales_with_products AS
                    SELECT
                        s.date,
                        s.product_id,
                        p.product_name,
                        s.region_id,
                        s.amount
                    FROM raw_sales s
                    JOIN raw_products p ON s.product_id = p.product_id
                """,
                ),
                (
                    "daily_summary",
                    """
                    CREATE TABLE daily_summary AS
                    SELECT
                        date,
                        count() as num_sales,
                        sum(amount) as total_amount
                    FROM sales_with_products
                    GROUP BY date
                """,
                ),
            ],
            dialect="clickhouse",
        )

        result = pipeline.to_mage_pipeline(
            pipeline_name="enterprise_pipeline",
        )

        metadata = result["metadata.yaml"]
        assert metadata["name"] == "enterprise_pipeline"
        assert len(metadata["blocks"]) == 4

        block_map = {b["name"]: b for b in metadata["blocks"]}

        # Source blocks are data_loaders
        assert block_map["raw_sales"]["type"] == "data_loader"
        assert block_map["raw_products"]["type"] == "data_loader"

        # Dependent blocks are transformers
        assert block_map["sales_with_products"]["type"] == "transformer"
        assert block_map["daily_summary"]["type"] == "transformer"

        # sales_with_products depends on both raw tables
        assert sorted(block_map["sales_with_products"]["upstream_blocks"]) == [
            "raw_products",
            "raw_sales",
        ]

        # daily_summary depends on sales_with_products
        assert block_map["daily_summary"]["upstream_blocks"] == ["sales_with_products"]

    def test_10_step_linear_chain(self):
        """Test 10-step linear chain has correct sequential dependencies."""
        queries = []
        for i in range(10):
            if i == 0:
                queries.append((f"step_{i}", f"CREATE TABLE step_{i} AS SELECT {i}"))
            else:
                queries.append(
                    (
                        f"step_{i}",
                        f"CREATE TABLE step_{i} AS SELECT * FROM step_{i - 1}",
                    )
                )

        pipeline = Pipeline(queries)

        result = pipeline.to_mage_pipeline(
            pipeline_name="long_chain",
        )

        metadata = result["metadata.yaml"]
        assert len(metadata["blocks"]) == 10

        block_map = {b["name"]: b for b in metadata["blocks"]}

        # First step is data_loader with no upstream
        assert block_map["step_0"]["type"] == "data_loader"
        assert block_map["step_0"]["upstream_blocks"] == []

        # All other steps are transformers with correct upstream
        for i in range(1, 10):
            assert block_map[f"step_{i}"]["type"] == "transformer"
            assert block_map[f"step_{i}"]["upstream_blocks"] == [f"step_{i - 1}"]

        # Check downstream wiring
        for i in range(9):
            assert block_map[f"step_{i}"]["downstream_blocks"] == [f"step_{i + 1}"]
        assert block_map["step_9"]["downstream_blocks"] == []
