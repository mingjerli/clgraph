"""
Tests for Dagster integration (to_dagster_assets, to_dagster_job).
"""

import pytest

from clgraph.pipeline import Pipeline

# Check if Dagster is available
try:
    import dagster  # noqa: F401

    DAGSTER_AVAILABLE = True
except ImportError:
    DAGSTER_AVAILABLE = False


class TestToDagsterAssetsBasic:
    """Basic tests for to_dagster_assets method."""

    def test_requires_dagster(self):
        """Test that to_dagster_assets raises error when Dagster is not installed."""
        queries = [("query1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        try:
            assets = pipeline.to_dagster_assets(executor=mock_executor, group_name="test")
            # If we get here, dagster is installed
            assert assets is not None
            assert len(assets) == 1
        except ImportError as e:
            # Expected if dagster not installed
            assert "Dagster is required" in str(e)

    @pytest.mark.skipif(not DAGSTER_AVAILABLE, reason="Dagster not installed")
    def test_basic_asset_generation(self):
        """Test basic asset generation from pipeline."""
        queries = [
            ("staging", "CREATE TABLE staging AS SELECT 1 as id, 'Alice' as name"),
            (
                "analytics",
                "CREATE TABLE analytics AS SELECT id, name FROM staging WHERE id = 1",
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        assets = pipeline.to_dagster_assets(executor=mock_executor, group_name="test_group")

        # Should create 2 assets
        assert len(assets) == 2

        # Check asset keys
        asset_keys = [asset.key.path[-1] for asset in assets]
        assert "staging" in asset_keys
        assert "analytics" in asset_keys

    @pytest.mark.skipif(not DAGSTER_AVAILABLE, reason="Dagster not installed")
    def test_asset_dependencies(self):
        """Test that asset dependencies are correctly wired."""
        queries = [
            ("raw", "CREATE TABLE raw AS SELECT 1 as id"),
            ("staging", "CREATE TABLE staging AS SELECT * FROM raw"),
            ("analytics", "CREATE TABLE analytics AS SELECT * FROM staging"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        assets = pipeline.to_dagster_assets(executor=mock_executor)

        # Find analytics asset (should depend on staging)
        analytics_asset = next(a for a in assets if "analytics" in a.key.path)

        # Check it has dependencies
        assert len(analytics_asset.deps) > 0

        # The dependency should be staging
        dep_keys = [str(d.asset_key) for d in analytics_asset.deps]
        assert any("staging" in k for k in dep_keys)

    @pytest.mark.skipif(not DAGSTER_AVAILABLE, reason="Dagster not installed")
    def test_asset_with_key_prefix(self):
        """Test asset key prefix is applied correctly."""
        queries = [("table1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        # Test with string prefix
        assets = pipeline.to_dagster_assets(executor=mock_executor, key_prefix="warehouse")
        assert assets[0].key.path == ["warehouse", "table1"]

        # Test with list prefix
        assets = pipeline.to_dagster_assets(
            executor=mock_executor, key_prefix=["prod", "analytics"]
        )
        assert assets[0].key.path == ["prod", "analytics", "table1"]

    @pytest.mark.skipif(not DAGSTER_AVAILABLE, reason="Dagster not installed")
    def test_asset_group_name(self):
        """Test asset group name is set correctly."""
        queries = [("table1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        assets = pipeline.to_dagster_assets(executor=mock_executor, group_name="my_group")

        assert assets[0].group_names_by_key[assets[0].key] == "my_group"

    @pytest.mark.skipif(not DAGSTER_AVAILABLE, reason="Dagster not installed")
    def test_asset_compute_kind(self):
        """Test asset compute kind is set correctly."""
        queries = [("table1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        assets = pipeline.to_dagster_assets(executor=mock_executor, compute_kind="clickhouse")

        # Check compute_kind in asset's op
        assert assets[0].op.tags.get("dagster/compute_kind") == "clickhouse"

    @pytest.mark.skipif(not DAGSTER_AVAILABLE, reason="Dagster not installed")
    def test_asset_skips_queries_without_target_table(self):
        """Test that queries without target tables are skipped."""
        queries = [
            ("insert_query", "INSERT INTO existing_table SELECT 1"),  # No target table
            ("create_query", "CREATE TABLE new_table AS SELECT 1 as id"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        assets = pipeline.to_dagster_assets(executor=mock_executor)

        # Should only create 1 asset (for CREATE TABLE)
        assert len(assets) == 1
        assert "new_table" in assets[0].key.path


class TestToDagsterAssetsExecution:
    """Tests for Dagster asset execution."""

    @pytest.mark.skipif(not DAGSTER_AVAILABLE, reason="Dagster not installed")
    def test_asset_execution_calls_executor(self):
        """Test that materializing an asset calls the executor."""
        queries = [("table1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        executed_sql = []

        def tracking_executor(sql: str):
            executed_sql.append(sql)

        assets = pipeline.to_dagster_assets(executor=tracking_executor, group_name="test")

        # Materialize the asset
        from dagster import materialize

        result = materialize(assets)

        assert result.success
        assert len(executed_sql) == 1
        assert "table1" in executed_sql[0]

    @pytest.mark.skipif(not DAGSTER_AVAILABLE, reason="Dagster not installed")
    def test_asset_execution_order(self):
        """Test that assets execute in correct dependency order."""
        queries = [
            ("level1", "CREATE TABLE level1 AS SELECT 1 as id"),
            ("level2", "CREATE TABLE level2 AS SELECT * FROM level1"),
            ("level3", "CREATE TABLE level3 AS SELECT * FROM level2"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        execution_order = []

        def tracking_executor(sql: str):
            if "level1" in sql:
                execution_order.append("level1")
            elif "level2" in sql:
                execution_order.append("level2")
            elif "level3" in sql:
                execution_order.append("level3")

        assets = pipeline.to_dagster_assets(executor=tracking_executor)

        from dagster import materialize

        result = materialize(assets)

        assert result.success
        assert execution_order == ["level1", "level2", "level3"]


class TestToDagsterJob:
    """Tests for to_dagster_job method."""

    def test_requires_dagster(self):
        """Test that to_dagster_job raises error when Dagster is not installed."""
        queries = [("query1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        try:
            job = pipeline.to_dagster_job(executor=mock_executor, job_name="test_job")
            # If we get here, dagster is installed
            assert job is not None
            assert job.name == "test_job"
        except ImportError as e:
            # Expected if dagster not installed
            assert "Dagster is required" in str(e)

    @pytest.mark.skipif(not DAGSTER_AVAILABLE, reason="Dagster not installed")
    def test_basic_job_generation(self):
        """Test basic job generation from pipeline."""
        queries = [
            ("staging", "CREATE TABLE staging AS SELECT 1 as id"),
            ("analytics", "CREATE TABLE analytics AS SELECT * FROM staging"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        job = pipeline.to_dagster_job(executor=mock_executor, job_name="my_pipeline")

        assert job.name == "my_pipeline"
        assert job.description is not None
        assert "2 queries" in job.description

    @pytest.mark.skipif(not DAGSTER_AVAILABLE, reason="Dagster not installed")
    def test_job_with_custom_description(self):
        """Test job with custom description."""
        queries = [("table1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        job = pipeline.to_dagster_job(
            executor=mock_executor,
            job_name="custom_job",
            description="My custom description",
        )

        assert job.description == "My custom description"

    @pytest.mark.skipif(not DAGSTER_AVAILABLE, reason="Dagster not installed")
    def test_job_with_tags(self):
        """Test job with custom tags."""
        queries = [("table1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        job = pipeline.to_dagster_job(
            executor=mock_executor,
            job_name="tagged_job",
            tags={"team": "data-eng", "env": "prod"},
        )

        assert job.tags.get("team") == "data-eng"
        assert job.tags.get("env") == "prod"

    @pytest.mark.skipif(not DAGSTER_AVAILABLE, reason="Dagster not installed")
    def test_job_execution_in_process(self):
        """Test job can be executed in process."""
        queries = [
            ("table1", "CREATE TABLE table1 AS SELECT 1 as id"),
            ("table2", "CREATE TABLE table2 AS SELECT * FROM table1"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        executed_sql = []

        def tracking_executor(sql: str):
            executed_sql.append(sql)

        job = pipeline.to_dagster_job(executor=tracking_executor, job_name="test_execution")

        result = job.execute_in_process()

        assert result.success
        assert len(executed_sql) == 2


class TestDagsterDefinitionsIntegration:
    """Tests for creating complete Dagster Definitions."""

    @pytest.mark.skipif(not DAGSTER_AVAILABLE, reason="Dagster not installed")
    def test_assets_with_definitions(self):
        """Test creating Dagster Definitions from assets."""
        from dagster import Definitions

        queries = [
            ("staging", "CREATE TABLE staging AS SELECT 1 as id"),
            ("analytics", "CREATE TABLE analytics AS SELECT * FROM staging"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        assets = pipeline.to_dagster_assets(executor=mock_executor, group_name="demo")

        # Create Definitions (this is what users do in definitions.py)
        defs = Definitions(assets=assets)

        assert defs is not None
        # Check that assets are registered
        all_asset_keys = defs.get_all_asset_keys()
        assert len(all_asset_keys) == 2

    @pytest.mark.skipif(not DAGSTER_AVAILABLE, reason="Dagster not installed")
    def test_job_with_definitions(self):
        """Test creating Dagster Definitions from job."""
        from dagster import Definitions

        queries = [("table1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        job = pipeline.to_dagster_job(executor=mock_executor, job_name="demo_job")

        # Create Definitions
        defs = Definitions(jobs=[job])

        assert defs is not None
        # Check that job is registered
        assert defs.get_job_def("demo_job") is not None
