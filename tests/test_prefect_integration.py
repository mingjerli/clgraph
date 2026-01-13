"""
Tests for Prefect integration (to_prefect_flow, to_prefect_deployment).
"""

import pytest

from clgraph.pipeline import Pipeline

# Check if Prefect is available
try:
    import prefect  # noqa: F401
    from prefect import flow, task  # noqa: F401

    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False


class TestToPrefectFlowBasic:
    """Basic tests for to_prefect_flow method."""

    def test_requires_prefect(self):
        """Test that to_prefect_flow raises error when Prefect is not installed."""
        queries = [("query1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        try:
            flow_fn = pipeline.to_prefect_flow(executor=mock_executor, flow_name="test_flow")
            # If we get here, prefect is installed
            assert flow_fn is not None
        except ImportError as e:
            # Expected if prefect not installed
            assert "Prefect is required" in str(e)

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_basic_flow_generation(self):
        """Test basic flow generation from pipeline."""
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

        flow_fn = pipeline.to_prefect_flow(executor=mock_executor, flow_name="test_pipeline")

        # Should return a flow function
        assert flow_fn is not None
        assert flow_fn.name == "test_pipeline"

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_flow_with_custom_description(self):
        """Test flow with custom description."""
        queries = [("query1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        flow_fn = pipeline.to_prefect_flow(
            executor=mock_executor,
            flow_name="custom_flow",
            description="My custom description",
        )

        assert flow_fn.description == "My custom description"

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_flow_auto_generated_description(self):
        """Test that description is auto-generated when not provided."""
        queries = [
            ("q1", "CREATE TABLE t1 AS SELECT 1"),
            ("q2", "CREATE TABLE t2 AS SELECT 2"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        flow_fn = pipeline.to_prefect_flow(executor=mock_executor, flow_name="auto_desc_flow")

        assert "2 queries" in flow_fn.description
        assert "clgraph" in flow_fn.description

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_flow_with_custom_retries(self):
        """Test flow with custom retry settings."""
        queries = [("query1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        flow_fn = pipeline.to_prefect_flow(
            executor=mock_executor,
            flow_name="retry_flow",
            retries=5,
            retry_delay_seconds=120,
        )

        assert flow_fn is not None
        assert flow_fn.name == "retry_flow"

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_flow_with_tags(self):
        """Test flow with custom tags."""
        queries = [("query1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        flow_fn = pipeline.to_prefect_flow(
            executor=mock_executor,
            flow_name="tagged_flow",
            tags=["production", "critical"],
        )

        assert flow_fn is not None


class TestToPrefectFlowDependencies:
    """Test dependency handling in Prefect flows."""

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_linear_dependencies(self):
        """Test linear dependency chain A -> B -> C."""
        queries = [
            ("step1", "CREATE TABLE step1 AS SELECT 1 as id"),
            ("step2", "CREATE TABLE step2 AS SELECT * FROM step1"),
            ("step3", "CREATE TABLE step3 AS SELECT * FROM step2"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        flow_fn = pipeline.to_prefect_flow(executor=mock_executor, flow_name="linear_flow")

        assert flow_fn is not None

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_diamond_dependencies(self):
        """Test diamond pattern: A -> B, A -> C, B -> D, C -> D."""
        queries = [
            ("source", "CREATE TABLE source AS SELECT 1 as id"),
            ("left", "CREATE TABLE left AS SELECT * FROM source"),
            ("right", "CREATE TABLE right AS SELECT * FROM source"),
            ("final", "CREATE TABLE final AS SELECT * FROM left, right"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        flow_fn = pipeline.to_prefect_flow(executor=mock_executor, flow_name="diamond_flow")

        assert flow_fn is not None

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_parallel_independent_queries(self):
        """Test parallel execution of independent queries."""
        queries = [
            ("table1", "CREATE TABLE table1 AS SELECT 1 as id"),
            ("table2", "CREATE TABLE table2 AS SELECT 2 as id"),
            ("table3", "CREATE TABLE table3 AS SELECT 3 as id"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        flow_fn = pipeline.to_prefect_flow(executor=mock_executor, flow_name="parallel_flow")

        assert flow_fn is not None


class TestToPrefectFlowExecution:
    """Test Prefect flow execution."""

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_flow_execution_tracking(self):
        """Test that queries execute in correct order."""
        executed_queries = []

        def tracking_executor(sql: str):
            executed_queries.append(sql)

        queries = [
            ("staging", "CREATE TABLE staging AS SELECT 1"),
            ("analytics", "CREATE TABLE analytics AS SELECT * FROM staging"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        flow_fn = pipeline.to_prefect_flow(executor=tracking_executor, flow_name="tracking_flow")

        # Run the flow
        flow_fn()

        # Check that both queries were executed
        assert len(executed_queries) == 2
        # Staging must execute before analytics
        staging_idx = next(i for i, sql in enumerate(executed_queries) if "staging" in sql.lower())
        analytics_idx = next(
            i for i, sql in enumerate(executed_queries) if "analytics" in sql.lower()
        )
        assert staging_idx < analytics_idx

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_flow_returns_results(self):
        """Test that flow returns query IDs as results."""

        def mock_executor(sql: str):
            pass

        queries = [
            ("query1", "CREATE TABLE table1 AS SELECT 1"),
            ("query2", "CREATE TABLE table2 AS SELECT 2"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        flow_fn = pipeline.to_prefect_flow(executor=mock_executor, flow_name="result_flow")

        result = flow_fn()

        assert "query1" in result
        assert "query2" in result


class TestToPrefectDeployment:
    """Tests for to_prefect_deployment method."""

    def test_requires_prefect_for_deployment(self):
        """Test that to_prefect_deployment raises error when Prefect is not installed."""
        queries = [("query1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        try:
            deployment = pipeline.to_prefect_deployment(
                executor=mock_executor,
                flow_name="test_flow",
                deployment_name="test_deployment",
            )
            # If we get here, prefect is installed
            assert deployment is not None
        except ImportError as e:
            # Expected if prefect not installed
            assert "Prefect is required" in str(e)

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_basic_deployment_creation(self):
        """Test basic deployment creation."""
        queries = [("query1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        deployment = pipeline.to_prefect_deployment(
            executor=mock_executor,
            flow_name="deploy_flow",
            deployment_name="test_deployment",
        )

        assert deployment is not None
        # In Prefect 3.x, RunnerDeployment has 'name' attribute
        assert deployment.name == "test_deployment"

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_deployment_with_cron_schedule(self):
        """Test deployment with cron schedule."""
        queries = [("query1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        deployment = pipeline.to_prefect_deployment(
            executor=mock_executor,
            flow_name="scheduled_flow",
            deployment_name="scheduled_deployment",
            cron="0 0 * * *",  # Daily at midnight
        )

        assert deployment is not None
        # In Prefect 3.x, schedules are stored differently
        assert deployment.schedules is not None or deployment.cron is not None

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_deployment_with_interval_schedule(self):
        """Test deployment with interval schedule."""
        queries = [("query1", "CREATE TABLE table1 AS SELECT 1 as id")]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        deployment = pipeline.to_prefect_deployment(
            executor=mock_executor,
            flow_name="interval_flow",
            deployment_name="interval_deployment",
            interval_seconds=3600,  # Every hour
        )

        assert deployment is not None
        # In Prefect 3.x, interval schedules are stored differently
        assert deployment.schedules is not None or deployment.interval is not None


class TestPrefectOrchestratorDirectUsage:
    """Test using PrefectOrchestrator directly."""

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_direct_orchestrator_usage(self):
        """Test using PrefectOrchestrator directly instead of pipeline methods."""
        from clgraph.orchestrators import PrefectOrchestrator

        queries = [
            ("staging", "CREATE TABLE staging AS SELECT 1 as id"),
            ("analytics", "CREATE TABLE analytics AS SELECT * FROM staging"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        def mock_executor(sql: str):
            pass

        orchestrator = PrefectOrchestrator(pipeline)
        flow_fn = orchestrator.to_flow(executor=mock_executor, flow_name="direct_flow")

        assert flow_fn is not None
        assert flow_fn.name == "direct_flow"

    @pytest.mark.skipif(not PREFECT_AVAILABLE, reason="Prefect not installed")
    def test_sanitize_name(self):
        """Test that task names are properly sanitized."""
        from clgraph.orchestrators import PrefectOrchestrator

        queries = [
            ("my-query.name", "CREATE TABLE my_table AS SELECT 1"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        orchestrator = PrefectOrchestrator(pipeline)

        # Check sanitization
        sanitized = orchestrator._sanitize_name("my-query.name")
        assert sanitized == "my_query_name"
        assert "-" not in sanitized
        assert "." not in sanitized
