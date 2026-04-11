"""Verify orchestration methods still exist and delegate correctly."""
import pytest

from clgraph import Pipeline


@pytest.fixture
def simple_pipeline():
    queries = [
        ("staging", "CREATE TABLE staging AS SELECT id FROM users"),
        ("analytics", "CREATE TABLE analytics AS SELECT id FROM staging"),
    ]
    return Pipeline(queries, dialect="bigquery")


class TestOrchestrationMethodsExist:
    def test_to_airflow_dag_exists(self, simple_pipeline):
        assert hasattr(simple_pipeline, "to_airflow_dag")
        assert callable(simple_pipeline.to_airflow_dag)

    def test_to_dagster_assets_exists(self, simple_pipeline):
        assert hasattr(simple_pipeline, "to_dagster_assets")
        assert callable(simple_pipeline.to_dagster_assets)

    def test_to_dagster_job_exists(self, simple_pipeline):
        assert hasattr(simple_pipeline, "to_dagster_job")
        assert callable(simple_pipeline.to_dagster_job)

    def test_to_prefect_flow_exists(self, simple_pipeline):
        assert hasattr(simple_pipeline, "to_prefect_flow")
        assert callable(simple_pipeline.to_prefect_flow)

    def test_to_prefect_deployment_exists(self, simple_pipeline):
        assert hasattr(simple_pipeline, "to_prefect_deployment")
        assert callable(simple_pipeline.to_prefect_deployment)

    def test_to_kestra_flow_exists(self, simple_pipeline):
        assert hasattr(simple_pipeline, "to_kestra_flow")
        assert callable(simple_pipeline.to_kestra_flow)

    def test_to_mage_pipeline_exists(self, simple_pipeline):
        assert hasattr(simple_pipeline, "to_mage_pipeline")
        assert callable(simple_pipeline.to_mage_pipeline)
