"""
Orchestrator integrations for clgraph.

This package provides integrations with various workflow orchestrators,
allowing clgraph pipelines to be deployed to production environments.

Supported orchestrators:
- Airflow (2.x and 3.x)
- Dagster (1.x)
- Prefect (2.x and 3.x)
- Kestra (YAML-based declarative workflows)

Example:
    from clgraph import Pipeline
    from clgraph.orchestrators import (
        AirflowOrchestrator,
        DagsterOrchestrator,
        PrefectOrchestrator,
        KestraOrchestrator,
    )

    pipeline = Pipeline.from_sql_files("queries/", dialect="bigquery")

    # Generate Airflow DAG
    airflow = AirflowOrchestrator(pipeline)
    dag = airflow.to_dag(executor=execute_sql, dag_id="my_pipeline")

    # Generate Dagster assets
    dagster = DagsterOrchestrator(pipeline)
    assets = dagster.to_assets(executor=execute_sql, group_name="analytics")

    # Generate Prefect flow
    prefect = PrefectOrchestrator(pipeline)
    flow = prefect.to_flow(executor=execute_sql, flow_name="my_pipeline")

    # Generate Kestra flow YAML
    kestra = KestraOrchestrator(pipeline)
    yaml_content = kestra.to_flow(flow_id="my_pipeline", namespace="clgraph")
"""

from .airflow import AirflowOrchestrator
from .base import BaseOrchestrator
from .dagster import DagsterOrchestrator
from .kestra import KestraOrchestrator
from .prefect import PrefectOrchestrator

__all__ = [
    "BaseOrchestrator",
    "AirflowOrchestrator",
    "DagsterOrchestrator",
    "KestraOrchestrator",
    "PrefectOrchestrator",
]
