"""
Airflow orchestrator integration for clgraph.

Converts clgraph pipelines to Airflow DAGs using the TaskFlow API.
Supports both Airflow 2.x and 3.x.
"""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable, Optional

from .base import BaseOrchestrator

if TYPE_CHECKING:
    pass


class AirflowOrchestrator(BaseOrchestrator):
    """
    Converts clgraph pipelines to Airflow DAGs.

    Uses the TaskFlow API (@dag and @task decorators) which is compatible
    across both Airflow 2.x and 3.x versions.

    Example:
        from clgraph.orchestrators import AirflowOrchestrator

        orchestrator = AirflowOrchestrator(pipeline)
        dag = orchestrator.to_dag(
            executor=execute_sql,
            dag_id="my_pipeline",
            schedule="@daily",
        )
    """

    def to_dag(
        self,
        executor: Callable[[str], None],
        dag_id: str,
        schedule: str = "@daily",
        start_date: Optional[datetime] = None,
        default_args: Optional[dict] = None,
        airflow_version: Optional[str] = None,
        **dag_kwargs,
    ):
        """
        Create Airflow DAG from the pipeline using TaskFlow API.

        Supports both Airflow 2.x and 3.x. The TaskFlow API (@dag and @task decorators)
        is fully compatible across both versions.

        Args:
            executor: Function that executes SQL (takes sql string)
            dag_id: Airflow DAG ID
            schedule: Schedule interval (default: "@daily")
            start_date: DAG start date (default: datetime(2024, 1, 1))
            default_args: Airflow default_args (default: owner='data_team', retries=2)
            airflow_version: Optional Airflow version ("2" or "3").
                            Auto-detected from installed Airflow if not provided.
            **dag_kwargs: Additional DAG parameters (catchup, tags, max_active_runs,
                         description, max_active_tasks, dagrun_timeout, etc.)
                         See Airflow DAG documentation for all available parameters.

        Returns:
            Airflow DAG instance

        Examples:
            # Basic usage (auto-detects Airflow version)
            dag = orchestrator.to_dag(
                executor=execute_sql,
                dag_id="my_pipeline"
            )

            # Explicit version specification (for testing)
            dag = orchestrator.to_dag(
                executor=execute_sql,
                dag_id="my_pipeline",
                airflow_version="3"
            )

            # Advanced usage with all DAG parameters
            dag = orchestrator.to_dag(
                executor=execute_sql,
                dag_id="my_pipeline",
                schedule="0 0 * * *",  # Daily at midnight
                description="Customer analytics pipeline",
                catchup=False,
                max_active_runs=3,
                max_active_tasks=10,
                tags=["analytics", "daily"],
            )

        Note:
            - Airflow 2.x: Fully supported (2.7.0+)
            - Airflow 3.x: Fully supported (3.0.0+)
            - TaskFlow API is compatible across both versions
        """
        try:
            import airflow  # type: ignore[import-untyped]
            from airflow.decorators import dag, task  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "Airflow is required for DAG generation. "
                "Install it with:\n"
                "  - Airflow 2.x: pip install 'apache-airflow>=2.7.0,<3.0.0'\n"
                "  - Airflow 3.x: pip install 'apache-airflow>=3.0.0'"
            ) from e

        # Detect Airflow version if not specified
        if airflow_version is None:
            detected_version = airflow.__version__
            major_version = int(detected_version.split(".")[0])
            airflow_version = str(major_version)

        # Validate version
        if airflow_version not in ("2", "3"):
            raise ValueError(
                f"Unsupported Airflow version: {airflow_version}. Supported versions: 2, 3"
            )

        if start_date is None:
            start_date = datetime(2024, 1, 1)

        if default_args is None:
            default_args = {
                "owner": "data_team",
                "retries": 2,
                "retry_delay": timedelta(minutes=5),
            }

        # Build DAG parameters
        dag_params = {
            "dag_id": dag_id,
            "schedule": schedule,
            "start_date": start_date,
            "default_args": default_args,
            **dag_kwargs,  # Allow user to override any parameter
        }

        # Set default values only if not provided by user
        dag_params.setdefault("catchup", False)
        dag_params.setdefault("tags", ["clgraph"])

        table_graph = self.table_graph

        @dag(**dag_params)
        def pipeline_dag():
            """Generated pipeline DAG"""

            # Create task callables for each query
            task_callables = {}

            for query_id in table_graph.topological_sort():
                query = table_graph.queries[query_id]
                sql_to_execute = query.sql

                # Create task with unique function name using closure
                def make_task(qid, sql):
                    @task(task_id=qid.replace("-", "_"))
                    def execute_query():
                        """Execute SQL query"""
                        executor(sql)
                        return f"Completed: {qid}"

                    return execute_query

                task_callables[query_id] = make_task(query_id, sql_to_execute)

            # Instantiate all tasks once before wiring dependencies
            task_instances = {qid: callable() for qid, callable in task_callables.items()}

            # Set up dependencies based on table lineage
            for _table_name, table_node in table_graph.tables.items():
                if table_node.created_by:
                    upstream_id = table_node.created_by
                    for downstream_id in table_node.read_by:
                        if upstream_id in task_instances and downstream_id in task_instances:
                            # Airflow: downstream >> upstream means upstream runs first
                            task_instances[upstream_id] >> task_instances[downstream_id]

        return pipeline_dag()


__all__ = ["AirflowOrchestrator"]
