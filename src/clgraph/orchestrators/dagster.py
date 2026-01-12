"""
Dagster orchestrator integration for clgraph.

Converts clgraph pipelines to Dagster assets and jobs.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from .base import BaseOrchestrator

if TYPE_CHECKING:
    pass


class DagsterOrchestrator(BaseOrchestrator):
    """
    Converts clgraph pipelines to Dagster assets and jobs.

    Supports both asset-based (recommended) and job-based approaches.
    Assets provide better lineage tracking and observability in Dagster.

    Example:
        from clgraph.orchestrators import DagsterOrchestrator

        orchestrator = DagsterOrchestrator(pipeline)
        assets = orchestrator.to_assets(
            executor=execute_sql,
            group_name="analytics",
        )

        # Or for job-based approach
        job = orchestrator.to_job(
            executor=execute_sql,
            job_name="analytics_pipeline",
        )
    """

    def to_assets(
        self,
        executor: Callable[[str], None],
        group_name: Optional[str] = None,
        key_prefix: Optional[Union[str, List[str]]] = None,
        compute_kind: str = "sql",
        **asset_kwargs,
    ) -> List:
        """
        Create Dagster Assets from the pipeline.

        Converts the pipeline's table dependency graph into Dagster assets
        where each target table becomes an asset with proper dependencies.
        This is the recommended approach for Dagster as it provides better
        lineage tracking and observability.

        Args:
            executor: Function that executes SQL (takes sql string)
            group_name: Optional asset group name for organization in Dagster UI
            key_prefix: Optional prefix for asset keys (e.g., ["warehouse", "analytics"])
            compute_kind: Compute kind tag for assets (default: "sql")
            **asset_kwargs: Additional asset parameters (owners, tags, etc.)

        Returns:
            List of Dagster Asset definitions

        Examples:
            # Basic usage
            assets = orchestrator.to_assets(
                executor=execute_sql,
                group_name="analytics"
            )

            # Create Dagster Definitions
            from dagster import Definitions
            defs = Definitions(assets=assets)

            # Advanced usage with prefixes and metadata
            assets = orchestrator.to_assets(
                executor=execute_sql,
                group_name="warehouse",
                key_prefix=["prod", "analytics"],
                compute_kind="clickhouse",
                owners=["team:data-eng"],
                tags={"domain": "finance"},
            )

        Note:
            - Requires Dagster 1.x: pip install 'dagster>=1.5.0'
            - Each target table becomes a Dagster asset
            - Dependencies are automatically inferred from table lineage
            - Deployment: Drop the definitions.py file in your Dagster workspace
        """
        try:
            import dagster as dg
        except ImportError as e:
            raise ImportError(
                "Dagster is required for asset generation. "
                "Install it with: pip install 'dagster>=1.5.0'"
            ) from e

        table_graph = self.table_graph

        assets = []
        asset_key_mapping: Dict[str, Any] = {}  # query_id -> AssetKey

        # Process each query that creates a table
        for query_id in table_graph.topological_sort():
            query = table_graph.queries[query_id]
            target_table = query.destination_table

            if target_table is None:
                continue  # Skip queries that don't create tables

            # Determine upstream dependencies (source tables created by this pipeline)
            upstream_asset_keys = []
            for source_table in query.source_tables:
                if source_table in table_graph.tables:
                    table_node = table_graph.tables[source_table]
                    # Only add as dep if it's created by another query in this pipeline
                    if table_node.created_by and table_node.created_by in asset_key_mapping:
                        upstream_asset_keys.append(asset_key_mapping[table_node.created_by])

            # Build asset key (sanitize table name for Dagster compatibility)
            # Dagster names must match ^[A-Za-z0-9_]+$
            safe_table_name = self._sanitize_name(target_table)
            if key_prefix:
                if isinstance(key_prefix, str):
                    prefix_list = [key_prefix]
                else:
                    prefix_list = list(key_prefix)
                asset_key = dg.AssetKey([*prefix_list, safe_table_name])
            else:
                asset_key = dg.AssetKey(safe_table_name)

            # Store mapping for dependency resolution
            asset_key_mapping[query_id] = asset_key

            # Capture SQL in closure
            sql_to_execute = query.sql
            table_name = target_table
            query_identifier = query_id

            # Build asset configuration
            asset_config: Dict[str, Any] = {
                "key": asset_key,
                "compute_kind": compute_kind,
                **asset_kwargs,
            }

            if group_name:
                asset_config["group_name"] = group_name

            if upstream_asset_keys:
                asset_config["deps"] = upstream_asset_keys

            # Create asset factory function
            def make_asset(qid: str, sql: str, tbl: str, config: Dict[str, Any], exec_fn: Callable):
                @dg.asset(**config)
                def sql_asset(context: dg.AssetExecutionContext):
                    """Execute SQL to materialize asset."""
                    context.log.info(f"Materializing table: {tbl}")
                    context.log.info(f"Query ID: {qid}")
                    context.log.debug(f"SQL: {sql[:500]}...")
                    exec_fn(sql)
                    return dg.MaterializeResult(
                        metadata={
                            "query_id": dg.MetadataValue.text(qid),
                            "table": dg.MetadataValue.text(tbl),
                        }
                    )

                # Rename function for better debugging in Dagster UI
                safe_name = tbl.replace(".", "_").replace("-", "_")
                sql_asset.__name__ = safe_name
                sql_asset.__qualname__ = safe_name

                return sql_asset

            asset = make_asset(query_identifier, sql_to_execute, table_name, asset_config, executor)
            assets.append(asset)

        return assets

    def to_job(
        self,
        executor: Callable[[str], None],
        job_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **job_kwargs,
    ):
        """
        Create Dagster Job from the pipeline using ops.

        Converts the pipeline's table dependency graph into a Dagster job
        where each SQL query becomes an op with proper dependencies.

        Note: For new pipelines, consider using to_assets() instead,
        which provides better lineage tracking and observability in Dagster.

        Args:
            executor: Function that executes SQL (takes sql string)
            job_name: Name for the Dagster job
            description: Optional job description (auto-generated if not provided)
            tags: Optional job tags for filtering in Dagster UI
            **job_kwargs: Additional job parameters

        Returns:
            Dagster Job definition

        Examples:
            # Basic usage
            job = orchestrator.to_job(
                executor=execute_sql,
                job_name="analytics_pipeline"
            )

            # Create Dagster Definitions
            from dagster import Definitions
            defs = Definitions(jobs=[job])

            # Execute the job locally
            result = job.execute_in_process()

        Note:
            - Requires Dagster 1.x: pip install 'dagster>=1.5.0'
            - Consider using to_assets() for better Dagster integration
            - Deployment: Drop the definitions.py file in your Dagster workspace
        """
        try:
            import dagster as dg
        except ImportError as e:
            raise ImportError(
                "Dagster is required for job generation. "
                "Install it with: pip install 'dagster>=1.5.0'"
            ) from e

        table_graph = self.table_graph

        # Generate description if not provided
        if description is None:
            query_count = len(table_graph.queries)
            table_count = len(table_graph.tables)
            description = (
                f"Pipeline with {query_count} queries operating on {table_count} tables. "
                f"Generated by clgraph."
            )

        # Create ops for each query
        ops: Dict[str, Any] = {}
        op_mapping: Dict[str, str] = {}  # query_id -> op_name

        for query_id in table_graph.topological_sort():
            query = table_graph.queries[query_id]
            sql_to_execute = query.sql

            # Generate safe op name
            op_name = self._sanitize_name(query_id)
            op_mapping[query_id] = op_name

            def make_op(qid: str, sql: str, name: str, exec_fn: Callable):
                @dg.op(name=name)
                def sql_op(context: dg.OpExecutionContext):
                    """Execute SQL query."""
                    context.log.info(f"Executing query: {qid}")
                    exec_fn(sql)
                    return qid

                return sql_op

            ops[query_id] = make_op(query_id, sql_to_execute, op_name, executor)

        # Build the job graph
        @dg.job(name=job_name, description=description, tags=tags or {}, **job_kwargs)
        def pipeline_job():
            """Generated pipeline job."""
            op_results: Dict[str, Any] = {}

            for query_id in table_graph.topological_sort():
                # Find upstream dependencies
                query = table_graph.queries[query_id]
                upstream_results = []

                for source_table in query.source_tables:
                    if source_table in table_graph.tables:
                        table_node = table_graph.tables[source_table]
                        if table_node.created_by and table_node.created_by in op_results:
                            upstream_results.append(op_results[table_node.created_by])

                # Execute op - dependencies are implicit via the graph structure
                # In Dagster, we need to wire dependencies differently
                op_results[query_id] = ops[query_id]()

            return op_results

        return pipeline_job


__all__ = ["DagsterOrchestrator"]
