"""
Pipeline execution module for clgraph.

Provides synchronous and asynchronous execution of SQL pipelines
with concurrent execution within dependency levels.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .pipeline import Pipeline


class PipelineExecutor:
    """
    Executes clgraph pipelines with concurrent execution support.

    Provides both synchronous and asynchronous execution modes,
    with automatic parallelization within dependency levels.

    Example:
        from clgraph.execution import PipelineExecutor

        executor = PipelineExecutor(pipeline)
        result = executor.run(execute_sql, max_workers=4)

        # Or async
        result = await executor.async_run(async_execute_sql, max_workers=4)
    """

    def __init__(self, pipeline: "Pipeline") -> None:
        """
        Initialize executor with a Pipeline instance.

        Args:
            pipeline: The clgraph Pipeline to execute
        """
        self.pipeline = pipeline
        self.table_graph = pipeline.table_graph

    def get_execution_levels(self) -> List[List[str]]:
        """
        Group queries into levels for concurrent execution.

        Level 0: Queries with no dependencies
        Level 1: Queries that depend only on Level 0
        Level 2: Queries that depend on Level 0 or 1
        etc.

        Queries in the same level can run concurrently.

        Returns:
            List of levels, where each level is a list of query IDs
        """
        levels = []
        completed = set()

        while len(completed) < len(self.table_graph.queries):
            current_level = []

            for query_id, query in self.table_graph.queries.items():
                if query_id in completed:
                    continue

                # Check if all dependencies are completed
                dependencies_met = True
                for source_table in query.source_tables:
                    # Find query that creates this table
                    table_node = self.table_graph.tables.get(source_table)
                    if table_node and table_node.created_by:
                        if table_node.created_by not in completed:
                            dependencies_met = False
                            break

                if dependencies_met:
                    current_level.append(query_id)

            if not current_level:
                # No progress - circular dependency
                raise RuntimeError("Circular dependency detected in pipeline")

            levels.append(current_level)
            completed.update(current_level)

        return levels

    def run(
        self,
        executor: Callable[[str], None],
        max_workers: int = 4,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute pipeline synchronously with concurrent execution.

        Args:
            executor: Function that executes SQL (takes sql string)
            max_workers: Max concurrent workers (default: 4)
            verbose: Print progress (default: True)

        Returns:
            dict with execution results: {
                "completed": list of completed query IDs,
                "failed": list of (query_id, error) tuples,
                "elapsed_seconds": total execution time,
                "total_queries": total number of queries
            }

        Example:
            def execute_sql(sql: str):
                import duckdb
                conn = duckdb.connect()
                conn.execute(sql)

            result = executor.run(execute_sql, max_workers=4)
            print(f"Completed {len(result['completed'])} queries")
        """
        logger.info("Starting pipeline execution (%d queries)", len(self.table_graph.queries))

        # Track completed queries
        completed = set()
        failed: List[Tuple[str, str]] = []
        start_time = time.time()

        # Group queries by level for concurrent execution
        levels = self.get_execution_levels()

        # Execute level by level
        for level_num, level_queries in enumerate(levels, 1):
            logger.info("Level %d: %d queries", level_num, len(level_queries))

            # Execute queries in this level concurrently
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {}

                for query_id in level_queries:
                    query = self.table_graph.queries[query_id]
                    future = pool.submit(executor, query.sql)
                    futures[future] = query_id

                # Wait for completion
                for future in as_completed(futures):
                    query_id = futures[future]

                    try:
                        future.result()
                        completed.add(query_id)
                        logger.info("Completed: %s", query_id)
                    except Exception as e:
                        failed.append((query_id, str(e)))
                        logger.debug("Query %s execution failed", query_id, exc_info=True)
                        logger.warning("Failed: %s: %s", query_id, e)

        elapsed = time.time() - start_time

        # Summary
        logger.info("Pipeline completed in %.2fs", elapsed)
        logger.info("Successful: %d", len(completed))
        logger.info("Failed: %d", len(failed))
        if failed:
            for query_id, error in failed:
                logger.warning("Failed query - %s: %s", query_id, error)

        return {
            "completed": list(completed),
            "failed": failed,
            "elapsed_seconds": elapsed,
            "total_queries": len(self.table_graph.queries),
        }

    async def async_run(
        self,
        executor: Callable[[str], Awaitable[None]],
        max_workers: int = 4,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute pipeline asynchronously with concurrent execution.

        Args:
            executor: Async function that executes SQL (takes sql string)
            max_workers: Max concurrent workers (controls semaphore, default: 4)
            verbose: Print progress (default: True)

        Returns:
            dict with execution results: {
                "completed": list of completed query IDs,
                "failed": list of (query_id, error) tuples,
                "elapsed_seconds": total execution time,
                "total_queries": total number of queries
            }

        Example:
            async def execute_sql(sql: str):
                # Your async database connection
                await async_conn.execute(sql)

            result = await executor.async_run(execute_sql, max_workers=4)
            print(f"Completed {len(result['completed'])} queries")
        """
        logger.info("Starting async pipeline execution (%d queries)", len(self.table_graph.queries))

        # Track completed queries
        completed = set()
        failed: List[Tuple[str, str]] = []
        start_time = time.time()

        # Group queries by level for concurrent execution
        levels = self.get_execution_levels()

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_workers)

        # Execute level by level
        for level_num, level_queries in enumerate(levels, 1):
            logger.info("Level %d: %d queries", level_num, len(level_queries))

            async def execute_with_semaphore(query_id: str, sql: str):
                """Execute query with semaphore for concurrency control"""
                async with semaphore:
                    try:
                        await executor(sql)
                        completed.add(query_id)
                        logger.info("Completed: %s", query_id)
                    except Exception as e:
                        failed.append((query_id, str(e)))
                        logger.debug("Async query %s execution failed", query_id, exc_info=True)
                        logger.warning("Failed: %s: %s", query_id, e)

            # Execute queries in this level concurrently
            tasks = []
            for query_id in level_queries:
                query = self.table_graph.queries[query_id]
                task = execute_with_semaphore(query_id, query.sql)
                tasks.append(task)

            # Wait for all tasks in this level to complete
            await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        # Summary
        logger.info("Pipeline completed in %.2fs", elapsed)
        logger.info("Successful: %d", len(completed))
        logger.info("Failed: %d", len(failed))
        if failed:
            for query_id, error in failed:
                logger.warning("Failed query - %s: %s", query_id, error)

        return {
            "completed": list(completed),
            "failed": failed,
            "elapsed_seconds": elapsed,
            "total_queries": len(self.table_graph.queries),
        }


__all__ = ["PipelineExecutor"]
