"""
Base classes and protocols for orchestrator integrations.

This module defines the interface that all orchestrator integrations must follow.
"""

from typing import TYPE_CHECKING, List, Protocol

if TYPE_CHECKING:
    from ..pipeline import Pipeline


class OrchestratorProtocol(Protocol):
    """Protocol defining the interface for orchestrator integrations."""

    def __init__(self, pipeline: "Pipeline") -> None:
        """Initialize with a Pipeline instance."""
        ...


class BaseOrchestrator:
    """
    Base class for orchestrator integrations.

    Provides common functionality and enforces the interface for
    converting clgraph pipelines to orchestrator-specific formats.

    Subclasses should implement orchestrator-specific methods like:
    - to_dag() for Airflow
    - to_assets() / to_job() for Dagster
    - to_flow() for Prefect
    """

    def __init__(self, pipeline: "Pipeline") -> None:
        """
        Initialize orchestrator with a Pipeline instance.

        Args:
            pipeline: The clgraph Pipeline to convert
        """
        self.pipeline = pipeline
        self.table_graph = pipeline.table_graph

    def _get_execution_levels(self) -> List[List[str]]:
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

    def _sanitize_name(self, name: str) -> str:
        """
        Sanitize a name for use in orchestrator identifiers.

        Replaces dots and dashes with underscores to ensure compatibility
        with orchestrator naming requirements.

        Args:
            name: The name to sanitize

        Returns:
            Sanitized name safe for use as identifier
        """
        return name.replace(".", "_").replace("-", "_")


__all__ = ["BaseOrchestrator", "OrchestratorProtocol"]
