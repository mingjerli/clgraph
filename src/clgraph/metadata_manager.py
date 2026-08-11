"""
Metadata management component for Pipeline.

This module provides the MetadataManager class which contains all metadata
management logic extracted from the Pipeline class.

The MetadataManager handles:
- LLM-powered description generation
- Metadata propagation (PII, owner, tags)
- Governance queries (get PII columns, get by owner, get by tag)
"""

import logging
from typing import TYPE_CHECKING, List

from .column import (
    generate_description,
    propagate_metadata,
    propagate_metadata_backward,
)
from .models import ColumnNode, DescriptionSource

if TYPE_CHECKING:
    from .pipeline import Pipeline

logger = logging.getLogger(__name__)


def needs_description(col: ColumnNode) -> bool:
    """True when a column has no description or only a rule-based placeholder."""
    return not col.description or col.description_source == DescriptionSource.FALLBACK


def target_table(query) -> str:
    """The table a query's output columns live under (shared by both bulk ops)."""
    return query.destination_table or f"{query.query_id}_result"


class MetadataManager:
    """
    Metadata management for Pipeline.

    This class is extracted from Pipeline to follow the Single Responsibility
    Principle. It contains all metadata management methods that operate on
    the Pipeline's columns.

    The manager is lazily initialized by Pipeline when first needed.

    Example (via Pipeline - recommended):
        pipeline = Pipeline(queries, dialect="bigquery")
        pii_cols = pipeline.get_pii_columns()

    Example (direct usage - advanced):
        from clgraph.metadata_manager import MetadataManager

        manager = MetadataManager(pipeline)
        pii_cols = manager.get_pii_columns()
    """

    def __init__(self, pipeline: "Pipeline"):
        """
        Initialize MetadataManager with a Pipeline reference.

        Args:
            pipeline: The Pipeline instance to manage metadata for.
        """
        self._pipeline = pipeline

    def generate_all_descriptions(
        self,
        batch_size: int = 10,
        verbose: bool = True,
        *,
        overwrite: bool = False,
        on_error: str = "fallback",
        include_sources: bool = False,
    ):
        """
        Generate descriptions for all columns using LLM.

        Processes columns in topological order (sources first).

        Args:
            batch_size: Number of columns per batch (currently processes sequentially)
            verbose: If True, print progress messages
            overwrite: By default only columns that have no description yet are
                processed. Pass ``True`` to also re-describe columns that already
                have one, including descriptions authored as SQL comments.
            on_error: ``"fallback"`` (default) writes a rule-based description
                when the LLM fails; ``"raise"`` propagates
                :class:`~clgraph.column.DescriptionGenerationError` instead. Use
                ``"raise"`` when a silent fallback would be mistaken for a real
                model-generated description.
            include_sources: If ``True``, also describe columns of source tables
                (tables not produced by any query in the pipeline), using their
                forward usage and sibling columns as context. Defaults to
                ``False`` since source columns have no lineage-derived context.
        """
        if not self._pipeline.llm:
            raise ValueError("LLM not configured. Set pipeline.llm before calling.")

        # Get columns in topological order
        sorted_query_ids = self._pipeline.table_graph.topological_sort()

        columns_to_process = []
        if include_sources:
            for table_name, node in self._pipeline.table_graph.tables.items():
                if not node.is_source:
                    continue
                by_column = {}
                for col in self._pipeline.columns.values():
                    if col.table_name == table_name and (overwrite or needs_description(col)):
                        by_column.setdefault(col.column_name, []).append(col)
                for _name, nodes in sorted(by_column.items()):
                    representative = max(
                        nodes,
                        key=lambda c: len(self._pipeline._get_outgoing_edges(c.full_name)),
                    )
                    columns_to_process.append(representative)

        for query_id in sorted_query_ids:
            query = self._pipeline.table_graph.queries[query_id]
            table = target_table(query)
            for col in self._pipeline.columns.values():
                if (
                    col.table_name == table
                    and (overwrite or needs_description(col))
                    and col.is_computed()
                ):
                    columns_to_process.append(col)

        logger.info("Generating descriptions for %d columns...", len(columns_to_process))

        # Process columns
        for i, col in enumerate(columns_to_process):
            if (i + 1) % batch_size == 0:
                logger.info("Processed %d/%d columns...", i + 1, len(columns_to_process))

            generate_description(
                col,
                self._pipeline.llm,
                self._pipeline,
                overwrite=overwrite,
                on_error=on_error,
            )

            # The same physical column may appear as several ColumnNodes (one
            # per consuming query); copy the result to every twin so all
            # consumers see it. The guard mirrors the candidate-filter
            # semantics above: only touch twins that overwrite allows or that
            # still need a description, so an adequate GENERATED twin is left
            # alone without overwrite, and overwrite=True can replace even a
            # SOURCE twin's stale text instead of leaving nodes disagreeing.
            if col.description:
                for twin in self._pipeline.columns.values():
                    if (
                        twin is not col
                        and twin.table_name == col.table_name
                        and twin.column_name == col.column_name
                        and (overwrite or needs_description(twin))
                    ):
                        twin.description = col.description
                        twin.description_source = col.description_source

        logger.info("Done! Generated %d descriptions", len(columns_to_process))

    def propagate_all_metadata(self, verbose: bool = True):
        """
        Propagate metadata (owner, PII, tags) through lineage.

        Uses a two-pass approach:
        1. Backward pass: Propagate metadata from output columns (with SQL comment
           metadata) to their input layer sources. This ensures that if an output
           column has PII from a comment, the source column also gets PII.
        2. Forward pass: Propagate metadata from source columns to downstream
           columns in topological order.

        Args:
            verbose: If True, print progress messages
        """
        # Get columns in topological order
        sorted_query_ids = self._pipeline.table_graph.topological_sort()

        # Pass 1: Backward propagation from output columns to input columns
        # This handles metadata set via SQL comments on output columns
        output_columns = [col for col in self._pipeline.columns.values() if col.layer == "output"]

        logger.info(
            "Pass 1: Propagating metadata backward from %d output columns...",
            len(output_columns),
        )

        for col in output_columns:
            propagate_metadata_backward(col, self._pipeline)

        # Pass 2: Forward propagation through lineage
        # Process all computed columns (output columns from each query)
        columns_to_process = []
        for query_id in sorted_query_ids:
            query = self._pipeline.table_graph.queries[query_id]
            table = target_table(query)
            for col in self._pipeline.columns.values():
                if col.table_name == table and col.is_computed():
                    columns_to_process.append(col)

        logger.info(
            "Pass 2: Propagating metadata forward for %d columns...",
            len(columns_to_process),
        )

        # Process columns
        for col in columns_to_process:
            propagate_metadata(col, self._pipeline)

        logger.info("Done! Propagated metadata for %d columns", len(columns_to_process))

    def get_pii_columns(self) -> List[ColumnNode]:
        """
        Get all columns marked as PII.

        Returns:
            List of columns where pii == True
        """
        return [col for col in self._pipeline.columns.values() if col.pii]

    def get_columns_by_owner(self, owner: str) -> List[ColumnNode]:
        """
        Get all columns with a specific owner.

        Args:
            owner: Owner name to filter by

        Returns:
            List of columns with matching owner
        """
        return [col for col in self._pipeline.columns.values() if col.owner == owner]

    def get_columns_by_tag(self, tag: str) -> List[ColumnNode]:
        """
        Get all columns containing a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of columns containing the tag
        """
        return [col for col in self._pipeline.columns.values() if tag in col.tags]


__all__ = ["MetadataManager", "target_table", "needs_description"]
