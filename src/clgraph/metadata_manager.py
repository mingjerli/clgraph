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
from .models import ColumnNode

if TYPE_CHECKING:
    from .pipeline import Pipeline

logger = logging.getLogger(__name__)


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

    def generate_all_descriptions(self, batch_size: int = 10, verbose: bool = True):
        """
        Generate descriptions for all columns using LLM.

        Processes columns in topological order (sources first).

        Args:
            batch_size: Number of columns per batch (currently processes sequentially)
            verbose: If True, print progress messages
        """
        if not self._pipeline.llm:
            raise ValueError("LLM not configured. Set pipeline.llm before calling.")

        # Get columns in topological order
        sorted_query_ids = self._pipeline.table_graph.topological_sort()

        columns_to_process = []
        for query_id in sorted_query_ids:
            query = self._pipeline.table_graph.queries[query_id]
            if query.destination_table:
                for col in self._pipeline.columns.values():
                    if (
                        col.table_name == query.destination_table
                        and not col.description
                        and col.is_computed()
                    ):
                        columns_to_process.append(col)

        logger.info("Generating descriptions for %d columns...", len(columns_to_process))

        # Process columns
        for i, col in enumerate(columns_to_process):
            if (i + 1) % batch_size == 0:
                logger.info("Processed %d/%d columns...", i + 1, len(columns_to_process))

            generate_description(col, self._pipeline.llm, self._pipeline)

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
            # Get the table name for this query's output
            # For CREATE TABLE queries, use destination_table
            # For plain SELECTs, use query_id_result pattern
            target_table = query.destination_table or f"{query_id}_result"
            for col in self._pipeline.columns.values():
                if col.table_name == target_table and col.is_computed():
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


__all__ = ["MetadataManager"]
