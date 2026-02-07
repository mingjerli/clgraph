"""Special source handling helper.

Handles parsing of special sources like UNNEST, Table-Valued Functions (TVFs),
VALUES clauses, and LATERAL subqueries.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import QueryUnit
    from ..query_parser import RecursiveQueryParser


class SpecialSourcesHandler:
    """Helper for handling special sources (UNNEST, TVF, VALUES, LATERAL)."""

    def __init__(self, parser: "RecursiveQueryParser"):
        """Initialize with reference to parent parser.

        Args:
            parser: The parent RecursiveQueryParser instance
        """
        self._parser = parser

    def is_unnest(self, source) -> bool:
        """Check if source is an UNNEST expression."""
        raise NotImplementedError("SpecialSourcesHandler.is_unnest not yet implemented")

    def is_tvf(self, source) -> bool:
        """Check if source is a Table-Valued Function."""
        raise NotImplementedError("SpecialSourcesHandler.is_tvf not yet implemented")

    def process_unnest(self, source, parent_unit: "QueryUnit", depth: int):
        """Process UNNEST source."""
        raise NotImplementedError("SpecialSourcesHandler.process_unnest not yet implemented")

    def process_tvf(self, source, parent_unit: "QueryUnit", depth: int):
        """Process Table-Valued Function source."""
        raise NotImplementedError("SpecialSourcesHandler.process_tvf not yet implemented")
