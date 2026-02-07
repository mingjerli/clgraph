"""FROM clause parsing helper.

Handles parsing of FROM clause sources including tables, subqueries,
and delegates to JoinHandler and SpecialSourcesHandler for complex cases.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import QueryUnit
    from ..query_parser import RecursiveQueryParser


class FromClauseParser:
    """Helper for parsing FROM clause sources."""

    def __init__(self, parser: "RecursiveQueryParser"):
        """Initialize with reference to parent parser.

        Args:
            parser: The parent RecursiveQueryParser instance
        """
        self._parser = parser
        # Sub-helpers will be initialized after extraction
        self._join_handler = None
        self._special_sources = None

    def parse(self, from_node, parent_unit: "QueryUnit", depth: int):
        """Parse all sources in FROM clause.

        Args:
            from_node: The FROM expression node
            parent_unit: The parent QueryUnit
            depth: Current recursion depth
        """
        # Will be implemented during extraction
        raise NotImplementedError("FromClauseParser.parse not yet implemented")
