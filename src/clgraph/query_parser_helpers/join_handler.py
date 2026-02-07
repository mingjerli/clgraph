"""JOIN clause handling helper.

Handles parsing of JOIN clauses including INNER, LEFT, RIGHT, FULL,
CROSS joins and their conditions.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import QueryUnit
    from ..query_parser import RecursiveQueryParser


class JoinHandler:
    """Helper for handling JOIN clauses."""

    def __init__(self, parser: "RecursiveQueryParser"):
        """Initialize with reference to parent parser.

        Args:
            parser: The parent RecursiveQueryParser instance
        """
        self._parser = parser

    def process_joins(self, from_node, parent_unit: "QueryUnit", depth: int):
        """Process JOIN clauses in FROM node.

        Args:
            from_node: The FROM expression node
            parent_unit: The parent QueryUnit
            depth: Current recursion depth
        """
        # Will be implemented during extraction
        raise NotImplementedError("JoinHandler.process_joins not yet implemented")
