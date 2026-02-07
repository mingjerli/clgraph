"""MERGE statement parsing helper.

Handles parsing of MERGE INTO statements including WHEN MATCHED/NOT MATCHED clauses.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..query_parser import RecursiveQueryParser


class MergeParser:
    """Helper for parsing MERGE INTO statements."""

    def __init__(self, parser: "RecursiveQueryParser"):
        """Initialize with reference to parent parser.

        Args:
            parser: The parent RecursiveQueryParser instance
        """
        self._parser = parser

    def parse(self, merge_node, name: str, depth: int):
        """Parse a MERGE INTO statement.

        Args:
            merge_node: The MERGE expression node
            name: Name for the merge unit
            depth: Current recursion depth
        """
        raise NotImplementedError("MergeParser.parse not yet implemented")
