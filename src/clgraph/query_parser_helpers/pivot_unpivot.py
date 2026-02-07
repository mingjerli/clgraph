"""PIVOT and UNPIVOT parsing helper.

Handles parsing of PIVOT and UNPIVOT operations in SQL queries.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import QueryUnit
    from ..query_parser import RecursiveQueryParser


class PivotUnpivotParser:
    """Helper for parsing PIVOT and UNPIVOT operations."""

    def __init__(self, parser: "RecursiveQueryParser"):
        """Initialize with reference to parent parser.

        Args:
            parser: The parent RecursiveQueryParser instance
        """
        self._parser = parser

    def parse_pivot(
        self, pivot_node, parent_unit: "QueryUnit", parent_source_alias: str, depth: int
    ):
        """Parse a PIVOT operation.

        Args:
            pivot_node: The PIVOT expression node
            parent_unit: The parent QueryUnit
            parent_source_alias: Alias of the source being pivoted
            depth: Current recursion depth
        """
        raise NotImplementedError("PivotUnpivotParser.parse_pivot not yet implemented")

    def parse_unpivot(
        self, unpivot_node, parent_unit: "QueryUnit", parent_source_alias: str, depth: int
    ):
        """Parse an UNPIVOT operation.

        Args:
            unpivot_node: The UNPIVOT expression node
            parent_unit: The parent QueryUnit
            parent_source_alias: Alias of the source being unpivoted
            depth: Current recursion depth
        """
        raise NotImplementedError("PivotUnpivotParser.parse_unpivot not yet implemented")
