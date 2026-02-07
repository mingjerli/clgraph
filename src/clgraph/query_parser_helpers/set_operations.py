"""Set operations parsing helper.

Handles parsing of UNION, INTERSECT, and EXCEPT operations.
"""

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from ..models import QueryUnit
    from ..query_parser import RecursiveQueryParser


class SetOperationsParser:
    """Helper for parsing set operations (UNION, INTERSECT, EXCEPT)."""

    def __init__(self, parser: "RecursiveQueryParser"):
        """Initialize with reference to parent parser.

        Args:
            parser: The parent RecursiveQueryParser instance
        """
        self._parser = parser

    def parse(self, set_node, operation_type: str, name: str, parent_unit: "QueryUnit", depth: int):
        """Parse a set operation (UNION/INTERSECT/EXCEPT).

        Args:
            set_node: The set operation expression node
            operation_type: Type of operation ('union', 'intersect', 'except')
            name: Name for the operation unit
            parent_unit: The parent QueryUnit (or None for top-level)
            depth: Current recursion depth
        """
        raise NotImplementedError("SetOperationsParser.parse not yet implemented")

    def collect_branches(self, set_node) -> List[Tuple]:
        """Collect all branches of a set operation.

        Args:
            set_node: The set operation expression node

        Returns:
            List of (branch_node, distinct_flag) tuples
        """
        raise NotImplementedError("SetOperationsParser.collect_branches not yet implemented")
