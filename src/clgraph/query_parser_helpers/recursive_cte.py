"""Recursive CTE parsing helper.

Handles parsing of WITH RECURSIVE common table expressions.
"""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..models import QueryUnit
    from ..query_parser import RecursiveQueryParser


class RecursiveCTEParser:
    """Helper for parsing recursive CTEs."""

    def __init__(self, parser: "RecursiveQueryParser"):
        """Initialize with reference to parent parser.

        Args:
            parser: The parent RecursiveQueryParser instance
        """
        self._parser = parser

    def is_recursive_cte(self, query, cte_name: str) -> bool:
        """Check if a CTE is recursive.

        Args:
            query: The CTE query expression
            cte_name: Name of the CTE

        Returns:
            True if the CTE references itself
        """
        raise NotImplementedError("RecursiveCTEParser.is_recursive_cte not yet implemented")

    def parse(self, cte, cte_name: str, parent_unit: Optional["QueryUnit"], depth: int):
        """Parse a recursive CTE.

        Args:
            cte: The CTE expression node
            cte_name: Name of the CTE
            parent_unit: The parent QueryUnit (or None)
            depth: Current recursion depth
        """
        raise NotImplementedError("RecursiveCTEParser.parse not yet implemented")

    def find_self_reference(self, recursive_part, cte_name: str) -> Optional[str]:
        """Find the self-reference table alias in recursive part.

        Args:
            recursive_part: The recursive part of the CTE
            cte_name: Name of the CTE

        Returns:
            Alias of the self-reference, or None if not found
        """
        raise NotImplementedError("RecursiveCTEParser.find_self_reference not yet implemented")

    def find_join_condition_for_alias(self, query, alias: str) -> Optional[str]:
        """Find the join condition for a given alias.

        Args:
            query: The SELECT expression
            alias: The alias to find condition for

        Returns:
            String representation of the join condition, or None
        """
        raise NotImplementedError(
            "RecursiveCTEParser.find_join_condition_for_alias not yet implemented"
        )

    def extract_select_column_names(self, query) -> List[str]:
        """Extract column names from SELECT clause.

        Args:
            query: The SELECT expression

        Returns:
            List of column names
        """
        raise NotImplementedError(
            "RecursiveCTEParser.extract_select_column_names not yet implemented"
        )
