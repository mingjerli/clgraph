"""Subquery parsing helper.

Handles parsing of subqueries in WHERE, HAVING, SELECT, and QUALIFY clauses.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import QueryUnit
    from ..query_parser import RecursiveQueryParser


class SubqueryParser:
    """Helper for parsing subqueries in various clauses."""

    def __init__(self, parser: "RecursiveQueryParser"):
        """Initialize with reference to parent parser.

        Args:
            parser: The parent RecursiveQueryParser instance
        """
        self._parser = parser

    def parse_where_subqueries(self, where_node, parent_unit: "QueryUnit", depth: int):
        """Parse subqueries in WHERE clause.

        Args:
            where_node: The WHERE expression node
            parent_unit: The parent QueryUnit
            depth: Current recursion depth
        """
        raise NotImplementedError("SubqueryParser.parse_where_subqueries not yet implemented")

    def parse_having_subqueries(self, having_node, parent_unit: "QueryUnit", depth: int):
        """Parse subqueries in HAVING clause.

        Args:
            having_node: The HAVING expression node
            parent_unit: The parent QueryUnit
            depth: Current recursion depth
        """
        raise NotImplementedError("SubqueryParser.parse_having_subqueries not yet implemented")

    def parse_select_subqueries(self, expr, parent_unit: "QueryUnit", depth: int):
        """Parse subqueries in SELECT clause.

        Args:
            expr: The SELECT expression
            parent_unit: The parent QueryUnit
            depth: Current recursion depth
        """
        raise NotImplementedError("SubqueryParser.parse_select_subqueries not yet implemented")

    def parse_qualify_clause(self, qualify_node, unit: "QueryUnit"):
        """Parse QUALIFY clause for window function filtering.

        Args:
            qualify_node: The QUALIFY expression node
            unit: The QueryUnit to add qualify information to
        """
        raise NotImplementedError("SubqueryParser.parse_qualify_clause not yet implemented")
