"""Subquery parsing helper.

Handles parsing of subqueries in WHERE, HAVING, SELECT, and QUALIFY clauses.
"""

from typing import TYPE_CHECKING, List

from sqlglot import exp

from ..models import QueryUnitType

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

    def parse_where_subqueries(
        self, where_node: exp.Expression, parent_unit: "QueryUnit", depth: int
    ):
        """Parse subqueries in WHERE clause.

        Args:
            where_node: The WHERE expression node
            parent_unit: The parent QueryUnit
            depth: Current recursion depth
        """
        for node in where_node.walk():
            if isinstance(node, exp.Subquery):
                subquery_select = node.this
                if isinstance(subquery_select, exp.Select):
                    subquery_name = f"where_subq_{self._parser.subquery_counter}"
                    self._parser.subquery_counter += 1

                    # Recursively parse
                    subquery_unit = self._parser._parse_select_unit(
                        select_node=subquery_select,
                        unit_type=QueryUnitType.SUBQUERY_WHERE,
                        name=subquery_name,
                        parent_unit=parent_unit,
                        depth=depth + 1,
                    )

                    parent_unit.depends_on_units.append(subquery_unit.unit_id)

    def parse_having_subqueries(
        self, having_node: exp.Expression, parent_unit: "QueryUnit", depth: int
    ):
        """Parse subqueries in HAVING clause.

        Args:
            having_node: The HAVING expression node
            parent_unit: The parent QueryUnit
            depth: Current recursion depth
        """
        for node in having_node.walk():
            if isinstance(node, exp.Subquery):
                subquery_select = node.this
                if isinstance(subquery_select, exp.Select):
                    subquery_name = f"having_subq_{self._parser.subquery_counter}"
                    self._parser.subquery_counter += 1

                    # Recursively parse
                    subquery_unit = self._parser._parse_select_unit(
                        select_node=subquery_select,
                        unit_type=QueryUnitType.SUBQUERY_HAVING,
                        name=subquery_name,
                        parent_unit=parent_unit,
                        depth=depth + 1,
                    )

                    parent_unit.depends_on_units.append(subquery_unit.unit_id)

    def parse_select_subqueries(self, expr: exp.Expression, parent_unit: "QueryUnit", depth: int):
        """Parse scalar subqueries in SELECT clause.

        Args:
            expr: The SELECT expression
            parent_unit: The parent QueryUnit
            depth: Current recursion depth
        """
        for node in expr.walk():
            if isinstance(node, exp.Subquery):
                subquery_select = node.this
                if isinstance(subquery_select, exp.Select):
                    subquery_name = f"select_subq_{self._parser.subquery_counter}"
                    self._parser.subquery_counter += 1

                    # Recursively parse
                    subquery_unit = self._parser._parse_select_unit(
                        select_node=subquery_select,
                        unit_type=QueryUnitType.SUBQUERY_SELECT,
                        name=subquery_name,
                        parent_unit=parent_unit,
                        depth=depth + 1,
                    )

                    parent_unit.depends_on_units.append(subquery_unit.unit_id)

    def parse_qualify_clause(self, qualify_node: exp.Qualify, unit: "QueryUnit"):
        """Parse QUALIFY clause to extract window function column dependencies.

        QUALIFY filters rows based on window function results.
        Example:
            QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1

        This extracts:
        - condition: The full QUALIFY condition as SQL
        - partition_columns: Columns used in PARTITION BY
        - order_columns: Columns used in ORDER BY
        - window_functions: Names of window functions used

        Args:
            qualify_node: The QUALIFY expression node
            unit: The QueryUnit to add qualify information to
        """
        condition = qualify_node.this
        partition_columns: List[str] = []
        order_columns: List[str] = []
        window_functions: List[str] = []

        # Walk the condition to find window functions
        for node in condition.walk():
            if isinstance(node, exp.Window):
                # Get function name
                func = node.this
                # Try sql_name() first (works for ROW_NUMBER, RANK, etc.),
                # fall back to type name
                if hasattr(func, "sql_name"):
                    func_name = func.sql_name()
                elif hasattr(func, "name") and func.name:
                    func_name = func.name
                else:
                    func_name = type(func).__name__
                window_functions.append(func_name.upper())

                # Get PARTITION BY columns
                partition_by = node.args.get("partition_by")
                if partition_by:
                    for partition_expr in partition_by:
                        for col in partition_expr.find_all(exp.Column):
                            table_ref = str(col.table) if col.table else None
                            col_name = col.name
                            full_name = f"{table_ref}.{col_name}" if table_ref else col_name
                            if full_name not in partition_columns:
                                partition_columns.append(full_name)

                # Get ORDER BY columns
                order_by = node.args.get("order")
                if order_by and hasattr(order_by, "expressions"):
                    for order_expr in order_by.expressions:
                        expr_node = (
                            order_expr.this if isinstance(order_expr, exp.Ordered) else order_expr
                        )
                        for col in expr_node.find_all(exp.Column):
                            table_ref = str(col.table) if col.table else None
                            col_name = col.name
                            full_name = f"{table_ref}.{col_name}" if table_ref else col_name
                            if full_name not in order_columns:
                                order_columns.append(full_name)

        # Store QUALIFY info on the unit
        unit.qualify_info = {
            "condition": condition.sql(),
            "partition_columns": partition_columns,
            "order_columns": order_columns,
            "window_functions": window_functions,
        }
