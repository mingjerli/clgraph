"""Grouping clause parsing helper.

Handles parsing of GROUP BY including GROUPING SETS, CUBE, and ROLLUP.
"""

from itertools import combinations
from typing import TYPE_CHECKING, List

from sqlglot import exp

if TYPE_CHECKING:
    from ..models import QueryUnit
    from ..query_parser import RecursiveQueryParser


class GroupingParser:
    """Helper for parsing GROUP BY and grouping sets."""

    def __init__(self, parser: "RecursiveQueryParser"):
        """Initialize with reference to parent parser.

        Args:
            parser: The parent RecursiveQueryParser instance
        """
        self._parser = parser

    def parse_grouping_sets(self, group_clause: exp.Group, unit: "QueryUnit"):
        """Parse GROUP BY clause for GROUPING SETS, CUBE, and ROLLUP constructs.

        These constructs generate multiple grouping levels in a single query:
        - CUBE(a, b): All combinations: (a,b), (a), (b), ()
        - ROLLUP(a, b): Hierarchical: (a,b), (a), ()
        - GROUPING SETS(...): Explicit list of grouping combinations

        Args:
            group_clause: The GROUP BY expression node
            unit: The QueryUnit to add grouping information to
        """
        # Check for CUBE
        cube_list = group_clause.args.get("cube", [])
        if cube_list:
            for cube_node in cube_list:
                if isinstance(cube_node, exp.Cube):
                    columns = self.extract_grouping_columns(cube_node.expressions)
                    # CUBE generates all 2^n combinations
                    grouping_sets = self.expand_cube(columns)
                    unit.grouping_config = {
                        "grouping_type": "cube",
                        "grouping_columns": columns,
                        "grouping_sets": grouping_sets,
                    }
                    return

        # Check for ROLLUP
        rollup_list = group_clause.args.get("rollup", [])
        if rollup_list:
            for rollup_node in rollup_list:
                if isinstance(rollup_node, exp.Rollup):
                    columns = self.extract_grouping_columns(rollup_node.expressions)
                    # ROLLUP generates n+1 hierarchical combinations
                    grouping_sets = self.expand_rollup(columns)
                    unit.grouping_config = {
                        "grouping_type": "rollup",
                        "grouping_columns": columns,
                        "grouping_sets": grouping_sets,
                    }
                    return

        # Check for GROUPING SETS
        gs_list = group_clause.args.get("grouping_sets", [])
        if gs_list:
            for gs_node in gs_list:
                if isinstance(gs_node, exp.GroupingSets):
                    grouping_sets = []
                    columns_set: set = set()
                    for set_expr in gs_node.expressions:
                        if isinstance(set_expr, exp.Tuple):
                            # Tuple: (a, b)
                            cols = self.extract_grouping_columns(set_expr.expressions)
                            grouping_sets.append(cols)
                            columns_set.update(cols)
                        elif isinstance(set_expr, exp.Paren):
                            # Single column: (a)
                            cols = self.extract_grouping_columns([set_expr.this])
                            grouping_sets.append(cols)
                            columns_set.update(cols)
                        else:
                            # Could be empty () for grand total
                            grouping_sets.append([])
                    unit.grouping_config = {
                        "grouping_type": "grouping_sets",
                        "grouping_columns": list(columns_set),
                        "grouping_sets": grouping_sets,
                    }
                    return

    def extract_grouping_columns(self, expressions: List[exp.Expression]) -> List[str]:
        """Extract column names from a list of expressions.

        Args:
            expressions: List of grouping expressions

        Returns:
            List of column names (qualified if table prefix is present)
        """
        columns = []
        for expr in expressions:
            if isinstance(expr, exp.Column):
                table_ref = str(expr.table) if expr.table else None
                col_name = expr.name
                full_name = f"{table_ref}.{col_name}" if table_ref else col_name
                if full_name not in columns:
                    columns.append(full_name)
            else:
                # Walk nested expressions for columns
                for col in expr.find_all(exp.Column):
                    table_ref = str(col.table) if col.table else None
                    col_name = col.name
                    full_name = f"{table_ref}.{col_name}" if table_ref else col_name
                    if full_name not in columns:
                        columns.append(full_name)
        return columns

    def expand_cube(self, columns: List[str]) -> List[List[str]]:
        """Expand CUBE into all 2^n combinations.

        Args:
            columns: List of column names in CUBE

        Returns:
            List of grouping sets (each is a list of column names)
        """
        result = []
        n = len(columns)
        # Generate all subsets from full set to empty set
        for r in range(n, -1, -1):
            for combo in combinations(columns, r):
                result.append(list(combo))
        return result

    def expand_rollup(self, columns: List[str]) -> List[List[str]]:
        """Expand ROLLUP into hierarchical combinations.

        Args:
            columns: List of column names in ROLLUP

        Returns:
            List of grouping sets (each is a list of column names)
        """
        result = []
        # From full set down to empty set hierarchically
        for i in range(len(columns), -1, -1):
            result.append(columns[:i])
        return result
