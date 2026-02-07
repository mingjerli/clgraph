"""Set operations parsing helper.

Handles parsing of UNION, INTERSECT, and EXCEPT operations.
"""

from typing import TYPE_CHECKING, List, Optional, Union

from sqlglot import exp

from ..models import QueryUnit, QueryUnitType

if TYPE_CHECKING:
    from ..query_parser import RecursiveQueryParser


class SetOperationsParser:
    """Helper for parsing set operations (UNION, INTERSECT, EXCEPT)."""

    def __init__(self, parser: "RecursiveQueryParser"):
        """Initialize with reference to parent parser.

        Args:
            parser: The parent RecursiveQueryParser instance
        """
        self._parser = parser

    def parse(
        self,
        set_node: Union[exp.Union, exp.Intersect, exp.Except],
        operation_type: str,
        name: str,
        parent_unit: Optional[QueryUnit] = None,
        depth: int = 0,
    ) -> QueryUnit:
        """Parse UNION/INTERSECT/EXCEPT set operations.

        Set operations combine results from multiple SELECT statements.
        Each branch is parsed as a separate query unit.

        Args:
            set_node: The set operation node (Union, Intersect, or Except)
            operation_type: Type of operation ("union", "intersect", "except")
            name: Name for this set operation unit
            parent_unit: Parent query unit (if nested)
            depth: Nesting depth

        Returns:
            QueryUnit representing the set operation

        sqlglot Structure:
            - Union.this = left SELECT
            - Union.expression = right SELECT
            - Union.distinct = True if UNION (not UNION ALL)
        """
        # Determine unit type based on operation
        unit_type_map = {
            "union": QueryUnitType.UNION,
            "intersect": QueryUnitType.INTERSECT,
            "except": QueryUnitType.EXCEPT,
        }
        unit_type = unit_type_map[operation_type]

        # Determine specific operation variant (e.g., UNION vs UNION ALL)
        if operation_type == "union":
            # Check if DISTINCT is explicitly set (UNION DISTINCT vs UNION ALL)
            is_distinct = set_node.args.get("distinct", False)
            set_op_variant = "union" if is_distinct else "union_all"
        else:
            set_op_variant = operation_type

        # Create unit for the set operation itself
        unit_id = self._parser._generate_unit_id(unit_type, name)
        unit = QueryUnit(
            unit_id=unit_id,
            unit_type=unit_type,
            name=name,
            select_node=None,  # Set operations don't have a select_node
            parent_unit=parent_unit,
            depth=depth,
            set_operation_type=set_op_variant,
            set_operation_branches=[],
        )

        # Collect all SELECT branches (handles nested set operations)
        branches = self.collect_branches(set_node, operation_type)

        # Parse each branch as a separate query unit
        for idx, branch_select in enumerate(branches):
            branch_name = f"{name}_branch_{idx}"
            branch_unit = self._parser._parse_select_unit(
                select_node=branch_select,
                unit_type=QueryUnitType.SUBQUERY_UNION,
                name=branch_name,
                parent_unit=unit,
                depth=depth + 1,
            )

            # Track branch in set operation
            unit.set_operation_branches.append(branch_unit.unit_id)
            unit.depends_on_units.append(branch_unit.unit_id)

        # Add to graph
        self._parser.unit_graph.add_unit(unit)

        return unit

    def collect_branches(
        self,
        set_node: Union[exp.Union, exp.Intersect, exp.Except],
        operation_type: str,
    ) -> List[exp.Select]:
        """Recursively collect all SELECT branches from a set operation.

        Handles nested set operations by flattening them into a list.
        Example: (A UNION B) UNION C → [A, B, C]

        Args:
            set_node: The set operation node
            operation_type: Type of operation ("union", "intersect", "except")

        Returns:
            List of SELECT statements in the set operation
        """
        branches = []

        # Determine the node type we're collecting
        node_class_map = {
            "union": exp.Union,
            "intersect": exp.Intersect,
            "except": exp.Except,
        }
        target_class = node_class_map[operation_type]

        # Process left side (this)
        left_node = set_node.this
        # Handle parenthesized expressions wrapped in Subquery
        if isinstance(left_node, exp.Subquery):
            left_node = left_node.this

        if isinstance(left_node, target_class):
            # Nested set operation - recurse
            branches.extend(self.collect_branches(left_node, operation_type))
        elif isinstance(left_node, exp.Select):
            # Base case - SELECT statement
            branches.append(left_node)
        else:
            raise ValueError(f"Unexpected node type in set operation: {type(left_node).__name__}")

        # Process right side (expression)
        right_node = set_node.expression
        # Handle parenthesized expressions wrapped in Subquery
        if isinstance(right_node, exp.Subquery):
            right_node = right_node.this

        if isinstance(right_node, target_class):
            # Nested set operation - recurse
            branches.extend(self.collect_branches(right_node, operation_type))
        elif isinstance(right_node, exp.Select):
            # Base case - SELECT statement
            branches.append(right_node)
        else:
            raise ValueError(f"Unexpected node type in set operation: {type(right_node).__name__}")

        return branches
