"""PIVOT and UNPIVOT parsing helper.

Handles parsing of PIVOT and UNPIVOT operations in SQL queries.
"""

from typing import TYPE_CHECKING

from sqlglot import exp

from ..models import QueryUnit, QueryUnitType

if TYPE_CHECKING:
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
        self,
        pivot_node: exp.Pivot,
        name: str,
        parent_unit: QueryUnit,
        depth: int,
        table_node,  # Can be exp.Table or exp.Subquery
    ) -> QueryUnit:
        """Parse PIVOT operation.

        PIVOT transforms rows into columns based on pivot values.
        Example: PIVOT(SUM(revenue) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4'))

        In sqlglot, PIVOT is stored as part of Table or Subquery nodes.

        Args:
            pivot_node: The PIVOT expression node
            name: Name for the pivot unit
            parent_unit: The parent QueryUnit
            depth: Current recursion depth
            table_node: The table or subquery source being pivoted

        Returns:
            QueryUnit representing the PIVOT operation
        """
        # Create unit for PIVOT operation
        unit_id = self._parser._generate_unit_id(QueryUnitType.PIVOT, name)
        unit = QueryUnit(
            unit_id=unit_id,
            unit_type=QueryUnitType.PIVOT,
            name=name,
            select_node=None,
            parent_unit=parent_unit,
            depth=depth,
        )

        # Extract PIVOT configuration
        pivot_config = {}

        # Get aggregation expressions (e.g., SUM(revenue))
        if hasattr(pivot_node, "expressions") and pivot_node.expressions:
            pivot_config["aggregations"] = [str(expr) for expr in pivot_node.expressions]

        # Get pivot column (the FOR column)
        # In sqlglot, the pivot column is in 'fields' which contains an In expression
        if hasattr(pivot_node, "fields") and pivot_node.fields:
            for field in pivot_node.fields:
                if isinstance(field, exp.In):
                    # The 'this' is the column being pivoted
                    pivot_config["pivot_column"] = str(field.this)

        # Get pivot values (the IN clause values)
        # In sqlglot, columns are stored in args, not as a direct attribute
        if hasattr(pivot_node, "args") and "columns" in pivot_node.args:
            columns = pivot_node.args["columns"]
            if columns:
                pivot_config["value_columns"] = [str(col) for col in columns]

        unit.pivot_config = pivot_config

        # Parse the source
        # table_node can be either a Table or a Subquery
        if isinstance(table_node, exp.Subquery):
            # PIVOT is applied to a subquery: (SELECT ...) PIVOT(...)
            source_select = table_node.this
            if isinstance(source_select, exp.Select):
                source_name = f"{name}_source"
                source_unit = self._parser._parse_select_unit(
                    select_node=source_select,
                    unit_type=QueryUnitType.SUBQUERY_PIVOT_SOURCE,
                    name=source_name,
                    parent_unit=unit,
                    depth=depth + 1,
                )
                unit.depends_on_units.append(source_unit.unit_id)
        elif isinstance(table_node, exp.Table):
            # PIVOT is applied to a table: table_name PIVOT(...)
            table_source = table_node.this

            # Check if it's a subquery or table reference
            if isinstance(table_source, exp.Subquery):
                # Shouldn't happen, but handle it
                source_select = table_source.this
                if isinstance(source_select, exp.Select):
                    source_name = f"{name}_source"
                    source_unit = self._parser._parse_select_unit(
                        select_node=source_select,
                        unit_type=QueryUnitType.SUBQUERY_PIVOT_SOURCE,
                        name=source_name,
                        parent_unit=unit,
                        depth=depth + 1,
                    )
                    unit.depends_on_units.append(source_unit.unit_id)
            else:
                # Source is a base table or CTE
                table_name = (
                    table_source.name if hasattr(table_source, "name") else str(table_source)
                )

                # Check if it's a CTE reference
                cte_unit = self._parser.unit_graph.get_unit_by_name(table_name)
                if cte_unit:
                    unit.depends_on_units.append(cte_unit.unit_id)
                else:
                    unit.depends_on_tables.append(table_name)

        # Add to graph
        self._parser.unit_graph.add_unit(unit)

        return unit

    def parse_unpivot(
        self,
        unpivot_node: exp.Pivot,  # Note: sqlglot uses Pivot class for both PIVOT and UNPIVOT
        name: str,
        parent_unit: QueryUnit,
        depth: int,
        table_node,  # Can be exp.Table or exp.Subquery
    ) -> QueryUnit:
        """Parse UNPIVOT operation.

        UNPIVOT transforms columns into rows.
        Example: UNPIVOT(revenue FOR quarter IN (q1_revenue, q2_revenue, q3_revenue, q4_revenue))

        In sqlglot, UNPIVOT is represented as a Pivot node with unpivot=True.

        Args:
            unpivot_node: The UNPIVOT expression node
            name: Name for the unpivot unit
            parent_unit: The parent QueryUnit
            depth: Current recursion depth
            table_node: The table or subquery source being unpivoted

        Returns:
            QueryUnit representing the UNPIVOT operation
        """
        # Create unit for UNPIVOT operation
        unit_id = self._parser._generate_unit_id(QueryUnitType.UNPIVOT, name)
        unit = QueryUnit(
            unit_id=unit_id,
            unit_type=QueryUnitType.UNPIVOT,
            name=name,
            select_node=None,
            parent_unit=parent_unit,
            depth=depth,
        )

        # Extract UNPIVOT configuration
        unpivot_config = {}

        # For UNPIVOT, we need to extract:
        # - value_column: The new column for unpivoted values (e.g., "revenue")
        # - name_column: The new column for column names (e.g., "quarter")
        # - unpivot_columns: The columns being unpivoted (e.g., [q1_revenue, q2_revenue, ...])

        # Get value column from expressions (e.g., revenue)
        if hasattr(unpivot_node, "expressions") and unpivot_node.expressions:
            unpivot_config["value_column"] = str(unpivot_node.expressions[0])

        # Get name column and unpivot columns from fields (the FOR ... IN clause)
        if hasattr(unpivot_node, "fields") and unpivot_node.fields:
            for field in unpivot_node.fields:
                if isinstance(field, exp.In):
                    # The 'this' is the name column (e.g., quarter)
                    unpivot_config["name_column"] = str(field.this)
                    # The 'expressions' are the columns being unpivoted
                    if hasattr(field, "expressions"):
                        unpivot_config["unpivot_columns"] = [str(col) for col in field.expressions]

        unit.unpivot_config = unpivot_config

        # Parse the source
        # table_node can be either a Table or a Subquery
        if isinstance(table_node, exp.Subquery):
            # UNPIVOT is applied to a subquery: (SELECT ...) UNPIVOT(...)
            source_select = table_node.this
            if isinstance(source_select, exp.Select):
                source_name = f"{name}_source"
                source_unit = self._parser._parse_select_unit(
                    select_node=source_select,
                    unit_type=QueryUnitType.SUBQUERY_PIVOT_SOURCE,
                    name=source_name,
                    parent_unit=unit,
                    depth=depth + 1,
                )
                unit.depends_on_units.append(source_unit.unit_id)
        elif isinstance(table_node, exp.Table):
            # UNPIVOT is applied to a base table: table_name UNPIVOT(...)
            table_name = (
                table_node.this.name if hasattr(table_node.this, "name") else table_node.name
            )
            unit.depends_on_tables.append(table_name)

        # Add to graph
        self._parser.unit_graph.add_unit(unit)

        return unit
