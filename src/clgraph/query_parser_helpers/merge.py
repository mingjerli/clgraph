"""MERGE statement parsing helper.

Handles parsing of MERGE INTO statements including WHEN MATCHED/NOT MATCHED clauses.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from sqlglot import exp

from ..models import QueryUnit, QueryUnitType

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

    def parse(
        self,
        merge_node: exp.Merge,
        name: str,
        depth: int,
    ) -> QueryUnit:
        """Parse MERGE INTO statement.

        MERGE combines INSERT, UPDATE, and DELETE operations based on match conditions.
        Example:
            MERGE INTO target t
            USING source s ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET t.value = s.new_value
            WHEN NOT MATCHED THEN INSERT (id, value) VALUES (s.id, s.new_value)

        Args:
            merge_node: The MERGE expression node
            name: Name for the merge unit
            depth: Current recursion depth

        Returns:
            QueryUnit representing the MERGE operation
        """
        # Create unit for MERGE operation
        unit_id = self._parser._generate_unit_id(QueryUnitType.MERGE, name)
        unit = QueryUnit(
            unit_id=unit_id,
            unit_type=QueryUnitType.MERGE,
            name=name,
            select_node=None,
            parent_unit=None,
            depth=depth,
        )

        # Extract target table
        target_table = merge_node.this
        target_name = None
        target_alias = None
        if isinstance(target_table, exp.Table):
            target_name = target_table.name
            if hasattr(target_table, "alias") and target_table.alias:
                target_alias = str(target_table.alias)

        # Extract source table (can be table or subquery)
        source = merge_node.args.get("using")
        source_name = None
        source_alias = None
        if isinstance(source, exp.Table):
            source_name = source.name
            if hasattr(source, "alias") and source.alias:
                source_alias = str(source.alias)
            unit.depends_on_tables.append(source_name)
        elif isinstance(source, exp.Subquery):
            # Source is a subquery - parse it
            source_select = source.this
            if isinstance(source_select, exp.Select):
                source_alias = (
                    str(source.alias) if hasattr(source, "alias") and source.alias else "source"
                )
                source_unit = self._parser._parse_select_unit(
                    select_node=source_select,
                    unit_type=QueryUnitType.MERGE_SOURCE,
                    name=source_alias,
                    parent_unit=unit,
                    depth=depth + 1,
                )
                unit.depends_on_units.append(source_unit.unit_id)
                source_name = source_alias

        # Add target to depends_on_tables (MERGE reads and modifies target)
        if target_name:
            unit.depends_on_tables.append(target_name)

        # Store alias mappings
        if target_alias and target_name:
            unit.alias_mapping[target_alias] = (target_name, False)
        if source_alias and source_name:
            unit.alias_mapping[source_alias] = (source_name, False)

        # Extract match condition
        match_condition = merge_node.args.get("on")
        match_condition_sql = match_condition.sql() if match_condition else None

        # Extract match columns from ON condition
        match_columns: List[Tuple[str, str]] = []
        if match_condition:
            for eq in match_condition.find_all(exp.EQ):
                left_col = eq.left
                right_col = eq.right
                if isinstance(left_col, exp.Column) and isinstance(right_col, exp.Column):
                    match_columns.append((left_col.name, right_col.name))

        # Parse WHEN clauses from the 'whens' arg
        whens = merge_node.args.get("whens")
        matched_actions: List[Dict[str, Any]] = []
        not_matched_actions: List[Dict[str, Any]] = []

        if whens and hasattr(whens, "expressions"):
            for when in whens.expressions:
                is_matched = when.args.get("matched", False)
                then_expr = when.args.get("then")
                condition = when.args.get("condition")
                condition_sql = condition.sql() if condition else None

                action: Dict[str, Any] = {
                    "condition": condition_sql,
                    "column_mappings": {},
                }

                if isinstance(then_expr, exp.Update):
                    action["action_type"] = "update"
                    # Extract SET clause mappings
                    for set_expr in then_expr.expressions:
                        if isinstance(set_expr, exp.EQ):
                            target_col = (
                                set_expr.left.name
                                if hasattr(set_expr.left, "name")
                                else str(set_expr.left)
                            )
                            source_expr = set_expr.right.sql()
                            action["column_mappings"][target_col] = source_expr
                    if is_matched:
                        matched_actions.append(action)
                    else:
                        not_matched_actions.append(action)

                elif isinstance(then_expr, exp.Insert):
                    action["action_type"] = "insert"
                    # Extract target columns and source values
                    target_cols = []
                    if then_expr.this and hasattr(then_expr.this, "expressions"):
                        target_cols = [col.name for col in then_expr.this.expressions]
                    source_vals = []
                    if then_expr.expression and hasattr(then_expr.expression, "expressions"):
                        source_vals = [val.sql() for val in then_expr.expression.expressions]
                    for i, target_col in enumerate(target_cols):
                        if i < len(source_vals):
                            action["column_mappings"][target_col] = source_vals[i]
                    not_matched_actions.append(action)

                elif isinstance(then_expr, exp.Delete):
                    action["action_type"] = "delete"
                    if is_matched:
                        matched_actions.append(action)

        # Store merge configuration in a custom attribute
        # Using unpivot_config as a general-purpose config storage
        unit.unpivot_config = {
            "merge_type": "merge",
            "target_table": target_name,
            "target_alias": target_alias,
            "source_table": source_name,
            "source_alias": source_alias,
            "match_condition": match_condition_sql,
            "match_columns": match_columns,
            "matched_actions": matched_actions,
            "not_matched_actions": not_matched_actions,
        }

        # Add to graph
        self._parser.unit_graph.add_unit(unit)

        return unit
