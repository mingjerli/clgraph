"""
Output column extraction for specialized SQL constructs.

Extracts column metadata from UNION, PIVOT, UNPIVOT, MERGE statements,
and expression strings.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import sqlglot
from sqlglot import exp

from .models import (
    ColumnLineageGraph,
    ColumnNode,
    QueryUnit,
    QueryUnitGraph,
)


@dataclass
class ExtractionContext:
    """Shared state needed by extraction functions."""

    unit_graph: QueryUnitGraph
    unit_columns_cache: Dict[str, List[ColumnNode]]
    external_table_columns: Dict[str, List[str]]
    lineage_graph: ColumnLineageGraph


def extract_union_columns(ctx: ExtractionContext, unit: QueryUnit) -> List[Dict]:
    """
    Extract output columns from a UNION/INTERSECT/EXCEPT operation.

    For set operations, all branches must have the same column structure.
    We use the first branch's columns as the output schema.
    """
    output_cols: List[Dict] = []

    if not unit.set_operation_branches:
        return output_cols

    # Get the first branch unit
    first_branch_id = unit.set_operation_branches[0]
    if first_branch_id not in ctx.unit_columns_cache:
        # Branch not yet processed — this shouldn't happen in topological order
        return output_cols

    # Use first branch's columns as the output schema
    first_branch_cols = ctx.unit_columns_cache[first_branch_id]

    for i, branch_col_node in enumerate(first_branch_cols):
        col_info = {
            "index": i,
            "name": branch_col_node.column_name,
            "is_star": branch_col_node.is_star,
            "type": "union_column",
            "expression": f"{unit.set_operation_type}({branch_col_node.column_name})",
            "source_branches": unit.set_operation_branches,
            "ast_node": None,
        }
        output_cols.append(col_info)

    return output_cols


def extract_pivot_columns(ctx: ExtractionContext, unit: QueryUnit) -> List[Dict]:
    """
    Extract output columns from a PIVOT operation.

    PIVOT transforms rows into columns.  The output has:
    - All columns from the source except the pivot column and aggregated column
    - New columns for each pivot value (e.g., Q1, Q2, Q3, Q4)
    """
    output_cols: List[Dict] = []

    if not unit.pivot_config:
        return output_cols

    # Get aggregated column names (e.g., "revenue" from "SUM(revenue)")
    aggregations = unit.pivot_config.get("aggregations", [])
    aggregated_cols: set = set()
    for agg in aggregations:
        if "(" in agg and ")" in agg:
            col_part = agg.split("(")[1].split(")")[0].strip()
            aggregated_cols.add(col_part)

    # Get source unit columns if available
    if unit.depends_on_units:
        source_unit_id = unit.depends_on_units[0]
        source_unit = ctx.unit_graph.units[source_unit_id]
        source_unit_name = source_unit.name  # Use name, not ID

        if source_unit_id in ctx.unit_columns_cache:
            source_cols = ctx.unit_columns_cache[source_unit_id]

            # Add non-pivoted columns
            pivot_column = unit.pivot_config.get("pivot_column", "")

            for i, source_col in enumerate(source_cols):
                if (
                    source_col.column_name != pivot_column
                    and source_col.column_name not in aggregated_cols
                    and not source_col.is_star
                ):
                    col_info = {
                        "index": i,
                        "name": source_col.column_name,
                        "is_star": False,
                        "type": "pivot_passthrough",
                        "expression": source_col.column_name,
                        "ast_node": None,
                        "source_columns": [(source_unit_name, source_col.column_name)],
                    }
                    output_cols.append(col_info)

    # Add pivot value columns (the new columns created by PIVOT)
    value_columns = unit.pivot_config.get("value_columns", [])
    base_idx = len(output_cols)

    # Get aggregated columns as source
    pivot_source_cols: List[Tuple[str, str]] = []
    if unit.depends_on_units:
        source_unit_id = unit.depends_on_units[0]
        source_unit = ctx.unit_graph.units[source_unit_id]
        source_unit_name = source_unit.name
        for agg_col in aggregated_cols:
            pivot_source_cols.append((source_unit_name, agg_col))

    for i, value_col in enumerate(value_columns):
        col_info = {
            "index": base_idx + i,
            "name": value_col,
            "is_star": False,
            "type": "pivot_value",
            "expression": f"PIVOT({value_col})",
            "ast_node": None,
            "source_columns": pivot_source_cols,
        }
        output_cols.append(col_info)

    return output_cols


def extract_unpivot_columns(ctx: ExtractionContext, unit: QueryUnit) -> List[Dict]:
    """
    Extract output columns from an UNPIVOT operation.

    UNPIVOT transforms columns into rows.  The output has:
    - All columns from the source except the unpivoted columns
    - A new value column (containing the values)
    - A new name column (containing the original column names)
    """
    output_cols: List[Dict] = []

    if not unit.unpivot_config:
        return output_cols

    # Get source unit columns if available
    if unit.depends_on_units:
        source_unit_id = unit.depends_on_units[0]
        source_unit = ctx.unit_graph.units[source_unit_id]
        source_unit_name = source_unit.name

        if source_unit_id in ctx.unit_columns_cache:
            source_cols = ctx.unit_columns_cache[source_unit_id]
            unpivot_columns = set(unit.unpivot_config.get("unpivot_columns", []))

            for i, source_col in enumerate(source_cols):
                if source_col.column_name not in unpivot_columns and not source_col.is_star:
                    col_info = {
                        "index": i,
                        "name": source_col.column_name,
                        "is_star": False,
                        "type": "unpivot_passthrough",
                        "expression": source_col.column_name,
                        "ast_node": None,
                        "source_columns": [(source_unit_name, source_col.column_name)],
                    }
                    output_cols.append(col_info)

    elif unit.depends_on_tables:
        # UNPIVOT on a table — use external_table_columns to infer passthrough columns
        table_name = unit.depends_on_tables[0]
        unpivot_columns = set(unit.unpivot_config.get("unpivot_columns", []))

        if table_name in ctx.external_table_columns:
            table_cols = ctx.external_table_columns[table_name]
            for i, col_name in enumerate(table_cols):
                if col_name not in unpivot_columns:
                    col_info = {
                        "index": i,
                        "name": col_name,
                        "is_star": False,
                        "type": "unpivot_passthrough",
                        "expression": col_name,
                        "ast_node": None,
                        "source_columns": [(table_name, col_name)],
                    }
                    output_cols.append(col_info)

    # Add the value column
    value_column = unit.unpivot_config.get("value_column", "value")
    unpivot_columns_list = unit.unpivot_config.get("unpivot_columns", [])

    # Build source_columns for the value column
    value_source_cols: List[Tuple[str, str]] = []
    if unit.depends_on_units:
        source_unit_id = unit.depends_on_units[0]
        source_unit = ctx.unit_graph.units[source_unit_id]
        source_unit_name = source_unit.name
        for unpivot_col in unpivot_columns_list:
            value_source_cols.append((source_unit_name, unpivot_col))
    elif unit.depends_on_tables:
        table_name = unit.depends_on_tables[0]
        for unpivot_col in unpivot_columns_list:
            value_source_cols.append((table_name, unpivot_col))

    col_info = {
        "index": len(output_cols),
        "name": value_column,
        "is_star": False,
        "type": "unpivot_value",
        "expression": f"UNPIVOT({value_column})",
        "ast_node": None,
        "source_columns": value_source_cols,
    }
    output_cols.append(col_info)

    # Add the name column
    name_column = unit.unpivot_config.get("name_column", "name")
    col_info = {
        "index": len(output_cols),
        "name": name_column,
        "is_star": False,
        "type": "unpivot_name",
        "expression": f"UNPIVOT({name_column})",
        "ast_node": None,
        "source_columns": [],
    }
    output_cols.append(col_info)

    return output_cols


def extract_merge_columns(ctx: ExtractionContext, unit: QueryUnit) -> List[Dict]:
    """
    Extract output columns from a MERGE operation.

    MERGE operations modify the target table.  The output represents:
    - Match condition columns (join columns)
    - Updated columns (from WHEN MATCHED THEN UPDATE)
    - Inserted columns (from WHEN NOT MATCHED THEN INSERT)
    """
    output_cols: List[Dict] = []

    if not unit.unpivot_config or unit.unpivot_config.get("merge_type") != "merge":
        return output_cols

    config = unit.unpivot_config
    target_table = config.get("target_table", "target")
    target_alias = config.get("target_alias") or target_table
    source_table = config.get("source_table", "source")
    source_alias = config.get("source_alias") or source_table
    match_columns = config.get("match_columns", [])
    matched_actions = config.get("matched_actions", [])
    not_matched_actions = config.get("not_matched_actions", [])

    idx = 0

    # 1. Match condition columns (edges for ON clause)
    for target_col, source_col in match_columns:
        col_info = {
            "index": idx,
            "name": target_col,
            "is_star": False,
            "type": "merge_match",
            "expression": f"{target_alias}.{target_col} = {source_alias}.{source_col}",
            "ast_node": None,
            "source_columns": [(source_alias, source_col)],
            "merge_action": "match",
        }
        output_cols.append(col_info)
        idx += 1

    # 1b. Literal-bound match filter columns (edges for ON clause literal predicates)
    match_filter_columns = config.get("match_filter_columns", [])
    for col_name, literal_val in match_filter_columns:
        col_info = {
            "index": idx,
            "name": col_name,
            "is_star": False,
            "type": "merge_match_filter",
            "expression": f"{target_alias}.{col_name} = {literal_val}",
            "ast_node": None,
            "source_columns": [(target_alias, col_name)],
            "merge_action": "match",
            "merge_column_role": "condition",
        }
        output_cols.append(col_info)
        idx += 1

    # 2. WHEN MATCHED -> UPDATE columns
    for action in matched_actions:
        if action.get("action_type") == "update":
            condition = action.get("condition")
            # Note: target_alias is used as default_table, but WHEN conditions
            # typically use qualified refs (t.name, s.name). extract_columns_from_expr
            # uses the qualified table ref when present, so the default_table only
            # applies to unqualified column names.
            condition_columns = (
                extract_columns_from_expr(condition, target_alias) if condition else []
            )
            for target_col, source_expr in action.get("column_mappings", {}).items():
                col_info = {
                    "index": idx,
                    "name": target_col,
                    "is_star": False,
                    "type": "merge_update",
                    "expression": source_expr,
                    "ast_node": None,
                    "source_columns": extract_columns_from_expr(source_expr, source_alias),
                    "merge_action": "update",
                    "merge_condition": condition,
                    "condition_columns": condition_columns,
                }
                output_cols.append(col_info)
                idx += 1

    # 3. WHEN NOT MATCHED -> INSERT columns
    for action in not_matched_actions:
        if action.get("action_type") == "insert":
            condition = action.get("condition")
            for target_col, source_expr in action.get("column_mappings", {}).items():
                col_info = {
                    "index": idx,
                    "name": target_col,
                    "is_star": False,
                    "type": "merge_insert",
                    "expression": source_expr,
                    "ast_node": None,
                    "source_columns": extract_columns_from_expr(source_expr, source_alias),
                    "merge_action": "insert",
                    "merge_condition": condition,
                }
                output_cols.append(col_info)
                idx += 1

    return output_cols


def extract_columns_from_expr(expr_str: str, default_table: str) -> List[Tuple[str, str]]:
    """
    Extract column references from a SQL expression string.  Pure function.

    Args:
        expr_str: SQL expression like "s.new_value" or "COALESCE(s.a, s.b)"
        default_table: Default table to use for unqualified columns

    Returns:
        List of (table, column) tuples
    """
    result: List[Tuple[str, str]] = []
    try:
        parsed = sqlglot.parse_one(expr_str, into=exp.Expression)
        for col in parsed.find_all(exp.Column):
            table_ref = default_table
            if hasattr(col, "table") and col.table:
                table_ref = str(col.table.name) if hasattr(col.table, "name") else str(col.table)
            col_name = col.name
            result.append((table_ref, col_name))
    except (sqlglot.errors.SqlglotError, ValueError, TypeError):
        # If parsing fails, try simple extraction for "table.column" format
        if "." in expr_str:
            parts = expr_str.split(".")
            if len(parts) == 2 and parts[0].isidentifier() and parts[1].isidentifier():
                result.append((parts[0], parts[1]))
    return result
