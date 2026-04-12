"""
Pure functions for parsing SQL aggregate expressions.

These functions are extracted from RecursiveLineageBuilder and have no
dependency on instance state. They operate solely on AST nodes and model types.
"""

from typing import Dict, List, Optional

from sqlglot import exp

from .models import AggregateSpec, AggregateType, OrderByColumn, QueryUnit

# ============================================================================
# Aggregate Registry
# ============================================================================

AGGREGATE_REGISTRY: Dict[str, AggregateType] = {
    # Array aggregates
    "array_agg": AggregateType.ARRAY,
    "array_concat_agg": AggregateType.ARRAY,
    "collect_list": AggregateType.ARRAY,
    "collect_set": AggregateType.ARRAY,
    "arrayagg": AggregateType.ARRAY,  # Alternative name
    # String aggregates
    "string_agg": AggregateType.STRING,
    "listagg": AggregateType.STRING,
    "group_concat": AggregateType.STRING,
    "concat_ws": AggregateType.STRING,
    # Object aggregates
    "object_agg": AggregateType.OBJECT,
    "map_agg": AggregateType.OBJECT,
    "json_agg": AggregateType.OBJECT,
    "jsonb_agg": AggregateType.OBJECT,
    "json_object_agg": AggregateType.OBJECT,
    "jsonb_object_agg": AggregateType.OBJECT,
    # Statistical aggregates
    "percentile_cont": AggregateType.STATISTICAL,
    "percentile_disc": AggregateType.STATISTICAL,
    "approx_quantiles": AggregateType.STATISTICAL,
    "median": AggregateType.STATISTICAL,
    "mode": AggregateType.STATISTICAL,
    "corr": AggregateType.STATISTICAL,
    "covar_pop": AggregateType.STATISTICAL,
    "covar_samp": AggregateType.STATISTICAL,
    "stddev": AggregateType.STATISTICAL,
    "stddev_pop": AggregateType.STATISTICAL,
    "stddev_samp": AggregateType.STATISTICAL,
    "variance": AggregateType.STATISTICAL,
    "var_pop": AggregateType.STATISTICAL,
    "var_samp": AggregateType.STATISTICAL,
    # Scalar aggregates
    "sum": AggregateType.SCALAR,
    "count": AggregateType.SCALAR,
    "avg": AggregateType.SCALAR,
    "min": AggregateType.SCALAR,
    "max": AggregateType.SCALAR,
    "any_value": AggregateType.SCALAR,
    "first_value": AggregateType.SCALAR,
    "last_value": AggregateType.SCALAR,
    "bit_and": AggregateType.SCALAR,
    "bit_or": AggregateType.SCALAR,
    "bit_xor": AggregateType.SCALAR,
    "bool_and": AggregateType.SCALAR,
    "bool_or": AggregateType.SCALAR,
}


def _get_aggregate_type(func_name: str) -> Optional[AggregateType]:
    """Get the aggregate type for a function name."""
    return AGGREGATE_REGISTRY.get(func_name.lower())


# ============================================================================
# Public Functions
# ============================================================================


def parse_aggregate_spec(ast_node: Optional[exp.Expression]) -> Optional[AggregateSpec]:
    """
    Parse an aggregate function expression and return its specification.

    Extracts function name, type, modifiers (DISTINCT), ORDER BY clauses,
    separators (for STRING_AGG), and value/key columns.

    Args:
        ast_node: The AST node to analyze for aggregate functions

    Returns:
        AggregateSpec if an aggregate function is found, None otherwise
    """
    if ast_node is None:
        return None

    # Find the first aggregate function in the expression
    agg_func = None
    for node in ast_node.walk():
        func_name = None

        # Check for known aggregate expression types
        if isinstance(node, exp.ArrayAgg):
            func_name = "ARRAY_AGG"
        elif isinstance(node, exp.GroupConcat):
            func_name = "GROUP_CONCAT"
        elif isinstance(node, exp.Count):
            func_name = "COUNT"
        elif isinstance(node, exp.Sum):
            func_name = "SUM"
        elif isinstance(node, exp.Avg):
            func_name = "AVG"
        elif isinstance(node, exp.Min):
            func_name = "MIN"
        elif isinstance(node, exp.Max):
            func_name = "MAX"
        elif isinstance(node, exp.AggFunc):
            # Generic aggregate function
            func_name = node.sql_name().upper() if hasattr(node, "sql_name") else "AGGREGATE"
        elif isinstance(node, (exp.Anonymous, exp.Func)):
            # Check if it's a known aggregate by name
            node_name = node.name if hasattr(node, "name") else ""
            if not node_name:
                node_name = node.sql_name() if hasattr(node, "sql_name") else ""
            if node_name and _get_aggregate_type(node_name):
                func_name = node_name.upper()

        if func_name and _get_aggregate_type(func_name):
            agg_func = node
            break

    if agg_func is None:
        return None

    # Get function name
    func_name = get_aggregate_func_name(agg_func)
    agg_type = _get_aggregate_type(func_name) or AggregateType.SCALAR

    # Extract value columns (columns being aggregated)
    value_columns: List[str] = []
    key_columns: List[str] = []
    order_by: List[OrderByColumn] = []
    distinct = False

    # Get function arguments - handle sqlglot's special wrapping
    args_to_process = []
    main_arg = getattr(agg_func, "this", None)

    # Check for Order wrapper (ARRAY_AGG(x ORDER BY y))
    if isinstance(main_arg, exp.Order):
        # The actual argument is inside the Order
        args_to_process = [main_arg.this] if main_arg.this else []
        # Extract ORDER BY from the Order node
        if hasattr(main_arg, "expressions"):
            for order_expr in main_arg.expressions:
                col_name = ""
                direction = "asc"
                nulls = None

                if isinstance(order_expr, exp.Ordered):
                    order_col = order_expr.this
                    if isinstance(order_col, exp.Column):
                        col_name = order_col.name
                        if order_col.table:
                            col_name = f"{order_col.table}.{order_col.name}"
                    else:
                        col_name = str(order_col)
                    direction = "desc" if order_expr.args.get("desc") else "asc"
                    nulls_first = order_expr.args.get("nulls_first")
                    if nulls_first is not None:
                        nulls = "first" if nulls_first else "last"
                elif isinstance(order_expr, exp.Column):
                    col_name = order_expr.name
                    if order_expr.table:
                        col_name = f"{order_expr.table}.{order_expr.name}"

                if col_name:
                    order_by.append(
                        OrderByColumn(column=col_name, direction=direction, nulls=nulls)
                    )
    # Check for Distinct wrapper (ARRAY_AGG(DISTINCT x))
    elif isinstance(main_arg, exp.Distinct):
        distinct = True
        args_to_process = list(main_arg.expressions) if hasattr(main_arg, "expressions") else []
    else:
        # Regular case
        args_to_process = list(agg_func.flatten()) if hasattr(agg_func, "flatten") else []
        if not args_to_process:
            args_to_process = getattr(agg_func, "expressions", [])
        if not args_to_process and main_arg:
            args_to_process = [main_arg]

    for arg in args_to_process:
        if isinstance(arg, exp.Column):
            col_name = arg.name
            if arg.table:
                col_name = f"{arg.table}.{arg.name}"
            value_columns.append(col_name)
        elif isinstance(arg, exp.Expression):
            # Try to extract column references from complex expressions
            for col in arg.find_all(exp.Column):
                col_name = col.name
                if col.table:
                    col_name = f"{col.table}.{col.name}"
                if col_name not in value_columns:
                    value_columns.append(col_name)

    # For OBJECT_AGG, second argument is value, first is key
    if agg_type == AggregateType.OBJECT and len(value_columns) >= 2:
        key_columns = [value_columns[0]]
        value_columns = [value_columns[1]]

    # Check for DISTINCT modifier (fallback for standard syntax)
    if not distinct and hasattr(agg_func, "distinct") and agg_func.distinct:
        distinct = True
    # Also check args.distinct
    if not distinct and agg_func.args.get("distinct"):
        distinct = True

    # Extract ORDER BY within aggregate (fallback for standard syntax)
    if not order_by and hasattr(agg_func, "order") and agg_func.order:
        for order_expr in agg_func.order.expressions:  # type: ignore[union-attr]
            col_name = ""
            direction = "asc"
            nulls = None

            if isinstance(order_expr, exp.Ordered):
                order_col = order_expr.this
                if isinstance(order_col, exp.Column):
                    col_name = order_col.name
                    if order_col.table:
                        col_name = f"{order_col.table}.{order_col.name}"
                else:
                    col_name = str(order_col)
                direction = "desc" if order_expr.args.get("desc") else "asc"
                nulls_first = order_expr.args.get("nulls_first")
                if nulls_first is not None:
                    nulls = "first" if nulls_first else "last"
            elif isinstance(order_expr, exp.Column):
                col_name = order_expr.name
                if order_expr.table:
                    col_name = f"{order_expr.table}.{order_expr.name}"

            if col_name:
                order_by.append(OrderByColumn(column=col_name, direction=direction, nulls=nulls))

    # Extract separator for STRING_AGG/LISTAGG
    separator = None
    if agg_type == AggregateType.STRING:
        # Look for separator argument (usually second argument)
        if isinstance(agg_func, exp.GroupConcat) and hasattr(agg_func, "separator"):
            sep = agg_func.separator
            if sep:
                separator = sep.this if isinstance(sep, exp.Literal) else str(sep)
        elif hasattr(agg_func, "expressions"):
            exprs = agg_func.expressions
            if len(exprs) >= 2:
                sep_arg = exprs[1]
                if isinstance(sep_arg, exp.Literal):
                    separator = sep_arg.this

    # Infer return type
    return_type = infer_aggregate_return_type(func_name, value_columns)

    return AggregateSpec(
        function_name=func_name,
        aggregate_type=agg_type,
        return_type=return_type,
        value_columns=value_columns,
        key_columns=key_columns,
        distinct=distinct,
        order_by=order_by,
        separator=separator,
    )


def get_aggregate_func_name(node: exp.Expression) -> str:
    """Get the function name from an aggregate expression."""
    if isinstance(node, exp.ArrayAgg):
        return "ARRAY_AGG"
    elif isinstance(node, exp.GroupConcat):
        return "GROUP_CONCAT"
    elif isinstance(node, exp.Count):
        return "COUNT"
    elif isinstance(node, exp.Sum):
        return "SUM"
    elif isinstance(node, exp.Avg):
        return "AVG"
    elif isinstance(node, exp.Min):
        return "MIN"
    elif isinstance(node, exp.Max):
        return "MAX"
    elif hasattr(node, "sql_name"):
        return node.sql_name().upper()  # type: ignore[union-attr]
    elif hasattr(node, "name") and node.name:
        return node.name.upper()
    return "AGGREGATE"


def infer_aggregate_return_type(func_name: str, value_columns: List[str]) -> str:
    """Infer the return type of an aggregate function."""
    func_lower = func_name.lower()

    if func_lower in ("array_agg", "collect_list", "collect_set", "array_concat_agg"):
        return "array"
    elif func_lower in ("string_agg", "listagg", "group_concat"):
        return "string"
    elif func_lower in ("object_agg", "map_agg", "json_agg", "jsonb_agg"):
        return "object"
    elif func_lower in ("count",):
        return "integer"
    elif func_lower in ("avg", "percentile_cont", "percentile_disc"):
        return "float"
    elif func_lower in ("sum", "min", "max"):
        return "numeric"
    elif func_lower in ("stddev", "variance", "var_pop", "var_samp"):
        return "float"
    return "any"


def has_star_in_aggregate(expr: Optional[exp.Expression]) -> bool:
    """
    Check if an expression contains an aggregate function with * (e.g., COUNT(*)).
    This is used to detect ambiguous lineage cases with JOINs.
    """
    if not expr:
        return False

    for node in expr.walk():
        # Check for aggregate functions
        if isinstance(node, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)):
            # Check if the aggregate has a * argument
            for child in node.walk():
                if isinstance(child, exp.Star):
                    return True

    return False


def unit_has_fully_resolved_columns(unit: QueryUnit) -> bool:
    """
    Check if a query unit has fully resolved columns (all explicit, no SELECT *).

    Returns True if we know ALL columns in this unit (no stars in SELECT).
    Returns False if the unit has SELECT * or other unresolvable column patterns.
    """
    # Skip for units without select_node (UNION, PIVOT, UNPIVOT)
    # For these types, assume columns are resolved from their source units
    if unit.select_node is None:
        return True

    # Check the SELECT expressions for any star notation
    for expr in unit.select_node.expressions:
        # Check for direct star: SELECT *
        if isinstance(expr, exp.Star):
            return False
        # Check for qualified star: SELECT table.*
        if isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star):
            return False

    # No stars found, all columns are explicit
    return True
