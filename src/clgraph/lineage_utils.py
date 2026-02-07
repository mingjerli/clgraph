"""
Lineage utility functions for SQL column lineage analysis.

This module contains:
- Type definitions (SourceColumnRef, BackwardLineageResult)
- JSON function detection constants and utilities
- Aggregate function registry and classification
- Nested access detection and extraction functions
- Schema qualification utilities

Extracted from lineage_builder.py to improve module organization.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

import sqlglot
from sqlglot import exp
from sqlglot.optimizer import qualify_columns

from .models import AggregateType

# ============================================================================
# Type Definitions
# ============================================================================


class SourceColumnRef(TypedDict, total=False):
    """Type for source column reference with optional JSON metadata."""

    table_ref: Optional[str]
    column_name: str
    json_path: Optional[str]
    json_function: Optional[str]


class BackwardLineageResult(TypedDict):
    """Type for backward lineage result."""

    required_inputs: Dict[str, List[str]]
    required_ctes: List[str]
    paths: List[Dict[str, Any]]


# ============================================================================
# JSON Function Detection Constants
# ============================================================================

# JSON extraction function names by dialect (case-insensitive matching)
JSON_FUNCTION_NAMES: Set[str] = {
    # BigQuery
    "JSON_EXTRACT",
    "JSON_EXTRACT_SCALAR",
    "JSON_VALUE",
    "JSON_QUERY",
    "JSON_EXTRACT_STRING_ARRAY",
    "JSON_EXTRACT_ARRAY",
    # Snowflake
    "GET_PATH",
    "GET",
    "JSON_EXTRACT_PATH_TEXT",
    "TRY_PARSE_JSON",
    "PARSE_JSON",
    # PostgreSQL
    "JSONB_EXTRACT_PATH",
    "JSONB_EXTRACT_PATH_TEXT",
    "JSON_EXTRACT_PATH",
    # MySQL
    "JSON_UNQUOTE",
    # Spark/Databricks
    "GET_JSON_OBJECT",
    "JSON_TUPLE",
    # DuckDB
    "JSON_EXTRACT_STRING",
}

# Map of sqlglot expression types to normalized function names
JSON_EXPRESSION_TYPES: Dict[type, str] = {
    exp.JSONExtract: "JSON_EXTRACT",  # -> operator
    exp.JSONExtractScalar: "JSON_EXTRACT_SCALAR",  # ->> operator
    exp.JSONBExtract: "JSONB_EXTRACT",  # PostgreSQL jsonb ->
    exp.JSONBExtractScalar: "JSONB_EXTRACT_SCALAR",  # PostgreSQL jsonb ->>
}


# ============================================================================
# JSON Function Detection Functions
# ============================================================================


def _is_json_extract_function(node: exp.Expression) -> bool:
    """Check if an expression is a JSON extraction function."""
    # Check for known JSON expression types (operators like -> and ->>)
    if type(node) in JSON_EXPRESSION_TYPES:
        return True

    # Check for anonymous function calls with JSON function names
    if isinstance(node, exp.Anonymous):
        func_name = node.name.upper() if node.name else ""
        return func_name in JSON_FUNCTION_NAMES

    # Check for named function calls
    if isinstance(node, exp.Func):
        func_name = node.sql_name().upper() if hasattr(node, "sql_name") else ""
        return func_name in JSON_FUNCTION_NAMES

    return False


def _get_json_function_name(node: exp.Expression) -> str:
    """Get the normalized JSON function name from an expression."""
    # Check for known expression types
    if type(node) in JSON_EXPRESSION_TYPES:
        return JSON_EXPRESSION_TYPES[type(node)]

    # Check for anonymous function calls
    if isinstance(node, exp.Anonymous):
        return node.name.upper() if node.name else "JSON_EXTRACT"

    # Check for named function calls
    if isinstance(node, exp.Func):
        return node.sql_name().upper() if hasattr(node, "sql_name") else "JSON_EXTRACT"

    return "JSON_EXTRACT"


def _extract_json_path(func_node: exp.Expression) -> Optional[str]:
    """
    Extract and normalize JSON path from a JSON function call.

    Handles various syntaxes:
    - JSON_EXTRACT(col, '$.path') -> '$.path'
    - col->'path' -> '$.path'
    - col->>'path' -> '$.path'
    - GET_PATH(col, 'path.nested') -> '$.path.nested'

    Returns normalized JSONPath format ($.field.nested) or None if not extractable.
    """
    path_value: Optional[str] = None

    # Handle JSON operators (-> and ->>)
    if isinstance(
        func_node,
        (exp.JSONExtract, exp.JSONExtractScalar, exp.JSONBExtract, exp.JSONBExtractScalar),
    ):
        # The path is the second argument
        if hasattr(func_node, "expression") and func_node.expression:
            path_expr = func_node.expression
            if isinstance(path_expr, exp.Literal):
                path_value = path_expr.this
            else:
                path_value = path_expr.sql()

    # Handle function calls like JSON_EXTRACT(col, '$.path')
    elif isinstance(func_node, (exp.Anonymous, exp.Func)):
        # Get the second argument (path)
        expressions = getattr(func_node, "expressions", [])
        if len(expressions) >= 2:
            path_arg = expressions[1]
            if isinstance(path_arg, exp.Literal):
                path_value = path_arg.this
            else:
                path_value = path_arg.sql()

    if path_value:
        return _normalize_json_path(path_value)

    return None


def _normalize_json_path(path: str) -> str:
    """
    Normalize JSON path to consistent format.

    Conversions:
    - '$.address.city' -> '$.address.city' (unchanged)
    - '$["address"]["city"]' -> '$.address.city'
    - 'address.city' (Snowflake) -> '$.address.city'
    - '{address,city}' (PostgreSQL) -> '$.address.city'

    Args:
        path: Raw JSON path string

    Returns:
        Normalized path in $.field.nested format
    """
    # Remove surrounding quotes if present
    path = path.strip("'\"")

    # PostgreSQL array format: {address,city} -> $.address.city
    if path.startswith("{") and path.endswith("}"):
        parts = path[1:-1].split(",")
        return "$." + ".".join(part.strip() for part in parts)

    # Handle paths starting with $ (including bracket notation like $["field"])
    if path.startswith("$"):
        # Convert bracket notation to dot notation
        # $["address"]["city"] -> $.address.city
        # $['address']['city'] -> $.address.city
        path = re.sub(r'\["([^"]+)"\]', r".\1", path)
        path = re.sub(r"\['([^']+)'\]", r".\1", path)
        path = re.sub(r"\[(\d+)\]", r".\1", path)  # Array indices
        # Ensure path starts with $. not $..
        if path.startswith("$") and not path.startswith("$."):
            path = "$." + path[1:].lstrip(".")
        return path

    # Snowflake format without $: address.city -> $.address.city
    # Handle bracket notation without $
    path = re.sub(r'\["([^"]+)"\]', r".\1", path)
    path = re.sub(r"\['([^']+)'\]", r".\1", path)
    path = re.sub(r"\[(\d+)\]", r".\1", path)  # Array indices
    return "$." + path.lstrip(".")


def _find_json_function_ancestor(
    column: exp.Column, root: exp.Expression
) -> Optional[exp.Expression]:
    """
    Find if a column is an argument to a JSON extraction function.

    Walks up the AST from the column to find the nearest JSON function.

    Args:
        column: The column expression to check
        root: The root expression to search within

    Returns:
        The JSON function expression if found, None otherwise
    """
    # Build parent map for efficient ancestor lookup
    parent_map: Dict[int, exp.Expression] = {}

    def build_parent_map(node: exp.Expression, parent: Optional[exp.Expression] = None):
        if parent is not None:
            parent_map[id(node)] = parent
        for child in node.iter_expressions():
            build_parent_map(child, node)

    build_parent_map(root)

    # Walk up from column to find JSON function
    current: Optional[exp.Expression] = column
    while current is not None:
        if _is_json_extract_function(current):
            return current
        current = parent_map.get(id(current))

    return None


# ============================================================================
# Complex Aggregate Function Registry
# ============================================================================

# Maps aggregate function names (lowercase) to their AggregateType
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


def _is_complex_aggregate(func_name: str) -> bool:
    """Check if a function is a complex aggregate (non-scalar)."""
    agg_type = _get_aggregate_type(func_name)
    return agg_type is not None and agg_type != AggregateType.SCALAR


# ============================================================================
# Nested Access (Struct/Array/Map) Detection and Extraction
# ============================================================================


def _is_nested_access_expression(expr: exp.Expression) -> bool:
    """
    Check if expression involves nested field/subscript access.

    Detects:
    - exp.Dot: struct.field (after array access like items[0].name)
    - exp.Bracket: array[index] or map['key']
    """
    return isinstance(expr, (exp.Dot, exp.Bracket))


def _extract_nested_path_from_expression(
    expr: exp.Expression,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Extract nested path from Dot or Bracket expressions.

    Args:
        expr: The expression to analyze (Dot or Bracket)

    Returns:
        Tuple of (table_ref, column_name, nested_path, access_type):
        - table_ref: Table/alias name or None
        - column_name: Base column name
        - nested_path: Normalized path like "[0].field" or "['key']"
        - access_type: "array", "map", "struct", or "mixed"
    """
    components: List[str] = []
    access_types: Set[str] = set()
    current = expr

    # Walk down the expression tree to build the path
    while True:
        if isinstance(current, exp.Dot):
            # Struct field access: items[0].product_id
            # exp.Dot has 'this' (the object) and 'expression' (the field name)
            if hasattr(current, "expression") and current.expression:
                field_name = (
                    current.expression.name
                    if hasattr(current.expression, "name")
                    else str(current.expression)
                )
                components.insert(0, f".{field_name}")
                access_types.add("struct")
            current = current.this

        elif isinstance(current, exp.Bracket):
            # Array index or map key access
            if current.expressions:
                key_expr = current.expressions[0]

                if isinstance(key_expr, exp.Literal):
                    if key_expr.is_int:
                        # Array index
                        idx = int(key_expr.this)
                        components.insert(0, f"[{idx}]")
                        access_types.add("array")
                    elif key_expr.is_string:
                        # Map key
                        key = str(key_expr.this)
                        components.insert(0, f"['{key}']")
                        access_types.add("map")
                else:
                    # Dynamic index/key (variable)
                    components.insert(0, "[*]")
                    access_types.add("array")
            current = current.this

        elif isinstance(current, exp.Column):
            # Reached the base column
            table_ref = None
            if hasattr(current, "table") and current.table:
                table_ref = (
                    str(current.table.name)
                    if hasattr(current.table, "name")
                    else str(current.table)
                )

            nested_path = "".join(components) if components else None

            # Determine access type
            if len(access_types) == 0:
                access_type = None
            elif len(access_types) == 1:
                access_type = access_types.pop()
            else:
                access_type = "mixed"

            return (table_ref, current.name, nested_path, access_type)

        else:
            # Unknown node type, stop
            break

    return (None, None, None, None)


def _find_nested_access_ancestor(
    column: exp.Column, root: exp.Expression
) -> Optional[exp.Expression]:
    """
    Find if a column is the base of a nested access expression.

    Walks up the AST from the column to find if it's inside a Dot or Bracket.

    Args:
        column: The column expression to check
        root: The root expression to search within

    Returns:
        The outermost nested access expression (Dot or Bracket) if found
    """
    # Build parent map for efficient ancestor lookup
    parent_map: Dict[int, exp.Expression] = {}

    def build_parent_map(node: exp.Expression, parent: Optional[exp.Expression] = None):
        if parent is not None:
            parent_map[id(node)] = parent
        for child in node.iter_expressions():
            build_parent_map(child, node)

    build_parent_map(root)

    # Walk up from column to find nested access expressions
    current: Optional[exp.Expression] = column
    outermost_nested: Optional[exp.Expression] = None

    while current is not None:
        if isinstance(current, (exp.Dot, exp.Bracket)):
            outermost_nested = current
        current = parent_map.get(id(current))

    return outermost_nested


# ============================================================================
# Schema Qualification Utilities
# ============================================================================


def _convert_to_nested_schema(
    flat_schema: Dict[str, List[str]],
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Convert flat table schema to nested format for sqlglot optimizer.

    The sqlglot optimizer.qualify_columns requires a nested schema format:
    {
        "schema_name": {
            "table_name": {
                "column_name": "type"
            }
        }
    }

    Our flat format is:
    {
        "schema.table": ["col1", "col2", ...]
    }

    Args:
        flat_schema: Dict mapping "schema.table" to list of column names

    Returns:
        Nested schema dict suitable for sqlglot optimizer
    """
    nested: Dict[str, Dict[str, Dict[str, str]]] = {}

    for qualified_table, columns in flat_schema.items():
        parts = qualified_table.split(".")

        if len(parts) >= 2:
            # Has schema prefix: "schema.table" or "catalog.schema.table"
            schema_name = parts[-2]  # Second to last part
            table_name = parts[-1]  # Last part
        else:
            # No schema prefix - use empty string as schema
            schema_name = ""
            table_name = qualified_table

        if schema_name not in nested:
            nested[schema_name] = {}

        if table_name not in nested[schema_name]:
            nested[schema_name][table_name] = {}

        for col in columns:
            # Use "UNKNOWN" as type since we don't have type info
            nested[schema_name][table_name][col] = "UNKNOWN"

    return nested


def _qualify_sql_with_schema(
    sql_query: str,
    external_table_columns: Dict[str, List[str]],
    dialect: str,
) -> str:
    """
    Qualify unqualified column references in SQL using schema information.

    When a SQL query has multiple tables joined and columns are unqualified
    (no table prefix), this function uses the schema to determine which table
    each column belongs to and adds the appropriate table prefix.

    Args:
        sql_query: The SQL query to qualify
        external_table_columns: Dict mapping table names to column lists
        dialect: SQL dialect for parsing

    Returns:
        The SQL query with qualified column references
    """
    if not external_table_columns:
        return sql_query

    try:
        # Parse the SQL
        parsed = sqlglot.parse_one(sql_query, read=dialect)

        # Convert to nested schema format
        nested_schema = _convert_to_nested_schema(external_table_columns)

        # Use sqlglot's qualify_columns to add table prefixes
        qualified = qualify_columns.qualify_columns(
            parsed,
            schema=nested_schema,
            dialect=dialect,
            infer_schema=True,
        )

        # Return the qualified SQL
        return qualified.sql(dialect=dialect)

    except (sqlglot.errors.SqlglotError, KeyError, ValueError, TypeError):
        # If qualification fails, return original SQL
        # The lineage builder will handle unqualified columns as before
        return sql_query


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    # Type definitions
    "SourceColumnRef",
    "BackwardLineageResult",
    # JSON constants
    "JSON_FUNCTION_NAMES",
    "JSON_EXPRESSION_TYPES",
    # JSON functions
    "_is_json_extract_function",
    "_get_json_function_name",
    "_extract_json_path",
    "_normalize_json_path",
    "_find_json_function_ancestor",
    # Aggregate registry and functions
    "AGGREGATE_REGISTRY",
    "_get_aggregate_type",
    "_is_complex_aggregate",
    # Nested access functions
    "_is_nested_access_expression",
    "_extract_nested_path_from_expression",
    "_find_nested_access_ancestor",
    # Schema qualification functions
    "_convert_to_nested_schema",
    "_qualify_sql_with_schema",
]
