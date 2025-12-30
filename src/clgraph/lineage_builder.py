"""
Recursive lineage builder for SQL column lineage.

Builds complete column lineage graphs by recursively tracing through query units.
Includes SQLColumnTracer wrapper for backward compatibility.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

import sqlglot
from sqlglot import exp

from .metadata_parser import MetadataExtractor
from .models import (
    ColumnEdge,
    ColumnLineageGraph,
    ColumnNode,
    IssueCategory,
    IssueSeverity,
    QueryUnit,
    QueryUnitGraph,
    QueryUnitType,
    ValidationIssue,
)
from .query_parser import RecursiveQueryParser

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
    import re

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
# Part 1: Recursive Lineage Builder
# ============================================================================


class RecursiveLineageBuilder:
    """
    Build column lineage graph by recursively tracing through query units.
    """

    def __init__(
        self,
        sql_query: str,
        external_table_columns: Optional[Dict[str, List[str]]] = None,
        dialect: str = "bigquery",
        query_id: Optional[str] = None,
    ):
        self.sql_query = sql_query
        self.external_table_columns = external_table_columns or {}
        self.dialect = dialect
        self.query_id = query_id

        # Parse query structure first
        parser = RecursiveQueryParser(sql_query, dialect=dialect)
        self.unit_graph = parser.parse()

        # Column lineage graph (to be built)
        self.lineage_graph = ColumnLineageGraph()

        # Cache for resolved columns per unit
        self.unit_columns_cache: Dict[str, List[ColumnNode]] = {}

        # Metadata extractor for parsing SQL comments
        self.metadata_extractor = MetadataExtractor()

    def build(self) -> ColumnLineageGraph:
        """
        Build complete column lineage graph.

        Algorithm:
        1. Process units in topological order (bottom-up)
        2. For each unit:
           a. Resolve input columns from dependencies
           b. Extract output columns
           c. Create column nodes
           d. Create edges
        3. Return final graph
        """
        # Get units in dependency order (leaves first)
        ordered_units = self.unit_graph.get_topological_order()

        for unit in ordered_units:
            self._process_unit(unit)

        return self.lineage_graph

    def _process_unit(self, unit: QueryUnit):
        """
        Process a single query unit and add its nodes/edges to lineage graph.
        This is the core recursive lineage building method.
        """
        # 1. Get output columns from this unit's SELECT
        output_cols = self._extract_output_columns(unit)

        # 2. Run validation checks on this unit
        self._run_validations(unit, output_cols)

        # 3. For each output column, trace to its sources
        for col_info in output_cols:
            # Create output node
            output_node = self._create_column_node(unit=unit, col_info=col_info, is_output=True)
            self.lineage_graph.add_node(output_node)

            # 4. Trace dependencies recursively
            self._trace_column_dependencies(unit=unit, output_node=output_node, col_info=col_info)

        # Cache this unit's columns for parent units to reference
        self.unit_columns_cache[unit.unit_id] = [
            self.lineage_graph.nodes[self._get_node_key(unit, col_info)] for col_info in output_cols
        ]

    def _extract_output_columns(self, unit: QueryUnit) -> List[Dict]:
        """
        Extract output columns from a query unit's SELECT.
        Handles star notation and expands stars when source columns are known.

        Also handles special query types: UNION, PIVOT, UNPIVOT.
        """
        # Handle special query types that don't have a select_node
        if unit.unit_type in (QueryUnitType.UNION, QueryUnitType.INTERSECT, QueryUnitType.EXCEPT):
            return self._extract_union_columns(unit)
        elif unit.unit_type == QueryUnitType.PIVOT:
            return self._extract_pivot_columns(unit)
        elif unit.unit_type == QueryUnitType.UNPIVOT:
            return self._extract_unpivot_columns(unit)

        select_node = unit.select_node
        output_cols = []

        # Guard against None select_node or missing expressions
        if not select_node or not select_node.expressions:
            return output_cols

        for i, expr in enumerate(select_node.expressions):
            col_info = {"index": i, "ast_node": expr}

            # Check if star
            is_star = isinstance(expr, exp.Star) or (
                isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star)
            )

            if is_star:
                # Star expression
                star_alias = None
                if isinstance(expr, exp.Column) and hasattr(expr, "table") and expr.table:
                    # expr.table can be a string or an Identifier object
                    star_alias = expr.table.name if hasattr(expr.table, "name") else str(expr.table)

                # Resolve star source
                if star_alias:
                    # Qualified star - find source unit or table
                    source_unit = self._resolve_source_unit(unit, star_alias)
                    if source_unit:
                        col_info["source_unit"] = source_unit.unit_id
                        col_info["source_table"] = source_unit.name
                    else:
                        # Resolve alias to actual base table name
                        actual_table = self._resolve_base_table_name(unit, star_alias)
                        col_info["source_table"] = actual_table if actual_table else star_alias
                else:
                    # Unqualified star - get default FROM table
                    default_table = self._get_default_from_table(unit)
                    if default_table:
                        col_info["source_table"] = default_table
                        # Also try to resolve as a source unit (CTE/subquery)
                        source_unit = self._resolve_source_unit(unit, default_table)
                        if source_unit:
                            col_info["source_unit"] = source_unit.unit_id

                    # VALIDATION: Check for unqualified SELECT * with multiple tables
                    table_count = len(unit.depends_on_tables) + len(unit.depends_on_units)
                    if table_count > 1:
                        # This is ambiguous - which table does * refer to?
                        tables = [str(t) for t in unit.depends_on_tables] + [
                            str(self.unit_graph.units[uid].name)
                            for uid in unit.depends_on_units
                            if uid in self.unit_graph.units
                        ]
                        issue = ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES,
                            message=f"Unqualified SELECT * with {table_count} tables: {', '.join(tables)}. Cannot determine column sources.",
                            query_id=self.query_id,
                            location="SELECT clause",
                            suggestion=f"Use qualified star (e.g., SELECT {tables[0]}.*, {tables[1]}.* ...) or list columns explicitly",
                            context={"tables": tables, "table_count": table_count},
                        )
                        self.lineage_graph.add_issue(issue)

                col_info["is_star"] = True
                col_info["name"] = "*"
                col_info["type"] = "star"
                col_info["expression"] = expr.sql()

                # Handle EXCEPT/REPLACE
                # Note: sqlglot 28.x uses 'except_' and 'replace_' (with underscore)
                # while sqlglot 27.x uses 'except' and 'replace' (without underscore)
                star_expr = expr.this if isinstance(expr, exp.Column) else expr
                except_clause = None
                if hasattr(star_expr, "args"):
                    # Try both old and new sqlglot key names
                    except_clause = star_expr.args.get("except") or star_expr.args.get("except_")
                if except_clause:
                    col_info["except_columns"] = {col.name for col in except_clause}
                    col_info["type"] = "star_except"
                else:
                    col_info["except_columns"] = set()

                replace_clause = None
                if hasattr(star_expr, "args"):
                    # Try both old and new sqlglot key names
                    replace_clause = star_expr.args.get("replace") or star_expr.args.get("replace_")
                if replace_clause:
                    col_info["replace_columns"] = {
                        replace_expr.alias: replace_expr.sql()
                        for replace_expr in replace_clause
                        if hasattr(replace_expr, "alias")
                    }
                else:
                    col_info["replace_columns"] = {}

                # STAR EXPANSION: Try to expand star if we know the source columns
                # This only applies to the main query output (not CTEs/subqueries)
                if unit.unit_type == QueryUnitType.MAIN_QUERY:
                    expanded_cols = None

                    # Case 1: Source is a CTE or subquery (internal query unit)
                    if col_info.get("source_unit"):
                        source_unit_id = str(col_info["source_unit"])
                        expanded_cols = self._try_expand_star(unit, source_unit_id, col_info)

                    # Case 2: Source is an external table with known columns
                    elif col_info.get("source_table"):
                        source_table = col_info["source_table"]
                        # Type guard: ensure source_table is a string
                        if isinstance(source_table, str):
                            expanded_cols = self._try_expand_star_from_external_table(
                                unit, source_table, col_info
                            )

                    if expanded_cols:
                        # Replace star with expanded columns
                        output_cols.extend(expanded_cols)
                        continue  # Skip adding the star itself

            else:
                # Regular column
                col_info["is_star"] = False
                col_info["name"] = expr.alias_or_name
                col_info["type"] = self._determine_expression_type(expr)
                col_info["expression"] = expr.sql()
                col_info["source_columns"] = self._extract_source_column_refs(expr)

            output_cols.append(col_info)

        return output_cols

    def _extract_union_columns(self, unit: QueryUnit) -> List[Dict]:
        """
        Extract output columns from a UNION/INTERSECT/EXCEPT operation.

        For set operations, all branches must have the same column structure.
        We use the first branch's columns as the output schema.
        """
        output_cols = []

        if not unit.set_operation_branches:
            # No branches - return empty
            return output_cols

        # Get the first branch unit
        first_branch_id = unit.set_operation_branches[0]
        if first_branch_id not in self.unit_columns_cache:
            # Branch not yet processed - this shouldn't happen in topological order
            # Return empty for now
            return output_cols

        # Use first branch's columns as the output schema
        first_branch_cols = self.unit_columns_cache[first_branch_id]

        for i, branch_col_node in enumerate(first_branch_cols):
            col_info = {
                "index": i,
                "name": branch_col_node.column_name,
                "is_star": branch_col_node.is_star,
                "type": "union_column",
                "expression": f"{unit.set_operation_type}({branch_col_node.column_name})",
                "source_branches": unit.set_operation_branches,
                "ast_node": None,  # No AST node for UNION columns
            }
            output_cols.append(col_info)

        return output_cols

    def _extract_pivot_columns(self, unit: QueryUnit) -> List[Dict]:
        """
        Extract output columns from a PIVOT operation.

        PIVOT transforms rows into columns. The output has:
        - All columns from the source except the pivot column and aggregated column
        - New columns for each pivot value (e.g., Q1, Q2, Q3, Q4)
        """
        output_cols = []

        if not unit.pivot_config:
            return output_cols

        # Get aggregated column names (e.g., "revenue" from "SUM(revenue)")
        # These are needed for both passthrough and pivot value columns
        aggregations = unit.pivot_config.get("aggregations", [])
        aggregated_cols = set()
        for agg in aggregations:
            # Extract column name from aggregation (e.g., "revenue" from "SUM(revenue)")
            # Simple extraction - assumes format like SUM(col_name)
            if "(" in agg and ")" in agg:
                col_part = agg.split("(")[1].split(")")[0].strip()
                aggregated_cols.add(col_part)

        # Get source unit columns if available
        if unit.depends_on_units:
            source_unit_id = unit.depends_on_units[0]
            source_unit = self.unit_graph.units[source_unit_id]
            source_unit_name = source_unit.name  # Use name, not ID

            if source_unit_id in self.unit_columns_cache:
                source_cols = self.unit_columns_cache[source_unit_id]

                # Add non-pivoted columns (columns that aren't the pivot column or aggregated columns)
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
                            "source_columns": [
                                (source_unit_name, source_col.column_name)
                            ],  # Use name, not ID
                        }
                        output_cols.append(col_info)

        # Add pivot value columns (the new columns created by PIVOT)
        value_columns = unit.pivot_config.get("value_columns", [])
        base_idx = len(output_cols)

        # Get aggregated columns as source (e.g., "revenue" from "SUM(revenue)")
        # These pivot value columns derive from the aggregated column
        pivot_source_cols = []
        if unit.depends_on_units:
            source_unit_id = unit.depends_on_units[0]
            source_unit = self.unit_graph.units[source_unit_id]
            source_unit_name = source_unit.name  # Use name, not ID
            for agg_col in aggregated_cols:
                pivot_source_cols.append((source_unit_name, agg_col))  # Use name, not ID

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

    def _extract_unpivot_columns(self, unit: QueryUnit) -> List[Dict]:
        """
        Extract output columns from an UNPIVOT operation.

        UNPIVOT transforms columns into rows. The output has:
        - All columns from the source except the unpivoted columns
        - A new value column (containing the values)
        - A new name column (containing the original column names)
        """
        output_cols = []

        if not unit.unpivot_config:
            return output_cols

        # Get source unit columns if available
        if unit.depends_on_units:
            source_unit_id = unit.depends_on_units[0]
            source_unit = self.unit_graph.units[source_unit_id]
            source_unit_name = source_unit.name  # Use name, not ID

            if source_unit_id in self.unit_columns_cache:
                source_cols = self.unit_columns_cache[source_unit_id]
                unpivot_columns = set(unit.unpivot_config.get("unpivot_columns", []))

                # Add non-unpivoted columns (passthrough columns)
                for i, source_col in enumerate(source_cols):
                    if source_col.column_name not in unpivot_columns and not source_col.is_star:
                        col_info = {
                            "index": i,
                            "name": source_col.column_name,
                            "is_star": False,
                            "type": "unpivot_passthrough",
                            "expression": source_col.column_name,
                            "ast_node": None,
                            "source_columns": [
                                (source_unit_name, source_col.column_name)
                            ],  # Add source
                        }
                        output_cols.append(col_info)
        elif unit.depends_on_tables:
            # UNPIVOT on a table - use external_table_columns to infer passthrough columns
            table_name = unit.depends_on_tables[0]
            unpivot_columns = set(unit.unpivot_config.get("unpivot_columns", []))

            if table_name in self.external_table_columns:
                table_cols = self.external_table_columns[table_name]
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
        # The value column derives from all the unpivoted columns
        value_column = unit.unpivot_config.get("value_column", "value")
        unpivot_columns = unit.unpivot_config.get("unpivot_columns", [])

        # Build source_columns for the value column
        value_source_cols = []
        if unit.depends_on_units:
            # UNPIVOT on a subquery - reference the source unit
            source_unit_id = unit.depends_on_units[0]
            source_unit = self.unit_graph.units[source_unit_id]
            source_unit_name = source_unit.name
            for unpivot_col in unpivot_columns:
                value_source_cols.append((source_unit_name, unpivot_col))
        elif unit.depends_on_tables:
            # UNPIVOT on a table - reference the table columns
            table_name = unit.depends_on_tables[0]
            for unpivot_col in unpivot_columns:
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
        # The name column is generated (doesn't have direct source columns)
        name_column = unit.unpivot_config.get("name_column", "name")
        col_info = {
            "index": len(output_cols),
            "name": name_column,
            "is_star": False,
            "type": "unpivot_name",
            "expression": f"UNPIVOT({name_column})",
            "ast_node": None,
            "source_columns": [],  # Generated column, no direct source
        }
        output_cols.append(col_info)

        return output_cols

    def _trace_column_dependencies(self, unit: QueryUnit, output_node: ColumnNode, col_info: Dict):
        """
        Recursively trace where an output column's data comes from.
        This creates edges from source nodes to the output node.
        """
        if col_info.get("is_star"):
            # Star column - trace to source star or table
            source_unit_id = col_info.get("source_unit")
            source_table = col_info.get("source_table")

            if source_unit_id:
                # Star from another query unit (CTE or subquery)
                source_unit = self.unit_graph.units[source_unit_id]
                source_star_node = self._find_or_create_star_node(
                    source_unit, source_table or source_unit.name or source_unit.unit_id
                )

                # Create edge
                edge = ColumnEdge(
                    from_node=source_star_node,
                    to_node=output_node,
                    edge_type="star_passthrough",
                    transformation="star_passthrough",
                    context=unit.unit_type.value,
                    expression=col_info["expression"],
                )
                self.lineage_graph.add_edge(edge)

            elif source_table:
                # Star from base table
                table_star_node = self._find_or_create_table_star_node(source_table)

                edge = ColumnEdge(
                    from_node=table_star_node,
                    to_node=output_node,
                    edge_type="star_passthrough",
                    transformation="star_passthrough",
                    context=unit.unit_type.value,
                    expression=col_info["expression"],
                )
                self.lineage_graph.add_edge(edge)

        else:
            # Regular column - trace each source column reference
            source_refs = col_info.get("source_columns", [])

            # Special case: Check for COUNT(*) or other aggregates with *
            # These need special handling because they depend on ALL rows/columns from source
            has_star_in_aggregate = self._has_star_in_aggregate(col_info.get("ast_node"))

            if has_star_in_aggregate:
                # Aggregate with * (e.g., COUNT(*)) - link to all source columns/rows
                # Add warning only if multiple tables are involved (ambiguous case)
                has_multiple_sources = len(unit.depends_on_tables) + len(unit.depends_on_units) > 1
                if has_multiple_sources:
                    warning = (
                        f"Ambiguous lineage: {col_info['expression']} uses * with multiple sources."
                    )
                    output_node.warnings.append(warning)

                # Link to ALL tables involved
                # For base tables: use individual columns if schema is known, else use *
                for table_name in unit.depends_on_tables:
                    # Check if we know the schema for this table from upstream queries
                    # Use resolver to match short names (e.g., 'events') to full names (e.g., 'staging.events')
                    resolved_table = self._resolve_external_table_name(table_name)
                    if resolved_table:
                        # Schema is known - link to individual columns
                        column_names = self.external_table_columns[resolved_table]
                        for col_name in column_names:
                            # Use the resolved full table name for the node
                            source_node = self._find_or_create_table_column_node(
                                resolved_table, col_name
                            )
                            edge = ColumnEdge(
                                from_node=source_node,
                                to_node=output_node,
                                edge_type="aggregate",
                                transformation="ambiguous_aggregate",
                                context=unit.unit_type.value,
                                expression=col_info["expression"],
                            )
                            self.lineage_graph.add_edge(edge)
                    else:
                        # Schema unknown - use star node
                        table_star_node = self._find_or_create_table_star_node(table_name)
                        edge = ColumnEdge(
                            from_node=table_star_node,
                            to_node=output_node,
                            edge_type="aggregate",
                            transformation="ambiguous_aggregate",
                            context=unit.unit_type.value,
                            expression=col_info["expression"],
                        )
                        self.lineage_graph.add_edge(edge)

                # For query units: link to all explicit columns if fully resolved, else use *
                for unit_id in unit.depends_on_units:
                    dep_unit = self.unit_graph.units[unit_id]

                    # Check if this unit has fully resolved columns (no stars in its SELECT)
                    if self._unit_has_fully_resolved_columns(dep_unit):
                        # Link to ALL explicit columns from this unit
                        if unit_id in self.unit_columns_cache:
                            for col_node in self.unit_columns_cache[unit_id]:
                                if not col_node.is_star:  # Skip any * nodes
                                    edge = ColumnEdge(
                                        from_node=col_node,
                                        to_node=output_node,
                                        edge_type="aggregate",
                                        transformation="ambiguous_aggregate",
                                        context=unit.unit_type.value,
                                        expression=col_info["expression"],
                                    )
                                    self.lineage_graph.add_edge(edge)
                    else:
                        # Unit has unresolved columns (has * in SELECT), use * node
                        unit_star_node = self._find_or_create_star_node(
                            dep_unit, dep_unit.name or dep_unit.unit_id
                        )
                        edge = ColumnEdge(
                            from_node=unit_star_node,
                            to_node=output_node,
                            edge_type="aggregate",
                            transformation="ambiguous_aggregate",
                            context=unit.unit_type.value,
                            expression=col_info["expression"],
                        )
                        self.lineage_graph.add_edge(edge)

                # Don't process normal source_refs for this column
                return

            # Special case: UNION/INTERSECT/EXCEPT columns
            # These need edges from all branch columns with the same position
            if col_info.get("type") == "union_column" and "source_branches" in col_info:
                source_branches = col_info["source_branches"]
                col_index = col_info.get("index", 0)

                # Create edges from each branch's corresponding column
                for branch_id in source_branches:
                    if branch_id in self.unit_columns_cache:
                        branch_cols = self.unit_columns_cache[branch_id]
                        # Get the column at the same position in the branch
                        if col_index < len(branch_cols):
                            branch_col_node = branch_cols[col_index]
                            edge = ColumnEdge(
                                from_node=branch_col_node,
                                to_node=output_node,
                                edge_type="union",
                                transformation=unit.set_operation_type or "union",
                                context=unit.unit_type.value,
                                expression=col_info["expression"],
                            )
                            self.lineage_graph.add_edge(edge)

                # Don't process normal source_refs for this column
                return

            for source_ref in source_refs:
                # Unpack source reference (now includes JSON metadata)
                # Handle both old 2-tuple and new 4-tuple formats for backward compatibility
                if len(source_ref) == 4:
                    table_ref, col_name, json_path, json_function = source_ref
                else:
                    table_ref, col_name = source_ref[:2]
                    json_path, json_function = None, None

                # Resolve table_ref to either a query unit or base table
                # If table_ref is None, try to infer the default source
                effective_table_ref = table_ref

                # Check if this is an UNNEST alias (check both table_ref and col_name)
                # Case 1: Qualified reference like unnest_alias.field (table_ref = unnest_alias)
                # Case 2: Unqualified reference where column name is the UNNEST alias itself
                unnest_info = None
                if table_ref and table_ref in unit.unnest_sources:
                    unnest_info = unit.unnest_sources[table_ref]
                elif not table_ref and col_name in unit.unnest_sources:
                    # Unqualified column name matches UNNEST alias
                    unnest_info = unit.unnest_sources[col_name]

                if not table_ref:
                    # No explicit table reference - infer from FROM clause
                    default_table = self._get_default_from_table(unit)
                    if default_table:
                        effective_table_ref = default_table

                if unnest_info:
                    # This is a reference to an UNNEST result
                    self._create_unnest_edge(
                        unit=unit,
                        output_node=output_node,
                        col_info=col_info,
                        unnest_info=unnest_info,
                        col_name=col_name,
                    )
                    continue

                source_unit = (
                    self._resolve_source_unit(unit, effective_table_ref)
                    if effective_table_ref
                    else None
                )

                if source_unit:
                    # Reference to another query unit (CTE or subquery)
                    source_node = self._find_column_in_unit(source_unit, col_name)

                    if source_node:
                        edge = ColumnEdge(
                            from_node=source_node,
                            to_node=output_node,
                            edge_type=col_info["type"],
                            transformation=col_info["type"],
                            context=unit.unit_type.value,
                            expression=col_info["expression"],
                            json_path=json_path,
                            json_function=json_function,
                        )
                        self.lineage_graph.add_edge(edge)

                else:
                    # Reference to base table - resolve alias to actual table name
                    base_table = (
                        self._resolve_base_table_name(unit, effective_table_ref)
                        if effective_table_ref
                        else None
                    )
                    if base_table:
                        source_node = self._find_or_create_table_column_node(base_table, col_name)

                        edge = ColumnEdge(
                            from_node=source_node,
                            to_node=output_node,
                            edge_type=col_info["type"],
                            transformation=col_info["type"],
                            context=unit.unit_type.value,
                            expression=col_info["expression"],
                            json_path=json_path,
                            json_function=json_function,
                        )
                        self.lineage_graph.add_edge(edge)

    def _resolve_source_unit(
        self, current_unit: QueryUnit, table_ref: Optional[str]
    ) -> Optional[QueryUnit]:
        """
        Resolve a table reference to a query unit (CTE or subquery).

        This checks:
        1. Alias mappings in current unit (handles aliases like "b" -> "base")
        2. Units this current_unit depends on
        3. CTEs in parent units (scope traversal)
        """
        if not table_ref:
            return None

        # First, resolve alias to actual name
        actual_name = table_ref
        if table_ref in current_unit.alias_mapping:
            mapped_name, is_unit = current_unit.alias_mapping[table_ref]
            if is_unit:
                # This is an alias to a unit (CTE or subquery), use the actual name
                actual_name = mapped_name
            else:
                # This is a base table, not a unit
                return None

        # Check direct dependencies using actual name
        for dep_unit_id in current_unit.depends_on_units:
            dep_unit = self.unit_graph.units[dep_unit_id]
            if dep_unit.name == actual_name:
                return dep_unit

        # Check CTEs in parent scope (for nested subqueries)
        if current_unit.parent_unit:
            return self._resolve_source_unit(current_unit.parent_unit, actual_name)

        return None

    def _resolve_base_table_name(
        self, current_unit: QueryUnit, table_ref: Optional[str]
    ) -> Optional[str]:
        """
        Resolve a table reference (which might be an alias) to the actual base table name.

        For example, if the query has "FROM products p", this resolves "p" -> "products".

        Args:
            current_unit: The current query unit
            table_ref: The table reference (might be an alias or actual table name)

        Returns:
            The actual base table name, or None if not found
        """
        if not table_ref:
            return None

        # Check if this is in the alias mapping
        if table_ref in current_unit.alias_mapping:
            mapped_name, is_unit = current_unit.alias_mapping[table_ref]
            if not is_unit:
                # This is a base table alias, return the actual table name
                return mapped_name
            else:
                # This is a unit (CTE/subquery), not a base table
                return None

        # If not in alias mapping, it might already be the actual table name
        # Check if it's in depends_on_tables
        if table_ref in current_unit.depends_on_tables:
            return table_ref

        return None

    def _find_column_in_unit(
        self, unit: QueryUnit, col_name: str, visited: Optional[set] = None
    ) -> Optional[ColumnNode]:
        """
        Find a column node in a processed unit's output.

        If the column is not explicitly defined but the unit has a star column
        (SELECT *), trace through to find the actual source column.
        """
        if visited is None:
            visited = set()

        # Prevent infinite recursion
        if unit.unit_id in visited:
            return None
        visited.add(unit.unit_id)

        if unit.unit_id not in self.unit_columns_cache:
            return None

        # First, look for explicit column
        for node in self.unit_columns_cache[unit.unit_id]:
            if node.column_name == col_name:
                return node

        # Not found - check if this unit has a star column
        # If so, the column might come through the star from an upstream source
        star_node = None
        for node in self.unit_columns_cache[unit.unit_id]:
            if node.is_star:
                star_node = node
                break

        if star_node:
            # Find where the star comes from by looking at edges
            for edge in self.lineage_graph.edges:
                if edge.to_node.full_name == star_node.full_name:
                    source_node = edge.from_node
                    if source_node.is_star and source_node.table_name:
                        # The star comes from another unit - look up the column there
                        source_unit = self._get_unit_by_name(source_node.table_name)
                        if source_unit:
                            result = self._find_column_in_unit(source_unit, col_name, visited)
                            if result:
                                return result

        return None

    def _get_unit_by_name(self, name: str) -> Optional[QueryUnit]:
        """Get a QueryUnit by its name (CTE name or alias)"""
        for unit in self.unit_graph.units.values():
            if unit.name == name:
                return unit
        return None

    def _find_or_create_star_node(self, unit: QueryUnit, source_table: str) -> ColumnNode:
        """Find or create star node for a query unit"""
        node_key = f"{unit.name}.*"

        if node_key in self.lineage_graph.nodes:
            return self.lineage_graph.nodes[node_key]

        # Create new star node
        node = ColumnNode(
            layer=self._get_layer_for_unit(unit),
            table_name=unit.name or unit.unit_id,
            column_name="*",
            full_name=node_key,
            expression="*",
            node_type="star",
            source_expression=None,
            unit_id=unit.unit_id,  # IMPORTANT: Set the unit_id so it's grouped correctly
            is_star=True,
            star_source_table=source_table,
        )
        self.lineage_graph.add_node(node)

        return node

    def _resolve_external_table_name(self, table_name: str) -> Optional[str]:
        """
        Resolve a table name to its full qualified version in external_table_columns.

        The parser may return just 'events' while external_table_columns has 'staging.events'.
        This method finds the matching full qualified name.

        Args:
            table_name: Short or full table name (e.g., 'events' or 'staging.events')

        Returns:
            The full qualified table name if found in external_table_columns, None otherwise
        """
        # Direct match
        if table_name in self.external_table_columns:
            return table_name

        # Check if any key ends with .{table_name}
        for full_name in self.external_table_columns:
            if full_name.endswith(f".{table_name}"):
                return full_name

        return None

    def _find_or_create_table_star_node(self, table_name: str) -> ColumnNode:
        """Find or create star node for a base table"""
        node_key = f"{table_name}.*"

        if node_key in self.lineage_graph.nodes:
            return self.lineage_graph.nodes[node_key]

        node = ColumnNode(
            layer="input",
            table_name=table_name,
            column_name="*",
            full_name=node_key,
            expression="*",
            node_type="star",
            source_expression=None,
            unit_id=None,  # External table, not part of any QueryUnit
            is_star=True,
            star_source_table=table_name,
        )
        self.lineage_graph.add_node(node)

        return node

    def _find_or_create_table_column_node(self, table_name: str, col_name: str) -> ColumnNode:
        """Find or create column node for a base table"""
        node_key = f"{table_name}.{col_name}"

        if node_key in self.lineage_graph.nodes:
            return self.lineage_graph.nodes[node_key]

        node = ColumnNode(
            layer="input",
            table_name=table_name,
            column_name=col_name,
            full_name=node_key,
            expression=col_name,
            node_type="base_column",
            source_expression=None,
            unit_id=None,  # External table, not part of any QueryUnit
        )
        self.lineage_graph.add_node(node)

        return node

    def _create_unnest_edge(
        self,
        unit: QueryUnit,
        output_node: ColumnNode,
        col_info: Dict,
        unnest_info: Dict[str, Any],
        col_name: str,
    ):
        """
        Create an array expansion edge from UNNEST source to output column.

        Args:
            unit: The current query unit
            output_node: The output column node
            col_info: Column info dictionary
            unnest_info: UNNEST metadata from query parser
            col_name: The column name being referenced (may be field of struct)
        """
        source_table = unnest_info.get("source_table")
        source_column = unnest_info.get("source_column")
        expansion_type = unnest_info.get("expansion_type", "unnest")
        offset_alias = unnest_info.get("offset_alias")

        # Handle offset reference (WITH OFFSET)
        if unnest_info.get("is_offset"):
            # This is a reference to the offset/position column
            # The source is still the array column
            actual_unnest_alias = unnest_info.get("unnest_alias")
            if actual_unnest_alias and actual_unnest_alias in unit.unnest_sources:
                actual_unnest_info = unit.unnest_sources[actual_unnest_alias]
                source_table = actual_unnest_info.get("source_table")
                source_column = actual_unnest_info.get("source_column")

        if not source_column:
            # Can't create edge without source column info
            return

        # Determine the source table (resolve alias if needed)
        actual_source_table = source_table
        if source_table and source_table in unit.alias_mapping:
            actual_source_table, _ = unit.alias_mapping[source_table]
        elif source_table is None:
            # Try to get from depends_on_tables
            if unit.depends_on_tables:
                actual_source_table = unit.depends_on_tables[0]

        if not actual_source_table:
            return

        # Create or find the source array column node
        source_node = self._find_or_create_table_column_node(actual_source_table, source_column)

        # Create edge with array expansion metadata
        edge = ColumnEdge(
            from_node=source_node,
            to_node=output_node,
            edge_type="array_expansion",
            transformation="array_expansion",
            context=unit.unit_type.value,
            expression=col_info["expression"],
            is_array_expansion=True,
            expansion_type=expansion_type,
            offset_column=offset_alias if unnest_info.get("is_offset") else None,
        )
        self.lineage_graph.add_edge(edge)

    def _create_column_node(
        self, unit: QueryUnit, col_info: Dict, is_output: bool = False
    ) -> ColumnNode:
        """Create a ColumnNode from column info"""
        layer = (
            "output"
            if is_output and unit.unit_type == QueryUnitType.MAIN_QUERY
            else self._get_layer_for_unit(unit)
        )
        table_name = unit.name if unit.name else "__output__"
        col_name = col_info["name"]

        full_name = f"{table_name}.{col_name}"
        if is_output and unit.unit_type == QueryUnitType.MAIN_QUERY:
            full_name = f"output.{col_name}"

        # Extract metadata from SQL comments if ast_node is available
        sql_metadata = None
        ast_node = col_info.get("ast_node")
        if ast_node is not None:
            sql_metadata = self.metadata_extractor.extract_from_expression(ast_node)
            # Only keep if metadata has any content
            if not (
                sql_metadata.description
                or sql_metadata.pii is not None
                or sql_metadata.owner
                or sql_metadata.tags
                or sql_metadata.custom_metadata
            ):
                sql_metadata = None

        node = ColumnNode(
            layer=layer,
            table_name=table_name,
            column_name=col_name,
            full_name=full_name,
            expression=col_info["expression"],
            node_type=col_info["type"],
            source_expression=ast_node,
            unit_id=unit.unit_id,
            is_star=col_info.get("is_star", False),
            star_source_table=col_info.get("source_table"),
            except_columns=col_info.get("except_columns", set()),
            replace_columns=col_info.get("replace_columns", {}),
            sql_metadata=sql_metadata,
        )

        return node

    def _get_layer_for_unit(self, unit: QueryUnit) -> str:
        """Determine layer name for a query unit"""
        if unit.unit_type == QueryUnitType.MAIN_QUERY:
            return "output"
        elif unit.unit_type == QueryUnitType.CTE:
            return "cte"
        else:
            return "subquery"

    def _get_node_key(self, unit: QueryUnit, col_info: Dict) -> str:
        """Get cache key for a column node"""
        is_main = unit.unit_type == QueryUnitType.MAIN_QUERY
        table_name = unit.name if unit.name else "__output__"
        col_name = col_info["name"]

        if is_main:
            return f"output.{col_name}"
        else:
            return f"{table_name}.{col_name}"

    def _extract_source_column_refs(
        self, expr: exp.Expression
    ) -> List[Tuple[Optional[str], str, Optional[str], Optional[str]]]:
        """
        Extract source column references from expression with JSON metadata.

        Args:
            expr: The SQL expression to analyze

        Returns:
            List of tuples: (table_ref, column_name, json_path, json_function)
            - table_ref: Table/alias name or None for unqualified columns
            - column_name: Column name
            - json_path: Normalized JSON path (e.g., "$.address.city") or None
            - json_function: JSON function name (e.g., "JSON_EXTRACT") or None
        """
        refs: List[Tuple[Optional[str], str, Optional[str], Optional[str]]] = []

        for node in expr.walk():
            if isinstance(node, exp.Column):
                table_ref: Optional[str] = None
                if hasattr(node, "table") and node.table:
                    table_ref = (
                        str(node.table.name) if hasattr(node.table, "name") else str(node.table)
                    )

                # Check if this column is inside a JSON function
                json_path: Optional[str] = None
                json_function: Optional[str] = None

                json_ancestor = _find_json_function_ancestor(node, expr)
                if json_ancestor:
                    json_path = _extract_json_path(json_ancestor)
                    json_function = _get_json_function_name(json_ancestor)

                refs.append((table_ref, node.name, json_path, json_function))

        return refs

    def _determine_expression_type(self, expr: exp.Expression) -> str:
        """Classify expression type"""
        # Check if expression contains aggregates (walk the AST)
        for node in expr.walk():
            if isinstance(node, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)):
                return "aggregate"
            elif isinstance(node, exp.Window):
                return "window"

        # Check top-level expression type
        if isinstance(expr, exp.Case):
            return "case"
        elif isinstance(expr, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
            return "arithmetic"
        elif isinstance(expr, exp.Column):
            return "direct_column"
        else:
            return "expression"

    def _get_default_from_table(self, unit: QueryUnit) -> Optional[str]:
        """Get the default FROM table for unqualified column references"""
        # Get the first table from depends_on_tables or depends_on_units
        if unit.depends_on_tables:
            return unit.depends_on_tables[0]
        elif unit.depends_on_units:
            # Get first unit's name
            first_unit = self.unit_graph.units[unit.depends_on_units[0]]
            return first_unit.name
        return None

    def _unit_has_fully_resolved_columns(self, unit: QueryUnit) -> bool:
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

    def _has_star_in_aggregate(self, expr: Optional[exp.Expression]) -> bool:
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

    def _try_expand_star(
        self, unit: QueryUnit, source_unit_id: str, star_col_info: Dict
    ) -> Optional[List[Dict]]:
        """
        Try to expand a star into individual columns if the source columns are known.

        This is key for star expansion: when we SELECT * FROM cte3 and cte3 has
        explicit columns (id, name), we expand the star to output.id and output.name.

        Args:
            unit: The current query unit (main query)
            source_unit_id: The unit ID we're selecting * from
            star_col_info: The star column info dict

        Returns:
            List of expanded column dicts, or None if expansion not possible
        """
        # Get the source unit
        source_unit = self.unit_graph.units.get(source_unit_id)
        if not source_unit:
            return None

        # Check if source unit has already been processed (cached columns exist)
        if source_unit_id not in self.unit_columns_cache:
            return None

        source_columns = self.unit_columns_cache[source_unit_id]

        # Check if source has explicit columns (no stars)
        # If source itself has stars, we can't expand
        has_stars = any(col.is_star for col in source_columns)
        if has_stars:
            return None

        # Great! We can expand. Create individual column entries
        expanded: List[Dict] = []
        except_cols = star_col_info.get("except_columns", set())

        for source_col in source_columns:
            # Skip columns in EXCEPT clause
            if source_col.column_name in except_cols:
                continue

            # Create a new column info for this expanded column
            expanded_col = {
                "index": len(expanded),
                "ast_node": None,  # No specific AST node for expanded columns
                "is_star": False,  # This is now an explicit column
                "name": source_col.column_name,
                "type": "direct_column",  # Direct pass-through from source
                "expression": source_col.column_name,
                "source_columns": [(source_unit.name, source_col.column_name)],
                # Mark this as star-expanded so we can trace it properly
                "star_expanded": True,
                "star_source_unit": source_unit_id,
            }
            expanded.append(expanded_col)

        return expanded if expanded else None

    def _try_expand_star_from_external_table(
        self, unit: QueryUnit, source_table: str, star_col_info: Dict
    ) -> Optional[List[Dict]]:
        """
        Try to expand a star from an external table using external_table_columns.

        This handles the cross-query scenario where Query 2 does SELECT * FROM staging.orders,
        and staging.orders was created by Query 1 with known columns.

        Args:
            unit: The current query unit (main query)
            source_table: The external table name (e.g., "staging.orders")
            star_col_info: The star column info dict

        Returns:
            List of expanded column dicts, or None if expansion not possible
        """
        # Check if we have column information for this external table
        if source_table not in self.external_table_columns:
            # VALIDATION: Missing schema information for external table
            issue = ValidationIssue(
                severity=IssueSeverity.INFO,
                category=IssueCategory.STAR_WITHOUT_SCHEMA,
                message=f"SELECT * from external table '{source_table}' without known schema. Star cannot be expanded to individual columns.",
                query_id=self.query_id,
                location="SELECT clause",
                suggestion=f"Provide schema for '{source_table}' or list columns explicitly",
                context={
                    "table": source_table,
                    "available_schemas": list(self.external_table_columns.keys()),
                },
            )
            self.lineage_graph.add_issue(issue)
            return None

        column_names = self.external_table_columns[source_table]
        if not column_names:
            # VALIDATION: Empty schema information
            issue = ValidationIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.MISSING_SCHEMA_INFO,
                message=f"Table '{source_table}' has empty schema. Star cannot be expanded.",
                query_id=self.query_id,
                location="SELECT clause",
                suggestion=f"Check schema definition for '{source_table}'",
                context={"table": source_table},
            )
            self.lineage_graph.add_issue(issue)
            return None

        # Great! We can expand. Create individual column entries
        expanded: List[Dict] = []
        except_cols = star_col_info.get("except_columns", set())
        replace_cols = star_col_info.get("replace_columns", {})

        for col_name in column_names:
            # Skip columns in EXCEPT clause
            if col_name in except_cols:
                continue

            # Create a new column info for this expanded column
            expanded_col = {
                "index": len(expanded),
                "ast_node": None,  # No specific AST node for expanded columns
                "is_star": False,  # This is now an explicit column
                "name": col_name,
                "type": "direct_column",  # Direct pass-through from source
                "expression": replace_cols.get(col_name, col_name),  # Use REPLACE if specified
                "source_columns": [(source_table, col_name)],
                # Mark this as star-expanded so we can trace it properly
                "star_expanded": True,
                "star_source_table": source_table,
            }
            expanded.append(expanded_col)

        return expanded if expanded else None

    # ========================================================================
    # Validation Methods (Static Analysis)
    # ========================================================================

    def _validate_column_aliases(self, unit: QueryUnit, output_cols: List[Dict]) -> List[str]:
        """
        Validate that expressions have proper aliases.

        Rules:
        - Aggregates MUST have aliases (ERROR)
        - Window functions MUST have aliases (ERROR)
        - Other expressions SHOULD have aliases (WARNING)
        - Plain column references don't need aliases
        """
        warnings = []

        for _i, col_info in enumerate(output_cols):
            # Skip stars
            if col_info.get("is_star"):
                continue

            expr = col_info.get("ast_node")
            if not expr:
                continue

            # Check if this has an alias
            has_alias = bool(expr.alias)

            # Determine expression category
            is_plain_column = isinstance(expr, exp.Column) and not isinstance(expr.this, exp.Star)

            # Check for aggregates
            is_aggregate = False
            for node in expr.walk():
                if isinstance(node, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max, exp.AggFunc)):
                    is_aggregate = True
                    break

            # Check for window functions
            is_window = False
            for node in expr.walk():
                if isinstance(node, exp.Window):
                    is_window = True
                    break

            # Apply validation rules
            if is_aggregate and not has_alias:
                col_expr = expr.sql()
                warnings.append(
                    f"[SUGGESTION] Aggregate function in {unit.unit_id} lacks alias: '{col_expr}'. "
                    f"Add 'AS alias_name' for reliable column lineage across databases."
                )
            elif is_window and not has_alias:
                col_expr = expr.sql()
                warnings.append(
                    f"[SUGGESTION] Window function in {unit.unit_id} lacks alias: '{col_expr}'. "
                    f"Add 'AS alias_name' for reliable column lineage across databases."
                )
            elif not is_plain_column and not has_alias:
                col_expr = expr.sql()
                warnings.append(
                    f"[SUGGESTION] Expression in {unit.unit_id} lacks explicit alias: '{col_expr}'. "
                    f"Add 'AS alias_name' for better lineage clarity."
                )

        return warnings

    def _validate_unique_column_names(self, unit: QueryUnit, output_cols: List[Dict]) -> List[str]:
        """
        Validate that all output column names are unique.
        Duplicate column names make lineage ambiguous.
        """
        warnings = []
        col_names = []

        for col_info in output_cols:
            # Skip star columns (they expand to multiple columns)
            if col_info.get("is_star"):
                continue

            col_name = col_info["name"]

            # Check for duplicates
            if col_name in col_names:
                warnings.append(
                    f"[SUGGESTION] Duplicate column name '{col_name}' in {unit.unit_id}. "
                    f"Use explicit aliases to make lineage clearer."
                )
            else:
                col_names.append(col_name)

        return warnings

    def _validate_qualified_columns_in_joins(
        self, unit: QueryUnit, output_cols: List[Dict]
    ) -> List[str]:
        """
        Flag unqualified column references when query has JOINs.
        This is a readability/correctness issue - lineage should be obvious from SQL.

        When multiple tables are joined, unqualified column references can be ambiguous
        and may resolve incorrectly. This validation:
        1. Walks ALL expressions (not just top-level columns)
        2. Finds unqualified column references inside expressions
        3. Adds ValidationIssue objects for proper issue tracking
        """
        warnings = []

        # Skip validation for units without select_node (UNION, PIVOT, UNPIVOT)
        if unit.select_node is None:
            return warnings

        # Check if this query has JOINs or multiple tables
        joins = unit.select_node.args.get("joins", [])
        has_multiple_tables = (len(unit.depends_on_tables) + len(unit.depends_on_units) > 1) or (
            joins and len(joins) > 0
        )

        if not has_multiple_tables:
            return warnings

        # Collect all tables/schemas available for resolution
        available_tables = list(unit.depends_on_tables) + [
            self.unit_graph.units[uid].name for uid in unit.depends_on_units
        ]

        # Query has JOINs or multiple tables - check for unqualified columns
        for col_info in output_cols:
            # Skip stars (they're explicit about being unqualified)
            if col_info.get("is_star"):
                continue

            expr = col_info.get("ast_node")
            if not expr:
                continue

            output_col_name = col_info.get("name", "unknown")

            # Walk the ENTIRE expression to find all column references
            for node in expr.walk():
                if isinstance(node, exp.Column) and not isinstance(node.this, exp.Star):
                    # Check if column is unqualified (no table prefix)
                    table_ref = node.table if hasattr(node, "table") else None
                    col_name = node.name

                    if not table_ref:  # Unqualified!
                        # Add string warning for backward compatibility
                        warnings.append(
                            f"Unqualified column '{col_name}' in expression for '{output_col_name}' "
                            f"with multiple tables. Cannot determine source table."
                        )

                        # Also add proper ValidationIssue
                        issue = ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.UNQUALIFIED_COLUMN,
                            message=(
                                f"Unqualified column '{col_name}' in expression for '{output_col_name}'. "
                                f"With multiple tables ({', '.join(available_tables)}), "
                                f"the source table is ambiguous."
                            ),
                            query_id=self.query_id,
                            location=f"SELECT clause: {output_col_name}",
                            suggestion=(
                                f"Qualify the column with table name: e.g., 'table_name.{col_name}'"
                            ),
                            context={
                                "column_name": col_name,
                                "output_column": output_col_name,
                                "available_tables": available_tables,
                                "expression": expr.sql() if expr else None,
                            },
                        )
                        self.lineage_graph.add_issue(issue)

        return warnings

    def _run_validations(self, unit: QueryUnit, output_cols: List[Dict]):
        """
        Run all validation checks on a query unit and add warnings to the lineage graph.
        """
        # Collect all warnings
        all_warnings = []

        # 1. Check column aliases
        all_warnings.extend(self._validate_column_aliases(unit, output_cols))

        # 2. Check for duplicate column names
        all_warnings.extend(self._validate_unique_column_names(unit, output_cols))

        # 3. Check for unqualified columns in JOINs
        all_warnings.extend(self._validate_qualified_columns_in_joins(unit, output_cols))

        # Add warnings to lineage graph
        for warning in all_warnings:
            self.lineage_graph.add_warning(warning)


# ============================================================================
# Part 2: SQLColumnTracer Wrapper (Backward Compatibility)
# ============================================================================


class SQLColumnTracer:
    """
    High-level wrapper that provides backward compatibility with existing code.
    Uses RecursiveLineageBuilder internally.
    """

    def __init__(
        self,
        sql_query: str,
        external_table_columns: Optional[Dict[str, List[str]]] = None,
        dialect: str = "bigquery",
    ):
        self.sql_query = sql_query
        self.external_table_columns = external_table_columns or {}
        self.dialect = dialect
        self.parsed = sqlglot.parse_one(sql_query, read=dialect)

        # Build lineage
        self.builder = RecursiveLineageBuilder(sql_query, external_table_columns, dialect=dialect)
        self.lineage_graph = None
        self._select_columns_cache = None

    def get_column_names(self) -> List[str]:
        """Get list of output column names"""
        # Build graph if not already built
        if self.lineage_graph is None:
            self.lineage_graph = self.builder.build()

        # Get output nodes
        output_nodes = self.lineage_graph.get_output_nodes()
        return [node.column_name for node in output_nodes]

    def build_column_lineage_graph(self) -> ColumnLineageGraph:
        """Build and return the complete lineage graph"""
        if self.lineage_graph is None:
            self.lineage_graph = self.builder.build()
        return self.lineage_graph

    def get_forward_lineage(self, input_columns: List[str]) -> Dict[str, Any]:
        """
        Get forward lineage (impact analysis) for given input columns.

        Args:
            input_columns: List of input column names (e.g., ["users.id", "orders.total"])

        Returns:
            Dict with:
                - impacted_outputs: List of output column names affected
                - impacted_ctes: List of CTE names in the path
                - paths: List of path dicts with input, intermediate, output, transformations
        """
        if self.lineage_graph is None:
            self.lineage_graph = self.builder.build()

        result = {"impacted_outputs": [], "impacted_ctes": [], "paths": []}

        impacted_outputs = set()
        impacted_ctes = set()

        for input_col in input_columns:
            # Find matching input nodes
            start_nodes = []
            for node in self.lineage_graph.nodes.values():
                # Match by full_name or table.column pattern
                if node.full_name == input_col:
                    start_nodes.append(node)
                elif node.layer == "input":
                    # Try matching table.column pattern
                    if f"{node.table_name}.{node.column_name}" == input_col:
                        start_nodes.append(node)
                    # Try matching just column name for star patterns
                    elif input_col.endswith(".*") and node.is_star:
                        if node.table_name == input_col.replace(".*", ""):
                            start_nodes.append(node)

            # BFS forward from each start node
            for start_node in start_nodes:
                visited = set()
                queue = [(start_node, [start_node.full_name], [])]

                while queue:
                    current, path, transformations = queue.pop(0)

                    if current.full_name in visited:
                        continue
                    visited.add(current.full_name)

                    # Track CTEs
                    if current.layer == "cte" or current.layer.startswith("cte_"):
                        cte_name = current.table_name
                        impacted_ctes.add(cte_name)

                    # Get outgoing edges
                    outgoing = self.lineage_graph.get_edges_from(current)

                    if not outgoing:
                        # Reached end - check if output
                        if current.layer == "output":
                            impacted_outputs.add(current.column_name)
                            result["paths"].append(
                                {
                                    "input": input_col,
                                    "intermediate": path[1:-1] if len(path) > 2 else [],
                                    "output": current.column_name,
                                    "transformations": list(set(transformations)),
                                }
                            )
                    else:
                        for edge in outgoing:
                            new_path = path + [edge.to_node.full_name]
                            new_transforms = transformations + [edge.transformation]
                            queue.append((edge.to_node, new_path, new_transforms))

        result["impacted_outputs"] = list(impacted_outputs)
        result["impacted_ctes"] = list(impacted_ctes)

        return result

    def get_backward_lineage(self, output_columns: List[str]) -> BackwardLineageResult:
        """
        Get backward lineage (source tracing) for given output columns.

        Args:
            output_columns: List of output column names (e.g., ["id", "total_amount"])

        Returns:
            Dict with:
                - required_inputs: Dict[table_name, List[column_names]]
                - required_ctes: List of CTE names in the path
                - paths: List of path dicts
        """
        if self.lineage_graph is None:
            self.lineage_graph = self.builder.build()

        result: BackwardLineageResult = {"required_inputs": {}, "required_ctes": [], "paths": []}

        required_ctes = set()

        for output_col in output_columns:
            # Find matching output nodes
            start_nodes = []
            for node in self.lineage_graph.nodes.values():
                if node.layer == "output":
                    if node.column_name == output_col or node.full_name == output_col:
                        start_nodes.append(node)

            # BFS backward from each start node
            for start_node in start_nodes:
                visited = set()
                queue = [(start_node, [start_node.full_name], [])]

                while queue:
                    current, path, transformations = queue.pop(0)

                    if current.full_name in visited:
                        continue
                    visited.add(current.full_name)

                    # Track CTEs
                    if current.layer == "cte" or current.layer.startswith("cte_"):
                        cte_name = current.table_name
                        required_ctes.add(cte_name)

                    # Get incoming edges
                    incoming = self.lineage_graph.get_edges_to(current)

                    if not incoming:
                        # Reached source - should be input layer
                        if current.layer == "input" and current.table_name:
                            table = current.table_name
                            col = current.column_name

                            if table not in result["required_inputs"]:
                                result["required_inputs"][table] = []
                            if col not in result["required_inputs"][table]:
                                result["required_inputs"][table].append(col)

                            result["paths"].append(
                                {
                                    "output": output_col,
                                    "intermediate": list(reversed(path[1:-1]))
                                    if len(path) > 2
                                    else [],
                                    "input": f"{table}.{col}",
                                    "transformations": list(set(transformations)),
                                }
                            )
                    else:
                        for edge in incoming:
                            new_path = path + [edge.from_node.full_name]
                            new_transforms = transformations + [edge.transformation]
                            queue.append((edge.from_node, new_path, new_transforms))

        result["required_ctes"] = list(required_ctes)

        return result

    def get_query_structure(self) -> QueryUnitGraph:
        """Get the query structure graph"""
        return self.builder.unit_graph

    def trace_column_dependencies(self, column_name: str) -> Set[Tuple[int, int]]:
        """
        Trace column dependencies and return SQL positions (for backward compatibility).

        NOTE: This is a stub implementation that returns empty set.
        The new design focuses on graph-based lineage, not position-based highlighting.
        """
        # For now, return empty set - position tracking is not part of the new design
        return set()

    def get_highlighted_sql(self, column_name: str) -> str:
        """
        Return SQL with highlighted sections (for backward compatibility).

        NOTE: Returns un-highlighted SQL for now.
        Position-based highlighting is not part of the new recursive design.
        """
        return self.sql_query

    def get_syntax_tree(self, column_name: Optional[str] = None) -> str:
        """
        Return a string representation of the syntax tree.
        """
        if self.lineage_graph is None:
            self.lineage_graph = self.builder.build()

        # Build a simple tree view of the query structure
        result = ["Query Structure:", ""]

        for unit in self.builder.unit_graph.get_topological_order():
            indent = "  " * unit.depth
            deps = unit.depends_on_units + unit.depends_on_tables
            deps_str = f" <- {', '.join(deps)}" if deps else ""
            result.append(f"{indent}{unit.unit_id} ({unit.unit_type.value}){deps_str}")

        result.append("")
        result.append("Column Lineage Graph:")
        result.append(f"  Nodes: {len(self.lineage_graph.nodes)}")
        result.append(f"  Edges: {len(self.lineage_graph.edges)}")

        # Show nodes by layer
        for layer in ["input", "cte", "subquery", "output"]:
            layer_nodes = [n for n in self.lineage_graph.nodes.values() if n.layer == layer]
            if layer_nodes:
                result.append(f"\n  {layer.upper()} Layer ({len(layer_nodes)} nodes):")
                for node in sorted(layer_nodes, key=lambda n: n.full_name)[:10]:  # Show first 10
                    star_indicator = " ⭐" if node.is_star else ""
                    result.append(f"    - {node.full_name}{star_indicator}")
                if len(layer_nodes) > 10:
                    result.append(f"    ... and {len(layer_nodes) - 10} more")

        return "\n".join(result)

    @property
    def select_columns(self) -> List[Dict]:
        """
        Get select columns info for backward compatibility with app.
        Returns list of dicts with 'alias', 'sql', 'index' keys.
        """
        if self._select_columns_cache is None:
            if self.lineage_graph is None:
                self.lineage_graph = self.builder.build()

            # Get output nodes and format them
            output_nodes = self.lineage_graph.get_output_nodes()
            self._select_columns_cache = [
                {"alias": node.column_name, "sql": node.expression, "index": i}
                for i, node in enumerate(output_nodes)
            ]

        return self._select_columns_cache


__all__ = ["RecursiveLineageBuilder", "SQLColumnTracer"]
