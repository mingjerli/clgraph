"""Special source handling helper.

Handles parsing of special sources like UNNEST, Table-Valued Functions (TVFs),
VALUES clauses, and LATERAL subqueries.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlglot import exp

from ..models import QueryUnitType, TVFInfo, TVFType, ValuesInfo
from ..tvf_registry import KNOWN_TVF_EXPRESSIONS, KNOWN_TVF_NAMES, TVF_DEFAULT_COLUMNS

if TYPE_CHECKING:
    from ..models import QueryUnit
    from ..query_parser import RecursiveQueryParser


class SpecialSourcesHandler:
    """Helper for handling special sources (UNNEST, TVF, VALUES, LATERAL)."""

    def __init__(self, parser: "RecursiveQueryParser"):
        """Initialize with reference to parent parser.

        Args:
            parser: The parent RecursiveQueryParser instance
        """
        self._parser = parser

    def is_unnest(self, source) -> bool:
        """Check if source is an UNNEST expression."""
        return isinstance(source, exp.Unnest)

    def is_tvf(self, source) -> bool:
        """Check if source is a Table-Valued Function."""
        if isinstance(source, exp.Table):
            if hasattr(source, "this") and self._is_tvf_expression(source.this):
                return True
        return False

    def is_lateral(self, source) -> bool:
        """Check if source is a LATERAL expression."""
        return isinstance(source, exp.Lateral)

    def is_values(self, source) -> bool:
        """Check if source is a VALUES clause."""
        return isinstance(source, exp.Values)

    def _is_tvf_expression(self, expr: exp.Expression) -> bool:
        """Check if expression is a Table-Valued Function."""
        # Check for known TVF expression types
        if type(expr) in KNOWN_TVF_EXPRESSIONS:
            return True

        # Check for Anonymous function calls with known TVF names
        if isinstance(expr, exp.Anonymous):
            func_name = expr.name.lower() if expr.name else ""
            return func_name in KNOWN_TVF_NAMES

        return False

    def process_unnest(self, unnest_node: exp.Unnest, parent_unit: "QueryUnit"):
        """Process UNNEST expression and store metadata in parent_unit."""
        # Extract the array column being unnested
        array_expr = None
        if unnest_node.expressions:
            array_expr = unnest_node.expressions[0]

        if not array_expr:
            return

        # Get source table and column from array expression
        source_table = None
        source_column = None
        if isinstance(array_expr, exp.Column):
            source_column = array_expr.name
            if hasattr(array_expr, "table") and array_expr.table:
                source_table = (
                    array_expr.table.name
                    if hasattr(array_expr.table, "name")
                    else str(array_expr.table)
                )

        # Get the alias for the unnested values
        unnest_alias = None
        alias_node = unnest_node.args.get("alias")
        if alias_node:
            # TableAlias has columns attribute for the value aliases
            if hasattr(alias_node, "columns") and alias_node.columns:
                unnest_alias = alias_node.columns[0].name
            elif hasattr(alias_node, "this"):
                unnest_alias = (
                    alias_node.this.name
                    if hasattr(alias_node.this, "name")
                    else str(alias_node.this)
                )

        if not unnest_alias:
            unnest_alias = f"_unnest_{self._parser.subquery_counter}"
            self._parser.subquery_counter += 1

        # Get offset alias if WITH OFFSET is used
        offset_alias = None
        offset_node = unnest_node.args.get("offset")
        if offset_node:
            if hasattr(offset_node, "name"):
                offset_alias = offset_node.name
            elif hasattr(offset_node, "this"):
                offset_alias = (
                    offset_node.this if isinstance(offset_node.this, str) else str(offset_node)
                )
            else:
                offset_alias = str(offset_node)

        # Store UNNEST info in parent_unit
        parent_unit.unnest_sources[unnest_alias] = {
            "source_table": source_table,
            "source_column": source_column,
            "offset_alias": offset_alias,
            "expansion_type": "unnest",
        }

        # Also add offset alias if present
        if offset_alias:
            parent_unit.unnest_sources[offset_alias] = {
                "source_table": source_table,
                "source_column": source_column,
                "is_offset": True,
                "unnest_alias": unnest_alias,
                "expansion_type": "unnest",
            }

    def process_lateral_flatten(self, lateral_node: exp.Lateral, parent_unit: "QueryUnit"):
        """Process Snowflake LATERAL FLATTEN and store metadata."""
        inner_expr = lateral_node.this
        if not isinstance(inner_expr, exp.Explode):
            return

        # Extract INPUT parameter from FLATTEN
        source_table = None
        source_column = None

        input_expr = inner_expr.this
        if isinstance(input_expr, exp.EQ):
            # INPUT => col format
            right = input_expr.right
            if isinstance(right, exp.Column):
                source_column = right.name
                if hasattr(right, "table") and right.table:
                    source_table = (
                        right.table.name if hasattr(right.table, "name") else str(right.table)
                    )
        elif isinstance(input_expr, exp.Column):
            source_column = input_expr.name
            if hasattr(input_expr, "table") and input_expr.table:
                source_table = (
                    input_expr.table.name
                    if hasattr(input_expr.table, "name")
                    else str(input_expr.table)
                )

        # Get alias
        flatten_alias = None
        if hasattr(lateral_node, "alias") and lateral_node.alias:
            if hasattr(lateral_node.alias, "this"):
                flatten_alias = (
                    lateral_node.alias.this.name
                    if hasattr(lateral_node.alias.this, "name")
                    else str(lateral_node.alias.this)
                )
            else:
                flatten_alias = str(lateral_node.alias)

        if not flatten_alias:
            flatten_alias = f"_flatten_{self._parser.subquery_counter}"
            self._parser.subquery_counter += 1

        # Ensure str type for type checker (flatten_alias is guaranteed non-empty)
        flatten_alias = str(flatten_alias)

        # Store FLATTEN info
        parent_unit.unnest_sources[flatten_alias] = {
            "source_table": source_table,
            "source_column": source_column,
            "offset_alias": None,  # FLATTEN uses .INDEX field instead
            "expansion_type": "flatten",
            "flatten_fields": ["VALUE", "INDEX", "KEY", "PATH", "SEQ", "THIS"],
        }

    def process_lateral_subquery(
        self,
        lateral_node: exp.Lateral,
        parent_unit: "QueryUnit",
        preceding_tables: List[str],
        depth: int,
    ):
        """Process LATERAL subquery and identify correlated column references.

        Args:
            lateral_node: The LATERAL AST node
            parent_unit: The parent query unit
            preceding_tables: List of table names/aliases that precede this LATERAL
            depth: Current recursion depth
        """
        inner_expr = lateral_node.this

        # Skip if this is a FLATTEN (handled separately)
        if isinstance(inner_expr, exp.Explode):
            self.process_lateral_flatten(lateral_node, parent_unit)
            return

        # Skip if not a Subquery
        if not isinstance(inner_expr, exp.Subquery):
            return

        subquery = inner_expr.this
        if not isinstance(subquery, exp.Select):
            return

        # Get LATERAL alias
        lateral_alias = None
        if hasattr(lateral_node, "alias") and lateral_node.alias:
            if hasattr(lateral_node.alias, "this"):
                lateral_alias = (
                    lateral_node.alias.this.name
                    if hasattr(lateral_node.alias.this, "name")
                    else str(lateral_node.alias.this)
                )
            else:
                lateral_alias = str(lateral_node.alias)

        if not lateral_alias:
            lateral_alias = f"_lateral_{self._parser.subquery_counter}"
            self._parser.subquery_counter += 1

        # Ensure str type for type checker (lateral_alias is guaranteed non-empty)
        lateral_alias = str(lateral_alias)

        # Find all column references in the subquery
        correlated_columns: List[str] = []
        for col in subquery.find_all(exp.Column):
            table_ref = None
            if hasattr(col, "table") and col.table:
                table_ref = str(col.table.name) if hasattr(col.table, "name") else str(col.table)

            # Check if this column references a preceding table (correlation)
            if table_ref and table_ref in preceding_tables:
                correlated_columns.append(f"{table_ref}.{col.name}")

        # Store LATERAL info
        parent_unit.lateral_sources[lateral_alias] = {
            "correlated_columns": correlated_columns,
            "preceding_tables": preceding_tables.copy(),
            "subquery_sql": subquery.sql(),
        }

        # Parse the LATERAL subquery as a unit
        subquery_name = lateral_alias
        subquery_unit = self._parser._parse_select_unit(
            select_node=subquery,
            unit_type=QueryUnitType.SUBQUERY_FROM,
            name=subquery_name,
            parent_unit=parent_unit,
            depth=depth + 1,
        )

        # Mark as LATERAL and store correlation info
        subquery_unit.is_lateral = True
        subquery_unit.correlated_columns = correlated_columns

        # Add dependency and alias mapping
        if subquery_unit.unit_id not in parent_unit.depends_on_units:
            parent_unit.depends_on_units.append(subquery_unit.unit_id)
        parent_unit.alias_mapping[lateral_alias] = (subquery_name, True)

    def _extract_tvf_info(
        self, tvf_expr: exp.Expression, alias: str, column_aliases: List[str]
    ) -> TVFInfo:
        """Extract TVF information from a TVF expression."""
        # Determine function name and type
        tvf_type: TVFType = TVFType.GENERATOR  # default
        func_name: str = ""
        parameters: Dict[str, Any] = {}
        input_columns: List[str] = []
        external_source: Optional[str] = None

        # Get type from expression class
        if type(tvf_expr) in KNOWN_TVF_EXPRESSIONS:
            tvf_type = KNOWN_TVF_EXPRESSIONS[type(tvf_expr)]
            # Get function name from class name
            func_name = type(tvf_expr).__name__.lower()
            # Map to standard name
            if func_name in ("explodinggenerateseries", "generateseries"):
                func_name = "generate_series"
            elif func_name == "generatedatearray":
                func_name = "generate_date_array"
            elif func_name == "readcsv":
                func_name = "read_csv"

        # Handle Anonymous function calls
        elif isinstance(tvf_expr, exp.Anonymous):
            func_name = tvf_expr.name.lower() if tvf_expr.name else "unknown"
            tvf_type = KNOWN_TVF_NAMES.get(func_name, TVFType.GENERATOR)

        # Extract parameters from expressions attribute
        if hasattr(tvf_expr, "expressions") and tvf_expr.expressions:
            for i, arg in enumerate(tvf_expr.expressions):
                if isinstance(arg, exp.Literal):
                    # Literal value
                    value = arg.this
                    # Detect file paths for external TVFs
                    if i == 0 and tvf_type == TVFType.EXTERNAL:
                        external_source = str(value)
                    parameters[f"arg_{i}"] = value
                elif isinstance(arg, exp.Column):
                    # Column reference - indicates COLUMN_INPUT type
                    col_ref = f"{arg.table}.{arg.name}" if arg.table else arg.name
                    input_columns.append(col_ref)
                    parameters[f"arg_{i}"] = col_ref
                elif isinstance(arg, exp.Kwarg):
                    # Named parameter (e.g., ROWCOUNT => 100)
                    key = str(arg.this) if arg.this else f"arg_{i}"
                    value = str(arg.expression) if arg.expression else None
                    parameters[key.lower()] = value
                else:
                    parameters[f"arg_{i}"] = str(arg)

        # Also extract parameters from args dict (for typed TVFs like ExplodingGenerateSeries)
        if hasattr(tvf_expr, "args"):
            args_dict = tvf_expr.args
            for key, value in args_dict.items():
                if key == "expressions":
                    continue  # Already handled above
                if isinstance(value, exp.Literal):
                    param_value = value.this
                    parameters[key] = param_value
                    # For external TVFs, extract the source path
                    if tvf_type == TVFType.EXTERNAL and key == "this":
                        external_source = str(param_value)
                elif isinstance(value, exp.Column):
                    col_ref = f"{value.table}.{value.name}" if value.table else value.name
                    input_columns.append(col_ref)
                    parameters[key] = col_ref
                elif value is not None and key != "this":
                    # Skip 'this' for non-external TVFs (it's often None or internal)
                    parameters[key] = str(value)

        # Get default output columns if not provided via alias
        output_columns = column_aliases if column_aliases else []
        if not output_columns:
            output_columns = TVF_DEFAULT_COLUMNS.get(func_name, ["value"])

        return TVFInfo(
            function_name=func_name,
            tvf_type=tvf_type,
            alias=alias,
            output_columns=output_columns,
            parameters=parameters,
            input_columns=input_columns,
            external_source=external_source,
        )

    def process_tvf(self, source_node: exp.Table, parent_unit: "QueryUnit"):
        """Process a Table-Valued Function in FROM clause."""
        inner_expr = source_node.this

        # Get alias
        alias = str(source_node.alias) if source_node.alias else None
        if not alias:
            alias = f"_tvf_{self._parser.subquery_counter}"
            self._parser.subquery_counter += 1

        # Extract column aliases from TableAlias (e.g., AS t(col1, col2))
        column_aliases: List[str] = []
        alias_node = source_node.args.get("alias")
        if alias_node and hasattr(alias_node, "columns") and alias_node.columns:
            column_aliases = [col.name for col in alias_node.columns if hasattr(col, "name")]

        # Extract TVF info
        tvf_info = self._extract_tvf_info(inner_expr, alias, column_aliases)

        # Store in parent unit
        parent_unit.tvf_sources[alias] = tvf_info

        # Also register alias mapping so columns can be resolved
        # TVFs are like virtual tables, so we map alias to itself with is_unit=False
        parent_unit.alias_mapping[alias] = (alias, False)

    def _extract_literal(self, expr: exp.Expression) -> Any:
        """Extract literal value from expression."""
        if isinstance(expr, exp.Literal):
            if expr.is_int:
                return int(expr.this)
            elif expr.is_number:
                return float(expr.this)
            elif expr.is_string:
                return expr.this
            return expr.this
        elif isinstance(expr, exp.Boolean):
            return expr.this
        elif isinstance(expr, exp.Null):
            return None
        # Complex expression - store as string
        return expr.sql()

    def _infer_value_types(self, rows: List[List[Any]]) -> List[str]:
        """Infer column types from sample values."""
        if not rows:
            return []

        num_cols = len(rows[0])
        types: List[str] = []

        for col_idx in range(num_cols):
            col_values = [
                row[col_idx] for row in rows if col_idx < len(row) and row[col_idx] is not None
            ]

            if not col_values:
                types.append("unknown")
            elif all(isinstance(v, bool) for v in col_values):
                types.append("boolean")
            elif all(isinstance(v, int) for v in col_values):
                types.append("integer")
            elif all(isinstance(v, (int, float)) for v in col_values):
                types.append("numeric")
            else:
                types.append("string")

        return types

    def process_values(
        self, values_node: exp.Values, alias: str, column_aliases: List[str]
    ) -> ValuesInfo:
        """Process a VALUES clause and extract its information."""
        rows: List[List[Any]] = []

        # Parse each row (tuple)
        for row_expr in values_node.expressions:
            if isinstance(row_expr, exp.Tuple):
                row = [self._extract_literal(v) for v in row_expr.expressions]
                rows.append(row)

        # If no column aliases provided, generate defaults
        num_cols = len(rows[0]) if rows else 0
        if not column_aliases and num_cols > 0:
            column_aliases = [f"column{i + 1}" for i in range(num_cols)]

        # Infer column types
        column_types = self._infer_value_types(rows)

        return ValuesInfo(
            alias=alias,
            column_names=column_aliases,
            row_count=len(rows),
            column_types=column_types,
            sample_values=rows[:3],  # Keep first 3 rows as sample
        )

    def handle_values_in_subquery(
        self, source_node: exp.Subquery, parent_unit: "QueryUnit"
    ) -> bool:
        """Handle VALUES clause wrapped in a Subquery.

        Returns:
            True if VALUES was processed, False otherwise
        """
        inner = source_node.this
        if not isinstance(inner, exp.Values):
            return False

        # Get alias
        alias = source_node.alias_or_name if hasattr(source_node, "alias") else None
        if not alias:
            alias = f"_values_{self._parser.subquery_counter}"
            self._parser.subquery_counter += 1

        # Extract column aliases from TableAlias (e.g., AS t(col1, col2))
        column_aliases: List[str] = []
        alias_node = source_node.args.get("alias")
        if alias_node and hasattr(alias_node, "columns") and alias_node.columns:
            column_aliases = [col.name for col in alias_node.columns if hasattr(col, "name")]

        # Process VALUES
        values_info = self.process_values(inner, alias, column_aliases)

        # Store in parent unit
        parent_unit.values_sources[alias] = values_info

        # Add alias mapping so columns can be resolved
        parent_unit.alias_mapping[alias] = (alias, False)

        return True

    def process_values_direct(self, values_node: exp.Values, parent_unit: "QueryUnit"):
        """Process VALUES clause directly in FROM (not wrapped in Subquery)."""
        # Get alias
        alias = str(values_node.alias) if values_node.alias else None
        if not alias:
            alias = f"_values_{self._parser.subquery_counter}"
            self._parser.subquery_counter += 1

        # Extract column aliases from alias node
        column_aliases: List[str] = []
        alias_node = values_node.args.get("alias")
        if alias_node and hasattr(alias_node, "columns") and alias_node.columns:
            column_aliases = [col.name for col in alias_node.columns if hasattr(col, "name")]

        # Process VALUES
        values_info = self.process_values(values_node, alias, column_aliases)

        # Store in parent unit
        parent_unit.values_sources[alias] = values_info

        # Add alias mapping so columns can be resolved
        parent_unit.alias_mapping[alias] = (alias, False)
