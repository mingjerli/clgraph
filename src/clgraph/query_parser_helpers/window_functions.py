"""Window function parsing helper.

Handles parsing of window functions including PARTITION BY, ORDER BY,
and frame specifications.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlglot import exp

if TYPE_CHECKING:
    from ..models import QueryUnit
    from ..query_parser import RecursiveQueryParser


class WindowFunctionsParser:
    """Helper for parsing window functions."""

    def __init__(self, parser: "RecursiveQueryParser"):
        """Initialize with reference to parent parser.

        Args:
            parser: The parent RecursiveQueryParser instance
        """
        self._parser = parser

    def parse(self, select_node: exp.Select, unit: "QueryUnit"):
        """
        Parse window functions in SELECT clause to extract:
        - Function name and arguments
        - PARTITION BY columns
        - ORDER BY columns with direction
        - Frame specification (ROWS/RANGE/GROUPS)
        - Named window definitions

        Args:
            select_node: The SELECT expression
            unit: The query unit to store window info
        """
        # 1. Parse named window definitions from WINDOW clause
        window_defs = select_node.args.get("windows", [])
        for window_def in window_defs:
            if isinstance(window_def, exp.Window):
                # Named window: window alias is in 'this' as Identifier
                window_name_node = window_def.args.get("this")
                if isinstance(window_name_node, exp.Identifier):
                    window_name = window_name_node.this
                    spec = self._parse_window_spec(window_def)
                    unit.window_definitions[window_name] = spec

        # 2. Parse window functions in SELECT expressions
        windows = []
        for i, col_expr in enumerate(select_node.expressions):
            # Get output column alias
            alias = None
            if isinstance(col_expr, exp.Alias):
                alias = col_expr.alias
                search_expr = col_expr.this
            else:
                search_expr = col_expr

            # Find window functions in this expression
            for window in search_expr.find_all(exp.Window):
                window_info = self._parse_single_window(window, alias, i, unit)
                if window_info:
                    windows.append(window_info)

        if windows:
            unit.window_info = {"windows": windows}

        # Populate ranking_window_columns for dedup qualify promotion (Gap 2)
        RANKING_FUNCTIONS = {"ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE"}
        for window_def in windows:
            func_name = window_def.get("function", "").upper()
            output_col = window_def.get("output_column")
            if func_name in RANKING_FUNCTIONS and output_col:
                unit.ranking_window_columns[output_col] = {
                    "function": func_name,
                    "partition_by": window_def.get("partition_by", []),
                    "order_by": window_def.get("order_by", []),
                }

    def _parse_single_window(
        self,
        window: exp.Window,
        output_alias: Optional[str],
        col_index: int,
        unit: "QueryUnit",
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a single window function expression.

        Args:
            window: The Window expression node
            output_alias: Alias of the output column (if any)
            col_index: Index of the column in SELECT
            unit: Query unit for resolving named window references

        Returns:
            Dictionary with window function details
        """
        # Get function name and arguments
        func = window.args.get("this")
        if func is None:
            return None

        func_name = ""
        func_args = []

        if hasattr(func, "sql_name"):
            func_name = func.sql_name()
        elif hasattr(func, "name"):
            func_name = func.name

        # Extract function arguments
        # Most aggregate functions have a single argument in 'this'
        if hasattr(func, "this") and func.this:
            arg_cols = self._extract_window_columns(func.this)
            func_args.extend(arg_cols)
        # Some functions (like NTILE) may have additional arguments in 'expressions'
        if hasattr(func, "expressions") and func.expressions:
            for arg in func.expressions:
                arg_cols = self._extract_window_columns(arg)
                func_args.extend(arg_cols)

        # Check for named window reference
        window_alias = window.alias
        base_spec: Dict[str, Any] = {}

        if window_alias and window_alias in unit.window_definitions:
            # Resolve named window reference
            base_spec = unit.window_definitions[window_alias].copy()

        # Parse inline window spec (may override or extend named window)
        inline_spec = self._parse_window_spec(window)

        # Merge: inline spec overrides base spec
        spec = {**base_spec, **inline_spec}

        # Build result
        output_column = output_alias if output_alias else f"_col{col_index}"

        return {
            "output_column": output_column,
            "function": func_name,
            "arguments": func_args,
            "partition_by": spec.get("partition_by", []),
            "order_by": spec.get("order_by", []),
            "frame_type": spec.get("frame_type"),
            "frame_start": spec.get("frame_start"),
            "frame_end": spec.get("frame_end"),
            "window_name": window_alias if window_alias else None,
        }

    def _parse_window_spec(self, window: exp.Window) -> Dict[str, Any]:
        """
        Parse window specification from a Window expression.

        Args:
            window: The Window expression node

        Returns:
            Dictionary with partition_by, order_by, and frame details
        """
        result: Dict[str, Any] = {}

        # Parse PARTITION BY
        partition_by = window.args.get("partition_by", [])
        if partition_by:
            partition_cols = []
            for expr in partition_by:
                cols = self._extract_window_columns(expr)
                partition_cols.extend(cols)
            result["partition_by"] = partition_cols

        # Parse ORDER BY
        order = window.args.get("order")
        if order:
            order_cols = []
            for order_expr in order.expressions:
                col_info = self._parse_order_by_column(order_expr)
                order_cols.append(col_info)
            result["order_by"] = order_cols

        # Parse frame specification
        spec = window.args.get("spec")
        if spec:
            frame_info = self._parse_frame_spec(spec)
            result.update(frame_info)

        return result

    def _parse_order_by_column(self, order_expr: exp.Expression) -> Dict[str, Any]:
        """
        Parse ORDER BY column with direction and nulls handling.

        Args:
            order_expr: The Ordered expression or column

        Returns:
            Dictionary with column, direction, and nulls info
        """
        if isinstance(order_expr, exp.Ordered):
            # Extract column(s) from the ordered expression
            cols = self._extract_window_columns(order_expr.this)
            column = cols[0] if cols else str(order_expr.this)

            # Get direction from args (desc is None for ASC, True for DESC)
            desc_val = order_expr.args.get("desc")
            if desc_val is True:
                direction = "desc"
            else:
                direction = "asc"

            # Get nulls handling from args
            nulls_first_val = order_expr.args.get("nulls_first")
            if nulls_first_val is True:
                nulls = "first"
            else:
                nulls = "last"

            return {"column": column, "direction": direction, "nulls": nulls}
        else:
            # Plain column without direction
            cols = self._extract_window_columns(order_expr)
            column = cols[0] if cols else str(order_expr)
            return {"column": column, "direction": "asc", "nulls": "last"}

    def _parse_frame_spec(self, spec: exp.WindowSpec) -> Dict[str, Any]:
        """
        Parse frame specification (ROWS/RANGE/GROUPS BETWEEN ... AND ...).

        Args:
            spec: The WindowSpec expression

        Returns:
            Dictionary with frame_type, frame_start, frame_end
        """
        result: Dict[str, Any] = {}

        # Frame type: ROWS, RANGE, or GROUPS
        kind = spec.args.get("kind")
        if kind:
            result["frame_type"] = kind.lower()

        # Frame start
        start = spec.args.get("start")
        start_side = spec.args.get("start_side")
        if start is not None:
            result["frame_start"] = self._format_frame_boundary(start, start_side)

        # Frame end
        end = spec.args.get("end")
        end_side = spec.args.get("end_side")
        if end is not None:
            result["frame_end"] = self._format_frame_boundary(end, end_side)

        return result

    def _format_frame_boundary(self, boundary: Any, side: Optional[str]) -> str:
        """
        Format a frame boundary as a string.

        Args:
            boundary: The boundary value (UNBOUNDED, number, CURRENT ROW)
            side: PRECEDING or FOLLOWING

        Returns:
            String like "unbounded preceding", "3 preceding", "current row"
        """
        if isinstance(boundary, str):
            # Already a string like "CURRENT ROW" or "UNBOUNDED"
            boundary_str = boundary.lower()
            if boundary_str == "current row":
                return "current row"
            elif boundary_str == "unbounded":
                if side:
                    return f"unbounded {side.lower()}"
                return "unbounded"
            else:
                if side:
                    return f"{boundary_str} {side.lower()}"
                return boundary_str
        elif isinstance(boundary, exp.Literal):
            # Numeric literal
            value = boundary.this
            if side:
                return f"{value} {side.lower()}"
            return str(value)
        elif hasattr(boundary, "this"):
            # Some other expression
            value = str(boundary.this)
            if side:
                return f"{value} {side.lower()}"
            return value
        else:
            return str(boundary)

    def _extract_window_columns(self, expr: exp.Expression) -> List[str]:
        """
        Extract column references from an expression.

        Args:
            expr: The expression to extract columns from

        Returns:
            List of column names (qualified if table is present)
        """
        columns = []
        if isinstance(expr, str):
            # Sometimes expr is a string (e.g., from certain window functions)
            # Just return empty - we can't extract columns from a string
            return columns
        if isinstance(expr, exp.Column):
            table_ref = str(expr.table) if expr.table else None
            col_name = expr.name
            if table_ref:
                columns.append(f"{table_ref}.{col_name}")
            else:
                columns.append(col_name)
        elif hasattr(expr, "find_all"):
            # Walk expression tree for columns
            for col in expr.find_all(exp.Column):
                table_ref = str(col.table) if col.table else None
                col_name = col.name
                if table_ref:
                    full_name = f"{table_ref}.{col_name}"
                else:
                    full_name = col_name
                if full_name not in columns:
                    columns.append(full_name)
        return columns
