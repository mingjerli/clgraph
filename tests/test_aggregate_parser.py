"""
Unit tests for aggregate_parser module.

Tests the 5 pure functions extracted from RecursiveLineageBuilder:
- parse_aggregate_spec
- get_aggregate_func_name
- infer_aggregate_return_type
- has_star_in_aggregate
- unit_has_fully_resolved_columns
"""

from typing import Optional
from unittest.mock import MagicMock

import sqlglot  # noqa: I001
from sqlglot import exp

from clgraph.aggregate_parser import (
    get_aggregate_func_name,
    has_star_in_aggregate,
    infer_aggregate_return_type,
    parse_aggregate_spec,
    unit_has_fully_resolved_columns,
)
from clgraph.models import AggregateType, QueryUnit, QueryUnitType

# ============================================================================
# Helpers
# ============================================================================


def _parse_expr(sql: str, dialect: str | None = None) -> exp.Expression:
    """Parse a SQL expression and return the first SELECT expression."""
    stmt = sqlglot.parse_one(f"SELECT {sql} FROM t", dialect=dialect)
    return stmt.expressions[0]  # type: ignore[union-attr]


def _find_agg_node(sql: str, dialect: str | None = None) -> exp.Expression:
    """Parse a SQL expression and find the first aggregate node."""
    stmt = sqlglot.parse_one(f"SELECT {sql} FROM t", dialect=dialect)
    expr = stmt.expressions[0]  # type: ignore[union-attr]
    for node in expr.walk():
        if isinstance(node, (exp.AggFunc, exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max, exp.ArrayAgg, exp.GroupConcat)):
            return node
    raise ValueError(f"No aggregate node found in: {sql}")


# ============================================================================
# Tests: parse_aggregate_spec
# ============================================================================


class TestParseAggregateSpec:
    """Tests for parse_aggregate_spec function."""

    def test_returns_none_for_none_input(self):
        result = parse_aggregate_spec(None)
        assert result is None

    def test_returns_none_for_non_aggregate_expression(self):
        expr = _parse_expr("col1 + col2")
        result = parse_aggregate_spec(expr)
        assert result is None

    def test_sum_basic(self):
        expr = _parse_expr("SUM(amount)")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.function_name == "SUM"
        assert result.aggregate_type == AggregateType.SCALAR
        assert result.return_type == "numeric"
        assert "amount" in result.value_columns

    def test_count_basic(self):
        expr = _parse_expr("COUNT(id)")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.function_name == "COUNT"
        assert result.aggregate_type == AggregateType.SCALAR
        assert result.return_type == "integer"

    def test_count_star(self):
        expr = _parse_expr("COUNT(*)")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.function_name == "COUNT"
        assert result.return_type == "integer"

    def test_count_distinct(self):
        expr = _parse_expr("COUNT(DISTINCT user_id)")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.function_name == "COUNT"
        assert result.distinct is True

    def test_array_agg_basic(self):
        expr = _parse_expr("ARRAY_AGG(product_id)", dialect="bigquery")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.function_name == "ARRAY_AGG"
        assert result.aggregate_type == AggregateType.ARRAY
        assert result.return_type == "array"
        assert "product_id" in result.value_columns

    def test_array_agg_with_order_by(self):
        expr = _parse_expr("ARRAY_AGG(product_id ORDER BY purchase_date)", dialect="bigquery")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.function_name == "ARRAY_AGG"
        assert result.aggregate_type == AggregateType.ARRAY
        assert len(result.order_by) > 0
        assert result.order_by[0].column == "purchase_date"
        assert result.order_by[0].direction == "asc"

    def test_string_agg_basic(self):
        expr = _parse_expr("STRING_AGG(name, ',')", dialect="bigquery")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.aggregate_type == AggregateType.STRING
        assert result.return_type == "string"

    def test_avg(self):
        expr = _parse_expr("AVG(score)")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.function_name == "AVG"
        assert result.return_type == "float"

    def test_min_max(self):
        for func in ("MIN", "MAX"):
            expr = _parse_expr(f"{func}(value)")
            result = parse_aggregate_spec(expr)

            assert result is not None
            assert result.function_name == func
            assert result.return_type == "numeric"

    def test_value_columns_extracted(self):
        expr = _parse_expr("SUM(revenue)")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert "revenue" in result.value_columns

    def test_distinct_flag_set(self):
        expr = _parse_expr("COUNT(DISTINCT customer_id)")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.distinct is True

    def test_non_distinct_flag_false(self):
        expr = _parse_expr("SUM(amount)")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.distinct is False


# ============================================================================
# Tests: get_aggregate_func_name
# ============================================================================


class TestGetAggregateFuncName:
    """Tests for get_aggregate_func_name function."""

    def test_sum_node(self):
        node = _find_agg_node("SUM(x)")
        assert get_aggregate_func_name(node) == "SUM"

    def test_count_node(self):
        node = _find_agg_node("COUNT(*)")
        assert get_aggregate_func_name(node) == "COUNT"

    def test_avg_node(self):
        node = _find_agg_node("AVG(score)")
        assert get_aggregate_func_name(node) == "AVG"

    def test_min_node(self):
        node = _find_agg_node("MIN(val)")
        assert get_aggregate_func_name(node) == "MIN"

    def test_max_node(self):
        node = _find_agg_node("MAX(val)")
        assert get_aggregate_func_name(node) == "MAX"

    def test_array_agg_node(self):
        node = _find_agg_node("ARRAY_AGG(x)", dialect="bigquery")
        assert get_aggregate_func_name(node) == "ARRAY_AGG"

    def test_group_concat_node(self):
        node = _find_agg_node("GROUP_CONCAT(x)", dialect="mysql")
        assert get_aggregate_func_name(node) == "GROUP_CONCAT"

    def test_fallback_for_unknown_node(self):
        # A node with sql_name but not a specific type
        mock_node = MagicMock()
        mock_node.sql_name.return_value = "custom_agg"
        del mock_node.name  # Ensure .name is not easily accessible
        # Remove special type matching
        result = get_aggregate_func_name(mock_node)
        assert isinstance(result, str)


# ============================================================================
# Tests: infer_aggregate_return_type
# ============================================================================


class TestInferAggregateReturnType:
    """Tests for infer_aggregate_return_type function."""

    def test_count_returns_integer(self):
        assert infer_aggregate_return_type("COUNT", []) == "integer"
        assert infer_aggregate_return_type("count", []) == "integer"

    def test_sum_returns_numeric(self):
        assert infer_aggregate_return_type("SUM", ["amount"]) == "numeric"
        assert infer_aggregate_return_type("sum", ["amount"]) == "numeric"

    def test_min_max_return_numeric(self):
        assert infer_aggregate_return_type("MIN", ["val"]) == "numeric"
        assert infer_aggregate_return_type("MAX", ["val"]) == "numeric"

    def test_avg_returns_float(self):
        assert infer_aggregate_return_type("AVG", ["score"]) == "float"
        assert infer_aggregate_return_type("avg", ["score"]) == "float"

    def test_array_agg_returns_array(self):
        assert infer_aggregate_return_type("ARRAY_AGG", ["col"]) == "array"
        assert infer_aggregate_return_type("array_agg", ["col"]) == "array"

    def test_collect_list_returns_array(self):
        assert infer_aggregate_return_type("collect_list", ["col"]) == "array"

    def test_collect_set_returns_array(self):
        assert infer_aggregate_return_type("collect_set", ["col"]) == "array"

    def test_string_agg_returns_string(self):
        assert infer_aggregate_return_type("string_agg", ["name"]) == "string"

    def test_listagg_returns_string(self):
        assert infer_aggregate_return_type("listagg", ["name"]) == "string"

    def test_group_concat_returns_string(self):
        assert infer_aggregate_return_type("group_concat", ["name"]) == "string"

    def test_object_agg_returns_object(self):
        assert infer_aggregate_return_type("object_agg", ["k", "v"]) == "object"

    def test_json_agg_returns_object(self):
        assert infer_aggregate_return_type("json_agg", ["data"]) == "object"

    def test_stddev_returns_float(self):
        assert infer_aggregate_return_type("stddev", ["val"]) == "float"

    def test_variance_returns_float(self):
        assert infer_aggregate_return_type("variance", ["val"]) == "float"

    def test_unknown_returns_any(self):
        assert infer_aggregate_return_type("my_custom_agg", ["col"]) == "any"

    def test_percentile_cont_returns_float(self):
        assert infer_aggregate_return_type("percentile_cont", ["val"]) == "float"


# ============================================================================
# Tests: has_star_in_aggregate
# ============================================================================


class TestHasStarInAggregate:
    """Tests for has_star_in_aggregate function."""

    def test_none_returns_false(self):
        assert has_star_in_aggregate(None) is False

    def test_count_star_returns_true(self):
        expr = _parse_expr("COUNT(*)")
        assert has_star_in_aggregate(expr) is True

    def test_count_column_returns_false(self):
        expr = _parse_expr("COUNT(id)")
        assert has_star_in_aggregate(expr) is False

    def test_sum_returns_false(self):
        expr = _parse_expr("SUM(amount)")
        assert has_star_in_aggregate(expr) is False

    def test_plain_column_returns_false(self):
        expr = _parse_expr("my_col")
        assert has_star_in_aggregate(expr) is False

    def test_arithmetic_expression_returns_false(self):
        expr = _parse_expr("price * quantity")
        assert has_star_in_aggregate(expr) is False


# ============================================================================
# Tests: unit_has_fully_resolved_columns
# ============================================================================


class TestUnitHasFullyResolvedColumns:
    """Tests for unit_has_fully_resolved_columns function."""

    def _make_unit(self, select_sql: Optional[str]) -> QueryUnit:
        """Create a minimal QueryUnit with the given SELECT."""
        if select_sql is None:
            return QueryUnit(
                unit_id="test_unit",
                unit_type=QueryUnitType.MAIN_QUERY,
                name="test_unit",
                select_node=None,
                parent_unit=None,
            )
        stmt = sqlglot.parse_one(f"SELECT {select_sql} FROM t")
        return QueryUnit(
            unit_id="test_unit",
            unit_type=QueryUnitType.MAIN_QUERY,
            name="test_unit",
            select_node=stmt,  # type: ignore[arg-type]
            parent_unit=None,
        )

    def test_none_select_node_returns_true(self):
        """Units without select_node (UNION, PIVOT) are considered resolved."""
        unit = self._make_unit(None)
        assert unit_has_fully_resolved_columns(unit) is True

    def test_explicit_columns_returns_true(self):
        unit = self._make_unit("id, name, email")
        assert unit_has_fully_resolved_columns(unit) is True

    def test_star_select_returns_false(self):
        unit = self._make_unit("*")
        assert unit_has_fully_resolved_columns(unit) is False

    def test_qualified_star_returns_false(self):
        unit = self._make_unit("t.*")
        assert unit_has_fully_resolved_columns(unit) is False

    def test_single_explicit_column_returns_true(self):
        unit = self._make_unit("id")
        assert unit_has_fully_resolved_columns(unit) is True

    def test_aggregate_without_star_returns_true(self):
        unit = self._make_unit("SUM(amount)")
        assert unit_has_fully_resolved_columns(unit) is True


# ============================================================================
# Additional coverage tests
# ============================================================================


class TestParseAggregateSpecCoverage:
    """Additional tests to cover more code paths in parse_aggregate_spec."""

    def test_array_agg_with_order_by_desc(self):
        """Test ARRAY_AGG with descending ORDER BY."""
        expr = _parse_expr("ARRAY_AGG(product_id ORDER BY purchase_date DESC)", dialect="bigquery")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.function_name == "ARRAY_AGG"
        assert len(result.order_by) > 0
        assert result.order_by[0].direction == "desc"

    def test_array_agg_with_qualified_order_by(self):
        """Test ARRAY_AGG with table-qualified ORDER BY column."""
        expr = _parse_expr(
            "ARRAY_AGG(t.product_id ORDER BY t.purchase_date)", dialect="bigquery"
        )
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.function_name == "ARRAY_AGG"

    def test_group_concat_with_separator(self):
        """Test GROUP_CONCAT with separator (covers GroupConcat separator path)."""
        expr = _parse_expr("GROUP_CONCAT(name SEPARATOR ',')", dialect="mysql")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.aggregate_type == AggregateType.STRING

    def test_string_agg_returns_string_type(self):
        """Test STRING_AGG captures separator via expressions path."""
        expr = _parse_expr("LISTAGG(name, ', ')", dialect="snowflake")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.aggregate_type == AggregateType.STRING

    def test_sum_with_table_qualified_column(self):
        """Test SUM with table.column reference."""
        expr = _parse_expr("SUM(t.amount)")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.function_name == "SUM"
        # Table-qualified columns are included
        assert any("amount" in col for col in result.value_columns)

    def test_object_agg_key_value_extraction(self):
        """Test OBJECT_AGG extracts key and value columns."""
        expr = _parse_expr("OBJECT_AGG(k, v)", dialect="snowflake")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.aggregate_type == AggregateType.OBJECT
        assert result.key_columns == ["k"]
        assert result.value_columns == ["v"]

    def test_group_concat_mysql(self):
        """Test GROUP_CONCAT in mysql dialect parses correctly."""
        expr = _parse_expr("GROUP_CONCAT(name ORDER BY name ASC SEPARATOR ', ')", dialect="mysql")
        result = parse_aggregate_spec(expr)

        assert result is not None
        assert result.aggregate_type == AggregateType.STRING


class TestGetAggregateFuncNameCoverage:
    """Additional coverage for get_aggregate_func_name fallback paths."""

    def test_generic_agg_func_with_sql_name(self):
        """Test that generic AggFunc nodes use sql_name()."""
        # BOOL_AND is an AggFunc subclass in some dialects
        try:
            node = _find_agg_node("BOOL_AND(flag)")
            result = get_aggregate_func_name(node)
            assert isinstance(result, str)
            assert len(result) > 0
        except ValueError:
            pass  # Skip if not parseable

    def test_node_with_name_but_no_sql_name(self):
        """Test node with .name but no .sql_name -> uses .name.upper()."""
        mock_node = MagicMock(spec=[])  # Empty spec: no sql_name, no standard attrs
        mock_node.name = "custom_agg"
        result = get_aggregate_func_name(mock_node)
        assert result == "CUSTOM_AGG"

    def test_node_with_no_name_no_sql_name(self):
        """Test node with neither .name nor .sql_name -> returns 'AGGREGATE'."""
        mock_node = MagicMock(spec=[])  # No attributes at all
        result = get_aggregate_func_name(mock_node)
        assert result == "AGGREGATE"


class TestInferAggregateReturnTypeCoverage:
    """Additional coverage for infer_aggregate_return_type."""

    def test_var_pop_returns_float(self):
        assert infer_aggregate_return_type("var_pop", []) == "float"

    def test_var_samp_returns_float(self):
        assert infer_aggregate_return_type("var_samp", []) == "float"

    def test_jsonb_agg_returns_object(self):
        assert infer_aggregate_return_type("jsonb_agg", ["data"]) == "object"
