"""
Tests for node_factory.py — pure factory functions for creating lineage nodes/edges.
"""

import pytest

from clgraph.models import (
    ColumnLineageGraph,
    QueryUnit,
    QueryUnitType,
    TVFInfo,
    TVFType,
    ValuesInfo,
)
from clgraph.node_factory import (
    find_or_create_star_node,
    find_or_create_table_column_node,
    find_or_create_table_star_node,
    find_or_create_tvf_column_node,
    find_or_create_values_column_node,
    get_layer_for_unit,
    get_node_key,
    resolve_external_table_name,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def graph():
    return ColumnLineageGraph()


@pytest.fixture
def main_unit():
    return QueryUnit(
        unit_id="main",
        unit_type=QueryUnitType.MAIN_QUERY,
        name=None,
        select_node=None,
        parent_unit=None,
    )


@pytest.fixture
def cte_unit():
    return QueryUnit(
        unit_id="cte_1",
        unit_type=QueryUnitType.CTE,
        name="my_cte",
        select_node=None,
        parent_unit=None,
    )


@pytest.fixture
def subquery_unit():
    return QueryUnit(
        unit_id="sub_1",
        unit_type=QueryUnitType.SUBQUERY_FROM,
        name="sub",
        select_node=None,
        parent_unit=None,
    )


@pytest.fixture
def col_info():
    return {
        "name": "revenue",
        "expression": "revenue",
        "type": "passthrough",
        "source_columns": [],
    }


# ============================================================================
# Tests: get_layer_for_unit
# ============================================================================


class TestGetLayerForUnit:
    def test_main_query_returns_output(self, main_unit):
        assert get_layer_for_unit(main_unit) == "output"

    def test_cte_returns_cte(self, cte_unit):
        assert get_layer_for_unit(cte_unit) == "cte"

    def test_subquery_from_returns_subquery(self, subquery_unit):
        assert get_layer_for_unit(subquery_unit) == "subquery"


# ============================================================================
# Tests: get_node_key
# ============================================================================


class TestGetNodeKey:
    def test_main_query_uses_output_prefix(self, main_unit, col_info):
        key = get_node_key(main_unit, col_info)
        assert key == "output.revenue"

    def test_cte_uses_cte_name(self, cte_unit, col_info):
        key = get_node_key(cte_unit, col_info)
        assert key == "my_cte.revenue"

    def test_subquery_uses_unit_name(self, subquery_unit, col_info):
        key = get_node_key(subquery_unit, col_info)
        assert key == "sub.revenue"

    def test_returns_string(self, main_unit, col_info):
        assert isinstance(get_node_key(main_unit, col_info), str)


# ============================================================================
# Tests: find_or_create_star_node
# ============================================================================


class TestFindOrCreateStarNode:
    def test_creates_star_node(self, graph, cte_unit):
        node = find_or_create_star_node(graph, cte_unit, "source_t")
        assert node is not None
        assert node.column_name == "*"
        assert node.is_star is True

    def test_returns_same_node_on_second_call(self, graph, cte_unit):
        node1 = find_or_create_star_node(graph, cte_unit, "source_t")
        node2 = find_or_create_star_node(graph, cte_unit, "source_t")
        assert node1 is node2

    def test_node_added_to_graph(self, graph, cte_unit):
        find_or_create_star_node(graph, cte_unit, "source_t")
        assert "my_cte.*" in graph.nodes

    def test_node_layer_matches_unit(self, graph, cte_unit):
        node = find_or_create_star_node(graph, cte_unit, "source_t")
        assert node.layer == "cte"


# ============================================================================
# Tests: resolve_external_table_name
# ============================================================================


class TestResolveExternalTableName:
    def test_direct_match(self):
        external = {"events": ["id", "ts"], "users": ["id", "name"]}
        assert resolve_external_table_name(external, "events") == "events"

    def test_suffix_match(self):
        external = {"staging.events": ["id", "ts"]}
        assert resolve_external_table_name(external, "events") == "staging.events"

    def test_returns_none_if_not_found(self):
        external = {"staging.events": ["id", "ts"]}
        assert resolve_external_table_name(external, "orders") is None

    def test_empty_dict(self):
        assert resolve_external_table_name({}, "events") is None


# ============================================================================
# Tests: find_or_create_table_star_node
# ============================================================================


class TestFindOrCreateTableStarNode:
    def test_creates_table_star_node(self, graph):
        node = find_or_create_table_star_node(graph, "orders")
        assert node is not None
        assert node.column_name == "*"
        assert node.is_star is True
        assert node.table_name == "orders"

    def test_layer_is_input(self, graph):
        node = find_or_create_table_star_node(graph, "orders")
        assert node.layer == "input"

    def test_returns_same_node_on_second_call(self, graph):
        node1 = find_or_create_table_star_node(graph, "orders")
        node2 = find_or_create_table_star_node(graph, "orders")
        assert node1 is node2

    def test_node_added_to_graph(self, graph):
        find_or_create_table_star_node(graph, "orders")
        assert "orders.*" in graph.nodes


# ============================================================================
# Tests: find_or_create_table_column_node
# ============================================================================


class TestFindOrCreateTableColumnNode:
    def test_creates_column_node(self, graph):
        node = find_or_create_table_column_node(graph, "orders", "amount")
        assert node is not None
        assert node.column_name == "amount"
        assert node.table_name == "orders"

    def test_layer_is_input(self, graph):
        node = find_or_create_table_column_node(graph, "orders", "amount")
        assert node.layer == "input"

    def test_returns_same_node_on_second_call(self, graph):
        node1 = find_or_create_table_column_node(graph, "orders", "amount")
        node2 = find_or_create_table_column_node(graph, "orders", "amount")
        assert node1 is node2

    def test_node_added_to_graph(self, graph):
        find_or_create_table_column_node(graph, "orders", "amount")
        assert "orders.amount" in graph.nodes

    def test_node_type_is_base_column(self, graph):
        node = find_or_create_table_column_node(graph, "orders", "amount")
        assert node.node_type == "base_column"


# ============================================================================
# Tests: find_or_create_tvf_column_node
# ============================================================================


class TestFindOrCreateTvfColumnNode:
    def test_creates_synthetic_tvf_node(self, graph):
        tvf_info = TVFInfo(
            alias="gs",
            function_name="generate_series",
            tvf_type=TVFType.GENERATOR,
            parameters={"start": 1, "end": 10},
            output_columns=["num"],
        )
        node = find_or_create_tvf_column_node(graph, tvf_info, "num")
        assert node is not None
        assert node.column_name == "num"
        assert node.node_type == "tvf_synthetic"
        assert node.is_synthetic is True

    def test_returns_same_node_on_second_call(self, graph):
        tvf_info = TVFInfo(
            alias="gs",
            function_name="generate_series",
            tvf_type=TVFType.GENERATOR,
            parameters={"start": 1, "end": 10},
            output_columns=["num"],
        )
        node1 = find_or_create_tvf_column_node(graph, tvf_info, "num")
        node2 = find_or_create_tvf_column_node(graph, tvf_info, "num")
        assert node1 is node2

    def test_node_added_to_graph(self, graph):
        tvf_info = TVFInfo(
            alias="gs",
            function_name="generate_series",
            tvf_type=TVFType.GENERATOR,
            parameters={"start": 1, "end": 10},
            output_columns=["num"],
        )
        find_or_create_tvf_column_node(graph, tvf_info, "num")
        assert "gs.num" in graph.nodes


# ============================================================================
# Tests: find_or_create_values_column_node
# ============================================================================


class TestFindOrCreateValuesColumnNode:
    def test_creates_literal_values_node(self, graph):
        values_info = ValuesInfo(
            alias="v",
            column_names=["a", "b"],
            sample_values=[["1", "x"], ["2", "y"]],
            column_types=["int", "str"],
        )
        node = find_or_create_values_column_node(graph, values_info, "a")
        assert node is not None
        assert node.column_name == "a"
        assert node.node_type == "literal"
        assert node.is_literal is True

    def test_returns_same_node_on_second_call(self, graph):
        values_info = ValuesInfo(
            alias="v",
            column_names=["a"],
            sample_values=[["1"]],
            column_types=["int"],
        )
        node1 = find_or_create_values_column_node(graph, values_info, "a")
        node2 = find_or_create_values_column_node(graph, values_info, "a")
        assert node1 is node2

    def test_node_added_to_graph(self, graph):
        values_info = ValuesInfo(
            alias="v",
            column_names=["a"],
            sample_values=[["1"]],
            column_types=["int"],
        )
        find_or_create_values_column_node(graph, values_info, "a")
        assert "v.a" in graph.nodes

    def test_extracts_sample_values(self, graph):
        values_info = ValuesInfo(
            alias="v",
            column_names=["a", "b"],
            sample_values=[["10", "foo"], ["20", "bar"]],
            column_types=["int", "str"],
        )
        node = find_or_create_values_column_node(graph, values_info, "a")
        assert node.literal_values == ["10", "20"]
