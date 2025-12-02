"""
Tests for data models with new UNION and PIVOT/UNPIVOT support.
"""

from clgraph.models import QueryUnit, QueryUnitGraph, QueryUnitType


class TestQueryUnitTypeEnum:
    """Test QueryUnitType enum has all required values"""

    def test_main_query_type_exists(self):
        """Test MAIN_QUERY type exists"""
        assert QueryUnitType.MAIN_QUERY.value == "main_query"

    def test_cte_type_exists(self):
        """Test CTE type exists"""
        assert QueryUnitType.CTE.value == "cte"

    def test_union_type_exists(self):
        """Test UNION type exists"""
        assert QueryUnitType.UNION.value == "union"

    def test_intersect_type_exists(self):
        """Test INTERSECT type exists"""
        assert QueryUnitType.INTERSECT.value == "intersect"

    def test_except_type_exists(self):
        """Test EXCEPT type exists"""
        assert QueryUnitType.EXCEPT.value == "except"

    def test_subquery_union_type_exists(self):
        """Test SUBQUERY_UNION type exists"""
        assert QueryUnitType.SUBQUERY_UNION.value == "subquery_union"

    def test_pivot_type_exists(self):
        """Test PIVOT type exists"""
        assert QueryUnitType.PIVOT.value == "pivot"

    def test_unpivot_type_exists(self):
        """Test UNPIVOT type exists"""
        assert QueryUnitType.UNPIVOT.value == "unpivot"

    def test_subquery_pivot_source_type_exists(self):
        """Test SUBQUERY_PIVOT_SOURCE type exists"""
        assert QueryUnitType.SUBQUERY_PIVOT_SOURCE.value == "subquery_pivot_source"


class TestQueryUnitSetOperations:
    """Test QueryUnit with set operation fields"""

    def test_query_unit_set_operation_defaults(self):
        """Test set operation fields have correct defaults"""
        unit = QueryUnit(
            unit_id="test",
            unit_type=QueryUnitType.UNION,
            name="test_union",
            select_node=None,
            parent_unit=None,
        )

        assert unit.set_operation_type is None
        assert unit.set_operation_branches == []

    def test_query_unit_union_all_metadata(self):
        """Test UNION ALL metadata storage"""
        unit = QueryUnit(
            unit_id="setop:main",
            unit_type=QueryUnitType.UNION,
            name="main",
            select_node=None,
            parent_unit=None,
            set_operation_type="union_all",
            set_operation_branches=["subq:main_branch_0", "subq:main_branch_1"],
        )

        assert unit.set_operation_type == "union_all"
        assert len(unit.set_operation_branches) == 2
        assert "subq:main_branch_0" in unit.set_operation_branches
        assert "subq:main_branch_1" in unit.set_operation_branches

    def test_query_unit_union_distinct_metadata(self):
        """Test UNION (distinct) metadata storage"""
        unit = QueryUnit(
            unit_id="setop:main",
            unit_type=QueryUnitType.UNION,
            name="main",
            select_node=None,
            parent_unit=None,
            set_operation_type="union",
            set_operation_branches=["subq:main_branch_0", "subq:main_branch_1"],
        )

        assert unit.set_operation_type == "union"

    def test_query_unit_intersect_metadata(self):
        """Test INTERSECT metadata storage"""
        unit = QueryUnit(
            unit_id="setop:main",
            unit_type=QueryUnitType.INTERSECT,
            name="main",
            select_node=None,
            parent_unit=None,
            set_operation_type="intersect",
            set_operation_branches=["subq:main_branch_0", "subq:main_branch_1"],
        )

        assert unit.set_operation_type == "intersect"
        assert unit.unit_type == QueryUnitType.INTERSECT

    def test_query_unit_except_metadata(self):
        """Test EXCEPT metadata storage"""
        unit = QueryUnit(
            unit_id="setop:main",
            unit_type=QueryUnitType.EXCEPT,
            name="main",
            select_node=None,
            parent_unit=None,
            set_operation_type="except",
            set_operation_branches=["subq:main_branch_0", "subq:main_branch_1"],
        )

        assert unit.set_operation_type == "except"
        assert unit.unit_type == QueryUnitType.EXCEPT


class TestQueryUnitPivotOperations:
    """Test QueryUnit with PIVOT/UNPIVOT fields"""

    def test_query_unit_pivot_defaults(self):
        """Test PIVOT fields have correct defaults"""
        unit = QueryUnit(
            unit_id="test",
            unit_type=QueryUnitType.MAIN_QUERY,
            name="test",
            select_node=None,
            parent_unit=None,
        )

        assert unit.pivot_config is None
        assert unit.unpivot_config is None

    def test_query_unit_pivot_config(self):
        """Test PIVOT configuration storage"""
        pivot_config = {
            "pivot_column": "quarter",
            "aggregations": ["SUM(revenue)"],
            "value_columns": ["Q1", "Q2", "Q3", "Q4"],
        }

        unit = QueryUnit(
            unit_id="main",
            unit_type=QueryUnitType.MAIN_QUERY,
            name="main",
            select_node=None,
            parent_unit=None,
            pivot_config=pivot_config,
        )

        assert unit.pivot_config is not None
        assert unit.pivot_config["pivot_column"] == "quarter"
        assert unit.pivot_config["aggregations"] == ["SUM(revenue)"]
        assert len(unit.pivot_config["value_columns"]) == 4
        assert "Q1" in unit.pivot_config["value_columns"]

    def test_query_unit_unpivot_config(self):
        """Test UNPIVOT configuration storage"""
        unpivot_config = {
            "value_column": "revenue",
            "unpivot_columns": ["q1", "q2", "q3", "q4"],
            "name_column": "quarter",
        }

        unit = QueryUnit(
            unit_id="main",
            unit_type=QueryUnitType.MAIN_QUERY,
            name="main",
            select_node=None,
            parent_unit=None,
            unpivot_config=unpivot_config,
        )

        assert unit.unpivot_config is not None
        assert unit.unpivot_config["value_column"] == "revenue"
        assert unit.unpivot_config["name_column"] == "quarter"
        assert len(unit.unpivot_config["unpivot_columns"]) == 4
        assert "q1" in unit.unpivot_config["unpivot_columns"]


class TestQueryUnitGraphSetOperations:
    """Test QueryUnitGraph with set operation units"""

    def test_union_unit_as_main_unit(self):
        """Test UNION unit can be set as main unit"""
        graph = QueryUnitGraph()

        union_unit = QueryUnit(
            unit_id="setop:main",
            unit_type=QueryUnitType.UNION,
            name="main",
            select_node=None,
            parent_unit=None,
        )

        graph.add_unit(union_unit)

        assert graph.main_unit_id == "setop:main"
        assert "setop:main" in graph.units

    def test_intersect_unit_as_main_unit(self):
        """Test INTERSECT unit can be set as main unit"""
        graph = QueryUnitGraph()

        intersect_unit = QueryUnit(
            unit_id="setop:main",
            unit_type=QueryUnitType.INTERSECT,
            name="main",
            select_node=None,
            parent_unit=None,
        )

        graph.add_unit(intersect_unit)

        assert graph.main_unit_id == "setop:main"

    def test_except_unit_as_main_unit(self):
        """Test EXCEPT unit can be set as main unit"""
        graph = QueryUnitGraph()

        except_unit = QueryUnit(
            unit_id="setop:main",
            unit_type=QueryUnitType.EXCEPT,
            name="main",
            select_node=None,
            parent_unit=None,
        )

        graph.add_unit(except_unit)

        assert graph.main_unit_id == "setop:main"

    def test_union_with_branches(self):
        """Test UNION graph with branch units"""
        graph = QueryUnitGraph()

        # Create UNION unit
        union_unit = QueryUnit(
            unit_id="setop:main",
            unit_type=QueryUnitType.UNION,
            name="main",
            select_node=None,
            parent_unit=None,
            set_operation_type="union_all",
            set_operation_branches=["subq:main_branch_0", "subq:main_branch_1"],
        )

        # Create branch units
        branch_0 = QueryUnit(
            unit_id="subq:main_branch_0",
            unit_type=QueryUnitType.SUBQUERY_UNION,
            name="main_branch_0",
            select_node=None,
            parent_unit=union_unit,
        )

        branch_1 = QueryUnit(
            unit_id="subq:main_branch_1",
            unit_type=QueryUnitType.SUBQUERY_UNION,
            name="main_branch_1",
            select_node=None,
            parent_unit=union_unit,
        )

        graph.add_unit(union_unit)
        graph.add_unit(branch_0)
        graph.add_unit(branch_1)

        # Verify structure
        assert graph.main_unit_id == "setop:main"
        assert len(graph.units) == 3
        assert len(union_unit.set_operation_branches) == 2

    def test_topological_order_with_union(self):
        """Test topological ordering works with UNION units"""
        graph = QueryUnitGraph()

        # Create UNION unit that depends on branches
        union_unit = QueryUnit(
            unit_id="setop:main",
            unit_type=QueryUnitType.UNION,
            name="main",
            select_node=None,
            parent_unit=None,
            depends_on_units=["subq:main_branch_0", "subq:main_branch_1"],
        )

        # Branch 0 depends on a table
        branch_0 = QueryUnit(
            unit_id="subq:main_branch_0",
            unit_type=QueryUnitType.SUBQUERY_UNION,
            name="main_branch_0",
            select_node=None,
            parent_unit=union_unit,
            depends_on_tables=["users"],
        )

        # Branch 1 depends on a table
        branch_1 = QueryUnit(
            unit_id="subq:main_branch_1",
            unit_type=QueryUnitType.SUBQUERY_UNION,
            name="main_branch_1",
            select_node=None,
            parent_unit=union_unit,
            depends_on_tables=["archived_users"],
        )

        graph.add_unit(branch_0)
        graph.add_unit(branch_1)
        graph.add_unit(union_unit)

        # Get topological order
        ordered = graph.get_topological_order()

        # Branches should come before union
        branch_indices = [
            i
            for i, u in enumerate(ordered)
            if u.unit_id in ["subq:main_branch_0", "subq:main_branch_1"]
        ]
        union_index = [i for i, u in enumerate(ordered) if u.unit_id == "setop:main"][0]

        assert all(idx < union_index for idx in branch_indices)
