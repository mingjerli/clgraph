"""
Tests for UNION/INTERSECT/EXCEPT parsing functionality.
"""

import pytest

from clgraph.models import QueryUnitType
from clgraph.query_parser import RecursiveQueryParser


class TestUnionBasic:
    """Test basic UNION operations"""

    def test_union_all_two_tables(self):
        """Test UNION ALL of two tables"""
        sql = """
        SELECT user_id, name FROM users
        UNION ALL
        SELECT user_id, name FROM archived_users
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Should have 3 units: main union + 2 branches
        assert len(graph.units) == 3

        # Check main unit is UNION
        main_unit = graph.units[graph.main_unit_id]
        assert main_unit.unit_type == QueryUnitType.UNION
        assert main_unit.set_operation_type == "union_all"
        assert len(main_unit.set_operation_branches) == 2

        # Check branches
        branch_0 = graph.units["subq:main_branch_0"]
        branch_1 = graph.units["subq:main_branch_1"]

        assert branch_0.unit_type == QueryUnitType.SUBQUERY_UNION
        assert branch_1.unit_type == QueryUnitType.SUBQUERY_UNION

        assert "users" in branch_0.depends_on_tables
        assert "archived_users" in branch_1.depends_on_tables

    def test_union_distinct_two_tables(self):
        """Test UNION (distinct) of two tables"""
        sql = """
        SELECT id FROM table_a
        UNION DISTINCT
        SELECT id FROM table_b
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Check main unit is UNION with distinct
        main_unit = graph.units[graph.main_unit_id]
        assert main_unit.unit_type == QueryUnitType.UNION
        assert main_unit.set_operation_type == "union"  # not union_all

    def test_union_three_way(self):
        """Test three-way UNION"""
        sql = """
        SELECT id FROM table_a
        UNION ALL
        SELECT id FROM table_b
        UNION ALL
        SELECT id FROM table_c
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Should have 4 units: main union + 3 branches
        assert len(graph.units) == 4

        main_unit = graph.units[graph.main_unit_id]
        assert len(main_unit.set_operation_branches) == 3

        # Verify all three branches exist
        assert "subq:main_branch_0" in graph.units
        assert "subq:main_branch_1" in graph.units
        assert "subq:main_branch_2" in graph.units

        # Verify table dependencies
        branch_0 = graph.units["subq:main_branch_0"]
        branch_1 = graph.units["subq:main_branch_1"]
        branch_2 = graph.units["subq:main_branch_2"]

        assert "table_a" in branch_0.depends_on_tables
        assert "table_b" in branch_1.depends_on_tables
        assert "table_c" in branch_2.depends_on_tables


class TestIntersectExcept:
    """Test INTERSECT and EXCEPT operations"""

    def test_intersect_basic(self):
        """Test basic INTERSECT"""
        sql = """
        SELECT id FROM set_a
        INTERSECT DISTINCT
        SELECT id FROM set_b
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        main_unit = graph.units[graph.main_unit_id]
        assert main_unit.unit_type == QueryUnitType.INTERSECT
        assert main_unit.set_operation_type == "intersect"
        assert len(main_unit.set_operation_branches) == 2

    def test_except_basic(self):
        """Test basic EXCEPT"""
        sql = """
        SELECT id FROM all_items
        EXCEPT DISTINCT
        SELECT id FROM excluded_items
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        main_unit = graph.units[graph.main_unit_id]
        assert main_unit.unit_type == QueryUnitType.EXCEPT
        assert main_unit.set_operation_type == "except"
        assert len(main_unit.set_operation_branches) == 2


class TestUnionWithCTEs:
    """Test UNION with CTEs in branches"""

    def test_union_with_cte_in_branch(self):
        """Test UNION where one branch has a CTE"""
        sql = """
        SELECT user_id FROM active_users
        UNION ALL
        WITH archived AS (
            SELECT user_id FROM archived_users WHERE deleted_at IS NOT NULL
        )
        SELECT user_id FROM archived
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Should have:
        # - 1 main UNION unit
        # - 2 branch units
        # - 1 CTE unit (inside second branch)
        assert len(graph.units) == 4

        # Find the branch with CTE
        branch_1 = graph.units["subq:main_branch_1"]
        assert len(branch_1.depends_on_units) == 1

        # Verify CTE exists
        cte_id = branch_1.depends_on_units[0]
        cte_unit = graph.units[cte_id]
        assert cte_unit.unit_type == QueryUnitType.CTE
        assert cte_unit.name == "archived"

    def test_union_branches_with_multiple_ctes(self):
        """Test UNION where both branches have CTEs"""
        sql = """
        WITH active AS (
            SELECT user_id FROM users WHERE status = 'active'
        )
        SELECT user_id FROM active
        UNION ALL
        WITH archived AS (
            SELECT user_id FROM users WHERE status = 'archived'
        )
        SELECT user_id FROM archived
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Note: In BigQuery, WITH clauses apply to their respective SELECT statements
        # The first CTE is at the top level (treats table "active" as non-existent in first branch)
        # The second CTE is within the second branch
        # So we should have:
        # - 1 main UNION unit
        # - 2 branch units
        # - 1 CTE unit (archived, in second branch)
        # First branch depends on non-existent table "active" (would fail at runtime)
        assert len(graph.units) == 4


class TestUnionWithSubqueries:
    """Test UNION with subqueries in branches"""

    def test_union_with_subquery_in_from(self):
        """Test UNION where branch has subquery in FROM"""
        sql = """
        SELECT id FROM base_table
        UNION ALL
        SELECT id FROM (
            SELECT id FROM derived_table WHERE active = true
        )
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Should have:
        # - 1 main UNION unit
        # - 2 branch units
        # - 1 subquery unit
        assert len(graph.units) == 4

        # Check second branch has subquery dependency
        branch_1 = graph.units["subq:main_branch_1"]
        assert len(branch_1.depends_on_units) == 1

    def test_union_with_complex_expressions(self):
        """Test UNION with expressions and aliases"""
        sql = """
        SELECT
            user_id,
            first_name || ' ' || last_name AS full_name,
            'active' AS status
        FROM users
        WHERE active = true
        UNION ALL
        SELECT
            user_id,
            archived_name AS full_name,
            'archived' AS status
        FROM archived_users
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        assert len(graph.units) == 3

        main_unit = graph.units[graph.main_unit_id]
        assert main_unit.unit_type == QueryUnitType.UNION


class TestNestedSetOperations:
    """Test nested and mixed set operations"""

    def test_union_of_unions(self):
        """Test nested UNION operations are flattened"""
        sql = """
        (SELECT id FROM table_a UNION ALL SELECT id FROM table_b)
        UNION ALL
        SELECT id FROM table_c
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Nested UNIONs should be flattened to 3 branches
        main_unit = graph.units[graph.main_unit_id]
        assert len(main_unit.set_operation_branches) == 3

    def test_four_way_union(self):
        """Test four-way UNION"""
        sql = """
        SELECT id FROM t1
        UNION ALL
        SELECT id FROM t2
        UNION ALL
        SELECT id FROM t3
        UNION ALL
        SELECT id FROM t4
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        main_unit = graph.units[graph.main_unit_id]
        assert len(main_unit.set_operation_branches) == 4


class TestUnionDependencies:
    """Test dependency tracking for UNION operations"""

    def test_union_depends_on_branches(self):
        """Test UNION unit correctly depends on its branches"""
        sql = """
        SELECT id FROM users
        UNION ALL
        SELECT id FROM archived_users
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        main_unit = graph.units[graph.main_unit_id]

        # Main UNION should depend on both branches
        assert len(main_unit.depends_on_units) == 2
        assert "subq:main_branch_0" in main_unit.depends_on_units
        assert "subq:main_branch_1" in main_unit.depends_on_units

    def test_topological_order_union(self):
        """Test topological ordering with UNION"""
        sql = """
        SELECT id FROM table_a
        UNION ALL
        SELECT id FROM table_b
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        ordered = graph.get_topological_order()

        # Branches should come before union
        branch_ids = {u.unit_id for u in ordered if u.unit_type == QueryUnitType.SUBQUERY_UNION}
        union_idx = next(i for i, u in enumerate(ordered) if u.unit_type == QueryUnitType.UNION)
        branch_indices = [i for i, u in enumerate(ordered) if u.unit_id in branch_ids]

        assert all(idx < union_idx for idx in branch_indices)


class TestUnionWithJoins:
    """Test UNION branches containing JOINs"""

    def test_union_branches_with_joins(self):
        """Test UNION where branches have JOINs"""
        sql = """
        SELECT u.id, o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE u.active = true
        UNION ALL
        SELECT au.id, ao.total
        FROM archived_users au
        JOIN archived_orders ao ON au.id = ao.user_id
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Check both branches have correct table dependencies
        branch_0 = graph.units["subq:main_branch_0"]
        branch_1 = graph.units["subq:main_branch_1"]

        assert "users" in branch_0.depends_on_tables
        assert "orders" in branch_0.depends_on_tables

        assert "archived_users" in branch_1.depends_on_tables
        assert "archived_orders" in branch_1.depends_on_tables


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_unsupported_top_level_type_raises_error(self):
        """Test that unsupported top-level query types raise an error"""
        sql = "INSERT INTO table VALUES (1, 2, 3)"

        with pytest.raises(ValueError, match="Unsupported top-level query type"):
            parser = RecursiveQueryParser(sql)
            parser.parse()

    def test_union_with_different_column_counts(self):
        """Test UNION with different column counts (parser should still work)"""
        # Note: This is syntactically invalid SQL but parser should handle it
        sql = """
        SELECT id, name FROM users
        UNION ALL
        SELECT id FROM archived_users
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Parser should still create the structure
        assert len(graph.units) == 3
        assert graph.units[graph.main_unit_id].unit_type == QueryUnitType.UNION


class TestUnitIdGeneration:
    """Test unit ID generation for set operations"""

    def test_setop_unit_id_format(self):
        """Test set operation units get correct ID format"""
        sql = """
        SELECT id FROM table_a
        UNION ALL
        SELECT id FROM table_b
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Main unit should have setop: prefix
        assert graph.main_unit_id == "setop:main"

        # Branches should have subq: prefix
        assert "subq:main_branch_0" in graph.units
        assert "subq:main_branch_1" in graph.units

    def test_intersect_unit_id(self):
        """Test INTERSECT unit ID"""
        sql = """
        SELECT id FROM table_a
        INTERSECT DISTINCT
        SELECT id FROM table_b
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        assert graph.main_unit_id == "setop:main"

    def test_except_unit_id(self):
        """Test EXCEPT unit ID"""
        sql = """
        SELECT id FROM table_a
        EXCEPT DISTINCT
        SELECT id FROM table_b
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        assert graph.main_unit_id == "setop:main"
