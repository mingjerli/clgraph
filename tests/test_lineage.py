"""
Comprehensive test suite for recursive column lineage system.

Test-Driven Development approach:
1. Write tests first
2. Run tests (they should fail)
3. Implement minimum code to pass tests
4. Refactor
5. Repeat

Test Structure:
- TestQueryUnitBasics: Basic data structures (3 tests)
- TestRecursiveParserSimple: Simple queries (4 tests)
- TestRecursiveParserCTEs: CTE handling (7 tests)
- TestRecursiveParserSubqueries: Subquery handling (6 tests - includes HAVING)
- TestStarValidation: Star usage validation (7 tests)
- TestLineageBuilderSimple: Basic lineage building (4 tests)
- TestLineageBuilderStar: Star notation support (7 tests)
- TestLineageBuilderDeep: Deep nesting (4 tests)
- TestForwardLineage: Impact analysis (5 tests)
- TestBackwardLineage: Source tracing (5 tests)
- TestIntegration: End-to-end scenarios (4 tests)

Total: 56 test cases
"""

import pytest

# Import from the clgraph library
from clgraph import (
    ColumnLineageGraph,
    QueryUnit,
    QueryUnitGraph,
    QueryUnitType,
    RecursiveLineageBuilder,
    RecursiveQueryParser,
    SQLColumnTracer,
)

# ============================================================================
# Test Group 1: Query Unit Basics
# ============================================================================


class TestQueryUnitBasics:
    """Test basic QueryUnit data structures"""

    def test_query_unit_creation(self):
        """Test creating a QueryUnit"""
        import sqlglot

        parsed = sqlglot.parse_one("SELECT * FROM users")

        unit = QueryUnit(
            unit_id="main",
            unit_type=QueryUnitType.MAIN_QUERY,
            name="main",
            select_node=parsed,
            parent_unit=None,
            depends_on_units=[],
            depends_on_tables=["users"],
            depth=0,
            order=0,
        )

        assert unit.unit_id == "main"
        assert unit.unit_type == QueryUnitType.MAIN_QUERY
        assert unit.is_leaf()
        assert "users" in unit.depends_on_tables

    def test_query_unit_graph_add(self):
        """Test adding units to QueryUnitGraph"""
        graph = QueryUnitGraph()

        import sqlglot

        parsed = sqlglot.parse_one("SELECT * FROM users")

        unit = QueryUnit(
            unit_id="main",
            unit_type=QueryUnitType.MAIN_QUERY,
            name="main",
            select_node=parsed,
            parent_unit=None,
        )

        graph.add_unit(unit)

        assert len(graph.units) == 1
        assert graph.main_unit_id == "main"
        assert "main" in graph.units

    def test_query_unit_graph_topological_order(self):
        """Test topological sorting of query units"""
        graph = QueryUnitGraph()

        import sqlglot

        # Create mock units with dependencies
        # cte1 depends on nothing
        # cte2 depends on cte1
        # main depends on cte2

        cte1 = QueryUnit(
            unit_id="cte:cte1",
            unit_type=QueryUnitType.CTE,
            name="cte1",
            select_node=sqlglot.parse_one("SELECT * FROM t1"),
            parent_unit=None,
            depends_on_units=[],
            depends_on_tables=["t1"],
        )

        cte2 = QueryUnit(
            unit_id="cte:cte2",
            unit_type=QueryUnitType.CTE,
            name="cte2",
            select_node=sqlglot.parse_one("SELECT * FROM cte1"),
            parent_unit=None,
            depends_on_units=["cte:cte1"],
        )

        main = QueryUnit(
            unit_id="main",
            unit_type=QueryUnitType.MAIN_QUERY,
            name="main",
            select_node=sqlglot.parse_one("SELECT * FROM cte2"),
            parent_unit=None,
            depends_on_units=["cte:cte2"],
        )

        graph.add_unit(cte1)
        graph.add_unit(cte2)
        graph.add_unit(main)

        ordered = graph.get_topological_order()
        unit_ids = [u.unit_id for u in ordered]

        # cte1 should come before cte2, cte2 before main
        assert unit_ids.index("cte:cte1") < unit_ids.index("cte:cte2")
        assert unit_ids.index("cte:cte2") < unit_ids.index("main")


# ============================================================================
# Test Group 2: Recursive Parser - Simple Queries
# ============================================================================


class TestRecursiveParserSimple:
    """Test parsing simple queries"""

    def test_parse_single_table(self):
        """Test parsing simple single-table query"""
        query = "SELECT id, name FROM users"
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        assert len(graph.units) == 1
        assert graph.main_unit_id == "main"

        main_unit = graph.units["main"]
        assert main_unit.unit_type == QueryUnitType.MAIN_QUERY
        assert "users" in main_unit.depends_on_tables
        assert len(main_unit.depends_on_units) == 0

    def test_parse_with_alias(self):
        """Test parsing query with table alias"""
        query = "SELECT u.id, u.name FROM users u"
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        assert len(graph.units) == 1
        main_unit = graph.units["main"]
        assert "users" in main_unit.depends_on_tables

    def test_parse_with_join(self):
        """Test parsing query with JOIN"""
        query = """
        SELECT u.id, o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        assert len(graph.units) == 1
        main_unit = graph.units["main"]
        assert "users" in main_unit.depends_on_tables
        assert "orders" in main_unit.depends_on_tables

    def test_parse_aggregate(self):
        """Test parsing query with aggregates"""
        query = """
        SELECT user_id, COUNT(*) as cnt, SUM(total) as sum_total
        FROM orders
        GROUP BY user_id
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        assert len(graph.units) == 1
        assert "orders" in graph.units["main"].depends_on_tables


# ============================================================================
# Test Group 3: Recursive Parser - CTEs
# ============================================================================


class TestRecursiveParserCTEs:
    """Test parsing queries with CTEs"""

    def test_parse_single_cte(self):
        """Test parsing query with one CTE"""
        query = """
        WITH user_orders AS (
            SELECT user_id, COUNT(*) as order_count
            FROM orders
            GROUP BY user_id
        )
        SELECT * FROM user_orders
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        # Should have 2 units: main + CTE
        assert len(graph.units) == 2
        assert "cte:user_orders" in graph.units
        assert "main" in graph.units

        # CTE should depend on orders
        cte_unit = graph.units["cte:user_orders"]
        assert "orders" in cte_unit.depends_on_tables

        # Main should depend on CTE
        main_unit = graph.units["main"]
        assert "cte:user_orders" in main_unit.depends_on_units

    def test_parse_multiple_ctes(self):
        """Test parsing query with multiple CTEs"""
        query = """
        WITH cte1 AS (SELECT * FROM table1),
             cte2 AS (SELECT * FROM table2)
        SELECT cte1.*, cte2.* FROM cte1 JOIN cte2 ON cte1.id = cte2.id
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        # Should have 3 units: main + 2 CTEs
        assert len(graph.units) == 3
        assert "cte:cte1" in graph.units
        assert "cte:cte2" in graph.units

        # Main should depend on both CTEs
        main_unit = graph.units["main"]
        assert "cte:cte1" in main_unit.depends_on_units
        assert "cte:cte2" in main_unit.depends_on_units

    def test_parse_nested_ctes(self):
        """Test parsing CTEs that reference other CTEs"""
        query = """
        WITH cte1 AS (SELECT * FROM table1),
             cte2 AS (SELECT * FROM cte1),
             cte3 AS (SELECT * FROM cte2)
        SELECT * FROM cte3
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        # Should have 4 units: main + 3 CTEs
        assert len(graph.units) == 4

        # Check dependencies
        assert "table1" in graph.units["cte:cte1"].depends_on_tables
        assert "cte:cte1" in graph.units["cte:cte2"].depends_on_units
        assert "cte:cte2" in graph.units["cte:cte3"].depends_on_units
        assert "cte:cte3" in graph.units["main"].depends_on_units

    def test_parse_complex_cte_dependencies(self):
        """Test parsing CTEs with complex dependencies"""
        query = """
        WITH base AS (SELECT * FROM table1),
             enriched AS (
                 SELECT b.*, t2.value
                 FROM base b
                 JOIN table2 t2 ON b.id = t2.id
             ),
             aggregated AS (
                 SELECT id, SUM(value) as total
                 FROM enriched
                 GROUP BY id
             )
        SELECT * FROM aggregated
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        # Should have 4 units
        assert len(graph.units) == 4

        # Check base depends on table1
        assert "table1" in graph.units["cte:base"].depends_on_tables

        # Check enriched depends on base AND table2
        enriched = graph.units["cte:enriched"]
        assert "cte:base" in enriched.depends_on_units
        assert "table2" in enriched.depends_on_tables

        # Check aggregated depends on enriched
        assert "cte:enriched" in graph.units["cte:aggregated"].depends_on_units

    def test_parse_cte_depth(self):
        """Test that CTE depth is tracked correctly"""
        query = """
        WITH cte1 AS (SELECT * FROM table1),
             cte2 AS (SELECT * FROM cte1)
        SELECT * FROM cte2
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        # All CTEs at depth 1, main at depth 0
        assert graph.units["cte:cte1"].depth == 1
        assert graph.units["cte:cte2"].depth == 1
        assert graph.units["main"].depth == 0

    def test_parse_topological_order_ctes(self):
        """Test that CTEs are in correct topological order"""
        query = """
        WITH cte1 AS (SELECT * FROM table1),
             cte2 AS (SELECT * FROM cte1),
             cte3 AS (SELECT * FROM cte2)
        SELECT * FROM cte3
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        ordered = graph.get_topological_order()
        unit_ids = [u.unit_id for u in ordered]

        # cte1 before cte2 before cte3 before main
        assert unit_ids.index("cte:cte1") < unit_ids.index("cte:cte2")
        assert unit_ids.index("cte:cte2") < unit_ids.index("cte:cte3")
        assert unit_ids.index("cte:cte3") < unit_ids.index("main")


# ============================================================================
# Test Group 4: Recursive Parser - Subqueries
# ============================================================================


class TestRecursiveParserSubqueries:
    """Test parsing queries with subqueries"""

    def test_parse_subquery_in_from(self):
        """Test parsing subquery in FROM clause"""
        query = """
        SELECT x.id, x.name
        FROM (SELECT id, name FROM users WHERE active = true) x
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        # Should have 2 units: main + subquery
        assert len(graph.units) == 2

        # Find the subquery unit
        subq_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.SUBQUERY_FROM]
        assert len(subq_units) == 1

        subq = subq_units[0]
        assert "users" in subq.depends_on_tables

        # Main should depend on subquery
        main_unit = graph.units["main"]
        assert subq.unit_id in main_unit.depends_on_units

    def test_parse_scalar_subquery_in_select(self):
        """Test parsing scalar subquery in SELECT clause"""
        query = """
        SELECT
            u.name,
            (SELECT COUNT(*) FROM orders WHERE user_id = u.id) as order_count
        FROM users u
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        # Should have 2 units: main + scalar subquery
        assert len(graph.units) == 2

        # Find scalar subquery
        subq_units = [
            u for u in graph.units.values() if u.unit_type == QueryUnitType.SUBQUERY_SELECT
        ]
        assert len(subq_units) == 1

        subq = subq_units[0]
        assert "orders" in subq.depends_on_tables

    def test_parse_subquery_in_where(self):
        """Test parsing subquery in WHERE clause"""
        query = """
        SELECT name
        FROM users
        WHERE id IN (SELECT user_id FROM orders WHERE total > 100)
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        # Should have 2 units: main + WHERE subquery
        assert len(graph.units) == 2

        # Find WHERE subquery
        subq_units = [
            u for u in graph.units.values() if u.unit_type == QueryUnitType.SUBQUERY_WHERE
        ]
        assert len(subq_units) == 1

        subq = subq_units[0]
        assert "orders" in subq.depends_on_tables

    def test_parse_subquery_in_having(self):
        """Test parsing subquery in HAVING clause"""
        query = """
        SELECT category, COUNT(*) as cnt
        FROM products
        GROUP BY category
        HAVING COUNT(*) > (SELECT AVG(product_count) FROM category_stats)
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        # Should have 2 units: main + HAVING subquery
        assert len(graph.units) == 2

        # Find HAVING subquery
        subq_units = [
            u for u in graph.units.values() if u.unit_type == QueryUnitType.SUBQUERY_HAVING
        ]
        assert len(subq_units) == 1

        subq = subq_units[0]
        assert "category_stats" in subq.depends_on_tables

    def test_parse_nested_subqueries(self):
        """Test parsing nested subqueries"""
        query = """
        SELECT x.id
        FROM (
            SELECT id FROM (
                SELECT id FROM users WHERE active = true
            ) y
        ) x
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        # Should have 3 units: main + 2 nested subqueries
        assert len(graph.units) == 3

        # All should be subquery_from type except main
        subq_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.SUBQUERY_FROM]
        assert len(subq_units) == 2

    def test_parse_mixed_cte_and_subquery(self):
        """Test parsing query with both CTEs and subqueries"""
        query = """
        WITH base AS (SELECT * FROM table1)
        SELECT
            b.id,
            (SELECT MAX(value) FROM table2 WHERE id = b.id) as max_value
        FROM base b
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()

        # Should have 3 units: main + CTE + scalar subquery
        assert len(graph.units) == 3

        assert "cte:base" in graph.units
        subq_units = [
            u for u in graph.units.values() if u.unit_type == QueryUnitType.SUBQUERY_SELECT
        ]
        assert len(subq_units) == 1


# ============================================================================
# Test Group 5: Star Validation
# ============================================================================


class TestStarValidation:
    """Test star usage validation"""

    def test_star_single_table_valid(self):
        """Unqualified star with single table should be valid"""
        query = "SELECT * FROM users"
        parser = RecursiveQueryParser(query)
        graph = parser.parse()  # Should not raise

        assert len(graph.units) == 1

    def test_star_qualified_valid(self):
        """Qualified stars should always be valid"""
        query = """
        SELECT u.*, o.*
        FROM users u
        JOIN orders o ON u.id = o.user_id
        """
        parser = RecursiveQueryParser(query)
        graph = parser.parse()  # Should not raise

        assert len(graph.units) == 1

    def test_star_unqualified_multiple_tables_invalid(self):
        """Unqualified star with multiple tables should generate validation issue"""
        from clgraph.lineage_builder import RecursiveLineageBuilder
        from clgraph.models import IssueCategory

        query = """
        SELECT *
        FROM users u
        JOIN orders o ON u.id = o.user_id
        """
        # Parser now allows this and validation happens in lineage builder
        builder = RecursiveLineageBuilder(query)
        lineage_graph = builder.build()

        # Should have validation issue
        star_issues = [
            i
            for i in lineage_graph.issues
            if i.category == IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES
        ]
        assert len(star_issues) >= 1

    def test_star_in_cte_validated(self):
        """Star usage in CTEs should also be validated"""
        from clgraph.lineage_builder import RecursiveLineageBuilder
        from clgraph.models import IssueCategory

        query = """
        WITH bad_cte AS (
            SELECT * FROM users u JOIN orders o ON u.id = o.user_id
        )
        SELECT * FROM bad_cte
        """
        # Parser now allows this and validation happens in lineage builder
        builder = RecursiveLineageBuilder(query)
        lineage_graph = builder.build()

        # Should have validation issue for the CTE
        star_issues = [
            i
            for i in lineage_graph.issues
            if i.category == IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES
        ]
        assert len(star_issues) >= 1

    def test_star_except_valid(self):
        """Star with EXCEPT should be valid"""
        query = "SELECT * EXCEPT(password, ssn) FROM users"
        parser = RecursiveQueryParser(query)
        graph = parser.parse()  # Should not raise

        assert len(graph.units) == 1

    def test_star_replace_valid(self):
        """Star with REPLACE should be valid"""
        query = "SELECT * REPLACE(UPPER(name) AS name) FROM users"
        parser = RecursiveQueryParser(query)
        graph = parser.parse()  # Should not raise

        assert len(graph.units) == 1


# ============================================================================
# Test Group 6: Lineage Builder - Simple Cases
# ============================================================================


class TestLineageBuilderSimple:
    """Test building lineage for simple queries"""

    def test_build_simple_query(self):
        """Test building lineage for simple query"""
        query = "SELECT id, name FROM users"
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # Should have input and output nodes
        assert "users.id" in graph.nodes
        assert "users.name" in graph.nodes
        assert "output.id" in graph.nodes
        assert "output.name" in graph.nodes

        # Check layers
        assert graph.nodes["users.id"].layer == "input"
        assert graph.nodes["output.id"].layer == "output"

        # Should have edges connecting them
        assert len(graph.edges) >= 2

        # Check specific edges
        edges_dict = {(e.from_node.full_name, e.to_node.full_name): e for e in graph.edges}
        assert ("users.id", "output.id") in edges_dict
        assert ("users.name", "output.name") in edges_dict

    def test_build_with_alias(self):
        """Test building lineage with aliased columns"""
        query = "SELECT id as user_id, name as user_name FROM users"
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        assert "users.id" in graph.nodes
        assert "users.name" in graph.nodes
        assert "output.user_id" in graph.nodes
        assert "output.user_name" in graph.nodes

        # Check edges
        edges_dict = {(e.from_node.full_name, e.to_node.full_name): e for e in graph.edges}
        assert ("users.id", "output.user_id") in edges_dict
        assert ("users.name", "output.user_name") in edges_dict

    def test_build_with_expression(self):
        """Test building lineage with expressions"""
        query = "SELECT id, name || ' ' || email as full_info FROM users"
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        assert "users.id" in graph.nodes
        assert "users.name" in graph.nodes
        assert "users.email" in graph.nodes
        assert "output.full_info" in graph.nodes

        # full_info should have edges from both name and email
        full_info_edges = [e for e in graph.edges if e.to_node.full_name == "output.full_info"]
        source_tables = {e.from_node.table_name for e in full_info_edges}
        assert "users" in source_tables

    def test_build_with_aggregate(self):
        """Test building lineage with aggregates"""
        query = """
        SELECT user_id, COUNT(*) as cnt, SUM(total) as sum_total
        FROM orders
        GROUP BY user_id
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # Check nodes
        assert "orders.user_id" in graph.nodes
        assert "orders.total" in graph.nodes
        assert "output.cnt" in graph.nodes
        assert "output.sum_total" in graph.nodes

        # Check node types
        assert graph.nodes["output.cnt"].node_type == "aggregate"
        assert graph.nodes["output.sum_total"].node_type == "aggregate"


# ============================================================================
# Test Group 7: Lineage Builder - Star Support
# ============================================================================


class TestLineageBuilderStar:
    """Test building lineage with star notation"""

    def test_build_simple_star(self):
        """Test building lineage for SELECT *"""
        query = "SELECT * FROM users"
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # Should have star node
        assert "users.*" in graph.nodes
        star_node = graph.nodes["users.*"]
        assert star_node.is_star
        assert star_node.layer == "input"

        # Check output
        assert any("*" in key for key in graph.nodes.keys() if "output" in key)

    def test_build_qualified_star(self):
        """Test building lineage for qualified star"""
        query = "SELECT u.* FROM users u"
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        assert "users.*" in graph.nodes
        assert graph.nodes["users.*"].is_star

    def test_build_multiple_qualified_stars(self):
        """Test building lineage for multiple qualified stars"""
        query = """
        SELECT u.*, o.*
        FROM users u
        JOIN orders o ON u.id = o.user_id
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # Should have star nodes for both tables
        assert "users.*" in graph.nodes
        assert "orders.*" in graph.nodes

    def test_build_star_except(self):
        """Test building lineage for star with EXCEPT"""
        query = "SELECT * EXCEPT(password, ssn) FROM users"
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # Find output star node
        output_stars = [n for n in graph.nodes.values() if n.layer == "output" and n.is_star]
        assert len(output_stars) > 0

        star = output_stars[0]
        assert "password" in star.except_columns
        assert "ssn" in star.except_columns

    def test_build_star_passthrough_cte(self):
        """Test star passthrough through CTE"""
        query = """
        WITH cte1 AS (SELECT t.* FROM table1 t)
        SELECT * FROM cte1
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # Should have star at each layer
        assert "table1.*" in graph.nodes
        assert "cte1.*" in graph.nodes

        # Check edges
        star_edges = [e for e in graph.edges if e.transformation == "star_passthrough"]
        assert len(star_edges) > 0

        # Verify chain: table1.* -> cte1.*
        edge_dict = {(e.from_node.full_name, e.to_node.full_name): e for e in star_edges}
        assert ("table1.*", "cte1.*") in edge_dict

    def test_build_star_with_specific_columns(self):
        """Test mixed star and specific columns"""
        query = """
        SELECT u.*, o.total, o.status
        FROM users u
        JOIN orders o ON u.id = o.user_id
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # Should have star for users
        assert "users.*" in graph.nodes

        # Should have specific columns for orders
        assert "orders.total" in graph.nodes
        assert "orders.status" in graph.nodes


# ============================================================================
# Test Group 8: Lineage Builder - Deep Nesting
# ============================================================================


class TestLineageBuilderDeep:
    """Test building lineage for deeply nested queries"""

    def test_build_single_cte(self):
        """Test building lineage with one CTE"""
        query = """
        WITH user_orders AS (
            SELECT user_id, COUNT(*) as order_count
            FROM orders
            GROUP BY user_id
        )
        SELECT u.name, uo.order_count
        FROM users u
        JOIN user_orders uo ON u.id = uo.user_id
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # Check input nodes
        assert "orders.user_id" in graph.nodes
        assert "users.name" in graph.nodes

        # Check CTE nodes
        assert "user_orders.user_id" in graph.nodes
        assert "user_orders.order_count" in graph.nodes

        # Check output nodes
        assert "output.name" in graph.nodes
        assert "output.order_count" in graph.nodes

        # Check edges exist
        edges_dict = {(e.from_node.full_name, e.to_node.full_name): e for e in graph.edges}
        assert ("orders.user_id", "user_orders.user_id") in edges_dict
        assert ("user_orders.order_count", "output.order_count") in edges_dict

    def test_build_nested_ctes(self):
        """Test building lineage with nested CTEs"""
        query = """
        WITH cte1 AS (SELECT id, name FROM table1),
             cte2 AS (SELECT id, name FROM cte1),
             cte3 AS (SELECT id, name FROM cte2)
        SELECT * FROM cte3
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # Check all layers exist
        assert "table1.id" in graph.nodes
        assert "cte1.id" in graph.nodes
        assert "cte2.id" in graph.nodes
        assert "cte3.id" in graph.nodes
        assert "output.id" in graph.nodes

        # Check chain of edges
        edges_dict = {(e.from_node.full_name, e.to_node.full_name): e for e in graph.edges}
        assert ("table1.id", "cte1.id") in edges_dict
        assert ("cte1.id", "cte2.id") in edges_dict
        assert ("cte2.id", "cte3.id") in edges_dict
        assert ("cte3.id", "output.id") in edges_dict

    def test_build_deep_star_chain(self):
        """Test building lineage for deep star passthrough"""
        query = """
        WITH cte1 AS (SELECT * FROM t1),
             cte2 AS (SELECT * FROM cte1),
             cte3 AS (SELECT * FROM cte2),
             cte4 AS (SELECT * FROM cte3)
        SELECT * FROM cte4
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # All layers should have star nodes
        assert "t1.*" in graph.nodes
        assert "cte1.*" in graph.nodes
        assert "cte2.*" in graph.nodes
        assert "cte3.*" in graph.nodes
        assert "cte4.*" in graph.nodes

        # Check complete chain
        star_edges = [e for e in graph.edges if ".*" in e.from_node.full_name]
        assert len(star_edges) >= 4  # At least 4 hops

    def test_build_complex_mixed(self):
        """Test building lineage for complex mixed query"""
        query = """
        WITH base AS (SELECT t.* FROM table1 t),
             enriched AS (
                 SELECT b.*, t2.value as extra_value
                 FROM base b
                 JOIN table2 t2 ON b.id = t2.id
             ),
             aggregated AS (
                 SELECT id, COUNT(*) as cnt
                 FROM enriched
                 GROUP BY id
             )
        SELECT e.*, a.cnt
        FROM enriched e
        JOIN aggregated a ON e.id = a.id
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # Check we have nodes at all layers
        input_nodes = [n for n in graph.nodes.values() if n.layer == "input"]
        cte_nodes = [n for n in graph.nodes.values() if n.layer == "cte"]
        output_nodes = [n for n in graph.nodes.values() if n.layer == "output"]

        assert len(input_nodes) > 0
        assert len(cte_nodes) > 0
        assert len(output_nodes) > 0

        # Check star nodes exist
        assert "table1.*" in graph.nodes
        assert "base.*" in graph.nodes


# ============================================================================
# Test Group 9: Forward Lineage Queries
# ============================================================================


class TestForwardLineage:
    """Test forward lineage (impact analysis)"""

    def test_forward_simple(self):
        """Test forward lineage for simple query"""
        query = "SELECT id, name FROM users"
        tracer = SQLColumnTracer(query)
        forward = tracer.get_forward_lineage(["users.id"])

        assert "id" in forward["impacted_outputs"]
        assert len(forward["paths"]) > 0

    def test_forward_through_cte(self):
        """Test forward lineage through CTE"""
        query = """
        WITH cte1 AS (SELECT id, name FROM users)
        SELECT id, name FROM cte1
        """
        tracer = SQLColumnTracer(query)
        forward = tracer.get_forward_lineage(["users.id"])

        assert "cte1" in forward["impacted_ctes"]
        assert "id" in forward["impacted_outputs"]

        # Check path
        paths = forward["paths"]
        assert len(paths) > 0
        path = paths[0]
        assert "users.id" in path["input"]
        assert any("cte1" in str(i) for i in path["intermediate"])

    def test_forward_multiple_outputs(self):
        """Test forward lineage affecting multiple outputs"""
        query = """
        SELECT id, id as user_id, id * 2 as double_id
        FROM users
        """
        tracer = SQLColumnTracer(query)
        forward = tracer.get_forward_lineage(["users.id"])

        # Should impact all three outputs
        assert len(forward["impacted_outputs"]) == 3

    def test_forward_star(self):
        """Test forward lineage from star node"""
        query = """
        WITH cte1 AS (SELECT t.* FROM table1 t)
        SELECT * FROM cte1
        """
        tracer = SQLColumnTracer(query)
        forward = tracer.get_forward_lineage(["table1.*"])

        assert "cte1" in forward["impacted_ctes"]
        assert len(forward["impacted_outputs"]) > 0

    def test_forward_transformation_tracking(self):
        """Test that transformations are tracked in forward lineage"""
        query = """
        SELECT id, SUM(total) as sum_total
        FROM orders
        GROUP BY id
        """
        tracer = SQLColumnTracer(query)
        forward = tracer.get_forward_lineage(["orders.total"])

        paths = forward["paths"]
        assert len(paths) > 0

        # Should show aggregate transformation
        path = paths[0]
        assert "aggregate" in path["transformations"]


# ============================================================================
# Test Group 10: Backward Lineage Queries
# ============================================================================


class TestBackwardLineage:
    """Test backward lineage (source tracing)"""

    def test_backward_simple(self):
        """Test backward lineage for simple query"""
        query = "SELECT id, name FROM users"
        tracer = SQLColumnTracer(query)
        backward = tracer.get_backward_lineage(["id"])

        assert "users" in backward["required_inputs"]
        assert "id" in backward["required_inputs"]["users"]

    def test_backward_through_cte(self):
        """Test backward lineage through CTE"""
        query = """
        WITH cte1 AS (SELECT id, name FROM users)
        SELECT id FROM cte1
        """
        tracer = SQLColumnTracer(query)
        backward = tracer.get_backward_lineage(["id"])

        assert "cte1" in backward["required_ctes"]
        assert "users" in backward["required_inputs"]
        assert "id" in backward["required_inputs"]["users"]

    def test_backward_deep_nesting(self):
        """Test backward lineage through deep nesting"""
        query = """
        WITH cte1 AS (SELECT id FROM table1),
             cte2 AS (SELECT id FROM cte1),
             cte3 AS (SELECT id FROM cte2)
        SELECT id FROM cte3
        """
        tracer = SQLColumnTracer(query)
        backward = tracer.get_backward_lineage(["id"])

        # All CTEs should be required
        assert "cte1" in backward["required_ctes"]
        assert "cte2" in backward["required_ctes"]
        assert "cte3" in backward["required_ctes"]

        # Base table should be required
        assert "table1" in backward["required_inputs"]

    def test_backward_star(self):
        """Test backward lineage for star output"""
        query = """
        WITH cte1 AS (SELECT t.* FROM table1 t)
        SELECT * FROM cte1
        """
        tracer = SQLColumnTracer(query)

        # Get all output columns
        graph = tracer.build_column_lineage_graph()
        output_nodes = graph.get_output_nodes()
        output_cols = [n.column_name for n in output_nodes]

        if output_cols:
            backward = tracer.get_backward_lineage([output_cols[0]])
            assert "table1" in backward["required_inputs"]

    def test_backward_expression(self):
        """Test backward lineage for expression"""
        query = """
        SELECT id, name || ' ' || email as full_info
        FROM users
        """
        tracer = SQLColumnTracer(query)
        backward = tracer.get_backward_lineage(["full_info"])

        # Should require both name and email
        assert "users" in backward["required_inputs"]
        required_cols = set(backward["required_inputs"]["users"])
        assert "name" in required_cols
        assert "email" in required_cols


# ============================================================================
# Test Group 11: Integration Tests
# ============================================================================


class TestIntegration:
    """End-to-end integration tests"""

    def test_real_world_example_1(self):
        """Test real-world example: user analytics"""
        query = """
        WITH user_orders AS (
            SELECT
                user_id,
                COUNT(*) as order_count,
                SUM(total) as total_spent
            FROM orders
            WHERE status = 'completed'
            GROUP BY user_id
        ),
        user_segments AS (
            SELECT
                u.id,
                u.name,
                u.email,
                uo.order_count,
                uo.total_spent,
                CASE
                    WHEN uo.total_spent > 1000 THEN 'high'
                    WHEN uo.total_spent > 100 THEN 'medium'
                    ELSE 'low'
                END as segment
            FROM users u
            LEFT JOIN user_orders uo ON u.id = uo.user_id
        )
        SELECT
            segment,
            COUNT(*) as user_count,
            AVG(total_spent) as avg_spent
        FROM user_segments
        GROUP BY segment
        """

        tracer = SQLColumnTracer(query)

        # Build graph
        graph = tracer.build_column_lineage_graph()
        assert len(graph.nodes) > 0

        # Test forward lineage from orders.total
        forward = tracer.get_forward_lineage(["orders.total"])
        assert (
            "total_spent" in forward["impacted_outputs"]
            or "avg_spent" in forward["impacted_outputs"]
        )

        # Test backward lineage for segment
        # segment is based on total_spent which comes from orders.total
        backward = tracer.get_backward_lineage(["segment"])
        assert "orders" in backward["required_inputs"]
        # Note: users table is not required for segment calculation
        # (segment only depends on total_spent from orders)

    def test_real_world_example_2(self):
        """Test real-world example with stars and except"""
        query = """
        WITH cte1 AS (
            SELECT t.* FROM table1 t
        ),
        cte2 AS (
            SELECT t2.* FROM table2 t2
        )
        SELECT c1.*, c2.* EXCEPT(id)
        FROM cte1 c1
        INNER JOIN cte2 c2 USING(id)
        """

        tracer = SQLColumnTracer(query)

        # Build graph
        graph = tracer.build_column_lineage_graph()

        # Should have star nodes
        star_nodes = [n for n in graph.nodes.values() if n.is_star]
        assert len(star_nodes) > 0

        # Should have EXCEPT columns
        output_stars = [n for n in star_nodes if n.layer == "output"]
        has_except = any("id" in n.except_columns for n in output_stars)
        assert has_except

    def test_real_world_example_3(self):
        """Test real-world example with subqueries"""
        query = """
        WITH base AS (
            SELECT * FROM table1
        ),
        enriched AS (
            SELECT
                b.*,
                (SELECT MAX(value) FROM table2 WHERE id = b.id) as max_value
            FROM base b
        )
        SELECT e.*
        FROM enriched e
        WHERE e.max_value > (SELECT AVG(max_value) FROM enriched)
        """

        tracer = SQLColumnTracer(query)

        # Should parse successfully
        structure = tracer.get_query_structure()
        assert len(structure.units) >= 4  # main + 2 CTEs + 2 subqueries

        # Build graph
        graph = tracer.build_column_lineage_graph()
        assert len(graph.nodes) > 0

        # Forward lineage from table1.*
        forward = tracer.get_forward_lineage(["table1.*"])
        assert len(forward["impacted_outputs"]) > 0

    def test_backward_compatibility(self):
        """Test that existing SQLColumnTracer functionality still works"""
        query = """
        SELECT u.name, COUNT(o.id) as order_count
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        GROUP BY u.name
        """

        tracer = SQLColumnTracer(query)

        # Old methods should still work
        columns = tracer.get_column_names()
        assert "name" in columns
        assert "order_count" in columns

        # New methods should work
        graph = tracer.build_column_lineage_graph()
        assert len(graph.nodes) > 0


# ============================================================================
# Test Group 12: Simplified Graph
# ============================================================================


class TestSimplifiedGraph:
    """Test the to_simplified() method for collapsing intermediate layers"""

    def test_simplified_single_cte(self):
        """Test simplifying a graph with one CTE"""
        query = """
        WITH monthly AS (
            SELECT user_id, SUM(amount) as total FROM orders GROUP BY user_id
        )
        SELECT * FROM monthly
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()
        simplified = graph.to_simplified()

        # Should only have input and output nodes
        assert all(n.layer in ("input", "output") for n in simplified.nodes.values())

        # Should have 2 input nodes (orders.user_id, orders.amount)
        # and 2 output nodes (output.user_id, output.total)
        input_nodes = simplified.get_input_nodes()
        output_nodes = simplified.get_output_nodes()
        assert len(input_nodes) == 2
        assert len(output_nodes) == 2

        # Should have direct edges from input to output
        assert len(simplified.edges) == 2

        edge_pairs = {(e.from_node.full_name, e.to_node.full_name) for e in simplified.edges}
        assert ("orders.user_id", "output.user_id") in edge_pairs
        assert ("orders.amount", "output.total") in edge_pairs

    def test_simplified_multiple_ctes(self):
        """Test simplifying a graph with multiple CTEs"""
        query = """
        WITH
          step1 AS (
            SELECT user_id, amount FROM orders WHERE status = 'completed'
          ),
          step2 AS (
            SELECT user_id, SUM(amount) as total FROM step1 GROUP BY user_id
          )
        SELECT user_id, total FROM step2
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()
        simplified = graph.to_simplified()

        # Original should have 8 nodes (2 input + 2 step1 + 2 step2 + 2 output)
        assert len(graph.nodes) == 8

        # Simplified should only have 4 nodes (2 input + 2 output)
        assert len(simplified.nodes) == 4

        # Check edges trace through both CTEs
        edge_pairs = {(e.from_node.full_name, e.to_node.full_name) for e in simplified.edges}
        assert ("orders.user_id", "output.user_id") in edge_pairs
        assert ("orders.amount", "output.total") in edge_pairs

    def test_simplified_preserves_warnings(self):
        """Test that simplified graph preserves warnings and issues"""
        query = "SELECT * FROM users, orders"  # Multiple tables with unqualified star
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()
        simplified = graph.to_simplified()

        # Issues should be preserved
        assert len(simplified.issues) == len(graph.issues)

    def test_simplified_no_intermediate_nodes(self):
        """Test that simplified graph has no CTE or subquery nodes"""
        query = """
        WITH cte1 AS (SELECT id FROM t1),
             cte2 AS (SELECT id FROM cte1)
        SELECT id FROM cte2
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()
        simplified = graph.to_simplified()

        # No CTE nodes in simplified
        cte_nodes = [n for n in simplified.nodes.values() if n.layer == "cte"]
        assert len(cte_nodes) == 0

        # Only input and output
        for node in simplified.nodes.values():
            assert node.layer in ("input", "output")


# ============================================================================
# Helper Functions for Tests
# ============================================================================


def print_graph_structure(graph: ColumnLineageGraph):
    """Helper to visualize graph structure during debugging"""
    print("\n=== Graph Structure ===")
    print(f"Total nodes: {len(graph.nodes)}")
    print(f"Total edges: {len(graph.edges)}")

    for layer in ["input", "cte", "subquery", "output"]:
        layer_nodes = [n for n in graph.nodes.values() if n.layer == layer]
        if layer_nodes:
            print(f"\n{layer.upper()} Layer ({len(layer_nodes)} nodes):")
            for node in sorted(layer_nodes, key=lambda n: n.full_name):
                star_indicator = " ⭐" if node.is_star else ""
                print(f"  - {node.full_name}{star_indicator}")

    print("\nEdges:")
    for edge in graph.edges[:10]:  # Show first 10
        print(f"  {edge.from_node.full_name} --[{edge.transformation}]--> {edge.to_node.full_name}")


def print_query_structure(graph: QueryUnitGraph):
    """Helper to visualize query structure during debugging"""
    print("\n=== Query Structure ===")
    print(f"Total units: {len(graph.units)}")

    for unit in graph.get_topological_order():
        indent = "  " * unit.depth
        deps = (
            f" (depends: {', '.join(unit.depends_on_units + unit.depends_on_tables)})"
            if unit.depends_on_units or unit.depends_on_tables
            else ""
        )
        print(f"{indent}{unit.unit_id} [{unit.unit_type.value}]{deps}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
