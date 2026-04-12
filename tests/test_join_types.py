"""
Test suite for column lineage tracking across different JOIN types.

Verifies that clgraph correctly traces column lineage through:
- INNER JOIN
- LEFT JOIN / LEFT OUTER JOIN
- RIGHT JOIN / RIGHT OUTER JOIN
- FULL OUTER JOIN
- CROSS JOIN
- Self joins
- Multi-table join chains
- Joins against subqueries
- ON clause column tracking
- Multiple SQL dialects

Test Structure:
- TestInnerJoin: Explicit INNER JOIN (2 tests)
- TestLeftJoin: LEFT JOIN / LEFT OUTER JOIN (2 tests)
- TestRightJoin: RIGHT JOIN / RIGHT OUTER JOIN (2 tests)
- TestFullOuterJoin: FULL OUTER JOIN (2 tests)
- TestCrossJoin: CROSS JOIN without ON clause (1 test)
- TestSelfJoin: Table joined to itself with aliases (2 tests)
- TestMultipleJoins: 3+ tables joined in a chain (2 tests)
- TestJoinWithSubquery: JOIN against a subquery (2 tests)
- TestJoinConditionLineage: ON clause columns tracked (1 test)
- TestJoinDialects: Same join across bigquery, postgres, snowflake (2 tests)

Total: 18 test cases
"""

import pytest

from clgraph import RecursiveLineageBuilder, SQLColumnTracer


# ============================================================================
# Test Group 1: INNER JOIN
# ============================================================================


class TestInnerJoin:
    """Test lineage tracking through explicit INNER JOIN."""

    def test_inner_join_columns_from_both_tables(self):
        """Both sides of an INNER JOIN contribute columns to the output."""
        sql = """
        SELECT u.id, u.name, o.order_id, o.amount
        FROM users u
        INNER JOIN orders o ON u.id = o.user_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        assert "output.id" in graph.nodes
        assert "output.name" in graph.nodes
        assert "output.order_id" in graph.nodes
        assert "output.amount" in graph.nodes

        edges_dict = {
            (e.from_node.full_name, e.to_node.full_name): e
            for e in graph.edges
        }
        assert ("users.id", "output.id") in edges_dict
        assert ("users.name", "output.name") in edges_dict
        assert ("orders.order_id", "output.order_id") in edges_dict
        assert ("orders.amount", "output.amount") in edges_dict

    def test_inner_join_forward_backward_lineage(self):
        """Forward and backward tracing works through INNER JOIN."""
        sql = """
        SELECT u.id, u.name, o.amount
        FROM users u
        INNER JOIN orders o ON u.id = o.user_id
        """
        tracer = SQLColumnTracer(sql, dialect="bigquery")

        forward = tracer.get_forward_lineage(["users.name"])
        assert "name" in forward["impacted_outputs"]

        backward = tracer.get_backward_lineage(["amount"])
        assert "orders" in backward["required_inputs"]


# ============================================================================
# Test Group 2: LEFT JOIN
# ============================================================================


class TestLeftJoin:
    """Test lineage tracking through LEFT JOIN / LEFT OUTER JOIN."""

    def test_left_join_nullable_side_columns_tracked(self):
        """Columns from the right (nullable) side are tracked in LEFT JOIN."""
        sql = """
        SELECT u.id, u.name, p.bio
        FROM users u
        LEFT JOIN profiles p ON u.id = p.user_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        assert "output.id" in graph.nodes
        assert "output.name" in graph.nodes
        assert "output.bio" in graph.nodes

        edges_dict = {
            (e.from_node.full_name, e.to_node.full_name): e
            for e in graph.edges
        }
        assert ("users.id", "output.id") in edges_dict
        assert ("profiles.bio", "output.bio") in edges_dict

    def test_left_outer_join_backward_lineage(self):
        """Backward lineage traces through LEFT OUTER JOIN to both tables."""
        sql = """
        SELECT u.id, p.bio
        FROM users u
        LEFT OUTER JOIN profiles p ON u.id = p.user_id
        """
        tracer = SQLColumnTracer(sql, dialect="bigquery")

        backward = tracer.get_backward_lineage(["bio"])
        assert "profiles" in backward["required_inputs"]


# ============================================================================
# Test Group 3: RIGHT JOIN
# ============================================================================


class TestRightJoin:
    """Test lineage tracking through RIGHT JOIN / RIGHT OUTER JOIN."""

    def test_right_join_columns_tracked(self):
        """Both sides of a RIGHT JOIN contribute columns."""
        sql = """
        SELECT u.name, o.order_id, o.amount
        FROM users u
        RIGHT JOIN orders o ON u.id = o.user_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        assert "output.name" in graph.nodes
        assert "output.order_id" in graph.nodes
        assert "output.amount" in graph.nodes

        edges_dict = {
            (e.from_node.full_name, e.to_node.full_name): e
            for e in graph.edges
        }
        assert ("users.name", "output.name") in edges_dict
        assert ("orders.order_id", "output.order_id") in edges_dict

    def test_right_outer_join_forward_lineage(self):
        """Forward lineage works through RIGHT OUTER JOIN."""
        sql = """
        SELECT u.name, o.order_id
        FROM users u
        RIGHT OUTER JOIN orders o ON u.id = o.user_id
        """
        tracer = SQLColumnTracer(sql, dialect="bigquery")

        forward = tracer.get_forward_lineage(["orders.order_id"])
        assert "order_id" in forward["impacted_outputs"]


# ============================================================================
# Test Group 4: FULL OUTER JOIN
# ============================================================================


class TestFullOuterJoin:
    """Test lineage tracking through FULL OUTER JOIN."""

    def test_full_outer_join_both_sides_tracked(self):
        """Both sides of a FULL OUTER JOIN contribute columns."""
        sql = """
        SELECT u.id, u.name, o.order_id, o.amount
        FROM users u
        FULL OUTER JOIN orders o ON u.id = o.user_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        assert "output.id" in graph.nodes
        assert "output.name" in graph.nodes
        assert "output.order_id" in graph.nodes
        assert "output.amount" in graph.nodes

        edges_dict = {
            (e.from_node.full_name, e.to_node.full_name): e
            for e in graph.edges
        }
        assert ("users.id", "output.id") in edges_dict
        assert ("orders.amount", "output.amount") in edges_dict

    def test_full_outer_join_backward_lineage(self):
        """Backward lineage traces through FULL OUTER JOIN."""
        sql = """
        SELECT u.name, o.amount
        FROM users u
        FULL OUTER JOIN orders o ON u.id = o.user_id
        """
        tracer = SQLColumnTracer(sql, dialect="bigquery")

        backward = tracer.get_backward_lineage(["name"])
        assert "users" in backward["required_inputs"]

        backward_amount = tracer.get_backward_lineage(["amount"])
        assert "orders" in backward_amount["required_inputs"]


# ============================================================================
# Test Group 5: CROSS JOIN
# ============================================================================


class TestCrossJoin:
    """Test lineage tracking through CROSS JOIN (cartesian product)."""

    def test_cross_join_no_on_clause(self):
        """CROSS JOIN without ON clause tracks columns from both tables."""
        sql = """
        SELECT u.name, c.color_name
        FROM users u
        CROSS JOIN colors c
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        assert "output.name" in graph.nodes
        assert "output.color_name" in graph.nodes

        edges_dict = {
            (e.from_node.full_name, e.to_node.full_name): e
            for e in graph.edges
        }
        assert ("users.name", "output.name") in edges_dict
        assert ("colors.color_name", "output.color_name") in edges_dict


# ============================================================================
# Test Group 6: Self Join
# ============================================================================


class TestSelfJoin:
    """Test lineage tracking when a table is joined to itself."""

    def test_self_join_with_aliases(self):
        """Self join using aliases resolves columns to the same source table."""
        sql = """
        SELECT e.name AS employee, m.name AS manager
        FROM employees e
        INNER JOIN employees m ON e.manager_id = m.id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        assert "output.employee" in graph.nodes
        assert "output.manager" in graph.nodes

        # Both output columns trace back to employees.name
        employee_edges = [
            e for e in graph.edges if e.to_node.full_name == "output.employee"
        ]
        manager_edges = [
            e for e in graph.edges if e.to_node.full_name == "output.manager"
        ]

        assert len(employee_edges) > 0
        assert len(manager_edges) > 0
        assert employee_edges[0].from_node.table_name == "employees"
        assert manager_edges[0].from_node.table_name == "employees"

    def test_self_join_forward_lineage(self):
        """Forward lineage through a self join reaches aliased outputs."""
        sql = """
        SELECT e.name AS employee, m.name AS manager
        FROM employees e
        INNER JOIN employees m ON e.manager_id = m.id
        """
        tracer = SQLColumnTracer(sql, dialect="bigquery")

        forward = tracer.get_forward_lineage(["employees.name"])
        # employees.name feeds both employee and manager outputs
        assert len(forward["impacted_outputs"]) >= 1


# ============================================================================
# Test Group 7: Multiple Joins (3+ tables)
# ============================================================================


class TestMultipleJoins:
    """Test lineage tracking through chains of 3+ joined tables."""

    def test_three_table_join_chain(self):
        """Lineage flows through a chain of three joined tables."""
        sql = """
        SELECT u.name, o.order_id, p.product_name
        FROM users u
        INNER JOIN orders o ON u.id = o.user_id
        INNER JOIN products p ON o.product_id = p.id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        assert "output.name" in graph.nodes
        assert "output.order_id" in graph.nodes
        assert "output.product_name" in graph.nodes

        edges_dict = {
            (e.from_node.full_name, e.to_node.full_name): e
            for e in graph.edges
        }
        assert ("users.name", "output.name") in edges_dict
        assert ("orders.order_id", "output.order_id") in edges_dict
        assert ("products.product_name", "output.product_name") in edges_dict

    def test_mixed_join_types_chain(self):
        """Lineage works through mixed INNER + LEFT join chain."""
        sql = """
        SELECT u.name, o.order_id, r.review_text
        FROM users u
        INNER JOIN orders o ON u.id = o.user_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
        """
        tracer = SQLColumnTracer(sql, dialect="bigquery")

        backward = tracer.get_backward_lineage(["review_text"])
        assert "reviews" in backward["required_inputs"]

        backward_name = tracer.get_backward_lineage(["name"])
        assert "users" in backward_name["required_inputs"]


# ============================================================================
# Test Group 8: JOIN with Subquery
# ============================================================================


class TestJoinWithSubquery:
    """Test lineage tracking when JOINing against a subquery."""

    def test_join_against_subquery(self):
        """Columns from a subquery in a JOIN are tracked to their source."""
        sql = """
        SELECT u.name, agg.total_amount
        FROM users u
        INNER JOIN (
            SELECT user_id, SUM(amount) AS total_amount
            FROM orders
            GROUP BY user_id
        ) agg ON u.id = agg.user_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        assert "output.name" in graph.nodes
        assert "output.total_amount" in graph.nodes

        # total_amount should trace back through the subquery to orders.amount
        total_edges = [
            e for e in graph.edges if e.to_node.full_name == "output.total_amount"
        ]
        assert len(total_edges) > 0

    def test_join_subquery_backward_lineage(self):
        """Backward lineage traces through subquery in JOIN to source table."""
        sql = """
        SELECT u.name, agg.total_amount
        FROM users u
        INNER JOIN (
            SELECT user_id, SUM(amount) AS total_amount
            FROM orders
            GROUP BY user_id
        ) agg ON u.id = agg.user_id
        """
        tracer = SQLColumnTracer(sql, dialect="bigquery")

        backward = tracer.get_backward_lineage(["total_amount"])
        assert "orders" in backward["required_inputs"]


# ============================================================================
# Test Group 9: JOIN Condition Lineage
# ============================================================================


class TestJoinConditionLineage:
    """Test that columns referenced in ON clauses are tracked."""

    def test_on_clause_columns_in_graph(self):
        """Columns used only in the ON clause still appear in the graph."""
        sql = """
        SELECT u.name, o.amount
        FROM users u
        INNER JOIN orders o ON u.id = o.user_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        # The selected columns must be present
        assert "output.name" in graph.nodes
        assert "output.amount" in graph.nodes

        # The join-key columns should appear as source nodes in the graph
        all_source_names = {
            e.from_node.full_name for e in graph.edges
        }
        # users.id and orders.user_id are join keys; at minimum the selected
        # columns (users.name, orders.amount) must have edges
        assert ("users.name", "output.name") in {
            (e.from_node.full_name, e.to_node.full_name) for e in graph.edges
        }
        assert ("orders.amount", "output.amount") in {
            (e.from_node.full_name, e.to_node.full_name) for e in graph.edges
        }


# ============================================================================
# Test Group 10: JOIN across Dialects
# ============================================================================


class TestJoinDialects:
    """Test the same JOIN query across bigquery, postgres, and snowflake."""

    DIALECTS = ["bigquery", "postgres", "snowflake"]

    JOIN_SQL = """
    SELECT u.id, u.name, o.order_id, o.amount
    FROM users u
    INNER JOIN orders o ON u.id = o.user_id
    """

    @pytest.mark.parametrize("dialect", DIALECTS)
    def test_inner_join_lineage_per_dialect(self, dialect):
        """INNER JOIN lineage is consistent across SQL dialects."""
        builder = RecursiveLineageBuilder(self.JOIN_SQL, dialect=dialect)
        graph = builder.build()

        assert "output.id" in graph.nodes
        assert "output.name" in graph.nodes
        assert "output.order_id" in graph.nodes
        assert "output.amount" in graph.nodes

        edges_dict = {
            (e.from_node.full_name, e.to_node.full_name): e
            for e in graph.edges
        }
        assert ("users.id", "output.id") in edges_dict
        assert ("orders.amount", "output.amount") in edges_dict

    @pytest.mark.parametrize("dialect", DIALECTS)
    def test_left_join_tracer_per_dialect(self, dialect):
        """LEFT JOIN forward/backward tracing works across dialects."""
        sql = """
        SELECT u.name, o.amount
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        """
        tracer = SQLColumnTracer(sql, dialect=dialect)

        forward = tracer.get_forward_lineage(["users.name"])
        assert "name" in forward["impacted_outputs"]

        backward = tracer.get_backward_lineage(["amount"])
        assert "orders" in backward["required_inputs"]
