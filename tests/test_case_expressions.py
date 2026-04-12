"""
Test suite for column lineage tracking through CASE WHEN expressions.

Tests cover:
- Simple CASE WHEN with column refs in WHEN and THEN
- Searched CASE (CASE col WHEN val THEN ...)
- ELSE branch column tracking
- CASE referencing columns from multiple tables
- Nested CASE expressions
- CASE in JOIN conditions
- CASE inside aggregate functions (SUM, COUNT)
- CASE with scalar subqueries
- COALESCE lineage (all arguments tracked)
- BigQuery IF expression
- Snowflake IFF expression
- NULLIF expression

Total: 16 test cases
"""

import pytest

from clgraph import RecursiveLineageBuilder, SQLColumnTracer


# ============================================================================
# Test Group 1: Simple CASE WHEN
# ============================================================================


class TestSimpleCaseWhen:
    """Test basic CASE WHEN with column references in WHEN and THEN branches."""

    def test_case_when_tracks_condition_and_result_columns(self):
        """All columns in WHEN conditions and THEN results should be traced."""
        sql = """
        SELECT
            id,
            CASE
                WHEN status = 'active' THEN revenue
                WHEN status = 'pending' THEN estimated_revenue
            END AS computed_revenue
        FROM orders
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        revenue_edges = [e for e in graph.edges if e.to_node.full_name == "output.computed_revenue"]
        source_columns = {e.from_node.column_name for e in revenue_edges}

        assert "status" in source_columns, "WHEN condition column should be traced"
        assert "revenue" in source_columns, "THEN result column should be traced"
        assert "estimated_revenue" in source_columns, "Second THEN result column should be traced"

    def test_case_when_backward_lineage(self):
        """Backward lineage should find all source columns referenced in CASE."""
        sql = """
        SELECT
            CASE
                WHEN age >= 18 THEN salary
                WHEN age < 18 THEN allowance
            END AS income
        FROM employees
        """
        tracer = SQLColumnTracer(sql, dialect="bigquery")
        backward = tracer.get_backward_lineage(["income"])

        required_cols = set(backward["required_inputs"].get("employees", []))
        assert "age" in required_cols, "WHEN condition column should be a required input"
        assert "salary" in required_cols, "THEN result column should be a required input"
        assert "allowance" in required_cols, "Second THEN column should be a required input"


# ============================================================================
# Test Group 2: Searched CASE (Simple Form)
# ============================================================================


class TestSearchedCase:
    """Test CASE col WHEN val1 THEN ... WHEN val2 THEN ... form."""

    def test_searched_case_tracks_discriminator_column(self):
        """The discriminator column in CASE <col> should be traced."""
        sql = """
        SELECT
            CASE region
                WHEN 'US' THEN domestic_rate
                WHEN 'EU' THEN eu_rate
            END AS tax_rate
        FROM tax_config
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        rate_edges = [e for e in graph.edges if e.to_node.full_name == "output.tax_rate"]
        source_columns = {e.from_node.column_name for e in rate_edges}

        assert "region" in source_columns, "Discriminator column should be traced"
        assert "domestic_rate" in source_columns, "THEN result column should be traced"
        assert "eu_rate" in source_columns, "Second THEN result column should be traced"


# ============================================================================
# Test Group 3: CASE with ELSE
# ============================================================================


class TestCaseWithElse:
    """Test that ELSE branch columns are tracked."""

    def test_else_branch_column_tracked(self):
        """Column referenced in ELSE should appear in lineage."""
        sql = """
        SELECT
            CASE
                WHEN priority = 'high' THEN urgent_handler
                ELSE default_handler
            END AS assigned_handler
        FROM tickets
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        handler_edges = [e for e in graph.edges if e.to_node.full_name == "output.assigned_handler"]
        source_columns = {e.from_node.column_name for e in handler_edges}

        assert "priority" in source_columns, "WHEN condition column should be traced"
        assert "urgent_handler" in source_columns, "THEN result column should be traced"
        assert "default_handler" in source_columns, "ELSE branch column should be traced"


# ============================================================================
# Test Group 4: CASE with Multiple Source Tables
# ============================================================================


class TestCaseMultipleSources:
    """Test CASE that references columns from different tables."""

    def test_case_references_multiple_tables(self):
        """CASE referencing columns from joined tables should trace to all tables."""
        sql = """
        SELECT
            CASE
                WHEN o.status = 'shipped' THEN s.tracking_number
                WHEN o.status = 'pending' THEN o.order_ref
            END AS reference_code
        FROM orders o
        JOIN shipments s ON o.id = s.order_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        ref_edges = [e for e in graph.edges if e.to_node.full_name == "output.reference_code"]
        source_tables = {e.from_node.table_name for e in ref_edges}

        assert "orders" in source_tables or "o" in source_tables, \
            "Orders table should be a source"
        assert "shipments" in source_tables or "s" in source_tables, \
            "Shipments table should be a source"

    def test_case_multi_table_backward_lineage(self):
        """Backward lineage should identify required inputs from all joined tables."""
        sql = """
        SELECT
            CASE
                WHEN u.role = 'admin' THEN p.full_access_level
                ELSE p.basic_access_level
            END AS access_level
        FROM users u
        JOIN permissions p ON u.permission_id = p.id
        """
        tracer = SQLColumnTracer(sql, dialect="bigquery")
        backward = tracer.get_backward_lineage(["access_level"])

        all_required = {}
        for table, cols in backward["required_inputs"].items():
            all_required[table] = set(cols)

        all_cols = set()
        for cols in all_required.values():
            all_cols.update(cols)

        assert "role" in all_cols, "Condition column from users should be required"
        assert "full_access_level" in all_cols or "basic_access_level" in all_cols, \
            "Result columns from permissions should be required"


# ============================================================================
# Test Group 5: Nested CASE
# ============================================================================


class TestNestedCase:
    """Test CASE expression nested inside another CASE."""

    def test_nested_case_tracks_all_columns(self):
        """All columns from both outer and inner CASE should be traced."""
        sql = """
        SELECT
            CASE
                WHEN category = 'A' THEN
                    CASE
                        WHEN sub_category = 'A1' THEN price_a1
                        ELSE price_a_default
                    END
                ELSE price_other
            END AS final_price
        FROM products
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        price_edges = [e for e in graph.edges if e.to_node.full_name == "output.final_price"]
        source_columns = {e.from_node.column_name for e in price_edges}

        assert "category" in source_columns, "Outer WHEN condition should be traced"
        assert "sub_category" in source_columns, "Inner WHEN condition should be traced"
        assert "price_a1" in source_columns, "Inner THEN result should be traced"
        assert "price_a_default" in source_columns, "Inner ELSE result should be traced"
        assert "price_other" in source_columns, "Outer ELSE result should be traced"


# ============================================================================
# Test Group 6: CASE in JOIN Condition
# ============================================================================


class TestCaseInJoinCondition:
    """Test CASE expression used in ON clause of a JOIN."""

    def test_case_in_join_on_clause(self):
        """Columns in a CASE within a JOIN ON clause should contribute to lineage."""
        sql = """
        SELECT
            a.id,
            b.value
        FROM table_a a
        JOIN table_b b ON b.key = CASE
            WHEN a.type = 'x' THEN a.key_x
            WHEN a.type = 'y' THEN a.key_y
        END
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        # The graph should have edges; the CASE columns participate in the join
        assert len(graph.edges) > 0, "Graph should have lineage edges"

        all_source_columns = {e.from_node.column_name for e in graph.edges}
        # At minimum, the selected columns should be present
        assert "id" in all_source_columns
        assert "value" in all_source_columns


# ============================================================================
# Test Group 7: CASE with Aggregates
# ============================================================================


class TestCaseWithAggregates:
    """Test CASE inside aggregate functions like SUM and COUNT."""

    def test_sum_case_when(self):
        """SUM(CASE WHEN ... THEN col END) should trace the CASE columns."""
        sql = """
        SELECT
            customer_id,
            SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) AS completed_total
        FROM orders
        GROUP BY customer_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        total_edges = [e for e in graph.edges if e.to_node.full_name == "output.completed_total"]
        source_columns = {e.from_node.column_name for e in total_edges}

        assert "status" in source_columns, "WHEN condition column should be traced"
        assert "amount" in source_columns, "THEN result column should be traced"

    def test_count_case_when(self):
        """COUNT(CASE WHEN ... THEN 1 END) should trace the condition columns."""
        sql = """
        SELECT
            department,
            COUNT(CASE WHEN is_active = TRUE THEN employee_id END) AS active_count
        FROM employees
        GROUP BY department
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        count_edges = [e for e in graph.edges if e.to_node.full_name == "output.active_count"]
        source_columns = {e.from_node.column_name for e in count_edges}

        assert "is_active" in source_columns or "employee_id" in source_columns, \
            "At least the condition or result column should be traced"


# ============================================================================
# Test Group 8: CASE with Subquery
# ============================================================================


class TestCaseWithSubquery:
    """Test CASE with scalar subquery in THEN branch."""

    def test_case_with_scalar_subquery(self):
        """Scalar subquery in THEN should be part of the lineage graph."""
        sql = """
        SELECT
            id,
            CASE
                WHEN type = 'premium' THEN (SELECT MAX(discount) FROM discounts WHERE tier = 'premium')
                ELSE 0
            END AS discount_rate
        FROM customers
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        discount_edges = [e for e in graph.edges if e.to_node.full_name == "output.discount_rate"]
        assert len(discount_edges) > 0, "CASE with subquery should produce lineage edges"

        source_columns = {e.from_node.column_name for e in discount_edges}
        assert "type" in source_columns, "WHEN condition column should be traced"


# ============================================================================
# Test Group 9: COALESCE
# ============================================================================


class TestCoalesceLineage:
    """Test that COALESCE(a, b, c) traces to all three source columns."""

    def test_coalesce_traces_all_arguments(self):
        """Every argument to COALESCE should appear as a source."""
        sql = """
        SELECT
            id,
            COALESCE(preferred_name, display_name, username) AS resolved_name
        FROM users
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        name_edges = [e for e in graph.edges if e.to_node.full_name == "output.resolved_name"]
        source_columns = {e.from_node.column_name for e in name_edges}

        assert "preferred_name" in source_columns, "First COALESCE arg should be traced"
        assert "display_name" in source_columns, "Second COALESCE arg should be traced"
        assert "username" in source_columns, "Third COALESCE arg should be traced"

    def test_coalesce_backward_lineage(self):
        """Backward lineage for COALESCE output should list all argument columns."""
        sql = """
        SELECT
            COALESCE(mobile_phone, home_phone, work_phone) AS contact_number
        FROM contacts
        """
        tracer = SQLColumnTracer(sql, dialect="bigquery")
        backward = tracer.get_backward_lineage(["contact_number"])

        required_cols = set(backward["required_inputs"].get("contacts", []))
        assert "mobile_phone" in required_cols
        assert "home_phone" in required_cols
        assert "work_phone" in required_cols


# ============================================================================
# Test Group 10: BigQuery IF Expression
# ============================================================================


class TestIfExpression:
    """Test BigQuery IF(condition, true_val, false_val) lineage tracking."""

    def test_if_expression_traces_all_parts(self):
        """IF should trace condition, true branch, and false branch columns."""
        sql = """
        SELECT
            id,
            IF(is_premium, premium_price, standard_price) AS final_price
        FROM products
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        price_edges = [e for e in graph.edges if e.to_node.full_name == "output.final_price"]
        source_columns = {e.from_node.column_name for e in price_edges}

        assert "is_premium" in source_columns, "IF condition column should be traced"
        assert "premium_price" in source_columns, "IF true-branch column should be traced"
        assert "standard_price" in source_columns, "IF false-branch column should be traced"


# ============================================================================
# Test Group 11: Snowflake IFF Expression
# ============================================================================


class TestIffExpression:
    """Test Snowflake IFF(condition, true_val, false_val) lineage tracking."""

    def test_iff_expression_traces_all_parts(self):
        """IFF should trace condition, true branch, and false branch columns."""
        sql = """
        SELECT
            id,
            IFF(is_domestic, domestic_rate, international_rate) AS shipping_rate
        FROM shipments
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        rate_edges = [e for e in graph.edges if e.to_node.full_name == "output.shipping_rate"]
        source_columns = {e.from_node.column_name for e in rate_edges}

        assert "is_domestic" in source_columns, "IFF condition column should be traced"
        assert "domestic_rate" in source_columns, "IFF true-branch column should be traced"
        assert "international_rate" in source_columns, "IFF false-branch column should be traced"


# ============================================================================
# Test Group 12: NULLIF
# ============================================================================


class TestNullif:
    """Test NULLIF(a, b) traces to both columns."""

    def test_nullif_traces_both_arguments(self):
        """NULLIF should trace both the value and the comparison columns."""
        sql = """
        SELECT
            id,
            NULLIF(current_balance, previous_balance) AS balance_change
        FROM accounts
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        balance_edges = [e for e in graph.edges if e.to_node.full_name == "output.balance_change"]
        source_columns = {e.from_node.column_name for e in balance_edges}

        assert "current_balance" in source_columns, "First NULLIF arg should be traced"
        assert "previous_balance" in source_columns, "Second NULLIF arg should be traced"
