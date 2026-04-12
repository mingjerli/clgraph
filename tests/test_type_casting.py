"""
Test suite for column lineage tracking through type casting expressions.

Tests cover:
- Standard CAST(col AS type) across various data types
- PostgreSQL double-colon (::) cast operator
- BigQuery SAFE_CAST function
- Snowflake/SQL Server TRY_CAST function
- CAST nested inside other expressions (CASE, COALESCE, function args)
- Cross-dialect consistency for CAST
- Implicit type conversions
"""

import pytest

from clgraph import RecursiveLineageBuilder, SQLColumnTracer

# ============================================================================
# Test Group 1: Standard CAST Expression
# ============================================================================


class TestCastExpression:
    """Test lineage tracking through CAST(col AS type) for various target types."""

    def test_cast_to_int(self):
        """CAST to INTEGER should preserve lineage to the source column."""
        sql = "SELECT CAST(price AS INT) AS price_int FROM products"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        output_cols = {c.column_name for c in graph.nodes.values()}
        assert "price_int" in output_cols

        edges_to_price = [e for e in graph.edges if e.to_node.column_name == "price_int"]
        assert len(edges_to_price) > 0
        source_cols = {e.from_node.column_name for e in edges_to_price}
        assert "price" in source_cols

    def test_cast_to_varchar(self):
        """CAST to VARCHAR should trace back to the original column."""
        sql = "SELECT CAST(user_id AS VARCHAR) AS user_id_str FROM users"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        edges = [e for e in graph.edges if e.to_node.column_name == "user_id_str"]
        assert len(edges) > 0
        assert any(e.from_node.column_name == "user_id" for e in edges)

    def test_cast_to_date_and_timestamp(self):
        """CAST to DATE and TIMESTAMP should both preserve lineage."""
        sql = """
        SELECT
            CAST(created_at AS DATE) AS created_date,
            CAST(created_at AS TIMESTAMP) AS created_ts
        FROM events
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        for alias in ("created_date", "created_ts"):
            edges = [e for e in graph.edges if e.to_node.column_name == alias]
            assert len(edges) > 0, f"No edges found for {alias}"
            assert any(e.from_node.column_name == "created_at" for e in edges)

    def test_cast_to_numeric_types(self):
        """CAST to FLOAT, NUMERIC, and BOOLEAN should preserve lineage."""
        sql = """
        SELECT
            CAST(amount AS FLOAT) AS amount_float,
            CAST(amount AS NUMERIC) AS amount_numeric,
            CAST(is_active AS BOOLEAN) AS active_flag
        FROM accounts
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        for alias, source in [
            ("amount_float", "amount"),
            ("amount_numeric", "amount"),
            ("active_flag", "is_active"),
        ]:
            edges = [e for e in graph.edges if e.to_node.column_name == alias]
            assert len(edges) > 0, f"No edges found for {alias}"
            assert any(e.from_node.column_name == source for e in edges)

    def test_cast_backward_lineage(self):
        """Backward lineage through CAST should find the source column."""
        sql = "SELECT CAST(revenue AS FLOAT) AS revenue_float FROM sales"
        tracer = SQLColumnTracer(sql, dialect="bigquery")
        backward = tracer.get_backward_lineage(["revenue_float"])

        assert "sales" in backward["required_inputs"]
        assert "revenue" in backward["required_inputs"]["sales"]

    def test_cast_forward_lineage(self):
        """Forward lineage should track impact through CAST."""
        sql = "SELECT CAST(score AS INT) AS score_int FROM exams"
        tracer = SQLColumnTracer(sql, dialect="bigquery")
        forward = tracer.get_forward_lineage(["exams.score"])

        assert "score_int" in forward["impacted_outputs"]


# ============================================================================
# Test Group 2: PostgreSQL Double-Colon Cast Operator
# ============================================================================


class TestPostgresDoublColon:
    """Test lineage tracking through PostgreSQL :: cast operator."""

    def test_double_colon_int(self):
        """PostgreSQL col::int should trace back to the source column."""
        sql = "SELECT price::int AS price_int FROM products"
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        edges = [e for e in graph.edges if e.to_node.column_name == "price_int"]
        assert len(edges) > 0
        assert any(e.from_node.column_name == "price" for e in edges)

    def test_double_colon_text(self):
        """PostgreSQL col::text should preserve lineage."""
        sql = "SELECT user_id::text AS user_id_str FROM users"
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        edges = [e for e in graph.edges if e.to_node.column_name == "user_id_str"]
        assert len(edges) > 0
        assert any(e.from_node.column_name == "user_id" for e in edges)

    def test_double_colon_backward_lineage(self):
        """Backward lineage through :: cast finds the original column."""
        sql = "SELECT created_at::date AS created_date FROM events"
        tracer = SQLColumnTracer(sql, dialect="postgres")
        backward = tracer.get_backward_lineage(["created_date"])

        assert "events" in backward["required_inputs"]
        assert "created_at" in backward["required_inputs"]["events"]


# ============================================================================
# Test Group 3: BigQuery SAFE_CAST
# ============================================================================


class TestSafeCast:
    """Test lineage tracking through BigQuery SAFE_CAST function."""

    def test_safe_cast_preserves_lineage(self):
        """SAFE_CAST should track lineage the same as CAST."""
        sql = "SELECT SAFE_CAST(amount AS INT64) AS amount_int FROM transactions"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        edges = [e for e in graph.edges if e.to_node.column_name == "amount_int"]
        assert len(edges) > 0
        assert any(e.from_node.column_name == "amount" for e in edges)

    def test_safe_cast_backward_lineage(self):
        """Backward lineage through SAFE_CAST should find the source."""
        sql = "SELECT SAFE_CAST(rating AS FLOAT64) AS rating_float FROM reviews"
        tracer = SQLColumnTracer(sql, dialect="bigquery")
        backward = tracer.get_backward_lineage(["rating_float"])

        assert "reviews" in backward["required_inputs"]
        assert "rating" in backward["required_inputs"]["reviews"]


# ============================================================================
# Test Group 4: Snowflake / SQL Server TRY_CAST
# ============================================================================


class TestTryCast:
    """Test lineage tracking through TRY_CAST function."""

    def test_try_cast_preserves_lineage(self):
        """TRY_CAST should track lineage the same as CAST."""
        sql = "SELECT TRY_CAST(quantity AS INT) AS quantity_int FROM orders"
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        edges = [e for e in graph.edges if e.to_node.column_name == "quantity_int"]
        assert len(edges) > 0
        assert any(e.from_node.column_name == "quantity" for e in edges)

    def test_try_cast_forward_lineage(self):
        """Forward lineage should track impact through TRY_CAST."""
        sql = "SELECT TRY_CAST(value AS FLOAT) AS value_float FROM measurements"
        tracer = SQLColumnTracer(sql, dialect="snowflake")
        forward = tracer.get_forward_lineage(["measurements.value"])

        assert "value_float" in forward["impacted_outputs"]


# ============================================================================
# Test Group 5: CAST Inside Complex Expressions
# ============================================================================


class TestCastInExpressions:
    """Test lineage when CAST is nested inside other expressions."""

    def test_cast_inside_case_when(self):
        """CAST inside CASE WHEN should trace to the source column."""
        sql = """
        SELECT
            CASE
                WHEN status = 'active' THEN CAST(balance AS FLOAT)
                ELSE 0.0
            END AS balance_float
        FROM accounts
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        edges = [e for e in graph.edges if e.to_node.column_name == "balance_float"]
        assert len(edges) > 0
        source_cols = {e.from_node.column_name for e in edges}
        assert "balance" in source_cols or "status" in source_cols

    def test_cast_inside_coalesce(self):
        """CAST inside COALESCE should trace to source columns."""
        sql = """
        SELECT COALESCE(CAST(nullable_val AS INT), 0) AS safe_val
        FROM raw_data
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        edges = [e for e in graph.edges if e.to_node.column_name == "safe_val"]
        assert len(edges) > 0
        source_cols = {e.from_node.column_name for e in edges}
        assert "nullable_val" in source_cols

    def test_cast_as_function_argument(self):
        """CAST used as an argument to another function should preserve lineage."""
        sql = """
        SELECT CONCAT(CAST(user_id AS VARCHAR), '-', CAST(order_id AS VARCHAR)) AS composite_key
        FROM orders
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        edges = [e for e in graph.edges if e.to_node.column_name == "composite_key"]
        assert len(edges) > 0
        source_cols = {e.from_node.column_name for e in edges}
        assert "user_id" in source_cols or "order_id" in source_cols


# ============================================================================
# Test Group 6: Cross-Dialect CAST Consistency
# ============================================================================


class TestCastDialects:
    """Test that CAST lineage works consistently across dialects."""

    @pytest.mark.parametrize("dialect", ["bigquery", "postgres", "snowflake"])
    def test_cast_lineage_across_dialects(self, dialect):
        """The same CAST query should produce correct lineage in all dialects."""
        sql = "SELECT CAST(amount AS INT) AS amount_int FROM payments"
        builder = RecursiveLineageBuilder(sql, dialect=dialect)
        graph = builder.build()

        edges = [e for e in graph.edges if e.to_node.column_name == "amount_int"]
        assert len(edges) > 0, f"No edges for amount_int in dialect={dialect}"
        assert any(
            e.from_node.column_name == "amount" for e in edges
        ), f"Source column 'amount' not found in dialect={dialect}"

    @pytest.mark.parametrize("dialect", ["bigquery", "postgres", "snowflake"])
    def test_cast_backward_lineage_across_dialects(self, dialect):
        """Backward lineage through CAST should work in all dialects."""
        sql = "SELECT CAST(name AS VARCHAR) AS name_str FROM customers"
        tracer = SQLColumnTracer(sql, dialect=dialect)
        backward = tracer.get_backward_lineage(["name_str"])

        assert "customers" in backward["required_inputs"], (
            f"Table 'customers' not in required_inputs for dialect={dialect}"
        )
        assert "name" in backward["required_inputs"]["customers"], (
            f"Column 'name' not found for dialect={dialect}"
        )


# ============================================================================
# Test Group 7: Implicit Type Conversions
# ============================================================================


class TestImplicitConversions:
    """Test lineage through expressions that imply type conversion."""

    def test_arithmetic_with_float_literal(self):
        """Adding 0.0 to a column implies float conversion; lineage should hold."""
        sql = "SELECT quantity + 0.0 AS quantity_float FROM inventory"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        edges = [e for e in graph.edges if e.to_node.column_name == "quantity_float"]
        assert len(edges) > 0
        assert any(e.from_node.column_name == "quantity" for e in edges)

    def test_concat_with_non_string(self):
        """CONCAT with a non-string column implies conversion; lineage should hold."""
        sql = "SELECT CONCAT(name, ' - ', CAST(age AS VARCHAR)) AS label FROM people"
        tracer = SQLColumnTracer(sql, dialect="bigquery")
        backward = tracer.get_backward_lineage(["label"])

        assert "people" in backward["required_inputs"]
        source_cols = backward["required_inputs"]["people"]
        assert "name" in source_cols or "age" in source_cols
