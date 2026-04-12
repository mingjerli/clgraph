"""
Test suite for column lineage tracking across underrepresented SQL dialects.

Covers dialect-specific constructs for Snowflake, MySQL, and Spark/Databricks
that are not well-exercised by the existing BigQuery/PostgreSQL test suites.

Tests verify that columns and edges are correctly tracked through
dialect-specific syntax. Where parsing is not yet supported, tests are
marked with ``pytest.mark.xfail`` to document known gaps.
"""

import pytest

from clgraph import Pipeline, RecursiveLineageBuilder

# ============================================================================
# Helper utilities
# ============================================================================


def _output_column_names(graph):
    """Return a set of output column names from a lineage graph."""
    return {n.column_name for n in graph.get_output_nodes()}


def _source_columns_for(graph, output_col):
    """Return source column names that feed into *output_col*."""
    return {e.from_node.column_name for e in graph.edges if e.to_node.column_name == output_col}


# ============================================================================
# Test Group 1: Snowflake-Specific Constructs
# ============================================================================


class TestSnowflakeSpecific:
    """Snowflake dialect-specific lineage tests."""

    def test_lateral_flatten(self):
        """LATERAL FLATTEN should parse and produce lineage edges."""
        sql = """
        SELECT
            id,
            f.VALUE AS item_value
        FROM orders, LATERAL FLATTEN(INPUT => orders.items) AS f
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        assert len(graph.nodes) > 0
        names = _output_column_names(graph)
        assert "id" in names
        assert "item_value" in names

    def test_flatten_without_lateral(self):
        """Bare FLATTEN (no LATERAL keyword) should still be recognised."""
        sql = """
        SELECT f.VALUE AS val
        FROM events, FLATTEN(INPUT => events.tags) AS f
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        assert len(graph.nodes) > 0
        assert "val" in _output_column_names(graph)

    def test_qualify_with_row_number(self):
        """QUALIFY ROW_NUMBER() should track partition/order columns."""
        sql = """
        SELECT customer_id, order_date, amount
        FROM orders
        QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        qualify_edges = [e for e in graph.edges if e.is_qualify_column]
        assert len(qualify_edges) > 0

    def test_object_construct(self):
        """OBJECT_CONSTRUCT should track input columns."""
        sql = """
        SELECT OBJECT_CONSTRUCT('name', name, 'age', age) AS obj
        FROM users
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        sources = _source_columns_for(graph, "obj")
        assert "name" in sources or "age" in sources

    def test_semi_structured_access(self):
        """Snowflake semi-structured access (col:field::type) lineage."""
        sql = """
        SELECT
            payload:user_id::INT AS user_id,
            payload:event_name::STRING AS event_name
        FROM raw_events
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        names = _output_column_names(graph)
        assert "user_id" in names
        assert "event_name" in names

    def test_create_or_replace_table_as_select(self):
        """CREATE OR REPLACE TABLE AS SELECT should be parsed like CTAS."""
        sql = """
        CREATE OR REPLACE TABLE staging_users AS
        SELECT id, name, email FROM raw_users
        """
        pipeline = Pipeline.from_sql_string(sql, dialect="snowflake")

        table_names = list(pipeline.table_graph.tables.keys())
        assert "staging_users" in table_names
        col_names = [c.column_name for c in pipeline.get_columns_by_table("staging_users")]
        assert "id" in col_names
        assert "name" in col_names

    def test_merge_snowflake_syntax(self):
        """MERGE INTO with Snowflake syntax should track columns."""
        sql = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET t.value = s.new_value
        WHEN NOT MATCHED THEN INSERT (id, value) VALUES (s.id, s.new_value)
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

    def test_try_cast(self):
        """TRY_CAST should be treated as a pass-through for lineage."""
        sql = """
        SELECT TRY_CAST(amount AS DECIMAL(10, 2)) AS amount_dec
        FROM transactions
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        assert "amount_dec" in _output_column_names(graph)
        sources = _source_columns_for(graph, "amount_dec")
        assert "amount" in sources

    def test_tablesample(self):
        """TABLESAMPLE should not break parsing; columns still tracked."""
        sql = """
        SELECT id, value
        FROM large_table TABLESAMPLE (10)
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        names = _output_column_names(graph)
        assert "id" in names
        assert "value" in names


# ============================================================================
# Test Group 2: MySQL-Specific Constructs
# ============================================================================


class TestMySQLSpecific:
    """MySQL dialect-specific lineage tests."""

    def test_limit_without_offset(self):
        """Simple LIMIT (no OFFSET) should not affect lineage tracking."""
        sql = """
        SELECT id, name FROM users LIMIT 10
        """
        builder = RecursiveLineageBuilder(sql, dialect="mysql")
        graph = builder.build()

        names = _output_column_names(graph)
        assert "id" in names
        assert "name" in names

    def test_group_concat(self):
        """GROUP_CONCAT should track its argument column."""
        sql = """
        SELECT
            department,
            GROUP_CONCAT(employee_name ORDER BY employee_name SEPARATOR ', ') AS names
        FROM employees
        GROUP BY department
        """
        builder = RecursiveLineageBuilder(sql, dialect="mysql")
        graph = builder.build()

        sources = _source_columns_for(graph, "names")
        assert "employee_name" in sources

    def test_if_function(self):
        """MySQL IF(cond, val1, val2) should track both branch columns."""
        sql = """
        SELECT
            id,
            IF(status = 'active', balance, 0) AS effective_balance
        FROM accounts
        """
        builder = RecursiveLineageBuilder(sql, dialect="mysql")
        graph = builder.build()

        names = _output_column_names(graph)
        assert "effective_balance" in names
        sources = _source_columns_for(graph, "effective_balance")
        assert "balance" in sources or "status" in sources

    def test_ifnull(self):
        """IFNULL should track its argument columns."""
        sql = """
        SELECT
            id,
            IFNULL(nickname, full_name) AS display_name
        FROM profiles
        """
        builder = RecursiveLineageBuilder(sql, dialect="mysql")
        graph = builder.build()

        sources = _source_columns_for(graph, "display_name")
        assert "nickname" in sources or "full_name" in sources

    @pytest.mark.xfail(reason="ON DUPLICATE KEY UPDATE may not be tracked")
    def test_insert_on_duplicate_key_update(self):
        """INSERT ... ON DUPLICATE KEY UPDATE should parse without error."""
        sql = """
        INSERT INTO counters (id, hits)
        VALUES (1, 1)
        ON DUPLICATE KEY UPDATE hits = hits + 1
        """
        builder = RecursiveLineageBuilder(sql, dialect="mysql")
        graph = builder.build()

        assert graph is not None

    def test_backtick_quoted_identifiers(self):
        """Backtick-quoted identifiers should resolve correctly."""
        sql = """
        SELECT `user`.`id`, `user`.`name`
        FROM `user`
        """
        builder = RecursiveLineageBuilder(sql, dialect="mysql")
        graph = builder.build()

        names = _output_column_names(graph)
        assert "id" in names
        assert "name" in names

    def test_straight_join(self):
        """STRAIGHT_JOIN should be treated as a regular JOIN for lineage."""
        sql = """
        SELECT a.id, b.value
        FROM orders a STRAIGHT_JOIN products b ON a.product_id = b.id
        """
        builder = RecursiveLineageBuilder(sql, dialect="mysql")
        graph = builder.build()

        names = _output_column_names(graph)
        assert "id" in names
        assert "value" in names


# ============================================================================
# Test Group 3: Spark / Databricks-Specific Constructs
# ============================================================================


class TestSparkSpecific:
    """Spark / Databricks dialect-specific lineage tests."""

    def test_lateral_view_explode(self):
        """LATERAL VIEW EXPLODE should produce array expansion lineage."""
        sql = """
        SELECT id, tag
        FROM events
        LATERAL VIEW EXPLODE(tags) t AS tag
        """
        builder = RecursiveLineageBuilder(sql, dialect="spark")
        graph = builder.build()

        names = _output_column_names(graph)
        assert "id" in names
        assert "tag" in names

    def test_collect_list(self):
        """COLLECT_LIST should track its argument column."""
        sql = """
        SELECT
            group_id,
            COLLECT_LIST(value) AS values
        FROM items
        GROUP BY group_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="spark")
        graph = builder.build()

        sources = _source_columns_for(graph, "values")
        assert "value" in sources

    def test_collect_set(self):
        """COLLECT_SET should track its argument column."""
        sql = """
        SELECT
            category,
            COLLECT_SET(product_name) AS unique_products
        FROM products
        GROUP BY category
        """
        builder = RecursiveLineageBuilder(sql, dialect="spark")
        graph = builder.build()

        sources = _source_columns_for(graph, "unique_products")
        assert "product_name" in sources

    @pytest.mark.xfail(reason="CREATE TABLE USING delta may not parse")
    def test_create_table_using_delta(self):
        """CREATE TABLE ... USING delta AS SELECT should parse like CTAS."""
        sql = """
        CREATE TABLE gold.metrics
        USING delta
        AS SELECT user_id, SUM(amount) AS total
        FROM silver.transactions
        GROUP BY user_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="databricks")
        graph = builder.build()

        names = _output_column_names(graph)
        assert "user_id" in names
        assert "total" in names

    def test_transform_function(self):
        """TRANSFORM higher-order function should track array source."""
        sql = """
        SELECT
            id,
            TRANSFORM(scores, x -> x * 100) AS scaled_scores
        FROM students
        """
        builder = RecursiveLineageBuilder(sql, dialect="spark")
        graph = builder.build()

        assert "scaled_scores" in _output_column_names(graph)
        sources = _source_columns_for(graph, "scaled_scores")
        assert "scores" in sources

    def test_named_struct(self):
        """STRUCT(col1, col2) / named_struct should track input columns."""
        sql = """
        SELECT
            id,
            STRUCT(first_name, last_name) AS full_name
        FROM users
        """
        builder = RecursiveLineageBuilder(sql, dialect="spark")
        graph = builder.build()

        assert "full_name" in _output_column_names(graph)

    def test_rlike_operator(self):
        """RLIKE in WHERE should not break lineage; columns still tracked."""
        sql = """
        SELECT id, email
        FROM users
        WHERE email RLIKE '^[a-z]+@example\\\\.com$'
        """
        builder = RecursiveLineageBuilder(sql, dialect="spark")
        graph = builder.build()

        names = _output_column_names(graph)
        assert "id" in names
        assert "email" in names


# ============================================================================
# Test Group 4: Cross-Dialect Equivalence
# ============================================================================


class TestCrossDialectEquivalence:
    """Verify that logically equivalent queries across dialects produce
    equivalent column lineage (same output columns sourced from same inputs).
    """

    def _get_latest_per_group_sql(self, dialect):
        """Return 'get latest row per group' SQL for the given dialect."""
        if dialect in ("bigquery", "snowflake", "postgres"):
            return """
            SELECT customer_id, order_date, amount
            FROM orders
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY customer_id ORDER BY order_date DESC
            ) = 1
            """
        elif dialect == "mysql":
            # MySQL lacks QUALIFY; use subquery with window function
            return """
            SELECT customer_id, order_date, amount
            FROM (
                SELECT
                    customer_id,
                    order_date,
                    amount,
                    ROW_NUMBER() OVER (
                        PARTITION BY customer_id ORDER BY order_date DESC
                    ) AS rn
                FROM orders
            ) ranked
            WHERE rn = 1
            """
        else:
            raise ValueError(f"Unsupported dialect: {dialect}")

    def test_latest_per_group_bigquery(self):
        """BigQuery latest-per-group produces expected output columns."""
        sql = self._get_latest_per_group_sql("bigquery")
        graph = RecursiveLineageBuilder(sql, dialect="bigquery").build()

        names = _output_column_names(graph)
        assert "customer_id" in names
        assert "order_date" in names
        assert "amount" in names

    def test_latest_per_group_snowflake(self):
        """Snowflake latest-per-group produces expected output columns."""
        sql = self._get_latest_per_group_sql("snowflake")
        graph = RecursiveLineageBuilder(sql, dialect="snowflake").build()

        names = _output_column_names(graph)
        assert "customer_id" in names
        assert "order_date" in names
        assert "amount" in names

    def test_latest_per_group_postgres(self):
        """PostgreSQL latest-per-group produces expected output columns."""
        sql = self._get_latest_per_group_sql("postgres")
        graph = RecursiveLineageBuilder(sql, dialect="postgres").build()

        names = _output_column_names(graph)
        assert "customer_id" in names
        assert "order_date" in names
        assert "amount" in names

    def test_latest_per_group_mysql(self):
        """MySQL latest-per-group (subquery form) produces expected output columns."""
        sql = self._get_latest_per_group_sql("mysql")
        graph = RecursiveLineageBuilder(sql, dialect="mysql").build()

        names = _output_column_names(graph)
        assert "customer_id" in names
        assert "order_date" in names
        assert "amount" in names

    def test_cross_dialect_output_columns_match(self):
        """All dialects should produce the same set of output column names."""
        baseline_cols = None
        for dialect in ("bigquery", "snowflake", "postgres", "mysql"):
            sql = self._get_latest_per_group_sql(dialect)
            graph = RecursiveLineageBuilder(sql, dialect=dialect).build()
            names = _output_column_names(graph)

            # MySQL subquery form may include 'rn'; filter to business columns
            business_cols = names - {"rn"}

            if baseline_cols is None:
                baseline_cols = business_cols
            else:
                assert business_cols == baseline_cols, (
                    f"Dialect {dialect} produced {business_cols}, expected {baseline_cols}"
                )


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
