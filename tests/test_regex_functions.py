"""
Test suite for column lineage tracking through regular expression functions.

Tests cover:
- REGEXP_CONTAINS (BigQuery)
- REGEXP_EXTRACT / REGEXP_SUBSTR (BigQuery, Snowflake)
- REGEXP_REPLACE across dialects
- RLIKE operator (Spark, Snowflake)
- PostgreSQL regex operators (~ , ~*, SIMILAR TO)
- Regex inside CASE WHEN expressions
- Regex in WHERE clauses
- Cross-dialect regex patterns
"""

import pytest

from clgraph import RecursiveLineageBuilder, SQLColumnTracer


# ============================================================================
# Test Group 1: REGEXP_CONTAINS (BigQuery)
# ============================================================================


class TestRegexpContains:
    """Test lineage tracking through BigQuery REGEXP_CONTAINS function."""

    def test_regexp_contains_in_select(self):
        """Test REGEXP_CONTAINS used as a boolean expression in SELECT."""
        sql = """
        SELECT
            email,
            REGEXP_CONTAINS(email, r'@example\\.com$') AS is_example_domain
        FROM users
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        is_example_edges = [
            e for e in graph.edges if e.to_node.column_name == "is_example_domain"
        ]
        assert len(is_example_edges) > 0

        source_columns = {e.from_node.column_name for e in is_example_edges}
        assert "email" in source_columns


# ============================================================================
# Test Group 2: REGEXP_EXTRACT / REGEXP_SUBSTR
# ============================================================================


class TestRegexpExtract:
    """Test lineage tracking through REGEXP_EXTRACT and REGEXP_SUBSTR."""

    def test_bigquery_regexp_extract(self):
        """Test BigQuery REGEXP_EXTRACT traces back to source column."""
        sql = """
        SELECT
            url,
            REGEXP_EXTRACT(url, r'https?://([^/]+)') AS domain
        FROM web_logs
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        domain_edges = [e for e in graph.edges if e.to_node.column_name == "domain"]
        assert len(domain_edges) > 0

        source_columns = {e.from_node.column_name for e in domain_edges}
        assert "url" in source_columns

    def test_snowflake_regexp_substr(self):
        """Test Snowflake REGEXP_SUBSTR traces back to source column."""
        sql = """
        SELECT
            phone_number,
            REGEXP_SUBSTR(phone_number, '\\\\d{3}') AS area_code
        FROM contacts
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        area_code_edges = [
            e for e in graph.edges if e.to_node.column_name == "area_code"
        ]
        assert len(area_code_edges) > 0

        source_columns = {e.from_node.column_name for e in area_code_edges}
        assert "phone_number" in source_columns


# ============================================================================
# Test Group 3: REGEXP_REPLACE
# ============================================================================


class TestRegexpReplace:
    """Test lineage tracking through REGEXP_REPLACE across dialects."""

    def test_bigquery_regexp_replace(self):
        """Test BigQuery REGEXP_REPLACE traces back to source column."""
        sql = """
        SELECT
            REGEXP_REPLACE(phone, r'[^0-9]', '') AS clean_phone
        FROM customers
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        clean_phone_edges = [
            e for e in graph.edges if e.to_node.column_name == "clean_phone"
        ]
        assert len(clean_phone_edges) > 0

        source_columns = {e.from_node.column_name for e in clean_phone_edges}
        assert "phone" in source_columns

    def test_snowflake_regexp_replace(self):
        """Test Snowflake REGEXP_REPLACE traces back to source column."""
        sql = """
        SELECT
            REGEXP_REPLACE(address, '[^a-zA-Z0-9 ]', '') AS sanitized_address
        FROM locations
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        sanitized_edges = [
            e for e in graph.edges if e.to_node.column_name == "sanitized_address"
        ]
        assert len(sanitized_edges) > 0

        source_columns = {e.from_node.column_name for e in sanitized_edges}
        assert "address" in source_columns


# ============================================================================
# Test Group 4: RLIKE Operator (Spark / Snowflake)
# ============================================================================


class TestRlike:
    """Test lineage tracking through RLIKE operator."""

    def test_rlike_in_where_clause(self):
        """Test that columns used with RLIKE in WHERE are tracked."""
        sql = """
        SELECT name, email
        FROM users
        WHERE email RLIKE '^[a-z]+@company\\\\.com$'
        """
        builder = RecursiveLineageBuilder(sql, dialect="spark")
        graph = builder.build()

        email_edges = [e for e in graph.edges if e.to_node.column_name == "email"]
        assert len(email_edges) > 0

        source_columns = {e.from_node.column_name for e in email_edges}
        assert "email" in source_columns


# ============================================================================
# Test Group 5: PostgreSQL Regex Operators
# ============================================================================


class TestPostgresRegex:
    """Test lineage tracking through PostgreSQL regex operators."""

    def test_tilde_operator_in_where(self):
        """Test PostgreSQL ~ operator tracks column in WHERE clause."""
        sql = """
        SELECT id, description
        FROM products
        WHERE description ~ 'organic|natural'
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        desc_edges = [
            e for e in graph.edges if e.to_node.column_name == "description"
        ]
        assert len(desc_edges) > 0

        source_columns = {e.from_node.column_name for e in desc_edges}
        assert "description" in source_columns


# ============================================================================
# Test Group 6: Regex in CASE WHEN
# ============================================================================


class TestRegexpInCaseWhen:
    """Test lineage tracking when regex functions are used in CASE expressions."""

    def test_regexp_extract_in_case_when(self):
        """Test REGEXP_EXTRACT inside CASE WHEN tracks source column."""
        sql = """
        SELECT
            url,
            CASE
                WHEN REGEXP_CONTAINS(url, r'^https') THEN 'secure'
                ELSE 'insecure'
            END AS protocol_type
        FROM web_logs
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        protocol_edges = [
            e for e in graph.edges if e.to_node.column_name == "protocol_type"
        ]
        assert len(protocol_edges) > 0

        source_columns = {e.from_node.column_name for e in protocol_edges}
        assert "url" in source_columns

    def test_regexp_replace_in_case_value(self):
        """Test REGEXP_REPLACE used as CASE WHEN result tracks source column."""
        sql = """
        SELECT
            CASE
                WHEN country = 'US'
                    THEN REGEXP_REPLACE(phone, r'^1', '')
                ELSE phone
            END AS normalized_phone
        FROM contacts
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        phone_edges = [
            e for e in graph.edges if e.to_node.column_name == "normalized_phone"
        ]
        assert len(phone_edges) > 0

        source_columns = {e.from_node.column_name for e in phone_edges}
        assert "phone" in source_columns


# ============================================================================
# Test Group 7: Regex in WHERE Clause (column still tracked in output)
# ============================================================================


class TestRegexpInWhereClause:
    """Test that regex filtering in WHERE does not break output column lineage."""

    def test_regexp_where_preserves_select_lineage(self):
        """Test that SELECT columns are still tracked when WHERE uses regex."""
        sql = """
        SELECT
            user_id,
            email,
            REGEXP_EXTRACT(email, r'@(.+)$') AS email_domain
        FROM users
        WHERE REGEXP_CONTAINS(email, r'@company\\.com$')
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        # user_id should still trace to source
        user_id_edges = [
            e for e in graph.edges if e.to_node.column_name == "user_id"
        ]
        assert len(user_id_edges) > 0

        # email_domain should trace back to email
        domain_edges = [
            e for e in graph.edges if e.to_node.column_name == "email_domain"
        ]
        assert len(domain_edges) > 0

        source_columns = {e.from_node.column_name for e in domain_edges}
        assert "email" in source_columns


# ============================================================================
# Test Group 8: Cross-Dialect Regex Patterns
# ============================================================================


class TestRegexpDialects:
    """Test the same regex pattern across BigQuery, PostgreSQL, and Snowflake."""

    def test_regexp_replace_bigquery(self):
        """Test REGEXP_REPLACE lineage in BigQuery dialect."""
        sql = """
        SELECT
            REGEXP_REPLACE(name, r'[^a-zA-Z]', '') AS clean_name
        FROM employees
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        clean_edges = [
            e for e in graph.edges if e.to_node.column_name == "clean_name"
        ]
        assert len(clean_edges) > 0

        source_columns = {e.from_node.column_name for e in clean_edges}
        assert "name" in source_columns

    def test_regexp_replace_postgres(self):
        """Test REGEXP_REPLACE lineage in PostgreSQL dialect."""
        sql = """
        SELECT
            REGEXP_REPLACE(name, '[^a-zA-Z]', '', 'g') AS clean_name
        FROM employees
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        clean_edges = [
            e for e in graph.edges if e.to_node.column_name == "clean_name"
        ]
        assert len(clean_edges) > 0

        source_columns = {e.from_node.column_name for e in clean_edges}
        assert "name" in source_columns

    def test_regexp_replace_snowflake(self):
        """Test REGEXP_REPLACE lineage in Snowflake dialect."""
        sql = """
        SELECT
            REGEXP_REPLACE(name, '[^a-zA-Z]', '') AS clean_name
        FROM employees
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        clean_edges = [
            e for e in graph.edges if e.to_node.column_name == "clean_name"
        ]
        assert len(clean_edges) > 0

        source_columns = {e.from_node.column_name for e in clean_edges}
        assert "name" in source_columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
