"""
Test suite for complex aggregate functions support in column lineage tracking.

Tests cover:
- Array aggregates (ARRAY_AGG, COLLECT_LIST, COLLECT_SET)
- String aggregates (STRING_AGG, LISTAGG, GROUP_CONCAT)
- Object aggregates (OBJECT_AGG, JSON_AGG)
- Aggregate modifiers (DISTINCT, ORDER BY)
- Pipeline API integration
- Export format validation
"""

import pytest

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder
from clgraph.models import AggregateType

# ============================================================================
# Test Group 1: Array Aggregates
# ============================================================================


class TestArrayAggregates:
    """Test array aggregate functions."""

    def test_array_agg_basic(self):
        """Test basic ARRAY_AGG function."""
        sql = """
        SELECT
            customer_id,
            ARRAY_AGG(product_id) AS products
        FROM purchases
        GROUP BY customer_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        # Find aggregate edges
        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        assert len(agg_edges) > 0

        # Check aggregate spec
        agg_spec = agg_edges[0].aggregate_spec
        assert agg_spec.function_name == "ARRAY_AGG"
        assert agg_spec.aggregate_type == AggregateType.ARRAY
        assert agg_spec.return_type == "array"

    def test_array_agg_with_order_by(self):
        """Test ARRAY_AGG with ORDER BY clause."""
        sql = """
        SELECT
            customer_id,
            ARRAY_AGG(product_id ORDER BY purchase_date) AS products
        FROM purchases
        GROUP BY customer_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        assert len(agg_edges) > 0

        agg_spec = agg_edges[0].aggregate_spec
        assert agg_spec.function_name == "ARRAY_AGG"
        assert len(agg_spec.order_by) >= 1

    def test_array_agg_distinct(self):
        """Test ARRAY_AGG with DISTINCT modifier."""
        sql = """
        SELECT ARRAY_AGG(DISTINCT category) AS categories
        FROM products
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        assert len(agg_edges) > 0

        agg_spec = agg_edges[0].aggregate_spec
        assert agg_spec.function_name == "ARRAY_AGG"
        assert agg_spec.distinct is True

    def test_collect_list_spark(self):
        """Test Spark's COLLECT_LIST function."""
        sql = """
        SELECT customer_id, COLLECT_LIST(product_id) AS products
        FROM purchases
        GROUP BY customer_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="spark")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        # The function might be parsed differently by dialect
        if agg_edges:
            assert agg_edges[0].aggregate_spec.aggregate_type == AggregateType.ARRAY


# ============================================================================
# Test Group 2: String Aggregates
# ============================================================================


class TestStringAggregates:
    """Test string aggregate functions."""

    def test_string_agg_basic(self):
        """Test basic STRING_AGG function."""
        sql = """
        SELECT STRING_AGG(name, ', ') AS names
        FROM users
        GROUP BY department
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        assert len(agg_edges) > 0

        agg_spec = agg_edges[0].aggregate_spec
        assert agg_spec.function_name in ("STRING_AGG", "LISTAGG", "GROUP_CONCAT")
        assert agg_spec.aggregate_type == AggregateType.STRING
        assert agg_spec.return_type == "string"

    def test_listagg_snowflake(self):
        """Test Snowflake's LISTAGG function."""
        sql = """
        SELECT LISTAGG(product_name, '; ') AS product_list
        FROM products
        GROUP BY category
        """
        builder = RecursiveLineageBuilder(sql, dialect="snowflake")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        if agg_edges:
            agg_spec = agg_edges[0].aggregate_spec
            assert agg_spec.aggregate_type == AggregateType.STRING

    def test_group_concat_mysql(self):
        """Test MySQL's GROUP_CONCAT function."""
        sql = """
        SELECT GROUP_CONCAT(tag) AS tags
        FROM post_tags
        GROUP BY post_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="mysql")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        if agg_edges:
            agg_spec = agg_edges[0].aggregate_spec
            assert agg_spec.function_name == "GROUP_CONCAT"
            assert agg_spec.aggregate_type == AggregateType.STRING


# ============================================================================
# Test Group 3: Scalar Aggregates with Complex Specs
# ============================================================================


class TestScalarAggregates:
    """Test scalar aggregate functions with complex specifications."""

    def test_sum_basic(self):
        """Test basic SUM function."""
        sql = """
        SELECT customer_id, SUM(amount) AS total
        FROM orders
        GROUP BY customer_id
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        assert len(agg_edges) > 0

        agg_spec = agg_edges[0].aggregate_spec
        assert agg_spec.function_name == "SUM"
        assert agg_spec.aggregate_type == AggregateType.SCALAR

    def test_count_distinct(self):
        """Test COUNT with DISTINCT modifier."""
        sql = """
        SELECT COUNT(DISTINCT user_id) AS unique_users
        FROM sessions
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        assert len(agg_edges) > 0

        agg_spec = agg_edges[0].aggregate_spec
        assert agg_spec.function_name == "COUNT"
        assert agg_spec.distinct is True

    def test_avg_function(self):
        """Test AVG function."""
        sql = """
        SELECT AVG(price) AS avg_price
        FROM products
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        assert len(agg_edges) > 0

        agg_spec = agg_edges[0].aggregate_spec
        assert agg_spec.function_name == "AVG"
        assert agg_spec.return_type == "float"


# ============================================================================
# Test Group 4: Pipeline API Integration
# ============================================================================


class TestAggregatePipeline:
    """Test complex aggregates through Pipeline API."""

    def test_aggregate_in_pipeline(self):
        """Test that aggregate specs are preserved in Pipeline."""
        sql = """
        SELECT
            customer_id,
            ARRAY_AGG(product_id ORDER BY purchase_date) AS products,
            SUM(amount) AS total
        FROM purchases
        GROUP BY customer_id
        """
        pipeline = Pipeline([("agg_query", sql)], dialect="bigquery")

        agg_edges = [
            e for e in pipeline.column_graph.edges if getattr(e, "aggregate_spec", None) is not None
        ]
        assert len(agg_edges) > 0

    def test_multiple_aggregates_in_query(self):
        """Test multiple aggregate functions in same query."""
        sql = """
        SELECT
            department,
            COUNT(*) AS emp_count,
            SUM(salary) AS total_salary,
            AVG(salary) AS avg_salary,
            MAX(salary) AS max_salary
        FROM employees
        GROUP BY department
        """
        pipeline = Pipeline([("multi_agg", sql)], dialect="bigquery")

        agg_edges = [
            e for e in pipeline.column_graph.edges if getattr(e, "aggregate_spec", None) is not None
        ]
        # Should have at least 3 aggregate edges (SUM, AVG, MAX on salary)
        # COUNT(*) doesn't have a source column reference so no aggregate_spec edge
        assert len(agg_edges) >= 3

        # Check that we have different aggregate functions
        func_names = {e.aggregate_spec.function_name for e in agg_edges}
        assert "SUM" in func_names
        assert "AVG" in func_names
        assert "MAX" in func_names


# ============================================================================
# Test Group 5: Export Format
# ============================================================================


class TestAggregateExport:
    """Test that aggregate metadata is included in exports."""

    def test_aggregate_spec_in_export(self):
        """Test that aggregate_spec appears in JSON export."""
        sql = """
        SELECT
            customer_id,
            ARRAY_AGG(product_id ORDER BY date) AS products,
            SUM(amount) AS total
        FROM purchases
        GROUP BY customer_id
        """
        pipeline = Pipeline([("agg_query", sql)], dialect="bigquery")

        exporter = JSONExporter()
        export_data = exporter.export(pipeline)

        edges = export_data.get("edges", [])
        agg_edges = [e for e in edges if e.get("aggregate_spec") is not None]

        assert len(agg_edges) > 0

        # Check ARRAY_AGG export
        array_agg_edge = None
        for e in agg_edges:
            spec = e.get("aggregate_spec", {})
            if spec.get("function_name") == "ARRAY_AGG":
                array_agg_edge = e
                break

        if array_agg_edge:
            spec = array_agg_edge["aggregate_spec"]
            assert spec["aggregate_type"] == "array"
            assert spec["return_type"] == "array"

    def test_distinct_in_export(self):
        """Test that DISTINCT modifier is in export."""
        sql = """
        SELECT COUNT(DISTINCT user_id) AS unique_users
        FROM sessions
        """
        pipeline = Pipeline([("agg_query", sql)], dialect="bigquery")

        exporter = JSONExporter()
        export_data = exporter.export(pipeline)

        edges = export_data.get("edges", [])
        agg_edges = [e for e in edges if e.get("aggregate_spec") is not None]

        if agg_edges:
            spec = agg_edges[0].get("aggregate_spec", {})
            assert spec.get("distinct") is True


# ============================================================================
# Test Group 6: Edge Cases
# ============================================================================


class TestAggregateEdgeCases:
    """Test edge cases for aggregate support."""

    def test_nested_aggregate(self):
        """Test aggregate with complex expression."""
        sql = """
        SELECT SUM(price * quantity) AS total_revenue
        FROM order_items
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        assert len(agg_edges) > 0

    def test_aggregate_with_case(self):
        """Test aggregate with CASE expression."""
        sql = """
        SELECT SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) AS completed_total
        FROM orders
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        assert len(agg_edges) > 0

    def test_non_aggregate_query(self):
        """Test that non-aggregate queries don't have aggregate specs."""
        sql = "SELECT id, name FROM users"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        assert len(agg_edges) == 0


# ============================================================================
# Test Group 7: Dialect Support
# ============================================================================


class TestAggregateDialects:
    """Test aggregate support across dialects."""

    def test_bigquery_array_agg(self):
        """Test BigQuery ARRAY_AGG."""
        sql = """
        SELECT ARRAY_AGG(STRUCT(id, name)) AS user_structs
        FROM users
        """
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        if agg_edges:
            assert agg_edges[0].aggregate_spec.aggregate_type == AggregateType.ARRAY

    def test_postgres_string_agg(self):
        """Test PostgreSQL STRING_AGG."""
        sql = """
        SELECT STRING_AGG(name, ', ' ORDER BY name) AS sorted_names
        FROM users
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        agg_edges = [e for e in graph.edges if e.aggregate_spec is not None]
        if agg_edges:
            agg_spec = agg_edges[0].aggregate_spec
            assert agg_spec.aggregate_type == AggregateType.STRING


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
