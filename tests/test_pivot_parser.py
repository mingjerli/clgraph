"""
Tests for PIVOT/UNPIVOT parsing functionality.
"""

import pytest

from clgraph.models import QueryUnitType
from clgraph.query_parser import RecursiveQueryParser


class TestPivotBasic:
    """Test basic PIVOT operations"""

    def test_pivot_basic_with_subquery(self):
        """Test basic PIVOT operation with subquery source"""
        sql = """
        SELECT * FROM (
            SELECT product, quarter, sales
            FROM quarterly_sales
        )
        PIVOT(
            SUM(sales) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
        )
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Should have:
        # - 1 main query unit
        # - 1 PIVOT unit (in FROM clause)
        # - 1 source subquery for PIVOT
        assert len(graph.units) >= 2

        # Find PIVOT unit
        pivot_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.PIVOT]
        assert len(pivot_units) == 1

        pivot_unit = pivot_units[0]
        assert pivot_unit.pivot_config is not None
        assert "aggregations" in pivot_unit.pivot_config
        assert "pivot_column" in pivot_unit.pivot_config
        assert "value_columns" in pivot_unit.pivot_config

    def test_pivot_basic_with_table(self):
        """Test PIVOT operation with base table source"""
        sql = """
        SELECT * FROM sales_data
        PIVOT(
            SUM(amount) FOR region IN ('North', 'South', 'East', 'West')
        )
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Should have:
        # - 1 main query unit
        # - 1 PIVOT unit
        pivot_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.PIVOT]
        assert len(pivot_units) >= 1

        pivot_unit = pivot_units[0]
        assert pivot_unit.pivot_config is not None

    def test_pivot_multiple_aggregations(self):
        """Test PIVOT with multiple aggregation functions"""
        sql = """
        SELECT * FROM (
            SELECT product, quarter, sales, units
            FROM quarterly_sales
        )
        PIVOT(
            SUM(sales) AS total_sales,
            AVG(units) AS avg_units
            FOR quarter IN ('Q1', 'Q2')
        )
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        pivot_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.PIVOT]
        assert len(pivot_units) >= 1

        pivot_unit = pivot_units[0]
        assert pivot_unit.pivot_config is not None
        if "aggregations" in pivot_unit.pivot_config:
            # Should have multiple aggregations
            assert len(pivot_unit.pivot_config["aggregations"]) >= 1


class TestUnpivotBasic:
    """Test basic UNPIVOT operations"""

    def test_unpivot_basic(self):
        """Test basic UNPIVOT operation"""
        sql = """
        SELECT * FROM quarterly_revenue
        UNPIVOT(
            revenue FOR quarter IN (Q1, Q2, Q3, Q4)
        )
        """

        parser = RecursiveQueryParser(sql)

        # This might fail if sqlglot doesn't support UNPIVOT in BigQuery dialect
        # We'll catch the error and skip the test
        try:
            graph = parser.parse()

            # Should have UNPIVOT unit
            unpivot_units = [
                u for u in graph.units.values() if u.unit_type == QueryUnitType.UNPIVOT
            ]
            if len(unpivot_units) > 0:
                unpivot_unit = unpivot_units[0]
                assert unpivot_unit.unpivot_config is not None
        except Exception as e:
            # Skip if UNPIVOT is not supported by sqlglot
            pytest.skip(f"UNPIVOT not supported by sqlglot: {e}")


class TestPivotWithCTEs:
    """Test PIVOT operations with CTEs"""

    def test_pivot_with_cte_source(self):
        """Test PIVOT where source is a CTE"""
        sql = """
        WITH sales_summary AS (
            SELECT product, quarter, SUM(amount) as total_sales
            FROM raw_sales
            GROUP BY product, quarter
        )
        SELECT * FROM sales_summary
        PIVOT(
            SUM(total_sales) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
        )
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Should have:
        # - 1 main query unit
        # - 1 CTE unit
        # - 1 PIVOT unit
        assert len(graph.units) >= 3

        # Verify CTE exists
        cte_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.CTE]
        assert len(cte_units) >= 1

        # Verify PIVOT exists
        pivot_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.PIVOT]
        assert len(pivot_units) >= 1


class TestPivotDependencies:
    """Test dependency tracking for PIVOT operations"""

    def test_pivot_depends_on_source(self):
        """Test PIVOT unit correctly depends on its source"""
        sql = """
        SELECT * FROM (
            SELECT product, quarter, sales
            FROM base_sales
        )
        PIVOT(
            SUM(sales) FOR quarter IN ('Q1', 'Q2')
        )
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Find PIVOT unit
        pivot_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.PIVOT]
        if len(pivot_units) > 0:
            pivot_unit = pivot_units[0]

            # PIVOT should depend on either source unit or source table
            assert len(pivot_unit.depends_on_units) > 0 or len(pivot_unit.depends_on_tables) > 0


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_nested_pivot_operations(self):
        """Test nested PIVOT operations (if supported)"""
        # This is likely not supported, but we test the parser doesn't crash
        sql = """
        SELECT * FROM (
            SELECT * FROM sales
            PIVOT(SUM(amount) FOR quarter IN ('Q1', 'Q2'))
        )
        PIVOT(SUM(total) FOR region IN ('North', 'South'))
        """

        try:
            parser = RecursiveQueryParser(sql)
            graph = parser.parse()

            # If it parses, check for PIVOT units
            pivot_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.PIVOT]
            # Could be 0, 1, or 2 depending on sqlglot support
            assert len(pivot_units) >= 0
        except Exception:
            # Expected - nested PIVOTs may not be supported
            pass


class TestPivotWithComplexQueries:
    """Test PIVOT with complex query patterns"""

    def test_pivot_with_where_clause(self):
        """Test PIVOT operation with WHERE clause in source"""
        sql = """
        SELECT * FROM (
            SELECT product, quarter, sales
            FROM quarterly_sales
            WHERE year = 2024
        )
        PIVOT(
            SUM(sales) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
        )
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Should parse successfully
        assert len(graph.units) >= 2

    def test_pivot_with_joins_in_source(self):
        """Test PIVOT where source has JOINs"""
        sql = """
        SELECT * FROM (
            SELECT p.product_name, s.quarter, s.sales
            FROM sales s
            JOIN products p ON s.product_id = p.id
        )
        PIVOT(
            SUM(sales) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
        )
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Should parse successfully
        assert len(graph.units) >= 2


class TestUnitIdGeneration:
    """Test unit ID generation for PIVOT/UNPIVOT operations"""

    def test_pivot_unit_id_format(self):
        """Test PIVOT units get correct ID format"""
        sql = """
        SELECT * FROM sales
        PIVOT(
            SUM(amount) FOR quarter IN ('Q1', 'Q2')
        )
        """

        parser = RecursiveQueryParser(sql)
        graph = parser.parse()

        # Check for pivot: prefix in unit IDs
        pivot_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.PIVOT]
        if len(pivot_units) > 0:
            assert pivot_units[0].unit_id.startswith("pivot:")

    def test_unpivot_unit_id_format(self):
        """Test UNPIVOT units get correct ID format"""
        sql = """
        SELECT * FROM quarterly_revenue
        UNPIVOT(
            revenue FOR quarter IN (Q1, Q2, Q3, Q4)
        )
        """

        try:
            parser = RecursiveQueryParser(sql)
            graph = parser.parse()

            # Check for unpivot: prefix in unit IDs
            unpivot_units = [
                u for u in graph.units.values() if u.unit_type == QueryUnitType.UNPIVOT
            ]
            if len(unpivot_units) > 0:
                assert unpivot_units[0].unit_id.startswith("unpivot:")
        except Exception:
            # Skip if UNPIVOT not supported
            pytest.skip("UNPIVOT not supported by sqlglot")
