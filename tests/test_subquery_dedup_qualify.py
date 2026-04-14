"""
Test suite for Gap 2: Subquery-Based Dedup Qualify Promotion.

Tests that the common dedup pattern:
    SELECT ... FROM (SELECT *, ROW_NUMBER() OVER (...) AS rn FROM t) WHERE rn = 1
is promoted to qualify metadata on the outer query unit.
"""

import pytest

from clgraph.query_parser import RecursiveQueryParser

# ============================================================================
# Test Group 1: Qualify Promotion from Subquery Dedup Pattern
# ============================================================================


class TestDedupQualifyPromotion:
    """Test promotion of subquery-based dedup WHERE to qualify_info."""

    def test_qualify_promotion_eq(self):
        """WHERE rn = 1 with ROW_NUMBER promotes qualify_info on outer unit."""
        sql = """
        SELECT id, name
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts DESC) AS rn
            FROM t
        )
        WHERE rn = 1
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        assert main_unit.qualify_info is not None
        assert main_unit.qualify_info["promoted_from_subquery"] is True
        assert "ROW_NUMBER" in main_unit.qualify_info["window_functions"]
        assert "id" in main_unit.qualify_info["partition_columns"]

    def test_qualify_promotion_lte(self):
        """WHERE rn <= 3 with ROW_NUMBER promotes qualify_info."""
        sql = """
        SELECT id, name
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts DESC) AS rn
            FROM t
        )
        WHERE rn <= 3
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        assert main_unit.qualify_info is not None
        assert main_unit.qualify_info["promoted_from_subquery"] is True
        assert "ROW_NUMBER" in main_unit.qualify_info["window_functions"]

    def test_qualify_promotion_lt(self):
        """WHERE rn < 3 with ROW_NUMBER promotes qualify_info."""
        sql = """
        SELECT id, name
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts DESC) AS rn
            FROM t
        )
        WHERE rn < 3
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        assert main_unit.qualify_info is not None
        assert main_unit.qualify_info["promoted_from_subquery"] is True
        assert "ROW_NUMBER" in main_unit.qualify_info["window_functions"]


# ============================================================================
# Test Group 2: Non-Ranking Functions Should NOT Promote
# ============================================================================


class TestNonRankingNotPromoted:
    """Test that non-ranking window functions are not promoted."""

    def test_sum_window_not_promoted(self):
        """SUM() OVER (...) + WHERE total > 100 should NOT produce qualify_info."""
        sql = """
        SELECT id, total
        FROM (
            SELECT id, SUM(amount) OVER (PARTITION BY id) AS total
            FROM t
        )
        WHERE total > 100
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        assert main_unit.qualify_info is None


# ============================================================================
# Test Group 3: Explicit QUALIFY Not Overwritten
# ============================================================================


class TestExplicitQualifyNotOverwritten:
    """Test that explicit QUALIFY clause is not overwritten by promotion."""

    def test_explicit_qualify_preserved(self):
        """Explicit QUALIFY should remain; promotion should not overwrite."""
        sql = """
        SELECT customer_id, order_date
        FROM orders
        QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        assert main_unit.qualify_info is not None
        # Explicit QUALIFY should NOT have promoted_from_subquery
        assert main_unit.qualify_info.get("promoted_from_subquery") is not True


# ============================================================================
# Test Group 4: rn Not in Output Columns
# ============================================================================


class TestRnNotInOutput:
    """Test that the ranking alias (rn) is not in the outer unit output columns."""

    def test_rn_not_in_output(self):
        """Outer SELECT id, name should not include rn in output columns."""
        sql = """
        SELECT id, name
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts DESC) AS rn
            FROM t
        )
        WHERE rn = 1
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        main_unit = graph.units["main"]
        output_col_names = [c.get("name", "") for c in main_unit.output_columns]
        assert "rn" not in output_col_names


# ============================================================================
# Test Group 5: ranking_window_columns Populated on Inner Unit
# ============================================================================


class TestRankingWindowColumns:
    """Test that inner subquery unit has ranking_window_columns metadata."""

    def test_ranking_window_columns_populated(self):
        """Inner unit should have ranking_window_columns with correct metadata."""
        sql = """
        SELECT id, name
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY ts DESC) AS rn
            FROM t
        )
        WHERE rn = 1
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        graph = parser.parse()

        # Find the inner subquery unit (not 'main')
        inner_units = [u for uid, u in graph.units.items() if uid != "main"]
        assert len(inner_units) >= 1

        # At least one inner unit should have ranking_window_columns
        inner_with_ranking = [u for u in inner_units if u.ranking_window_columns]
        assert len(inner_with_ranking) >= 1

        inner_unit = inner_with_ranking[0]
        assert "rn" in inner_unit.ranking_window_columns
        rn_meta = inner_unit.ranking_window_columns["rn"]
        assert rn_meta["function"] == "ROW_NUMBER"
        assert "id" in rn_meta["partition_by"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
