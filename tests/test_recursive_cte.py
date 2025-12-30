"""
Test suite for Recursive CTE (Common Table Expression) support in column lineage tracking.

Tests cover:
- Recursive CTE detection (WITH RECURSIVE keyword)
- Base case vs recursive case parsing
- Self-reference identification
- Column lineage through recursive CTEs
- Pipeline API integration
- Export format validation
"""

import pytest

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder
from clgraph.models import QueryUnitType, RecursiveCTEInfo
from clgraph.query_parser import RecursiveQueryParser

# ============================================================================
# Test Group 1: Recursive CTE Detection
# ============================================================================


class TestRecursiveCTEDetection:
    """Test detection of recursive CTEs."""

    def test_simple_recursive_cte_detected(self):
        """Test that a simple recursive CTE is detected."""
        sql = """
        WITH RECURSIVE nums AS (
            SELECT 1 AS n
            UNION ALL
            SELECT n + 1 FROM nums WHERE n < 10
        )
        SELECT * FROM nums
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        # The CTE should be detected as recursive
        cte_unit = unit_graph.get_unit_by_name("nums")
        assert cte_unit is not None
        assert cte_unit.recursive_cte_info is not None
        assert cte_unit.recursive_cte_info.is_recursive is True

    def test_non_recursive_union_cte_not_marked_recursive(self):
        """Test that a non-recursive UNION CTE is not marked as recursive."""
        sql = """
        WITH combined AS (
            SELECT id, name FROM users
            UNION ALL
            SELECT id, name FROM admins
        )
        SELECT * FROM combined
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        # Should not have recursive CTE info
        # Non-recursive UNION CTE will be parsed as a set operation
        # Check that no unit named "combined" has recursive_cte_info
        for unit in unit_graph.units.values():
            if unit.name == "combined" and unit.recursive_cte_info is not None:
                assert unit.recursive_cte_info.is_recursive is False

    def test_regular_cte_not_recursive(self):
        """Test that a regular CTE is not detected as recursive."""
        sql = """
        WITH sales AS (
            SELECT product_id, SUM(amount) AS total
            FROM orders
            GROUP BY product_id
        )
        SELECT * FROM sales
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        cte_unit = unit_graph.get_unit_by_name("sales")
        assert cte_unit is not None
        assert cte_unit.recursive_cte_info is None


# ============================================================================
# Test Group 2: Recursive CTE Structure Parsing
# ============================================================================


class TestRecursiveCTEStructure:
    """Test parsing of recursive CTE structure."""

    def test_base_and_recursive_units_created(self):
        """Test that base and recursive case units are created."""
        sql = """
        WITH RECURSIVE hierarchy AS (
            SELECT id, parent_id, name FROM employees WHERE parent_id IS NULL
            UNION ALL
            SELECT e.id, e.parent_id, e.name FROM employees e
            JOIN hierarchy h ON e.parent_id = h.id
        )
        SELECT * FROM hierarchy
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        # Check for base and recursive units
        base_unit = unit_graph.units.get("cte_base:hierarchy_base")
        recursive_unit = unit_graph.units.get("cte_recursive:hierarchy_recursive")

        assert base_unit is not None, "Base case unit should exist"
        assert recursive_unit is not None, "Recursive case unit should exist"
        assert base_unit.unit_type == QueryUnitType.CTE_BASE
        assert recursive_unit.unit_type == QueryUnitType.CTE_RECURSIVE

    def test_self_reference_alias_captured(self):
        """Test that the self-reference alias is captured."""
        sql = """
        WITH RECURSIVE org AS (
            SELECT id, manager_id, 1 AS level FROM employees WHERE manager_id IS NULL
            UNION ALL
            SELECT e.id, e.manager_id, o.level + 1
            FROM employees e
            JOIN org o ON e.manager_id = o.id
        )
        SELECT * FROM org
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        cte_unit = unit_graph.get_unit_by_name("org")
        assert cte_unit.recursive_cte_info.self_reference_alias == "o"

    def test_join_condition_captured(self):
        """Test that the join condition is captured."""
        sql = """
        WITH RECURSIVE tree AS (
            SELECT id, parent_id FROM nodes WHERE parent_id IS NULL
            UNION ALL
            SELECT n.id, n.parent_id FROM nodes n
            JOIN tree t ON n.parent_id = t.id
        )
        SELECT * FROM tree
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        cte_unit = unit_graph.get_unit_by_name("tree")
        assert cte_unit.recursive_cte_info.join_condition is not None
        assert "parent_id" in cte_unit.recursive_cte_info.join_condition.lower()

    def test_union_type_detected(self):
        """Test that UNION vs UNION ALL is detected."""
        sql_union_all = """
        WITH RECURSIVE nums AS (
            SELECT 1 AS n
            UNION ALL
            SELECT n + 1 FROM nums WHERE n < 10
        )
        SELECT * FROM nums
        """
        parser_all = RecursiveQueryParser(sql_union_all, dialect="postgres")
        unit_graph_all = parser_all.parse()
        cte_all = unit_graph_all.get_unit_by_name("nums")
        assert cte_all.recursive_cte_info.union_type == "union_all"

    def test_column_names_extracted(self):
        """Test that column names are extracted from base and recursive cases."""
        sql = """
        WITH RECURSIVE paths AS (
            SELECT id, name, 0 AS depth FROM nodes WHERE parent_id IS NULL
            UNION ALL
            SELECT n.id, n.name, p.depth + 1
            FROM nodes n
            JOIN paths p ON n.parent_id = p.id
        )
        SELECT * FROM paths
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        cte_unit = unit_graph.get_unit_by_name("paths")
        info = cte_unit.recursive_cte_info

        assert "id" in info.base_columns
        assert "name" in info.base_columns
        assert "depth" in info.base_columns


# ============================================================================
# Test Group 3: Recursive CTE Lineage Building
# ============================================================================


class TestRecursiveCTELineage:
    """Test lineage building for recursive CTEs."""

    def test_recursive_cte_lineage_builds(self):
        """Test that lineage builds successfully for recursive CTE."""
        sql = """
        WITH RECURSIVE nums AS (
            SELECT 1 AS n
            UNION ALL
            SELECT n + 1 FROM nums WHERE n < 10
        )
        SELECT n FROM nums
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        # Should have nodes for output columns
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

    def test_hierarchy_lineage(self):
        """Test lineage for a typical hierarchy query."""
        sql = """
        WITH RECURSIVE org AS (
            SELECT id, name, manager_id, 1 AS level
            FROM employees
            WHERE manager_id IS NULL

            UNION ALL

            SELECT e.id, e.name, e.manager_id, o.level + 1
            FROM employees e
            JOIN org o ON e.manager_id = o.id
        )
        SELECT id, name, level FROM org
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        # Check that we have output nodes
        output_nodes = [n for n in graph.nodes.values() if n.layer == "output"]
        assert len(output_nodes) >= 3  # id, name, level


# ============================================================================
# Test Group 4: Pipeline Integration
# ============================================================================


class TestRecursiveCTEPipeline:
    """Test recursive CTE support through Pipeline API."""

    def test_recursive_cte_in_pipeline(self):
        """Test recursive CTE works in Pipeline."""
        sql = """
        CREATE TABLE hierarchy AS
        WITH RECURSIVE org AS (
            SELECT id, parent_id, 1 AS level FROM nodes WHERE parent_id IS NULL
            UNION ALL
            SELECT n.id, n.parent_id, o.level + 1
            FROM nodes n JOIN org o ON n.parent_id = o.id
        )
        SELECT * FROM org
        """
        pipeline = Pipeline([("build_hierarchy", sql)], dialect="postgres")

        # Check that parsing succeeded
        assert len(pipeline.table_graph.tables) > 0

    def test_recursive_cte_multi_query_pipeline(self):
        """Test recursive CTE with multiple queries in pipeline."""
        queries = [
            ("staging", "CREATE TABLE staging AS SELECT id, parent_id FROM raw_data"),
            (
                "hierarchy",
                """
                CREATE TABLE hierarchy AS
                WITH RECURSIVE tree AS (
                    SELECT id, parent_id, 1 AS depth FROM staging WHERE parent_id IS NULL
                    UNION ALL
                    SELECT s.id, s.parent_id, t.depth + 1
                    FROM staging s JOIN tree t ON s.parent_id = t.id
                )
                SELECT * FROM tree
            """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="postgres")

        # Check that both queries are in the pipeline
        assert "staging" in pipeline.table_graph.tables or len(pipeline.queries) == 2


# ============================================================================
# Test Group 5: Export Format
# ============================================================================


class TestRecursiveCTEExport:
    """Test recursive CTE metadata in exports."""

    def test_recursive_cte_info_exportable(self):
        """Test that recursive CTE info can be exported."""
        sql = """
        WITH RECURSIVE nums AS (
            SELECT 1 AS n
            UNION ALL
            SELECT n + 1 FROM nums WHERE n < 10
        )
        SELECT n FROM nums
        """
        pipeline = Pipeline([("numbers", sql)], dialect="postgres")

        exporter = JSONExporter()
        export_data = exporter.export(pipeline)

        # Export should succeed
        assert "columns" in export_data
        assert "edges" in export_data


# ============================================================================
# Test Group 6: Edge Cases
# ============================================================================


class TestRecursiveCTEEdgeCases:
    """Test edge cases for recursive CTE support."""

    def test_deeply_nested_recursive_cte(self):
        """Test recursive CTE with multiple levels of nesting."""
        sql = """
        WITH RECURSIVE
            base_data AS (
                SELECT id, parent_id FROM raw_table
            ),
            hierarchy AS (
                SELECT id, parent_id, 1 AS level FROM base_data WHERE parent_id IS NULL
                UNION ALL
                SELECT b.id, b.parent_id, h.level + 1
                FROM base_data b
                JOIN hierarchy h ON b.parent_id = h.id
            )
        SELECT * FROM hierarchy
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        # Should have both CTEs
        base_cte = unit_graph.get_unit_by_name("base_data")
        hierarchy_cte = unit_graph.get_unit_by_name("hierarchy")

        assert base_cte is not None
        assert hierarchy_cte is not None
        assert hierarchy_cte.recursive_cte_info is not None

    def test_recursive_cte_without_explicit_alias(self):
        """Test recursive CTE where self-reference has no explicit alias."""
        sql = """
        WITH RECURSIVE nums AS (
            SELECT 1 AS n
            UNION ALL
            SELECT nums.n + 1 FROM nums WHERE nums.n < 10
        )
        SELECT * FROM nums
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        cte_unit = unit_graph.get_unit_by_name("nums")
        assert cte_unit.recursive_cte_info is not None
        # Without alias, the alias should default to the CTE name
        assert cte_unit.recursive_cte_info.self_reference_alias in ["nums", "NUMS"]


# ============================================================================
# Test Group 7: Dialect Support
# ============================================================================


class TestRecursiveCTEDialects:
    """Test recursive CTE support across dialects."""

    def test_postgres_recursive_cte(self):
        """Test PostgreSQL recursive CTE."""
        sql = """
        WITH RECURSIVE nums AS (
            SELECT 1 AS n
            UNION ALL
            SELECT n + 1 FROM nums WHERE n < 10
        )
        SELECT * FROM nums
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        cte_unit = unit_graph.get_unit_by_name("nums")
        assert cte_unit.recursive_cte_info is not None

    def test_bigquery_recursive_cte(self):
        """Test BigQuery recursive CTE."""
        sql = """
        WITH RECURSIVE nums AS (
            SELECT 1 AS n
            UNION ALL
            SELECT n + 1 FROM nums WHERE n < 10
        )
        SELECT * FROM nums
        """
        parser = RecursiveQueryParser(sql, dialect="bigquery")
        unit_graph = parser.parse()

        cte_unit = unit_graph.get_unit_by_name("nums")
        assert cte_unit.recursive_cte_info is not None

    def test_snowflake_recursive_cte(self):
        """Test Snowflake recursive CTE."""
        sql = """
        WITH RECURSIVE nums AS (
            SELECT 1 AS n
            UNION ALL
            SELECT n + 1 FROM nums WHERE n < 10
        )
        SELECT * FROM nums
        """
        parser = RecursiveQueryParser(sql, dialect="snowflake")
        unit_graph = parser.parse()

        cte_unit = unit_graph.get_unit_by_name("nums")
        assert cte_unit.recursive_cte_info is not None


# ============================================================================
# Test Group 8: Complex Recursive Patterns
# ============================================================================


class TestComplexRecursivePatterns:
    """Test complex recursive CTE patterns."""

    def test_path_accumulation(self):
        """Test recursive CTE that accumulates a path string."""
        sql = """
        WITH RECURSIVE paths AS (
            SELECT id, name, CAST(name AS VARCHAR(1000)) AS path
            FROM nodes
            WHERE parent_id IS NULL

            UNION ALL

            SELECT n.id, n.name, CONCAT(p.path, ' > ', n.name)
            FROM nodes n
            JOIN paths p ON n.parent_id = p.id
        )
        SELECT * FROM paths
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        cte_unit = unit_graph.get_unit_by_name("paths")
        assert cte_unit.recursive_cte_info is not None
        assert "path" in cte_unit.recursive_cte_info.base_columns

    def test_graph_traversal(self):
        """Test recursive CTE for graph edge traversal."""
        sql = """
        WITH RECURSIVE reachable AS (
            SELECT target_id AS node_id, 1 AS hops
            FROM edges
            WHERE source_id = 1

            UNION ALL

            SELECT e.target_id, r.hops + 1
            FROM edges e
            JOIN reachable r ON e.source_id = r.node_id
            WHERE r.hops < 5
        )
        SELECT DISTINCT node_id, MIN(hops) AS min_hops
        FROM reachable
        GROUP BY node_id
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        cte_unit = unit_graph.get_unit_by_name("reachable")
        assert cte_unit.recursive_cte_info is not None


# ============================================================================
# Test Group 9: RecursiveCTEInfo Model
# ============================================================================


class TestRecursiveCTEInfoModel:
    """Test RecursiveCTEInfo dataclass."""

    def test_recursive_cte_info_creation(self):
        """Test creating RecursiveCTEInfo instance."""
        info = RecursiveCTEInfo(
            cte_name="hierarchy",
            is_recursive=True,
            base_columns=["id", "name", "level"],
            recursive_columns=["id", "name", "level"],
            union_type="union_all",
            self_reference_alias="h",
            join_condition="e.parent_id = h.id",
        )

        assert info.cte_name == "hierarchy"
        assert info.is_recursive is True
        assert len(info.base_columns) == 3
        assert info.self_reference_alias == "h"

    def test_recursive_cte_info_repr(self):
        """Test RecursiveCTEInfo string representation."""
        info = RecursiveCTEInfo(
            cte_name="test_cte",
            self_reference_alias="t",
        )

        repr_str = repr(info)
        assert "test_cte" in repr_str
        assert "t" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
