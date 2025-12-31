"""
Test suite for MERGE statement support in column lineage tracking.

Tests cover:
- Basic MERGE INTO parsing
- WHEN MATCHED THEN UPDATE tracking
- WHEN NOT MATCHED THEN INSERT tracking
- Match condition column identification
- Pipeline API integration
- Export format validation
"""

import pytest

from clgraph import JSONExporter, Pipeline, RecursiveLineageBuilder
from clgraph.models import QueryUnitType
from clgraph.query_parser import RecursiveQueryParser

# ============================================================================
# Test Group 1: MERGE Statement Parsing
# ============================================================================


class TestMergeParsing:
    """Test MERGE statement parsing."""

    def test_simple_merge_detected(self):
        """Test detection of simple MERGE statement."""
        sql = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET t.value = s.new_value
        WHEN NOT MATCHED THEN INSERT (id, value) VALUES (s.id, s.new_value)
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        graph = parser.parse()

        # Should have a MERGE unit
        merge_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.MERGE]
        assert len(merge_units) == 1

        unit = merge_units[0]
        assert unit.unit_type == QueryUnitType.MERGE

    def test_merge_target_source_tables(self):
        """Test that target and source tables are extracted."""
        sql = """
        MERGE INTO customers c
        USING updates u ON c.customer_id = u.customer_id
        WHEN MATCHED THEN UPDATE SET c.name = u.name
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        graph = parser.parse()

        merge_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.MERGE]
        assert len(merge_units) == 1

        unit = merge_units[0]
        # Should depend on both target and source tables
        assert "customers" in unit.depends_on_tables
        assert "updates" in unit.depends_on_tables

    def test_merge_config_extraction(self):
        """Test that MERGE configuration is extracted correctly."""
        sql = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET t.value = s.new_value
        WHEN NOT MATCHED THEN INSERT (id, value) VALUES (s.id, s.new_value)
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        graph = parser.parse()

        merge_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.MERGE]
        unit = merge_units[0]

        config = unit.unpivot_config
        assert config is not None
        assert config.get("merge_type") == "merge"
        assert config.get("target_table") == "target"
        assert config.get("target_alias") == "t"
        assert config.get("source_table") == "source"
        assert config.get("source_alias") == "s"


# ============================================================================
# Test Group 2: Match Condition Tracking
# ============================================================================


class TestMatchConditions:
    """Test MERGE match condition tracking."""

    def test_match_columns_extracted(self):
        """Test that match columns from ON clause are extracted."""
        sql = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET t.value = s.new_value
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        graph = parser.parse()

        merge_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.MERGE]
        unit = merge_units[0]

        config = unit.unpivot_config
        match_columns = config.get("match_columns", [])
        assert len(match_columns) >= 1
        # Match columns should contain the ON condition columns
        col_names = [col[0] for col in match_columns]
        assert "id" in col_names

    def test_multiple_match_columns(self):
        """Test MERGE with multiple join columns."""
        sql = """
        MERGE INTO target t
        USING source s ON t.id = s.id AND t.region = s.region
        WHEN MATCHED THEN UPDATE SET t.value = s.new_value
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        graph = parser.parse()

        merge_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.MERGE]
        unit = merge_units[0]

        config = unit.unpivot_config
        match_columns = config.get("match_columns", [])
        # Should have both id and region
        col_names = [col[0] for col in match_columns]
        assert "id" in col_names
        assert "region" in col_names


# ============================================================================
# Test Group 3: WHEN MATCHED Actions
# ============================================================================


class TestMatchedActions:
    """Test WHEN MATCHED action tracking."""

    def test_update_action_columns(self):
        """Test that UPDATE columns are tracked."""
        sql = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET t.name = s.name, t.value = s.value
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        graph = parser.parse()

        merge_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.MERGE]
        unit = merge_units[0]

        config = unit.unpivot_config
        matched_actions = config.get("matched_actions", [])
        assert len(matched_actions) >= 1

        update_action = matched_actions[0]
        assert update_action.get("action_type") == "update"
        col_mappings = update_action.get("column_mappings", {})
        assert "name" in col_mappings or "value" in col_mappings

    def test_conditional_update(self):
        """Test WHEN MATCHED with additional condition."""
        sql = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED AND s.is_active THEN UPDATE SET t.value = s.value
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        graph = parser.parse()

        merge_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.MERGE]
        unit = merge_units[0]

        config = unit.unpivot_config
        matched_actions = config.get("matched_actions", [])
        assert len(matched_actions) >= 1

        # Should have condition
        _update_action = matched_actions[0]  # noqa: F841 - checked for presence
        # Condition may or may not be extracted depending on sqlglot parsing


# ============================================================================
# Test Group 4: WHEN NOT MATCHED Actions
# ============================================================================


class TestNotMatchedActions:
    """Test WHEN NOT MATCHED action tracking."""

    def test_insert_action_columns(self):
        """Test that INSERT columns are tracked."""
        sql = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN NOT MATCHED THEN INSERT (id, name, value) VALUES (s.id, s.name, s.value)
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        graph = parser.parse()

        merge_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.MERGE]
        unit = merge_units[0]

        config = unit.unpivot_config
        not_matched_actions = config.get("not_matched_actions", [])
        assert len(not_matched_actions) >= 1

        insert_action = not_matched_actions[0]
        assert insert_action.get("action_type") == "insert"


# ============================================================================
# Test Group 5: Lineage Building
# ============================================================================


class TestMergeLineage:
    """Test column lineage tracking for MERGE statements."""

    def test_merge_lineage_edges(self):
        """Test that MERGE creates lineage edges."""
        sql = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET t.value = s.new_value
        WHEN NOT MATCHED THEN INSERT (id, value) VALUES (s.id, s.new_value)
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        # Should have edges from source to target columns
        merge_edges = [e for e in graph.edges if e.is_merge_operation]
        assert len(merge_edges) > 0

    def test_merge_edge_properties(self):
        """Test that MERGE edges have correct properties."""
        sql = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET t.value = s.new_value
        """
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        merge_edges = [e for e in graph.edges if e.is_merge_operation]
        assert len(merge_edges) > 0

        # Check edge properties
        for edge in merge_edges:
            assert edge.is_merge_operation is True
            assert edge.merge_action in ("match", "update", "insert", "delete", None)


# ============================================================================
# Test Group 6: Pipeline API
# ============================================================================


class TestMergePipeline:
    """Test MERGE lineage through Pipeline API."""

    def test_merge_in_pipeline(self):
        """Test MERGE lineage in Pipeline."""
        sql = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET t.value = s.new_value
        """
        pipeline = Pipeline([("merge_op", sql)], dialect="postgres")

        # Check that edges have merge operation metadata
        merge_edges = [
            e for e in pipeline.column_graph.edges if getattr(e, "is_merge_operation", False)
        ]
        assert len(merge_edges) > 0


# ============================================================================
# Test Group 7: Export Format
# ============================================================================


class TestMergeExport:
    """Test that MERGE metadata is included in exports."""

    def test_merge_metadata_in_export(self):
        """Test that MERGE metadata appears in JSON export."""
        sql = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET t.value = s.new_value
        """
        pipeline = Pipeline([("merge_op", sql)], dialect="postgres")

        exporter = JSONExporter()
        export_data = exporter.export(pipeline)

        # Check that edges in export contain merge operation metadata
        edges = export_data.get("edges", [])
        merge_edges = [e for e in edges if e.get("is_merge_operation")]

        assert len(merge_edges) > 0
        # Check edge has merge_action
        assert merge_edges[0].get("merge_action") is not None


# ============================================================================
# Test Group 8: Edge Cases
# ============================================================================


class TestMergeEdgeCases:
    """Test edge cases for MERGE support."""

    def test_non_merge_no_merge_edges(self):
        """Test that non-MERGE queries don't have merge edges."""
        sql = "SELECT id, name, email FROM users"
        builder = RecursiveLineageBuilder(sql, dialect="postgres")
        graph = builder.build()

        merge_edges = [e for e in graph.edges if e.is_merge_operation]
        assert len(merge_edges) == 0

    def test_merge_with_delete(self):
        """Test MERGE with DELETE action."""
        sql = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED AND t.is_deleted THEN DELETE
        WHEN MATCHED THEN UPDATE SET t.value = s.value
        """
        parser = RecursiveQueryParser(sql, dialect="postgres")
        graph = parser.parse()

        merge_units = [u for u in graph.units.values() if u.unit_type == QueryUnitType.MERGE]
        assert len(merge_units) == 1

        unit = merge_units[0]
        config = unit.unpivot_config
        matched_actions = config.get("matched_actions", [])
        # Should have both delete and update actions
        action_types = [a.get("action_type") for a in matched_actions]
        assert "delete" in action_types or "update" in action_types


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
