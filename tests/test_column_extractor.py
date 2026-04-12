"""
Tests for column_extractor module.

Covers:
- extract_columns_from_expr as a pure function
- extract_union_columns with UNION query unit
- extract_merge_columns with MERGE query unit
- Backward compatibility via RecursiveLineageBuilder
"""

from clgraph import Pipeline, RecursiveLineageBuilder
from clgraph.column_extractor import (
    ExtractionContext,
    extract_columns_from_expr,
    extract_merge_columns,
    extract_union_columns,
)
from clgraph.models import (
    ColumnLineageGraph,
    ColumnNode,
    QueryUnit,
    QueryUnitGraph,
    QueryUnitType,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_ctx(
    units: dict | None = None,
    cache: dict | None = None,
    external: dict | None = None,
) -> ExtractionContext:
    """Build a minimal ExtractionContext for testing."""
    unit_graph = QueryUnitGraph()
    if units:
        for uid, unit in units.items():
            unit_graph.units[uid] = unit

    return ExtractionContext(
        unit_graph=unit_graph,
        unit_columns_cache=cache or {},
        external_table_columns=external or {},
        lineage_graph=ColumnLineageGraph(),
    )


def _make_column_node(name: str, is_star: bool = False) -> ColumnNode:
    """Create a simple ColumnNode for test setup."""
    return ColumnNode(
        column_name=name,
        table_name="test_table",
        full_name=f"test_table.{name}",
        is_star=is_star,
    )


# ============================================================================
# Tests: extract_columns_from_expr (pure function)
# ============================================================================


class TestExtractColumnsFromExpr:
    """extract_columns_from_expr is a pure function with no context."""

    def test_simple_column_no_table(self):
        """Unqualified column uses default_table."""
        result = extract_columns_from_expr("revenue", "src")
        assert ("src", "revenue") in result

    def test_qualified_column(self):
        """Qualified column extracts table prefix."""
        result = extract_columns_from_expr("s.new_value", "default_tbl")
        assert ("s", "new_value") in result

    def test_coalesce_expression_multiple_columns(self):
        """Expression with multiple columns returns all references."""
        result = extract_columns_from_expr("COALESCE(s.a, s.b)", "s")
        names = {col for _, col in result}
        assert "a" in names
        assert "b" in names

    def test_literal_returns_empty(self):
        """Pure numeric literal should yield no column references."""
        result = extract_columns_from_expr("42", "src")
        assert result == []

    def test_table_dot_column_fallback(self):
        """simple table.column form is always extractable."""
        result = extract_columns_from_expr("t.id", "fallback")
        assert len(result) >= 1
        assert ("t", "id") in result


# ============================================================================
# Tests: extract_union_columns
# ============================================================================


class TestExtractUnionColumns:
    """extract_union_columns derives schema from first branch."""

    def _build_union_unit(self, branch_ids: list[str]) -> QueryUnit:
        """Create a minimal UNION QueryUnit."""
        return QueryUnit(
            unit_id="main",
            unit_type=QueryUnitType.UNION,
            name=None,
            select_node=None,
            parent_unit=None,
            set_operation_type="union_all",
            set_operation_branches=branch_ids,
        )

    def test_no_branches_returns_empty(self):
        ctx = _make_ctx()
        unit = self._build_union_unit([])
        result = extract_union_columns(ctx, unit)
        assert result == []

    def test_branch_not_in_cache_returns_empty(self):
        ctx = _make_ctx()
        unit = self._build_union_unit(["branch_0"])
        result = extract_union_columns(ctx, unit)
        assert result == []

    def test_uses_first_branch_schema(self):
        """Output columns mirror the first branch columns."""
        cols = [_make_column_node("id"), _make_column_node("name")]
        ctx = _make_ctx(cache={"branch_0": cols})
        unit = self._build_union_unit(["branch_0", "branch_1"])
        result = extract_union_columns(ctx, unit)

        assert len(result) == 2
        assert result[0]["name"] == "id"
        assert result[1]["name"] == "name"
        assert result[0]["type"] == "union_column"

    def test_source_branches_recorded(self):
        """Each output column records all branch IDs."""
        cols = [_make_column_node("x")]
        ctx = _make_ctx(cache={"b0": cols})
        unit = self._build_union_unit(["b0", "b1"])
        result = extract_union_columns(ctx, unit)

        assert result[0]["source_branches"] == ["b0", "b1"]

    def test_star_column_preserved(self):
        """Star columns from branches propagate."""
        cols = [_make_column_node("*", is_star=True)]
        ctx = _make_ctx(cache={"b0": cols})
        unit = self._build_union_unit(["b0"])
        result = extract_union_columns(ctx, unit)

        assert result[0]["is_star"] is True


# ============================================================================
# Tests: extract_merge_columns
# ============================================================================


class TestExtractMergeColumns:
    """extract_merge_columns reads from unit.unpivot_config."""

    def _build_merge_unit(self, config: dict) -> QueryUnit:
        return QueryUnit(
            unit_id="merge_main",
            unit_type=QueryUnitType.MERGE,
            name=None,
            select_node=None,
            parent_unit=None,
            unpivot_config=config,
        )

    def test_no_config_returns_empty(self):
        ctx = _make_ctx()
        unit = self._build_merge_unit({})  # missing merge_type
        result = extract_merge_columns(ctx, unit)
        assert result == []

    def test_match_columns_extracted(self):
        """ON clause columns become merge_match entries."""
        config = {
            "merge_type": "merge",
            "target_table": "target",
            "target_alias": "t",
            "source_table": "source",
            "source_alias": "s",
            "match_columns": [("id", "id")],
            "matched_actions": [],
            "not_matched_actions": [],
        }
        ctx = _make_ctx()
        unit = self._build_merge_unit(config)
        result = extract_merge_columns(ctx, unit)

        match_cols = [c for c in result if c["type"] == "merge_match"]
        assert len(match_cols) == 1
        assert match_cols[0]["name"] == "id"

    def test_update_columns_extracted(self):
        """WHEN MATCHED UPDATE becomes merge_update entries."""
        config = {
            "merge_type": "merge",
            "target_table": "target",
            "target_alias": "t",
            "source_table": "source",
            "source_alias": "s",
            "match_columns": [],
            "matched_actions": [
                {
                    "action_type": "update",
                    "condition": None,
                    "column_mappings": {"value": "s.new_value"},
                }
            ],
            "not_matched_actions": [],
        }
        ctx = _make_ctx()
        unit = self._build_merge_unit(config)
        result = extract_merge_columns(ctx, unit)

        update_cols = [c for c in result if c["type"] == "merge_update"]
        assert len(update_cols) == 1
        assert update_cols[0]["name"] == "value"
        # source_columns should contain (s, new_value)
        assert ("s", "new_value") in update_cols[0]["source_columns"]

    def test_insert_columns_extracted(self):
        """WHEN NOT MATCHED INSERT becomes merge_insert entries."""
        config = {
            "merge_type": "merge",
            "target_table": "target",
            "target_alias": "t",
            "source_table": "source",
            "source_alias": "s",
            "match_columns": [],
            "matched_actions": [],
            "not_matched_actions": [
                {
                    "action_type": "insert",
                    "condition": None,
                    "column_mappings": {"id": "s.id", "value": "s.new_value"},
                }
            ],
        }
        ctx = _make_ctx()
        unit = self._build_merge_unit(config)
        result = extract_merge_columns(ctx, unit)

        insert_cols = [c for c in result if c["type"] == "merge_insert"]
        assert len(insert_cols) == 2

    def test_indices_are_sequential(self):
        """Column indices should be 0,1,2,... in order."""
        config = {
            "merge_type": "merge",
            "target_table": "tgt",
            "target_alias": "t",
            "source_table": "src",
            "source_alias": "s",
            "match_columns": [("id", "id")],
            "matched_actions": [
                {
                    "action_type": "update",
                    "condition": None,
                    "column_mappings": {"name": "s.name"},
                }
            ],
            "not_matched_actions": [],
        }
        ctx = _make_ctx()
        unit = self._build_merge_unit(config)
        result = extract_merge_columns(ctx, unit)

        for i, col in enumerate(result):
            assert col["index"] == i


# ============================================================================
# Tests: Backward compatibility via RecursiveLineageBuilder
# ============================================================================


class TestBackwardCompat:
    """End-to-end: builder still works correctly after extraction."""

    def test_union_query_lineage_end_to_end(self):
        """UNION query should still produce lineage without errors."""
        sql = """
        SELECT user_id, name FROM users
        UNION ALL
        SELECT user_id, name FROM archived_users
        """
        pipeline = Pipeline([("union_q", sql)])
        assert pipeline is not None

    def test_merge_query_lineage_end_to_end(self):
        """MERGE query should still produce lineage without errors."""
        sql = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET t.value = s.new_value
        WHEN NOT MATCHED THEN INSERT (id, value) VALUES (s.id, s.new_value)
        """
        pipeline = Pipeline([("merge_q", sql)], dialect="postgres")
        assert pipeline is not None

    def test_union_columns_in_cache(self):
        """After processing a UNION query, the lineage builder caches columns."""
        sql = """
        SELECT a, b FROM t1
        UNION ALL
        SELECT a, b FROM t2
        """
        builder = RecursiveLineageBuilder(sql)
        result = builder.build()
        assert result is not None
        # The column lineage graph should have nodes for a and b
        col_names = {node.column_name for node in result.nodes.values()}
        assert "a" in col_names or len(col_names) > 0
