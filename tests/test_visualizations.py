"""
Tests for visualization functions in clgraph.visualizations.

These tests verify that visualization functions:
1. Return valid graphviz.Digraph objects
2. Generate DOT source with expected structure
3. Handle edge cases appropriately
"""

import graphviz
import pytest

from clgraph import (
    Pipeline,
    RecursiveQueryParser,
    visualize_column_lineage,
    visualize_column_path,
    visualize_lineage_path,
    visualize_pipeline_lineage,
    visualize_query_structure_from_lineage,
    visualize_query_units,
    visualize_table_dependencies,
    visualize_table_dependencies_with_levels,
)
from clgraph.visualizations import _sanitize_graphviz_id, visualize_column_lineage_simple


class TestSanitizeGraphvizId:
    """Tests for the _sanitize_graphviz_id helper function."""

    def test_sanitize_colons(self):
        """Colons should be replaced with double underscores."""
        assert _sanitize_graphviz_id("cte:my_cte") == "cte__my_cte"
        assert _sanitize_graphviz_id("query:table:column") == "query__table__column"

    def test_sanitize_dots(self):
        """Dots should be replaced with single underscores."""
        assert _sanitize_graphviz_id("schema.table") == "schema_table"
        assert _sanitize_graphviz_id("a.b.c") == "a_b_c"

    def test_sanitize_mixed(self):
        """Mixed colons and dots should both be sanitized."""
        assert _sanitize_graphviz_id("cte:schema.table") == "cte__schema_table"

    def test_sanitize_no_special_chars(self):
        """Strings without special chars should be unchanged."""
        assert _sanitize_graphviz_id("simple_name") == "simple_name"


class TestVisualizeQueryUnits:
    """Tests for visualize_query_units function."""

    @pytest.fixture
    def simple_parser(self):
        """Create a simple parser for testing."""
        parser = RecursiveQueryParser("SELECT id, name FROM source")
        parser.parse()
        return parser

    @pytest.fixture
    def cte_parser(self):
        """Create a parser with CTEs."""
        parser = RecursiveQueryParser("""
            WITH cte_data AS (
                SELECT id, name FROM source
            )
            SELECT * FROM cte_data
        """)
        parser.parse()
        return parser

    def test_returns_digraph(self, simple_parser):
        """Should return a graphviz.Digraph object."""
        dot = visualize_query_units(simple_parser.unit_graph)
        assert isinstance(dot, graphviz.Digraph)

    def test_contains_main_query_node(self, simple_parser):
        """Should contain the main query node."""
        dot = visualize_query_units(simple_parser.unit_graph)
        source = dot.source
        assert "main" in source

    def test_contains_base_table_node(self, simple_parser):
        """Should contain base table nodes."""
        dot = visualize_query_units(simple_parser.unit_graph)
        source = dot.source
        assert "source" in source

    def test_cte_parser_contains_cte_node(self, cte_parser):
        """Should contain CTE nodes for queries with CTEs."""
        dot = visualize_query_units(cte_parser.unit_graph)
        source = dot.source
        assert "cte_data" in source or "CTE" in source


class TestVisualizeQueryStructureFromLineage:
    """Tests for visualize_query_structure_from_lineage function."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline for testing."""
        queries = [
            ("staging", "CREATE TABLE staging AS SELECT id, name FROM source"),
        ]
        return Pipeline(queries, dialect="bigquery")

    @pytest.fixture
    def parser(self):
        """Create a parser for testing."""
        parser = RecursiveQueryParser("SELECT id, name FROM source")
        parser.parse()
        return parser

    def test_returns_digraph(self, pipeline, parser):
        """Should return a graphviz.Digraph object."""
        query_lineage = pipeline.query_graphs["staging"]
        dot = visualize_query_structure_from_lineage(query_lineage, parser.unit_graph)
        assert isinstance(dot, graphviz.Digraph)

    def test_contains_table_nodes(self, pipeline, parser):
        """Should contain table nodes."""
        query_lineage = pipeline.query_graphs["staging"]
        dot = visualize_query_structure_from_lineage(query_lineage, parser.unit_graph)
        source = dot.source
        # Should contain either the table name or "columns" indicator
        assert "source" in source or "Main Query" in source


class TestVisualizeColumnLineage:
    """Tests for visualize_column_lineage function."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline for testing."""
        queries = [
            (
                "staging",
                "CREATE TABLE staging AS SELECT id, name, UPPER(name) as upper_name FROM source",
            ),
        ]
        return Pipeline(queries, dialect="bigquery")

    def test_returns_digraph(self, pipeline):
        """Should return a graphviz.Digraph object."""
        query_lineage = pipeline.query_graphs["staging"]
        dot = visualize_column_lineage(query_lineage)
        assert isinstance(dot, graphviz.Digraph)

    def test_contains_column_nodes(self, pipeline):
        """Should contain column nodes."""
        query_lineage = pipeline.query_graphs["staging"]
        dot = visualize_column_lineage(query_lineage)
        source = dot.source
        # Should contain column names
        assert "id" in source
        assert "name" in source

    def test_respects_max_nodes(self, pipeline):
        """Should respect max_nodes parameter."""
        query_lineage = pipeline.query_graphs["staging"]
        dot = visualize_column_lineage(query_lineage, max_nodes=2)
        assert isinstance(dot, graphviz.Digraph)

    def test_simplified_graph(self, pipeline):
        """Should work with simplified graph."""
        query_lineage = pipeline.query_graphs["staging"]
        simplified = query_lineage.to_simplified()
        dot = visualize_column_lineage(simplified)
        assert isinstance(dot, graphviz.Digraph)


class TestVisualizeColumnLineageSimple:
    """Tests for visualize_column_lineage_simple function."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline for testing."""
        queries = [
            ("staging", "CREATE TABLE staging AS SELECT id, name FROM source"),
        ]
        return Pipeline(queries, dialect="bigquery")

    def test_returns_digraph(self, pipeline):
        """Should return a graphviz.Digraph object."""
        query_lineage = pipeline.query_graphs["staging"]
        dot = visualize_column_lineage_simple(query_lineage)
        assert isinstance(dot, graphviz.Digraph)

    def test_respects_max_nodes(self, pipeline):
        """Should respect max_nodes parameter."""
        query_lineage = pipeline.query_graphs["staging"]
        dot = visualize_column_lineage_simple(query_lineage, max_nodes=2)
        assert isinstance(dot, graphviz.Digraph)


class TestVisualizeColumnPath:
    """Tests for visualize_column_path function."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline for testing."""
        queries = [
            ("staging", "CREATE TABLE staging AS SELECT id, name FROM source"),
            ("output", "CREATE TABLE output AS SELECT id FROM staging"),
        ]
        return Pipeline(queries, dialect="bigquery")

    def test_returns_digraph_single_query(self, pipeline):
        """Should return a graphviz.Digraph for single query graph."""
        query_lineage = pipeline.query_graphs["staging"]
        dot = visualize_column_path(query_lineage, "staging.id")
        assert isinstance(dot, graphviz.Digraph)

    def test_returns_digraph_pipeline(self, pipeline):
        """Should return a graphviz.Digraph for pipeline graph."""
        dot = visualize_column_path(pipeline.column_graph, "output.id")
        assert isinstance(dot, graphviz.Digraph)

    def test_column_not_found(self, pipeline):
        """Should handle column not found gracefully."""
        dot = visualize_column_path(pipeline.column_graph, "nonexistent.column")
        assert isinstance(dot, graphviz.Digraph)
        assert "not found" in dot.source

    def test_contains_target_column(self, pipeline):
        """Should contain the target column."""
        dot = visualize_column_path(pipeline.column_graph, "output.id")
        source = dot.source
        assert "id" in source


class TestVisualizeTableDependencies:
    """Tests for visualize_table_dependencies function."""

    @pytest.fixture
    def pipeline(self):
        """Create a multi-query pipeline."""
        queries = [
            ("staging", "CREATE TABLE staging AS SELECT id FROM source"),
            ("output", "CREATE TABLE output AS SELECT id FROM staging"),
        ]
        return Pipeline(queries, dialect="bigquery")

    def test_returns_digraph(self, pipeline):
        """Should return a graphviz.Digraph object."""
        dot = visualize_table_dependencies(pipeline.table_graph)
        assert isinstance(dot, graphviz.Digraph)

    def test_contains_table_nodes(self, pipeline):
        """Should contain table nodes."""
        dot = visualize_table_dependencies(pipeline.table_graph)
        source = dot.source
        assert "source" in source
        assert "staging" in source
        assert "output" in source

    def test_contains_source_label(self, pipeline):
        """Should label source tables correctly."""
        dot = visualize_table_dependencies(pipeline.table_graph)
        source = dot.source
        assert "source" in source

    def test_contains_final_label(self, pipeline):
        """Should label final tables correctly."""
        dot = visualize_table_dependencies(pipeline.table_graph)
        source = dot.source
        assert "final" in source


class TestVisualizeTableDependenciesWithLevels:
    """Tests for visualize_table_dependencies_with_levels function."""

    @pytest.fixture
    def pipeline(self):
        """Create a multi-query pipeline."""
        queries = [
            ("staging", "CREATE TABLE staging AS SELECT id FROM source"),
            ("output", "CREATE TABLE output AS SELECT id FROM staging"),
        ]
        return Pipeline(queries, dialect="bigquery")

    def test_returns_digraph(self, pipeline):
        """Should return a graphviz.Digraph object."""
        dot = visualize_table_dependencies_with_levels(pipeline.table_graph, pipeline)
        assert isinstance(dot, graphviz.Digraph)

    def test_contains_level_info(self, pipeline):
        """Should contain execution level information."""
        dot = visualize_table_dependencies_with_levels(pipeline.table_graph, pipeline)
        source = dot.source
        assert "Level" in source

    def test_contains_table_nodes(self, pipeline):
        """Should contain table nodes."""
        dot = visualize_table_dependencies_with_levels(pipeline.table_graph, pipeline)
        source = dot.source
        assert "staging" in source
        assert "output" in source


class TestVisualizePipelineLineage:
    """Tests for visualize_pipeline_lineage function."""

    @pytest.fixture
    def pipeline(self):
        """Create a multi-query pipeline."""
        queries = [
            ("staging", "CREATE TABLE staging AS SELECT id, name FROM source"),
            ("output", "CREATE TABLE output AS SELECT id FROM staging"),
        ]
        return Pipeline(queries, dialect="bigquery")

    def test_returns_digraph(self, pipeline):
        """Should return a graphviz.Digraph object."""
        dot = visualize_pipeline_lineage(pipeline.column_graph)
        assert isinstance(dot, graphviz.Digraph)

    def test_returns_tuple_with_debug_info(self, pipeline):
        """Should return tuple with debug info when requested."""
        result = visualize_pipeline_lineage(pipeline.column_graph, return_debug_info=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        dot, debug_info = result
        assert isinstance(dot, graphviz.Digraph)
        assert isinstance(debug_info, dict)
        assert "columns_displayed" in debug_info
        assert "edges_displayed" in debug_info

    def test_contains_column_nodes(self, pipeline):
        """Should contain column nodes."""
        dot = visualize_pipeline_lineage(pipeline.column_graph)
        source = dot.source
        assert "id" in source

    def test_respects_max_columns(self, pipeline):
        """Should respect max_columns parameter."""
        dot = visualize_pipeline_lineage(pipeline.column_graph, max_columns=2)
        assert isinstance(dot, graphviz.Digraph)

    def test_edges_have_styling(self, pipeline):
        """Edges should have proper styling."""
        dot = visualize_pipeline_lineage(pipeline.column_graph)
        source = dot.source
        # Should have edge styling (either cross_query orange or regular gray)
        assert "#E65100" in source or "#666666" in source


class TestVisualizeLineagePath:
    """Tests for visualize_lineage_path function."""

    @pytest.fixture
    def pipeline(self):
        """Create a multi-query pipeline."""
        queries = [
            ("staging", "CREATE TABLE staging AS SELECT id, name FROM source"),
            ("output", "CREATE TABLE output AS SELECT id FROM staging"),
        ]
        return Pipeline(queries, dialect="bigquery")

    def test_returns_digraph_backward(self, pipeline):
        """Should return a graphviz.Digraph for backward trace."""
        nodes, edges = pipeline.trace_column_backward_full("output", "id")
        dot = visualize_lineage_path(nodes, edges, is_backward=True)
        assert isinstance(dot, graphviz.Digraph)

    def test_returns_digraph_forward(self, pipeline):
        """Should return a graphviz.Digraph for forward trace."""
        nodes, edges = pipeline.trace_column_forward_full("source", "id")
        dot = visualize_lineage_path(nodes, edges, is_backward=False)
        assert isinstance(dot, graphviz.Digraph)

    def test_backward_label(self, pipeline):
        """Should have 'Backward' in comment for backward trace."""
        nodes, edges = pipeline.trace_column_backward_full("output", "id")
        dot = visualize_lineage_path(nodes, edges, is_backward=True)
        assert "Backward" in dot.comment

    def test_forward_label(self, pipeline):
        """Should have 'Forward' in comment for forward trace."""
        nodes, edges = pipeline.trace_column_forward_full("source", "id")
        dot = visualize_lineage_path(nodes, edges, is_backward=False)
        assert "Forward" in dot.comment

    def test_empty_path(self, pipeline):
        """Should handle empty path gracefully."""
        dot = visualize_lineage_path([], [], is_backward=True)
        assert isinstance(dot, graphviz.Digraph)

    def test_contains_traced_nodes(self, pipeline):
        """Should contain the traced nodes."""
        nodes, edges = pipeline.trace_column_backward_full("output", "id")
        dot = visualize_lineage_path(nodes, edges, is_backward=True)
        source = dot.source
        assert "id" in source


class TestVisualizationIntegration:
    """Integration tests for visualization functions."""

    def test_all_visualizations_work_together(self):
        """All visualization functions should work on the same pipeline."""
        queries = [
            (
                "staging",
                """
                WITH cte AS (SELECT id, name FROM source)
                CREATE TABLE staging AS SELECT * FROM cte
                """,
            ),
            ("output", "CREATE TABLE output AS SELECT id FROM staging"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # All these should work without errors
        dot1 = visualize_table_dependencies(pipeline.table_graph)
        dot2 = visualize_table_dependencies_with_levels(pipeline.table_graph, pipeline)
        dot3 = visualize_pipeline_lineage(pipeline.column_graph)

        query_lineage = pipeline.query_graphs["staging"]
        dot5 = visualize_column_lineage(query_lineage)

        # Test visualize_query_units with a parser
        parser = RecursiveQueryParser("SELECT id FROM source")
        parser.parse()
        dot4 = visualize_query_units(parser.unit_graph)

        nodes, edges = pipeline.trace_column_backward_full("output", "id")
        dot6 = visualize_lineage_path(nodes, edges)

        dot7 = visualize_column_path(pipeline.column_graph, "output.id")

        # All should be valid Digraph objects
        for dot in [dot1, dot2, dot3, dot4, dot5, dot6, dot7]:
            assert isinstance(dot, graphviz.Digraph)
            assert len(dot.source) > 0
