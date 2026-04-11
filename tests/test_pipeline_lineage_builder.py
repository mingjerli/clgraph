"""
Tests for PipelineLineageBuilder extracted module.

Covers:
- Import from new module (clgraph.pipeline_lineage_builder)
- Import from old module (clgraph.pipeline) - backward compat
- Import from clgraph package - backward compat
- Functional tests for PipelineLineageBuilder
"""


from clgraph.pipeline import Pipeline


class TestPipelineLineageBuilderImports:
    """Test that PipelineLineageBuilder can be imported from all expected locations."""

    def test_import_from_new_module(self):
        """Import from the new module clgraph.pipeline_lineage_builder."""
        from clgraph.pipeline_lineage_builder import PipelineLineageBuilder

        assert PipelineLineageBuilder is not None

    def test_import_from_old_module(self):
        """Import from old module clgraph.pipeline — backward compatibility."""
        from clgraph.pipeline import PipelineLineageBuilder

        assert PipelineLineageBuilder is not None

    def test_import_from_package(self):
        """Import from top-level clgraph package — backward compatibility."""
        from clgraph import PipelineLineageBuilder

        assert PipelineLineageBuilder is not None

    def test_all_imports_are_same_class(self):
        """All import paths should resolve to the same class."""
        from clgraph import PipelineLineageBuilder as PipelineLineageBuilderFromPackage
        from clgraph.pipeline import PipelineLineageBuilder as PipelineLineageBuilderFromPipeline
        from clgraph.pipeline_lineage_builder import (
            PipelineLineageBuilder as PipelineLineageBuilderFromModule,
        )

        assert (
            PipelineLineageBuilderFromModule
            is PipelineLineageBuilderFromPipeline
            is PipelineLineageBuilderFromPackage
        )

    def test_class_has_expected_methods(self):
        """PipelineLineageBuilder should have all expected public/private methods."""
        from clgraph.pipeline_lineage_builder import PipelineLineageBuilder

        expected_methods = [
            "build",
            "_expand_star_nodes_in_pipeline",
            "_collect_upstream_table_schemas",
            "_get_upstream_table_columns",
            "_add_query_columns",
            "_add_query_edges",
            "_add_cross_query_edges",
            "_infer_table_name",
            "_make_full_name",
            "_is_physical_table_column",
            "_extract_select_from_query",
        ]
        for method in expected_methods:
            assert hasattr(PipelineLineageBuilder, method), (
                f"PipelineLineageBuilder missing method: {method}"
            )


class TestPipelineLineageBuilderFunctional:
    """Functional tests for PipelineLineageBuilder."""

    def test_build_returns_pipeline(self):
        """build() should return a Pipeline instance."""

        queries = [
            ("q1", "CREATE TABLE staging AS SELECT a, b FROM raw"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # PipelineLineageBuilder is used internally by Pipeline.__init__
        # We verify the result is a pipeline with lineage built
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.columns) > 0

    def test_multi_query_lineage(self):
        """PipelineLineageBuilder correctly builds cross-query lineage."""
        queries = [
            ("q1", "CREATE TABLE staging AS SELECT customer_id, amount FROM raw_orders"),
            ("q2", "CREATE TABLE final AS SELECT customer_id, SUM(amount) AS total FROM staging GROUP BY customer_id"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Should have columns from both queries
        col_names = {col.full_name for col in pipeline.columns.values()}
        assert any("staging" in name for name in col_names)
        assert any("final" in name for name in col_names)

        # Should have customer_id in staging
        assert "staging.customer_id" in col_names or any(
            "customer_id" in name and "staging" in name for name in col_names
        )

    def test_pipeline_lineage_builder_can_be_instantiated_directly(self):
        """PipelineLineageBuilder can be instantiated and used directly."""
        from clgraph.multi_query import MultiQueryParser
        from clgraph.pipeline_lineage_builder import PipelineLineageBuilder
        from clgraph.table import TableDependencyGraph

        # Build a simple pipeline via the builder directly using TableDependencyGraph
        # (backward compatibility path - accepts TableDependencyGraph directly)
        parser = MultiQueryParser()
        # parse_queries takes a list of plain SQL strings
        table_graph = parser.parse_queries(["CREATE TABLE staging AS SELECT a, b FROM raw"])

        assert isinstance(table_graph, TableDependencyGraph)

        builder = PipelineLineageBuilder()
        result = builder.build(table_graph)

        assert isinstance(result, Pipeline)
        assert len(result.columns) > 0

    def test_star_expansion_across_queries(self):
        """PipelineLineageBuilder expands * nodes when upstream schema is known."""
        queries = [
            ("q1", "CREATE TABLE staging AS SELECT col_a, col_b, col_c FROM source"),
            ("q2", "CREATE TABLE final AS SELECT * FROM staging"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # With star expansion, final should have explicit columns
        final_cols = [
            col for col in pipeline.columns.values()
            if col.table_name == "final" and col.layer == "output" and not col.is_star
        ]
        assert len(final_cols) > 0, "Star expansion should produce explicit output columns"
        final_col_names = {col.column_name for col in final_cols}
        assert "col_a" in final_col_names
        assert "col_b" in final_col_names
        assert "col_c" in final_col_names

    def test_three_query_chain(self):
        """PipelineLineageBuilder handles 3-query chain with proper cross-query edges."""
        queries = [
            ("q1", "CREATE TABLE raw_clean AS SELECT id, val FROM raw_source"),
            ("q2", "CREATE TABLE aggregated AS SELECT id, SUM(val) AS total FROM raw_clean GROUP BY id"),
            ("q3", "CREATE TABLE report AS SELECT id, total FROM aggregated WHERE total > 0"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # All tables should have columns
        table_names = {col.table_name for col in pipeline.columns.values()}
        assert "aggregated" in table_names
        assert "report" in table_names

    def test_cte_query_lineage(self):
        """PipelineLineageBuilder handles CTEs within a pipeline query."""
        queries = [
            ("q1", "CREATE TABLE source AS SELECT id, amount FROM raw"),
            (
                "q2",
                """
                CREATE TABLE result AS
                WITH cte AS (
                    SELECT id, amount * 2 AS doubled FROM source
                )
                SELECT id, doubled FROM cte
                """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        result_cols = [
            col for col in pipeline.columns.values()
            if col.table_name == "result" and col.layer == "output"
        ]
        assert len(result_cols) > 0

    def test_pipeline_edges_connect_queries(self):
        """Edges should be created connecting output from q1 to input of q2."""
        queries = [
            ("q1", "CREATE TABLE staging AS SELECT user_id FROM users"),
            ("q2", "CREATE TABLE final AS SELECT user_id FROM staging"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Should have edges
        assert len(pipeline.edges) > 0

        # There should be an edge from staging.user_id to final.user_id
        edge_pairs = {(e.from_node.full_name, e.to_node.full_name) for e in pipeline.edges}
        assert any("staging.user_id" in pair[0] and "final.user_id" in pair[1] for pair in edge_pairs)
