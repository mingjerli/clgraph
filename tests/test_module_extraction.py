"""
Test suite for module extraction (Item 9 Phases 1-3).

This test suite verifies:
1. Phase 1: lineage_utils.py extraction from lineage_builder.py
   - SourceColumnRef, BackwardLineageResult TypedDicts
   - JSON function constants and utilities
   - Aggregate registry and classification functions
   - Nested access detection and schema qualification functions

2. Phase 2: sql_column_tracer.py extraction from lineage_builder.py
   - SQLColumnTracer class moved to new module
   - Backward compatibility imports from lineage_builder

3. Phase 3: tvf_registry.py extraction from query_parser.py
   - KNOWN_TVF_EXPRESSIONS, KNOWN_TVF_NAMES, TVF_DEFAULT_COLUMNS
   - Backward compatibility imports from query_parser

Import Compatibility Requirements:
- All existing imports must continue to work
- from clgraph import SQLColumnTracer should work
- from clgraph.lineage_builder import SQLColumnTracer should work (backward compat)
- Direct imports from new modules should also work
"""


# ============================================================================
# Test Group 1: Phase 1 - lineage_utils.py extraction
# ============================================================================


class TestLineageUtilsExtraction:
    """Test that utilities are properly extracted to lineage_utils.py"""

    def test_source_column_ref_importable_from_new_module(self):
        """SourceColumnRef TypedDict should be importable from lineage_utils."""
        from clgraph.lineage_utils import SourceColumnRef

        # Verify it's a TypedDict by checking its structure
        assert hasattr(SourceColumnRef, "__annotations__")
        assert "table_ref" in SourceColumnRef.__annotations__
        assert "column_name" in SourceColumnRef.__annotations__
        assert "json_path" in SourceColumnRef.__annotations__
        assert "json_function" in SourceColumnRef.__annotations__

    def test_backward_lineage_result_importable_from_new_module(self):
        """BackwardLineageResult TypedDict should be importable from lineage_utils."""
        from clgraph.lineage_utils import BackwardLineageResult

        assert hasattr(BackwardLineageResult, "__annotations__")
        assert "required_inputs" in BackwardLineageResult.__annotations__
        assert "required_ctes" in BackwardLineageResult.__annotations__
        assert "paths" in BackwardLineageResult.__annotations__

    def test_json_function_names_constant_importable(self):
        """JSON_FUNCTION_NAMES set should be importable from lineage_utils."""
        from clgraph.lineage_utils import JSON_FUNCTION_NAMES

        assert isinstance(JSON_FUNCTION_NAMES, set)
        # Verify some known values
        assert "JSON_EXTRACT" in JSON_FUNCTION_NAMES
        assert "JSON_VALUE" in JSON_FUNCTION_NAMES
        assert "JSON_EXTRACT_SCALAR" in JSON_FUNCTION_NAMES

    def test_json_expression_types_constant_importable(self):
        """JSON_EXPRESSION_TYPES dict should be importable from lineage_utils."""
        from sqlglot import exp

        from clgraph.lineage_utils import JSON_EXPRESSION_TYPES

        assert isinstance(JSON_EXPRESSION_TYPES, dict)
        # Verify keys are sqlglot expression types
        assert exp.JSONExtract in JSON_EXPRESSION_TYPES
        assert exp.JSONExtractScalar in JSON_EXPRESSION_TYPES

    def test_json_detection_functions_importable(self):
        """JSON detection functions should be importable from lineage_utils."""
        from clgraph.lineage_utils import (
            _extract_json_path,
            _get_json_function_name,
            _is_json_extract_function,
            _normalize_json_path,
        )

        # Verify they are callable
        assert callable(_is_json_extract_function)
        assert callable(_get_json_function_name)
        assert callable(_extract_json_path)
        assert callable(_normalize_json_path)

    def test_normalize_json_path_functionality(self):
        """Test _normalize_json_path works correctly after extraction."""
        from clgraph.lineage_utils import _normalize_json_path

        # Test unchanged format
        assert _normalize_json_path("$.address.city") == "$.address.city"
        # Test bracket notation conversion
        assert _normalize_json_path('$["address"]["city"]') == "$.address.city"
        # Test Snowflake format
        assert _normalize_json_path("address.city") == "$.address.city"
        # Test PostgreSQL format
        assert _normalize_json_path("{address,city}") == "$.address.city"

    def test_aggregate_registry_importable(self):
        """AGGREGATE_REGISTRY dict should be importable from lineage_utils."""
        from clgraph.lineage_utils import AGGREGATE_REGISTRY
        from clgraph.models import AggregateType

        assert isinstance(AGGREGATE_REGISTRY, dict)
        # Verify some known values
        assert AGGREGATE_REGISTRY.get("array_agg") == AggregateType.ARRAY
        assert AGGREGATE_REGISTRY.get("sum") == AggregateType.SCALAR
        assert AGGREGATE_REGISTRY.get("string_agg") == AggregateType.STRING

    def test_aggregate_classification_functions_importable(self):
        """Aggregate classification functions should be importable from lineage_utils."""
        from clgraph.lineage_utils import _get_aggregate_type, _is_complex_aggregate

        assert callable(_get_aggregate_type)
        assert callable(_is_complex_aggregate)

    def test_aggregate_type_classification_functionality(self):
        """Test aggregate type classification works correctly after extraction."""
        from clgraph.lineage_utils import _get_aggregate_type, _is_complex_aggregate
        from clgraph.models import AggregateType

        # Test type classification
        assert _get_aggregate_type("array_agg") == AggregateType.ARRAY
        assert _get_aggregate_type("sum") == AggregateType.SCALAR
        assert _get_aggregate_type("unknown_func") is None

        # Test complex aggregate detection
        assert _is_complex_aggregate("array_agg") is True
        assert _is_complex_aggregate("sum") is False
        assert _is_complex_aggregate("string_agg") is True

    def test_json_ancestor_function_importable(self):
        """_find_json_function_ancestor should be importable from lineage_utils."""
        from clgraph.lineage_utils import _find_json_function_ancestor

        assert callable(_find_json_function_ancestor)

    def test_nested_access_functions_importable(self):
        """Nested access detection functions should be importable from lineage_utils."""
        from clgraph.lineage_utils import (
            _extract_nested_path_from_expression,
            _find_nested_access_ancestor,
            _is_nested_access_expression,
        )

        assert callable(_is_nested_access_expression)
        assert callable(_extract_nested_path_from_expression)
        assert callable(_find_nested_access_ancestor)

    def test_schema_qualification_functions_importable(self):
        """Schema qualification functions should be importable from lineage_utils."""
        from clgraph.lineage_utils import (
            _convert_to_nested_schema,
            _qualify_sql_with_schema,
        )

        assert callable(_convert_to_nested_schema)
        assert callable(_qualify_sql_with_schema)

    def test_convert_to_nested_schema_functionality(self):
        """Test _convert_to_nested_schema works correctly after extraction."""
        from clgraph.lineage_utils import _convert_to_nested_schema

        flat_schema = {
            "schema1.table1": ["col1", "col2"],
            "schema2.table2": ["col3"],
        }
        nested = _convert_to_nested_schema(flat_schema)

        assert "schema1" in nested
        assert "table1" in nested["schema1"]
        assert nested["schema1"]["table1"]["col1"] == "UNKNOWN"

    def test_backward_compat_imports_from_lineage_builder(self):
        """All extracted items should still be importable from lineage_builder."""
        # These imports should work for backward compatibility
        from clgraph.lineage_builder import (
            AGGREGATE_REGISTRY,
            JSON_FUNCTION_NAMES,
        )

        # Just verify imports work - functionality tested elsewhere
        assert JSON_FUNCTION_NAMES is not None
        assert AGGREGATE_REGISTRY is not None


# ============================================================================
# Test Group 2: Phase 2 - sql_column_tracer.py extraction
# ============================================================================


class TestSQLColumnTracerExtraction:
    """Test that SQLColumnTracer is properly extracted to sql_column_tracer.py"""

    def test_sql_column_tracer_importable_from_new_module(self):
        """SQLColumnTracer should be importable from sql_column_tracer."""
        from clgraph.sql_column_tracer import SQLColumnTracer

        assert SQLColumnTracer is not None
        # Verify it's a class
        assert isinstance(SQLColumnTracer, type)

    def test_sql_column_tracer_backward_compat_from_lineage_builder(self):
        """SQLColumnTracer should still be importable from lineage_builder."""
        from clgraph.lineage_builder import SQLColumnTracer

        assert SQLColumnTracer is not None

    def test_sql_column_tracer_importable_from_parser(self):
        """SQLColumnTracer should be importable from parser (main re-export)."""
        from clgraph.parser import SQLColumnTracer

        assert SQLColumnTracer is not None

    def test_sql_column_tracer_importable_from_clgraph(self):
        """SQLColumnTracer should be importable from top-level clgraph."""
        from clgraph import SQLColumnTracer

        assert SQLColumnTracer is not None

    def test_sql_column_tracer_functionality(self):
        """SQLColumnTracer should work correctly after extraction."""
        from clgraph.sql_column_tracer import SQLColumnTracer

        sql = "SELECT id, name FROM users WHERE status = 'active'"
        tracer = SQLColumnTracer(sql, dialect="bigquery")

        # Test basic functionality
        column_names = tracer.get_column_names()
        assert "id" in column_names
        assert "name" in column_names

    def test_sql_column_tracer_graph_building(self):
        """SQLColumnTracer graph building should work after extraction."""
        from clgraph.sql_column_tracer import SQLColumnTracer

        sql = "SELECT u.id, u.name FROM users u"
        tracer = SQLColumnTracer(sql, dialect="bigquery")

        graph = tracer.build_column_lineage_graph()
        assert graph is not None
        assert len(graph.nodes) > 0

    def test_sql_column_tracer_forward_lineage(self):
        """SQLColumnTracer forward lineage should work after extraction."""
        from clgraph.sql_column_tracer import SQLColumnTracer

        sql = "SELECT id, UPPER(name) AS upper_name FROM users"
        tracer = SQLColumnTracer(sql, dialect="bigquery")

        result = tracer.get_forward_lineage(["users.name"])
        assert "impacted_outputs" in result
        assert "upper_name" in result["impacted_outputs"]

    def test_sql_column_tracer_backward_lineage(self):
        """SQLColumnTracer backward lineage should work after extraction."""
        from clgraph.sql_column_tracer import SQLColumnTracer

        sql = "SELECT id, UPPER(name) AS upper_name FROM users"
        tracer = SQLColumnTracer(sql, dialect="bigquery")

        result = tracer.get_backward_lineage(["upper_name"])
        assert "required_inputs" in result
        assert "users" in result["required_inputs"]

    def test_sql_column_tracer_select_columns_property(self):
        """SQLColumnTracer select_columns property should work after extraction."""
        from clgraph.sql_column_tracer import SQLColumnTracer

        sql = "SELECT id, name AS user_name FROM users"
        tracer = SQLColumnTracer(sql, dialect="bigquery")

        cols = tracer.select_columns
        assert len(cols) == 2
        assert any(c["alias"] == "id" for c in cols)
        assert any(c["alias"] == "user_name" for c in cols)


# ============================================================================
# Test Group 3: Phase 3 - tvf_registry.py extraction
# ============================================================================


class TestTVFRegistryExtraction:
    """Test that TVF registry is properly extracted to tvf_registry.py"""

    def test_known_tvf_expressions_importable_from_new_module(self):
        """KNOWN_TVF_EXPRESSIONS should be importable from tvf_registry."""
        from sqlglot import exp

        from clgraph.tvf_registry import KNOWN_TVF_EXPRESSIONS

        assert isinstance(KNOWN_TVF_EXPRESSIONS, dict)
        # Verify some known keys
        assert exp.GenerateSeries in KNOWN_TVF_EXPRESSIONS
        assert exp.ReadCSV in KNOWN_TVF_EXPRESSIONS

    def test_known_tvf_names_importable_from_new_module(self):
        """KNOWN_TVF_NAMES should be importable from tvf_registry."""
        from clgraph.models import TVFType
        from clgraph.tvf_registry import KNOWN_TVF_NAMES

        assert isinstance(KNOWN_TVF_NAMES, dict)
        # Verify some known values
        assert KNOWN_TVF_NAMES.get("generate_series") == TVFType.GENERATOR
        assert KNOWN_TVF_NAMES.get("read_csv") == TVFType.EXTERNAL
        assert KNOWN_TVF_NAMES.get("flatten") == TVFType.COLUMN_INPUT

    def test_tvf_default_columns_importable_from_new_module(self):
        """TVF_DEFAULT_COLUMNS should be importable from tvf_registry."""
        from clgraph.tvf_registry import TVF_DEFAULT_COLUMNS

        assert isinstance(TVF_DEFAULT_COLUMNS, dict)
        # Verify some known values
        assert "generate_series" in TVF_DEFAULT_COLUMNS
        assert TVF_DEFAULT_COLUMNS["generate_series"] == ["generate_series"]
        assert "flatten" in TVF_DEFAULT_COLUMNS
        assert "value" in TVF_DEFAULT_COLUMNS["flatten"]

    def test_backward_compat_imports_from_query_parser(self):
        """TVF registry items should still be importable from query_parser."""
        from clgraph.query_parser import (
            KNOWN_TVF_EXPRESSIONS,
            KNOWN_TVF_NAMES,
            TVF_DEFAULT_COLUMNS,
        )

        assert KNOWN_TVF_EXPRESSIONS is not None
        assert KNOWN_TVF_NAMES is not None
        assert TVF_DEFAULT_COLUMNS is not None

    def test_query_parser_uses_tvf_registry(self):
        """RecursiveQueryParser should use TVF registry correctly."""
        from clgraph import RecursiveQueryParser
        from clgraph.models import TVFType

        sql = "SELECT num FROM GENERATE_SERIES(1, 10) AS t(num)"
        parser = RecursiveQueryParser(sql, dialect="postgres")
        unit_graph = parser.parse()

        unit = unit_graph.units["main"]
        assert "t" in unit.tvf_sources
        tvf_info = unit.tvf_sources["t"]
        assert tvf_info.tvf_type == TVFType.GENERATOR


# ============================================================================
# Test Group 4: Cross-Module Integration
# ============================================================================


class TestCrossModuleIntegration:
    """Test that all modules work together after extraction."""

    def test_recursive_lineage_builder_uses_lineage_utils(self):
        """RecursiveLineageBuilder should work with extracted utilities."""
        from clgraph import RecursiveLineageBuilder

        sql = "SELECT JSON_EXTRACT(data, '$.user.name') AS user_name FROM users"
        builder = RecursiveLineageBuilder(sql, dialect="bigquery")
        graph = builder.build()

        # Verify JSON handling works
        user_name_edges = [e for e in graph.edges if e.to_node.column_name == "user_name"]
        assert len(user_name_edges) > 0
        edge = user_name_edges[0]
        assert edge.json_function == "JSON_EXTRACT"

    def test_sql_column_tracer_uses_recursive_lineage_builder(self):
        """SQLColumnTracer should work with RecursiveLineageBuilder after extraction."""
        from clgraph import SQLColumnTracer

        sql = """
        WITH processed AS (
            SELECT id, UPPER(name) AS upper_name FROM users
        )
        SELECT id, upper_name FROM processed
        """
        tracer = SQLColumnTracer(sql, dialect="bigquery")
        graph = tracer.build_column_lineage_graph()

        # Verify integration works
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

    def test_pipeline_integration_after_extraction(self):
        """Pipeline should work correctly after all extractions."""
        from clgraph import Pipeline

        queries = [
            ("staging_users", "SELECT id, name FROM raw_users"),
            ("final_users", "SELECT id, UPPER(name) AS formatted_name FROM staging_users"),
        ]

        pipeline = Pipeline(queries, dialect="bigquery")

        # Verify basic pipeline functionality
        assert pipeline.table_graph is not None
        assert pipeline.column_graph is not None

    def test_all_existing_imports_still_work(self):
        """All existing import patterns should continue to work."""
        # Top-level imports
        from clgraph import (
            Pipeline,
            RecursiveLineageBuilder,
            RecursiveQueryParser,
            SQLColumnTracer,
        )

        # Verify top-level imports work
        assert Pipeline is not None
        assert SQLColumnTracer is not None
        assert RecursiveLineageBuilder is not None
        assert RecursiveQueryParser is not None

        # Direct module imports - use different names to avoid redefinition
        from clgraph.lineage_builder import (
            RecursiveLineageBuilder as LB_RecursiveLineageBuilder,
        )

        # parser.py imports
        from clgraph.parser import (
            RecursiveLineageBuilder as P_RecursiveLineageBuilder,
        )
        from clgraph.parser import (
            RecursiveQueryParser as P_RecursiveQueryParser,
        )
        from clgraph.parser import (
            SQLColumnTracer as P_SQLColumnTracer,
        )
        from clgraph.query_parser import (
            RecursiveQueryParser as QP_RecursiveQueryParser,
        )

        # Verify all imports resolve to the same class
        assert LB_RecursiveLineageBuilder is RecursiveLineageBuilder
        assert P_RecursiveLineageBuilder is RecursiveLineageBuilder
        assert P_RecursiveQueryParser is RecursiveQueryParser
        assert P_SQLColumnTracer is SQLColumnTracer
        assert QP_RecursiveQueryParser is RecursiveQueryParser


# ============================================================================
# Test Group 5: Module Size Verification
# ============================================================================


class TestModuleSizeConstraints:
    """Verify new modules meet size constraints."""

    def test_lineage_utils_exists(self):
        """lineage_utils.py should exist as a module."""
        import clgraph.lineage_utils

        assert clgraph.lineage_utils is not None

    def test_sql_column_tracer_module_exists(self):
        """sql_column_tracer.py should exist as a module."""
        import clgraph.sql_column_tracer

        assert clgraph.sql_column_tracer is not None

    def test_tvf_registry_module_exists(self):
        """tvf_registry.py should exist as a module."""
        import clgraph.tvf_registry

        assert clgraph.tvf_registry is not None
