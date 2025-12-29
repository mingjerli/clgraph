"""
Tests for export functionality.

Tests JSON and CSV exporters.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clgraph.export import CSVExporter, JSONExporter
from clgraph.models import DescriptionSource
from clgraph.pipeline import Pipeline


def create_test_graph():
    """Create a simple test pipeline for export tests"""
    # Create pipeline with SQL that produces similar structure
    queries = [
        ("q1", "CREATE TABLE staging.orders AS SELECT order_id, customer_email FROM raw.orders"),
    ]
    pipeline = Pipeline(queries, dialect="bigquery")

    # Set metadata on columns to match test expectations
    # Use get_column for lookup since keys now include query_id prefix
    col = pipeline.get_column("raw.orders", "order_id")
    if col:
        col.description = "Order ID"
        col.description_source = DescriptionSource.SOURCE
        col.owner = "data-team"
        col.pii = False

    col = pipeline.get_column("raw.orders", "customer_email")
    if col:
        col.description = "Customer email address"
        col.description_source = DescriptionSource.SOURCE
        col.owner = "data-team"
        col.pii = True
        col.tags = {"contact", "sensitive"}

    col = pipeline.get_column("staging.orders", "order_id")
    if col:
        col.owner = "data-team"
        col.pii = False

    return pipeline


def test_json_export():
    """Test JSON exporter basic functionality"""
    graph = create_test_graph()

    data = JSONExporter.export(graph, include_metadata=True)

    # Check structure
    assert "columns" in data
    assert "edges" in data
    assert "tables" in data

    # Check columns (4 total: 2 source + 2 output)
    assert len(data["columns"]) == 4
    # Check by table_name + column_name since full_name now includes query_id prefix
    col_table_names = {(c["table_name"], c["column_name"]) for c in data["columns"]}
    assert ("raw.orders", "order_id") in col_table_names
    assert ("raw.orders", "customer_email") in col_table_names
    assert ("staging.orders", "order_id") in col_table_names
    assert ("staging.orders", "customer_email") in col_table_names

    # Check metadata on source column
    email_cols = [c for c in data["columns"] if c["column_name"] == "customer_email"]
    # Find the one with metadata set
    email_col = next((c for c in email_cols if c.get("pii")), email_cols[0])
    assert email_col["pii"] is True
    assert email_col["owner"] == "data-team"
    assert "contact" in email_col["tags"]
    assert email_col["description"] == "Customer email address"

    # Check edges (2 edges: one for each column)
    assert len(data["edges"]) == 2
    # Check edge endpoints by parsing table.column from full_name (which has query_id prefix)
    for edge in data["edges"]:
        # Format is "query_id:table.column", extract table.column part
        from_table_col = (
            edge["from_column"].split(":", 1)[-1]
            if ":" in edge["from_column"]
            else edge["from_column"]
        )
        to_table_col = (
            edge["to_column"].split(":", 1)[-1] if ":" in edge["to_column"] else edge["to_column"]
        )
        assert from_table_col in ["raw.orders.order_id", "raw.orders.customer_email"]
        assert to_table_col in ["staging.orders.order_id", "staging.orders.customer_email"]

    # Check tables
    assert len(data["tables"]) == 2


def test_json_export_without_metadata():
    """Test JSON export without metadata"""
    graph = create_test_graph()

    data = JSONExporter.export(graph, include_metadata=False)

    # Columns should not have metadata fields
    col = data["columns"][0]
    assert "description" not in col
    assert "owner" not in col
    assert "pii" not in col
    assert "tags" not in col


def test_json_export_to_file():
    """Test JSON export to file"""
    graph = create_test_graph()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.json"
        JSONExporter.export_to_file(graph, str(file_path))

        assert file_path.exists()

        # Read and verify
        with open(file_path) as f:
            data = json.load(f)

        assert "columns" in data
        assert len(data["columns"]) == 4


def test_csv_columns_export():
    """Test CSV columns export"""
    graph = create_test_graph()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "columns.csv"
        CSVExporter.export_columns_to_file(graph, str(file_path))

        assert file_path.exists()

        # Read and verify
        with open(file_path) as f:
            lines = f.readlines()

        # Header + 4 rows
        assert len(lines) == 5

        # Check header
        assert "full_name" in lines[0]
        assert "pii" in lines[0]
        assert "owner" in lines[0]

        # Check PII column
        email_line = next(line for line in lines if "customer_email" in line)
        assert "Yes" in email_line  # PII=True
        assert "contact" in email_line  # tags


def test_csv_tables_export():
    """Test CSV tables export"""
    graph = create_test_graph()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "tables.csv"
        CSVExporter.export_tables_to_file(graph, str(file_path))

        assert file_path.exists()

        # Read and verify
        with open(file_path) as f:
            lines = f.readlines()

        # Header + 2 tables
        assert len(lines) == 3

        # Check header
        assert "table_name" in lines[0]
        assert "is_source" in lines[0]


def test_json_export_includes_queries():
    """Test JSON export includes queries for round-trip serialization"""
    graph = create_test_graph()

    data = JSONExporter.export(graph, include_queries=True)

    # Check queries are included
    assert "queries" in data
    assert "dialect" in data
    assert data["dialect"] == "bigquery"

    # Check query structure
    assert len(data["queries"]) >= 1
    query = data["queries"][0]
    assert "query_id" in query
    assert "sql" in query
    assert query["query_id"] == "q1"
    assert "CREATE TABLE staging.orders" in query["sql"]


def test_json_export_without_queries():
    """Test JSON export without queries"""
    graph = create_test_graph()

    data = JSONExporter.export(graph, include_queries=False)

    # Queries should not be present
    assert "queries" not in data
    assert "dialect" not in data


def test_json_round_trip_basic():
    """Test basic JSON round-trip: export → import → verify"""
    # Create original pipeline
    queries = [
        ("staging", "CREATE TABLE staging.orders AS SELECT id, amount FROM raw.orders"),
        (
            "analytics",
            "CREATE TABLE analytics.totals AS SELECT SUM(amount) as total FROM staging.orders",
        ),
    ]
    original = Pipeline.from_tuples(queries, dialect="bigquery")

    # Export to JSON
    data = JSONExporter.export(original, include_queries=True)

    # Import from JSON
    restored = Pipeline.from_json(data)

    # Verify structure matches
    assert restored.dialect == original.dialect
    assert len(restored.columns) == len(original.columns)
    assert len(restored.edges) == len(original.edges)
    assert len(restored.table_graph.tables) == len(original.table_graph.tables)

    # Verify column names match
    original_cols = {c.full_name for c in original.columns.values()}
    restored_cols = {c.full_name for c in restored.columns.values()}
    assert original_cols == restored_cols


def test_json_round_trip_with_metadata():
    """Test JSON round-trip preserves metadata"""
    # Create pipeline with metadata
    queries = [
        ("q1", "CREATE TABLE staging.users AS SELECT id, email FROM raw.users"),
    ]
    original = Pipeline.from_tuples(queries, dialect="bigquery")

    # Set metadata on a column
    email_col = original.get_column("raw.users", "email")
    if email_col:
        email_col.description = "User email address"
        email_col.description_source = DescriptionSource.GENERATED
        email_col.owner = "privacy-team"
        email_col.pii = True
        email_col.tags = {"contact", "sensitive"}
        email_col.custom_metadata = {"classification": "restricted"}

    # Export and import
    data = JSONExporter.export(original, include_queries=True, include_metadata=True)
    restored = Pipeline.from_json(data, apply_metadata=True)

    # Verify metadata preserved
    restored_email = restored.get_column("raw.users", "email")
    assert restored_email is not None
    assert restored_email.description == "User email address"
    assert restored_email.description_source == DescriptionSource.GENERATED
    assert restored_email.owner == "privacy-team"
    assert restored_email.pii is True
    assert restored_email.tags == {"contact", "sensitive"}
    assert restored_email.custom_metadata == {"classification": "restricted"}


def test_json_round_trip_skip_metadata():
    """Test JSON round-trip can skip applying metadata"""
    # Create pipeline with metadata
    queries = [
        ("q1", "CREATE TABLE staging.users AS SELECT id, email FROM raw.users"),
    ]
    original = Pipeline.from_tuples(queries, dialect="bigquery")

    # Set metadata
    email_col = original.get_column("raw.users", "email")
    if email_col:
        email_col.description = "Original description"
        email_col.pii = True

    # Export with metadata, import without
    data = JSONExporter.export(original, include_queries=True, include_metadata=True)
    restored = Pipeline.from_json(data, apply_metadata=False)

    # Metadata should not be applied
    restored_email = restored.get_column("raw.users", "email")
    assert restored_email is not None
    assert restored_email.description is None
    assert restored_email.pii is False


def test_json_round_trip_file():
    """Test JSON round-trip through file"""
    queries = [
        ("q1", "CREATE TABLE staging.orders AS SELECT id, amount FROM raw.orders"),
    ]
    original = Pipeline.from_tuples(queries, dialect="bigquery")

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "pipeline.json"

        # Export to file
        JSONExporter.export_to_file(original, str(file_path), include_queries=True)

        # Import from file
        restored = Pipeline.from_json_file(str(file_path))

        # Verify
        assert restored.dialect == original.dialect
        assert len(restored.columns) == len(original.columns)
        assert len(restored.edges) == len(original.edges)


def test_json_round_trip_with_template_context():
    """Test JSON round-trip preserves template context"""
    queries = [
        ("q1", "CREATE TABLE {{ env }}.orders AS SELECT id FROM raw.orders"),
    ]
    template_context = {"env": "staging"}
    original = Pipeline.from_tuples(queries, dialect="bigquery", template_context=template_context)

    # Export and import
    data = JSONExporter.export(original, include_queries=True)
    restored = Pipeline.from_json(data)

    # Verify template context preserved
    assert restored.template_context == template_context


def test_json_from_json_missing_queries():
    """Test from_json raises error when queries missing"""
    import pytest

    data = {"columns": [], "edges": [], "tables": [], "dialect": "bigquery"}

    with pytest.raises(ValueError, match="missing 'queries'"):
        Pipeline.from_json(data)


def test_json_from_json_missing_dialect():
    """Test from_json raises error when dialect missing"""
    import pytest

    data = {"columns": [], "edges": [], "tables": [], "queries": []}

    with pytest.raises(ValueError, match="missing 'dialect'"):
        Pipeline.from_json(data)


def test_json_from_json_file_not_found():
    """Test from_json_file raises error when file not found"""
    import pytest

    with pytest.raises(FileNotFoundError, match="not found"):
        Pipeline.from_json_file("/nonexistent/path/pipeline.json")


if __name__ == "__main__":
    # Run tests
    test_json_export()
    test_json_export_without_metadata()
    test_json_export_to_file()
    test_json_export_includes_queries()
    test_json_export_without_queries()
    test_json_round_trip_basic()
    test_json_round_trip_with_metadata()
    test_json_round_trip_skip_metadata()
    test_json_round_trip_file()
    test_json_round_trip_with_template_context()
    test_json_from_json_missing_queries()
    test_json_from_json_missing_dialect()
    test_json_from_json_file_not_found()
    test_csv_columns_export()
    test_csv_tables_export()

    print("✅ All export tests passed!")
