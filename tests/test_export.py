"""
Tests for export functionality.

Tests JSON, CSV, and GraphViz exporters.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clpipe.column import PipelineColumnNode
from clpipe.export import CSVExporter, GraphVizExporter, JSONExporter
from clpipe.models import DescriptionSource
from clpipe.pipeline import Pipeline


def create_test_graph():
    """Create a simple test pipeline for export tests"""
    # Create pipeline with SQL that produces similar structure
    queries = [
        ("q1", "CREATE TABLE staging.orders AS SELECT order_id, customer_email FROM raw.orders"),
    ]
    pipeline = Pipeline(queries, dialect="bigquery")

    # Set metadata on columns to match test expectations
    if "raw.orders.order_id" in pipeline.columns:
        col = pipeline.columns["raw.orders.order_id"]
        col.description = "Order ID"
        col.description_source = DescriptionSource.SOURCE
        col.owner = "data-team"
        col.pii = False

    if "raw.orders.customer_email" in pipeline.columns:
        col = pipeline.columns["raw.orders.customer_email"]
        col.description = "Customer email address"
        col.description_source = DescriptionSource.SOURCE
        col.owner = "data-team"
        col.pii = True
        col.tags = {"contact", "sensitive"}

    if "staging.orders.order_id" in pipeline.columns:
        col = pipeline.columns["staging.orders.order_id"]
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
    col_names = {c["full_name"] for c in data["columns"]}
    assert "raw.orders.order_id" in col_names
    assert "raw.orders.customer_email" in col_names
    assert "staging.orders.order_id" in col_names
    assert "staging.orders.customer_email" in col_names

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
    edge_pairs = {(e["from_column"], e["to_column"]) for e in data["edges"]}
    assert ("raw.orders.order_id", "staging.orders.order_id") in edge_pairs
    assert ("raw.orders.customer_email", "staging.orders.customer_email") in edge_pairs

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


def test_graphviz_export():
    """Test GraphViz DOT export"""
    graph = create_test_graph()

    dot_content = GraphVizExporter.export(graph, layout="LR")

    # Check basic structure
    assert "digraph lineage {" in dot_content
    assert "rankdir=LR" in dot_content

    # Check nodes
    assert "raw_orders_order_id" in dot_content
    assert "staging_orders_order_id" in dot_content

    # Check edge
    assert "->" in dot_content


def test_graphviz_export_with_descriptions():
    """Test GraphViz export includes descriptions in labels"""
    graph = create_test_graph()

    dot_content = GraphVizExporter.export(graph)

    # Check descriptions in labels
    assert "Order ID" in dot_content
    assert "Customer email address" in dot_content


def test_graphviz_export_to_file():
    """Test GraphViz export to file"""
    graph = create_test_graph()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "lineage.dot"
        GraphVizExporter.export_to_file(graph, str(file_path), layout="TB")

        assert file_path.exists()

        # Read and verify
        with open(file_path) as f:
            content = f.read()

        assert "digraph lineage" in content
        assert "rankdir=TB" in content


def test_graphviz_max_columns():
    """Test GraphViz export with max_columns limit"""
    graph = create_test_graph()

    # Add more columns
    for i in range(10):
        col = PipelineColumnNode(
            column_name=f"col_{i}",
            table_name="raw.orders",
            query_id=None,
            node_type="source",
            full_name=f"raw.orders.col_{i}",
        )
        graph.add_column(col)

    # Export with max_columns=3
    dot_content = GraphVizExporter.export(graph, max_columns=3)

    # Should only include 3 columns
    node_count = dot_content.count('[label="')
    assert node_count == 3


if __name__ == "__main__":
    # Run tests
    test_json_export()
    test_json_export_without_metadata()
    test_json_export_to_file()
    test_csv_columns_export()
    test_csv_tables_export()
    test_graphviz_export()
    test_graphviz_export_with_descriptions()
    test_graphviz_export_to_file()
    test_graphviz_max_columns()

    print("✅ All export tests passed!")
