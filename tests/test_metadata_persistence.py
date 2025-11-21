"""
Tests for Phase 3 metadata persistence functionality.

Week 3: Testing save/load/apply_metadata.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clpipe.column import PipelineColumnNode
from clpipe.models import DescriptionSource
from clpipe.pipeline import Pipeline
from clpipe.table import TableDependencyGraph, TableNode


def test_save_and_load_metadata():
    """Test save and load metadata round-trip"""
    # Create graph with metadata
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Add column with metadata
    col = PipelineColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
        description="Unique user identifier",
        description_source=DescriptionSource.SOURCE,
        owner="analytics_team",
        pii=True,
        tags={"important", "key"},
        custom_metadata={"format": "UUID"},
    )
    graph.add_column(col)

    # Add table with description
    table = TableNode(table_name="users", is_source=True, description="User data table")
    graph.table_graph.tables["users"] = table

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
        temp_path = f.name

    try:
        graph.save(temp_path)

        # Load metadata
        metadata = Pipeline.load_metadata(temp_path)

        # Verify metadata structure
        assert "version" in metadata
        assert "columns" in metadata
        assert "tables" in metadata

        # Verify column metadata
        assert "users.user_id" in metadata["columns"]
        col_meta = metadata["columns"]["users.user_id"]
        assert col_meta["description"] == "Unique user identifier"
        assert col_meta["description_source"] == "source"
        assert col_meta["owner"] == "analytics_team"
        assert col_meta["pii"] is True
        assert set(col_meta["tags"]) == {"important", "key"}
        assert col_meta["custom_metadata"] == {"format": "UUID"}

        # Verify table metadata
        assert "users" in metadata["tables"]
        table_meta = metadata["tables"]["users"]
        assert table_meta["description"] == "User data table"

    finally:
        # Clean up
        Path(temp_path).unlink()


def test_apply_metadata():
    """Test applying loaded metadata to new graph"""
    # Create original graph
    table_graph1 = TableDependencyGraph()
    graph1 = Pipeline._create_empty(table_graph=table_graph1)

    col1 = PipelineColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
        description="Unique user identifier",
        description_source=DescriptionSource.SOURCE,
        owner="analytics_team",
        pii=True,
        tags={"important"},
    )
    graph1.add_column(col1)

    # Save metadata
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
        temp_path = f.name

    try:
        graph1.save(temp_path)

        # Create new graph with same structure but no metadata
        table_graph2 = TableDependencyGraph()
        graph2 = Pipeline._create_empty(table_graph=table_graph2)

        col2 = PipelineColumnNode(
            column_name="user_id",
            table_name="users",
            query_id="q1",
            node_type="source",
            full_name="users.user_id",
        )
        graph2.add_column(col2)

        # Verify no metadata initially
        assert col2.description is None
        assert col2.owner is None
        assert col2.pii is False
        assert col2.tags == set()

        # Load and apply metadata
        metadata = Pipeline.load_metadata(temp_path)
        graph2.apply_metadata(metadata)

        # Verify metadata was applied
        assert col2.description == "Unique user identifier"
        assert col2.description_source == DescriptionSource.SOURCE
        assert col2.owner == "analytics_team"
        assert col2.pii is True
        assert col2.tags == {"important"}

    finally:
        # Clean up
        Path(temp_path).unlink()


def test_save_preserves_metadata_only():
    """Test that save only stores metadata, not SQL structure"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    col = PipelineColumnNode(
        column_name="revenue",
        table_name="orders",
        query_id="q1",
        node_type="intermediate",
        full_name="orders.revenue",
        expression="price * quantity",  # SQL structure
        description="Total revenue",  # Metadata
        owner="finance_team",  # Metadata
    )
    graph.add_column(col)

    # Save
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
        temp_path = f.name

    try:
        graph.save(temp_path)
        metadata = Pipeline.load_metadata(temp_path)

        # Verify metadata is saved
        col_meta = metadata["columns"]["orders.revenue"]
        assert col_meta["description"] == "Total revenue"
        assert col_meta["owner"] == "finance_team"

        # Verify SQL structure is NOT in metadata
        # (expression is part of the graph structure, not metadata)
        assert "expression" not in col_meta

    finally:
        Path(temp_path).unlink()


def test_apply_metadata_partial_match():
    """Test applying metadata when only some columns match"""
    # Create graph with metadata
    table_graph1 = TableDependencyGraph()
    graph1 = Pipeline._create_empty(table_graph=table_graph1)

    col1 = PipelineColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
        description="User ID",
        owner="team_a",
    )
    graph1.add_column(col1)

    # Save
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
        temp_path = f.name

    try:
        graph1.save(temp_path)

        # Create new graph with different columns
        table_graph2 = TableDependencyGraph()
        graph2 = Pipeline._create_empty(table_graph=table_graph2)

        col2a = PipelineColumnNode(
            column_name="user_id",
            table_name="users",
            query_id="q1",
            node_type="source",
            full_name="users.user_id",  # Matches saved metadata
        )
        graph2.add_column(col2a)

        col2b = PipelineColumnNode(
            column_name="email",
            table_name="users",
            query_id="q1",
            node_type="source",
            full_name="users.email",  # Does NOT match
        )
        graph2.add_column(col2b)

        # Apply metadata
        metadata = Pipeline.load_metadata(temp_path)
        graph2.apply_metadata(metadata)

        # Verify matching column got metadata
        assert col2a.description == "User ID"
        assert col2a.owner == "team_a"

        # Verify non-matching column unchanged
        assert col2b.description is None
        assert col2b.owner is None

    finally:
        Path(temp_path).unlink()


def test_apply_metadata_with_table_descriptions():
    """Test that table descriptions are also applied"""
    # Create graph
    table_graph1 = TableDependencyGraph()
    table = TableNode(table_name="users", is_source=True, description="User information table")
    table_graph1.tables["users"] = table
    graph1 = Pipeline._create_empty(table_graph=table_graph1)

    # Save
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
        temp_path = f.name

    try:
        graph1.save(temp_path)

        # Create new graph
        table_graph2 = TableDependencyGraph()
        table2 = TableNode(table_name="users", is_source=True)
        table_graph2.tables["users"] = table2
        graph2 = Pipeline._create_empty(table_graph=table_graph2)

        # Verify no description initially
        assert table2.description is None

        # Apply metadata
        metadata = Pipeline.load_metadata(temp_path)
        graph2.apply_metadata(metadata)

        # Verify table description applied
        assert table2.description == "User information table"

    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    # Run tests
    test_save_and_load_metadata()
    test_apply_metadata()
    test_save_preserves_metadata_only()
    test_apply_metadata_partial_match()
    test_apply_metadata_with_table_descriptions()

    print("✅ All persistence tests passed!")
