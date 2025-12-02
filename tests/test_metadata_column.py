"""
Tests for Phase 3 metadata fields and methods on ColumnNode.

Week 1: Testing metadata fields and basic methods.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clgraph.parser import ColumnNode, DescriptionSource


def test_column_metadata_fields_default():
    """Test that metadata fields have correct default values"""
    col = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="base_column",
        full_name="users.user_id",
        expression="user_id",
    )

    # Test default metadata fields
    assert col.description is None
    assert col.description_source is None
    assert col.owner is None
    assert col.pii is False
    assert col.tags == set()
    assert col.custom_metadata == {}


def test_column_metadata_fields_assignment():
    """Test that metadata fields can be assigned"""
    col = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="base_column",
        full_name="users.user_id",
    )

    # Assign metadata
    col.description = "Unique user identifier"
    col.description_source = DescriptionSource.SOURCE
    col.owner = "analytics_team"
    col.pii = True
    col.tags = {"important", "key"}
    col.custom_metadata = {"format": "UUID"}

    # Verify assignments
    assert col.description == "Unique user identifier"
    assert col.description_source == DescriptionSource.SOURCE
    assert col.owner == "analytics_team"
    assert col.pii is True
    assert col.tags == {"important", "key"}
    assert col.custom_metadata == {"format": "UUID"}


def test_set_source_description():
    """Test set_source_description method"""
    col = ColumnNode(
        column_name="email",
        table_name="users",
        query_id="q1",
        node_type="base_column",
        full_name="users.email",
    )

    # Set source description
    col.set_source_description("User email address")

    # Verify description and source
    assert col.description == "User email address"
    assert col.description_source == DescriptionSource.SOURCE


def test_set_source_description_overwrites():
    """Test that set_source_description overwrites existing description"""
    col = ColumnNode(
        column_name="email",
        table_name="users",
        query_id="q1",
        node_type="base_column",
        full_name="users.email",
    )

    # Set initial description
    col.description = "Old description"
    col.description_source = DescriptionSource.GENERATED

    # Overwrite with source description
    col.set_source_description("User email address")

    # Verify new description
    assert col.description == "User email address"
    assert col.description_source == DescriptionSource.SOURCE


def test_is_computed_base_column():
    """Test is_computed returns True for base columns with query_id (in derived table)"""
    col = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",  # Has query_id, so it's computed (in a derived table)
        node_type="base_column",
        full_name="users.user_id",
    )

    # Even though it's a base_column, if it has query_id it's computed
    assert col.is_computed()


def test_is_computed_source_column():
    """Test is_computed returns False for source columns"""
    col = ColumnNode(
        column_name="user_id",
        table_name="raw_users",
        query_id=None,
        node_type="source",
        full_name="raw_users.user_id",
    )

    assert not col.is_computed()


def test_is_computed_intermediate_column():
    """Test is_computed returns True for intermediate columns"""
    col = ColumnNode(
        column_name="total_revenue",
        table_name="user_metrics",
        query_id="q2",
        node_type="intermediate",
        full_name="user_metrics.total_revenue",
        expression="SUM(amount)",
    )

    assert col.is_computed()


def test_is_computed_output_column():
    """Test is_computed returns True for output columns"""
    col = ColumnNode(
        column_name="total_revenue",
        table_name="user_metrics",
        query_id="q2",
        node_type="output",
        full_name="user_metrics.total_revenue",
        expression="SUM(amount)",
    )

    assert col.is_computed()


def test_hash_and_equality():
    """Test that hash and equality still work with metadata fields"""
    col1 = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="base_column",
        full_name="users.user_id",
    )

    col2 = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="base_column",
        full_name="users.user_id",
        description="Different description",  # Different metadata
        owner="different_owner",
    )

    # Should be equal based on full_name
    assert col1 == col2
    assert hash(col1) == hash(col2)

    # Can be added to set
    col_set = {col1, col2}
    assert len(col_set) == 1  # Same column despite different metadata


def test_tags_are_mutable_set():
    """Test that tags can be modified"""
    col = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="base_column",
        full_name="users.user_id",
    )

    # Start empty
    assert col.tags == set()

    # Add tags
    col.tags.add("important")
    col.tags.add("key")
    assert col.tags == {"important", "key"}

    # Remove tag
    col.tags.remove("key")
    assert col.tags == {"important"}


if __name__ == "__main__":
    # Run tests
    test_column_metadata_fields_default()
    test_column_metadata_fields_assignment()
    test_set_source_description()
    test_set_source_description_overwrites()
    test_is_computed_base_column()
    test_is_computed_source_column()
    test_is_computed_intermediate_column()
    test_is_computed_output_column()
    test_hash_and_equality()
    test_tags_are_mutable_set()

    print("✅ All tests passed!")
