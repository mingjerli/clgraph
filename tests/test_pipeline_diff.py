"""
Tests for Phase 3 PipelineDiff functionality.

Week 4: Testing diff and incremental update detection.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clgraph.models import ColumnEdge, ColumnNode
from clgraph.pipeline import Pipeline
from clgraph.table import TableDependencyGraph


def test_diff_no_changes():
    """Test diff when pipelines are identical"""
    # Create two identical graphs
    table_graph1 = TableDependencyGraph()
    graph1 = Pipeline._create_empty(table_graph=table_graph1)

    col1 = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
        expression="user_id",
    )
    graph1.add_column(col1)

    table_graph2 = TableDependencyGraph()
    graph2 = Pipeline._create_empty(table_graph=table_graph2)

    col2 = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
        expression="user_id",
    )
    graph2.add_column(col2)

    # Compute diff
    diff = graph2.diff(graph1)

    # Verify no changes
    assert not diff.has_changes()
    assert len(diff.columns_added) == 0
    assert len(diff.columns_removed) == 0
    assert len(diff.columns_modified) == 0


def test_diff_column_added():
    """Test diff detects new columns"""
    # Create old graph
    table_graph1 = TableDependencyGraph()
    graph1 = Pipeline._create_empty(table_graph=table_graph1)

    col1 = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
    )
    graph1.add_column(col1)

    # Create new graph with additional column
    table_graph2 = TableDependencyGraph()
    graph2 = Pipeline._create_empty(table_graph=table_graph2)

    col2a = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
    )
    graph2.add_column(col2a)

    col2b = ColumnNode(
        column_name="email",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.email",
    )
    graph2.add_column(col2b)

    # Compute diff
    diff = graph2.diff(graph1)

    # Verify addition detected
    assert diff.has_changes()
    assert len(diff.columns_added) == 1
    assert "users.email" in diff.columns_added
    assert len(diff.columns_removed) == 0
    assert len(diff.columns_modified) == 0


def test_diff_column_removed():
    """Test diff detects removed columns"""
    # Create old graph with two columns
    table_graph1 = TableDependencyGraph()
    graph1 = Pipeline._create_empty(table_graph=table_graph1)

    col1a = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
    )
    graph1.add_column(col1a)

    col1b = ColumnNode(
        column_name="email",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.email",
    )
    graph1.add_column(col1b)

    # Create new graph with one column removed
    table_graph2 = TableDependencyGraph()
    graph2 = Pipeline._create_empty(table_graph=table_graph2)

    col2 = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
    )
    graph2.add_column(col2)

    # Compute diff
    diff = graph2.diff(graph1)

    # Verify removal detected
    assert diff.has_changes()
    assert len(diff.columns_added) == 0
    assert len(diff.columns_removed) == 1
    assert "users.email" in diff.columns_removed
    assert len(diff.columns_modified) == 0


def test_diff_sql_expression_changed():
    """Test diff detects SQL expression changes"""
    # Create old graph
    table_graph1 = TableDependencyGraph()
    graph1 = Pipeline._create_empty(table_graph=table_graph1)

    col1 = ColumnNode(
        column_name="total",
        table_name="metrics",
        query_id="q1",
        node_type="intermediate",
        full_name="metrics.total",
        expression="SUM(amount)",  # Old expression
    )
    graph1.add_column(col1)

    # Create new graph with changed expression
    table_graph2 = TableDependencyGraph()
    graph2 = Pipeline._create_empty(table_graph=table_graph2)

    col2 = ColumnNode(
        column_name="total",
        table_name="metrics",
        query_id="q1",
        node_type="intermediate",
        full_name="metrics.total",
        expression="SUM(amount * 1.1)",  # New expression
    )
    graph2.add_column(col2)

    # Compute diff
    diff = graph2.diff(graph1)

    # Verify expression change detected
    assert diff.has_changes()
    assert len(diff.columns_added) == 0
    assert len(diff.columns_removed) == 0
    assert len(diff.columns_modified) == 1

    # Check SQL changes
    sql_changes = diff.get_sql_changes()
    assert len(sql_changes) == 1
    assert sql_changes[0].column_name == "total"
    assert sql_changes[0].field_name == "expression"
    assert sql_changes[0].old_value == "SUM(amount)"
    assert sql_changes[0].new_value == "SUM(amount * 1.1)"


def test_diff_lineage_changed():
    """Test diff detects lineage changes (different source columns)"""
    # Create old graph
    table_graph1 = TableDependencyGraph()
    graph1 = Pipeline._create_empty(table_graph=table_graph1)

    source1 = ColumnNode(
        column_name="price",
        table_name="orders",
        query_id="q1",
        node_type="source",
        full_name="orders.price",
    )
    graph1.add_column(source1)

    derived1 = ColumnNode(
        column_name="total",
        table_name="metrics",
        query_id="q2",
        node_type="intermediate",
        full_name="metrics.total",
    )
    graph1.add_column(derived1)

    edge1 = ColumnEdge(from_node=source1, to_node=derived1, edge_type="direct")
    graph1.add_edge(edge1)

    # Create new graph with different source
    table_graph2 = TableDependencyGraph()
    graph2 = Pipeline._create_empty(table_graph=table_graph2)

    source2 = ColumnNode(
        column_name="amount",  # Different source column
        table_name="orders",
        query_id="q1",
        node_type="source",
        full_name="orders.amount",
    )
    graph2.add_column(source2)

    derived2 = ColumnNode(
        column_name="total",
        table_name="metrics",
        query_id="q2",
        node_type="intermediate",
        full_name="metrics.total",
    )
    graph2.add_column(derived2)

    edge2 = ColumnEdge(from_node=source2, to_node=derived2, edge_type="direct")
    graph2.add_edge(edge2)

    # Compute diff
    diff = graph2.diff(graph1)

    # Verify lineage change detected
    assert diff.has_changes()
    lineage_changes = diff.get_lineage_changes()
    assert len(lineage_changes) == 1
    assert lineage_changes[0].column_name == "total"
    assert lineage_changes[0].field_name == "source_columns"


def test_diff_get_columns_needing_update():
    """Test getting columns that need metadata regeneration"""
    # Create old graph
    table_graph1 = TableDependencyGraph()
    graph1 = Pipeline._create_empty(table_graph=table_graph1)

    col1a = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
    )
    graph1.add_column(col1a)

    col1b = ColumnNode(
        column_name="total",
        table_name="metrics",
        query_id="q2",
        node_type="intermediate",
        full_name="metrics.total",
        expression="SUM(amount)",
    )
    graph1.add_column(col1b)

    # Create new graph with:
    # - One new column (users.email)
    # - One modified column (metrics.total)
    # - One unchanged column (users.user_id)
    table_graph2 = TableDependencyGraph()
    graph2 = Pipeline._create_empty(table_graph=table_graph2)

    col2a = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
    )
    graph2.add_column(col2a)

    col2b = ColumnNode(
        column_name="email",  # NEW
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.email",
    )
    graph2.add_column(col2b)

    col2c = ColumnNode(
        column_name="total",
        table_name="metrics",
        query_id="q2",
        node_type="intermediate",
        full_name="metrics.total",
        expression="SUM(amount * 1.1)",  # MODIFIED
    )
    graph2.add_column(col2c)

    # Compute diff
    diff = graph2.diff(graph1)

    # Get columns needing update
    needing_update = diff.get_columns_needing_update()

    # Should include new and modified columns
    assert len(needing_update) == 2
    assert "users.email" in needing_update
    assert "metrics.total" in needing_update
    assert "users.user_id" not in needing_update  # Unchanged


def test_diff_summary():
    """Test diff summary generation"""
    # Create graphs with various changes
    table_graph1 = TableDependencyGraph()
    graph1 = Pipeline._create_empty(table_graph=table_graph1)

    col1a = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
    )
    graph1.add_column(col1a)

    col1b = ColumnNode(
        column_name="email",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.email",
    )
    graph1.add_column(col1b)

    table_graph2 = TableDependencyGraph()
    graph2 = Pipeline._create_empty(table_graph=table_graph2)

    col2a = ColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
    )
    graph2.add_column(col2a)

    col2b = ColumnNode(
        column_name="name",  # New column
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.name",
    )
    graph2.add_column(col2b)

    # Compute diff
    diff = graph2.diff(graph1)

    # Generate summary
    summary = diff.summary()

    # Verify summary contains key information
    assert "Pipeline Diff Summary" in summary
    assert "Columns Added" in summary or "Columns Removed" in summary
    assert isinstance(summary, str)
    assert len(summary) > 0


if __name__ == "__main__":
    # Run tests
    test_diff_no_changes()
    test_diff_column_added()
    test_diff_column_removed()
    test_diff_sql_expression_changed()
    test_diff_lineage_changed()
    test_diff_get_columns_needing_update()
    test_diff_summary()

    print("✅ All diff tests passed!")
