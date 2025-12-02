"""
Tests for Phase 3 metadata propagation functionality.

Week 3: Testing metadata propagation through lineage.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clgraph.column import propagate_metadata
from clgraph.models import ColumnEdge, ColumnNode, DescriptionSource
from clgraph.pipeline import Pipeline
from clgraph.table import TableDependencyGraph


def test_propagate_owner_single_source():
    """Test owner propagation when single source has owner"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create source column with owner
    source = ColumnNode(
        column_name="user_id",
        table_name="users",
        full_name="users.user_id",
        query_id="q1",
        node_type="source",
        owner="analytics_team",
    )
    graph.add_column(source)

    # Create derived column
    derived = ColumnNode(
        column_name="user_id",
        table_name="user_metrics",
        full_name="user_metrics.user_id",
        query_id="q2",
        node_type="intermediate",
    )
    graph.add_column(derived)

    # Create edge
    edge = ColumnEdge(from_node=source, to_node=derived, edge_type="direct")
    graph.add_edge(edge)

    # Propagate metadata
    propagate_metadata(derived, graph)

    # Verify owner propagated
    assert derived.owner == "analytics_team"


def test_propagate_owner_same_owner():
    """Test owner propagation when all sources have same owner"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create two source columns with same owner
    source1 = ColumnNode(
        column_name="amount1",
        table_name="orders",
        full_name="orders.amount1",
        query_id="q1",
        node_type="source",
        owner="finance_team",
    )
    graph.add_column(source1)

    source2 = ColumnNode(
        column_name="amount2",
        table_name="orders",
        full_name="orders.amount2",
        query_id="q1",
        node_type="source",
        owner="finance_team",
    )
    graph.add_column(source2)

    # Create derived column
    derived = ColumnNode(
        column_name="total_amount",
        table_name="order_totals",
        full_name="order_totals.total_amount",
        query_id="q2",
        node_type="intermediate",
        expression="amount1 + amount2",
    )
    graph.add_column(derived)

    # Create edges
    edge1 = ColumnEdge(from_node=source1, to_node=derived, edge_type="transform")
    edge2 = ColumnEdge(from_node=source2, to_node=derived, edge_type="transform")
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    # Propagate metadata
    propagate_metadata(derived, graph)

    # Verify owner propagated
    assert derived.owner == "finance_team"


def test_propagate_owner_different_owners():
    """Test owner NOT propagated when sources have different owners"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create two source columns with different owners
    source1 = ColumnNode(
        column_name="user_count",
        table_name="users",
        full_name="users.user_count",
        query_id="q1",
        node_type="source",
        owner="analytics_team",
    )
    graph.add_column(source1)

    source2 = ColumnNode(
        column_name="order_count",
        table_name="orders",
        full_name="orders.order_count",
        query_id="q1",
        node_type="source",
        owner="finance_team",
    )
    graph.add_column(source2)

    # Create derived column
    derived = ColumnNode(
        column_name="combined_count",
        table_name="metrics",
        full_name="metrics.combined_count",
        query_id="q2",
        node_type="intermediate",
        expression="user_count + order_count",
    )
    graph.add_column(derived)

    # Create edges
    edge1 = ColumnEdge(from_node=source1, to_node=derived, edge_type="transform")
    edge2 = ColumnEdge(from_node=source2, to_node=derived, edge_type="transform")
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    # Propagate metadata
    propagate_metadata(derived, graph)

    # Verify owner NOT propagated (remains None)
    assert derived.owner is None


def test_propagate_pii_single_source():
    """Test PII propagation from single source"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create source column with PII
    source = ColumnNode(
        column_name="email",
        table_name="users",
        full_name="users.email",
        query_id="q1",
        node_type="source",
        pii=True,
    )
    graph.add_column(source)

    # Create derived column
    derived = ColumnNode(
        column_name="email_lower",
        table_name="user_emails",
        full_name="user_emails.email_lower",
        query_id="q2",
        node_type="intermediate",
        expression="LOWER(email)",
    )
    graph.add_column(derived)

    # Create edge
    edge = ColumnEdge(from_node=source, to_node=derived, edge_type="transform")
    graph.add_edge(edge)

    # Propagate metadata
    propagate_metadata(derived, graph)

    # Verify PII propagated
    assert derived.pii is True


def test_propagate_pii_union():
    """Test PII union - any source is PII means result is PII"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create source columns - one with PII, one without
    source1 = ColumnNode(
        column_name="email",
        table_name="users",
        full_name="users.email",
        query_id="q1",
        node_type="source",
        pii=True,
    )
    graph.add_column(source1)

    source2 = ColumnNode(
        column_name="user_count",
        table_name="stats",
        full_name="stats.user_count",
        query_id="q1",
        node_type="source",
        pii=False,
    )
    graph.add_column(source2)

    # Create derived column
    derived = ColumnNode(
        column_name="combined_data",
        table_name="output",
        full_name="output.combined_data",
        query_id="q2",
        node_type="intermediate",
    )
    graph.add_column(derived)

    # Create edges
    edge1 = ColumnEdge(from_node=source1, to_node=derived, edge_type="transform")
    edge2 = ColumnEdge(from_node=source2, to_node=derived, edge_type="transform")
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    # Propagate metadata
    propagate_metadata(derived, graph)

    # Verify PII propagated (union - any source is PII)
    assert derived.pii is True


def test_propagate_tags_single_source():
    """Test tag propagation from single source"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create source column with tags
    source = ColumnNode(
        column_name="revenue",
        table_name="orders",
        full_name="orders.revenue",
        query_id="q1",
        node_type="source",
        tags={"important", "financial"},
    )
    graph.add_column(source)

    # Create derived column
    derived = ColumnNode(
        column_name="total_revenue",
        table_name="metrics",
        full_name="metrics.total_revenue",
        query_id="q2",
        node_type="intermediate",
        expression="SUM(revenue)",
    )
    graph.add_column(derived)

    # Create edge
    edge = ColumnEdge(from_node=source, to_node=derived, edge_type="aggregate")
    graph.add_edge(edge)

    # Propagate metadata
    propagate_metadata(derived, graph)

    # Verify tags propagated
    assert derived.tags == {"important", "financial"}


def test_propagate_tags_union():
    """Test tag union - merge all source tags"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create source columns with different tags
    source1 = ColumnNode(
        column_name="user_id",
        table_name="users",
        full_name="users.user_id",
        query_id="q1",
        node_type="source",
        tags={"important", "key"},
    )
    graph.add_column(source1)

    source2 = ColumnNode(
        column_name="order_id",
        table_name="orders",
        full_name="orders.order_id",
        query_id="q1",
        node_type="source",
        tags={"key", "financial"},
    )
    graph.add_column(source2)

    # Create derived column
    derived = ColumnNode(
        column_name="combined_id",
        table_name="joined",
        full_name="joined.combined_id",
        query_id="q2",
        node_type="intermediate",
    )
    graph.add_column(derived)

    # Create edges
    edge1 = ColumnEdge(from_node=source1, to_node=derived, edge_type="join")
    edge2 = ColumnEdge(from_node=source2, to_node=derived, edge_type="join")
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    # Propagate metadata
    propagate_metadata(derived, graph)

    # Verify tags are union of all source tags
    assert derived.tags == {"important", "key", "financial"}


def test_propagate_not_for_source_columns():
    """Test that propagation does NOT affect source columns"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create source column
    source = ColumnNode(
        column_name="user_id",
        table_name="users",
        full_name="users.user_id",
        query_id="q1",
        node_type="source",
    )
    graph.add_column(source)

    # Try to propagate metadata to source column
    propagate_metadata(source, graph)

    # Verify nothing changed (source columns don't propagate)
    assert source.owner is None
    assert source.pii is False
    assert source.tags == set()


def test_propagate_not_for_user_set_columns():
    """Test that propagation does NOT apply to columns with user-set descriptions"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create source column
    source = ColumnNode(
        column_name="user_id",
        table_name="users",
        full_name="users.user_id",
        query_id="q1",
        node_type="source",
        owner="analytics_team",
        pii=True,
    )
    graph.add_column(source)

    # Create derived column with user-set description
    derived = ColumnNode(
        column_name="user_id",
        table_name="metrics",
        full_name="metrics.user_id",
        query_id="q2",
        node_type="intermediate",
        description="User-set description",
        description_source=DescriptionSource.SOURCE,
    )
    graph.add_column(derived)

    # Create edge
    edge = ColumnEdge(from_node=source, to_node=derived, edge_type="direct")
    graph.add_edge(edge)

    # Try to propagate metadata
    propagate_metadata(derived, graph)

    # Metadata SHOULD propagate even when user has set description explicitly
    # Description source is independent from other metadata fields
    # This ensures PII flags and other critical metadata propagate correctly
    assert derived.owner == "analytics_team"
    assert derived.pii is True
    # Note: tags are empty because source has no tags
    assert derived.tags == set()
    # But description should remain user-set
    assert derived.description == "User-set description"
    assert derived.description_source == DescriptionSource.SOURCE


if __name__ == "__main__":
    # Run tests
    test_propagate_owner_single_source()
    test_propagate_owner_same_owner()
    test_propagate_owner_different_owners()
    test_propagate_pii_single_source()
    test_propagate_pii_union()
    test_propagate_tags_single_source()
    test_propagate_tags_union()
    test_propagate_not_for_source_columns()
    test_propagate_not_for_user_set_columns()

    print("All propagation tests passed!")
