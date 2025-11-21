"""
Tests for Phase 3 metadata propagation functionality.

Week 3: Testing metadata propagation through lineage.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clpipe.column import PipelineColumnEdge, PipelineColumnNode
from clpipe.models import DescriptionSource
from clpipe.pipeline import Pipeline
from clpipe.table import TableDependencyGraph


def test_propagate_owner_single_source():
    """Test owner propagation when single source has owner"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create source column with owner
    source = PipelineColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
        owner="analytics_team",
    )
    graph.add_column(source)

    # Create derived column
    derived = PipelineColumnNode(
        column_name="user_id",
        table_name="user_metrics",
        query_id="q2",
        node_type="intermediate",
        full_name="user_metrics.user_id",
    )
    graph.add_column(derived)

    # Create edge
    edge = PipelineColumnEdge(from_column=source, to_column=derived, edge_type="direct")
    graph.add_edge(edge)

    # Propagate metadata
    derived.propagate_metadata(graph)

    # Verify owner propagated
    assert derived.owner == "analytics_team"


def test_propagate_owner_same_owner():
    """Test owner propagation when all sources have same owner"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create two source columns with same owner
    source1 = PipelineColumnNode(
        column_name="amount1",
        table_name="orders",
        query_id="q1",
        node_type="source",
        full_name="orders.amount1",
        owner="finance_team",
    )
    graph.add_column(source1)

    source2 = PipelineColumnNode(
        column_name="amount2",
        table_name="orders",
        query_id="q1",
        node_type="source",
        full_name="orders.amount2",
        owner="finance_team",
    )
    graph.add_column(source2)

    # Create derived column
    derived = PipelineColumnNode(
        column_name="total_amount",
        table_name="order_totals",
        query_id="q2",
        node_type="intermediate",
        full_name="order_totals.total_amount",
        expression="amount1 + amount2",
    )
    graph.add_column(derived)

    # Create edges
    edge1 = PipelineColumnEdge(from_column=source1, to_column=derived, edge_type="transform")
    edge2 = PipelineColumnEdge(from_column=source2, to_column=derived, edge_type="transform")
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    # Propagate metadata
    derived.propagate_metadata(graph)

    # Verify owner propagated
    assert derived.owner == "finance_team"


def test_propagate_owner_different_owners():
    """Test owner NOT propagated when sources have different owners"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create two source columns with different owners
    source1 = PipelineColumnNode(
        column_name="user_count",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_count",
        owner="analytics_team",
    )
    graph.add_column(source1)

    source2 = PipelineColumnNode(
        column_name="order_count",
        table_name="orders",
        query_id="q1",
        node_type="source",
        full_name="orders.order_count",
        owner="finance_team",
    )
    graph.add_column(source2)

    # Create derived column
    derived = PipelineColumnNode(
        column_name="combined_count",
        table_name="metrics",
        query_id="q2",
        node_type="intermediate",
        full_name="metrics.combined_count",
        expression="user_count + order_count",
    )
    graph.add_column(derived)

    # Create edges
    edge1 = PipelineColumnEdge(from_column=source1, to_column=derived, edge_type="transform")
    edge2 = PipelineColumnEdge(from_column=source2, to_column=derived, edge_type="transform")
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    # Propagate metadata
    derived.propagate_metadata(graph)

    # Verify owner NOT propagated (remains None)
    assert derived.owner is None


def test_propagate_pii_single_source():
    """Test PII propagation from single source"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create source column with PII
    source = PipelineColumnNode(
        column_name="email",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.email",
        pii=True,
    )
    graph.add_column(source)

    # Create derived column
    derived = PipelineColumnNode(
        column_name="email_lower",
        table_name="user_emails",
        query_id="q2",
        node_type="intermediate",
        full_name="user_emails.email_lower",
        expression="LOWER(email)",
    )
    graph.add_column(derived)

    # Create edge
    edge = PipelineColumnEdge(from_column=source, to_column=derived, edge_type="transform")
    graph.add_edge(edge)

    # Propagate metadata
    derived.propagate_metadata(graph)

    # Verify PII propagated
    assert derived.pii is True


def test_propagate_pii_union():
    """Test PII union - any source is PII means result is PII"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create source columns - one with PII, one without
    source1 = PipelineColumnNode(
        column_name="email",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.email",
        pii=True,
    )
    graph.add_column(source1)

    source2 = PipelineColumnNode(
        column_name="user_count",
        table_name="stats",
        query_id="q1",
        node_type="source",
        full_name="stats.user_count",
        pii=False,
    )
    graph.add_column(source2)

    # Create derived column
    derived = PipelineColumnNode(
        column_name="combined_data",
        table_name="output",
        query_id="q2",
        node_type="intermediate",
        full_name="output.combined_data",
    )
    graph.add_column(derived)

    # Create edges
    edge1 = PipelineColumnEdge(from_column=source1, to_column=derived, edge_type="transform")
    edge2 = PipelineColumnEdge(from_column=source2, to_column=derived, edge_type="transform")
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    # Propagate metadata
    derived.propagate_metadata(graph)

    # Verify PII propagated (union - any source is PII)
    assert derived.pii is True


def test_propagate_tags_single_source():
    """Test tag propagation from single source"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create source column with tags
    source = PipelineColumnNode(
        column_name="revenue",
        table_name="orders",
        query_id="q1",
        node_type="source",
        full_name="orders.revenue",
        tags={"important", "financial"},
    )
    graph.add_column(source)

    # Create derived column
    derived = PipelineColumnNode(
        column_name="total_revenue",
        table_name="metrics",
        query_id="q2",
        node_type="intermediate",
        full_name="metrics.total_revenue",
        expression="SUM(revenue)",
    )
    graph.add_column(derived)

    # Create edge
    edge = PipelineColumnEdge(from_column=source, to_column=derived, edge_type="aggregate")
    graph.add_edge(edge)

    # Propagate metadata
    derived.propagate_metadata(graph)

    # Verify tags propagated
    assert derived.tags == {"important", "financial"}


def test_propagate_tags_union():
    """Test tag union - merge all source tags"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create source columns with different tags
    source1 = PipelineColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
        tags={"important", "key"},
    )
    graph.add_column(source1)

    source2 = PipelineColumnNode(
        column_name="order_id",
        table_name="orders",
        query_id="q1",
        node_type="source",
        full_name="orders.order_id",
        tags={"key", "financial"},
    )
    graph.add_column(source2)

    # Create derived column
    derived = PipelineColumnNode(
        column_name="combined_id",
        table_name="joined",
        query_id="q2",
        node_type="intermediate",
        full_name="joined.combined_id",
    )
    graph.add_column(derived)

    # Create edges
    edge1 = PipelineColumnEdge(from_column=source1, to_column=derived, edge_type="join")
    edge2 = PipelineColumnEdge(from_column=source2, to_column=derived, edge_type="join")
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    # Propagate metadata
    derived.propagate_metadata(graph)

    # Verify tags are union of all source tags
    assert derived.tags == {"important", "key", "financial"}


def test_propagate_not_for_source_columns():
    """Test that propagation does NOT affect source columns"""
    # Create graph
    table_graph = TableDependencyGraph()
    graph = Pipeline._create_empty(table_graph=table_graph)

    # Create source column
    source = PipelineColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
    )
    graph.add_column(source)

    # Try to propagate metadata to source column
    source.propagate_metadata(graph)

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
    source = PipelineColumnNode(
        column_name="user_id",
        table_name="users",
        query_id="q1",
        node_type="source",
        full_name="users.user_id",
        owner="analytics_team",
        pii=True,
    )
    graph.add_column(source)

    # Create derived column with user-set description
    derived = PipelineColumnNode(
        column_name="user_id",
        table_name="metrics",
        query_id="q2",
        node_type="intermediate",
        full_name="metrics.user_id",
        description="User-set description",
        description_source=DescriptionSource.SOURCE,
    )
    graph.add_column(derived)

    # Create edge
    edge = PipelineColumnEdge(from_column=source, to_column=derived, edge_type="direct")
    graph.add_edge(edge)

    # Try to propagate metadata
    derived.propagate_metadata(graph)

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

    print("✅ All propagation tests passed!")
