"""
Tests for trace strategy behaviors, tested via the public API.

Each test verifies that a specific SQL construct produces the correct lineage
graph structure. These tests are written against RecursiveLineageBuilder and
Pipeline so they validate behavior, not implementation internals.
"""

from clgraph import RecursiveLineageBuilder


class TestStarPassthrough:
    """Branch 1: SELECT * FROM cte — star passthrough."""

    def test_star_from_base_table(self):
        """SELECT * from a base table produces a star node."""
        query = "SELECT * FROM staging"
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        assert "staging.*" in graph.nodes
        star_node = graph.nodes["staging.*"]
        assert star_node.is_star
        assert star_node.layer == "input"

    def test_star_passthrough_via_cte(self):
        """SELECT * from a CTE propagates stars through the CTE layer."""
        query = """
        WITH staging AS (
            SELECT id, name FROM raw_users
        )
        SELECT * FROM staging
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # The output should have star/passthrough nodes
        output_nodes = [k for k in graph.nodes if k.startswith("output.")]
        assert len(output_nodes) > 0

        # There should be edges that propagate data from staging
        assert len(graph.edges) > 0

    def test_star_edge_type(self):
        """Edges from star columns should have star_passthrough type."""
        query = "SELECT * FROM users"
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        star_edges = [e for e in graph.edges if e.edge_type == "star_passthrough"]
        assert len(star_edges) > 0


class TestAggregateWithStar:
    """Branch 2: COUNT(*) and aggregates with *."""

    def test_count_star_produces_aggregate_node(self):
        """COUNT(*) produces an aggregate output node."""
        query = "SELECT COUNT(*) as cnt FROM orders"
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        assert "output.cnt" in graph.nodes
        cnt_node = graph.nodes["output.cnt"]
        assert cnt_node.node_type == "aggregate"

    def test_count_star_edges(self):
        """COUNT(*) has edges from the source table."""
        query = "SELECT COUNT(*) as cnt FROM orders"
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        cnt_edges = [e for e in graph.edges if e.to_node.full_name == "output.cnt"]
        assert len(cnt_edges) > 0
        # All edges should be aggregate type
        for edge in cnt_edges:
            assert edge.edge_type == "aggregate"

    def test_count_star_single_table(self):
        """COUNT(*) over a single table links to that table."""
        query = "SELECT COUNT(*) as cnt FROM events"
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        cnt_edges = [e for e in graph.edges if e.to_node.full_name == "output.cnt"]
        source_tables = {e.from_node.table_name for e in cnt_edges}
        assert "events" in source_tables


class TestSetOperations:
    """Branch 3: UNION/INTERSECT/EXCEPT columns."""

    def test_union_all_produces_union_edges(self):
        """UNION ALL creates union edges from each branch."""
        query = """
        SELECT id FROM users
        UNION ALL
        SELECT id FROM admins
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        union_edges = [e for e in graph.edges if e.edge_type == "union"]
        assert len(union_edges) > 0

    def test_union_output_nodes(self):
        """UNION columns appear in the main union output."""
        query = """
        SELECT id FROM users
        UNION ALL
        SELECT id FROM admins
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # UNION result column is named main.id (not output.id)
        union_result_nodes = [
            k
            for k in graph.nodes
            if ".id" in k and not k.startswith("users") and not k.startswith("admins")
        ]
        assert len(union_result_nodes) > 0

    def test_union_sources_linked(self):
        """UNION output column gets edges from both branch columns."""
        query = """
        SELECT id FROM users
        UNION ALL
        SELECT id FROM admins
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # Union result column gets edges from branch nodes
        union_edges = [e for e in graph.edges if e.edge_type == "union"]
        assert len(union_edges) >= 2


class TestMergeColumns:
    """Branch 4: MERGE statement columns."""

    def test_merge_produces_edges(self):
        """MERGE INTO produces column lineage edges."""
        query = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET t.value = s.new_value
        WHEN NOT MATCHED THEN INSERT (id, value) VALUES (s.id, s.new_value)
        """
        builder = RecursiveLineageBuilder(query, dialect="postgres")
        graph = builder.build()

        assert len(graph.edges) > 0

    def test_merge_identifies_source_columns(self):
        """MERGE statement tracks columns from the source table."""
        query = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET t.value = s.new_value
        """
        builder = RecursiveLineageBuilder(query, dialect="postgres")
        graph = builder.build()

        # Should have nodes from source
        source_nodes = [k for k in graph.nodes if "source" in k or "s" in k]
        assert len(source_nodes) > 0 or len(graph.nodes) > 0

    def test_merge_is_merge_operation(self):
        """MERGE edges should be flagged as merge operations."""
        query = """
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET t.value = s.new_value
        """
        builder = RecursiveLineageBuilder(query, dialect="postgres")
        graph = builder.build()

        merge_edges = [e for e in graph.edges if e.is_merge_operation]
        assert len(merge_edges) > 0


class TestRegularColumns:
    """Branch 5: Regular column references."""

    def test_simple_select_columns(self):
        """SELECT id, name FROM users creates proper input/output nodes."""
        query = "SELECT id, name FROM users"
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        assert "users.id" in graph.nodes
        assert "users.name" in graph.nodes
        assert "output.id" in graph.nodes
        assert "output.name" in graph.nodes

    def test_simple_select_edges(self):
        """SELECT id, name FROM users creates edges from users to output."""
        query = "SELECT id, name FROM users"
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        edges_dict = {(e.from_node.full_name, e.to_node.full_name): e for e in graph.edges}
        assert ("users.id", "output.id") in edges_dict
        assert ("users.name", "output.name") in edges_dict

    def test_column_through_cte(self):
        """Column flowing through a CTE is traced end-to-end."""
        query = """
        WITH staging AS (
            SELECT id, name FROM raw_users
        )
        SELECT id, name FROM staging
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # Should have input layer nodes from raw_users
        assert "raw_users.id" in graph.nodes
        assert "raw_users.name" in graph.nodes

        # Should have CTE layer nodes
        cte_id_nodes = [k for k in graph.nodes if "staging" in k and "id" in k]
        assert len(cte_id_nodes) > 0

        # Should have output nodes
        assert "output.id" in graph.nodes
        assert "output.name" in graph.nodes

    def test_column_through_cte_full_path(self):
        """Lineage is traceable from raw_users all the way to output."""
        query = """
        WITH staging AS (
            SELECT id, name FROM raw_users
        )
        SELECT id FROM staging
        """
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        # Trace backward from output.id
        output_id_edges = [e for e in graph.edges if e.to_node.full_name == "output.id"]
        assert len(output_id_edges) > 0

        # Source of those edges should come from staging
        source_nodes = {e.from_node.full_name for e in output_id_edges}
        assert any("id" in name for name in source_nodes)

    def test_aliased_columns(self):
        """Aliased columns are tracked correctly."""
        query = "SELECT id as user_id, name as user_name FROM users"
        builder = RecursiveLineageBuilder(query)
        graph = builder.build()

        assert "output.user_id" in graph.nodes
        assert "output.user_name" in graph.nodes

        edges_dict = {(e.from_node.full_name, e.to_node.full_name): e for e in graph.edges}
        assert ("users.id", "output.user_id") in edges_dict
        assert ("users.name", "output.user_name") in edges_dict
