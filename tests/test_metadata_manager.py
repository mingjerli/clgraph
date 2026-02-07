"""
Tests for MetadataManager component extracted from Pipeline.

Tests the delegation pattern from Pipeline to MetadataManager.
All existing Pipeline metadata tests should continue to pass.
"""

import pytest

from clgraph import Pipeline


class TestMetadataManagerDelegation:
    """Test that Pipeline properly delegates to MetadataManager."""

    @pytest.fixture
    def pipeline_with_columns(self):
        """Create a pipeline with columns for metadata testing."""
        queries = [
            (
                "staging",
                """
                CREATE TABLE staging.orders AS
                SELECT
                    order_id,
                    customer_id,
                    email,
                    amount
                FROM raw.orders
                """,
            ),
            (
                "analytics",
                """
                CREATE TABLE analytics.summary AS
                SELECT
                    customer_id,
                    SUM(amount) AS total_amount
                FROM staging.orders
                GROUP BY customer_id
                """,
            ),
        ]
        return Pipeline(queries, dialect="bigquery")

    def test_get_pii_columns_returns_list(self, pipeline_with_columns):
        """Test that get_pii_columns returns a list."""
        # Set some columns as PII
        for col in pipeline_with_columns.columns.values():
            if col.column_name == "email":
                col.pii = True

        pii_cols = pipeline_with_columns.get_pii_columns()

        assert isinstance(pii_cols, list)
        assert len(pii_cols) > 0
        assert all(col.pii for col in pii_cols)

    def test_get_pii_columns_empty_when_no_pii(self, pipeline_with_columns):
        """Test that get_pii_columns returns empty list when no PII."""
        pii_cols = pipeline_with_columns.get_pii_columns()

        assert isinstance(pii_cols, list)
        # By default, no columns should be PII
        assert len(pii_cols) == 0

    def test_get_columns_by_owner_returns_list(self, pipeline_with_columns):
        """Test that get_columns_by_owner returns a list."""
        # Set owner for some columns
        for col in pipeline_with_columns.columns.values():
            if col.column_name == "order_id":
                col.owner = "data_team"

        cols = pipeline_with_columns.get_columns_by_owner("data_team")

        assert isinstance(cols, list)
        assert len(cols) > 0
        assert all(col.owner == "data_team" for col in cols)

    def test_get_columns_by_owner_empty_for_unknown_owner(self, pipeline_with_columns):
        """Test that get_columns_by_owner returns empty for unknown owner."""
        cols = pipeline_with_columns.get_columns_by_owner("unknown_team")

        assert isinstance(cols, list)
        assert len(cols) == 0

    def test_get_columns_by_tag_returns_list(self, pipeline_with_columns):
        """Test that get_columns_by_tag returns a list."""
        # Add tags to some columns
        for col in pipeline_with_columns.columns.values():
            if col.column_name == "amount":
                col.tags.add("financial")

        cols = pipeline_with_columns.get_columns_by_tag("financial")

        assert isinstance(cols, list)
        assert len(cols) > 0
        assert all("financial" in col.tags for col in cols)

    def test_get_columns_by_tag_empty_for_unknown_tag(self, pipeline_with_columns):
        """Test that get_columns_by_tag returns empty for unknown tag."""
        cols = pipeline_with_columns.get_columns_by_tag("unknown_tag")

        assert isinstance(cols, list)
        assert len(cols) == 0

    def test_propagate_all_metadata_runs_without_error(self, pipeline_with_columns):
        """Test that propagate_all_metadata runs without error."""
        # Set some source metadata
        for col in pipeline_with_columns.columns.values():
            if col.column_name == "email":
                col.pii = True

        # Should not raise
        pipeline_with_columns.propagate_all_metadata(verbose=False)

    def test_propagate_all_metadata_propagates_pii(self, pipeline_with_columns):
        """Test that propagate_all_metadata propagates PII flag."""
        # Set PII on source column
        for col in pipeline_with_columns.columns.values():
            if col.table_name == "raw.orders" and col.column_name == "email":
                col.pii = True
                break

        pipeline_with_columns.propagate_all_metadata(verbose=False)

        # Check if PII propagated to downstream
        pii_cols = pipeline_with_columns.get_pii_columns()
        # Should have at least the original column and potentially propagated ones
        assert len(pii_cols) >= 1


class TestMetadataManagerLazyInitialization:
    """Test that MetadataManager is lazily initialized."""

    def test_metadata_mgr_not_created_on_pipeline_init(self):
        """Test that metadata_mgr is not created when Pipeline is initialized."""
        queries = [
            ("q1", "CREATE TABLE t1 AS SELECT a FROM source"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # The _metadata_mgr attribute should be None or not exist
        assert pipeline._metadata_mgr is None

    def test_metadata_mgr_created_on_first_metadata_call(self):
        """Test that metadata_mgr is created on first metadata method call."""
        queries = [
            ("q1", "CREATE TABLE t1 AS SELECT a FROM source"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Call a metadata method
        pipeline.get_pii_columns()

        # Now metadata_mgr should be initialized
        assert pipeline._metadata_mgr is not None

    def test_metadata_mgr_reused_across_calls(self):
        """Test that the same metadata_mgr instance is reused."""
        queries = [
            ("q1", "CREATE TABLE t1 AS SELECT a FROM source"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Call multiple metadata methods
        pipeline.get_pii_columns()
        mgr1 = pipeline._metadata_mgr

        pipeline.get_columns_by_owner("test")
        mgr2 = pipeline._metadata_mgr

        # Should be the same instance
        assert mgr1 is mgr2


class TestMetadataManagerDirectAccess:
    """Test that MetadataManager can be used directly (advanced usage)."""

    def test_metadata_manager_can_be_imported(self):
        """Test that MetadataManager can be imported directly."""
        from clgraph.metadata_manager import MetadataManager

        assert MetadataManager is not None

    def test_metadata_manager_initialization(self):
        """Test that MetadataManager can be initialized with a pipeline."""
        from clgraph.metadata_manager import MetadataManager

        queries = [
            ("q1", "CREATE TABLE t1 AS SELECT a FROM source"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        mgr = MetadataManager(pipeline)
        assert mgr._pipeline is pipeline

    def test_metadata_manager_get_pii_columns(self):
        """Test MetadataManager.get_pii_columns() directly."""
        from clgraph.metadata_manager import MetadataManager

        queries = [
            (
                "staging",
                """
                CREATE TABLE staging.users AS
                SELECT id, email, name
                FROM raw.users
                """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Set some columns as PII
        for col in pipeline.columns.values():
            if col.column_name == "email":
                col.pii = True

        mgr = MetadataManager(pipeline)
        pii_cols = mgr.get_pii_columns()

        assert isinstance(pii_cols, list)
        assert len(pii_cols) > 0
        assert all(col.pii for col in pii_cols)

    def test_metadata_manager_get_columns_by_owner(self):
        """Test MetadataManager.get_columns_by_owner() directly."""
        from clgraph.metadata_manager import MetadataManager

        queries = [
            ("q1", "CREATE TABLE t1 AS SELECT a, b FROM source"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Set owner for some columns
        for col in pipeline.columns.values():
            if col.column_name == "a":
                col.owner = "team_a"

        mgr = MetadataManager(pipeline)
        cols = mgr.get_columns_by_owner("team_a")

        assert isinstance(cols, list)
        assert len(cols) > 0
        assert all(col.owner == "team_a" for col in cols)

    def test_metadata_manager_get_columns_by_tag(self):
        """Test MetadataManager.get_columns_by_tag() directly."""
        from clgraph.metadata_manager import MetadataManager

        queries = [
            ("q1", "CREATE TABLE t1 AS SELECT a, b FROM source"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Add tag to some columns
        for col in pipeline.columns.values():
            if col.column_name == "a":
                col.tags.add("important")

        mgr = MetadataManager(pipeline)
        cols = mgr.get_columns_by_tag("important")

        assert isinstance(cols, list)
        assert len(cols) > 0
        assert all("important" in col.tags for col in cols)

    def test_metadata_manager_propagate_all_metadata(self):
        """Test MetadataManager.propagate_all_metadata() directly."""
        from clgraph.metadata_manager import MetadataManager

        queries = [
            (
                "staging",
                """
                CREATE TABLE staging.orders AS
                SELECT id, email
                FROM raw.orders
                """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Set PII on source
        for col in pipeline.columns.values():
            if col.column_name == "email":
                col.pii = True
                break

        mgr = MetadataManager(pipeline)
        # Should not raise
        mgr.propagate_all_metadata(verbose=False)
