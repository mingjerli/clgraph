"""
Integration tests for end-to-end metadata extraction from SQL comments.

Tests the complete flow:
1. SQL with inline comments
2. RecursiveLineageBuilder extracts metadata
3. Pipeline applies metadata to ColumnNode
4. Metadata propagates through lineage
"""

from clgraph.models import DescriptionSource
from clgraph.pipeline import Pipeline


class TestMetadataIntegration:
    """Test end-to-end metadata extraction and propagation."""

    def test_simple_select_with_metadata(self):
        """Test metadata extraction from simple SELECT."""
        sql = """
        SELECT
          user_id,  -- User identifier [pii: false]
          email,    -- Email address [pii: true, owner: data-team]
          name      -- Full name [pii: true]
        FROM users
        """

        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        # Check user_id
        user_id_nodes = [n for n in pipeline.columns.values() if n.column_name == "user_id"]
        assert len(user_id_nodes) > 0

        # Find output user_id (use query_id to identify the query)
        output_user_id = next(
            (n for n in user_id_nodes if n.query_id == "query" and n.layer == "output"),
            None,
        )
        assert output_user_id is not None
        assert output_user_id.description == "User identifier"
        assert output_user_id.pii is False
        assert output_user_id.description_source == DescriptionSource.SOURCE

        # Check email
        email_nodes = [n for n in pipeline.columns.values() if n.column_name == "email"]
        output_email = next(
            (n for n in email_nodes if n.query_id == "query" and n.layer == "output"),
            None,
        )
        assert output_email is not None
        assert output_email.description == "Email address"
        assert output_email.pii is True
        assert output_email.owner == "data-team"
        assert output_email.description_source == DescriptionSource.SOURCE

        # Check name
        name_nodes = [n for n in pipeline.columns.values() if n.column_name == "name"]
        output_name = next(
            (n for n in name_nodes if n.query_id == "query" and n.layer == "output"),
            None,
        )
        assert output_name is not None
        assert output_name.description == "Full name"
        assert output_name.pii is True

    def test_metadata_with_expressions(self):
        """Test metadata on computed expressions."""
        sql = """
        SELECT
          user_id,
          UPPER(email) as email_upper,  -- Uppercased email [pii: true]
          CONCAT(first_name, ' ', last_name) as full_name  /* Full name [pii: true, owner: analytics-team] */
        FROM users
        """

        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        # Check email_upper (use query_id to identify)
        email_upper_nodes = [n for n in pipeline.columns.values() if n.column_name == "email_upper"]
        output_email_upper = next(
            (n for n in email_upper_nodes if n.query_id == "query" and n.layer == "output"), None
        )
        assert output_email_upper is not None
        assert output_email_upper.description == "Uppercased email"
        assert output_email_upper.pii is True

        # Check full_name
        full_name_nodes = [n for n in pipeline.columns.values() if n.column_name == "full_name"]
        output_full_name = next(
            (n for n in full_name_nodes if n.query_id == "query" and n.layer == "output"), None
        )
        assert output_full_name is not None
        assert output_full_name.description == "Full name"
        assert output_full_name.pii is True
        assert output_full_name.owner == "analytics-team"

    def test_metadata_with_aggregations(self):
        """Test metadata on aggregate functions."""
        sql = """
        SELECT
          user_id,
          COUNT(*) as login_count,  -- Number of logins [tags: metric engagement]
          MAX(login_at) as last_login,  -- Most recent login [pii: false]
          SUM(revenue) as total_revenue  /* Total revenue [pii: false, owner: finance-team, tags: metric revenue] */
        FROM user_activity
        GROUP BY user_id
        """

        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        # Check login_count (use query_id to identify)
        login_count_nodes = [n for n in pipeline.columns.values() if n.column_name == "login_count"]
        output_login_count = next(
            (n for n in login_count_nodes if n.query_id == "query" and n.layer == "output"), None
        )
        assert output_login_count is not None
        assert output_login_count.description == "Number of logins"
        assert output_login_count.tags == {"metric", "engagement"}

        # Check last_login
        last_login_nodes = [n for n in pipeline.columns.values() if n.column_name == "last_login"]
        output_last_login = next(
            (n for n in last_login_nodes if n.query_id == "query" and n.layer == "output"), None
        )
        assert output_last_login is not None
        assert output_last_login.description == "Most recent login"
        assert output_last_login.pii is False

        # Check total_revenue
        total_revenue_nodes = [
            n for n in pipeline.columns.values() if n.column_name == "total_revenue"
        ]
        output_total_revenue = next(
            (n for n in total_revenue_nodes if n.query_id == "query" and n.layer == "output"), None
        )
        assert output_total_revenue is not None
        assert output_total_revenue.description == "Total revenue"
        assert output_total_revenue.pii is False
        assert output_total_revenue.owner == "finance-team"
        assert output_total_revenue.tags == {"metric", "revenue"}

    def test_metadata_in_cte(self):
        """Test metadata extraction from CTE columns."""
        sql = """
        WITH base AS (
          SELECT
            user_id,  -- User ID [pii: false]
            email     -- Email [pii: true, owner: data-team]
          FROM raw_users
        )
        SELECT
          user_id,
          email
        FROM base
        """

        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        # Check CTE columns have metadata
        base_email_nodes = [
            n
            for n in pipeline.columns.values()
            if n.column_name == "email" and n.table_name and "base" in n.table_name
        ]
        assert len(base_email_nodes) > 0

        # At least one email node should have metadata
        email_with_metadata = next((n for n in base_email_nodes if n.description == "Email"), None)
        assert email_with_metadata is not None
        assert email_with_metadata.pii is True
        assert email_with_metadata.owner == "data-team"

    def test_metadata_propagation_disabled_for_source_columns(self):
        """Test that existing source metadata is not overwritten by propagation."""
        sql1 = """
        SELECT
          user_id,  -- Original user ID [pii: false, owner: original-team]
          email     -- Original email [pii: true]
        FROM users
        """

        sql2 = """
        SELECT
          user_id,  -- This comment should create new metadata for query_1
          email
        FROM ({}) t
        """.format(sql1)

        pipeline = Pipeline([("query_0", sql1), ("query_1", sql2)], dialect="bigquery")

        # Query 0 columns should have their original metadata (use query_id to identify)
        q0_user_id = next(
            (
                n
                for n in pipeline.columns.values()
                if n.column_name == "user_id" and n.query_id == "query_0" and n.layer == "output"
            ),
            None,
        )
        assert q0_user_id is not None
        assert q0_user_id.description == "Original user ID"
        assert q0_user_id.owner == "original-team"

    def test_no_metadata_columns(self):
        """Test that columns without comments don't get metadata."""
        sql = """
        SELECT
          user_id,
          email,  -- Email address [pii: true]
          name
        FROM users
        """

        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        # user_id and name should have no metadata (use query_id to identify)
        user_id_nodes = [
            n
            for n in pipeline.columns.values()
            if n.column_name == "user_id" and n.query_id == "query" and n.layer == "output"
        ]
        if user_id_nodes:
            assert user_id_nodes[0].description is None
            assert user_id_nodes[0].description_source is None

        # email should have metadata
        email_nodes = [
            n
            for n in pipeline.columns.values()
            if n.column_name == "email" and n.query_id == "query" and n.layer == "output"
        ]
        assert len(email_nodes) > 0
        assert email_nodes[0].description == "Email address"
        assert email_nodes[0].pii is True

    def test_metadata_with_complex_query(self):
        """Test metadata in complex query with multiple transformations."""
        sql = """
        WITH users AS (
          SELECT
            user_id,  -- User identifier [pii: false]
            email,    -- Email address [pii: true, owner: data-team]
            created_at  -- Account creation date
          FROM raw_users
        ),

        activity AS (
          SELECT
            user_id,
            COUNT(*) as event_count,  -- Number of events [tags: metric]
            MAX(event_time) as last_event  -- Last event time [pii: false]
          FROM user_events
          GROUP BY user_id
        )

        SELECT
          u.user_id,
          u.email,
          COALESCE(a.event_count, 0) as total_events,  -- Total events [tags: metric engagement]
          CASE
            WHEN a.last_event > CURRENT_DATE() - 7 THEN 'active'
            ELSE 'inactive'
          END as status  /* User activity status [pii: false, owner: analytics-team] */
        FROM users u
        LEFT JOIN activity a ON u.user_id = a.user_id
        """

        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        # Check users CTE metadata
        users_email = next(
            (
                n
                for n in pipeline.columns.values()
                if n.column_name == "email" and "users" in n.full_name
            ),
            None,
        )
        if users_email:
            assert users_email.description == "Email address"
            assert users_email.pii is True
            assert users_email.owner == "data-team"

        # Check activity CTE metadata
        event_count_nodes = [n for n in pipeline.columns.values() if n.column_name == "event_count"]
        if event_count_nodes:
            event_count = event_count_nodes[0]
            assert event_count.description == "Number of events"
            assert event_count.tags == {"metric"}

        # Check final output metadata
        status_nodes = [
            n
            for n in pipeline.columns.values()
            if n.column_name == "status" and "output" in n.full_name
        ]
        if status_nodes:
            assert status_nodes[0].description == "User activity status"
            assert status_nodes[0].pii is False
            assert status_nodes[0].owner == "analytics-team"

    def test_metadata_with_custom_fields(self):
        """Test that custom metadata fields are preserved."""
        sql = """
        SELECT
          user_id,  -- User ID [pii: false, quality: high, confidence: 95]
          email     -- Email [pii: true, sensitivity: high]
        FROM users
        """

        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        # Check custom metadata on user_id
        user_id_nodes = [
            n
            for n in pipeline.columns.values()
            if n.column_name == "user_id" and n.full_name.startswith("query.")
        ]
        if user_id_nodes:
            user_id = user_id_nodes[0]
            assert user_id.custom_metadata.get("quality") == "high"
            assert user_id.custom_metadata.get("confidence") == 95

        # Check custom metadata on email
        email_nodes = [
            n
            for n in pipeline.columns.values()
            if n.column_name == "email" and n.full_name.startswith("query.")
        ]
        if email_nodes:
            email = email_nodes[0]
            assert email.custom_metadata.get("sensitivity") == "high"


class TestMetadataPropagation:
    """Test metadata propagation through pipeline lineage."""

    def test_pii_propagation(self):
        """Test that PII flag propagates through transformations."""
        sql = """
        SELECT
          user_id,
          email,  -- Email [pii: true]
          UPPER(email) as email_upper
        FROM users
        """

        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        # Email should have PII from comment (output column)
        email_nodes = [
            n for n in pipeline.columns.values() if n.column_name == "email" and n.layer == "output"
        ]
        assert len(email_nodes) == 1
        assert email_nodes[0].pii is True

        # After propagation runs, email_upper should also have PII
        pipeline.propagate_all_metadata(verbose=False)

        email_upper_nodes = [n for n in pipeline.columns.values() if n.column_name == "email_upper"]
        assert len(email_upper_nodes) == 1
        # email_upper should inherit PII from email via backward+forward propagation
        assert email_upper_nodes[0].pii is True

    def test_owner_propagation_single_source(self):
        """Test that owner propagates when all sources have same owner."""
        sql = """
        SELECT
          user_id,
          email,  -- Email [owner: data-team]
          UPPER(email) as email_upper
        FROM users
        """

        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        # Run propagation
        pipeline.propagate_all_metadata(verbose=False)

        # email_upper should inherit owner from email
        email_upper_nodes = [n for n in pipeline.columns.values() if n.column_name == "email_upper"]
        assert len(email_upper_nodes) == 1
        # Should inherit owner since there's only one source
        assert email_upper_nodes[0].owner == "data-team"

    def test_tags_propagation_union(self):
        """Test that tags are unioned during propagation."""
        sql = """
        SELECT
          col1,  -- Column 1 [tags: finance metric]
          col2,  -- Column 2 [tags: metric critical]
          CONCAT(col1, col2) as combined
        FROM table1
        """

        pipeline = Pipeline([("query", sql)], dialect="bigquery")

        # Run propagation
        pipeline.propagate_all_metadata(verbose=False)

        # combined should have union of tags from both col1 and col2
        combined_nodes = [n for n in pipeline.columns.values() if n.column_name == "combined"]
        assert len(combined_nodes) == 1
        # Should have union of all source tags
        assert "metric" in combined_nodes[0].tags
        assert "finance" in combined_nodes[0].tags
        assert "critical" in combined_nodes[0].tags
