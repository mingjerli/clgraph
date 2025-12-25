"""
Tests for metadata extraction from SQL comments.
"""

import sqlglot

from clgraph.metadata_parser import ColumnMetadata, MetadataExtractor


class TestColumnMetadata:
    """Test ColumnMetadata dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        metadata = ColumnMetadata()
        assert metadata.description is None
        assert metadata.pii is None
        assert metadata.owner is None
        assert metadata.tags == set()
        assert metadata.custom_metadata == {}

    def test_with_values(self):
        """Test initialization with values."""
        metadata = ColumnMetadata(
            description="Test description",
            pii=True,
            owner="data-team",
            tags={"finance", "metric"},
            custom_metadata={"quality": "high"},
        )
        assert metadata.description == "Test description"
        assert metadata.pii is True
        assert metadata.owner == "data-team"
        assert metadata.tags == {"finance", "metric"}
        assert metadata.custom_metadata == {"quality": "high"}

    def test_merge_empty(self):
        """Test merging with empty metadata."""
        m1 = ColumnMetadata(description="Test", pii=True)
        m2 = ColumnMetadata()

        result = m1.merge(m2)
        assert result.description == "Test"
        assert result.pii is True

    def test_merge_non_conflicting(self):
        """Test merging non-conflicting metadata."""
        m1 = ColumnMetadata(description="Test", pii=True)
        m2 = ColumnMetadata(owner="team-a", tags={"finance"})

        result = m1.merge(m2)
        assert result.description == "Test"
        assert result.pii is True
        assert result.owner == "team-a"
        assert result.tags == {"finance"}

    def test_merge_conflicting_first_wins(self):
        """Test that first value wins for conflicting fields."""
        m1 = ColumnMetadata(description="First", pii=True, owner="team-a")
        m2 = ColumnMetadata(description="Second", pii=False, owner="team-b")

        result = m1.merge(m2)
        assert result.description == "First"
        assert result.pii is True
        assert result.owner == "team-a"

    def test_merge_tags_union(self):
        """Test that tags are unioned when merging."""
        m1 = ColumnMetadata(tags={"finance", "metric"})
        m2 = ColumnMetadata(tags={"metric", "critical"})

        result = m1.merge(m2)
        assert result.tags == {"finance", "metric", "critical"}


class TestMetadataExtractor:
    """Test MetadataExtractor class."""

    def test_parse_description_only(self):
        """Test parsing comment with description only."""
        extractor = MetadataExtractor()
        metadata = extractor._parse_comment_metadata("Simple description")

        assert metadata["description"] == "Simple description"
        assert len(metadata) == 1

    def test_parse_empty_comment(self):
        """Test parsing empty comment."""
        extractor = MetadataExtractor()
        metadata = extractor._parse_comment_metadata("")

        assert metadata == {}

    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only comment."""
        extractor = MetadataExtractor()
        metadata = extractor._parse_comment_metadata("   \t  ")

        assert metadata == {}

    def test_parse_description_with_metadata(self):
        """Test parsing comment with description and metadata."""
        extractor = MetadataExtractor()
        metadata = extractor._parse_comment_metadata(
            "User email address [pii: true, owner: data-team]"
        )

        assert metadata["description"] == "User email address"
        assert metadata["pii"] is True
        assert metadata["owner"] == "data-team"

    def test_parse_metadata_only(self):
        """Test parsing comment with metadata but no description."""
        extractor = MetadataExtractor()
        metadata = extractor._parse_comment_metadata("[pii: true, owner: data-team]")

        assert "description" not in metadata
        assert metadata["pii"] is True
        assert metadata["owner"] == "data-team"

    def test_parse_boolean_values(self):
        """Test parsing boolean values in metadata."""
        extractor = MetadataExtractor()

        # Lowercase true/false
        metadata = extractor._parse_comment_metadata("[pii: true, public: false]")
        assert metadata["pii"] is True
        assert metadata["public"] is False

        # Mixed case
        metadata = extractor._parse_comment_metadata("[pii: True, public: False]")
        assert metadata["pii"] is True
        assert metadata["public"] is False

        # Uppercase
        metadata = extractor._parse_comment_metadata("[pii: TRUE, public: FALSE]")
        assert metadata["pii"] is True
        assert metadata["public"] is False

    def test_parse_integer_values(self):
        """Test parsing integer values in metadata."""
        extractor = MetadataExtractor()
        metadata = extractor._parse_comment_metadata("[priority: 10, level: 5]")

        assert metadata["priority"] == 10
        assert metadata["level"] == 5

    def test_parse_string_values(self):
        """Test parsing string values in metadata."""
        extractor = MetadataExtractor()
        metadata = extractor._parse_comment_metadata("[owner: data-team, category: finance]")

        assert metadata["owner"] == "data-team"
        assert metadata["category"] == "finance"

    def test_parse_tags(self):
        """Test parsing tags (space-separated values)."""
        extractor = MetadataExtractor()
        metadata = extractor._parse_comment_metadata("[tags: finance metric critical]")

        assert metadata["tags"] == {"finance", "metric", "critical"}

    def test_parse_tags_single(self):
        """Test parsing single tag."""
        extractor = MetadataExtractor()
        metadata = extractor._parse_comment_metadata("[tags: finance]")

        assert metadata["tags"] == {"finance"}

    def test_parse_mixed_metadata(self):
        """Test parsing mixed metadata types."""
        extractor = MetadataExtractor()
        metadata = extractor._parse_comment_metadata(
            "Revenue metric [pii: false, owner: finance-team, priority: 10, tags: metric critical]"
        )

        assert metadata["description"] == "Revenue metric"
        assert metadata["pii"] is False
        assert metadata["owner"] == "finance-team"
        assert metadata["priority"] == 10
        assert metadata["tags"] == {"metric", "critical"}

    def test_parse_keys_normalized_to_lowercase(self):
        """Test that keys are normalized to lowercase."""
        extractor = MetadataExtractor()
        metadata = extractor._parse_comment_metadata("[PII: true, Owner: data-team, TAGS: finance]")

        assert metadata["pii"] is True
        assert metadata["owner"] == "data-team"
        assert metadata["tags"] == {"finance"}

    def test_parse_malformed_pairs_skipped(self):
        """Test that malformed key-value pairs are skipped."""
        extractor = MetadataExtractor()
        metadata = extractor._parse_comment_metadata("[pii: true, invalid_no_colon, owner: team]")

        assert metadata["pii"] is True
        assert metadata["owner"] == "team"
        assert "invalid_no_colon" not in metadata

    def test_parse_extra_whitespace(self):
        """Test parsing with extra whitespace."""
        extractor = MetadataExtractor()
        metadata = extractor._parse_comment_metadata(
            "  Description  [ pii : true ,  owner : data-team  ]  "
        )

        assert metadata["description"] == "Description"
        assert metadata["pii"] is True
        assert metadata["owner"] == "data-team"

    def test_dict_to_metadata(self):
        """Test converting dict to ColumnMetadata."""
        extractor = MetadataExtractor()
        metadata_dict = {
            "description": "Test",
            "pii": True,
            "owner": "team-a",
            "tags": {"finance", "metric"},
            "quality": "high",
        }

        metadata = extractor._dict_to_metadata(metadata_dict)

        assert metadata.description == "Test"
        assert metadata.pii is True
        assert metadata.owner == "team-a"
        assert metadata.tags == {"finance", "metric"}
        assert metadata.custom_metadata == {"quality": "high"}

    def test_extract_from_expression_no_comments(self):
        """Test extracting from expression without comments."""
        sql = "SELECT user_id FROM users"
        parsed = sqlglot.parse_one(sql, dialect="bigquery")

        extractor = MetadataExtractor()
        user_id_expr = parsed.expressions[0]
        metadata = extractor.extract_from_expression(user_id_expr)

        assert metadata.description is None
        assert metadata.pii is None
        assert metadata.owner is None

    def test_extract_from_expression_with_comment(self):
        """Test extracting from expression with single-line comment."""
        sql = "SELECT user_id, email  -- User email [pii: true] FROM users"
        parsed = sqlglot.parse_one(sql, dialect="bigquery")

        extractor = MetadataExtractor()
        email_expr = parsed.expressions[1]
        metadata = extractor.extract_from_expression(email_expr)

        assert metadata.description == "User email"
        assert metadata.pii is True

    def test_extract_from_expression_with_multiline_comment(self):
        """Test extracting from expression with multi-line comment."""
        sql = "SELECT user_id, email  /* User email address [pii: true, owner: data-team] */ FROM users"
        parsed = sqlglot.parse_one(sql, dialect="bigquery")

        extractor = MetadataExtractor()
        email_expr = parsed.expressions[1]
        metadata = extractor.extract_from_expression(email_expr)

        assert metadata.description == "User email address"
        assert metadata.pii is True
        assert metadata.owner == "data-team"

    def test_extract_from_alias_expression(self):
        """Test extracting from aliased expression."""
        sql = "SELECT SUM(amount) as total  -- Total amount [owner: finance] FROM orders"
        parsed = sqlglot.parse_one(sql, dialect="bigquery")

        extractor = MetadataExtractor()
        total_expr = parsed.expressions[0]
        metadata = extractor.extract_from_expression(total_expr)

        assert metadata.description == "Total amount"
        assert metadata.owner == "finance"

    def test_extract_from_complex_expression(self):
        """Test extracting from complex CASE expression."""
        sql = """
        SELECT
          CASE
            WHEN status = 'active' THEN 1
            ELSE 0
          END as is_active  /* Active status [pii: false, tags: status flag] */
        FROM users
        """
        parsed = sqlglot.parse_one(sql, dialect="bigquery")

        extractor = MetadataExtractor()
        expr = parsed.expressions[0]
        metadata = extractor.extract_from_expression(expr)

        assert metadata.description == "Active status"
        assert metadata.pii is False
        assert metadata.tags == {"status", "flag"}

    def test_extract_multiple_comments_merged(self):
        """Test that multiple comments on same expression are merged."""
        # This is a theoretical test - in practice, sqlglot usually captures
        # one comment per expression, but we test the merge logic
        extractor = MetadataExtractor()

        # Simulate expression with multiple comments
        class MockExpr:
            comments = [" First description [pii: true]", " [owner: data-team, tags: finance]"]

        metadata = extractor.extract_from_expression(MockExpr())

        # First description wins
        assert metadata.description == "First description"
        assert metadata.pii is True
        assert metadata.owner == "data-team"
        assert metadata.tags == {"finance"}


class TestEndToEnd:
    """End-to-end tests with realistic SQL."""

    def test_select_with_multiple_columns(self):
        """Test extracting metadata from multiple columns."""
        sql = """
        SELECT
          user_id,  -- User identifier [pii: false]
          email,    -- Email address [pii: true, owner: data-team]
          name,     -- Full name [pii: true]
          created_at  -- Account creation timestamp
        FROM users
        """
        parsed = sqlglot.parse_one(sql, dialect="bigquery")
        extractor = MetadataExtractor()

        # user_id
        user_id_meta = extractor.extract_from_expression(parsed.expressions[0])
        assert user_id_meta.description == "User identifier"
        assert user_id_meta.pii is False

        # email
        email_meta = extractor.extract_from_expression(parsed.expressions[1])
        assert email_meta.description == "Email address"
        assert email_meta.pii is True
        assert email_meta.owner == "data-team"

        # name
        name_meta = extractor.extract_from_expression(parsed.expressions[2])
        assert name_meta.description == "Full name"
        assert name_meta.pii is True

        # created_at
        created_meta = extractor.extract_from_expression(parsed.expressions[3])
        assert created_meta.description == "Account creation timestamp"
        assert created_meta.pii is None

    def test_cte_with_metadata(self):
        """Test extracting metadata from CTE columns."""
        sql = """
        WITH base AS (
          SELECT
            user_id,  -- User ID [pii: false]
            email     -- Email [pii: true, owner: data-team]
          FROM raw.users
        )
        SELECT * FROM base
        """
        parsed = sqlglot.parse_one(sql, dialect="bigquery")
        extractor = MetadataExtractor()

        # Get CTE (use ctes property for compatibility across sqlglot versions)
        cte = parsed.ctes[0]
        cte_select = cte.this

        # user_id
        user_id_meta = extractor.extract_from_expression(cte_select.expressions[0])
        assert user_id_meta.description == "User ID"
        assert user_id_meta.pii is False

        # email
        email_meta = extractor.extract_from_expression(cte_select.expressions[1])
        assert email_meta.description == "Email"
        assert email_meta.pii is True
        assert email_meta.owner == "data-team"

    def test_aggregations_with_metadata(self):
        """Test extracting metadata from aggregate expressions."""
        sql = """
        SELECT
          user_id,
          COUNT(*) as login_count,  -- Number of logins [tags: metric engagement]
          MAX(login_at) as last_login,  -- Most recent login [pii: false]
          SUM(revenue) as total_revenue  /* Total revenue [pii: false, owner: finance-team, tags: metric revenue] */
        FROM user_activity
        GROUP BY user_id
        """
        parsed = sqlglot.parse_one(sql, dialect="bigquery")
        extractor = MetadataExtractor()

        # login_count
        login_meta = extractor.extract_from_expression(parsed.expressions[1])
        assert login_meta.description == "Number of logins"
        assert login_meta.tags == {"metric", "engagement"}

        # last_login
        last_meta = extractor.extract_from_expression(parsed.expressions[2])
        assert last_meta.description == "Most recent login"
        assert last_meta.pii is False

        # total_revenue
        revenue_meta = extractor.extract_from_expression(parsed.expressions[3])
        assert revenue_meta.description == "Total revenue"
        assert revenue_meta.pii is False
        assert revenue_meta.owner == "finance-team"
        assert revenue_meta.tags == {"metric", "revenue"}
