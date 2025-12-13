"""
Tests for the validation framework in clgraph.

This module tests the validation issue detection and reporting system
that helps users identify SQL quality issues preventing proper column lineage.
"""

from clgraph import IssueCategory, IssueSeverity, Pipeline, ValidationIssue


class TestValidationIssueModel:
    """Test the ValidationIssue data model."""

    def test_create_issue_minimal(self):
        """Test creating a validation issue with minimal fields."""
        issue = ValidationIssue(
            severity=IssueSeverity.ERROR,
            category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES,
            message="Test error message",
        )
        assert issue.severity == IssueSeverity.ERROR
        assert issue.category == IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES
        assert issue.message == "Test error message"
        assert issue.query_id is None
        assert issue.location is None
        assert issue.suggestion is None
        assert issue.context == {}

    def test_create_issue_full(self):
        """Test creating a validation issue with all fields."""
        issue = ValidationIssue(
            severity=IssueSeverity.WARNING,
            category=IssueCategory.AMBIGUOUS_COLUMN,
            message="Ambiguous column reference",
            query_id="query1",
            location="SELECT clause",
            suggestion="Qualify column with table name",
            context={"column": "id", "tables": ["users", "orders"]},
        )
        assert issue.severity == IssueSeverity.WARNING
        assert issue.category == IssueCategory.AMBIGUOUS_COLUMN
        assert issue.message == "Ambiguous column reference"
        assert issue.query_id == "query1"
        assert issue.location == "SELECT clause"
        assert issue.suggestion == "Qualify column with table name"
        assert issue.context["column"] == "id"
        assert len(issue.context["tables"]) == 2

    def test_issue_string_representation(self):
        """Test string representation of validation issues."""
        issue = ValidationIssue(
            severity=IssueSeverity.ERROR,
            category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES,
            message="Cannot determine column sources",
            query_id="bad_query",
            location="SELECT clause",
        )
        issue_str = str(issue)
        assert "ERROR" in issue_str
        assert "bad_query" in issue_str
        assert "Cannot determine column sources" in issue_str


class TestUnqualifiedStarDetection:
    """Test detection of unqualified SELECT * with multiple tables."""

    def test_unqualified_star_two_tables(self):
        """Test error detection for unqualified * with 2 tables."""
        queries = [
            (
                "bad_query",
                """
            CREATE TABLE result AS
            SELECT *
            FROM users, orders
            WHERE users.id = orders.user_id
            """,
            )
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        issues = pipeline.get_issues(severity=IssueSeverity.ERROR)
        assert len(issues) >= 1

        star_issue = next(
            (i for i in issues if i.category == IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES),
            None,
        )
        assert star_issue is not None
        assert "users" in star_issue.message and "orders" in star_issue.message
        assert star_issue.suggestion is not None
        assert "qualified star" in star_issue.suggestion.lower()

    def test_unqualified_star_three_tables(self):
        """Test error detection for unqualified * with 3+ tables."""
        queries = [
            (
                "bad_query",
                """
            CREATE TABLE result AS
            SELECT *
            FROM users, orders, products
            WHERE users.id = orders.user_id
            AND orders.product_id = products.id
            """,
            )
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        issues = pipeline.get_issues(category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES)
        assert len(issues) >= 1

        issue = issues[0]
        assert issue.severity == IssueSeverity.ERROR
        assert issue.context.get("table_count") == 3
        assert len(issue.context.get("tables", [])) == 3

    def test_qualified_star_no_error(self):
        """Test that qualified star with multiple tables doesn't trigger error."""
        queries = [
            (
                "good_query",
                """
            CREATE TABLE result AS
            SELECT users.*, orders.order_id, orders.amount
            FROM users, orders
            WHERE users.id = orders.user_id
            """,
            )
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        star_issues = pipeline.get_issues(category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES)
        assert len(star_issues) == 0

    def test_single_table_star_no_error(self):
        """Test that SELECT * from single table doesn't trigger error."""
        queries = [
            (
                "good_query",
                """
            CREATE TABLE result AS
            SELECT * FROM users WHERE active = true
            """,
            )
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        star_issues = pipeline.get_issues(category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES)
        assert len(star_issues) == 0


class TestExternalTableStarDetection:
    """Test detection of SELECT * from external tables without schema."""

    def test_star_from_external_table_info(self):
        """Test INFO issue for SELECT * from unknown external table."""
        queries = [
            (
                "external_query",
                """
            CREATE TABLE analytics.summary AS
            SELECT * FROM external_source.data
            """,
            )
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        issues = pipeline.get_issues(
            severity=IssueSeverity.INFO, category=IssueCategory.STAR_WITHOUT_SCHEMA
        )
        assert len(issues) >= 1

        issue = issues[0]
        # Table name is normalized, so it might be just 'data' instead of full qualified name
        assert "data" in issue.message.lower()
        assert issue.suggestion is not None

    def test_star_from_known_upstream_table_no_info(self):
        """Test that SELECT * from known upstream table doesn't trigger INFO."""
        queries = [
            (
                "query1",
                """
            CREATE TABLE users_clean AS
            SELECT id, name, email FROM raw.users
            """,
            ),
            (
                "query2",
                """
            CREATE TABLE users_final AS
            SELECT * FROM users_clean
            """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Query 2 should not have STAR_WITHOUT_SCHEMA issue because
        # users_clean is created by query1 with known columns
        query2_issues = pipeline.get_issues(
            query_id="query2", category=IssueCategory.STAR_WITHOUT_SCHEMA
        )
        assert len(query2_issues) == 0


class TestPipelineValidationAPI:
    """Test the Pipeline validation API methods."""

    def test_get_all_issues(self):
        """Test getting all issues from pipeline."""
        queries = [
            (
                "bad_query1",
                """
            CREATE TABLE result1 AS
            SELECT * FROM users, orders
            """,
            ),
            (
                "bad_query2",
                """
            CREATE TABLE result2 AS
            SELECT * FROM external.unknown_table
            """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        all_issues = pipeline.get_all_issues()
        assert len(all_issues) >= 2  # At least one ERROR and one INFO

    def test_get_issues_by_severity(self):
        """Test filtering issues by severity."""
        queries = [
            (
                "bad_query",
                """
            CREATE TABLE result AS
            SELECT * FROM users, orders
            """,
            ),
            (
                "external_query",
                """
            CREATE TABLE summary AS
            SELECT * FROM external.data
            """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        errors = pipeline.get_issues(severity=IssueSeverity.ERROR)
        infos = pipeline.get_issues(severity=IssueSeverity.INFO)

        assert len(errors) >= 1
        assert len(infos) >= 1
        assert all(i.severity == IssueSeverity.ERROR for i in errors)
        assert all(i.severity == IssueSeverity.INFO for i in infos)

    def test_get_issues_by_category(self):
        """Test filtering issues by category."""
        queries = [
            (
                "bad_query",
                """
            CREATE TABLE result AS
            SELECT * FROM users, orders
            """,
            )
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        star_issues = pipeline.get_issues(category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES)
        assert len(star_issues) >= 1
        assert all(
            i.category == IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES for i in star_issues
        )

    def test_get_issues_by_query_id(self):
        """Test filtering issues by query_id."""
        queries = [
            (
                "query1",
                """
            CREATE TABLE result1 AS
            SELECT * FROM users, orders
            """,
            ),
            (
                "query2",
                """
            CREATE TABLE result2 AS
            SELECT id, name FROM users
            """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        query1_issues = pipeline.get_issues(query_id="query1")
        query2_issues = pipeline.get_issues(query_id="query2")

        # query1 should have issues, query2 should not
        assert len(query1_issues) >= 1
        # query2 might have 0 issues since it's valid SQL
        assert len(query2_issues) == 0

    def test_get_issues_multiple_filters(self):
        """Test filtering with multiple criteria."""
        queries = [
            (
                "bad_query",
                """
            CREATE TABLE result AS
            SELECT * FROM users, orders
            """,
            )
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        filtered = pipeline.get_issues(
            severity=IssueSeverity.ERROR,
            category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES,
            query_id="bad_query",
        )
        assert len(filtered) >= 1
        assert all(
            i.severity == IssueSeverity.ERROR
            and i.category == IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES
            and i.query_id == "bad_query"
            for i in filtered
        )

    def test_has_errors(self):
        """Test has_errors() method."""
        # Pipeline with errors
        queries_bad = [("bad", "CREATE TABLE r AS SELECT * FROM t1, t2")]
        pipeline_bad = Pipeline(queries_bad, dialect="bigquery")
        assert pipeline_bad.has_errors() is True

        # Pipeline without errors
        queries_good = [("good", "CREATE TABLE r AS SELECT id, name FROM users")]
        pipeline_good = Pipeline(queries_good, dialect="bigquery")
        assert pipeline_good.has_errors() is False

    def test_has_warnings(self):
        """Test has_warnings() method."""
        # For now, we don't have WARNING-level issues implemented,
        # but test the API works
        queries = [("query", "CREATE TABLE r AS SELECT id FROM users")]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Should not crash
        result = pipeline.has_warnings()
        assert isinstance(result, bool)


class TestValidationReporting:
    """Test validation reporting and output formatting."""

    def test_print_issues_no_crash(self, capsys):
        """Test that print_issues() doesn't crash."""
        queries = [
            (
                "bad",
                """
            CREATE TABLE result AS
            SELECT * FROM users, orders
            """,
            )
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Should not crash
        pipeline.print_issues()

        captured = capsys.readouterr()
        assert "ERROR" in captured.out or "❌" in captured.out

    def test_print_issues_severity_filter(self, capsys):
        """Test printing issues filtered by severity."""
        queries = [
            ("bad", "CREATE TABLE r AS SELECT * FROM t1, t2"),
            ("ext", "CREATE TABLE s AS SELECT * FROM external.data"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Print only errors
        pipeline.print_issues(severity=IssueSeverity.ERROR)

        captured = capsys.readouterr()
        # Should show ERROR but not INFO
        assert "ERROR" in captured.out or "❌" in captured.out


class TestValidationIntegration:
    """Integration tests for validation with real SQL scenarios."""

    def test_multi_query_pipeline_with_issues(self):
        """Test validation in a multi-query pipeline."""
        queries = [
            (
                "extract",
                """
            CREATE TABLE users_clean AS
            SELECT id, name, email FROM raw.users
            """,
            ),
            (
                "bad_join",
                """
            CREATE TABLE user_orders AS
            SELECT * FROM users_clean, orders
            WHERE users_clean.id = orders.user_id
            """,
            ),
            (
                "aggregate",
                """
            CREATE TABLE summary AS
            SELECT user_id, COUNT(*) as order_count
            FROM user_orders
            GROUP BY user_id
            """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Should have error in bad_join query
        all_issues = pipeline.get_all_issues()
        assert len(all_issues) >= 1

        bad_join_issues = pipeline.get_issues(query_id="bad_join")
        assert len(bad_join_issues) >= 1

    def test_clean_pipeline_no_issues(self):
        """Test that a clean pipeline has no validation issues."""
        queries = [
            (
                "extract",
                """
            CREATE TABLE users_clean AS
            SELECT id, name, email FROM raw.users
            """,
            ),
            (
                "transform",
                """
            CREATE TABLE users_final AS
            SELECT id, UPPER(name) as name, email
            FROM users_clean
            WHERE email IS NOT NULL
            """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Should have no errors
        assert pipeline.has_errors() is False
        errors = pipeline.get_issues(severity=IssueSeverity.ERROR)
        assert len(errors) == 0

    def test_star_expansion_with_validation(self):
        """Test that star expansion works alongside validation."""
        queries = [
            (
                "query1",
                """
            CREATE TABLE users_clean AS
            SELECT id, name, email FROM raw.users
            """,
            ),
            (
                "query2",
                """
            CREATE TABLE users_final AS
            SELECT * FROM users_clean
            """,
            ),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Should have no errors (star is from single table in known schema)
        assert pipeline.has_errors() is False

        # Should have star expansion working
        query2_columns = [
            col
            for col in pipeline.columns.values()
            if col.query_id == "query2" and col.layer == "output"
        ]
        # Should have expanded columns, not a star
        non_star_cols = [c for c in query2_columns if not c.is_star]
        assert len(non_star_cols) >= 3  # id, name, email


class TestIssueSeverityLevels:
    """Test different severity levels are assigned correctly."""

    def test_error_severity_for_ambiguous_star(self):
        """Test ERROR severity for unqualified star with multiple tables."""
        queries = [("bad", "CREATE TABLE r AS SELECT * FROM t1, t2")]
        pipeline = Pipeline(queries, dialect="bigquery")

        issues = pipeline.get_issues(category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES)
        assert len(issues) >= 1
        assert issues[0].severity == IssueSeverity.ERROR

    def test_info_severity_for_external_star(self):
        """Test INFO severity for star from external table."""
        queries = [("ext", "CREATE TABLE s AS SELECT * FROM external.unknown")]
        pipeline = Pipeline(queries, dialect="bigquery")

        issues = pipeline.get_issues(category=IssueCategory.STAR_WITHOUT_SCHEMA)
        assert len(issues) >= 1
        assert issues[0].severity == IssueSeverity.INFO


class TestIssueSuggestions:
    """Test that validation issues include helpful suggestions."""

    def test_unqualified_star_has_suggestion(self):
        """Test suggestion for unqualified star issue."""
        queries = [("bad", "CREATE TABLE r AS SELECT * FROM users, orders")]
        pipeline = Pipeline(queries, dialect="bigquery")

        issues = pipeline.get_issues(category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES)
        assert len(issues) >= 1
        assert issues[0].suggestion is not None
        assert "qualified" in issues[0].suggestion.lower()

    def test_external_star_has_suggestion(self):
        """Test suggestion for external star issue."""
        queries = [("ext", "CREATE TABLE s AS SELECT * FROM external.data")]
        pipeline = Pipeline(queries, dialect="bigquery")

        issues = pipeline.get_issues(category=IssueCategory.STAR_WITHOUT_SCHEMA)
        assert len(issues) >= 1
        assert issues[0].suggestion is not None


class TestIssueContext:
    """Test that validation issues include useful context."""

    def test_star_issue_includes_table_list(self):
        """Test context includes table names for star issues."""
        queries = [("bad", "CREATE TABLE r AS SELECT * FROM users, orders, products")]
        pipeline = Pipeline(queries, dialect="bigquery")

        issues = pipeline.get_issues(category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES)
        assert len(issues) >= 1

        context = issues[0].context
        assert "tables" in context
        assert "table_count" in context
        assert context["table_count"] == 3
        assert len(context["tables"]) == 3
