"""Tests for the clgraph CLI."""

import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from clgraph.cli import app

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture
def sql_dir():
    """Create a temp directory with SQL files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "staging.sql").write_text(
            "CREATE TABLE staging AS SELECT id, name, email FROM raw_users"
        )
        (Path(tmpdir) / "analytics.sql").write_text(
            "CREATE TABLE analytics AS SELECT id, name FROM staging WHERE id > 0"
        )
        yield tmpdir


@pytest.fixture
def single_sql_file():
    """Create a temp SQL file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
        f.write("""
            WITH monthly AS (
                SELECT user_id, SUM(amount) as total
                FROM orders
                GROUP BY 1
            )
            SELECT u.name, m.total
            FROM users u
            JOIN monthly m ON u.id = m.user_id
        """)
        f.flush()
        yield f.name
    os.unlink(f.name)


class TestAnalyzeCommand:
    def test_analyze_directory(self, sql_dir):
        result = runner.invoke(app, ["analyze", sql_dir])
        assert result.exit_code == 0
        assert "staging" in result.stdout
        assert "analytics" in result.stdout

    def test_analyze_single_file(self, single_sql_file):
        result = runner.invoke(app, ["analyze", single_sql_file])
        assert result.exit_code == 0
        assert "users" in result.stdout or "orders" in result.stdout

    def test_analyze_with_dialect(self, sql_dir):
        result = runner.invoke(app, ["analyze", sql_dir, "--dialect", "snowflake"])
        assert result.exit_code == 0

    def test_analyze_json_output(self, sql_dir):
        result = runner.invoke(app, ["analyze", sql_dir, "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "tables" in data
        assert "columns" in data

    def test_analyze_dot_output(self, sql_dir):
        result = runner.invoke(app, ["analyze", sql_dir, "--format", "dot"])
        assert result.exit_code == 0
        assert "digraph" in result.stdout

    def test_analyze_json_file(self, sql_dir):
        """Test analyze with a pre-built JSON pipeline file."""
        from clgraph import Pipeline

        pipeline = Pipeline.from_sql_files(sql_dir)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(pipeline.to_json(), f)
            json_path = f.name

        try:
            result = runner.invoke(app, ["analyze", json_path])
            assert result.exit_code == 0
            assert "staging" in result.stdout
        finally:
            os.unlink(json_path)

    def test_analyze_unsupported_file_type(self):
        """Test analyze with an unsupported file extension."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"col1,col2\n")
            csv_path = f.name

        try:
            result = runner.invoke(app, ["analyze", csv_path])
            assert result.exit_code != 0
            assert "unsupported" in result.output.lower()
        finally:
            os.unlink(csv_path)

    def test_analyze_invalid_format(self, sql_dir):
        result = runner.invoke(app, ["analyze", sql_dir, "--format", "xml"])
        assert result.exit_code != 0

    def test_analyze_nonexistent_path(self):
        result = runner.invoke(app, ["analyze", "/nonexistent/path"])
        assert result.exit_code != 0

    def test_analyze_shows_lineage_summary(self, sql_dir):
        result = runner.invoke(app, ["analyze", sql_dir])
        assert result.exit_code == 0
        assert "staging" in result.stdout
        assert "analytics" in result.stdout


class TestDiffCommand:
    def test_diff_two_directories(self):
        with tempfile.TemporaryDirectory() as old_dir, tempfile.TemporaryDirectory() as new_dir:
            (Path(old_dir) / "q1.sql").write_text("CREATE TABLE t1 AS SELECT id, name FROM raw")
            (Path(new_dir) / "q1.sql").write_text(
                "CREATE TABLE t1 AS SELECT id, name, email FROM raw"
            )
            result = runner.invoke(app, ["diff", old_dir, new_dir])
            assert result.exit_code == 0
            assert "email" in result.stdout.lower() or "added" in result.stdout.lower()

    def test_diff_json_output(self):
        with tempfile.TemporaryDirectory() as old_dir, tempfile.TemporaryDirectory() as new_dir:
            (Path(old_dir) / "q1.sql").write_text("CREATE TABLE t1 AS SELECT id FROM raw")
            (Path(new_dir) / "q1.sql").write_text("CREATE TABLE t1 AS SELECT id, name FROM raw")
            result = runner.invoke(app, ["diff", old_dir, new_dir, "--format", "json"])
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert "columns_added" in data
            assert "columns_removed" in data
            assert "columns_modified" in data
            assert "has_changes" in data

    def test_diff_no_changes(self):
        with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
            sql = "CREATE TABLE t1 AS SELECT id FROM raw"
            (Path(dir1) / "q1.sql").write_text(sql)
            (Path(dir2) / "q1.sql").write_text(sql)
            result = runner.invoke(app, ["diff", dir1, dir2])
            assert result.exit_code == 0
            assert "no" in result.stdout.lower()

    def test_diff_single_sql_files(self, single_sql_file):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
            f.write("""
                WITH monthly AS (
                    SELECT user_id, SUM(amount) as total, COUNT(*) as cnt
                    FROM orders
                    GROUP BY 1
                )
                SELECT u.name, m.total, m.cnt
                FROM users u
                JOIN monthly m ON u.id = m.user_id
            """)
            f.flush()
            new_file = f.name

        try:
            result = runner.invoke(app, ["diff", single_sql_file, new_file])
            assert result.exit_code == 0
        finally:
            os.unlink(new_file)

    def test_diff_invalid_format(self):
        with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
            sql = "CREATE TABLE t1 AS SELECT id FROM raw"
            (Path(dir1) / "q1.sql").write_text(sql)
            (Path(dir2) / "q1.sql").write_text(sql)
            result = runner.invoke(app, ["diff", dir1, dir2, "--format", "xml"])
            assert result.exit_code != 0

    def test_diff_nonexistent_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["diff", "/nonexistent", tmpdir])
            assert result.exit_code != 0


class TestMCPCommand:
    def test_mcp_help(self):
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0
        output = _strip_ansi(result.stdout)
        assert "--pipeline" in output
        assert "--transport" in output

    def test_mcp_nonexistent_path(self):
        result = runner.invoke(app, ["mcp", "--pipeline", "/nonexistent"])
        assert result.exit_code != 0

    def test_mcp_missing_dependency(self, sql_dir):
        with patch.dict("sys.modules", {"clgraph.mcp": None}):
            result = runner.invoke(app, ["mcp", "--pipeline", sql_dir])
            assert result.exit_code != 0
            assert "mcp" in result.output.lower()


class TestHelpOutput:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "analyze" in result.stdout
        assert "diff" in result.stdout
        assert "mcp" in result.stdout

    def test_analyze_help(self):
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        output = _strip_ansi(result.stdout)
        assert "--dialect" in output
        assert "--format" in output
