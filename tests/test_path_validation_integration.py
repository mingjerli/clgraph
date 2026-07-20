"""Integration tests: path validation reached through the public Pipeline API.

These exercise the factory entry points a caller actually uses. The unit tests
in test_path_validation.py cover PathValidator internals directly.
"""

import json
from pathlib import Path

import pytest

from clgraph import Pipeline


class TestFromJsonFilePathValidation:
    def test_non_json_extension_rejected(self, tmp_path: Path):
        bad = tmp_path / "data.txt"
        bad.write_text("{}")
        with pytest.raises(ValueError, match="Invalid file extension"):
            Pipeline.from_json_file(str(bad))

    def test_symlink_rejected_by_default(self, tmp_path: Path):
        real = tmp_path / "real.json"
        real.write_text(json.dumps({"columns": [], "edges": []}))
        link = tmp_path / "link.json"
        link.symlink_to(real)
        with pytest.raises(ValueError, match="Symbolic links are not allowed"):
            Pipeline.from_json_file(str(link))

    def test_missing_file_raises_filenotfound(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            Pipeline.from_json_file(str(tmp_path / "nope.json"))


class TestFromSqlFilesPathValidation:
    def _write_sql(self, d: Path, name: str = "q.sql") -> None:
        (d / name).write_text("SELECT 1 AS a")

    def test_valid_directory_loads(self, tmp_path: Path):
        self._write_sql(tmp_path)
        pipeline = Pipeline.from_sql_files(str(tmp_path), dialect="bigquery")
        assert pipeline is not None

    def test_traversal_pattern_rejected(self, tmp_path: Path):
        self._write_sql(tmp_path)
        with pytest.raises(
            ValueError, match="Glob pattern must not contain directory traversal components"
        ):
            Pipeline.from_sql_files(str(tmp_path), pattern="../*.sql")

    def test_symlinked_dir_rejected_by_default(self, tmp_path: Path):
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        self._write_sql(real_dir)
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir, target_is_directory=True)
        with pytest.raises(ValueError, match="Symbolic links are not allowed"):
            Pipeline.from_sql_files(str(link_dir))

    def test_symlinked_dir_allowed_with_optin(self, tmp_path: Path):
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        self._write_sql(real_dir)
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir, target_is_directory=True)
        pipeline = Pipeline.from_sql_files(str(link_dir), allow_symlinks=True)
        assert pipeline is not None
