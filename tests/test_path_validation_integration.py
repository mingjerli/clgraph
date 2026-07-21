"""Integration tests: path validation reached through the public Pipeline API.

These exercise the factory entry points a caller actually uses. The unit tests
in test_path_validation.py cover PathValidator internals directly.
"""

import json
import logging
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

    def test_symlink_allowed_with_optin_logs_warning_once(self, tmp_path: Path, caplog):
        """`create_from_json_file` must not add its own unconditional SECURITY
        warning on top of PathValidator's -- that would double-log every time
        allow_symlinks=True is passed, symlink or not.
        """
        real = tmp_path / "real.json"
        real.write_text(json.dumps({"queries": [], "dialect": "bigquery"}))
        link = tmp_path / "link.json"
        link.symlink_to(real)

        with caplog.at_level(logging.WARNING):
            pipeline = Pipeline.from_json_file(str(link), allow_symlinks=True)

        assert pipeline is not None
        security_warnings = [r for r in caplog.records if "SECURITY" in r.message]
        assert len(security_warnings) == 1

    def test_non_symlink_with_allow_symlinks_true_logs_no_warning(self, tmp_path: Path, caplog):
        """Passing allow_symlinks=True for an ordinary (non-symlink) path must
        not trigger a SECURITY warning -- there is no symlink being followed.
        """
        real = tmp_path / "real.json"
        real.write_text(json.dumps({"queries": [], "dialect": "bigquery"}))

        with caplog.at_level(logging.WARNING):
            pipeline = Pipeline.from_json_file(str(real), allow_symlinks=True)

        assert pipeline is not None
        security_warnings = [r for r in caplog.records if "SECURITY" in r.message]
        assert security_warnings == []


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

    def test_symlinked_dir_allowed_with_optin_logs_warning_once(self, tmp_path: Path, caplog):
        """`create_from_sql_files` must not add its own unconditional SECURITY
        warning on top of PathValidator's -- that would double-log every time
        allow_symlinks=True is passed, symlink or not.
        """
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        self._write_sql(real_dir)
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir, target_is_directory=True)

        with caplog.at_level(logging.WARNING):
            pipeline = Pipeline.from_sql_files(str(link_dir), allow_symlinks=True)

        assert pipeline is not None
        security_warnings = [r for r in caplog.records if "SECURITY" in r.message]
        assert len(security_warnings) == 1

    def test_non_symlink_dir_with_allow_symlinks_true_logs_no_warning(self, tmp_path: Path, caplog):
        """Passing allow_symlinks=True for an ordinary (non-symlink) directory
        must not trigger a SECURITY warning -- there is no symlink being
        followed.
        """
        self._write_sql(tmp_path)

        with caplog.at_level(logging.WARNING):
            pipeline = Pipeline.from_sql_files(str(tmp_path), allow_symlinks=True)

        assert pipeline is not None
        security_warnings = [r for r in caplog.records if "SECURITY" in r.message]
        assert security_warnings == []


class TestFromDbtModelsPathValidation:
    def test_valid_dbt_layout_loads(self, tmp_path: Path):
        staging = tmp_path / "models" / "staging"
        staging.mkdir(parents=True)
        (staging / "stg_orders.sql").write_text("SELECT id AS order_id, amount FROM raw.raw_orders")

        pipeline = Pipeline.from_dbt_models(tmp_path, schema_map={"staging": "staging"})

        assert pipeline is not None
        assert "staging.stg_orders" in pipeline.table_graph.tables

    def test_symlinked_model_file_rejected_by_default(self, tmp_path: Path):
        staging = tmp_path / "models" / "staging"
        staging.mkdir(parents=True)
        real = staging / "real_model.sql"
        real.write_text("SELECT id AS order_id, amount FROM raw.raw_orders")
        link = staging / "linked_model.sql"
        link.symlink_to(real)

        with pytest.raises(ValueError, match="Symbolic links are not allowed"):
            Pipeline.from_dbt_models(tmp_path, schema_map={"staging": "staging"})

    def test_symlinked_model_file_allowed_with_optin(self, tmp_path: Path):
        staging = tmp_path / "models" / "staging"
        staging.mkdir(parents=True)
        real = staging / "real_model.sql"
        real.write_text("SELECT id AS order_id, amount FROM raw.raw_orders")
        link = staging / "linked_model.sql"
        link.symlink_to(real)

        pipeline = Pipeline.from_dbt_models(
            tmp_path, schema_map={"staging": "staging"}, allow_symlinks=True
        )

        assert pipeline is not None

    def test_symlink_escaping_models_dir_rejected(self, tmp_path: Path):
        staging = tmp_path / "models" / "staging"
        staging.mkdir(parents=True)
        outside = tmp_path / "outside.sql"
        outside.write_text("SELECT id AS order_id, amount FROM raw.raw_orders")
        link = staging / "escape_model.sql"
        link.symlink_to(outside)

        # Confinement is checked before the symlink check, so even opting
        # into symlinks does not allow escaping the models/ directory.
        with pytest.raises(ValueError, match="Path escapes the base directory"):
            Pipeline.from_dbt_models(
                tmp_path, schema_map={"staging": "staging"}, allow_symlinks=True
            )
