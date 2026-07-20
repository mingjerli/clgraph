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
        with pytest.raises(ValueError):
            Pipeline.from_json_file(str(bad))

    def test_symlink_rejected_by_default(self, tmp_path: Path):
        real = tmp_path / "real.json"
        real.write_text(json.dumps({"columns": [], "edges": []}))
        link = tmp_path / "link.json"
        link.symlink_to(real)
        with pytest.raises(ValueError):
            Pipeline.from_json_file(str(link))

    def test_missing_file_raises_filenotfound(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            Pipeline.from_json_file(str(tmp_path / "nope.json"))
