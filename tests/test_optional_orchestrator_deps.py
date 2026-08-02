"""Importing clgraph must not require optional orchestrator dependencies.

Regression tests for a bug shipped in 0.0.5: ``clgraph.orchestrators.kestra``
did ``import yaml`` at module scope, and ``orchestrators/__init__.py`` imports
every backend eagerly, so a bare ``pip install clgraph`` followed by
``import clgraph`` raised ``ModuleNotFoundError: No module named 'yaml'``.
PyYAML is not a declared dependency; it only ever arrived transitively in
development environments, which is why the whole test suite stayed green.

These tests run in a subprocess with ``yaml`` made unimportable, because the
development environment genuinely has PyYAML installed - the failure only
reproduces when the module is absent.
"""

import subprocess
import sys

import pytest

# Installed ahead of every import so `yaml` is unavailable even to modules that
# have not been loaded yet. A meta_path finder is used rather than deleting
# sys.modules entries, which a later import would simply repopulate.
_BLOCK_YAML = """
import sys


class _BlockYaml:
    def find_spec(self, name, path=None, target=None):
        if name == "yaml" or name.startswith("yaml."):
            raise ImportError("No module named 'yaml'")
        return None


sys.meta_path.insert(0, _BlockYaml())
for _mod in [m for m in list(sys.modules) if m == "yaml" or m.startswith("yaml.")]:
    del sys.modules[_mod]
"""


def _run_without_yaml(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_YAML + code],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_yaml_blocker_actually_blocks():
    """Guard the guard: if the blocker stopped working these tests would pass
    vacuously against a dev environment that has PyYAML installed."""
    result = _run_without_yaml("import yaml")
    assert result.returncode != 0
    assert "No module named 'yaml'" in result.stderr


def test_import_clgraph_without_pyyaml():
    result = _run_without_yaml("import clgraph; print(clgraph.__version__)")
    assert result.returncode == 0, f"importing clgraph needs PyYAML:\n{result.stderr}"


def test_import_orchestrators_package_without_pyyaml():
    result = _run_without_yaml(
        "from clgraph.orchestrators import AirflowOrchestrator, DagsterOrchestrator; print('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_kestra_orchestrator_is_importable_without_pyyaml():
    """The class must import; only *using* it should need PyYAML."""
    result = _run_without_yaml(
        "from clgraph.orchestrators import KestraOrchestrator; print(KestraOrchestrator.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert "KestraOrchestrator" in result.stdout


@pytest.mark.parametrize(
    "method_call",
    [
        'k.to_flow(flow_id="f", namespace="n")',
        'k.to_flow_with_triggers(flow_id="f", namespace="n", cron="0 0 * * *")',
        "k.to_flow_dict(flow_id='f', namespace='n')",
    ],
)
def test_using_kestra_without_pyyaml_raises_actionable_error(method_call):
    """A missing optional dependency must name itself and how to install it,
    rather than surfacing a bare ModuleNotFoundError from deep in the stack."""
    result = _run_without_yaml(f"""
from clgraph import Pipeline
from clgraph.orchestrators import KestraOrchestrator

p = Pipeline([("q", "CREATE TABLE mart_orders AS SELECT id FROM raw_orders")], dialect="bigquery")
k = KestraOrchestrator(p)
try:
    {method_call}
except ImportError as exc:
    print("RAISED:", exc)
else:
    print("NO ERROR RAISED")
""")
    assert result.returncode == 0, result.stderr
    assert "RAISED:" in result.stdout, result.stdout
    message = result.stdout.split("RAISED:", 1)[1].lower()
    assert "pyyaml" in message
    assert "kestra" in message


def test_pipeline_works_end_to_end_without_pyyaml():
    """The core product - lineage - must not be collateral damage."""
    result = _run_without_yaml("""
from clgraph import Pipeline

p = Pipeline([("q", "CREATE TABLE mart_orders AS SELECT id, amount FROM raw_orders")], dialect="bigquery")
print("COLUMNS:", len(p.columns))
""")
    assert result.returncode == 0, result.stderr
    assert "COLUMNS:" in result.stdout
    assert int(result.stdout.split("COLUMNS:")[1].strip()) > 0
