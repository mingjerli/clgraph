"""End-to-end integration tests against the bundled Jaffle Shop dbt example."""

from __future__ import annotations

import pathlib

import pytest

from clgraph import Pipeline, wrap_dbt_models

JAFFLE_DIR = pathlib.Path(__file__).resolve().parents[1] / "examples" / "jaffle_shop"


@pytest.mark.skipif(not JAFFLE_DIR.exists(), reason="jaffle_shop example missing")
def test_wrap_dbt_models_returns_three_tuples():
    queries = wrap_dbt_models(JAFFLE_DIR)
    assert len(queries) == 12
    assert all(len(q) == 3 for q in queries)
    target_tables = {q[2] for q in queries}
    assert "staging.stg_customers" in target_tables
    assert "marts.orders" in target_tables
    # Ordering: staging rows come before marts rows
    staging_count_before_first_mart = 0
    for _, _, tgt in queries:
        if tgt.startswith("marts."):
            break
        if tgt.startswith("staging."):
            staging_count_before_first_mart += 1
    assert staging_count_before_first_mart == 6


@pytest.mark.skipif(not JAFFLE_DIR.exists(), reason="jaffle_shop example missing")
def test_from_dbt_models_with_template_context():
    def source(src, tbl):
        return f"raw.{tbl}"

    def ref(model):
        return f"staging.{model}" if model.startswith("stg_") else f"marts.{model}"

    def cents_to_dollars(col):
        return f"({col} / 100)"

    class _Dbt:
        def date_trunc(self, g, c):
            return f"DATE_TRUNC({c}, {g.upper()})"

    class _DbtUtils:
        def generate_surrogate_key(self, cols):
            return f"MD5({','.join(cols)})"

    pipeline = Pipeline.from_dbt_models(
        JAFFLE_DIR,
        dialect="bigquery",
        template_context={
            "source": source,
            "ref": ref,
            "cents_to_dollars": cents_to_dollars,
            "dbt": _Dbt(),
            "dbt_utils": _DbtUtils(),
        },
    )
    raw_sources = [t for t in pipeline.table_graph.get_source_tables() if str(t).startswith("raw.")]
    assert len(raw_sources) == 6
    # All 12 dbt models should be represented in the table graph
    assert "staging.stg_customers" in pipeline.table_graph.tables
    assert "marts.customers" in pipeline.table_graph.tables
    # Staging tables should have explicit columns resolved (not just '*'),
    # because stg_orders lists columns explicitly after its `SELECT * FROM source` CTE.
    stg_orders_cols = {
        c.column_name
        for c in pipeline.columns.values()
        if c.table_name == "staging.stg_orders" and not c.is_star
    }
    assert {"order_id", "customer_id", "order_total"}.issubset(stg_orders_cols)


def test_from_dbt_models_synthetic_dbt_style_chain(tmp_path):
    """End-to-end: a mini dbt-like project without Jinja templates should
    produce full column lineage from marts back to raw via the new CTE
    cross-query edges and 3-tuple wrapping."""
    models = tmp_path / "models"
    staging = models / "staging"
    marts = models / "marts"
    staging.mkdir(parents=True)
    marts.mkdir(parents=True)

    (staging / "stg_orders.sql").write_text("SELECT id AS order_id, amount FROM raw.raw_orders")
    (marts / "orders.sql").write_text(
        """
        WITH stg AS (SELECT * FROM staging.stg_orders)
        SELECT order_id, amount FROM stg
        """
    )

    pipeline = Pipeline.from_dbt_models(tmp_path, dialect="bigquery")
    sources = pipeline.trace_column_backward("marts.orders", "amount")
    source_names = {(s.table_name, s.column_name) for s in sources}
    assert ("raw.raw_orders", "amount") in source_names
