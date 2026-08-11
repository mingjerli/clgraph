"""Terminal SELECTs get descriptions; both bulk ops visit the same columns."""

import pytest

from clgraph import Pipeline
from clgraph.metadata_manager import target_table
from clgraph.models import DescriptionSource


class _OkLLM:
    def invoke(self, _):
        class _R:
            pass

        r = _R()
        r.content = "A generated description."
        return r

    def __call__(self, value):
        return self.invoke(value)


@pytest.fixture
def pipeline_with_terminal_select():
    return Pipeline.from_dict(
        {
            "staging_users": """
                CREATE TABLE staging.users AS
                SELECT id, email FROM raw.users
            """,
            "report": """
                SELECT id, UPPER(email) AS email_norm FROM staging.users
            """,
        },
        dialect="bigquery",
    )


def test_target_table_falls_back_to_result_convention(pipeline_with_terminal_select):
    query = pipeline_with_terminal_select.table_graph.queries["report"]
    assert query.destination_table is None
    assert target_table(query) == "report_result"


def test_terminal_select_columns_get_descriptions(pipeline_with_terminal_select):
    pipeline = pipeline_with_terminal_select
    pipeline.llm = _OkLLM()
    pipeline.generate_all_descriptions(verbose=False)

    described = [
        col
        for col in pipeline.columns.values()
        if col.table_name == "report_result"
        and col.description_source == DescriptionSource.GENERATED
    ]
    assert described, "computed columns of the terminal SELECT must be described"


def test_bulk_ops_visit_identical_computed_columns(pipeline_with_terminal_select, monkeypatch):
    import clgraph.metadata_manager as mm

    pipeline = pipeline_with_terminal_select

    described = []
    monkeypatch.setattr(
        mm,
        "generate_description",
        lambda col, llm, p, **kw: described.append((col.table_name, col.column_name)),
    )
    pipeline.llm = _OkLLM()
    pipeline.generate_all_descriptions(verbose=False)

    propagated = []
    monkeypatch.setattr(
        mm,
        "propagate_metadata",
        lambda col, p: propagated.append((col.table_name, col.column_name)),
    )
    pipeline.propagate_all_metadata(verbose=False)

    # fresh pipeline: every computed column needs a description, so the sets match exactly
    assert set(described) == set(propagated)
    assert ("report_result", "email_norm") in described
