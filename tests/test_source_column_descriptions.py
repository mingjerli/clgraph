"""D1: describe source-table columns from forward usage context."""

import pytest

import clgraph.column as column_mod
from clgraph import Pipeline
from clgraph.column import build_source_description_prompt
from clgraph.models import DescriptionSource


class _OkLLM:
    def __init__(self, text="Raw user record identifier."):
        self.text = text
        self.calls = 0

    def invoke(self, _):
        self.calls += 1

        class _R:
            pass

        r = _R()
        r.content = self.text
        return r

    def __call__(self, value):
        return self.invoke(value)


@pytest.fixture
def pipeline():
    return Pipeline.from_dict(
        {
            "staging_users": """
                CREATE TABLE staging.users AS
                SELECT id, UPPER(email) AS email_norm FROM raw.users
            """,
        },
        dialect="bigquery",
    )


def _source_nodes(pipeline, table, name):
    return [c for c in pipeline.columns.values() if c.table_name == table and c.column_name == name]


def test_source_prompt_contains_forward_usage_and_siblings(pipeline):
    email_nodes = _source_nodes(pipeline, "raw.users", "email")
    assert email_nodes, "fixture must expose raw.users.email as an input node"
    prompt = build_source_description_prompt(email_nodes[0], pipeline)
    assert "raw.users" in prompt
    assert "Used downstream as:" in prompt
    assert "email_norm" in prompt  # forward usage
    assert "Sibling columns:" in prompt
    assert "id" in prompt  # sibling


def test_include_sources_describes_source_columns(pipeline):
    pipeline.llm = _OkLLM()
    pipeline.generate_all_descriptions(verbose=False, include_sources=True)
    for node in _source_nodes(pipeline, "raw.users", "email"):
        assert node.description == "Raw user record identifier."
        assert node.description_source == DescriptionSource.GENERATED


def test_default_excludes_source_columns(pipeline):
    pipeline.llm = _OkLLM()
    pipeline.generate_all_descriptions(verbose=False)
    for node in _source_nodes(pipeline, "raw.users", "email"):
        assert not node.description


def test_dispatch_routes_source_columns_to_source_builder(pipeline, monkeypatch):
    calls = {"source": 0, "computed": 0}
    real_source = column_mod.build_source_description_prompt
    real_computed = column_mod.build_description_prompt

    def spy_source(col, pipe):
        calls["source"] += 1
        return real_source(col, pipe)

    def spy_computed(col, pipe):
        calls["computed"] += 1
        return real_computed(col, pipe)

    monkeypatch.setattr(column_mod, "build_source_description_prompt", spy_source)
    monkeypatch.setattr(column_mod, "build_description_prompt", spy_computed)

    email = _source_nodes(pipeline, "raw.users", "email")[0]
    assert email.is_computed() is True  # the trap the dispatch must avoid
    column_mod.generate_description(email, _OkLLM(), pipeline)
    assert calls == {"source": 1, "computed": 0}

    norm = pipeline.get_column("staging.users", "email_norm")
    column_mod.generate_description(norm, _OkLLM(), pipeline)
    assert calls == {"source": 1, "computed": 1}


def test_twin_copy_respects_overwrite_semantics(pipeline):
    from clgraph.models import ColumnNode, DescriptionSource

    twin = ColumnNode(
        column_name="email",
        table_name="raw.users",
        full_name="twin:raw.users.email",
        expression="email",
    )
    twin.description = "Existing generated text"
    twin.description_source = DescriptionSource.GENERATED
    pipeline.columns["twin:raw.users.email"] = twin

    pipeline.llm = _OkLLM("Fresh model text.")
    pipeline.generate_all_descriptions(verbose=False, include_sources=True)
    # twin already had an adequate description -> untouched without overwrite
    assert twin.description == "Existing generated text"

    pipeline.generate_all_descriptions(verbose=False, include_sources=True, overwrite=True)
    assert twin.description == "Fresh model text."
