"""FALLBACK description source: distinguishable, retryable, excluded from prompts."""

import pytest

from clgraph import Pipeline
from clgraph.column import build_description_prompt, generate_description
from clgraph.models import DescriptionSource


class _FailingLLM:
    def invoke(self, _):
        raise RuntimeError("model unavailable")

    def __call__(self, value):
        return self.invoke(value)


class _OkLLM:
    def __init__(self, text="Total order amount per user."):
        self.text = text

    def invoke(self, _):
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


def _output_column(pipeline, table, name):
    col = pipeline.get_column(table, name)
    assert col is not None, f"{table}.{name} not found"
    return col


def test_fallback_is_stamped_as_fallback(pipeline):
    col = _output_column(pipeline, "staging.users", "email_norm")
    produced = generate_description(col, _FailingLLM(), pipeline)
    assert produced is False
    assert col.description  # rule-based text was written
    assert col.description_source == DescriptionSource.FALLBACK


def test_fallback_serializes_as_fallback_string(pipeline):
    # exporters emit description_source.value (export.py:113,332) — pin the string
    col = _output_column(pipeline, "staging.users", "email_norm")
    generate_description(col, _FailingLLM(), pipeline)
    assert col.description_source.value == "fallback"


def test_rerun_retries_fallback_columns(pipeline):
    pipeline.llm = _FailingLLM()
    pipeline.generate_all_descriptions(verbose=False)
    col = _output_column(pipeline, "staging.users", "email_norm")
    assert col.description_source == DescriptionSource.FALLBACK

    pipeline.llm = _OkLLM("Uppercased user email.")
    pipeline.generate_all_descriptions(verbose=False)
    assert col.description == "Uppercased user email."
    assert col.description_source == DescriptionSource.GENERATED


def test_prompt_lists_all_sources_but_hides_fallback_text(pipeline):
    target = _output_column(pipeline, "staging.users", "email_norm")
    sources = [e.from_node for e in pipeline.get_incoming_edges(target.full_name)]
    assert sources, "fixture must have lineage into email_norm"

    described = sources[0]
    described.description = "User email address"
    described.description_source = DescriptionSource.GENERATED

    prompt = build_description_prompt(target, pipeline)
    assert f"- {described.full_name}: User email address" in prompt

    described.description = "Email placeholder"  # now make it a fallback
    described.description_source = DescriptionSource.FALLBACK
    prompt = build_description_prompt(target, pipeline)
    assert f"- {described.full_name}" in prompt  # still listed by name
    assert "Email placeholder" not in prompt  # but its text is withheld
