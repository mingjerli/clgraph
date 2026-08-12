"""T2: BFS expansion with configurable depth, self-reference safety, ordering."""

import pytest

from clgraph import Pipeline
from clgraph.tools import ContextBuilder, ContextConfig


@pytest.fixture
def chain_pipeline():
    return Pipeline.from_dict(
        {
            "q_b": "CREATE TABLE b AS SELECT id FROM a",
            "q_c": "CREATE TABLE c AS SELECT id FROM b",
        },
        dialect="bigquery",
    )


def test_depth_one_matches_old_behavior(chain_pipeline):
    builder = ContextBuilder(chain_pipeline)
    assert set(builder.expand_with_lineage(["c"], depth=1)) == {"c", "b"}


def test_depth_two_reaches_grandparent(chain_pipeline):
    builder = ContextBuilder(chain_pipeline)
    assert set(builder.expand_with_lineage(["c"], depth=2)) == {"c", "b", "a"}


def test_default_depth_comes_from_config(chain_pipeline):
    builder = ContextBuilder(chain_pipeline, ContextConfig(lineage_expansion_depth=1))
    assert set(builder.expand_with_lineage(["c"])) == {"c", "b"}


def test_result_is_ordered_shallow_first(chain_pipeline):
    builder = ContextBuilder(chain_pipeline)
    assert builder.expand_with_lineage(["c"], depth=2) == ["c", "b", "a"]


def test_self_referencing_query_terminates():
    pipeline = Pipeline.from_dict({"q_t": "INSERT INTO t SELECT id FROM t"}, dialect="bigquery")
    builder = ContextBuilder(pipeline)
    result = builder.expand_with_lineage(["t"], depth=5)
    assert result.count("t") == 1


def test_include_lineage_false_disables_expansion(chain_pipeline):
    builder = ContextBuilder(chain_pipeline, ContextConfig(include_lineage=False))
    assert builder.expand_with_lineage(["c"], depth=3) == ["c"]
