"""T3a: extract equi-join predicates from parsed queries — zero fabrication."""

from clgraph import Pipeline
from clgraph.tools import ContextBuilder


def _pairs(joins):
    return {(j["left_table"], j["left_column"], j["right_table"], j["right_column"]) for j in joins}


def test_on_equality_with_aliases_resolves_physical_tables():
    pipeline = Pipeline.from_dict(
        {
            "q": """
                CREATE TABLE mart.user_orders AS
                SELECT u.id, o.amount
                FROM raw.users u
                JOIN raw.orders o ON u.id = o.user_id
            """
        },
        dialect="bigquery",
    )
    joins = ContextBuilder(pipeline).get_observed_joins()
    assert _pairs(joins) == {("raw.users", "id", "raw.orders", "user_id")}
    assert joins[0]["query_id"] == "q"


def test_composite_key_emits_one_entry_per_pair():
    pipeline = Pipeline.from_dict(
        {
            "q": """
                CREATE TABLE m.t AS
                SELECT a.x FROM s.a a
                JOIN s.b b ON a.x = b.x AND a.y = b.y
            """
        },
        dialect="bigquery",
    )
    joins = ContextBuilder(pipeline).get_observed_joins()
    assert _pairs(joins) == {("s.a", "x", "s.b", "x"), ("s.a", "y", "s.b", "y")}


def test_using_with_single_table_left_input():
    pipeline = Pipeline.from_dict(
        {"q": "CREATE TABLE m.t AS SELECT a.id FROM s.a a JOIN s.b b USING (id)"},
        dialect="bigquery",
    )
    joins = ContextBuilder(pipeline).get_observed_joins()
    assert _pairs(joins) == {("s.a", "id", "s.b", "id")}


def test_chained_using_skips_composite_left_input():
    pipeline = Pipeline.from_dict(
        {
            "q": """
                CREATE TABLE m.t AS
                SELECT a.id FROM s.a a
                JOIN s.b b USING (id)
                JOIN s.c c USING (id)
            """
        },
        dialect="bigquery",
    )
    joins = ContextBuilder(pipeline).get_observed_joins()
    assert _pairs(joins) == {("s.a", "id", "s.b", "id")}  # nothing fabricated for c


def test_non_equi_join_emits_nothing():
    pipeline = Pipeline.from_dict(
        {"q": "CREATE TABLE m.t AS SELECT a.ts FROM s.a a JOIN s.b b ON a.ts > b.ts"},
        dialect="bigquery",
    )
    assert ContextBuilder(pipeline).get_observed_joins() == []


def test_build_join_context_formats_hints():
    pipeline = Pipeline.from_dict(
        {
            "q": """
                CREATE TABLE mart.user_orders AS
                SELECT u.id FROM raw.users u JOIN raw.orders o ON u.id = o.user_id
            """
        },
        dialect="bigquery",
    )
    ctx = ContextBuilder(pipeline).build_join_context(list(pipeline.table_graph.tables))
    assert "## Join Hints" in ctx
    assert "raw.users.id = raw.orders.user_id (observed in q)" in ctx


def test_nested_subquery_alias_collision_does_not_fabricate():
    pipeline = Pipeline.from_dict(
        {
            "q": """
                CREATE TABLE m.t AS
                SELECT a.x FROM s.a a
                JOIN s.b b ON a.x = b.x
                WHERE a.y IN (SELECT a.z FROM s.c a WHERE a.w = 1)
            """
        },
        dialect="bigquery",
    )
    joins = ContextBuilder(pipeline).get_observed_joins()
    assert _pairs(joins) == {("s.a", "x", "s.b", "x")}


def test_repeated_join_across_queries_keeps_all_provenance():
    pipeline = Pipeline.from_dict(
        {
            "q1": """
                CREATE TABLE m.t1 AS
                SELECT u.id FROM raw.users u JOIN raw.orders o ON u.id = o.user_id
            """,
            "q2": """
                CREATE TABLE m.t2 AS
                SELECT u.id FROM raw.users u JOIN raw.orders o ON u.id = o.user_id
            """,
        },
        dialect="bigquery",
    )
    builder = ContextBuilder(pipeline)
    joins = builder.get_observed_joins()
    assert len(joins) == 1
    assert joins[0]["query_ids"] == ["q1", "q2"]
    ctx = builder.build_join_context(list(pipeline.table_graph.tables))
    assert "(observed in q1, q2)" in ctx
