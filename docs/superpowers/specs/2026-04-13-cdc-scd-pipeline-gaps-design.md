# CDC/SCD Pipeline: Gap Analysis & Example Notebook

**Date:** 2026-04-13 (updated 2026-04-14)
**TODO item:** B — `examples/cdc_scd_pipeline.ipynb`
**Goal:** Stress-test clgraph with a realistic CDC/SCD Type 2 pipeline, surface gaps in column-lineage capture, fix what's practical, and deliver a showcase notebook that honestly documents what works and what doesn't.

## Progress Summary

**All 10 gaps closed.** No remaining open issues.

| Closed | Fix |
|--------|-----|
| Gap 1 | Struct dot-access fallback for unresolvable table refs with `nested_path`/`access_type="struct"` (67859c7) |
| Gap 2 | Promote qualify metadata from subquery-based dedup `WHERE rn = 1` pattern (622e651) |
| Gap 3/10 | MERGE condition-gating edges with `merge_column_role='condition'` (778f918) |
| Gap 4 | Statement-scoped self-read nodes for self-referencing targets (5ef8dad) |
| Gap 5 | Verified: literal-only columns appear as terminal nodes with zero upstream edges |
| Gap 6 | Verified: `current_timestamp()` columns appear as output nodes, no incoming edges |
| Gap 7 | Tagged predicate edges for JOIN ON columns with `is_join_predicate=True` (12f2b62) |
| Gap 8 | WHERE filter lineage with `is_where_filter=True` and `where_condition` edges (bf6972d) |
| Gap 9 | Literal-bound MERGE ON predicate extraction as `merge_match_filter` edges (dfbd6b7) |

Design docs: [Gap 4](2026-04-13-gap4-self-referencing-target-design.md), [Gap 7](2026-04-13-gap7-join-predicate-columns-design.md), [Gaps 1+2+8](2026-04-14-gaps-1-2-8-design.md)

Test suites: `test_struct_dot_access.py` (Gap 1), `test_subquery_dedup_qualify.py` (Gap 2), `test_where_filter_lineage.py` (Gap 8), `test_cdc_scd_pipeline.py` (integration), `test_join_predicate_columns.py` (Gap 7)

## Background

clgraph already has MERGE parsing (`query_parser._parse_merge_statement`) that captures target, source, match columns, and matched/not-matched actions with per-column mappings. An example exists at `examples/merge_lineage.ipynb`. What is *not* yet proven is whether clgraph produces **correct and complete** column lineage for the specific MERGE patterns that appear in real-world CDC/SCD Type 2 pipelines.

The canonical real-world pattern (from [Satwanth/scd-type-2-implementation](https://github.com/Satwanth/scd-type-2-implementation), aligned with Databricks and Delta Lake docs) is a **two-step SCD2**: a MERGE that closes changed rows, followed by an INSERT that opens new versions. This design uses that pattern as the nucleus and wraps it in the surrounding layers (raw CDC envelope, dedup staging, fact table, mart) that a realistic pipeline contains.

## Pipeline Shape (Test Fixture)

Dialect: **Databricks / Delta Lake** (where Debezium-style CDC most commonly lands).

Four layers, ~6 SQL statements total:

```
raw_customer_cdc           (Debezium envelope: op, ts_ms, before STRUCT, after STRUCT)
        │
        ▼
staging_customer_latest    (flatten after.*, dedup by PK via ROW_NUMBER + QUALIFY)
        │
        ├──────────────────────────────► dim_customer (SCD2)
        │                                 ├─ Step 1: MERGE WHEN MATCHED UPDATE (close old row)
        │                                 └─ Step 2: INSERT ... LEFT JOIN (open new version)
        │
        ▼
fact_orders                (append-only; joins dim_customer with BETWEEN valid_from/valid_to)
        │
        ▼
mart_daily_revenue         (rollup by day/customer)
```

### Concrete SQL (abbreviated; full SQL lives in the notebook)

**L1 → L2: staging with CDC envelope flatten + dedup**
```sql
CREATE OR REPLACE TABLE staging_customer_latest AS
SELECT after.id, after.name, after.city, after.email, ts_ms, op
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY after.id ORDER BY ts_ms DESC) AS rn
  FROM raw_customer_cdc
  WHERE op IN ('c', 'u')   -- exclude deletes for dim
)
WHERE rn = 1;
```

**L2 → L3a: SCD2 close old row**
```sql
MERGE INTO dim_customer t
USING staging_customer_latest s ON t.id = s.id AND t.is_active = 'Y'
WHEN MATCHED AND (t.name <> s.name OR t.city <> s.city OR t.email <> s.email) THEN
  UPDATE SET t.end_time = current_timestamp(), t.is_active = 'N';
```

**L2 → L3a: SCD2 open new version**
```sql
INSERT INTO dim_customer
SELECT s.id, s.name, s.city, s.email,
       current_timestamp() AS start_time,
       TIMESTAMP '9999-12-31 00:00:00' AS end_time,
       'Y' AS is_active
FROM staging_customer_latest s
LEFT JOIN dim_customer t
  ON s.id = t.id AND t.is_active = 'Y'
WHERE t.id IS NULL OR (t.name <> s.name OR t.city <> s.city OR t.email <> s.email);
```

**L3 → fact with point-in-time join**
```sql
INSERT INTO fact_orders
SELECT o.order_id, o.customer_id, o.order_ts, o.amount,
       d.city AS customer_city_at_order
FROM raw_orders o
LEFT JOIN dim_customer d
  ON o.customer_id = d.id
 AND o.order_ts BETWEEN d.start_time AND d.end_time;
```

**Fact → mart**
```sql
CREATE OR REPLACE TABLE mart_daily_revenue AS
SELECT DATE(order_ts) AS order_date, customer_city_at_order,
       SUM(amount) AS revenue, COUNT(*) AS orders
FROM fact_orders
GROUP BY 1, 2;
```

## Gaps to Probe (with severity)

Each gap will be tested by running the fixture through `Pipeline` and inspecting the resulting column graph. Classified as:
- **P** — *Parses but lineage wrong* (silent correctness bug)
- **I** — *Parses but lineage incomplete* (missing edges)
- **F** — *Fails to parse*

| # | Gap | Where | Expected classification | Status |
|---|-----|-------|------------------------|--------|
| 1 | Struct field access on CDC envelope: `after.id` → `staging.id` | L1→L2 | **I** — sqlglot parses `after.id` as `Column(table="after", name="id")`, indistinguishable from table-qualified ref. | **Closed** (67859c7) — struct fallback emits edges with `nested_path`/`access_type="struct"` when `Column.table` doesn't resolve |
| 2 | Dedup pattern: `QUALIFY rn = 1` / `WHERE rn = 1` after `ROW_NUMBER()` | L2 | **I** — Subquery-based dedup did not propagate qualify metadata to final output. | **Closed** (622e651) — detects ranking functions in subquery + `WHERE rn = 1` outer filter, promotes qualify metadata |
| 3 | **MERGE WHEN MATCHED trigger columns** — condition `t.name <> s.name` should contribute to the lineage of `end_time` and `is_active`, not just the assigned exprs | L3a Step 1 | **P / I** (likely current behavior only records assigned exprs) | **Closed** (778f918) |
| 4 | **Self-referencing target** — Step 2 `LEFT JOIN dim_customer` on a table Step 1 just mutated. Pipeline lineage must treat the same table as both an input and output between statements | L3a | I (pipeline_lineage_builder behavior unknown) | **Closed** (5ef8dad) |
| 5 | Literal-only columns (`'Y' AS is_active`, `TIMESTAMP '9999-...'`) — should appear as terminal nodes, not be silently dropped | L3a Step 2 | Verified: terminal nodes with zero upstream edges. Minor: `is_literal` flag not set (only used for VALUES clauses). | **Closed** (verified) |
| 6 | `current_timestamp()` — function-only source (no column deps) | L3a both steps | Verified: output nodes with `node_type=expression`, no incoming edges. Works in both MERGE UPDATE and INSERT contexts. | **Closed** (verified) |
| 7 | `BETWEEN d.start_time AND d.end_time` in JOIN ON — does `fact_orders.customer_city_at_order` lineage include `d.start_time/end_time` as condition columns? | L3→fact | **P / I** (join predicate columns often omitted from column lineage) | **Closed** (12f2b62) |
| 8 | `WHERE t.id IS NULL OR (...)` as sentinel for new-vs-versioned inserts — does the NULL branch get recorded? | L3a Step 2 | **I** — WHERE clause columns not tracked in column lineage | **Closed** (bf6972d) — `where_filter` edges with `is_where_filter=True` and `where_condition` metadata |
| 9 | MERGE's `ON t.id = s.id AND t.is_active = 'Y'` with literal predicate — match_columns extraction looks at `EQ` pairs only; literal-bound predicate may be dropped | L3a Step 1 | I | **Closed** (dfbd6b7) |
| 10 | Does MERGE's close-action (UPDATE of `end_time`) show `is_active` as a **dependency** of `end_time` via the WHEN MATCHED condition? Impact analysis depends on this | L3a Step 1 | P (suspected) | **Closed** (778f918, same fix as Gap 3) |

**Probably-already-works (sanity checks, not gaps):** cross-CTE propagation (fixed by 8aaa454), column extraction for `DATE(order_ts)`, `SUM(amount)` aggregates, GROUP BY columns.

## Deliverables

Three artifacts, in order:

### 1. Test fixture + runner (`tests/test_cdc_scd_pipeline.py`)
- Fixture of the 6 SQL statements above (parametrized for Databricks dialect).
- One test per gap row in the table above, asserting the **expected** (post-fix) column lineage. Tests that exercise currently-broken behavior are marked `xfail` with a reference to the gap number — they flip to passing as fixes land.
- Runs in CI; the xfail list is the living gap backlog.

### 2. Gap fixes (scope-limited, each in its own commit)
Only fixes that are **localized and low-risk** land in this effort. Larger architectural changes (e.g., introducing a new edge type for "predicate-conditional" columns) are documented as follow-ups in `TODO.md`, not attempted here.

All fixes completed (each in its own commit):
- **Gap 1** (67859c7): Struct dot-access fallback — when `Column.table` doesn't resolve to a known table/alias/subquery in scope, emits a lineage edge with `nested_path` and `access_type="struct"` using the first base table from the dependency chain as the source. Handles recursive base table resolution for CDC-like subquery patterns.
- **Gap 2** (622e651): Subquery-based dedup promotion — detects the common pattern (ROW_NUMBER/RANK/DENSE_RANK/NTILE in subquery + `WHERE rn = 1` in outer query) and promotes qualify metadata (`is_qualify_column`, `qualify_context`, `qualify_function`) to the outer unit. Adds `ranking_window_columns` to `QueryUnit` for cross-unit metadata propagation.
- **Gap 3/10** (778f918): Extended MERGE parsing to extract column references from WHEN MATCHED conditions and emit edges with `merge_column_role='condition'`. Condition columns (e.g., `staging.name`) now appear as upstream inputs to assigned target columns (e.g., `dim_customer.end_time`).
- **Gap 4** (5ef8dad): Implemented statement-scoped table versioning with `{query_id}:self_read:{table}.{col}` naming. Self-read nodes represent the prior table state, enabling correct lineage when the same table is both input and output across pipeline statements.
- **Gap 7** (12f2b62): Emits tagged predicate edges for JOIN ON clause columns with `is_join_predicate=True`, `join_condition`, and `join_side` metadata. Supports equi-joins, range/BETWEEN, function-wrapped predicates, and multi-join chains.
- **Gap 8** (bf6972d): WHERE filter lineage — columns referenced in WHERE clauses now produce `where_filter` edges to all non-star output columns, with `is_where_filter=True` and `where_condition` metadata. Subquery columns within WHERE are excluded from the outer query's predicates. Also fixes `trace_forward` BFS to treat nodes as terminals when all outgoing targets are already visited.
- **Gap 9** (dfbd6b7): Extracts literal-bound ON predicates in MERGE (e.g., `t.is_active = 'Y'`) and emits lineage edges with `merge_match_filter` edge type and `merge_column_role="condition"`.

### 3. Showcase notebook (`examples/cdc_scd_pipeline.ipynb`)
Structure:
1. **Narrative intro** — what a CDC/SCD2 pipeline is, why it matters, what we're testing.
2. **The pipeline SQL** — the 6 statements above, with comments pointing out the interesting structures.
3. **Build + visualize** — `Pipeline(...)` → lineage graph rendered (GraphViz), table-level DAG.
4. **What clgraph captures** — concrete impact-analysis queries (e.g., "what downstream columns depend on `raw_customer_cdc.after.city`?"), showcasing all 10 resolved gaps: struct dot-access (1), dedup qualify promotion (2), MERGE condition-gating (3/10), self-referencing targets (4), literal terminals (5), function-only sources (6), JOIN predicate columns (7), WHERE filter lineage (8), and MERGE literal predicates (9).
5. **Edge semantics showcase** — demonstrate the new edge types and metadata: `access_type="struct"`, `is_qualify_column`, `merge_column_role`, `is_join_predicate`, `is_where_filter`, `merge_match_filter`.

## Acceptance Criteria

- [ ] `tests/test_cdc_scd_pipeline.py` exists, runs in CI, and has tests for every row in the gap table.
- [ ] All 10 gap tests pass (no xfails remaining).
- [ ] Every fix commit has: failing test → fix → passing test (TDD, per repo convention).
- [ ] The notebook runs end-to-end via `run_all_notebooks.py` with no errors.
- [ ] The notebook showcases all 10 resolved gaps with concrete lineage queries.
- [ ] `TODO.md` updated: item B checked off.

## Out of Scope

- Fact and mart layers are included in the fixture but we don't make them *perfect*; their role is to provide realistic downstream context for gap 7. If gap 7 isn't fixed, the fact-layer test is xfail; the notebook still renders.
- Other SCD variants (Type 1 overwrite-only, Type 3 previous-value column, Type 6 hybrid) — out of scope for this iteration.
- CDC sources other than Debezium envelope shape.
- Dialects other than Databricks/Delta.
- Performance (e.g., parse speed on large pipelines).

## Risks & Open Questions

1. **~~Risk: gap 3/10 fix has wider blast radius than expected.~~** **RESOLVED.** The fix uses `merge_column_role` to distinguish condition vs assignment edges, avoiding duplication with `match_columns`. No blast radius issues encountered.
2. **~~Open: how should "trigger columns" be represented in the column graph?~~** **RESOLVED.** Chose option (b): edges carry `merge_column_role='condition'` vs `merge_column_role='assignment'`, giving users the ability to distinguish trigger vs assignment semantics without flattening.
3. **Open: should the staging CTE preserve `op` column for dim?** CDC deletes usually produce SCD2 tombstones (`is_active='N', end_time=now()`), not hard deletes. For this iteration we filter `op IN ('c','u')` and list delete-handling as a follow-up gap. **Analysis:** The `op` column is selected into `staging_customer_latest` but never referenced by the downstream MERGE or INSERT statements — it's present but unused. The test fixture in `test_cdc_scd_pipeline.py` assumes `staging_customer_latest` already exists, so the staging CTE is not yet tested. This is a data modeling choice, not a lineage bug. Reconsider if future requirements need full-envelope preservation or delete-handling.

### Additional risks surfaced from code inspection

4. **~~Risk: gap 3/10 may already be half-plumbed.~~** **RESOLVED.** The fix (778f918) parses the stored WHEN condition into column refs and adds them as condition-gating edges with `merge_column_role='condition'`. Reused the existing column-extraction pass as recommended.

5. **~~Risk: gap 4 (self-referencing target) is load-bearing and unmitigated.~~** **RESOLVED.** Promoted to in-scope; implemented statement-scoped table versioning with self-read nodes (`{query_id}:self_read:{table}.{col}`). Design doc: `2026-04-13-gap4-self-referencing-target-design.md`. Tests in `test_cdc_scd_pipeline.py` (16 test classes).

6. **~~Risk: gap 7 is worse than the design implies.~~** **RESOLVED.** Implemented tagged predicate edges for JOIN ON clause columns with `is_join_predicate=True` metadata. Design doc: `2026-04-13-gap7-join-predicate-columns-design.md`. Tests in `test_join_predicate_columns.py` (38 test methods).

7. **~~Risk: gap 9 framing is inaccurate.~~** **RESOLVED.** Fix (dfbd6b7) extracts literal-bound ON predicates and emits `merge_match_filter` edges. Unit tests confirm behavior on `ON t.id = s.id AND t.is_active = 'Y'`.

8. **~~Risk: struct-column node naming is underspecified for test assertions.~~** **RESOLVED.** Gap 1 fix (67859c7) implements the struct fallback using the same `nested_path`/`access_type` metadata pattern. Tests in `test_struct_dot_access.py` assert `edge.nested_path == ".id"`, `edge.access_type == "struct"`, consistent with existing `test_struct_array_subscript.py` conventions.

9. **Risk: CI execution model.** **OPEN — integration gap, not correctness bug.** All 25 existing notebooks use static SQL + `Pipeline()` only (no `spark.sql()`, `dbutils`). Convention is sound. However, `run_all_notebooks.py` is **not wired into CI** — `.github/workflows/ci.yml` runs only `uv run pytest tests/`. The acceptance criterion "runs via `run_all_notebooks.py`" is not enforced. Action: add CI step `python run_all_notebooks.py --skip-llm` after CDC/SCD notebook is created.

10. **~~Open: de-duplicate with existing MERGE tests.~~** **RESOLVED.** Verified: `test_merge_statements.py` covers single-statement MERGE parsing (match_columns, action column mappings). `test_cdc_scd_pipeline.py` covers multi-statement cross-query semantics (self-reference, topological sort, cross-query edges). Zero overlap — properly scoped by design.

11. **~~Open: QUALIFY assertion target.~~** **RESOLVED.** Gap 2 fix (622e651) promotes qualify metadata from subquery-based dedup patterns. Tests in `test_subquery_dedup_qualify.py` use the same metadata assertion pattern (`is_qualify_column`, `qualify_context`, `qualify_function`) as `test_qualify_clause.py`.

12. **~~Open: should gap 4 and gap 7 graduate to their own design docs?~~** **RESOLVED.** Both now have dedicated design docs in `docs/superpowers/specs/`.
