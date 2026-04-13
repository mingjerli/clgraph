# CDC/SCD Pipeline: Gap Analysis & Example Notebook

**Date:** 2026-04-13
**TODO item:** B — `examples/cdc_scd_pipeline.ipynb`
**Goal:** Stress-test clgraph with a realistic CDC/SCD Type 2 pipeline, surface gaps in column-lineage capture, fix what's practical, and deliver a showcase notebook that honestly documents what works and what doesn't.

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

| # | Gap | Where | Expected classification |
|---|-----|-------|------------------------|
| 1 | Struct field access on CDC envelope: `after.id` → `staging.id` | L1→L2 | I (may work via existing struct support, needs verification) |
| 2 | Dedup pattern: `QUALIFY rn = 1` / `WHERE rn = 1` after `ROW_NUMBER()` | L2 | Likely works (`qualify_lineage` exists); verify not dropped through CTE |
| 3 | **MERGE WHEN MATCHED trigger columns** — condition `t.name <> s.name` should contribute to the lineage of `end_time` and `is_active`, not just the assigned exprs | L3a Step 1 | **P / I** (likely current behavior only records assigned exprs) |
| 4 | **Self-referencing target** — Step 2 `LEFT JOIN dim_customer` on a table Step 1 just mutated. Pipeline lineage must treat the same table as both an input and output between statements | L3a | I (pipeline_lineage_builder behavior unknown) |
| 5 | Literal-only columns (`'Y' AS is_active`, `TIMESTAMP '9999-...'`) — should appear as terminal nodes, not be silently dropped | L3a Step 2 | Likely works; verify |
| 6 | `current_timestamp()` — function-only source (no column deps) | L3a both steps | Likely works; verify |
| 7 | `BETWEEN d.start_time AND d.end_time` in JOIN ON — does `fact_orders.customer_city_at_order` lineage include `d.start_time/end_time` as condition columns? | L3→fact | **P / I** (join predicate columns often omitted from column lineage) |
| 8 | `WHERE t.id IS NULL OR (...)` as sentinel for new-vs-versioned inserts — does the NULL branch get recorded? | L3a Step 2 | I (expected; logical branch not in column graph today) |
| 9 | MERGE's `ON t.id = s.id AND t.is_active = 'Y'` with literal predicate — match_columns extraction looks at `EQ` pairs only; literal-bound predicate may be dropped | L3a Step 1 | I |
| 10 | Does MERGE's close-action (UPDATE of `end_time`) show `is_active` as a **dependency** of `end_time` via the WHEN MATCHED condition? Impact analysis depends on this | L3a Step 1 | P (suspected) |

**Probably-already-works (sanity checks, not gaps):** cross-CTE propagation (fixed by 8aaa454), column extraction for `DATE(order_ts)`, `SUM(amount)` aggregates, GROUP BY columns.

## Deliverables

Three artifacts, in order:

### 1. Test fixture + runner (`tests/test_cdc_scd_pipeline.py`)
- Fixture of the 6 SQL statements above (parametrized for Databricks dialect).
- One test per gap row in the table above, asserting the **expected** (post-fix) column lineage. Tests that exercise currently-broken behavior are marked `xfail` with a reference to the gap number — they flip to passing as fixes land.
- Runs in CI; the xfail list is the living gap backlog.

### 2. Gap fixes (scope-limited, each in its own commit)
Only fixes that are **localized and low-risk** land in this effort. Larger architectural changes (e.g., introducing a new edge type for "predicate-conditional" columns) are documented as follow-ups in `TODO.md`, not attempted here.

Likely in-scope fixes:
- Gap 3/10: extend `_parse_merge_statement` to record WHEN MATCHED `condition` columns as inputs to each assigned target column's mapping.
- Gap 9: generalize match_columns extraction to skip literal-bound predicates cleanly rather than dropping the whole predicate.

Likely out-of-scope (documented as follow-ups):
- Gap 7 (join-predicate columns in column lineage) — this is a project-wide convention question, not a CDC-specific fix.
- Gap 8 (logical-branch lineage from NULL sentinels) — needs a new edge semantic.

### 3. Showcase notebook (`examples/cdc_scd_pipeline.ipynb`)
Structure:
1. **Narrative intro** — what a CDC/SCD2 pipeline is, why it matters, what we're testing.
2. **The pipeline SQL** — the 6 statements above, with comments pointing out the interesting structures.
3. **Build + visualize** — `Pipeline(...)` → lineage graph rendered (GraphViz), table-level DAG.
4. **What clgraph captures** — 4-5 concrete impact-analysis queries (e.g., "what downstream columns depend on `raw_customer_cdc.after.city`?"), showing them returning correct results.
5. **Known limitations** — honest subsection listing the remaining gaps (7, 8, and any others not fixed), each with a short SQL snippet and what the ideal answer would be. Links to the xfail tests.

## Acceptance Criteria

- [ ] `tests/test_cdc_scd_pipeline.py` exists, runs in CI, and has tests for every row in the gap table.
- [ ] Each xfail is tagged with a specific gap number and a reason string.
- [ ] Every fix commit has: failing test → fix → passing test (TDD, per repo convention).
- [ ] The notebook runs end-to-end via `run_all_notebooks.py` with no errors.
- [ ] The notebook's "Known limitations" section matches the current xfail list (no silent discrepancies).
- [ ] `TODO.md` updated: item B checked off; follow-up gaps listed as their own new entries.

## Out of Scope

- Fact and mart layers are included in the fixture but we don't make them *perfect*; their role is to provide realistic downstream context for gap 7. If gap 7 isn't fixed, the fact-layer test is xfail; the notebook still renders.
- Other SCD variants (Type 1 overwrite-only, Type 3 previous-value column, Type 6 hybrid) — out of scope for this iteration.
- CDC sources other than Debezium envelope shape.
- Dialects other than Databricks/Delta.
- Performance (e.g., parse speed on large pipelines).

## Risks & Open Questions

1. **Risk: gap 3/10 fix has wider blast radius than expected.** The MERGE WHEN-MATCHED condition column inputs may already be partially captured elsewhere (e.g., in `match_columns`) in a different way; we need to not duplicate or contradict that. Mitigation: read `lineage_builder`'s MERGE handling before editing `query_parser`.
2. **Open: how should "trigger columns" be represented in the column graph?** Options: (a) same edge type as assignment inputs — simplest, but flattens semantics; (b) new edge attribute `role: "trigger" | "assignment"`. Recommend (a) for this iteration — keep it simple, revisit if users ask for the distinction.
3. **Open: should the staging CTE preserve `op` column for dim?** CDC deletes usually produce SCD2 tombstones (`is_active='N', end_time=now()`), not hard deletes. For this iteration we filter `op IN ('c','u')` and list delete-handling as a follow-up gap.

### Additional risks surfaced from code inspection

4. **Risk: gap 3/10 may already be half-plumbed, which makes the "fix" subtler than it looks.** `trace_strategies.trace_merge_columns` (`trace_strategies.py:186-229`) already stamps each MERGE `ColumnEdge` with `merge_condition` (raw SQL of the WHEN clause) and `merge_action`. The condition is therefore *recorded as edge metadata* but the columns *referenced by that condition* are not emitted as upstream inputs of the assigned target columns. The fix is not "start capturing the condition" — it's "parse the stored condition into column refs and add them as edges." Mitigation: the parser already has a column-extraction pass we can reuse; don't re-implement it in the MERGE path.

5. **Risk: gap 4 (self-referencing target) is load-bearing and unmitigated today.** Confirmed: clgraph has *no* multi-statement table versioning — `depends_on_tables` / `depends_on_units` in `pipeline_lineage_builder.py:76-108` reference tables by name only, with no N-vs-N+1 snapshot distinction. Step 2's `LEFT JOIN dim_customer t` will collapse onto the same node Step 1 just wrote, producing a self-loop in the pipeline graph. This is the dominant correctness issue for SCD2 and is *not* in the "likely in-scope fixes" list. Decide explicitly: either (a) accept the self-loop and document it as gap 4's known limitation in the notebook, or (b) promote gap 4 to in-scope and design a minimal "statement-index-scoped table ref" extension. Recommend (a) for this iteration — the architectural change for (b) is out of proportion with the stated deliverable.

6. **Risk: gap 7 is worse than the design implies.** Design says "join predicate columns often omitted." Code search confirms JOIN ON predicate columns produce **zero** lineage edges today (no handling in `lineage_builder`). The fact-layer BETWEEN join therefore produces no evidence of `dim_customer.start_time/end_time` influence on `fact_orders.customer_city_at_order` beyond the equi-join on `customer_id`. The test assertion should be a hard `xfail` with a clear message, not a soft "incomplete" — and the notebook's "known limitations" section needs an explicit callout so users don't silently trust the temporal join.

7. **Risk: gap 9 framing is inaccurate.** `match_columns` extraction (`query_parser.py:612-619`) walks EQ nodes and pairs columns — a literal-bound EQ like `t.is_active = 'Y'` produces an EQ with a literal on one side, which the current code likely filters (column-column pairs only) but may or may not drop cleanly. Before writing the fix, add a unit test that asserts current behavior on `ON t.id = s.id AND t.is_active = 'Y'`, then decide. The design's claim that "literal-bound predicate may be dropped" needs to be reduced to fact before coding.

8. **Risk: struct-column node naming is underspecified for test assertions.** `_extract_nested_path_from_expression` (`lineage_utils.py:335-395`) emits a `(table_ref, column_name, json_path, json_function, nested_path, access_type)` tuple for `after.id` — meaning the column node shape is not a single string but a structured reference. Tests that assert `"after.id"` as a node label will be brittle. Pin the exact shape by reading one existing struct-access test before writing fixture assertions.

9. **Risk: CI execution model.** The notebook acceptance says "runs via `run_all_notebooks.py` with no errors." clgraph is a static parser, so the SQL does not need a live Delta/Databricks engine — but any cell that calls `spark.sql(...)`, `dbutils`, or prints a rendered Delta table will fail in CI. Keep all cells to: (a) strings of SQL, (b) `Pipeline(...)` calls, (c) `pyvis`/graphviz renders against file output only.

10. **Open: de-duplicate with existing MERGE tests.** `tests/test_merge_statements.py` already covers basic MERGE parsing, match_columns, matched/not-matched actions, and edge properties. The new `tests/test_cdc_scd_pipeline.py` should cover *only* the CDC-pipeline-shaped assertions (multi-statement, self-reference, envelope flatten, SCD2-pair semantics) and reference existing tests for the single-statement primitives. Otherwise the gap backlog becomes ambiguous.

11. **Open: QUALIFY assertion target.** QUALIFY partition/order columns are emitted via edges with `context="qualify_partition"` / `context="qualify_order"` and `is_qualify_column=True` (`lineage_builder.py:376-442`). When writing gap 2's test, assert on `is_qualify_column`/`qualify_context` metadata, not on the plain node set — that's what will catch the "silently dropped through CTE" regression the design worries about.

12. **Open: should gap 4 (self-reference) and gap 7 (join-predicate columns) graduate to their own design docs?** Both are cross-cutting architectural shifts (statement-scoped table refs; a new edge semantic for predicate-conditional columns). If we leave them as one-liners in `TODO.md`, they will rot. Recommend: this effort closes with two short follow-up design stubs, not just backlog entries.
