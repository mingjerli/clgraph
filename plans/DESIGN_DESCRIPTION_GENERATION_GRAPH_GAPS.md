# Design: Description Generation — Graph Utilization Gaps

## Overview

`generate_all_descriptions()` is architecturally sound: it walks computed columns in
topological order and builds each prompt from the column's direct lineage sources, so
descriptions cascade through the graph. This document specifies fixes for four gaps
that keep it from fully using the graph:

| ID | Gap | Severity |
|----|-----|----------|
| D1 | Source-table columns are never described (cold-start at the graph boundary) | High |
| D2 | Incoming edges found by O(E) linear scan instead of the adjacency index | Medium (perf) |
| D3 | Rule-based fallback descriptions are indistinguishable from LLM output and cascade downstream | High |
| D4 | Queries without a destination table are skipped, unlike metadata propagation | Medium |

Each gap is specified independently; D1 and D3 share a change to
`build_description_prompt()` and should land together or in sequence D3 → D1.

---

## D1: Source-Column Cold Start

### Problem Statement

`MetadataManager.generate_all_descriptions()` only processes columns that satisfy
`col.is_computed()` on a query's `destination_table`
(`metadata_manager.py:90-100`). Columns of raw source tables (e.g. `raw.users.email`)
are never candidates. `build_description_prompt()` (`column.py:152-160`) only includes
source columns *that already have a description*, so for the first computed layer the
"Source columns" section is silently empty — the cascade starts from nothing exactly
where context matters most.

### Current Behavior

```python
pipeline = Pipeline(queries, dialect="bigquery")
pipeline.llm = llm
pipeline.generate_all_descriptions()
# raw.users.email          -> no description, never processed
# staging.users.email_norm -> prompt contains only: column name, table, SQL expression
```

### Design

Describe source columns using **forward usage context** — the graph knows how every
source column is consumed downstream even when nothing upstream exists.

1. **New prompt builder** in `column.py`:

   ```python
   def build_source_description_prompt(column: ColumnNode, pipeline: "Pipeline") -> str:
   ```

   Contents (all values through `sanitize_for_prompt` / `sanitize_sql_for_prompt`):
   - `Column:` / `Table:` lines as today.
   - `Sibling columns:` up to 15 other column names in the same table (domain signal).
   - `Used downstream as:` up to 5 entries derived from outgoing edges
     (`pipeline._get_outgoing_edges(column.full_name)`), each formatted as
     `<target.full_name> = <target.expression>`.
   - Same instruction block as `build_description_prompt` (one sentence, ≤15 words,
     no SQL jargon).

2. **New opt-in parameter** on `MetadataManager.generate_all_descriptions()` and the
   `Pipeline` wrapper:

   ```python
   def generate_all_descriptions(..., include_sources: bool = False)
   ```

   When `True`, source-table columns (columns of tables where
   `table_graph.tables[t].is_source`) are processed **before** the topological walk of
   computed columns, so their descriptions feed the first computed layer in the same
   run. Existing skip rules apply unchanged: authored (`DescriptionSource.SOURCE`)
   descriptions are never overwritten unless `overwrite=True`.

   Opt-in default rationale: the library is released (PyPI); flipping the default
   changes LLM call volume and export diffs for existing users. Revisit the default in
   a minor release.

3. `generate_description()` dispatches on layer: source columns (no incoming edges,
   `is_computed()` is False) use `build_source_description_prompt`; computed columns
   keep `build_description_prompt`.

### Test Cases

- Two-layer fixture (`raw.users` → `staging.users` → `mart.user_stats`):
  - `include_sources=True` produces descriptions on `raw.users.*` with
    `description_source == GENERATED`.
  - The prompt built for `staging.users.email_norm` afterwards contains the
    `raw.users.email` description (assert via `build_description_prompt`).
- `include_sources=False` (default): behavior identical to today (regression test).
- Source column with an authored SQL-comment description is skipped unless
  `overwrite=True`.
- Source column consumed by zero described targets still yields a valid prompt
  (name + siblings only).

### Risks and Mitigations

- **Cost**: more LLM calls. Mitigated by opt-in flag; log the source-column count in
  the existing progress logging.
- **Hallucination on thin context**: a bare column name invites guessing. The prompt
  instruction already demands ≤15 words; usage context constrains the model further.
  Output still passes `_validate_description_output`.

---

## D2: O(V·E) Incoming-Edge Scans

### Problem Statement

Three functions in `column.py` find incoming edges with a full scan of every edge in
the pipeline:

- `build_description_prompt` — `column.py:152`
- `propagate_metadata_backward` — `column.py:225`
- `propagate_metadata` — `column.py:268`

```python
incoming_edges = [e for e in pipeline.edges if e.to_node == column]
```

`PipelineLineageGraph._incoming_index` (`column.py:315`, populated at `column.py:327`
keyed by `edge.to_node.full_name`) exists precisely for this lookup, and
`Pipeline._get_incoming_edges(full_name)` (`pipeline.py:193`) already wraps it.
Bulk operations (`generate_all_descriptions`, `propagate_all_metadata`) therefore run
in O(V·E) instead of O(V+E).

### Design

1. Replace all three scans with:

   ```python
   incoming_edges = pipeline._get_incoming_edges(column.full_name)
   ```

2. Promote the accessor to a public method `Pipeline.get_incoming_edges(full_name)`
   (keep the underscore alias for backward compatibility), since `column.py` module
   functions are public API and should not depend on a private method.

3. **Precondition to verify in implementation**: `_incoming_index` is keyed by
   `to_node.full_name`; confirm every caller passes `column.full_name` for a node that
   is registered in `column_graph` (an unregistered node returns `[]` where the scan
   would also return `[]` — equivalent, but assert this in tests, not assumptions).

### Test Cases

- Equivalence: for every column in an existing multi-query fixture pipeline,
  `pipeline.get_incoming_edges(col.full_name)` equals the linear-scan result
  (set equality on `(from_node.full_name, to_node.full_name)`).
- Existing `generate_all_descriptions` / `propagate_all_metadata` test suites pass
  unchanged (behavior-preserving refactor).
- Optional benchmark note: synthetic pipeline with ~1k columns / ~5k edges shows the
  bulk description pass no longer scales with E per column.

### Risks and Mitigations

- **Index staleness** if edges are appended without going through the graph's
  `add_edge` path. Audit write paths for `column_graph.edges` during implementation;
  the equivalence test above catches divergence.

---

## D3: Fallback Descriptions Are Indistinguishable and Cascade

### Problem Statement

When the LLM call fails or validation rejects its output,
`_generate_fallback_description` (`column.py:187-193`) writes a humanized column name
("customer_ltv" → "Customer Ltv") and stamps it
`description_source = DescriptionSource.GENERATED` — the same value a real model
description gets. Consequences:

1. Nothing persisted distinguishes placeholder text from model output; the
   `return False` signal (added in PR #75) is transient.
2. `generate_all_descriptions` filters on `not col.description`
   (`metadata_manager.py:97`), so a later re-run — e.g. after fixing LLM
   configuration — **skips** every fallback column instead of retrying it.
3. Downstream prompts include the fallback text as source context
   (`column.py:156-160`), feeding noise into the cascade.

### Design

1. **New enum member** in `models.py`:

   ```python
   class DescriptionSource(Enum):
       SOURCE = "source"
       GENERATED = "generated"
       PROPAGATED = "propagated"
       FALLBACK = "fallback"   # rule-based placeholder, no model involved
   ```

   `_generate_fallback_description` stamps `FALLBACK`.

2. **Retry semantics**: the candidate filter in
   `MetadataManager.generate_all_descriptions` treats `FALLBACK` as "no description":

   ```python
   needs_description = (
       not col.description or col.description_source == DescriptionSource.FALLBACK
   )
   ```

   A re-run with a working LLM upgrades placeholders to real descriptions. This is an
   intentional behavior change; call it out in the changelog.

3. **Prompt hygiene** in `build_description_prompt`: list **all** direct source
   columns by `full_name` (fixes the silent drop of undescribed sources), and append
   `: <description>` only when a description exists and its source is not `FALLBACK`:

   ```
   Source columns:
   - raw.users.email: User's primary email address
   - raw.users.signup_ts
   ```

4. **Serialization**: exporters emit `description_source` via `.value`; the new
   `"fallback"` string flows through JSON/CSV export automatically. Verify the diff
   tooling treats it as an ordinary value; if any deserializer whitelists enum
   values, add `"fallback"`.

### Test Cases

- LLM raises → column gets fallback text with `description_source == FALLBACK`;
  function returns `False`.
- Re-running `generate_all_descriptions` with a working LLM overwrites `FALLBACK`
  columns (and still skips `SOURCE`/`GENERATED` unless `overwrite=True`).
- `build_description_prompt` for a column whose sources are (a) described,
  (b) fallback-described, (c) undescribed lists all three by name and attaches text
  only to (a).
- JSON export round-trip preserves `"fallback"`.

### Risks and Mitigations

- **Behavior change** (retry semantics): users relying on fallback text persisting
  across runs will see regeneration attempts. Changelog + the `on_error="raise"` path
  already exists for users who prefer hard failures.

---

## D4: Queries Without a Destination Table Are Skipped

### Problem Statement

The two bulk operations disagree about which columns belong to a query:

- `generate_all_descriptions` — only `query.destination_table`
  (`metadata_manager.py:92-100`); terminal plain `SELECT` queries contribute nothing.
- `propagate_all_metadata` — `query.destination_table or f"{query_id}_result"`
  (`metadata_manager.py:156`).

Computed columns of a pipeline-final `SELECT` therefore receive propagated metadata
but never receive generated descriptions.

### Design

1. Extract one shared helper on `MetadataManager` (or module level):

   ```python
   def _target_table(query: ParsedQuery) -> str:
       return query.destination_table or f"{query.query_id}_result"
   ```

2. Use it in both `generate_all_descriptions` and `propagate_all_metadata` so the
   candidate column sets are computed identically (description generation keeps its
   additional `needs_description` and `is_computed()` filters).

### Test Cases

- Pipeline ending in a plain `SELECT`: its computed output columns receive
  descriptions.
- Parity test: the set of `(table, column)` pairs visited by description generation
  equals the `is_computed()` subset of pairs visited by metadata propagation.
- Regression: pipelines where every query has a destination table are unaffected.

### Risks and Mitigations

- **Cost**: additional columns processed. Marginal — only terminal SELECTs; noted in
  changelog.

---

## Implementation Phases

1. **Phase 1 — D2 (index lookups)**: behavior-preserving, unblocks perf; smallest
   review surface.
2. **Phase 2 — D3 (FALLBACK source + prompt hygiene)**: changes `models.py`,
   `column.py`, `metadata_manager.py`; includes the "list all sources" prompt change
   D1 builds on.
3. **Phase 3 — D4 (target-table helper)**: small, isolated.
4. **Phase 4 — D1 (source-column descriptions)**: new prompt builder + `include_sources`
   flag; depends on Phase 2's prompt shape.

Each phase: `make pre-commit` (ruff check/format + tests) green; separate PRs
following `feat:`/`fix:` conventions.

## Success Metrics

- First-layer computed columns' prompts contain source context when
  `include_sources=True` (D1).
- Bulk description generation does no per-column full-edge scans (D2).
- Zero fallback strings served as source context in any prompt; re-runs retry
  fallback columns (D3).
- Description coverage includes terminal SELECT outputs; candidate sets of the two
  bulk operations agree (D4).
