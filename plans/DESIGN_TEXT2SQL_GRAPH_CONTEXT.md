# Design: Text2SQL — Deeper Graph Context

## Overview

`GenerateSQLTool` (`tools/sql.py`) currently uses the graph mostly as a schema
catalog: tables, columns, descriptions, PII flags, and table-level "derived from"
lines. The deep graph assets — transitive column traversal, join-key knowledge,
final-table identification — are either gated behind the non-default `two_stage`
strategy or not surfaced at all. This document specifies five gaps:

| ID | Gap | Severity |
|----|-----|----------|
| T1 | Column lineage absent from `direct` mode prompts | Medium |
| T2 | `expand_with_lineage` is 1-hop, not transitive | Medium |
| T3 | Join keys never surfaced (only "derived from" lines) | High |
| T4 | Final-table identification unused; intermediates presented as equally queryable | Medium |
| T5 | Non-LLM table-selection fallback ignores the graph | Low |

T1–T3 all modify how the `relationship_section` prompt slot is assembled; T3 is the
highest-value change for generated-SQL correctness (join quality).

### Shared change: graph-context assembly point

Both `_generate_direct` (`sql.py:182`) and `_generate_two_stage` (`sql.py:238`)
currently fill `relationship_section` with a single builder call. Refactor both to a
shared assembly:

```python
def _build_graph_context(self, builder: ContextBuilder, tables: List[str]) -> str:
    parts = [
        builder.build_relationship_context(tables),   # existing, T4 adjusts labels
        builder.build_lineage_context(tables),        # T1 wires into direct mode
        builder.build_join_context(tables),           # T3, new
    ]
    return "\n\n".join(p for p in parts if p)
```

The existing `GENERATE_SQL_PROMPT` / `GENERATE_SQL_WITH_EXPLANATION_PROMPT` templates
keep their `{relationship_section}` slot; no template signature change. Sanitization
stays where it is today (each section passed through `sanitize_for_prompt` at the
`prompt.format(...)` call sites).

---

## T1: Column Lineage in Direct Mode

### Problem Statement

`_generate_direct` fills `relationship_section` with table-level
`build_relationship_context()` only (`sql.py:194`). `build_lineage_context()` — which
performs real transitive traversal via `trace_column_backward` (`context.py:451-483`)
— is only invoked in `_generate_two_stage` (`sql.py:255`). The default strategy never
shows the model how output columns relate to source columns, which is exactly the
information needed to pick the right table for a metric.

### Design

1. Use the shared `_build_graph_context` above in **both** strategies, so `direct`
   mode gains the `## Column Lineage` section.
2. Make the existing hard-coded caps configurable on `ContextConfig`:

   ```python
   max_lineage_columns_per_table: int = 10   # today: literal 10 (context.py:470)
   max_lineage_lines: int = 20               # today: literal 20 (context.py:483)
   ```

3. `include_lineage=False` in `ContextConfig` disables the section (already the
   behavior of `build_lineage_context`; unchanged).

### Test Cases

- Direct-mode prompt for a two-layer fixture contains `## Column Lineage` with
  `mart.x <- raw.y` lines.
- Caps respected: a table with 30 output columns contributes at most
  `max_lineage_columns_per_table` lines; total lines ≤ `max_lineage_lines`.
- `include_lineage=False` removes the section from both strategies.

### Risks and Mitigations

- **Prompt growth** on wide pipelines — bounded by the two caps; both configurable.

---

## T2: Transitive Lineage Expansion

### Problem Statement

`expand_with_lineage` (`context.py:274-303`) adds only the *direct* parent tables of
each selected table. In `two_stage` mode, a question whose answer requires a
grandparent table (e.g. join key only present two levels up) produces context that
omits it, and the generated SQL references a table the model never saw or invents
joins.

### Current Behavior

```python
# a -> b -> c   (c selected)
builder.expand_with_lineage(["c"])   # ["c", "b"] — never "a"
```

### Design

1. BFS to a configurable depth with a hard size cap:

   ```python
   def expand_with_lineage(self, tables: List[str], depth: Optional[int] = None) -> List[str]:
   ```

   - `depth=None` reads `ContextConfig.lineage_expansion_depth` (new field,
     default `2`).
   - Traversal: frontier = selected tables; each round maps table →
     `table_graph.tables[t].created_by` → `queries[qid].source_tables`; stop at
     `depth` rounds or fixpoint. Maintain a visited set (the table graph is a DAG,
     but self-referencing tables exist — `ParsedQuery.self_referenced_tables` — so
     guard anyway).
   - **Priority on truncation**: if the expanded set exceeds
     `ContextConfig.max_tables`, keep shallower tables first (original selection,
     then depth-1 parents, then depth-2, ...).

2. Backward compatibility: `depth=1` reproduces today's behavior; the default of `2`
   is a deliberate improvement documented in the changelog.

### Test Cases

- Chain `a -> b -> c`, select `["c"]`: depth 1 → `{b, c}`; depth 2 → `{a, b, c}`;
  `None` → config default.
- Self-referencing query (`INSERT INTO t SELECT ... FROM t`) terminates.
- Truncation keeps the originally selected tables and nearest ancestors.

### Risks and Mitigations

- **Context bloat** in deep DAGs — depth default of 2 plus `max_tables` cap with
  shallow-first priority.

---

## T3: Join-Key Surfacing

### Problem Statement

The only relationship the model sees is `"- X is derived from Y"`
(`context.py:447`). Which columns actually join two tables — the single most
error-prone part of text2sql — is left for the model to guess from column names,
even though the pipeline holds this knowledge in two forms: observed join predicates
in the parsed SQL, and shared upstream ancestry in the column graph.

### Design

Two mechanisms, one output section.

#### 1. Observed joins (from query ASTs)

`ParsedQuery.ast` retains the sqlglot AST (`models.py:901`). New builder method:

```python
def get_observed_joins(self, tables: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    # [{left_table, left_column, right_table, right_column, query_id}, ...]
```

Implementation sketch:
- For each `ParsedQuery`, walk `ast.find_all(exp.Join)`.
- From each join's `on` condition, collect top-level `exp.EQ` nodes whose both sides
  are `exp.Column`; treat multiple EQs under one `AND` as a composite key (emit one
  entry per column pair, same `query_id`).
- Resolve table aliases to real table names using the FROM/JOIN `exp.Table` nodes of
  that query; consult `ParsedQuery.self_ref_aliases` for self-referencing tables.
  **Phase 1 scope**: skip predicates whose alias resolves to a CTE-internal name
  rather than a pipeline table; log at debug level. (CTE mapping via
  `query_lineage` is a follow-up.)
- `USING (col)` joins (`exp.Join` args) emit `left.col = right.col`.
- Non-equi joins and `ON` conditions that are not column-to-column equality are
  skipped.

#### 2. Candidate joins (from shared lineage sources)

For table pairs never joined in the pipeline (e.g. two marts), infer candidates:

- For each output column of each context table, call
  `pipeline.trace_column_backward(table, column)` once and record
  `ultimate_source_full_name -> [(table, column), ...]`.
- Any source mapped to columns in ≥2 distinct context tables yields candidate pairs.
- Deduplicate against observed joins; cap total candidates
  (`ContextConfig.max_join_hints: int = 15`, shared with observed joins,
  observed first).

#### 3. Prompt section

```python
def build_join_context(self, tables: List[str]) -> str:
```

```
## Join Hints

- orders.customer_id = customers.id (observed in query_3)
- candidate: mart_ltv.user_id = mart_churn.user_id (both derive from raw.users.id)
```

Wired into `_build_graph_context` (see Overview) for both strategies. Candidates are
explicitly labeled `candidate:` so the model can weigh them below observed joins.

### Test Cases

- Fixture with `JOIN ... ON o.customer_id = c.id` and aliases → observed join with
  real table names and `query_id`.
- Composite key (`ON a.x = b.x AND a.y = b.y`) → two entries, same query.
- `USING (id)` → resolved to both tables.
- Two marts sharing `raw.users.id` ancestry, never joined directly → one
  `candidate:` line; no candidates between unrelated tables.
- Non-equi join (`ON a.ts > b.ts`) produces nothing.
- Cap: hints truncated to `max_join_hints`, observed joins prioritized.

### Risks and Mitigations

- **Alias/CTE resolution complexity** — phase 1 restricts to directly resolvable
  aliases and logs skips; correctness over coverage.
- **False-positive candidates** (shared source ≠ valid join) — labeled `candidate:`,
  deduplicated, capped; observed joins always listed first.
- **O(cols × trace) cost** — one backward trace per context column, bounded by
  `max_lineage_columns_per_table`; traces are already the cost profile of
  `build_lineage_context`.

---

## T4: Final-Table Steering

### Problem Statement

`table_graph.get_final_tables()` (`table.py:304`, `read_by == []`) is never consulted
by `ContextBuilder`. `_format_table_context` annotates only `(Source table)`
(`context.py:398`); intermediates and marts look identical, so the model may answer
from a staging table when a mart exists — or from an intermediate that a later query
overwrites semantically.

### Design

1. **Role annotation** in `_format_table_context`:
   - `(Source table)` — unchanged.
   - `(Final table)` — `len(table_node.read_by) == 0` and not a source.
   - `(Intermediate table)` — everything else.
   Controlled by `ContextConfig.annotate_table_roles: bool = True`.

2. **Prompt instruction**: add one line to the `## Instructions` block of both SQL
   templates (via the existing `extra_instructions` slot assembly, applied in both
   strategies):

   ```
   - Prefer final tables when they answer the question; use intermediate tables only when required
   ```

3. **Truncation priority** in `build_schema_context` (`context.py:356-361`): when
   trimming to `max_tables`, keep order final > intermediate > source (today it is
   derived > source with no final/intermediate distinction).

### Test Cases

- Fixture `raw -> staging -> mart`: context labels `raw` source, `staging`
  intermediate, `mart` final; instruction line present.
- `annotate_table_roles=False` restores current output (regression).
- Truncation with `max_tables=2` keeps `mart` and `staging`, drops `raw`.

### Risks and Mitigations

- **Fragment pipelines**: analyzing a subset of a real pipeline makes mid-DAG tables
  look final. Acceptable — the annotation reflects the graph as loaded; document in
  the tool docstring.

---

## T5: Graph-Aware Selection Fallback

### Problem Statement

When the LLM table-selection call fails, `select_tables_by_keywords`
(`context.py:489-544`) scores tables purely lexically, and its `min_tables` padding
appends arbitrary tables in dict-insertion order. A question matching a mart by name
can miss the parent table required for its join, while padding adds noise tables.

### Design

1. **Score diffusion (1 hop)**: after lexical scoring, each table with score > 0
   contributes `0.5 × score` to its direct graph neighbors (parents via
   `created_by`/`source_tables`, children via `read_by`). Single pass, applied on the
   pre-diffusion scores (no iteration/feedback).
2. **Graph-aware padding**: to reach `min_tables`, prefer (in order) unselected
   graph neighbors of already-selected tables, then final tables, then the current
   arbitrary order. Deterministic tie-break: alphabetical.
3. Diffusion factor as a module constant (`_NEIGHBOR_SCORE_FACTOR = 0.5`); not
   config-exposed until there is evidence tuning matters.

### Test Cases

- Mart matches keywords, its parent (needed for the join, zero lexical score) is
  selected via diffusion before an unrelated lexically-weak table.
- Padding prefers neighbors of selected tables over unrelated tables; result order
  deterministic across runs.
- Pure-lexical results unchanged when the graph has a single table.

### Risks and Mitigations

- Minimal — fallback path only. Diffusion is one pass and cannot cycle.

---

## Implementation Phases

1. **Phase 1 — T1 (lineage in direct mode) + shared `_build_graph_context`**:
   creates the assembly point T3 plugs into; config caps added.
2. **Phase 2 — T4 (role annotation + instruction + truncation priority)**: isolated
   in `context.py` + one instruction line in `sql.py`.
3. **Phase 3 — T2 (transitive expansion)**: `expand_with_lineage(depth)` +
   `lineage_expansion_depth` config.
4. **Phase 4 — T3 (join hints)**: largest change; observed joins first, candidate
   inference second (can ship as two PRs).
5. **Phase 5 — T5 (fallback diffusion)**: independent, lowest priority.

Each phase: `make pre-commit` green; separate PRs (`feat:` prefix); docs-site update
for new `ContextConfig` fields after the API settles.

## Success Metrics

- Default (`direct`) prompts contain column lineage and join hints for pipelines
  where they exist (T1, T3).
- `two_stage` context includes all ancestor tables within the configured depth (T2).
- Every observed equi-join in the pipeline's SQL appears in `## Join Hints` for
  in-context tables, with zero fabricated observed joins (T3).
- Final/intermediate/source roles labeled in every schema context; truncation never
  drops a final table while keeping a source table (T4).
- Fallback selection includes join-required parent tables for the fixture suite (T5).
