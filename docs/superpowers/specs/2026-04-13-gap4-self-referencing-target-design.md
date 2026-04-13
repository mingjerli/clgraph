# Gap 4: Self-Referencing Target Across Statements

**Date:** 2026-04-13
**Parent:** `docs/superpowers/specs/2026-04-13-cdc-scd-pipeline-gaps-design.md`, Risk #5
**TODO.md ref:** Gap 4 under Architectural Gaps

## Problem

When a multi-statement pipeline writes and then reads the same table in consecutive statements, clgraph collapses both references onto a single graph node. The pipeline graph produces a self-loop instead of expressing "Step 2 reads the prior state of `dim_customer`."

Concrete SCD2 pattern:

```sql
-- Step 1 (MERGE): close old rows
MERGE INTO dim_customer t
USING staging_customer_latest s ON t.id = s.id AND t.is_active = 'Y'
WHEN MATCHED AND (t.name <> s.name OR t.city <> s.city) THEN
  UPDATE SET t.end_time = current_timestamp(), t.is_active = 'N';

-- Step 2 (INSERT): open new version rows
INSERT INTO dim_customer
SELECT s.id, s.name, s.city, s.email,
       current_timestamp() AS start_time,
       TIMESTAMP '9999-12-31 00:00:00' AS end_time,
       'Y' AS is_active
FROM staging_customer_latest s
LEFT JOIN dim_customer t
  ON s.id = t.id AND t.is_active = 'Y'
WHERE t.id IS NULL OR (t.name <> s.name OR t.city <> s.city);
```

Step 2's `LEFT JOIN dim_customer t` must read the state of `dim_customer` **after** Step 1's MERGE. Today clgraph sees both references as the same bare name `dim_customer` and produces a single `TableNode` with no temporal distinction.

## Deep Analysis of Current Behavior

### Where the collapse happens

**1. Source-table exclusion in `multi_query.py:286`.**
`_extract_source_tables` filters out any table matching `destination_table`:

```python
if not table_name or table_name == destination_table:
    continue
```

For Step 2 (`INSERT INTO dim_customer ... LEFT JOIN dim_customer`), `destination_table` is `dim_customer`. The `LEFT JOIN dim_customer` reference is therefore **silently dropped** from `source_tables`. Step 2's `ParsedQuery` records `source_tables = {"staging_customer_latest"}` only. This is the first and most severe collapse: the self-read is invisible to the entire downstream pipeline.

**2. Single `TableNode` per name in `table.py:160-164`.**
`TableDependencyGraph.add_table` deduplicates by name. Both the Step 1 MERGE target and the Step 2 INSERT target resolve to the same `TableNode("dim_customer")`. That node has a single `created_by` slot (set by the first DDL; MERGE/INSERT are DML so they go to `modified_by`). There is no concept of "version N" vs "version N+1."

**3. Shared column naming in `pipeline_lineage_builder.py:734-736`.**
`_make_full_name` for physical table columns returns `{table_name}.{column_name}` with no query-scoped prefix. Both Step 1's input `dim_customer.id` and Step 2's output `dim_customer.id` map to the same `full_name`, so the pipeline graph has exactly one `ColumnNode` for each column of `dim_customer`, shared across all statements.

**4. Topological sort self-dependency in `table.py:188-204`.**
`_build_query_dependencies` walks `source_tables` and looks up `table_node.created_by` / `table_node.modified_by`. Because Step 2 does not even list `dim_customer` in its `source_tables` (collapsed at step 1 above), there is no dependency edge from Step 2 back to Step 1 via `dim_customer`. If it *were* listed, the dependency would be `query_1 depends on query_0` (correct), but the topological sorter would also see `query_0` in `modified_by` for `dim_customer`, potentially introducing a cycle if both queries modify the same table.

### What the graph looks like today

For the two-step SCD2, clgraph currently produces:

```
staging_customer_latest.id ──► dim_customer.id        (from Step 1 MERGE)
staging_customer_latest.name ──► dim_customer.name     (from Step 1 MERGE)
staging_customer_latest.id ──► dim_customer.id         (from Step 2 INSERT, if it parses)
staging_customer_latest.name ──► dim_customer.name     (from Step 2 INSERT)
```

The `LEFT JOIN dim_customer` in Step 2 produces **zero** edges because `dim_customer` was excluded from `source_tables`. The self-read semantic is completely lost. A downstream impact query like "what does `dim_customer.is_active` depend on?" would miss the fact that the INSERT's WHERE clause reads prior `is_active` values.

### Tests encoding current (collapsed) behavior

- `tests/test_merge_statements.py`: all tests use single-statement MERGE. No test exercises a two-statement pipeline where the MERGE target is also read. These tests are **not directly affected** by a fix, but the pipeline-level MERGE integration test (`TestMergePipeline.test_merge_in_pipeline`, line 266) creates a single-query pipeline; it would remain unchanged.

- `tests/test_pipeline_lineage_builder.py`: functional tests use simple CREATE-then-SELECT chains. No test has a table that is both written and read across statements.

- No existing test asserts self-loop behavior. The collapse is untested and unintentional.

## Design Options

### Option A: Statement-Scoped Virtual Table Names

**Idea:** When a table appears as both destination and source across statements, introduce a virtual name like `dim_customer@after_stmt0` for the post-write version. The original name `dim_customer` becomes a read-only alias for the pre-pipeline state.

**Data model change:**
- `TableNode` gains an optional `version: int` field (default 0 = original source state).
- `TableDependencyGraph.tables` keys become `(table_name, version)` tuples or versioned strings like `dim_customer@v1`.
- `ParsedQuery.source_tables` and `destination_table` carry versioned names.

**Scope of code touched:**
- `multi_query.py`: version assignment pass after all queries are parsed; rewrite `_extract_source_tables` to not filter self-references but instead assign them versioned names.
- `table.py`: `TableNode`, `add_table`, `add_query`, `_build_query_dependencies`, `_build_table_dependencies`, `topological_sort` all need version awareness.
- `pipeline_lineage_builder.py`: `_make_full_name`, `_infer_table_name`, `_is_physical_table_column` need version-qualified names.
- `pipeline.py`: user-facing APIs (`get_column`, impact queries) must accept both bare and versioned names.
- All exporters (JSON, GraphViz) would expose versioned names.

**User-facing graph shape:**
```
staging_customer_latest.id ──► dim_customer@v1.id     (Step 1 MERGE)
dim_customer@v1.id ──► dim_customer@v2.id             (Step 2 reads v1, writes v2)
staging_customer_latest.id ──► dim_customer@v2.id     (Step 2 INSERT)
```

**Pros:** Fully general; works for N-step chains on the same table.
**Cons:** Invasive; changes the naming contract for every downstream consumer. Versioned names leak into exports, notebooks, and user queries. Migration burden is high.

### Option B: Snapshot Nodes with Shadow Edges

**Idea:** Keep the physical `TableNode` and `ColumnNode` names unchanged. For each statement that reads a table it also writes, insert a "snapshot" copy of the table's columns scoped to the reading query. Connect the snapshot to the physical node with a `snapshot_of` edge.

**Data model change:**
- New `ColumnNode` attribute: `is_snapshot: bool = False`, `snapshot_of: Optional[str] = None`.
- New edge type `"snapshot"` connecting `dim_customer.id` (physical) to `dim_customer.id` (snapshot, query-scoped).
- Snapshot nodes use query-scoped naming: `{query_id}:snapshot:dim_customer.id`.

**Scope of code touched:**
- `multi_query.py`: stop filtering self-references in `_extract_source_tables`; add `self_referenced_tables: Set[str]` to `ParsedQuery`.
- `pipeline_lineage_builder.py`: in `_add_query_columns`, detect self-referenced input columns and create snapshot nodes instead of sharing the physical node. In `_add_query_edges`, wire edges through snapshot nodes.
- `models.py`: add `is_snapshot`, `snapshot_of` to `ColumnNode`.
- `table.py`: add `is_self_referenced: bool` to `TableNode`; adjust `_build_query_dependencies` to handle self-refs without creating cycles.

**User-facing graph shape:**
```
staging.id ──► dim_customer.id          (Step 1 output)
dim_customer.id ──[snapshot]──► q1:snapshot:dim_customer.id
q1:snapshot:dim_customer.id ──► dim_customer.id   (Step 2 output, via INSERT)
staging.id ──► dim_customer.id          (Step 2 output)
```

**Pros:** Physical names unchanged for most queries; only self-referencing queries see snapshot nodes.
**Cons:** Snapshot naming is internal jargon; adds a new node type that renderers and exporters must handle. The snapshot-to-physical edge direction is confusing (is it "reads from" or "becomes"?).

### Option C: Read-Before-Write Detection with Ordered DML Edges

**Idea:** Do not create new nodes. Instead, recognize self-referencing tables at the pipeline level, stop excluding them from `source_tables`, and annotate the resulting column edges with a `statement_order` attribute. The self-read edges are distinguished from self-write edges by their `statement_order` and a new `edge_role` ("prior_state_read" vs "write").

**Data model change:**
- `ParsedQuery` gains `self_referenced_tables: Set[str]` (tables that appear in both `destination_table` and `source_tables` for the same query, or across consecutive queries targeting the same table).
- `ColumnEdge` gains `statement_order: Optional[int]` and `edge_role: Optional[str]` (values: `"prior_state_read"`, `"write"`, or `None` for normal edges).
- `TableNode` gains `self_referencing_queries: List[str]` tracking which queries read-before-write.

**Scope of code touched:**
- `multi_query.py:286`: remove the `table_name == destination_table` filter; instead, when a source table matches the destination, add it to both `source_tables` and `self_referenced_tables`.
- `table.py:188-204`: when building query dependencies for a self-referencing query, the query depends on all *prior* modifications to the same table, but not on itself. Add cycle-prevention logic.
- `pipeline_lineage_builder.py`:
  - `_infer_table_name` (line 680-700): for input columns from a self-referenced table, qualify the node name with `{query_id}:self_read:{table_name}` to avoid colliding with the output node.
  - `_make_full_name`: add a branch for self-read input nodes.
  - `_add_query_edges`: stamp edges from self-read nodes with `edge_role="prior_state_read"` and `statement_order`.
  - `_add_cross_query_edges`: connect the prior output (from the previous statement) to the self-read input node of the current statement.
- `models.py`: add `statement_order` and `edge_role` to `ColumnEdge`; add `self_referenced_tables` to `ParsedQuery`.
- Exporters: include new edge attributes in JSON/GraphViz output.

**User-facing graph shape:**
```
                                    ┌── staging.id ──────────────────┐
                                    │                                ▼
staging.id ──► dim_customer.id ──► q1:self_read:dim_customer.id ──► dim_customer.id
               (Step 1 output)     (Step 2 reads prior state)      (Step 2 output)
```

The self-read node is query-scoped and represents "dim_customer as it existed before this query ran." Cross-query edges connect Step 1's output `dim_customer.id` to Step 2's `q1:self_read:dim_customer.id`.

**Pros:** Minimal new concepts; edges carry explicit semantics; physical output nodes keep their canonical names; no version numbering scheme.
**Cons:** Self-read nodes are still a new node variant; `q1:self_read:dim_customer.id` naming needs renderer support.

## Recommendation: Option C (Read-Before-Write with Ordered DML Edges)

Option C is recommended because:

1. **Smallest blast radius.** Options A and B change the core naming contract for *all* table nodes or introduce a whole new node type. Option C only introduces query-scoped self-read nodes for the specific queries that exhibit self-reference. Non-self-referencing pipelines are completely untouched.

2. **No version numbering.** Option A's `@v1`/`@v2` naming leaks implementation detail into every user-facing surface. Option C's self-read nodes are internal to the query that reads its own target; the physical output nodes keep their canonical names.

3. **Cycle-safe.** The topological sort already handles `modified_by` dependencies. Option C adds a targeted cycle-prevention check (a query cannot depend on itself), which is simpler than Option A's global version-aware sort.

4. **Composable with future work.** If gap 7 (join-predicate columns) lands later, the `edge_role` attribute provides a natural extension point for predicate-conditional reads. Option B's snapshot semantics would compete with that.

### Worked Example: Before and After

**Before (current):**

```
staging_customer_latest.id ─────────────────────────────► dim_customer.id
staging_customer_latest.name ───────────────────────────► dim_customer.name
staging_customer_latest.city ───────────────────────────► dim_customer.city
staging_customer_latest.email ──────────────────────────► dim_customer.email
  (Step 1 MERGE edges only; Step 2 INSERT also writes to dim_customer
   but LEFT JOIN dim_customer is invisible — filtered at multi_query.py:286)
```

No edge from `dim_customer` to itself. The self-read in Step 2 is lost.

**After (Option C):**

```
Step 1 (MERGE):
  staging_customer_latest.id ──[merge,write]──► dim_customer.id
  staging_customer_latest.name ──[merge,write]──► dim_customer.name
  staging_customer_latest.city ──[merge,write]──► dim_customer.city
  dim_customer.id ──[merge,prior_state_read]──► dim_customer.end_time
  dim_customer.is_active ──[merge,prior_state_read]──► dim_customer.end_time

Cross-query (Step 1 output -> Step 2 self-read):
  dim_customer.id ──[cross_query]──► step2:self_read:dim_customer.id
  dim_customer.is_active ──[cross_query]──► step2:self_read:dim_customer.is_active

Step 2 (INSERT):
  staging_customer_latest.id ──[write]──► dim_customer.id
  staging_customer_latest.name ──[write]──► dim_customer.name
  step2:self_read:dim_customer.id ──[prior_state_read]──► dim_customer.id
  step2:self_read:dim_customer.is_active ──[prior_state_read]──► dim_customer.is_active
  step2:self_read:dim_customer.name ──[prior_state_read]──► dim_customer.name
```

The key difference: `dim_customer` columns that Step 2 reads from the LEFT JOIN flow through query-scoped `self_read` nodes, which are connected to Step 1's output via cross-query edges. The graph is a DAG (no self-loops).

## Implementation Sketch

### Phase 1: Stop filtering self-references

File: `multi_query.py`, method `_extract_source_tables`.

Change: when `table_name == destination_table`, do not `continue`. Instead, add the table to both `source_tables` and a new field `self_referenced_tables` on `ParsedQuery`.

```python
if table_name == destination_table:
    self_referenced = True
    # Still add to source_tables — it IS a source
```

### Phase 2: Cycle-safe dependency graph

File: `table.py`, method `_build_query_dependencies`.

Change: when computing deps for a query, exclude the query itself from its own dependency set. A query that writes to and reads from `dim_customer` should depend on *prior* queries that wrote `dim_customer`, not on itself.

```python
if table_node.created_by and table_node.created_by != query_id:
    deps[query_id].add(table_node.created_by)
for mod_id in table_node.modified_by:
    if mod_id != query_id:
        deps[query_id].add(mod_id)
```

### Phase 3: Self-read node creation

File: `pipeline_lineage_builder.py`, methods `_infer_table_name` and `_make_full_name`.

Change: for input-layer columns whose table is in `query.self_referenced_tables`, produce a query-scoped name like `{query_id}:self_read:{table_name}.{column_name}` instead of the shared physical name.

### Phase 4: Cross-query wiring

File: `pipeline_lineage_builder.py`, method `_add_cross_query_edges`.

Change: after the existing cross-query edge logic, add a pass that connects prior-statement output columns to self-read input columns. For each self-referenced table in a query, find the most recent query that wrote to that table (from topological order), and connect its output columns to the self-read nodes.

### Phase 5: Edge annotation

File: `models.py`, class `ColumnEdge`.

Change: add `statement_order: Optional[int] = None` and `edge_role: Optional[str] = None`. Populate these during Phase 3 and Phase 4.

## Scope

### In scope
- Multi-statement pipelines where consecutive DML statements target the same table and one reads the prior state.
- SCD Type 2 two-step pattern (MERGE + INSERT on same target).
- DELETE-then-INSERT (common anti-pattern that has the same shape).
- Single pipeline, linear execution order.

### Non-goals
- **Multi-branch pipelines** where two independent branches write to the same table. This requires a merge-point semantic that is out of scope.
- **Conditional execution** (e.g., IF/ELSE blocks in stored procedures). clgraph is a static parser; it cannot model runtime branching.
- **Cross-notebook state.** Each `Pipeline` instance is self-contained; table state does not carry across separate `Pipeline()` calls.
- **Idempotent re-runs.** The design assumes statements execute once in order. Re-execution semantics (e.g., Airflow retries) are not modeled.
- **Gap 7 (join-predicate columns).** The self-read nodes will not capture join-predicate influence on output columns. That is a separate design.
- **MERGE within a single statement that reads its own target.** A single MERGE statement's `USING` subquery referencing the target table is already handled by the query parser's alias resolution within one `QueryUnit`. This design addresses *cross-statement* self-reference only.

## Acceptance Criteria

### Structural

1. `ParsedQuery` has a `self_referenced_tables: Set[str]` field.
2. `ColumnEdge` has `statement_order: Optional[int]` and `edge_role: Optional[str]` fields.
3. `_extract_source_tables` does not filter `destination_table` from `source_tables` when the table appears in the query body (not just as the INSERT/MERGE target).
4. `_build_query_dependencies` does not create self-dependency cycles.
5. Self-read input columns use query-scoped naming (`{query_id}:self_read:{table}.{column}`).

### Functional

6. For the SCD2 two-step fixture (MERGE + INSERT on `dim_customer`), the pipeline graph contains:
   - Edges from `staging_customer_latest` columns to `dim_customer` output columns (both steps).
   - Self-read nodes for `dim_customer` columns read by Step 2.
   - Cross-query edges from Step 1's `dim_customer` output to Step 2's self-read nodes.
   - No self-loop edges (no edge where `from_node.table_name == to_node.table_name` on the same physical node with no intermediate).
7. Impact analysis: "what depends on `staging_customer_latest.city`?" returns `dim_customer.city` (both steps) and `dim_customer.end_time` (Step 1 close), `dim_customer.is_active` (Step 1 close).
8. Backward lineage: "where does `dim_customer.id` come from?" returns both `staging_customer_latest.id` (direct) and the self-read chain through prior `dim_customer.id`.

### Test Plan

Tests live in `tests/test_cdc_scd_pipeline.py` (created by the parent design's deliverable 1).

**Test 1: Self-reference detected.**
Given the SCD2 two-step SQL, assert that Step 2's `ParsedQuery.self_referenced_tables` contains `"dim_customer"` and `ParsedQuery.source_tables` contains `"dim_customer"`.

**Test 2: No topological cycle.**
Given the SCD2 two-step SQL, assert that `pipeline.table_graph.topological_sort()` succeeds (does not raise `CycleError`) and returns Step 1 before Step 2.

**Test 3: Self-read nodes exist.**
Assert that the pipeline column graph contains nodes matching `*:self_read:dim_customer.*` with `layer="input"`.

**Test 4: Cross-query edges connect prior output to self-read.**
Assert edges exist from `dim_customer.id` (output, Step 1) to the self-read node `*:self_read:dim_customer.id` (input, Step 2).

**Test 5: No self-loop.**
Assert that no edge in `pipeline.column_graph.edges` has `from_node.full_name == to_node.full_name`.

**Test 6: Impact analysis traversal.**
Starting from `staging_customer_latest.city`, forward-traverse the graph and assert `dim_customer.city` is reachable through both Step 1 and Step 2 paths.

**Test 7: DELETE-then-INSERT pattern.**
Same shape as SCD2 but using `DELETE FROM dim_customer WHERE ...` followed by `INSERT INTO dim_customer SELECT ...`. Assert self-read nodes and cross-query edges are created.

**Test 8: Non-self-referencing pipeline unchanged.**
A pipeline with `CREATE TABLE a AS SELECT ... FROM b; CREATE TABLE c AS SELECT ... FROM a` should produce zero self-read nodes and zero `edge_role="prior_state_read"` edges. Regression guard.

## Hostile Review

### Attack 1: API compatibility — self-read node names break existing user code

**Concern:** Users who call `pipeline.get_column("dim_customer", "id")` or iterate `pipeline.columns` filtering by `table_name == "dim_customer"` will not find the self-read nodes. Impact queries that walk edges will silently miss the self-read path.

**Response:** This is valid. The `get_column` API already accepts an optional `query_id` parameter (`pipeline.py:201`). Self-read nodes should be findable via `get_column("dim_customer", "id", query_id="step2")`. Additionally, impact-analysis traversal must be updated to follow cross-query edges through self-read nodes. The `LineageTracer` (lazy-loaded in `pipeline.py:148`) must be audited to confirm it traverses `edge_role="prior_state_read"` edges.

**Design revision:** Add a convenience method `pipeline.get_self_read_columns(table_name)` that returns all self-read nodes for a given physical table. Document that `pipeline.columns` includes self-read nodes (they are not hidden). Ensure `LineageTracer.trace_forward` and `trace_backward` traverse all edge roles by default.

### Attack 2: Graph rendering blow-up — self-read nodes double the node count for self-referenced tables

**Concern:** For a table with 50 columns referenced in 3 consecutive DML statements, this creates 100 extra self-read nodes (50 per self-referencing step). GraphViz and pyvis renders become cluttered.

**Response:** Accepted limitation with mitigation. The number of self-read nodes scales with `(number of self-referencing statements - 1) * columns_read`, not total columns. In practice, SCD2 Step 2 reads only the match columns (id, is_active) from the prior state, not all 50 columns. The column extractor only creates input nodes for columns actually referenced in the SQL.

For rendering, add an optional `collapse_self_reads=True` parameter to the GraphViz/pyvis exporters that merges self-read nodes back into their physical counterparts (restoring the current collapsed view) for visual simplicity while keeping the full graph for programmatic queries.

### Attack 3: Performance on large pipelines — extra pass over all queries for self-reference detection

**Concern:** The implementation adds a pass to detect self-referenced tables and a pass to wire cross-query edges for them. On a pipeline with 500 statements, this could be slow.

**Response:** The detection is O(1) per query (set intersection of `{destination_table}` and `source_tables`). The cross-query wiring pass iterates only self-referenced tables, not all tables. For a 500-statement pipeline with 2 self-referencing tables, the overhead is negligible compared to the existing lineage building which is O(queries * columns). No design revision needed; add a benchmark test for a 100-query pipeline to catch regressions.

### Attack 4: Dialect-specific semantics — MERGE in different dialects handles self-reference differently

**Concern:** In BigQuery, MERGE has snapshot isolation: the USING clause sees the target table as it was before the MERGE began. In PostgreSQL (via `MERGE` in v15+), the semantics are row-level with read-committed visibility. In Databricks Delta, MERGE uses snapshot isolation. The "read prior state" semantic encoded by self-read nodes may not match the actual dialect behavior.

**Response:** This is a real concern but is inherent to static analysis. clgraph does not execute SQL; it infers lineage from the SQL text. The "read prior state" assumption is correct for the dominant use case (Databricks/Delta SCD2) and is a reasonable default for all dialects. For dialects where MERGE has different visibility semantics, the lineage is still a useful approximation.

**Design revision:** Add a `dialect_note` to the self-read edge metadata explaining the assumption: "Assumes snapshot isolation: self-read sees table state before this statement executed." This is informational, not behavioral.

### Attack 5: User mental-model confusion — what IS a self-read node?

**Concern:** Users see `step2:self_read:dim_customer.id` in the graph and don't understand what it means. The naming convention (`query_id:self_read:table.column`) is internal jargon.

**Response:** Valid. The naming follows the existing convention for CTE nodes (`query_id:cte:name.column`) and subquery nodes (`query_id:subq:name.column`), both of which are already exposed in the graph. Users who understand CTE nodes will understand self-read nodes by analogy.

**Design revision:** Add a `node_type="self_read"` attribute to self-read `ColumnNode` instances (alongside the existing types like `"direct_column"`, `"expression"`, etc.). Renderers can use this to display a human-friendly label like "dim_customer (prior state)" instead of the raw full_name. Add documentation in the notebook's "How to read the graph" section.

### Attack 6: Test maintenance burden — existing tests need updating for new ColumnEdge fields

**Concern:** Adding `statement_order` and `edge_role` to `ColumnEdge` means every test that constructs or asserts on edges must account for new optional fields. If defaults are `None`, existing tests should pass, but assertion helpers that do exact-match comparisons may break.

**Response:** Both fields default to `None`. The `ColumnEdge` dataclass uses keyword arguments with defaults, so existing construction code is unaffected. Assertion helpers in tests use attribute access (`edge.is_merge_operation`, `edge.merge_action`), not positional matching. No existing test should break.

**Verification:** grep all test files for `ColumnEdge(` construction and `assert.*edge` patterns to confirm no exact-match comparisons exist.

### Attack 7: What happens when the same table is self-referenced within a SINGLE statement?

**Concern:** The design says "this addresses cross-statement self-reference only" (Non-goals). But a single `INSERT INTO t SELECT ... FROM t` is a valid SQL pattern. Does the current design accidentally create self-read nodes for single-statement self-references?

**Response:** No. Single-statement self-reference is handled within the `RecursiveQueryParser` / `RecursiveLineageBuilder` scope. The self-read detection in `multi_query.py` only fires when `destination_table` appears in `source_tables` of the same `ParsedQuery`, which is the per-statement level. For a single `INSERT INTO t SELECT ... FROM t`, the `_extract_source_tables` change will now include `t` in `source_tables` and mark it as self-referenced.

This is actually **desirable** — a single `INSERT INTO t SELECT ... FROM t` does read the prior state of `t`. The self-read node correctly models this. However, the single-statement case needs its own test to confirm the behavior is correct.

**Design revision:** Add Test 9 to the test plan: single-statement `INSERT INTO t SELECT ... FROM t` produces self-read nodes and no self-loop.

### Attack 8: The `modified_by` list ordering assumption

**Concern:** `table.py` stores `modified_by` as a `List[str]` but the design assumes that list ordering reflects execution order. If queries are added to the graph in non-execution order (e.g., user provides them out of order), the "most recent prior writer" lookup in Phase 4 would be wrong.

**Response:** Valid. `Pipeline.__init__` processes queries in the order provided by the user (`multi_query.py:147`, `for i, sql in enumerate(queries)`), and the `query_id` is `query_{i}`. The topological sort in `PipelineLineageBuilder.build` (line 68) processes queries in dependency order, not insertion order. The `modified_by` list therefore reflects insertion order, which may differ from execution order.

**Design revision:** Phase 4 (cross-query wiring) must use the topological sort order, not `modified_by` list order, to determine the "most recent prior writer." The `sorted_query_ids` list from `PipelineLineageBuilder.build` (line 68) is the correct ordering. Store it and pass it to the cross-query edge builder.

### Revised Test Plan Addition

**Test 9: Single-statement self-reference.**
`INSERT INTO t SELECT a, b FROM source LEFT JOIN t ON source.id = t.id` as a single-query pipeline. Assert self-read nodes exist for `t.id` and no self-loop edges.

**Test 10: Out-of-order query submission.**
Submit Step 2 (INSERT) before Step 1 (MERGE) in the pipeline constructor. Assert that topological sort still places MERGE before INSERT and self-read wiring is correct.
