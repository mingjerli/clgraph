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

**3. Shared column naming in `pipeline_lineage_builder.py:708-736` (`_make_full_name`, physical-table return at line 736).**
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
dim_customer.id ──[snapshot]──► {query_id}:snapshot:dim_customer.id
{query_id}:snapshot:dim_customer.id ──► dim_customer.id   (Step 2 output, via INSERT)
staging.id ──► dim_customer.id          (Step 2 output)
```

**Pros:** Physical names unchanged for most queries; only self-referencing queries see snapshot nodes.
**Cons:** Snapshot naming is internal jargon; adds a new node type that renderers and exporters must handle. The snapshot-to-physical edge direction is confusing (is it "reads from" or "becomes"?).

### Option C: Read-Before-Write Detection with Ordered DML Edges

**Idea:** Do not create new nodes. Instead, recognize self-referencing tables at the pipeline level, stop excluding them from `source_tables`, and annotate the resulting column edges with a `statement_order` attribute. The self-read edges are distinguished from normal edges by their `statement_order` and a new `edge_role` (`"prior_state_read"` for within-query self-reads, `"cross_query_self_ref"` for cross-query wiring, `None` for normal edges).

**Data model change:**
- `ParsedQuery` gains `self_referenced_tables: Set[str]` (tables that appear in both `destination_table` and `source_tables` for the same query, or across consecutive queries targeting the same table).
- `ColumnEdge` gains `statement_order: Optional[int]` and `edge_role: Optional[str]` (values: `"prior_state_read"`, `"cross_query_self_ref"`, or `None` for normal edges).
- `TableNode` gains `self_referencing_queries: List[str]` tracking which queries read-before-write.

**Scope of code touched:**
- `multi_query.py:286`: remove the `table_name == destination_table` filter; instead, when a source table matches the destination, add it to both `source_tables` and `self_referenced_tables`.
- `table.py:188-204`: when building query dependencies for a self-referencing query, the query depends on all *prior* modifications to the same table, but not on itself. Add cycle-prevention logic.
- `pipeline_lineage_builder.py`:
  - `_infer_table_name` (starts at line ~654): for input columns from a self-referenced table, qualify the node name with `{query_id}:self_read:{table_name}` to avoid colliding with the output node.
  - `_make_full_name`: add a branch for self-read input nodes.
  - `_add_query_edges`: stamp edges from self-read nodes with `edge_role="prior_state_read"` and `statement_order`.
  - `_add_cross_query_edges`: connect the prior output (from the previous statement) to the self-read input node of the current statement.
- `models.py`: add `statement_order` and `edge_role` to `ColumnEdge`; add `self_referenced_tables` to `ParsedQuery`.
- Exporters: include new edge attributes in JSON/GraphViz output.

**User-facing graph shape** (using concrete query IDs `query_0` for Step 1, `query_1` for Step 2):
```
                                    ┌── staging.id ──────────────────┐
                                    │                                ▼
staging.id ──► dim_customer.id ──► query_1:self_read:dim_customer.id ──► dim_customer.id
               (Step 1 output)     (Step 2 reads prior state)         (Step 2 output)
```

The self-read node is query-scoped and represents "dim_customer as it existed before this query ran." Cross-query edges connect Step 1's output `dim_customer.id` to Step 2's `{query_id}:self_read:dim_customer.id`.

**Pros:** Minimal new concepts; edges carry explicit semantics; physical output nodes keep their canonical names; no version numbering scheme.
**Cons:** Self-read nodes are still a new node variant; `{query_id}:self_read:dim_customer.id` naming needs renderer support.

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
Step 1 (MERGE, query_0):
  staging_customer_latest.id ──► dim_customer.id              (is_merge=true, edge_role=None)
  staging_customer_latest.name ──► dim_customer.name          (is_merge=true, edge_role=None)
  staging_customer_latest.city ──► dim_customer.city          (is_merge=true, edge_role=None)
  dim_customer.id ──► dim_customer.end_time                   (is_merge=true, edge_role=None)
  dim_customer.is_active ──► dim_customer.end_time            (is_merge=true, edge_role=None)

Cross-query (Step 1 output -> Step 2 self-read):
  dim_customer.id ──► query_1:self_read:dim_customer.id       (edge_role=cross_query_self_ref)
  dim_customer.is_active ──► query_1:self_read:dim_customer.is_active
                                                              (edge_role=cross_query_self_ref)

Step 2 (INSERT, query_1):
  staging_customer_latest.id ──► dim_customer.id              (edge_role=None)
  staging_customer_latest.name ──► dim_customer.name          (edge_role=None)
  query_1:self_read:dim_customer.id ──► dim_customer.id       (edge_role=prior_state_read)
  query_1:self_read:dim_customer.is_active ──► dim_customer.is_active
                                                              (edge_role=prior_state_read)
  query_1:self_read:dim_customer.name ──► dim_customer.name   (edge_role=prior_state_read)
```

Note: `is_merge_operation` and `merge_action` are existing `ColumnEdge` fields; `edge_role` is the new field added by this design. They are independent attributes on the same edge, not comma-separated compound values.

The key difference: `dim_customer` columns that Step 2 reads from the LEFT JOIN flow through query-scoped `self_read` nodes, which are connected to Step 1's output via cross-query edges. The graph is a DAG (no self-loops).

## Implementation Sketch

### Phase 1: Stop filtering self-references (source-scope only)

File: `multi_query.py`, method `_extract_source_tables`.

Change: when `table_name == destination_table`, check whether the `Table` node appears in a source scope (FROM, JOIN, USING subquery) rather than in the target slot. `sqlglot.ast.find_all(exp.Table)` walks into the target slot (`Insert.this`, `Merge.this`, `Update.this`), so naively removing the filter would mark every INSERT/MERGE as self-referencing even when the body never reads the target.

**Detection heuristic:** a Table node is a self-reference if it is a descendant of one of the source-scope AST containers: `exp.From`, `exp.Join`, `exp.Subquery`, or `exp.Merge.args["using"]` — but NOT if its immediate parent chain leads to `Insert.this`, `Merge.this`, `Update.this`, or `Create.this`.

```python
# Build set of Table nodes that sit in the target slot (not source scope).
# IMPORTANT: for INSERT INTO t (col1, col2) SELECT ... and CREATE TABLE t (col1 ...),
# sqlglot wraps the target in an exp.Schema (not exp.Table). The Schema's .this
# child is the actual exp.Table. We must collect both the Schema and its inner
# Table to avoid false-positive self-reference detection.
target_table_nodes = set()
if isinstance(ast, (exp.Create, exp.Insert, exp.Merge, exp.Update, exp.Delete)):
    if ast.this:
        # ast.this may be exp.Table, exp.Schema, or other node
        target_table_nodes.add(id(ast.this))
        # Also collect all Table nodes nested inside the target slot
        # (handles Schema -> Table, Schema -> Table with column list, etc.)
        for t in ast.this.find_all(exp.Table):
            target_table_nodes.add(id(t))

for table_node in ast.find_all(exp.Table):
    table_name = self._get_table_name(table_node, tokenizer)
    if not table_name:
        continue
    # Skip Table nodes that are the target slot itself
    if id(table_node) in target_table_nodes:
        continue
    # Self-reference: table appears in source scope with same name as destination
    if table_name == destination_table:
        self_referenced_tables.add(table_name)
        # Still add to source_tables — it IS a source
    # ... existing CTE alias filter ...
    tables.add(table_name)
```

**Note on DELETE statements:** `_extract_operation_and_destination` (`multi_query.py:212-257`) currently does not handle `exp.Delete`. The following changes are required:

(a) Add `DELETE = "DELETE"` to the `SQLOperation` enum (`models.py:817-841`) and include `DELETE` in `SQLOperation.is_dml()` so it is recognized as a DML operation. Note: the existing `DELETE_AND_INSERT = "DELETE+INSERT"` (`models.py:835`) is a pre-existing composite operation for the atomic delete-then-insert pattern, parsed as a single unit. The new `DELETE` is for standalone DELETE statements. Both enum values coexist and serve distinct purposes. `DELETE` must be added to the `is_dml()` method's member list alongside the existing DML operations.

(b) Add an `exp.Delete` branch to `_extract_operation_and_destination` that extracts the target table name from `exp.Delete.this` and returns `(SQLOperation.DELETE, table_name)`.

(c) Add `exp.Delete` to the `isinstance` check at `_extract_source_tables` line 274 (the one that sets `destination_table` via `_extract_operation_and_destination`), not just the `target_table_nodes` check. Without this, DELETE statements will not populate `destination_table`, and the self-reference detection logic will not fire for DELETE targets.

Without these three changes, DELETE statements will not be recognized as targeting a table, and Test 7 (DELETE-then-INSERT) will fail.

This ensures `INSERT INTO dim_customer SELECT ... FROM staging` does NOT mark `dim_customer` as self-referenced (the Table node is in the target slot only), while `INSERT INTO dim_customer SELECT ... FROM staging LEFT JOIN dim_customer` DOES (the JOIN's Table node is in source scope).

Acceptance criterion #3 updated to match: `_extract_source_tables` does not filter `destination_table` from `source_tables` when the table appears in a **source scope** (FROM/JOIN/USING), as detected by AST node identity rather than name matching alone.

### Phase 2: Cycle-safe dependency graph

#### Phase 2a: `_build_query_dependencies` self-exclusion

File: `table.py`, method `_build_query_dependencies`.

Change: when computing deps for a query, exclude the query itself from its own dependency set. A query that writes to and reads from `dim_customer` should depend on *prior* queries that wrote `dim_customer`, not on itself.

Note: this self-exclusion applies **globally** to all queries in `_build_query_dependencies`, not only to self-referencing ones. This is correct because a query should never depend on itself regardless of the reason. The guard is harmless for non-self-referencing queries (their `query_id` never appears in their own `modified_by`).

```python
if table_node.created_by and table_node.created_by != query_id:
    deps[query_id].add(table_node.created_by)
for mod_id in table_node.modified_by:
    if mod_id != query_id:
        deps[query_id].add(mod_id)
```

#### Phase 2b: `_build_table_dependencies` self-exclusion

File: `table.py`, method `_build_table_dependencies` (lines 206-243).

This method builds table-level dependencies and is used by `get_dependencies`, `get_downstream`, and `get_execution_order`. Without a fix, a self-referencing table like `dim_customer` would list itself as its own dependency (since it appears in both source and destination roles). This creates a spurious self-dependency cycle at the table level even after Phase 2a fixes the query level.

Change: when iterating source tables for a given table, skip entries where `source_table == table_name`:

```python
for source_table in query.source_tables:
    if source_table == table_name:
        continue  # skip self-dependency
    if source_table in self.tables:
        deps[table_name].add(source_table)
```

### Phase 3: Self-read node creation

Files: `pipeline_lineage_builder.py`, methods `_make_full_name`, `_add_query_columns`, and `_is_physical_table_column`.

**3a. Naming (`_make_full_name`, line 708).** Add a branch before the physical-table check: if the node is an input-layer column whose inferred table is in `query.self_referenced_tables`, return query-scoped naming instead of shared physical naming.

```python
def _make_full_name(self, node: ColumnNode, query: ParsedQuery) -> str:
    table_name = self._infer_table_name(node, query)
    unit_id = node.unit_id

    # Self-read input columns get query-scoped naming to avoid colliding
    # with the physical output node for the same table.column.
    #
    # Fallback: _infer_table_name may return None for ambiguous columns
    # (e.g., when no alias qualifier is present). For input-layer columns
    # in a query with self-referenced tables, fall back to node.table_name
    # (the raw alias/table name from the SQL AST) before giving up.
    if table_name is None and node.layer == "input":
        candidate = node.table_name
        if candidate and candidate in getattr(query, "self_referenced_tables", set()):
            table_name = candidate

    if (
        node.layer == "input"
        and table_name
        and table_name in getattr(query, "self_referenced_tables", set())
    ):
        return f"{query.query_id}:self_read:{table_name}.{node.column_name}"

    # ... existing physical / CTE / subquery branches unchanged ...
```

When `_infer_table_name` returns `None` for an input-layer column in a query with self-referenced tables, the implementation attempts to match by `node.table_name` (the raw alias from the SQL AST) before falling through to physical-table naming. Without this fallback, ambiguous columns would silently receive shared physical naming, causing the exact node collapse this design is intended to fix. The fallback is safe because `node.table_name` is only used when it matches a known self-referenced table; for non-self-referenced tables, the existing physical-table path applies as before.

The `getattr` guard ensures backward compatibility if `self_referenced_tables` is not yet populated on older `ParsedQuery` instances (e.g., during incremental rollout or tests that construct `ParsedQuery` manually without the new field).

**3b. Node attributes (`_add_query_columns`, line 285).** When creating the `ColumnNode` for a self-read input column, set `node_type="self_read"` on the pipeline-level node (not the query-lineage-level node). This enables renderers to display a human-friendly label like "dim_customer (prior state)." `operation` retains the original query-lineage-level `node.node_type` (e.g., `"direct_column"`, `"expression"`); only the pipeline-level `node_type` is set to `"self_read"`.

**Diagnostic note:** In `_add_query_columns`, when a column with a physical-table `full_name` is dropped due to the deduplication guard (`if full_name in pipeline.columns: continue`) AND the current query has non-empty `self_referenced_tables`, emit a debug-level log warning. This helps diagnose cases where self-read detection fails and a self-read column silently falls back to physical naming, colliding with an existing output node.

**Recommendation:** Both Phase 3a (`_make_full_name`) and Phase 3b (`_add_query_columns`) independently compute whether a node is a self-read column using slightly different expressions. To avoid drift if one evolves without the other, extract a shared helper `_is_self_read_column(node, query) -> bool` that encapsulates the self-read detection logic (`node.layer == "input"` and inferred table name is in `query.self_referenced_tables`). Both call sites should delegate to this helper.

```python
# Inside the node creation loop in _add_query_columns:
is_self_read = (
    node.layer == "input"
    and (self._infer_table_name(node, query) or node.table_name)
        in getattr(query, "self_referenced_tables", set())
)

column = ColumnNode(
    column_name=node.column_name,
    table_name=self._infer_table_name(node, query) or node.table_name,
    full_name=full_name,
    query_id=query.query_id,
    unit_id=node.unit_id,
    node_type="self_read" if is_self_read else node.node_type,
    layer=node.layer,
    # ... remaining fields unchanged ...
)
```

**3d. Alias resolution for self-referenced tables.** When SQL aliases a self-referenced table (e.g., `LEFT JOIN dim_customer t`), the query-lineage-level `ColumnNode` has `table_name="t"` (the alias), not `"dim_customer"`. Phase 3a's fallback checks `node.table_name in self_referenced_tables`, but `self_referenced_tables` contains `"dim_customer"`, not `"t"`. Without alias resolution, aliased self-references are never detected as self-reads.

Alias-to-table resolution belongs in `MultiQueryParser`, where `_get_table_name` and the `TemplateTokenizer` are available. The resolved mapping is stored on `ParsedQuery` so that `PipelineLineageBuilder` can consume it without needing AST-walking or tokenizer access.

**During parsing** (`MultiQueryParser._parse_single_query` or `_extract_source_tables`): after populating `self_referenced_tables`, build the alias mapping in the same scope that already walks `exp.Table` nodes and calls `_get_table_name`.

```python
# Inside _extract_source_tables (or _parse_single_query), after self_referenced_tables is populated:
self_ref_aliases: Dict[str, str] = {}
for table_node in ast.find_all(exp.Table):
    resolved_name = self._get_table_name(table_node, tokenizer)
    if resolved_name in self_referenced_tables:
        alias = table_node.alias
        if alias:
            self_ref_aliases[alias] = resolved_name

# Store on ParsedQuery:
# parsed_query.self_ref_aliases = self_ref_aliases
```

`ParsedQuery` gains a new field: `self_ref_aliases: Dict[str, str]` — a mapping from SQL alias to resolved table name, populated only for tables in `self_referenced_tables`.

**At `PipelineLineageBuilder` level** (Phase 3a fallback and Phase 3b): read `query.self_ref_aliases` directly — no AST walking or `_get_table_name` calls needed.

```python
# In the self-read check (Phase 3a fallback and condition):
candidate = table_name or node.table_name
resolved = query.self_ref_aliases.get(candidate, candidate)
if node.layer == "input" and resolved in query.self_referenced_tables:
    return f"{query.query_id}:self_read:{resolved}.{node.column_name}"
```

The `self_ref_aliases` mapping must be available to both Phase 3a (`_make_full_name`) and Phase 3b (`_add_query_columns`). Store it as a per-query local or pass it through the shared helper `_is_self_read_column`. All `candidate in self_referenced_tables` checks in Phases 3a and 3b must resolve through this alias mapping first.

**3c. Physical-table exclusion (`_is_physical_table_column`, line 757).** Self-read input columns must NOT be classified as physical table columns (which would route them to shared naming in `_make_full_name`). The current `_make_full_name` code calls `_is_physical_table_column` at line 732 and branches on it at line 734 **before** the self-read check would execute. Therefore, `_make_full_name` must be restructured so the self-read branch (from Phase 3a) is evaluated **before** the existing `_is_physical_table_column` call. The Phase 3a pseudocode already shows this correct ordering (the self-read check appears first); the implementation must match that ordering by moving the `_is_physical_table_column` call below the self-read branch.

### Phase 4: Cross-query wiring (column-granular, topo-ordered)

File: `pipeline_lineage_builder.py`, method `_add_cross_query_edges`.

Change: the self-read cross-query wiring needs its own dedicated loop (or a separate method such as `_add_self_read_cross_query_edges`), because the existing `_add_cross_query_edges` loop starts with `if not table_node.created_by: continue`, which skips DML-only tables. MERGE/INSERT targets have `created_by=None` (they use `modified_by`), so the existing loop filters out the exact tables this feature targets. Do not attempt to add self-read wiring inside the existing loop.

The new loop connects prior-statement output columns to self-read input columns. For each self-referenced table in a query, find the most recent query **that wrote each specific column** (from topological order, not `modified_by` list order), and connect that column's output node to the self-read input node. This lookup assumes all output nodes already exist when Phase 4 executes; this is guaranteed by the existing `PipelineLineageBuilder.build()` call sequence, which calls `_add_query_columns` (creating all output nodes) before `_add_cross_query_edges` (wiring cross-query relationships).

**Note on DELETE queries:** DELETE queries produce no column lineage because `_extract_select_from_query` returns `None` for `exp.Delete`. The cross-query wiring loop naturally skips them (no output nodes to connect from). No special handling is needed — the loop simply finds no output columns for DELETE queries when searching for "most recent writer" of a given column.

Column-granular lookup matters when multiple prior statements write to the same table but touch different columns. Example: Step 1 writes `id` + `name`, Step 2 writes only `name`, Step 3 self-reads both — Step 3's `self_read:id` should wire to Step 1 (only writer of `id`), while `self_read:name` wires to Step 2 (most recent writer of `name`).

Implementation: the `sorted_query_ids` list from `PipelineLineageBuilder.build` (line 68, topological sort output) is the authoritative execution order. Pass it to the cross-query edge builder. For each self-read column, walk `sorted_query_ids` in reverse up to (but not including) the current query, and find the most recent query whose output nodes include that `{table_name}.{column_name}`. Connect that output node to the self-read input node. If no prior writer is found for a column, connect to the original source-state node (the physical input column from the table's pre-pipeline state). If no such pre-pipeline source-state node exists in `pipeline.columns` (e.g., the table is only ever self-referenced and was never an explicit source in a non-self-referencing query), the implementation must create one — a physical input `ColumnNode` with `full_name="{table_name}.{column_name}"`, `layer="input"`, representing the table's state before the pipeline began.

### Phase 5: Edge annotation

File: `models.py`, class `ColumnEdge`.

Change: add `statement_order: Optional[int] = None` and `edge_role: Optional[str] = None`. Populate these during Phase 3 and Phase 4.

`statement_order` is the **topological sort index** (0-based position in the `sorted_query_ids` list from `PipelineLineageBuilder.build`), not the insertion order or `query_id` string. This matches the execution-order semantics required by Phase 4 and aligns with Attack 8's design revision.

`statement_order` is populated on **all** edges (not only self-read edges) for consistency — every edge carries the topo sort index of the query that produced it. This enables consumers to sort or filter edges by execution order without checking `edge_role` first. To make this possible, `PipelineLineageBuilder` stores `sorted_query_ids` as an instance attribute (set once during `build()` after topological sort completes), and `_add_query_edges` reads `self.sorted_query_ids` to look up the index for the current `query_id` when stamping each edge.

`edge_role` values: `"prior_state_read"` (self-read edge within a query), `"cross_query_self_ref"` (cross-query edge connecting prior output to self-read input), or `None` (normal edge). These values are string literals, consistent with the existing `edge_type` convention on `ColumnEdge` (which also uses string literals like `"lineage"`, `"rename"`, etc.). An enum is not introduced to avoid a new import dependency for a three-value field.

### Phase 6: `get_self_read_columns` convenience method

File: `pipeline.py`, class `Pipeline`.

```python
def get_self_read_columns(self, table_name: str) -> list[ColumnNode]:
    return [
        node
        for node in self.column_graph.nodes
        if node.node_type == "self_read" and node.table_name == table_name
    ]
```

This method filters the pipeline's column graph for nodes tagged `node_type="self_read"` with a matching `table_name`, providing a convenient API for acceptance criterion 7. It returns an empty list for tables that are never self-referenced.

**`get_column` disambiguation:** After this change, `get_column(table_name, column_name)` could match both an output node and a self-read node for the same table and column. To avoid nondeterministic results, `get_column` should prefer `layer="output"` over `layer="input"` when multiple candidates match, regardless of whether `query_id` is specified. Even with a `query_id`, there can be two matches for the same `(table_name, column_name, query_id)` tuple: a self-read input node and an output node. Implementation: filter candidates as before, then prefer `layer="output"` over `layer="input"` when multiple matches exist.

Note on backward compatibility: `ColumnEdge.__eq__` (`models.py:632-639`) compares only `from_node`, `to_node`, and `edge_type` — not all fields. `__hash__` uses the same three fields. Adding optional fields with `None` defaults does not change equality or hash behavior. All existing test assertions use either attribute access (`edge.is_merge_operation`) or keyword construction (`ColumnEdge(from_node=..., to_node=..., edge_type=...)`) — neither is affected. Verified by grep: 14 `ColumnEdge(` constructions across `test_metadata_propagation.py` and `test_pipeline_diff.py`, all use keyword arguments with no positional args.

`ColumnNode.__hash__` and `__eq__` use `full_name` only, so self-read nodes with unique `{query_id}:self_read:*` names introduce no collision risk with existing physical or CTE column nodes.

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
- **MERGE within a single statement that reads its own target via USING.** A single MERGE statement's `USING` subquery referencing the target table is already handled by the query parser's alias resolution within one `QueryUnit`. The USING clause creates an explicit source alias, so the target and source are already distinct within the single-query lineage builder. This non-goal applies specifically to MERGE's USING clause. **Note:** single-statement `INSERT INTO t SELECT ... FROM t` is **in scope** — it has a genuine self-reference (the FROM/JOIN reads the target's prior state) that is not resolved by alias separation. Test 9 covers this case. The distinction: MERGE's USING clause already separates target from source by design; INSERT's FROM clause does not.

## Acceptance Criteria

### Structural

1. `ParsedQuery` has a `self_referenced_tables: Set[str]` field and a `self_ref_aliases: Dict[str, str]` field (mapping SQL aliases to resolved table names for self-referenced tables only, populated during parsing).
2. `ColumnEdge` has `statement_order: Optional[int]` and `edge_role: Optional[str]` fields.
3. `_extract_source_tables` does not filter `destination_table` from `source_tables` when the table appears in a **source scope** (FROM/JOIN/USING), as detected by AST node identity (target-slot vs. source-scope), not name matching alone. Table nodes in the target slot (`Insert.this`, `Merge.this`, etc.) are still excluded.
4. `_build_query_dependencies` does not create self-dependency cycles.
5. Self-read input columns use query-scoped naming (`{query_id}:self_read:{table}.{column}`).
6. Self-read `ColumnNode` instances have `node_type="self_read"`.
7. `Pipeline` exposes `get_self_read_columns(table_name: str) -> List[ColumnNode]` returning all self-read nodes for a given physical table.
8. `Pipeline.trace_column_forward` and `Pipeline.trace_column_backward` traverse all edge roles (including `"prior_state_read"` and `"cross_query_self_ref"`) by default. Note: the underlying BFS functions in `lineage_tracer.py` are module-level functions, not methods on a `LineageTracer` class. Verify that these functions traverse all edge types without filtering by `edge_role`.

### Functional

9. For the SCD2 two-step fixture (MERGE + INSERT on `dim_customer`), the pipeline graph contains:
   - Edges from `staging_customer_latest` columns to `dim_customer` output columns (both steps).
   - Self-read nodes for `dim_customer` columns read by Step 2.
   - Cross-query edges from Step 1's `dim_customer` output to Step 2's self-read nodes.
   - No self-loop edges (no edge where `from_node.table_name == to_node.table_name` on the same physical node with no intermediate).
10. Impact analysis: "what depends on `staging_customer_latest.city`?" returns `dim_customer.city` (both steps) and `dim_customer.end_time` (Step 1 close), `dim_customer.is_active` (Step 1 close).
11. Backward lineage: "where does `dim_customer.id` come from?" returns both `staging_customer_latest.id` (direct) and the self-read chain through prior `dim_customer.id`.

### Test Plan

Tests live in `tests/test_cdc_scd_pipeline.py` (created by the parent design's deliverable 1).

**Test 1: Self-reference detected.**
Given the SCD2 two-step SQL, assert that Step 2's `ParsedQuery.self_referenced_tables` contains `"dim_customer"` and `ParsedQuery.source_tables` contains `"dim_customer"`.

**Test 2a: No topological cycle.**
Given the SCD2 two-step SQL, assert that `pipeline.table_graph.topological_sort()` succeeds (does not raise `CycleError`) and returns Step 1 before Step 2.

**Test 2b: Direct self-exclusion in `_build_query_dependencies`.**
Given the SCD2 two-step SQL, call `pipeline.table_graph._build_query_dependencies()` directly and assert that `deps["query_1"]` does not contain `"query_1"` (no self-dependency). This tests the Phase 2 fix independently of topological sort's cycle-handling behavior.

**Test 3: Self-read nodes exist.**
Assert that the pipeline column graph contains nodes matching `*:self_read:dim_customer.*` with `layer="input"`.

**Test 4: Cross-query edges connect prior output to self-read.**
Assert edges exist from `dim_customer.id` (output, Step 1) to the self-read node `*:self_read:dim_customer.id` (input, Step 2).

**Test 5: No self-loop.**
Assert that no edge in `pipeline.column_graph.edges` has `from_node.full_name == to_node.full_name`.

**Test 6: Impact analysis traversal.**
Starting from `staging_customer_latest.city`, forward-traverse the graph and assert `dim_customer.city` is reachable through both Step 1 and Step 2 paths.

**Test 7: DELETE-then-INSERT pattern.**
Using `DELETE FROM dim_customer WHERE ...` followed by `INSERT INTO dim_customer SELECT ... FROM staging LEFT JOIN dim_customer ...`. Assert:
(a) DELETE is recognized as DML targeting `dim_customer` (requires `_extract_operation_and_destination` to handle `exp.Delete`).
(b) INSERT's self-read nodes exist for the `dim_customer` columns referenced in the LEFT JOIN.
(c) INSERT's self-read nodes wire to the original source-state `dim_customer` columns (pre-pipeline), not to DELETE output — because DELETE produces no column lineage (no output columns).
(d) No cross-query edges from DELETE to INSERT exist (since DELETE has no column output nodes to connect from).

**Test 8: Non-self-referencing pipeline unchanged.**
A pipeline with `CREATE TABLE a AS SELECT ... FROM b; CREATE TABLE c AS SELECT ... FROM a` should produce zero self-read nodes, zero `edge_role="prior_state_read"` edges, **and `self_referenced_tables == set()` on every `ParsedQuery`**. This locks down both the detection step (ParsedQuery field) and the wiring step (graph nodes/edges) independently.

**Test 9: Single-statement self-reference.**
`INSERT INTO t SELECT a, b FROM source LEFT JOIN t ON source.id = t.id` as a single-query pipeline. Assert self-read nodes exist for `t.id` and no self-loop edges.

**Test 10: `statement_order` reflects topo sort output for independent statements.**
Submit two DML statements that both target `dim_customer` but have no inter-dependency (neither reads a table the other creates — e.g., both read only from `staging_customer_latest`). Since there is no dependency edge between them, topo sort may return them in any valid order. The test asserts that `statement_order` on edges matches the topo sort output (whatever it is), not the submission order. Specifically: retrieve `sorted_query_ids` from the pipeline's topological sort, then for each edge verify that `edge.statement_order` equals the index of `edge.query_id` in `sorted_query_ids`. Note: for independent statements the topo sort order may coincide with submission order — the test validates the assignment mechanism, not forced reordering. Forced reordering requires an explicit dependency between statements (covered by Tests 2a, 4, and 11).

**Test 11: Column-granular cross-query wiring (three-step chain).**
Three-query pipeline: Step 1 writes `{id, name, city}` to `dim_customer`, Step 2 writes only `{name}` to `dim_customer`, Step 3 self-reads `{id, name}` from `dim_customer`. Assert that Step 3's `self_read:id` wires to Step 1's output (the only writer of `id`) and `self_read:name` wires to Step 2's output (the most recent writer of `name`). This validates Phase 4's per-column "most recent writer" resolution.

**Test 12: INSERT with explicit column list does not spuriously self-reference.**
`INSERT INTO dim_customer (id, name, city) SELECT s.id, s.name, s.city FROM staging s` (no self-read in the body). Assert `self_referenced_tables == set()` — the target's `exp.Schema`-wrapped Table node must be excluded by Phase 1's `target_table_nodes` set.

**Test 13: MERGE with USING does not spuriously self-reference.**
`MERGE INTO dim_customer t USING staging s ON t.id = s.id WHEN MATCHED THEN UPDATE SET t.name = s.name WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name)` as a single-query pipeline. Assert `self_referenced_tables == set()`. The MERGE's USING clause already separates target from source by design (the target `t` is resolved via `Merge.this`, which is in the target slot). This validates the non-goal boundary claim that MERGE's USING-based self-reference is handled by existing alias resolution and should not trigger self-read node creation. Additionally, the test should include a structural assertion: verify that `_extract_source_tables` returns only `{"staging"}` (not `{"staging", "dim_customer"}`) for the MERGE statement, confirming the target's `exp.Table` node is correctly identified as target-slot-only and excluded from `source_tables`.

**Test 14: `get_self_read_columns` API.**
Given the SCD2 two-step SQL, assert that `pipeline.get_self_read_columns("dim_customer")` returns a non-empty list of `ColumnNode` instances, each with `node_type="self_read"` and `table_name="dim_customer"`. For the non-self-referencing pipeline from Test 8, assert it returns an empty list.

**Test 15: LineageTracer traverses self-read edges.**
Given the SCD2 two-step SQL, use `pipeline.trace_column_forward("staging_customer_latest", "city")` and assert the result includes `dim_customer.city` reached via the self-read path (through `query_1:self_read:dim_customer.*` nodes). Similarly, `pipeline.trace_column_backward("dim_customer", "id")` must include the self-read chain. This validates acceptance criterion 8 (Pipeline traversal).

**Test 16: Aliased self-reference detected.**
`INSERT INTO dim_customer SELECT s.id, s.name, s.city, s.email FROM staging s LEFT JOIN dim_customer t ON s.id = t.id WHERE t.id IS NULL` as a single-query pipeline. The alias `t` refers to `dim_customer`. Assert that self-read nodes are created for the aliased reference (e.g., `query_0:self_read:dim_customer.id`), not silently missed because `node.table_name == "t"` fails to match `self_referenced_tables == {"dim_customer"}`. This validates Phase 3d's alias resolution.

## Hostile Review

### Attack 1: API compatibility — self-read node names break existing user code

**Concern:** Users who call `pipeline.get_column("dim_customer", "id")` or iterate `pipeline.columns` filtering by `table_name == "dim_customer"` will not find the self-read nodes. Impact queries that walk edges will silently miss the self-read path.

**Response:** This is valid. The `get_column` API already accepts an optional `query_id` parameter (`pipeline.py:201`). Self-read nodes should be findable via `get_column("dim_customer", "id", query_id="query_1")`. Additionally, impact-analysis traversal must be updated to follow cross-query edges through self-read nodes. The `LineageTracer` (lazy-loaded in `pipeline.py:148`) must be audited to confirm it traverses `edge_role="prior_state_read"` edges.

**Design revision:** Add a convenience method `pipeline.get_self_read_columns(table_name)` that returns all self-read nodes for a given physical table. Document that `pipeline.columns` includes self-read nodes (they are not hidden). Ensure `LineageTracer.trace_column_forward` and `trace_column_backward` traverse all edge roles by default.

### Attack 2: Graph rendering blow-up — self-read nodes double the node count for self-referenced tables

**Concern:** For a table with 50 columns referenced in 3 consecutive DML statements, this creates 100 extra self-read nodes (50 per self-referencing step). GraphViz and pyvis renders become cluttered.

**Response:** Accepted limitation with mitigation. The number of self-read nodes scales with `(number of self-referencing statements - 1) * columns_read`, not total columns. In practice, SCD2 Step 2 reads only the match columns (id, is_active) from the prior state, not all 50 columns. The column extractor only creates input nodes for columns actually referenced in the SQL.

For rendering, a future follow-up could add an optional `collapse_self_reads=True` parameter to the GraphViz/pyvis exporters that merges self-read nodes back into their physical counterparts (restoring the current collapsed view) for visual simplicity while keeping the full graph for programmatic queries. **This exporter enhancement is deferred to a separate PR** to keep this change focused on correctness. The core graph is always complete; rendering collapse is a presentation concern. **Tracked as:** Deferred Item D1 in the Deferred Work section below.

### Attack 3: Performance on large pipelines — extra pass over all queries for self-reference detection

**Concern:** The implementation adds a pass to detect self-referenced tables and a pass to wire cross-query edges for them. On a pipeline with 500 statements, this could be slow.

**Response:** The detection is O(1) per query (set intersection of `{destination_table}` and `source_tables`). The cross-query wiring pass iterates only self-referenced tables, not all tables. For a 500-statement pipeline with 2 self-referencing tables, the overhead is negligible compared to the existing lineage building which is O(queries * columns). No design revision needed; add a benchmark test for a 100-query pipeline to catch regressions.

### Attack 4: Dialect-specific semantics — MERGE in different dialects handles self-reference differently

**Concern:** In BigQuery, MERGE has snapshot isolation: the USING clause sees the target table as it was before the MERGE began. In PostgreSQL (via `MERGE` in v15+), the semantics are row-level with read-committed visibility. In Databricks Delta, MERGE uses snapshot isolation. The "read prior state" semantic encoded by self-read nodes may not match the actual dialect behavior.

**Response:** This is a real concern but is inherent to static analysis. clgraph does not execute SQL; it infers lineage from the SQL text. The "read prior state" assumption is correct for the dominant use case (Databricks/Delta SCD2) and is a reasonable default for all dialects. For dialects where MERGE has different visibility semantics, the lineage is still a useful approximation.

**Design revision:** Add a `dialect_note` to the self-read edge metadata explaining the assumption: "Assumes snapshot isolation: self-read sees table state before this statement executed." This is informational, not behavioral. **Tracked as:** Deferred Item D2 in the Deferred Work section below. The `dialect_note` field is a nice-to-have that does not affect correctness; it can land in a follow-up PR without blocking this design.

### Attack 5: User mental-model confusion — what IS a self-read node?

**Concern:** Users see `{query_id}:self_read:dim_customer.id` in the graph and don't understand what it means. The naming convention (`query_id:self_read:table.column`) is internal jargon.

**Response:** Valid. The naming follows the existing convention for CTE nodes (`query_id:cte:name.column`) and subquery nodes (`query_id:subq:name.column`), both of which are already exposed in the graph. Users who understand CTE nodes will understand self-read nodes by analogy.

**Design revision:** Add a `node_type="self_read"` attribute to self-read `ColumnNode` instances (alongside the existing types like `"direct_column"`, `"expression"`, etc.). Renderers can use this to display a human-friendly label like "dim_customer (prior state)" instead of the raw full_name. Add documentation in the notebook's "How to read the graph" section.

### Attack 6: Test maintenance burden — existing tests need updating for new ColumnEdge fields

**Concern:** Adding `statement_order` and `edge_role` to `ColumnEdge` means every test that constructs or asserts on edges must account for new optional fields. If defaults are `None`, existing tests should pass, but assertion helpers that do exact-match comparisons may break.

**Response:** Both fields default to `None`. The `ColumnEdge` dataclass uses keyword arguments with defaults, so existing construction code is unaffected.

**Verification (completed):** Grep results confirm safety:
- 14 `ColumnEdge(` constructions across `test_metadata_propagation.py` (12) and `test_pipeline_diff.py` (2). All use keyword arguments (`from_node=..., to_node=..., edge_type=...`), no positional args.
- `ColumnEdge.__eq__` (`models.py:632-639`) compares only `from_node`, `to_node`, and `edge_type` — not all fields. `__hash__` (`models.py:629-630`) uses the same three fields.
- All test edge assertions use attribute access (`edge.is_merge_operation`, `edge.qualify_function`, etc.) or count checks (`len(edges) == N`), not full-object equality comparisons.
- No existing test should break.

### Attack 7: What happens when the same table is self-referenced within a SINGLE statement?

**Concern:** The design says "this addresses cross-statement self-reference only" (Non-goals). But a single `INSERT INTO t SELECT ... FROM t` is a valid SQL pattern. Does the current design accidentally create self-read nodes for single-statement self-references?

**Response:** No. Single-statement self-reference is handled within the `RecursiveQueryParser` / `RecursiveLineageBuilder` scope. The self-read detection in `multi_query.py` only fires when `destination_table` appears in `source_tables` of the same `ParsedQuery`, which is the per-statement level. For a single `INSERT INTO t SELECT ... FROM t`, the `_extract_source_tables` change will now include `t` in `source_tables` and mark it as self-referenced.

This is actually **desirable** — a single `INSERT INTO t SELECT ... FROM t` does read the prior state of `t`. The self-read node correctly models this. However, the single-statement case needs its own test to confirm the behavior is correct.

**Design revision:** Add Test 9 to the test plan: single-statement `INSERT INTO t SELECT ... FROM t` produces self-read nodes and no self-loop.

### Attack 8: The `modified_by` list ordering assumption

**Concern:** `table.py` stores `modified_by` as a `List[str]` but the design assumes that list ordering reflects execution order. If queries are added to the graph in non-execution order (e.g., user provides them out of order), the "most recent prior writer" lookup in Phase 4 would be wrong.

**Response:** Valid. `Pipeline.__init__` processes queries in the order provided by the user (`multi_query.py:147`, `for i, sql in enumerate(queries)`), and the `query_id` is `query_{i}`. The topological sort in `PipelineLineageBuilder.build` (line 68) processes queries in dependency order, not insertion order. The `modified_by` list therefore reflects insertion order, which may differ from execution order.

**Design revision:** Phase 4 (cross-query wiring) must use the topological sort order, not `modified_by` list order, to determine the "most recent prior writer." The `sorted_query_ids` list from `PipelineLineageBuilder.build` (line 68) is the correct ordering. Store it and pass it to the cross-query edge builder.

*(Tests 9–15 are in the main Test Plan section above.)*

## Deferred Work

Items deferred from this design to keep the PR focused on correctness. Each should be tracked as a follow-up issue or PR.

**D1: Exporter collapse mode for self-read nodes.**
Add an optional `collapse_self_reads=True` parameter to GraphViz/pyvis exporters that merges self-read nodes back into their physical counterparts for visual simplicity. Origin: Attack 2.

**D2: `dialect_note` metadata on self-read edges.**
Add a `dialect_note: Optional[str]` field to `ColumnEdge` populated with "Assumes snapshot isolation: self-read sees table state before this statement executed." for self-read edges. Informational only, no behavioral impact. Origin: Attack 4.

## Known Limitations

**Unqualified columns in self-referencing queries remain ambiguous.**
When a self-referencing query references columns without a table qualifier (e.g., bare `id` instead of `t.id`) and the query has multiple source tables, `_infer_table_name` may return `None` and `node.table_name` may also be `None`. In this case, the column falls through to physical naming and will not receive self-read treatment. This is consistent with existing behavior for any multi-source query where unqualified columns are ambiguous — it is not a regression introduced by this design. Users should qualify columns with table aliases in self-referencing queries for correct lineage tracking.
