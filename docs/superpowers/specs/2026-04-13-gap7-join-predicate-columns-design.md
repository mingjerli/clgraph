# Gap 7: JOIN ON Predicate Columns in Column Lineage

**Date:** 2026-04-13
**Parent:** `docs/superpowers/specs/2026-04-13-cdc-scd-pipeline-gaps-design.md` (Gap #7, Risk #6)
**TODO.md ref:** "Gap 7. JOIN ON predicate columns not recorded in column lineage"

## Problem

JOIN ON predicate columns produce **zero** column-lineage edges today. The equi-join identity resolution (`a.id = b.id`) implicitly handles value flow for columns that appear in both the ON clause and the SELECT list, but predicate-only columns -- those referenced in the ON clause but not projected -- are invisible in the column graph.

Concrete example from the CDC/SCD pipeline fixture:

```sql
INSERT INTO fact_orders
SELECT o.order_id, o.customer_id, o.order_ts, o.amount,
       d.city AS customer_city_at_order
FROM raw_orders o
LEFT JOIN dim_customer d
  ON o.customer_id = d.id
 AND o.order_ts BETWEEN d.start_time AND d.end_time;
```

Today: `fact_orders.customer_city_at_order` traces back to `dim_customer.city` only. The temporal predicate columns `dim_customer.start_time` and `dim_customer.end_time` have zero edges. An impact analysis asking "what breaks if `dim_customer.start_time` changes?" returns nothing -- a silent false-negative.

This is not CDC-specific. Any non-equi join condition has the same gap: range joins, band joins, function-based joins, and theta joins.

## Current Behavior: Code Analysis

### 1. JOIN parsing stops at table extraction

`query_parser.py:175-178` iterates JOIN clauses and delegates to `_parse_from_sources`:

```python
joins = select_node.args.get("joins", [])
for join in joins:
    self._parse_from_sources(join, unit, depth)
```

`_parse_from_sources` (`query_parser.py:1370-1415`) processes the `join.this` (the table/subquery being joined) to extract table names, aliases, and subqueries. The ON clause (`join.args.get("on")`) is **never accessed** in the non-MERGE path. No column references from the ON clause are extracted or stored on the `QueryUnit`.

### 2. The QueryUnit has no field for join predicates

`models.py:181-296` defines `QueryUnit`. There is no `join_predicates` or `join_conditions` field. The only JOIN-adjacent data stored is `alias_mapping` (alias -> table name) and `depends_on_tables` (table names). The ON clause's column references are discarded at parse time.

### 3. The lineage builder never sees ON-clause columns

`lineage_builder.py:130-158` processes each unit by extracting output columns (`_extract_output_columns`) and tracing their dependencies (`_trace_column_dependencies`). Both operate exclusively on SELECT-list expressions. The ON clause is not examined.

`_extract_source_column_refs` (`lineage_builder.py:889-948`) walks expression ASTs to find `exp.Column` nodes -- but it only receives SELECT-list expressions, never ON-clause expressions.

### 4. MERGE already has the pattern we need

`query_parser.py:608-619` extracts `match_columns` from MERGE's ON clause by walking `exp.EQ` nodes. More importantly, `trace_strategies.py:186-229` stamps every MERGE `ColumnEdge` with `merge_condition` (the raw SQL of the WHEN clause) and `merge_action`. The condition is recorded as **edge metadata** but its referenced columns are not emitted as upstream inputs.

The MERGE precedent tells us:

- The project already has edge metadata fields for condition context (`merge_condition` on `ColumnEdge`, `models.py:601`).
- The pattern of extracting columns from a predicate expression and emitting edges for them is mechanically straightforward -- `_extract_source_column_refs` already does the AST walk.
- The open question is not "can we extract them" but "what graph shape should they produce."

### 5. The `ColumnEdge` data model already has role-like fields

`ColumnEdge` (`models.py:556-628`) carries typed metadata for several edge categories: `is_merge_operation`/`merge_action`/`merge_condition`, `is_qualify_column`/`qualify_context`, `is_window_function`/`window_role`. A `join_predicate` metadata group would be consistent with the existing pattern.

## Design Options

### Option A: Flatten predicate columns as plain inputs

Treat every column in the ON clause as a plain upstream input of every projected column from the joined relation. For the BETWEEN example, `d.start_time` and `d.end_time` each get a `direct` edge to `output.customer_city_at_order`, same as `d.city`.

**Data model change:** None. Reuse existing `ColumnEdge` with `edge_type="direct"`.

**Code touched:**
- `query_parser.py`: extract ON-clause column refs and store on `QueryUnit` (new field `join_conditions`).
- `lineage_builder.py`: in `_process_unit`, after tracing SELECT-list deps, iterate join conditions and add edges from each ON-clause column to every output column sourced from the joined table.

**Graph shape:** Every ON-clause column fans out to every projected column from its side of the join. For a join with 3 predicate columns and 10 projected columns from the joined table, this produces 30 new edges.

**Trade-offs:**
- Pro: simplest implementation, no new edge types, existing impact queries work unchanged.
- Con: massive false-positive amplification. A wide dimension table (50 columns projected) with a 3-column join predicate adds 150 edges, all marked `direct`. Downstream consumers cannot distinguish "this column's value flows into the output" from "this column constrains which rows appear." Impact analysis becomes noisy.
- Con: semantically incorrect. `d.start_time` does not contribute its *value* to `customer_city_at_order` -- it constrains which *row* of `dim_customer` is selected. Calling this a `direct` edge misleads users.

### Option B: Add `join_predicate` edge role (recommended)

Add a new edge attribute group to `ColumnEdge` that tags predicate-derived edges, analogous to the existing `is_merge_operation` / `merge_condition` group. Predicate columns get edges to every projected column from the joined table, but each edge carries `is_join_predicate=True` and `join_condition` (the raw SQL of the ON clause).

**Data model change:** Three new optional fields on `ColumnEdge`:

```python
# --- JOIN Predicate Metadata ---
is_join_predicate: bool = False
join_condition: Optional[str] = None   # Raw SQL of the ON clause
join_side: Optional[str] = None        # "left" or "right" (which side of the join this column is on)
```

**Code touched:**
- `models.py`: add three fields to `ColumnEdge`.
- `query_parser.py`: in `_parse_select_statement`, after parsing JOINs, extract ON-clause column refs and store on `QueryUnit` as `join_predicates: List[JoinPredicateInfo]`.
- `models.py`: add `JoinPredicateInfo` dataclass (join_condition_sql, left_columns, right_columns, join_type).
- `lineage_builder.py`: in `_process_unit`, after step 4 (trace dependencies), add a new step to create join-predicate edges. For each join predicate, for each projected output column from the joined table, emit an edge from each ON-clause-only column to that output column with `is_join_predicate=True`.
- `trace_strategies.py`: no change needed (regular column tracing stays the same).

**Graph shape:** Same edge count as Option A, but every predicate edge is tagged. Impact analysis can filter: `if edge.is_join_predicate: skip` for value-only lineage, or include for full dependency analysis.

**Trade-offs:**
- Pro: preserves the distinction between value lineage and predicate lineage. Downstream tools (impact analysis, data quality checks) can opt in or out.
- Pro: consistent with the MERGE pattern (`is_merge_operation` + `merge_condition`).
- Pro: no graph-shape change -- nodes are still `table.column`, edges are still column-to-column. Existing queries, visualizations, and tests continue to work; they just see more edges (which they can filter).
- Con: still connects predicate columns to every projected column from the joined table, which can be noisy for wide tables. Mitigation: the `is_join_predicate` flag makes this filterable.
- Con: slightly more complex than Option A, but the delta is small (3 fields, 1 new dataclass).

### Option C: Emit predicate columns to a synthetic JOIN node

Instead of connecting predicate columns directly to projected output columns, create a synthetic intermediate node (e.g., `__join__raw_orders__dim_customer`) and route predicate columns there. Projected columns from the joined table also connect through this node.

**Data model change:** New synthetic `ColumnNode` type for join nodes. Edges from predicate columns point to the join node; edges from the join node point to projected columns.

**Code touched:**
- `models.py`: extend `ColumnNode` with a `is_join_node` flag or new layer type.
- `lineage_builder.py`: create synthetic join nodes and wire edges through them.

**Graph shape:** Introduces a new node type that does not correspond to any SQL column. The graph becomes a hypergraph with intermediary routing nodes.

**Trade-offs:**
- Pro: cleanest semantic model -- predicate columns and projected columns have different relationships to the join, and this makes it visible in graph topology.
- Con: breaks the fundamental invariant that every node is a `table.column`. Existing graph traversal, visualization, and serialization code all assume this. The blast radius is large.
- Con: makes simple impact queries harder -- "what depends on `dim_customer.start_time`?" now requires traversing through synthetic nodes.
- Con: no precedent in the codebase. MERGE, QUALIFY, WINDOW, GROUPING SETS all use edge metadata, not synthetic nodes.

### Option D: Store predicate columns on existing edges as metadata only (no new edges)

Instead of creating new edges, annotate existing value-lineage edges with the predicate columns that constrain the join. For example, the edge `dim_customer.city -> output.customer_city_at_order` would carry metadata listing `dim_customer.start_time, dim_customer.end_time` as predicate dependencies.

**Data model change:** New field `join_predicate_columns: Optional[List[str]]` on `ColumnEdge`.

**Code touched:** Similar to Option B for parsing; different wiring in `lineage_builder.py` (annotate existing edges instead of creating new ones).

**Trade-offs:**
- Pro: zero new edges, zero graph blow-up.
- Con: predicate columns are not first-class nodes in the graph. Impact analysis asking "what depends on `dim_customer.start_time`?" cannot answer by graph traversal -- it requires scanning all edges for metadata. This defeats the purpose of a graph representation.
- Con: if `start_time` is not projected anywhere, it appears in no node at all -- only as a string in edge metadata. You cannot trace through it.

## Recommendation: Option B

Option B (tagged predicate edges) is the recommended approach. Rationale:

1. **Consistent with existing patterns.** MERGE uses `is_merge_operation` + `merge_condition`. QUALIFY uses `is_qualify_column` + `qualify_context`. Window functions use `is_window_function` + `window_role`. A `is_join_predicate` + `join_condition` group follows the same convention.

2. **Preserves graph traversal.** Predicate columns become first-class nodes with edges. Impact analysis works by default (`dim_customer.start_time` -> `output.customer_city_at_order`). Users who want value-only lineage filter on `is_join_predicate`.

3. **Minimal blast radius.** Three new fields on `ColumnEdge`, one new dataclass, changes in two files (parser + builder). No new node types, no graph topology changes.

4. **Correct semantics.** The edge communicates "this column influences the output by constraining which row is selected" rather than "this column's value flows into the output." The `is_join_predicate` flag makes this distinction machine-readable.

### Worked Example: Before and After

**SQL:**
```sql
SELECT o.order_id, o.amount, d.city AS customer_city
FROM raw_orders o
LEFT JOIN dim_customer d
  ON o.customer_id = d.id
 AND o.order_ts BETWEEN d.start_time AND d.end_time
```

**Before (current):**

```
raw_orders.order_id ──────────────────> output.order_id
raw_orders.amount ────────────────────> output.amount
dim_customer.city ────────────────────> output.customer_city

(dim_customer.start_time: NO NODE, NO EDGE)
(dim_customer.end_time:   NO NODE, NO EDGE)
(raw_orders.customer_id:  NO NODE unless also projected)
(raw_orders.order_ts:     NO NODE unless also projected)
```

**After (Option B):**

```
raw_orders.order_id ──────────────────> output.order_id
raw_orders.amount ────────────────────> output.amount
dim_customer.city ────────────────────> output.customer_city

dim_customer.start_time ──[P]────────> output.customer_city
dim_customer.end_time ────[P]────────> output.customer_city
raw_orders.customer_id ───[P]────────> output.order_id
raw_orders.customer_id ───[P]────────> output.amount
raw_orders.customer_id ───[P]────────> output.customer_city
dim_customer.id ──────────[P]────────> output.customer_city
raw_orders.order_ts ──────[P]────────> output.order_id
raw_orders.order_ts ──────[P]────────> output.amount
raw_orders.order_ts ──────[P]────────> output.customer_city

[P] = is_join_predicate=True, join_condition="o.customer_id = d.id AND ..."
```

Note: the equi-join columns (`o.customer_id`, `d.id`) also get predicate edges. The equi-join case does not currently produce edges either (unless those columns are projected), so this is strictly additive.

**Refinement: scope predicate edges to the joined table's projected columns only.**

A predicate column from the left side of the join (`o.customer_id`, `o.order_ts`) influences which rows are selected from the *right* side (`dim_customer`). Therefore predicate columns from the left side should connect to projected columns sourced from the *right* side, and vice versa. This avoids the worst of the fan-out: `o.order_ts` connects to `output.customer_city` (sourced from `d`) but not to `output.order_id` (sourced from `o`).

**Refined graph:**

```
raw_orders.order_id ──────────────────> output.order_id
raw_orders.amount ────────────────────> output.amount
dim_customer.city ────────────────────> output.customer_city

dim_customer.start_time ──[P]────────> output.customer_city   (right-side pred -> right-side projected)
dim_customer.end_time ────[P]────────> output.customer_city   (right-side pred -> right-side projected)
dim_customer.id ──────────[P]────────> output.customer_city   (right-side pred -> right-side projected)
raw_orders.customer_id ───[P]────────> output.customer_city   (left-side pred -> right-side projected)
raw_orders.order_ts ──────[P]────────> output.customer_city   (left-side pred -> right-side projected)
```

This is the final recommended shape. Predicate columns from both sides connect to projected columns from the *opposite* side of the join (the side whose row selection they constrain). Self-side predicate columns (e.g., `d.start_time` constraining which `d` row is selected) also connect to their own side's projected columns, since they constrain the row that contributes the value.

**Final refined rule:** all ON-clause columns connect to all projected output columns from the join's non-FROM side (the joined table). For multi-way joins, each JOIN's predicate columns connect to projected columns from that specific JOIN's right-side table.

## Scope

### In scope

- Extract column references from JOIN ON clauses during parsing.
- Store join predicate info on `QueryUnit`.
- Emit `is_join_predicate=True` edges from predicate columns to projected output columns.
- Support: equi-joins, range joins (BETWEEN), theta joins (>, <, >=, <=, <>), function-wrapped join keys (e.g., `UPPER(a.name) = UPPER(b.name)`), compound predicates (AND/OR).
- All existing dialects (bigquery, postgres, snowflake, databricks, duckdb, spark, trino, redshift, mysql).

### Non-goals

- **USING clauses.** USING(col) is syntactic sugar for ON a.col = b.col. It can be handled as a follow-up; the AST shape is different (`exp.Using` vs `exp.On`).
- **NATURAL JOIN.** Implicit equi-join on all same-named columns. Requires schema knowledge to resolve. Out of scope.
- **LATERAL JOIN correlation.** Already handled by the lateral correlation path (`lineage_builder.py:160-162`). This design does not change that path.
- **Correlated subqueries in ON.** E.g., `ON a.id = (SELECT max(id) FROM ...)`. Rare in practice; the subquery parsing path already handles subqueries in WHERE/HAVING but not in ON. Follow-up.
- **CROSS JOIN.** No ON clause; nothing to extract.
- **Implicit joins (comma joins with WHERE predicate).** WHERE-clause predicates that act as join conditions are a separate problem. Out of scope.
- **Predicate push-down through CTEs.** If a CTE wraps a join, predicate edges are emitted at the CTE's query unit level. Cross-CTE propagation of the `is_join_predicate` flag is not attempted.

## Implementation Plan

### Phase 1: Parse join predicates (query_parser.py, models.py)

1. Add `JoinPredicateInfo` dataclass to `models.py`:
   ```python
   @dataclass
   class JoinPredicateInfo:
       condition_sql: str                              # Raw SQL of the ON clause
       columns: List[Tuple[Optional[str], str]]        # (table_ref, col_name) pairs
       join_type: str                                  # "inner", "left", "right", "full", "cross"
       right_table: Optional[str]                      # Name/alias of the joined (right-side) table
   ```

2. Add `join_predicates: List[JoinPredicateInfo]` field to `QueryUnit`.

3. In `query_parser._parse_select_statement`, after the JOIN loop (line 178), iterate joins again and extract ON-clause columns:
   ```python
   for join in joins:
       on_clause = join.args.get("on")
       if on_clause:
           cols = self._extract_join_predicate_columns(on_clause)
           join_type = self._get_join_type(join)
           right_table = self._get_join_right_table(join, unit)
           unit.join_predicates.append(JoinPredicateInfo(
               condition_sql=on_clause.sql(),
               columns=cols,
               join_type=join_type,
               right_table=right_table,
           ))
   ```

4. Implement `_extract_join_predicate_columns` by reusing `_extract_source_column_refs` or by walking `exp.Column` nodes in the ON clause.

### Phase 2: Emit predicate edges (lineage_builder.py, models.py)

1. Add three fields to `ColumnEdge` in `models.py`:
   ```python
   is_join_predicate: bool = False
   join_condition: Optional[str] = None
   join_side: Optional[str] = None
   ```

2. In `lineage_builder._process_unit`, add a new step after step 4 (window function edges):
   ```python
   # 9. Create join predicate edges
   if unit.join_predicates:
       self._create_join_predicate_edges(unit, output_cols)
   ```

3. Implement `_create_join_predicate_edges`:
   - For each `JoinPredicateInfo` on the unit:
     - Identify which output columns are sourced from the joined (right-side) table.
     - For each column ref in the predicate, resolve to a source node.
     - Emit a `ColumnEdge` from each predicate column to each right-side projected column, with `is_join_predicate=True`, `join_condition=info.condition_sql`, `edge_type="join_predicate"`.

### Phase 3: Tests

See Acceptance Criteria below.

## Acceptance Criteria and Test Plan

### Test 1: CDC/SCD2 point-in-time join (from parent design fixture)

```sql
SELECT o.order_id, o.customer_id, o.order_ts, o.amount,
       d.city AS customer_city_at_order
FROM raw_orders o
LEFT JOIN dim_customer d
  ON o.customer_id = d.id
 AND o.order_ts BETWEEN d.start_time AND d.end_time
```

Assert:
- `dim_customer.start_time` has an edge to `output.customer_city_at_order` with `is_join_predicate=True`.
- `dim_customer.end_time` has an edge to `output.customer_city_at_order` with `is_join_predicate=True`.
- `raw_orders.order_ts` has an edge to `output.customer_city_at_order` with `is_join_predicate=True`.
- `raw_orders.customer_id` has an edge to `output.customer_city_at_order` with `is_join_predicate=True`.
- `dim_customer.id` has an edge to `output.customer_city_at_order` with `is_join_predicate=True`.
- Existing value edges unchanged (`dim_customer.city` -> `output.customer_city_at_order`, `is_join_predicate=False`).

### Test 2: Band join (non-CDC)

```sql
SELECT e.event_id, e.event_ts, s.sensor_id, s.reading
FROM events e
INNER JOIN sensor_data s
  ON e.event_ts BETWEEN s.reading_ts - INTERVAL '5' MINUTE AND s.reading_ts + INTERVAL '5' MINUTE
 AND e.location_id = s.location_id
```

Assert:
- `sensor_data.reading_ts` has a predicate edge to projected columns from `sensor_data` (i.e., `output.sensor_id`, `output.reading`).
- `events.event_ts` has a predicate edge to projected columns from `sensor_data`.
- `events.location_id` and `sensor_data.location_id` have predicate edges.

### Test 3: Function-based join

```sql
SELECT a.id, b.name
FROM table_a a
INNER JOIN table_b b ON UPPER(a.key) = UPPER(b.key)
```

Assert:
- `table_a.key` has a predicate edge to `output.name` (right-side projected).
- `table_b.key` has a predicate edge to `output.name`.

### Test 4: Multi-join chain

```sql
SELECT a.id, b.val, c.label
FROM table_a a
INNER JOIN table_b b ON a.id = b.a_id
INNER JOIN table_c c ON b.id = c.b_id AND b.category = c.category
```

Assert:
- First join: `a.id` predicate edge to `output.val`; `b.a_id` predicate edge to `output.val`.
- Second join: `b.id` predicate edge to `output.label`; `c.b_id` predicate edge to `output.label`; `b.category` and `c.category` predicate edges to `output.label`.
- First join's predicates do NOT produce edges to `output.label` (they belong to a different join).

### Test 5: Existing equi-join tests still pass

All tests in `tests/test_join_types.py` must continue to pass. The new predicate edges are additive; no existing edge is removed or changed.

### Test 6: Impact analysis opt-in/opt-out

Using `SQLColumnTracer`, verify that:
- Forward lineage from `dim_customer.start_time` includes `customer_city_at_order` (when predicate edges are included).
- The `is_join_predicate` flag is accessible on returned edges for filtering.

### Test 7: Dialect consistency

The CDC BETWEEN join produces identical predicate edges across bigquery, postgres, snowflake, databricks.

## Hostile Review

### Attack 1: Graph blow-up on wide schemas

**Concern:** A fact table joining a 50-column dimension on 3 predicate columns. If all 50 dim columns are projected, that is 3 x 50 = 150 new predicate edges per join. With 5 dimension joins, that is 750 new edges. The graph becomes unwieldy.

**Response:** This is real but mitigated by three factors:

1. The scoping rule (predicate edges connect to the joined table's projected columns only, not all output columns) already limits fan-out. A query that projects 5 columns from a 50-column dimension gets 15 edges, not 150.

2. The `is_join_predicate` flag allows visualization and analysis tools to hide predicate edges by default and show them on demand. The default view stays clean.

3. The alternative (zero edges) is worse. A 150-edge graph with tagged noise is more useful than a graph that silently omits 3 columns that determine which rows appear.

**Accepted limitation:** Wide projections with many predicate columns will produce many edges. Tooling should default to hiding `is_join_predicate` edges in visualization. Document this in the API/notebook when the feature ships.

### Attack 2: False signal in impact analysis

**Concern:** "What downstream columns are affected if `dim_customer.start_time` changes?" Today: nothing (false negative). After this change: `customer_city_at_order` and everything downstream (false... positive? It is actually correct). But if the user interprets "affected" as "the value changes," a predicate column change does not change the *value* of `customer_city_at_order` -- it changes *which row's* `city` value appears. Is that "affected"?

**Response:** Yes, it is affected. If `start_time` changes, a different row of `dim_customer` may be selected, producing a different `city` value in the output. This is precisely the kind of impact that should surface. The edge is semantically correct.

However, the *type* of impact is different from a value-flow edge. A tool that distinguishes "value changed" from "row selection changed" can use `is_join_predicate` to make this distinction. The design supports both use cases without conflating them.

**No revision needed.** The flag provides the distinction; the default behavior (include predicate edges in impact analysis) is correct.

### Attack 3: Confusion between value lineage and predicate lineage for downstream consumers

**Concern:** A consumer iterating `graph.edges` today expects every edge to mean "this column's value flows into that column." After this change, some edges mean "this column constrains which row is selected." If the consumer does not check `is_join_predicate`, they get wrong results.

**Response:** This is the same concern that existed when `is_merge_operation`, `is_qualify_column`, and `is_window_function` were added. Those features also introduced edges with non-value-flow semantics (e.g., a QUALIFY partition column does not contribute its value to the output, it constrains which rows pass the filter). The project has established the pattern of "edge metadata disambiguates semantics; consumers who care must check."

**Revision:** Add a section to the feature's documentation (notebook, API docs) that explicitly lists the edge metadata fields that distinguish edge semantics. Consider adding a utility method `graph.get_value_edges()` that filters out predicate/qualify/window edges for consumers who want value-only lineage.

### Attack 4: Dialect quirks (Spark vs Snowflake vs Postgres ON-clause semantics)

**Concern:** Different dialects handle ON clauses differently. Spark allows non-equi joins but has performance implications. Snowflake has LATERAL join ON semantics. Postgres allows subqueries in ON. Does the parser handle all dialects consistently?

**Response:** The implementation relies on sqlglot's AST, which normalizes ON clauses across dialects into `exp.Join` -> `args["on"]` -> expression tree. Column extraction via `exp.Column` walking is dialect-agnostic. The dialect-specific parsing is handled by sqlglot before we see the AST.

Known exceptions:
- **Snowflake FLATTEN in JOIN:** Already handled by the LATERAL path; no ON clause to extract.
- **Spark USING clause:** Out of scope (see Non-goals).
- **Postgres `ON EXISTS (SELECT ...)`:** Subqueries in ON are out of scope.

**No revision needed for the core design.** The dialect consistency test (Test 7) will catch regressions.

### Attack 5: Interaction with MERGE `merge_condition` metadata

**Concern:** MERGE already has `merge_condition` on edges. This design adds `join_condition` on edges. Are they consistent or contradictory? Could a MERGE with a JOIN in its USING clause produce edges with both `merge_condition` and `join_condition` set?

**Response:** MERGE uses a separate code path (`trace_merge_columns` in `trace_strategies.py:186-229`). The MERGE ON clause is processed by `query_parser._parse_merge_statement` (line 608-619), not by the regular JOIN parsing. The two paths do not overlap:

- Regular SELECT with JOIN -> `join_predicates` on `QueryUnit` -> `is_join_predicate` edges.
- MERGE statement -> `match_columns` + WHEN conditions -> `is_merge_operation` edges with `merge_condition`.

A MERGE whose USING clause contains a JOIN (e.g., `MERGE INTO t USING (SELECT ... FROM a JOIN b ON ...) s ON ...`) would have the inner JOIN's predicates handled at the subquery's `QueryUnit` level, and the MERGE ON handled at the MERGE unit level. No single edge would carry both `merge_condition` and `join_condition`.

**No revision needed.** The paths are cleanly separated. Add a test for MERGE-with-JOIN-in-USING to confirm.

### Attack 6: Test churn

**Concern:** Existing tests assert specific edge counts or specific edge sets. Adding predicate edges will break tests that assert `len(graph.edges) == N` or that enumerate all edges.

**Response:** Code search of `tests/test_join_types.py` shows tests assert specific `(from, to)` pairs in `edges_dict`, not edge counts. The new predicate edges are additive -- they add new `(from, to)` pairs that existing tests do not assert on. No existing test should break.

However, tests that use `graph.edges` directly (e.g., `total_edges = [e for e in graph.edges if e.to_node.full_name == "output.total_amount"]`) will now include predicate edges in that list. If a test asserts `len(total_edges) == 1`, it could break if a predicate column also targets `output.total_amount`.

**Revision:** Before landing, run the full test suite against the implementation and fix any count-based assertions. The likely fix is to filter: `[e for e in graph.edges if e.to_node.full_name == X and not e.is_join_predicate]` in tests that care about value-only edges, or update count expectations.

### Attack 7: Self-join predicate edges

**Concern:** In a self-join (`SELECT a.id, b.name FROM users a JOIN users b ON a.manager_id = b.id`), the "left" and "right" tables are the same physical table. Does `join_side` still work? Does `right_table` resolve correctly?

**Response:** `right_table` is the alias of the joined table, not the physical table name. In this example, `right_table = "b"`. The alias mapping resolves `b` -> `users`. Predicate columns `a.manager_id` and `b.id` are correctly attributed to their respective alias sides. Output columns from `b` (`b.name`) get predicate edges from both `a.manager_id` and `b.id`.

**No revision needed.** The alias-based resolution handles self-joins correctly.

### Attack 8: Predicate columns that are also projected

**Concern:** If `o.customer_id` is both in the ON clause and in the SELECT list, it gets both a value edge (`raw_orders.customer_id -> output.customer_id`) and a predicate edge (`raw_orders.customer_id -> output.customer_city_at_order`). The value edge is correct; the predicate edge is also correct (it constrains which dim row is selected). But is the double-duty confusing?

**Response:** No. The edges go to different output columns. The value edge goes to `output.customer_id` (where it is projected); the predicate edge goes to `output.customer_city_at_order` (where it constrains the join). A column can legitimately serve both roles. The graph accurately represents this.

**No revision needed.**

### Attack 9: ON clause with OR creates ambiguous predicate semantics

**Concern:** `ON a.id = b.id OR a.alt_id = b.alt_id` -- all four columns are predicate columns, but the OR means only one pair needs to match. Does the design handle this correctly?

**Response:** The design extracts all column references from the ON clause regardless of logical structure (AND, OR, NOT). All four columns get predicate edges. This is conservative (over-connects) but correct: any of the four columns changing could change which rows match. The `join_condition` field carries the full SQL so downstream tools can inspect the logical structure if needed.

**Accepted limitation:** OR-based predicates produce the same predicate edges as AND-based predicates. The design does not model the logical structure of the predicate, only the column references. This is consistent with how `merge_condition` works today.

### Attack 10: Performance impact of edge creation

**Concern:** Large queries with many joins and wide projections could see significant edge count growth. Does this affect parse/build time or memory?

**Response:** Edge creation is O(predicate_columns * projected_columns_per_join). For typical queries (2-5 joins, 5-20 projected columns per join, 2-4 predicate columns per join), this adds 20-400 edges. The `ColumnLineageGraph.add_edge` method (`models.py:665`) does a constant-time dict insert and two list appends. The performance impact is negligible for realistic queries.

For adversarial cases (50 joins, 100 columns each), the edge count could reach tens of thousands. This is a pre-existing concern (star expansion on wide tables already produces similar edge counts) and is not specific to this change.

**Accepted limitation:** No performance guardrails added. Monitor in practice; add edge-count warnings if needed (similar to existing `UNQUALIFIED_STAR_MULTIPLE_TABLES` warnings).
