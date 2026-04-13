# Multi-Domain Data Mesh Pipeline: Gap Analysis & Example Notebook

**Date:** 2026-04-13
**TODO item:** C — `examples/data_mesh_pipeline.ipynb`
**Goal:** Stress-test clgraph against a realistic multi-domain dbt Mesh pipeline, surface gaps in cross-project column lineage and governance metadata, fix what's practical, and deliver a showcase notebook that honestly documents what works and what doesn't.

## Background

"Data mesh" is primarily an organizational pattern — independent domains publish data products that other domains consume through well-defined interfaces. In SQL-land, the most widely-adopted concrete implementation today is **dbt Mesh**: multiple dbt projects linked by `dependencies.yml`, using two-argument `{{ ref('project', 'model') }}` calls, governed by groups/access modifiers, and optionally by model contracts.

dbt Labs publishes a canonical three-project Jaffle Shop mesh:

- [`jaffle-shop-mesh-platform`](https://github.com/dbt-labs/jaffle-shop-mesh-platform) — shared staging layer (`stg_customers`, `stg_orders`, `stg_order_items`, `stg_products`, `stg_supplies`, `stg_locations`).
- [`jaffle-shop-mesh-finance`](https://github.com/dbt-labs/jaffle-shop-mesh-finance) — finance domain marts (`orders`, `order_items`, `products`, `supplies`) consuming platform.
- [`jaffle-shop-mesh-marketing`](https://github.com/dbt-labs/jaffle-shop-mesh-marketing) — marketing domain marts (`customers`, `locations`) consuming **both** platform and finance.

This is a real, documented, battle-tested pattern. It's also a close neighbour of clgraph's existing single-project `examples/jaffle_shop/` — the diff is exactly what we want to probe.

### Sample cross-project SQL (from jaffle-shop-mesh-finance/models/marts/order_items.sql)

```sql
with
  order_items as (select * from {{ ref('jaffle_shop_mesh_platform', 'stg_order_items') }}),
  orders     as (select * from {{ ref('jaffle_shop_mesh_platform', 'stg_orders') }}),
  products   as (select * from {{ ref('jaffle_shop_mesh_platform', 'stg_products') }}),
  supplies   as (select * from {{ ref('jaffle_shop_mesh_platform', 'stg_supplies') }}),
  order_supplies_summary as (
    select product_id, sum(supply_cost) as supply_cost from supplies group by 1
  ),
  joined as (
    select order_items.*, products.product_price, order_supplies_summary.supply_cost,
           products.is_food_item, products.is_drink_item, orders.ordered_at
    from order_items
    left join orders                  on order_items.order_id  = orders.order_id
    left join products                on order_items.product_id = products.product_id
    left join order_supplies_summary  on order_items.product_id = order_supplies_summary.product_id
  )
select * from joined
```

And `jaffle-shop-mesh-marketing/models/marts/customers.sql` consumes **finance** (one-arg `ref('orders')`, resolved via `dependencies.yml`) *plus* **platform** (`ref('stg_customers')`) — the file itself looks local, but is a cross-project consumer.

## Current clgraph State

- `Pipeline.from_dbt_models()` works on a **single** dbt project directory (verified by `tests/test_dbt_integration.py`, `examples/jaffle_shop/`).
- `multi_query.TemplateTokenizer` handles `{{ ref('table') }}` via a user-supplied `ref` callable in context — signature assumed to be `ref(model)`, one arg.
- Table identity is a single string; no project/package namespace.
- Metadata propagation (PII, owners) works within one pipeline; no concept of *domain* ownership or *access modifiers*.
- Project-level DAG: not a first-class concept (table-level is).

## Pipeline Shape (Test Fixture)

Vendored subset of the three dbt Labs mesh projects, placed under `examples/data_mesh/` with three sibling directories mirroring the real repos:

```
examples/data_mesh/
  platform/
    dbt_project.yml
    models/staging/__sources.yml, stg_customers.sql, stg_orders.sql, stg_order_items.sql,
                    stg_products.sql, stg_supplies.sql, stg_locations.sql
    models/_groups.yml        (owner: platform-eng, access: public)
  finance/
    dbt_project.yml
    dependencies.yml          (depends on: platform)
    models/marts/orders.sql, order_items.sql, products.sql, supplies.sql
    models/_groups.yml        (owner: finance-analytics, access: public)
  marketing/
    dbt_project.yml
    dependencies.yml          (depends on: platform, finance)
    models/marts/customers.sql, locations.sql
    models/_groups.yml        (owner: marketing-analytics, access: public)
```

Dependency graph: `platform → finance → marketing` plus `platform → marketing` (fan-in).

SQL is copied verbatim from the dbt Labs repos (small project, MIT/Apache-compatible license check required — if not redistributable, we regenerate minimal equivalents that exercise the same surface).

Multi-environment deployment is exercised via a `{{ target.schema }}`-style prefix resolved through `template_context`, producing e.g. `analytics_dev.stg_orders` vs `analytics_prod.stg_orders`.

## Gaps to Probe (with severity)

Each gap tested by loading the three projects into one (or several) `Pipeline` objects and asserting graph shape.

- **P** — parses but wrong lineage
- **I** — parses but lineage incomplete
- **F** — fails to parse / load

| # | Gap | Where | Expected |
|---|-----|-------|----------|
| 1 | **Two-argument `{{ ref('project', 'model') }}`** — `TemplateTokenizer` regex `\w+\([^)]*\)` captures it, but the user-supplied `ref` callable in tests is one-arg. Need convention for 2-arg ref + project-qualified table name | finance → platform | **F / I** — likely fails or silently drops project name |
| 2 | **Multi-project load API** — is there a `Pipeline.from_mesh([dir1, dir2, dir3])` or equivalent? Or must users stitch three pipelines manually? | load step | **F** (no API) |
| 3 | **Project-qualified table identity** — if `platform.stg_orders` and `finance.stg_orders` both existed, are they distinct nodes? Today, table name is a bare string | all | I (conflict would merge silently) |
| 4 | **Intra-project `ref('orders')` resolving across project boundary** via `dependencies.yml` (marketing's `ref('orders')` → finance's orders) | marketing → finance | I (clgraph has no concept of `dependencies.yml`) |
| 5 | **Fan-in across domains** — marketing's `customers` reads from both platform (`stg_customers`) and finance (`orders`, `order_items`). Does the pipeline DAG show both edges? | marketing | Likely works at table-level; verify column-level |
| 6 | **Project/domain ownership metadata** propagated from `_groups.yml` onto every node in that project | all | F (no reader for `_groups.yml`) |
| 7 | **Cross-domain PII propagation** — platform tags `stg_customers.email` as PII; does it flow into finance (never touches it) and marketing (joins on customer_id, may expose email)? | platform → marketing | Likely works if (1) resolves; verify |
| 8 | **Project-level DAG / rollup view** — ability to render `platform → finance → marketing` as a 3-node graph, not just the 12-table graph | export / viz | I (not a first-class concept) |
| 9 | **Access modifiers** (public/private/protected) — dbt convention; does clgraph flag or at least record them? | governance | F (no support) |
| 10 | **Multi-environment template variables** — `{{ target.schema }}.stg_orders` resolved differently for dev/prod. Does lineage remain stable under resolution? | template resolution | Likely works (existing template_variables_example covers single-env); verify cross-env equivalence |
| 11 | **Cross-project impact analysis query** — "if platform changes `stg_orders.ordered_at`, what marketing columns break?" — end-to-end traversal across three loaded projects | tracer | Depends on (1), (2), (3) |
| 12 | **Contract-aware consumption** — finance's `orders.yml` declares a contract; marketing consumes only contracted columns. Does clgraph flag violations? | governance | F (no support; out-of-scope candidate) |

**Probably-already-works (sanity checks):** single-project dbt load, one-arg `ref`, table-level DAG within a project, column lineage through joins, aggregate lineage.

## Deliverables

Three artifacts, in order:

### 1. Test fixture + runner (`tests/test_data_mesh_pipeline.py`)
- Vendored mesh project tree under `tests/fixtures/data_mesh/` (or `examples/data_mesh/` if shared with the notebook).
- One test per gap row above. Tests for unfixed gaps are `xfail` with the gap number. Xfail list *is* the living backlog.
- CI-runnable; no network required (fixture is local).

### 2. Gap fixes (scope-limited, each in its own commit)

**In-scope fixes** (small, well-bounded):

- **Gap 1:** extend `TemplateTokenizer` / dbt context to support 2-arg `ref('project', 'model')`. User-supplied `ref` callable receives both args; default behavior for tests is to return `"project__model"` or `"project.model"` (decision below). All existing one-arg `ref` call sites stay working.
- **Gap 2:** add `Pipeline.from_dbt_mesh(project_dirs: List[Path])` that loads each project and concatenates their queries with project-aware qualification. Reuses `from_dbt_models` per project.
- **Gap 3:** table identity keyed by `(project, table_name)` when loaded via mesh API; bare-string identity preserved for single-project mode (no breaking change).
- **Gap 6:** minimal `_groups.yml` reader that attaches `owner` and `access` attributes to each model's metadata, using the existing `MetadataManager` extension surface.

**Out-of-scope (documented as follow-ups in TODO.md):**

- **Gap 4:** full `dependencies.yml` resolution and cross-project `ref('orders')` where `orders` is not in the current project. This is non-trivial — requires a two-pass resolver. Notebook will use explicit 2-arg ref everywhere to sidestep.
- **Gap 8:** project-level DAG rollup view (nice to have, separate effort).
- **Gap 9:** access modifier enforcement (governance feature, separate effort).
- **Gap 12:** contract-aware consumption checks (governance feature, separate effort).

### 3. Showcase notebook (`examples/data_mesh_pipeline.ipynb`)

Structure:

1. **Narrative intro** — what a data mesh is, why dbt Mesh is the standard implementation, what we're testing.
2. **The three projects** — render the directory tree, show one representative SQL file per project with callouts on 2-arg ref, groups, access.
3. **Build + visualize** — `Pipeline.from_dbt_mesh([...])` → column and table graphs; project-qualified node names visible.
4. **Cross-domain queries** — 4-5 concrete examples:
   - Impact analysis across domains: "change `platform.stg_orders.ordered_at` → affected columns in finance and marketing?"
   - PII propagation: "which marketing columns are downstream of a platform PII column?"
   - Ownership rollup: "per domain, how many columns are owned?"
   - Fan-in check: "all upstream domains feeding `marketing.customers`."
5. **Multi-environment** — same pipeline resolved for dev vs. prod; show both graphs are isomorphic.
6. **Known limitations** — honest subsection for gaps 4, 8, 9, 12, each with a code snippet and the ideal answer. Links to xfails.

## Decision Points (resolve before implementation)

1. **Project-qualified table name format.** Options: (a) `project.table` — collides with schema-qualified names in some dialects; (b) `project__table` — dbt's relation naming convention for cross-project, safer; (c) structured tuple `(project, table)` carried alongside the string. Recommend **(b)** for string form plus **(c)** internally on node objects.
2. **Reuse `examples/data_mesh/` as both fixture and demo.** Fewer moving parts; the notebook and tests read the same SQL. Recommend **yes**.
3. **Vendor dbt Labs SQL verbatim, or regenerate.** Need license check. Recommend verbatim if Apache-2.0 / MIT; regenerate minimal equivalents otherwise. Placeholder: verify before writing the fixture.
4. **One `Pipeline` per mesh vs. three linked pipelines.** dbt Mesh's real runtime is *per-project*, with cross-project refs resolved at compile time. For clgraph, loading into **one** `Pipeline` keyed by (project, table) is simpler and matches how a lineage consumer wants to query. Recommend **one `Pipeline`**.

## Acceptance Criteria

- [ ] `tests/test_data_mesh_pipeline.py` exists, CI-green, with a test per gap row (xfail-tagged by gap number for unfixed gaps).
- [ ] `Pipeline.from_dbt_mesh(...)` public API documented with a docstring and exercised in the notebook.
- [ ] Each fix commit is TDD: failing test → fix → passing test.
- [ ] Notebook runs end-to-end via `run_all_notebooks.py`.
- [ ] Notebook "Known limitations" matches the current xfail list exactly.
- [ ] `TODO.md`: item C checked off; remaining gaps (4, 8, 9, 12) listed as new entries.
- [ ] No regression in `test_dbt_integration.py` (single-project path unchanged).

## Out of Scope

- Non-dbt mesh implementations (e.g., Airflow DAG-of-DAGs, bespoke orchestration).
- Runtime semantics — clgraph stays static-analysis-only, no compile-time dbt behavior simulation.
- Cross-dialect mesh (all three projects assumed same dialect).
- Semantic layer / metrics (dbt's `metrics/`), dbt Cloud-specific features, Snowflake RBAC wiring.
- Performance on very large meshes (>50 projects).

## Risks & Open Questions

1. **Risk: vendored SQL licensing.** dbt Labs repos — verify license before copying. If unclear, regenerate minimal equivalents that exercise the same clgraph surface.
2. **Risk: scope creep into `dependencies.yml` resolution (gap 4).** A full resolver is much bigger than the rest combined. Kept out of scope; the fixture uses explicit 2-arg refs everywhere so the notebook is complete without it. If user insists on 1-arg ref across project boundary, we stop and escalate.
3. **Risk: backwards compatibility of table identity.** Changing table keys is a minefield. Mitigation: mesh mode opts in explicitly via `from_dbt_mesh`; single-project mode keeps bare-string keys. Verified by keeping `test_dbt_integration.py` unchanged.
4. **Open: should cross-domain edges be visually distinguished in GraphViz output?** Nice-to-have for the notebook — deferred unless trivial.
5. **Open: how much of `_groups.yml` to support?** Minimal: `owner` and `access`. Anything deeper (group-scoped macros, config inheritance) = out of scope.
