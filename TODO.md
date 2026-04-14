# Real-World Data Pipeline Examples

Track progress on adding real-world data pipeline example notebooks.

## Examples

- [x] **A. dbt Jaffle Shop-style ELT** (`examples/jaffle_shop_elt_pipeline.ipynb`)
  - E-commerce pipeline: raw sources (customers, orders, payments) -> staging -> intermediate -> marts
  - Multi-source fan-in, dimension/fact modeling, PII tracking, impact analysis

- [ ] **B. CDC/SCD Pipeline** (`examples/cdc_scd_pipeline.ipynb`)
  - Debezium-style CDC: raw CDC events -> MERGE into slowly-changing dimensions -> fact tables -> aggregated metrics
  - Showcases MERGE lineage support with realistic use case

- [ ] **C. Multi-Domain Data Mesh** (`examples/data_mesh_pipeline.ipynb`)
  - Two independent domain pipelines (e-commerce + marketing) converging into cross-domain reporting
  - Template variables for multi-environment deployment + complex dependency graphs

## Architectural Gaps (blocking B)

Discovered during CDC/SCD design review (see `docs/superpowers/specs/2026-04-13-cdc-scd-pipeline-gaps-design.md` §Risks 5, 6). Both are cross-cutting changes that must land before the CDC/SCD notebook (item B) can honestly showcase lineage.

- [x] **Gap 4. Self-referencing target across statements** (statement-scoped table versioning)
  - Implemented in PR #61: self-read node detection via AST node identity, cycle-safe dependency resolution, query-scoped `{query_id}:self_read:{table}.{col}` naming, column-granular cross-query wiring, edge role/order annotations.
  - Design: `docs/superpowers/specs/2026-04-13-gap4-self-referencing-target-design.md`

- [ ] **Gap 7. JOIN ON predicate columns not recorded in column lineage**
  - Today: JOIN ON predicates produce **zero** column-lineage edges (no handling in `lineage_builder` for ON clause columns beyond the equi-join's identity resolution).
  - Symptom: point-in-time joins like `o.order_ts BETWEEN d.start_time AND d.end_time` leave `start_time`/`end_time` invisible as influences on downstream columns.
  - Needs its own design doc — new edge semantic for "predicate-conditional" columns.
