# Real-World Data Pipeline Examples

Track progress on adding real-world data pipeline example notebooks.

## Examples

- [ ] **A. dbt Jaffle Shop-style ELT** (`examples/jaffle_shop_elt_pipeline.ipynb`)
  - E-commerce pipeline: raw sources (customers, orders, payments) -> staging -> intermediate -> marts
  - Multi-source fan-in, dimension/fact modeling, PII tracking, impact analysis

- [ ] **B. CDC/SCD Pipeline** (`examples/cdc_scd_pipeline.ipynb`)
  - Debezium-style CDC: raw CDC events -> MERGE into slowly-changing dimensions -> fact tables -> aggregated metrics
  - Showcases MERGE lineage support with realistic use case

- [ ] **C. Multi-Domain Data Mesh** (`examples/data_mesh_pipeline.ipynb`)
  - Two independent domain pipelines (e-commerce + marketing) converging into cross-domain reporting
  - Template variables for multi-environment deployment + complex dependency graphs
