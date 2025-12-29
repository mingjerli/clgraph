# ClickHouse Example: Enterprise Data Pipeline

This directory contains SQL files demonstrating a realistic 4-layer enterprise data pipeline for ClickHouse.

## Pipeline Architecture

```
Raw Layer (01-04)         Staging Layer (10-12)      Analytics Layer (20-22)    Marts Layer (30-31)
┌─────────────────┐       ┌──────────────────┐       ┌────────────────────┐     ┌──────────────────┐
│ raw_{{env}}.*   │ ───▶  │ staging_{{env}}.*│ ───▶  │ analytics_{{env}}.*│ ──▶ │ marts_{{env}}.*  │
└─────────────────┘       └──────────────────┘       └────────────────────┘     └──────────────────┘
```

## Files

| File | Table | Description |
|------|-------|-------------|
| `00_init_schema.sql` | - | Creates database schemas |
| `01_raw_orders.sql` | `raw_{{env}}.orders` | Raw order data |
| `02_raw_customers.sql` | `raw_{{env}}.customers` | Raw customer data |
| `03_raw_products.sql` | `raw_{{env}}.products` | Raw product catalog |
| `04_raw_order_items.sql` | `raw_{{env}}.order_items` | Raw order line items |
| `10_staging_orders.sql` | `staging_{{env}}.orders` | Cleaned orders with validation |
| `11_staging_customers.sql` | `staging_{{env}}.customers` | PII-masked customers |
| `12_staging_products.sql` | `staging_{{env}}.products` | Enriched product data |
| `20_analytics_customer_metrics.sql` | `analytics_{{env}}.customer_metrics` | Customer-level aggregations |
| `21_analytics_product_metrics.sql` | `analytics_{{env}}.product_metrics` | Product performance metrics |
| `22_analytics_daily_sales.sql` | `analytics_{{env}}.daily_sales` | Daily sales aggregations |
| `30_marts_customer_360.sql` | `marts_{{env}}.customer_360` | Complete customer view |
| `31_marts_sales_dashboard.sql` | `marts_{{env}}.sales_dashboard` | Sales dashboard metrics |

## Features Demonstrated

- **ClickHouse dialect**: `CREATE OR REPLACE TABLE`, `ENGINE = MergeTree()`, ClickHouse functions
- **Template variables**: `{{env}}` substituted at runtime (e.g., `dev`, `prod`)
- **Multi-query pipeline**: 13 queries with cross-query dependencies
- **Aggregations**: COUNT, SUM, AVG, MAX, MIN
- **CASE expressions**: Data validation and categorization
- **JOINs**: LEFT JOIN for combining data sources
- **GROUP BY**: Customer and product-level aggregations

## Usage

```python
from pathlib import Path
from clgraph import Pipeline

# Load SQL files
sql_dir = Path("examples/clickhouse_example")
queries = []
for sql_file in sorted(sql_dir.glob("*.sql")):
    if not sql_file.name.startswith("00_"):  # Skip init schema
        queries.append((sql_file.stem, sql_file.read_text()))

# Create pipeline with dev environment
pipeline = Pipeline.from_tuples(
    queries,
    dialect="clickhouse",
    template_context={"env": "dev"},
)

# Trace lineage
sources = pipeline.trace_column_backward("marts_dev.customer_360", "lifetime_value")
```

## Tests

These files are used by `tests/test_enterprise_demo.py` for integration testing.

```bash
uv run pytest tests/test_enterprise_demo.py -v
```
