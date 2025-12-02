# E-Commerce SQL Pipeline Example

This example demonstrates clgraph's SQL lineage analysis and pipeline execution using DuckDB.

## Two Ways to Run

| Script | What it does |
|--------|--------------|
| `run_lineage.py` | **Lineage only** - Parses SQL, builds dependency graph, no execution |
| `run_with_duckdb.py` | **Full execution** - Generates fake data, executes pipeline against DuckDB |

## Quick Start

```bash
cd clgraph

# Lineage analysis only
uv run python examples/sql_files/run_lineage.py

# Full execution with fake data
uv run python examples/sql_files/run_with_duckdb.py
```

## Files

| File | Layer | Description |
|------|-------|-------------|
| `01_raw_orders.sql` | Raw | Order transactions |
| `02_raw_customers.sql` | Raw | Customer data (PII) |
| `03_raw_products.sql` | Raw | Product catalog |
| `04_raw_order_items.sql` | Raw | Order line items |
| `05_stg_orders_enriched.sql` | Staging | Enriched orders (CTEs, JOINs, window functions) |
| `06_int_daily_metrics.sql` | Intermediate | Daily aggregations (running totals) |
| `07_mart_customer_ltv.sql` | Mart | Customer lifetime value (RFM scoring) |
| `08_mart_product_perf.sql` | Mart | Product performance (rankings) |

## Pipeline Architecture

```
source_orders ─────┐
source_customers ──┼──► raw_* tables ──► stg_orders_enriched ──┬──► int_daily_metrics
source_products ───┤                                           ├──► mart_customer_ltv
source_order_items─┘                                           └──► mart_product_performance
```

## SQL Features Demonstrated

- **CTEs** - Multi-stage transformations within queries
- **Window Functions** - LAG, ROW_NUMBER, RANK, PERCENT_RANK, running totals
- **Aggregations** - GROUP BY, COUNT DISTINCT, AVG, SUM
- **Joins** - INNER JOIN, LEFT JOIN, CROSS JOIN
- **CASE statements** - Customer segmentation, RFM scoring

## Code Examples

### 1. Load SQL Files and Build Pipeline

```python
from pathlib import Path
from clgraph import Pipeline

# Load all SQL files
sql_dir = Path("examples/sql_files")
queries = []
for sql_file in sorted(sql_dir.glob("*.sql")):
    queries.append((sql_file.stem, sql_file.read_text()))

# Build pipeline
pipeline = Pipeline(queries, dialect="duckdb")
print(f"Queries: {len(pipeline.table_graph.queries)}")
print(f"Columns: {len(pipeline.columns)}")
```

### 2. Backward Lineage - Where does data come from?

```python
# Trace lifetime_revenue back to its sources
sources = pipeline.trace_column_backward("mart_customer_ltv", "lifetime_revenue")
for source in sources:
    print(f"  ← {source.table_name}.{source.column_name}")
```

### 3. Forward Lineage - What does this column affect?

```python
# See what downstream columns are affected by total_amount
impacts = pipeline.trace_column_forward("raw_orders", "total_amount")
for impact in impacts:
    print(f"  → {impact.table_name}.{impact.column_name}")
```

### 4. PII Tracking

```python
# Mark columns as PII
pipeline.columns["raw_customers.email"].pii = True
pipeline.columns["raw_customers.phone_number"].pii = True

# Propagate PII flags through lineage
pipeline.propagate_all_metadata()

# Find all PII columns (including propagated)
pii_cols = list(pipeline.get_pii_columns())
for col in pii_cols:
    print(f"  ⚠️ {col.table_name}.{col.column_name}")
```

### 5. Execute Pipeline Against DuckDB

```python
import duckdb
from clgraph import Pipeline

conn = duckdb.connect(":memory:")

# Create source tables with data first...
# (see run_with_duckdb.py for fake data generation)

# Build pipeline
pipeline = Pipeline(queries, dialect="duckdb")

# Define executor
def execute_sql(sql: str):
    conn.execute(sql)

# Run pipeline in dependency order
result = pipeline.run(executor=execute_sql, max_workers=1, verbose=True)

print(f"Completed: {len(result['completed'])}")
print(f"Failed: {len(result['failed'])}")
print(f"Time: {result['elapsed_seconds']:.2f}s")
```

## Expected Output (run_with_duckdb.py)

```
🚀 Starting pipeline execution (8 queries)

📊 Level 1: 4 queries
  ✅ 01_raw_orders
  ✅ 02_raw_customers
  ✅ 03_raw_products
  ✅ 04_raw_order_items

📊 Level 2: 1 queries
  ✅ 05_stg_orders_enriched

📊 Level 3: 3 queries
  ✅ 06_int_daily_metrics
  ✅ 07_mart_customer_ltv
  ✅ 08_mart_product_perf

✅ Pipeline completed in 0.06s
   Successful: 8
   Failed: 0
```

## Customization

Modify `run_with_duckdb.py` to:
- Change data volumes: `generate_fake_data(conn, num_customers=1000, num_orders=5000)`
- Use persistent database: `conn = duckdb.connect("my_db.duckdb")`
- Use async execution: `await pipeline.async_run(executor=async_execute_sql)`
