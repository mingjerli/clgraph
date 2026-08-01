# clgraph Library - Simplified Architecture

## The 4-Stage Flow

```
┌────────────────────────────────────────────────────────────────┐
│                   1️⃣  SQL CODE INPUT                           │
│                                                                │
│   from_sql_files() | from_sql_string() | from_sql_list()     │
│        from_dict() | Pipeline([(id, sql), ...])              │
│                                                                │
└───────────────────────────────┬────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                   2️⃣  PIPELINE OBJECT                          │
│                                                                │
│          Parses SQL → Builds lineage → Creates graphs         │
│                                                                │
└───────────────────────────────┬────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                   3️⃣  TWO GRAPH TYPES                          │
│                                                                │
│  ┌──────────────────────┐    ┌──────────────────────┐        │
│  │ pipeline.table_graph │    │ pipeline.column_graph│        │
│  │                      │    │                      │        │
│  │ Table-level          │    │ Column-level         │        │
│  │ dependencies         │    │ lineage              │        │
│  └──────────────────────┘    └──────────────────────┘        │
│                                                                │
└───────────────────────────────┬────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                   4️⃣  APPLICATIONS                             │
│                                                                │
│  🚀 Orchestrator     📖 Data Catalog    🏷️  Metadata          │
│  • to_airflow_dag() • trace_backward()  • propagate_metadata()│
│  • run()            • trace_forward()   • get_pii_columns()   │
│  • async_run()      • get_lineage_path()• tags & ownership    │
│  • build_subpipeline()                                        │
│                                                                │
│  🤖 LLM Apps         💾 Export           🔄 Serialization      │
│  • generate_        • to_json()         • from_json()         │
│    descriptions()   • CSVExporter       • from_json_file()    │
│                     • visualize_*()                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Quick Example

```python
from clgraph import Pipeline
from clgraph.export import CSVExporter

# 1. Input SQL
pipeline = Pipeline.from_sql_files("examples/sql_files/", dialect="duckdb")

# 2. Pipeline object created automatically

# 3. Access graphs
table_deps = pipeline.table_graph  # Table dependencies
col_lineage = pipeline.column_graph  # Column lineage

# 4. Use applications

# Data Catalog - trace lineage
sources = pipeline.trace_column_backward("mart_customer_ltv", "lifetime_revenue")
impact = pipeline.trace_column_forward("raw_orders", "total_amount")

# Metadata - track PII
pipeline.columns["raw_customers.email"].pii = True
pipeline.propagate_all_metadata()
pii_cols = list(pipeline.get_pii_columns())

# Export
data = pipeline.to_json()
```

**Additional capabilities (require extra dependencies):**

<!-- skip-test -->
```python
# Orchestration (requires executor function)
result = pipeline.run(executor=my_execute_sql, max_workers=4)
dag = pipeline.to_airflow_dag(executor=my_execute_sql, dag_id="my_pipeline")

# LLM-powered descriptions (requires langchain)
from langchain_openai import ChatOpenAI

pipeline.llm = ChatOpenAI()
pipeline.generate_all_descriptions()
```

## Key Concepts

### Input Flexibility
Multiple ways to load SQL:
- **Files**: Directory of .sql files
- **String**: Semicolon-separated SQL
- **List**: Array of SQL statements
- **Dict/Tuples**: Structured query definitions

### Dual-Level Analysis
- **Table Graph**: Which tables depend on which tables
- **Column Graph**: Which columns derive from which columns

### Rich Applications
- **Run pipelines**: Sync, async, or Airflow DAG
- **Trace lineage**: Backward and forward analysis
- **Manage metadata**: PII, ownership, tags with auto-propagation
- **AI-powered docs**: Auto-generate column descriptions
- **Export**: JSON, CSV, GraphViz (via visualize_*() functions)

## Use Cases

| Use Case | Pipeline Method | Output |
|----------|----------------|---------|
| Execute data pipeline | `pipeline.run()` | Execution results |
| Create Airflow DAG | `to_airflow_dag()` | Airflow DAG |
| Build subset of pipeline | `build_subpipeline()` | Filtered Pipeline |
| Find data sources | `trace_column_backward()` | Source columns |
| Impact analysis | `trace_column_forward()` | Affected columns |
| Find path between columns | `get_lineage_path()` | Column path |
| Track PII | `propagate_all_metadata()` | Auto-propagated flags |
| Generate docs | `generate_all_descriptions()` | AI descriptions |
| Export lineage | `to_json()` | JSON/CSV/DOT files |
| Load from JSON | `from_json()` | Pipeline object |
