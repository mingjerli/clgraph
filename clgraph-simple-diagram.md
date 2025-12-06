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
│  • async_run()      • get_lineage()     • tags & ownership    │
│                                                                │
│  🤖 LLM Apps         💾 Export                                 │
│  • generate_        • to_json()                               │
│    descriptions()   • CSVExporter                             │
│                     • GraphVizExporter                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Quick Example

```python
# 1. Input SQL
pipeline = Pipeline.from_sql_files("queries/", dialect="bigquery")

# 2. Pipeline object created automatically

# 3. Access graphs
table_deps = pipeline.table_graph    # Table dependencies
col_lineage = pipeline.column_graph  # Column lineage

# 4. Use applications

# Orchestration
dag = pipeline.to_airflow_dag(executor=execute_sql)
result = pipeline.run(executor=execute_sql)

# Data Catalog
sources = pipeline.trace_column_backward("final_table", "revenue")
impact = pipeline.trace_column_forward("raw_table", "user_id")

# Metadata
pipeline.columns["raw.email"].pii = True
pipeline.propagate_all_metadata()
pii_cols = pipeline.get_pii_columns()

# LLM
pipeline.llm = ChatOpenAI()
pipeline.generate_all_descriptions()

# Export
data = pipeline.to_json()
CSVExporter.export_columns_to_file(pipeline, "columns.csv")
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
- **Export**: JSON, CSV, GraphViz

## Use Cases

| Use Case | Pipeline Method | Output |
|----------|----------------|---------|
| Execute data pipeline | `pipeline.run()` | Execution results |
| Create Airflow DAG | `to_airflow_dag()` | Airflow DAG |
| Find data sources | `trace_column_backward()` | Source columns |
| Impact analysis | `trace_column_forward()` | Affected columns |
| Track PII | `propagate_all_metadata()` | Auto-propagated flags |
| Generate docs | `generate_all_descriptions()` | AI descriptions |
| Export lineage | `to_json()` | JSON/CSV/DOT files |
