# clgraph

A Python library for parsing SQL queries into lineage graphs, enabling column-level dependency tracking, metadata propagation, and impact analysis.

## Features

### Column Lineage Analysis
- **Perfect column lineage** for any single SQL query, no matter how complex
- **Recursive query parsing** - handles arbitrary nesting of CTEs and subqueries
- **Bottom-up lineage building** - dependency-ordered processing
- **Star notation preservation** - no forced expansion, with EXCEPT/REPLACE support
- **Forward and backward lineage tracing** - impact analysis and source tracing

### Multi-Query Pipeline Analysis
- **Cross-query lineage** - trace columns through multiple dependent queries
- **Table dependency graphs** - understand pipeline structure
- **Template variable support** - handle parameterized SQL with {{variable}} syntax
- **Pipeline-level impact analysis** - see how changes propagate through your data pipeline

### Metadata Management
- **Column metadata** - track descriptions, ownership, PII flags, and custom tags
- **Metadata propagation** - automatically inherit metadata through lineage
- **Inline comment parsing** - extract metadata from SQL comments (`-- description [pii: true]`)
- **LLM integration** - generate natural language descriptions using Ollama, OpenAI, etc.
- **Diff tracking** - detect changes between pipeline versions

### Export Functionality
- **JSON export** - machine-readable format for system integration
- **CSV export** - column and table metadata for spreadsheets
- **GraphViz export** - DOT format for visualization tools

## Installation

```bash
pip install clgraph
```

Or with uv:
```bash
uv pip install clgraph
```

## Quick Start

### Single Query Column Lineage

```python
from clgraph import Pipeline

sql = """
WITH monthly_sales AS (
  SELECT
    user_id,
    DATE_TRUNC(order_date, MONTH) as month,
    SUM(amount) as total_amount
  FROM orders
  GROUP BY 1, 2
)
SELECT
  u.name,
  ms.month,
  ms.total_amount
FROM users u
JOIN monthly_sales ms ON u.id = ms.user_id
"""

# Pipeline works for single queries too
pipeline = Pipeline.from_sql_string(sql, dialect="bigquery")

# Get output columns from the query's lineage
query_lineage = pipeline.query_graphs["select"]
print(query_lineage)

# Get source tables
input_nodes = query_lineage.get_input_nodes()
source_tables = {node.table_name for node in input_nodes if node.table_name}
print("-"*60)
print(f"{len(source_tables)} source tables:")
for table in source_tables:
    print(f"  {table}")
```

### Multi-Query Pipeline Lineage

```python
from clgraph import Pipeline

queries = [
    ("raw_events", """
        CREATE TABLE raw_events AS
        SELECT user_id, event_type, event_timestamp, session_id
        FROM source_events
        WHERE event_timestamp >= '2024-01-01'
    """),
    ("daily_active_users", """
        CREATE TABLE daily_active_users AS
        SELECT user_id, DATE(event_timestamp) as activity_date, COUNT(*) as event_count
        FROM raw_events
        GROUP BY user_id, DATE(event_timestamp)
    """),
    ("user_summary", """
        CREATE TABLE user_summary AS
        SELECT u.name, u.email, dau.activity_date, dau.event_count
        FROM users u
        JOIN daily_active_users dau ON u.id = dau.user_id
    """),
]

pipeline = Pipeline(queries, dialect="bigquery")

# Show pipeline structure
print(f"Pipeline with {len(pipeline.table_graph.queries)} queries")
print("-" * 60)

# Table execution order
execution_order = pipeline.table_graph.get_execution_order()
print(f"Execution order ({len(execution_order)} tables):")
for i, table in enumerate(execution_order, 1):
    print(f"  {i}. {table}")

print("-" * 60)

# Trace a column backward through the pipeline
sources = pipeline.trace_column_backward("user_summary", "event_count")
print(f"Backward lineage for user_summary.event_count ({len(sources)} sources):")
for source in sources:
    print(f"  {source}")

print("-" * 60)

# Forward lineage / Impact analysis
impacts = pipeline.trace_column_forward("source_events", "event_timestamp")
print(f"Forward lineage for source_events.event_timestamp ({len(impacts)} impacts):")
for impact in impacts:
    print(f"  {impact}")
```

### Metadata from SQL Comments

```python
from clgraph import Pipeline

sql = """
SELECT
  user_id,  -- User identifier [pii: false]
  email,    -- Email address [pii: true, owner: data-team]
  COUNT(*) as login_count  -- Number of logins [tags: metric engagement]
FROM user_activity
GROUP BY user_id, email
"""

pipeline = Pipeline.from_sql_string(sql, dialect="bigquery")

# Metadata is automatically extracted from comments
print(f"Total columns: {len(pipeline.columns)}")
print("-" * 60)

pii_columns = [col for col in pipeline.columns.values() if col.pii]
print(f"PII columns ({len(pii_columns)}):")
for col in pii_columns:
    print(f"  {col.full_name}")
    if col.owner:
        print(f"    Owner: {col.owner}")
    if col.tags:
        print(f"    Tags: {', '.join(col.tags)}")

print("-" * 60)

# Show all column metadata
for col in pipeline.columns.values():
    if col.sql_metadata:
        print(f"{col.full_name}:")
        if col.sql_metadata.description:
            print(f"  Description: {col.sql_metadata.description}")
        if col.sql_metadata.pii is not None:
            print(f"  PII: {col.sql_metadata.pii}")
```

### Metadata Management and Export

```python
from clgraph import Pipeline, JSONExporter, CSVExporter, GraphVizExporter

# Build pipeline
queries = [
    ("raw.orders", """
        CREATE TABLE raw.orders AS
        SELECT order_id, user_email, amount, order_date
        FROM source.orders
    """),
    ("analytics.revenue", """
        CREATE TABLE analytics.revenue AS
        SELECT user_email, SUM(amount) as total_revenue
        FROM raw.orders
        GROUP BY user_email
    """),
]

pipeline = Pipeline(queries, dialect="bigquery")

# Set source metadata
for col in pipeline.columns.values():
    if col.table_name == "raw.orders" and col.column_name == "user_email":
        col.set_source_description("Customer email address")
        col.owner = "data-team"
        col.pii = True
        col.tags = {"contact", "sensitive"}

# Propagate metadata through lineage
pipeline.propagate_all_metadata()

# Find all PII columns
pii_columns = pipeline.column_graph.get_pii_columns()
print(f"Found {len(pii_columns)} PII columns:")
for col in pii_columns:
    print(f"  {col}")
    if col.owner:
        print(f"    Owner: {col.owner}")
    if col.tags:
        print(f"    Tags: {', '.join(col.tags)}")

print("-" * 60)

# Export to different formats
print("Exporting to multiple formats...")
JSONExporter.export_to_file(pipeline.column_graph, "lineage.json")
CSVExporter.export_columns_to_file(pipeline.column_graph, "columns.csv")
GraphVizExporter.export_to_file(pipeline.column_graph, "lineage.dot")
print("✓ Exported to lineage.json, columns.csv, lineage.dot")
```

### LLM-Powered Description Generation

```python
from clgraph import Pipeline
from langchain_ollama import ChatOllama

# Build pipeline
queries = [
    ("raw.orders", """
        CREATE TABLE raw.orders AS
        SELECT order_id, user_email, amount, order_date
        FROM source.orders
    """),
    ("analytics.revenue", """
        CREATE TABLE analytics.revenue AS
        SELECT user_email, SUM(amount) as total_revenue
        FROM raw.orders
        GROUP BY user_email
    """),
]

pipeline = Pipeline(queries, dialect="bigquery")

# Configure LLM (Ollama - free, local), or replace to any LangChain Chat models.
llm = ChatOllama(model="qwen3-coder:30b", temperature=0.3)
pipeline.column_graph.llm = llm

# Generate descriptions for all columns
print(f"Generating descriptions for {len(pipeline.columns)} columns...")
pipeline.generate_all_descriptions(verbose=True)

print("-" * 60)

# View generated descriptions
columns_with_descriptions = [
    col for col in pipeline.columns.values() if col.description
]
print(f"Generated descriptions for {len(columns_with_descriptions)} columns:")
for col in columns_with_descriptions:
    print(f"  {col.full_name}:")
    print(f"    {col.description}")
```

## Architecture

### Conceptual Structure

clgraph analyzes SQL through a hierarchical decomposition:

1. **Pipeline** - A collection of SQL statements that together form a data pipeline
   - Example: Multiple CREATE TABLE statements that depend on each other
   - Represents the entire data transformation workflow

2. **SQL Statement** - Each statement can break into multiple query units
   - Example: A CREATE TABLE statement with CTEs contains multiple query units
   - Typically mutates or creates a database object (table, view, etc.)

3. **Query Unit** - A SELECT statement representing a table-like object
   - Can be a main query, CTE (Common Table Expression), or subquery
   - Represents a temporary or real table in the dependency graph
   - Each query unit reads from tables and produces columns

4. **Column Expressions** - Within each query unit, individual column definitions
   - Example: `SUM(amount) as total_revenue` is a column expression
   - Represents the transformation logic for a single output column
   - Tracks dependencies on input columns

This hierarchy allows clgraph to trace column lineage at any level: from pipeline-wide dependencies down to individual expression transformations.

## Pipeline Graph Objects

A `Pipeline` contains two graph structures for lineage analysis:

<!-- skip-test -->
```python
from clgraph import Pipeline

pipeline = Pipeline(queries, dialect="bigquery")

# Two graph objects available:
# 1. Table-level graph (TableDependencyGraph)
# 2. Column-level graph (PipelineLineageGraph)
```

### Table Graph (`pipeline.table_graph`)

The `TableDependencyGraph` tracks table-level dependencies:

<!-- skip-test -->
```python
# Access tables and queries
pipeline.table_graph.tables      # Dict[str, TableNode]
pipeline.table_graph.queries     # Dict[str, ParsedQuery]

# Get source tables (external inputs, not created by any query)
source_tables = pipeline.table_graph.get_source_tables()

# Get final tables (not read by any downstream query)
final_tables = pipeline.table_graph.get_final_tables()

# Get upstream dependencies for a table
deps = pipeline.table_graph.get_dependencies("analytics.user_metrics")

# Get downstream tables (impact analysis)
downstream = pipeline.table_graph.get_downstream("raw.orders")

# Get query execution order (topologically sorted)
query_order = pipeline.table_graph.topological_sort()

# Get table execution order
table_order = pipeline.table_graph.get_execution_order()
```

### Column Graph (`pipeline.column_graph`)

The `PipelineLineageGraph` tracks column-level lineage:

<!-- skip-test -->
```python
# Access columns and edges
pipeline.column_graph.columns    # Dict[str, ColumnNode]
pipeline.column_graph.edges      # List[ColumnEdge]

# Backward compatible access (property aliases)
pipeline.columns  # Same as pipeline.column_graph.columns
pipeline.edges    # Same as pipeline.column_graph.edges

# Get source columns (no incoming edges)
source_cols = pipeline.column_graph.get_source_columns()

# Get final columns (no outgoing edges)
final_cols = pipeline.column_graph.get_final_columns()

# Get direct upstream columns (one hop back)
upstream = pipeline.column_graph.get_upstream("analytics.metrics.total_revenue")

# Get direct downstream columns (one hop forward)
downstream = pipeline.column_graph.get_downstream("raw.orders.amount")
```

### Full Lineage Tracing

For complete lineage (not just direct dependencies), use Pipeline methods:

<!-- skip-test -->
```python
# Trace backward to ultimate sources (recursive)
sources = pipeline.trace_column_backward("final_table", "metric")

# Trace forward to all impacts (recursive)
impacts = pipeline.trace_column_forward("raw.orders", "amount")

# Find specific lineage path between two columns
path = pipeline.get_lineage_path(
    "raw.orders", "amount",
    "analytics.metrics", "total_revenue"
)
```

## Supported SQL Dialects

Built on [sqlglot](https://github.com/tobymao/sqlglot), supporting:
- BigQuery
- PostgreSQL
- MySQL
- Snowflake
- Redshift
- DuckDB
- And many more

Specify dialect when creating the tracer:

<!-- skip-test -->
```python
tracer = SQLColumnTracer(sql, dialect="postgres")
pipeline = Pipeline(queries, dialect="snowflake")
```

## Why We Built This

Column lineage is notoriously difficult. Traditional tools reverse-engineer lineage from query logs and execution metadata, requiring expensive platform integration and complex infrastructure. Most open-source alternatives focus only on table-level lineage or single-query column analysis.

**Our insight**: When SQL is written with explicit column names and clear transformations (what we call "lineage-friendly SQL"), static analysis can provide *perfect* column lineage—without database access, without runtime integration, and without query logs.

We built clgraph to prove this approach works. By combining lineage-friendly SQL with perfect static analysis, we solve 90% of column lineage needs with 10% of the complexity of enterprise tools. No database required. No infrastructure to maintain. Just pure Python analyzing your SQL files.

**Read more**: [Why We Built This (Full Story)](https://clgraph.dev/concepts/why-we-built-this/)

## Use Cases

- **Data Governance**: Track data lineage for compliance and auditing
- **Impact Analysis**: Understand downstream effects of schema changes
- **PII Tracking**: Automatically identify and propagate PII flags through pipelines
- **Pipeline Optimization**: Identify unused columns and redundant transformations
- **Data Quality**: Trace data issues back to their source
- **Documentation**: Auto-generate data flow diagrams and column descriptions

## Development

```bash
# Clone the repository
git clone https://github.com/mingjerli/clgraph.git
cd clgraph

# Install dependencies with uv
uv pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/ tests/
ruff format src/ tests/
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## Credits

Built with:
- [sqlglot](https://github.com/tobymao/sqlglot) - SQL parsing and transpilation
- [LangChain](https://github.com/langchain-ai/langchain) - LLM integration
- Python's `graphlib` - Topological sorting for dependency resolution
