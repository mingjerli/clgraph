# clpipe

A powerful Python library for SQL column lineage analysis and pipeline dependency tracking.

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
pip install clpipe
```

Or with uv:
```bash
uv pip install clpipe
```

## Quick Start

### Single Query Column Lineage

```python
from clpipe import SQLColumnTracer

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

tracer = SQLColumnTracer(sql, dialect="bigquery")
lineage = tracer.build_column_lineage_graph()

# Get output columns
output_cols = lineage.get_output_nodes()
for col in output_cols:
    print(f"  {col.column_name}")

# Get source tables
input_nodes = lineage.get_input_nodes()
source_tables = {node.table_name for node in input_nodes if node.table_name}
```

### Multi-Query Pipeline Lineage

```python
from clpipe import Pipeline

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

# Table execution order
execution_order = pipeline.table_graph.get_execution_order()

# Trace a column backward through the pipeline
sources = pipeline.trace_column_backward("user_summary", "event_count")
for source in sources:
    print(f"  {source.table_name}.{source.column_name}")

# Forward lineage / Impact analysis
impacts = pipeline.trace_column_forward("source_events", "event_timestamp")
for impact in impacts:
    print(f"  {impact.table_name}.{impact.column_name}")
```

### Metadata from SQL Comments

```python
from clpipe import Pipeline

sql = """
SELECT
  user_id,  -- User identifier [pii: false]
  email,    -- Email address [pii: true, owner: data-team]
  COUNT(*) as login_count  -- Number of logins [tags: metric engagement]
FROM user_activity
GROUP BY user_id, email
"""

pipeline = Pipeline([("user_metrics", sql)], dialect="bigquery")

# Metadata is automatically extracted from comments
for col in pipeline.columns.values():
    if col.pii:
        print(f"PII Column: {col.full_name}")
    if col.owner:
        print(f"Owner: {col.owner}")
```

### Metadata Management and Export

```python
from clpipe import (
    MultiQueryParser,
    PipelineLineageBuilder,
    JSONExporter,
    CSVExporter,
    GraphVizExporter
)

# Build pipeline
parser = MultiQueryParser()
table_graph = parser.parse_queries(sql_queries)
builder = PipelineLineageBuilder()
lineage_graph = builder.build(table_graph)

# Set source metadata
for col in lineage_graph.columns.values():
    if col.table_name == "raw.orders" and col.column_name == "user_email":
        col.set_source_description("Customer email address")
        col.owner = "data-team"
        col.pii = True
        col.tags = {"contact", "sensitive"}

# Propagate metadata through lineage
lineage_graph.propagate_all_metadata()

# Find all PII columns
pii_columns = lineage_graph.get_pii_columns()
print(f"Found {len(pii_columns)} PII columns")

# Export to different formats
JSONExporter.export_to_file(lineage_graph, "lineage.json")
CSVExporter.export_columns_to_file(lineage_graph, "columns.csv")
GraphVizExporter.export_to_file(lineage_graph, "lineage.dot")

# Track changes between versions
diff = new_lineage_graph.diff(old_lineage_graph)
print(diff.summary())
```

### LLM-Powered Description Generation

```python
from clpipe import MultiQueryParser, PipelineLineageBuilder
from langchain_ollama import ChatOllama

# Build pipeline
parser = MultiQueryParser()
table_graph = parser.parse_queries(sql_queries)
builder = PipelineLineageBuilder()
lineage_graph = builder.build(table_graph)

# Configure LLM (Ollama - free, local)
llm = ChatOllama(model="qwen3-coder:30b", temperature=0.3)
lineage_graph.llm = llm

# Generate descriptions for all columns
lineage_graph.generate_all_descriptions(verbose=True)

# View generated descriptions
for col in lineage_graph.columns.values():
    print(f"{col.full_name}: {col.description}")
```

## Architecture

The library consists of three main components:

### 1. Query Parser (`RecursiveQueryParser`)
Parses SQL into a `QueryUnitGraph` representing the structure:
- Main query
- CTEs (Common Table Expressions)
- Subqueries (in FROM, SELECT, WHERE, HAVING clauses)
- Dependency relationships

### 2. Lineage Builder (`RecursiveLineageBuilder`)
Builds a `ColumnLineageGraph` showing column-level dependencies:
- Column nodes (sources and derived columns)
- Edges representing transformations
- Support for star notation and modifiers
- Forward and backward lineage queries

### 3. Multi-Query Support (`MultiQueryParser`, `PipelineLineageBuilder`)
Handles multiple related queries as a pipeline:
- Table dependency resolution
- Cross-query column lineage
- Template variable support
- Pipeline-wide impact analysis

## Pipeline Graph Objects

A `Pipeline` contains two graph structures for lineage analysis:

```python
from clpipe import Pipeline

pipeline = Pipeline(queries, dialect="bigquery")

# Two graph objects available:
# 1. Table-level graph (TableDependencyGraph)
# 2. Column-level graph (PipelineLineageGraph)
```

### Table Graph (`pipeline.table_graph`)

The `TableDependencyGraph` tracks table-level dependencies:

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
```python
tracer = SQLColumnTracer(sql, dialect="postgres")
pipeline = Pipeline(queries, dialect="snowflake")
```

## Why We Built This

Column lineage is notoriously difficult. Traditional tools reverse-engineer lineage from query logs and execution metadata, requiring expensive platform integration and complex infrastructure. Most open-source alternatives focus only on table-level lineage or single-query column analysis.

**Our insight**: When SQL is written with explicit column names and clear transformations (what we call "lineage-friendly SQL"), static analysis can provide *perfect* column lineage—without database access, without runtime integration, and without query logs.

We built clpipe to prove this approach works. By combining lineage-friendly SQL with perfect static analysis, we solve 90% of column lineage needs with 10% of the complexity of enterprise tools. No database required. No infrastructure to maintain. Just pure Python analyzing your SQL files.

**Read more**: [Why We Built This (Full Story)](https://clpipe.dev/concepts/why-we-built-this/)

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
git clone https://github.com/mingjerli/clpipe.git
cd clpipe

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
