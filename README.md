# SQL Lineage

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

### Metadata Management (NEW)
- **Column metadata** - track descriptions, ownership, PII flags, and custom tags
- **Metadata propagation** - automatically inherit metadata through lineage
- **LLM integration** - generate natural language descriptions using OpenAI, Ollama, etc.
- **Diff tracking** - detect changes between pipeline versions

### Export Functionality (NEW)
- **JSON export** - machine-readable format for system integration
- **CSV export** - column and table metadata for spreadsheets
- **GraphViz export** - DOT format for visualization tools

## Installation

```bash
pip install clpipe
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

tracer = SQLColumnTracer(sql)
lineage = tracer.build_column_lineage_graph()

# Get backward lineage (sources) for a column
sources = lineage.get_backward_lineage("name")
print(sources)  # {'users.name'}

# Get forward lineage (impacts) from a source
impacts = lineage.get_forward_lineage("orders.amount")
print(impacts)  # {'monthly_sales.total_amount', 'output.total_amount'}
```

### Multi-Query Pipeline Lineage

```python
from clpipe import MultiQueryParser, PipelineLineageBuilder

queries = [
    ("raw_events", "CREATE TABLE raw_events AS SELECT user_id, event_type, created_at FROM events"),
    ("daily_active_users", "CREATE TABLE daily_active_users AS SELECT user_id, DATE(created_at) as date FROM raw_events"),
    ("user_summary", "CREATE TABLE user_summary AS SELECT u.name, dau.date FROM users u JOIN daily_active_users dau ON u.id = dau.user_id")
]

parser = MultiQueryParser(queries)
graph = parser.parse()

builder = PipelineLineageBuilder(graph)
lineage = builder.build()

# Trace a column through the entire pipeline
sources = lineage.get_backward_lineage("user_summary.date")
print(sources)  # {'raw_events.created_at', 'events.created_at'}
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
        col.pii = True  # Mark as PII
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
print(diff.summary())  # Shows added, removed, and modified columns
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
# 2. Column-level graph (ColumnGraph)
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
for dep in deps:
    print(f"  depends on: {dep.table_name}")

# Get downstream tables (impact analysis)
downstream = pipeline.table_graph.get_downstream("raw.orders")
for table in downstream:
    print(f"  impacts: {table.table_name}")

# Get query execution order (topologically sorted)
query_order = pipeline.table_graph.topological_sort()

# Get table execution order
table_order = pipeline.table_graph.get_execution_order()

# Build graphlib-style dependency map
deps_map = pipeline.table_graph._build_table_dependencies()
# Returns: Dict[str, Set[str]] - {table_name: {upstream_tables}}
```

### Column Graph (`pipeline.column_graph`)

The `ColumnGraph` tracks column-level lineage:

```python
# Access columns and edges
pipeline.column_graph.columns    # Dict[str, PipelineColumnNode]
pipeline.column_graph.edges      # List[PipelineColumnEdge]

# Backward compatible access (these are property aliases)
pipeline.columns  # Same as pipeline.column_graph.columns
pipeline.edges    # Same as pipeline.column_graph.edges

# Get source columns (no incoming edges)
source_cols = pipeline.column_graph.get_source_columns()

# Get final columns (no outgoing edges)
final_cols = pipeline.column_graph.get_final_columns()

# Get direct upstream columns (one hop back)
upstream = pipeline.column_graph.get_upstream("analytics.metrics.total_revenue")
for col in upstream:
    print(f"  depends on: {col.full_name}")

# Get direct downstream columns (one hop forward)
downstream = pipeline.column_graph.get_downstream("raw.orders.amount")
for col in downstream:
    print(f"  impacts: {col.full_name}")

# Build graphlib-style dependency map
col_deps = pipeline.column_graph._build_column_dependencies()
# Returns: Dict[str, Set[str]] - {column_full_name: {upstream_column_full_names}}
```

### Full Lineage Tracing

For complete lineage (not just direct dependencies), use Pipeline methods:

```python
# Trace backward to ultimate sources (recursive)
sources = pipeline.trace_column_backward("final_table", "metric")
# Returns all source columns across the entire pipeline

# Trace forward to all impacts (recursive)
impacts = pipeline.trace_column_forward("raw.orders", "amount")
# Returns all downstream columns that depend on this column

# Find specific lineage path between two columns
path = pipeline.get_lineage_path(
    "raw.orders", "amount",
    "analytics.metrics", "total_revenue"
)
# Returns list of edges connecting the two columns
```

## Supported SQL Dialects

Built on [sqlglot](https://github.com/tobymao/sqlglot), supporting:
- BigQuery
- PostgreSQL
- MySQL
- Snowflake
- Redshift
- And many more

Specify dialect when creating the tracer:
```python
tracer = SQLColumnTracer(sql, dialect="postgres")
```

## Use Cases

- **Data Governance**: Track data lineage for compliance and auditing
- **Impact Analysis**: Understand downstream effects of schema changes
- **Pipeline Optimization**: Identify unused columns and redundant transformations
- **Data Quality**: Trace data issues back to their source
- **Documentation**: Auto-generate data flow diagrams

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/clpipe.git
cd clpipe

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## Credits

Built with:
- [sqlglot](https://github.com/tobymao/sqlglot) - SQL parsing and transpilation
- Python's `graphlib` - Topological sorting for dependency resolution
