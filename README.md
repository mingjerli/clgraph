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
