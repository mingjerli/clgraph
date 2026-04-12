# Frontend Codemap

> Freshness: 2026-04-11

## No Frontend

clgraph is a Python backend library. It has no web frontend.

### Visual Output

Visualization is handled via **Graphviz** (server-side rendering):

- `visualize_column_lineage()` - Full column lineage graph
- `visualize_table_dependencies()` - Table-level DAG
- `visualize_lineage_path()` - Path between two columns
- `visualize_query_structure_from_lineage()` - Query unit structure

Output formats: PNG, SVG, PDF (via Graphviz `render()`)

### Interactive Usage

- **Jupyter notebooks**: 25+ examples in `examples/` directory
- **MCP server**: `python -m clgraph.mcp` for Claude Desktop integration
- **CLI**: No dedicated CLI (library usage via Python imports)

### Export Formats

- JSON via `JSONExporter`
- CSV via `CSVExporter`
- Graphviz DOT (via visualizations module)
