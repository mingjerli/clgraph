# MCP Server for Column Lineage

This MCP (Model Context Protocol) server exposes column lineage analysis and semantic search capabilities, allowing LLMs and AI assistants to query your data pipelines using natural language.

## What is MCP?

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open protocol that enables secure, controlled interactions between AI assistants and external tools/data sources. It allows LLMs like Claude to:

- Call tools and functions
- Access data sources
- Execute operations in your systems
- All while maintaining security and control

## Features

The clgraph MCP server provides these tools to LLMs:

### 1. **search_columns** 🔍
Search for columns using natural language with semantic understanding.

```
Example: "Find all revenue-related columns owned by the finance team"
```

### 2. **trace_lineage_backward** 📊
Trace column dependencies backward to source columns.

```
Example: "What are the source columns for user_summary.lifetime_revenue?"
```

### 3. **trace_lineage_forward** 🎯
Trace column impact forward to downstream columns.

```
Example: "Which columns depend on orders.customer_email?"
```

### 4. **get_column_info** ℹ️
Get detailed information about a specific column.

```
Example: "Show me details about user_events.revenue_amount"
```

### 5. **get_pii_columns** ⚠️
List all columns containing PII data.

```
Example: "Which columns contain personally identifiable information?"
```

### 6. **list_tables** 📋
List all tables with statistics.

```
Example: "What tables are in this pipeline?"
```

### 7. **get_table_lineage** 🔗
Show table-level dependencies.

```
Example: "What tables does user_summary depend on?"
```

## Installation

### 1. Install Dependencies

```bash
# Install clgraph with MCP support
pip install clgraph[mcp]

# For full features including NLP search
pip install clgraph[mcp,nlp]
```

### 2. Set Up Configuration

Create a configuration file (e.g., `lineage_config.json`):

```json
{
  "dialect": "bigquery",
  "enable_nlp": true,
  "embedding_model": "all-MiniLM-L6-v2",
  "collection_name": "my_lineage",
  "retrieval_k": 5,
  "rebuild_index": false,
  "queries": [
    {
      "name": "table1",
      "sql": "CREATE TABLE table1 AS SELECT ..."
    },
    {
      "name": "table2",
      "sql": "CREATE TABLE table2 AS SELECT ..."
    }
  ]
}
```

### 3. Test the Server

```bash
# Run the server directly
python examples/mcp_server.py --config lineage_config.json

# The server will start and wait for MCP client connections
```

## Configuration Reference

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `dialect` | string | SQL dialect (bigquery, snowflake, postgres, etc.) | "bigquery" |
| `enable_nlp` | boolean | Enable semantic search with vector database | true |
| `embedding_model` | string | Sentence transformer model for embeddings | "all-MiniLM-L6-v2" |
| `collection_name` | string | Name for the vector database collection | "lineage" |
| `retrieval_k` | integer | Number of results to retrieve for NLP search | 5 |
| `rebuild_index` | boolean | Rebuild vector index on startup | false |
| `vector_store_path` | string | Path to persist vector database (optional) | null |
| `queries` | array | List of query objects with name and sql | [] |

### Query Object Format

```json
{
  "name": "table_name",
  "sql": "CREATE TABLE table_name AS SELECT ..."
}
```

## Integration Examples

### Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "clgraph-lineage": {
      "command": "python",
      "args": [
        "/absolute/path/to/clgraph/examples/mcp_server.py",
        "--config",
        "/absolute/path/to/your/lineage_config.json"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/clgraph/src"
      }
    }
  }
}
```

**Important:** Use absolute paths, not relative paths.

After adding this configuration:
1. Restart Claude Desktop
2. You should see the MCP server connected
3. Claude can now use the lineage tools!

### Cline (VS Code Extension)

Add to Cline's MCP settings:

```json
{
  "mcpServers": {
    "clgraph-lineage": {
      "command": "python",
      "args": [
        "/absolute/path/to/clgraph/examples/mcp_server.py",
        "--config",
        "/absolute/path/to/your/lineage_config.json"
      ]
    }
  }
}
```

### Continue.dev

Similar configuration in Continue's settings.

### Custom MCP Client

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create server parameters
server_params = StdioServerParameters(
    command="python",
    args=["examples/mcp_server.py", "--config", "lineage_config.json"]
)

# Connect to server
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize
        await session.initialize()

        # List available tools
        tools = await session.list_tools()
        print(f"Available tools: {[t.name for t in tools.tools]}")

        # Call a tool
        result = await session.call_tool(
            "search_columns",
            arguments={"query": "revenue columns", "limit": 5}
        )
        print(result.content)
```

## Usage Examples

Once connected, you can ask Claude (or any MCP client) questions like:

### Data Discovery

**You:** "Find all columns related to customer revenue"

**Claude:** *Uses search_columns tool with query "customer revenue"*

**You:** "Which columns contain PII?"

**Claude:** *Uses get_pii_columns tool*

### Lineage Analysis

**You:** "What are the source columns for user_summary.lifetime_revenue?"

**Claude:** *Uses trace_lineage_backward tool*

**You:** "If I change orders.amount, what downstream columns are affected?"

**Claude:** *Uses trace_lineage_forward tool starting from orders.amount*

### Metadata Queries

**You:** "Show me all columns owned by the finance team"

**Claude:** *Uses search_columns with filter_owner="finance-team"*

**You:** "What's the detailed information for user_events.revenue_amount?"

**Claude:** *Uses get_column_info tool*

### Pipeline Understanding

**You:** "What tables are in this pipeline?"

**Claude:** *Uses list_tables tool*

**You:** "Show me the dependencies for the user_summary table"

**Claude:** *Uses get_table_lineage tool*

## Advanced Configuration

### Loading Queries from Files

Instead of inline SQL in config, you can load from files:

```python
# Create a custom config loader
import json
from pathlib import Path

def load_queries_from_directory(dir_path):
    queries = []
    for sql_file in Path(dir_path).glob("*.sql"):
        queries.append({
            "name": sql_file.stem,
            "sql": sql_file.read_text()
        })
    return queries

# Save to config
config = {
    "dialect": "bigquery",
    "enable_nlp": True,
    "queries": load_queries_from_directory("./sql")
}

with open("lineage_config.json", "w") as f:
    json.dump(config, f, indent=2)
```

### Persistent Vector Store

To avoid rebuilding the vector index on every startup:

```json
{
  "vector_store_path": "./lineage_vectors",
  "rebuild_index": false,
  ...
}
```

The first run will build and persist the index. Subsequent runs will load it from disk.

### Different Embedding Models

Choose based on your needs:

```json
{
  "embedding_model": "all-MiniLM-L6-v2",  // Fast, good quality (default)
  // OR
  "embedding_model": "all-mpnet-base-v2",  // Higher quality, slower
  // OR
  "embedding_model": "sentence-t5-xl"      // Best quality, slowest
}
```

### Dynamic Pipeline Loading

You can extend the server to load pipelines dynamically:

```python
class LineageMCPServer:
    def _load_config(self, config_path):
        config = super()._load_config(config_path)

        # Load from database
        if config.get('load_from_db'):
            config['queries'] = self._load_queries_from_db(
                config['db_connection']
            )

        # Load from dbt project
        if config.get('dbt_project_path'):
            config['queries'] = self._load_from_dbt(
                config['dbt_project_path']
            )

        return config
```

## Troubleshooting

### Server Not Appearing in Claude Desktop

1. Check logs: `~/Library/Logs/Claude/mcp*.log`
2. Verify paths are absolute
3. Test server manually: `python mcp_server.py --config config.json`
4. Check Python version (requires 3.10+)

### Import Errors

```bash
# Install all dependencies
pip install clgraph[mcp,nlp]

# Or install individually
pip install mcp chromadb sentence-transformers
```

### NLP Search Not Working

If you get "NLP search not available":

```bash
pip install clgraph[nlp]
# Or
pip install chromadb sentence-transformers
```

### Slow First Startup

First run downloads embedding model from HuggingFace:
- Requires internet connection
- Takes 1-2 minutes
- Models cached in `~/.cache/huggingface/`
- Subsequent starts are fast

### Configuration Not Loading

1. Check JSON syntax with `json.verify`
2. Use absolute paths
3. Verify file exists and is readable
4. Check server logs for errors

## Architecture

```
┌─────────────┐
│ MCP Client  │ (Claude Desktop, Cline, etc.)
│  (LLM)      │
└──────┬──────┘
       │ MCP Protocol (stdio)
       │
┌──────▼──────────────────────────┐
│   Lineage MCP Server            │
│                                 │
│  ┌──────────────────────────┐  │
│  │ Tool Handlers            │  │
│  │ - search_columns         │  │
│  │ - trace_lineage_*        │  │
│  │ - get_column_info        │  │
│  └────────┬─────────────────┘  │
│           │                     │
│  ┌────────▼─────────────────┐  │
│  │ Pipeline                 │  │
│  │ - Column lineage graph   │  │
│  │ - Table dependency graph │  │
│  └────────┬─────────────────┘  │
│           │                     │
│  ┌────────▼─────────────────┐  │
│  │ Vector Store (Optional)  │  │
│  │ - ChromaDB               │  │
│  │ - Embeddings             │  │
│  │ - Semantic search        │  │
│  └──────────────────────────┘  │
└─────────────────────────────────┘
```

## Security Considerations

### Read-Only Operations

The MCP server provides **read-only** access to lineage data. It cannot:
- Modify your database
- Execute queries against your data warehouse
- Change pipeline configurations
- Write files

### Sensitive Data

The server may expose:
- Column names and descriptions
- SQL queries and transformations
- Metadata (ownership, PII flags)

**Best practices:**
1. Don't include actual data values in descriptions
2. Use environment-specific configs (dev/staging/prod)
3. Restrict MCP server access appropriately
4. Review logs for sensitive information

### Network Isolation

MCP uses stdio (standard input/output) communication:
- No network ports opened
- No HTTP endpoints
- Communication only with parent process
- Inherits parent process permissions

## Performance Tips

### 1. Use Persistent Vector Store

```json
{
  "vector_store_path": "./vectors",
  "rebuild_index": false
}
```

### 2. Choose Appropriate Embedding Model

- Small pipelines (<1000 columns): `all-MiniLM-L6-v2`
- Medium pipelines (1000-10000): `all-mpnet-base-v2`
- Large pipelines (10000+): Consider Qdrant or Weaviate

### 3. Limit Retrieval Results

```json
{
  "retrieval_k": 3  // Return fewer results for faster queries
}
```

### 4. Optimize Query Configuration

Only include queries you need:

```json
{
  "queries": [
    // Include only production queries, not test queries
  ]
}
```

## Extending the Server

### Adding Custom Tools

```python
@self.server.list_tools()
async def list_tools() -> List[Tool]:
    tools = [
        # ... existing tools ...
        Tool(
            name="find_unused_columns",
            description="Find columns that aren't used downstream",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {"type": "string"}
                }
            }
        )
    ]
    return tools

@self.server.call_tool()
async def call_tool(name: str, arguments: Any):
    if name == "find_unused_columns":
        return await self._find_unused_columns(arguments)
    # ... existing handlers ...
```

### Custom Metadata Extractors

```python
def _initialize_pipeline(self):
    self.pipeline = Pipeline(queries, dialect=dialect)

    # Add custom metadata from external source
    for col in self.pipeline.columns.values():
        external_metadata = fetch_metadata_from_catalog(col.full_name)
        if external_metadata:
            col.description = external_metadata.get('description')
            col.owner = external_metadata.get('owner')

    self.pipeline.propagate_all_metadata()
```

### Integration with Data Catalogs

```python
# Add to server initialization
def _load_metadata_from_catalog(self):
    """Load metadata from external data catalog"""
    catalog_api = DataCatalogAPI(self.config['catalog_url'])

    for col in self.pipeline.columns.values():
        metadata = catalog_api.get_column_metadata(
            col.table_name,
            col.column_name
        )
        if metadata:
            col.description = metadata.description
            col.pii = metadata.is_pii
            col.owner = metadata.owner
```

## Use Cases

### 1. Data Discovery
Enable data analysts to find relevant columns:
- "Find all customer contact information"
- "Show me revenue metrics from sales"

### 2. Impact Analysis
Understand downstream effects of changes:
- "What breaks if I rename this column?"
- "Which reports use customer_email?"

### 3. Compliance & Governance
Support compliance queries:
- "List all PII columns"
- "What data is owned by privacy team?"

### 4. Documentation
Help teams understand pipelines:
- "Explain the lifetime value calculation"
- "How is daily active users computed?"

### 5. Onboarding
Help new team members learn:
- "What tables are in the user analytics pipeline?"
- "Where does user engagement data come from?"

## Roadmap

Future enhancements:

- [ ] Support for multiple pipelines/projects
- [ ] Real-time pipeline updates
- [ ] Query history and analytics
- [ ] Column similarity recommendations
- [ ] Auto-generated documentation
- [ ] Integration with dbt, Airflow, etc.
- [ ] Graph visualization generation
- [ ] Schema change impact reports

## Contributing

Have ideas for improving the MCP server? Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Resources

- **MCP Documentation**: https://modelcontextprotocol.io/
- **MCP Python SDK**: https://github.com/modelcontextprotocol/python-sdk
- **Claude Desktop**: https://claude.ai/download
- **clgraph Documentation**: https://github.com/mingjerli/clgraph

## License

MIT License - see LICENSE file for details
