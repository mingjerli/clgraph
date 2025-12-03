# Installing MCP Server

Quick start guide for setting up the clgraph MCP server.

## Step 1: Install Dependencies

```bash
# Option 1: Install from PyPI (when published)
pip install clgraph[mcp,nlp]

# Option 2: Install from source
cd clgraph
pip install -e ".[mcp,nlp]"

# Option 3: Install dependencies manually
pip install mcp chromadb sentence-transformers
```

## Step 2: Create Configuration

Create a file `my_lineage_config.json`:

```json
{
  "dialect": "bigquery",
  "enable_nlp": true,
  "queries": [
    {
      "name": "your_table",
      "sql": "CREATE TABLE your_table AS SELECT ..."
    }
  ]
}
```

See `mcp_config_example.json` for a complete example.

## Step 3: Test the Server

```bash
# Test that everything is working
python examples/test_mcp_server.py

# If all tests pass, try running the server
python examples/mcp_server.py --config my_lineage_config.json
```

## Step 4: Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "clgraph-lineage": {
      "command": "python",
      "args": [
        "/absolute/path/to/clgraph/examples/mcp_server.py",
        "--config",
        "/absolute/path/to/my_lineage_config.json"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/clgraph/src"
      }
    }
  }
}
```

**Important:** Use absolute paths!

## Step 5: Restart Claude Desktop

1. Quit Claude Desktop completely
2. Relaunch Claude Desktop
3. Look for MCP server connection indicator
4. Try asking: "What tables are in my lineage pipeline?"

## Troubleshooting

### "No module named 'mcp'"

Install MCP:
```bash
pip install mcp
```

### "No module named 'clgraph'"

Make sure PYTHONPATH is set correctly in the config, or install clgraph:
```bash
pip install -e /path/to/clgraph
```

### Server not appearing in Claude Desktop

1. Check Claude Desktop logs: `~/Library/Logs/Claude/`
2. Verify paths are absolute (not relative)
3. Test server manually first
4. Check Python version (requires 3.10+)

### NLP features not working

Install NLP dependencies:
```bash
pip install chromadb sentence-transformers
```

## Next Steps

See `README_MCP.md` for:
- Complete feature list
- Usage examples
- Advanced configuration
- Extension guide
