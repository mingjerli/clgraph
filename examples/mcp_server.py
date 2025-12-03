#!/usr/bin/env python3
"""
MCP Server for Column Lineage and Semantic Search

This MCP server exposes column lineage analysis and semantic search
capabilities through the Model Context Protocol, allowing LLMs to:

1. Search for columns using natural language
2. Trace column lineage (backward/forward)
3. Query metadata (PII, ownership, tags)
4. Discover data relationships
5. Answer questions about data pipelines

Usage:
    # Run the server
    python mcp_server.py --config config.json

    # Or use with uvx (recommended)
    uvx mcp_server.py

Example queries through MCP:
    - "Find all PII columns"
    - "What are the sources of user_summary.lifetime_revenue?"
    - "Show me revenue-related columns owned by finance team"
    - "Trace the lineage of orders.customer_email"
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
    )
except ImportError:
    print("ERROR: mcp not installed. Install with: pip install mcp")
    raise

from clgraph.pipeline import Pipeline

# Import our NLP components (optional, falls back gracefully)
try:
    from nlp_vector_database_rag import LineageVectorStore, LineageRAG
    HAS_NLP = True
except ImportError:
    HAS_NLP = False


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clgraph-mcp")


class LineageMCPServer:
    """MCP Server for column lineage analysis and semantic search"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MCP server

        Args:
            config_path: Path to configuration JSON file
        """
        self.server = Server("clgraph-lineage")
        self.pipeline: Optional[Pipeline] = None
        self.vector_store: Optional[Any] = None
        self.rag: Optional[Any] = None
        self.config = self._load_config(config_path) if config_path else {}

        # Register handlers
        self._register_handlers()

        logger.info("Lineage MCP Server initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _initialize_pipeline(self):
        """Initialize pipeline from configuration"""
        if self.pipeline is not None:
            return

        queries = self.config.get('queries', [])
        dialect = self.config.get('dialect', 'bigquery')

        if not queries:
            logger.warning("No queries configured, using demo data")
            queries = self._get_demo_queries()

        # Convert queries to tuples
        query_tuples = [
            (q['name'], q['sql']) for q in queries
        ]

        logger.info(f"Initializing pipeline with {len(query_tuples)} queries")
        self.pipeline = Pipeline(query_tuples, dialect=dialect)
        self.pipeline.propagate_all_metadata()
        logger.info(f"Pipeline initialized with {len(self.pipeline.columns)} columns")

        # Initialize vector store if available
        if HAS_NLP and self.config.get('enable_nlp', True):
            self._initialize_nlp()

    def _initialize_nlp(self):
        """Initialize NLP components"""
        if not HAS_NLP:
            logger.warning("NLP components not available")
            return

        try:
            persist_dir = self.config.get('vector_store_path')
            embedding_model = self.config.get('embedding_model', 'all-MiniLM-L6-v2')

            logger.info("Initializing vector store...")
            self.vector_store = LineageVectorStore(
                collection_name=self.config.get('collection_name', 'lineage'),
                embedding_model=embedding_model,
                persist_directory=persist_dir
            )

            # Check if we need to rebuild the index
            if self.config.get('rebuild_index', False):
                logger.info("Rebuilding vector index...")
                self.vector_store.clear()
                self.vector_store.add_pipeline(self.pipeline)

            # Initialize RAG
            self.rag = LineageRAG(
                vector_store=self.vector_store,
                llm=None,  # We'll use MCP client's LLM
                retrieval_k=self.config.get('retrieval_k', 5)
            )

            logger.info("NLP components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize NLP: {e}")
            self.vector_store = None
            self.rag = None

    def _get_demo_queries(self) -> List[Dict[str, str]]:
        """Get demo queries for testing"""
        return [
            {
                "name": "raw_events",
                "sql": """
                    CREATE TABLE raw_events AS
                    SELECT
                        event_id,
                        user_id,              -- [owner: data-team]
                        user_email,           -- [pii: true, owner: privacy-team]
                        event_type,
                        event_timestamp,
                        revenue_amount        -- [tags: financial revenue, owner: finance-team]
                    FROM source_events
                """
            },
            {
                "name": "user_summary",
                "sql": """
                    CREATE TABLE user_summary AS
                    SELECT
                        user_id,
                        user_email,
                        SUM(revenue_amount) as lifetime_revenue,  -- [tags: financial ltv kpi]
                        COUNT(*) as total_events
                    FROM raw_events
                    GROUP BY user_id, user_email
                """
            }
        ]

    def _register_handlers(self):
        """Register MCP tool handlers"""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            tools = [
                Tool(
                    name="search_columns",
                    description=(
                        "Search for columns using natural language. "
                        "Finds columns semantically related to the query. "
                        "Supports metadata filters like pii, owner, tags."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query (e.g., 'revenue columns', 'PII data')"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 5
                            },
                            "filter_pii": {
                                "type": "boolean",
                                "description": "Only return PII columns"
                            },
                            "filter_owner": {
                                "type": "string",
                                "description": "Filter by owner (e.g., 'finance-team')"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="trace_lineage_backward",
                    description=(
                        "Trace column lineage backward to find all source columns. "
                        "Shows the complete dependency chain from output to sources."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table"
                            },
                            "column_name": {
                                "type": "string",
                                "description": "Name of the column"
                            }
                        },
                        "required": ["table_name", "column_name"]
                    }
                ),
                Tool(
                    name="trace_lineage_forward",
                    description=(
                        "Trace column lineage forward to find all downstream columns. "
                        "Shows impact analysis - which columns depend on this one."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table"
                            },
                            "column_name": {
                                "type": "string",
                                "description": "Name of the column"
                            }
                        },
                        "required": ["table_name", "column_name"]
                    }
                ),
                Tool(
                    name="get_column_info",
                    description=(
                        "Get detailed information about a specific column including "
                        "metadata, transformations, and lineage."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table"
                            },
                            "column_name": {
                                "type": "string",
                                "description": "Name of the column"
                            }
                        },
                        "required": ["table_name", "column_name"]
                    }
                ),
                Tool(
                    name="get_pii_columns",
                    description="Get all columns marked as containing PII (Personally Identifiable Information)",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="list_tables",
                    description="List all tables in the pipeline with their statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_table_lineage",
                    description="Get table-level lineage showing dependencies between tables",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table (optional, shows all if not provided)"
                            }
                        }
                    }
                ),
            ]

            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            """Handle tool calls"""
            # Initialize pipeline if needed
            self._initialize_pipeline()

            try:
                if name == "search_columns":
                    return await self._search_columns(arguments)
                elif name == "trace_lineage_backward":
                    return await self._trace_lineage_backward(arguments)
                elif name == "trace_lineage_forward":
                    return await self._trace_lineage_forward(arguments)
                elif name == "get_column_info":
                    return await self._get_column_info(arguments)
                elif name == "get_pii_columns":
                    return await self._get_pii_columns(arguments)
                elif name == "list_tables":
                    return await self._list_tables(arguments)
                elif name == "get_table_lineage":
                    return await self._get_table_lineage(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}", exc_info=True)
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def _search_columns(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """Search for columns using natural language"""
        if not HAS_NLP or not self.vector_store:
            return [TextContent(
                type="text",
                text="NLP search not available. Please install: pip install clgraph[nlp]"
            )]

        query = args["query"]
        limit = args.get("limit", 5)

        # Build metadata filter
        where = {}
        if args.get("filter_pii"):
            where["pii"] = "True"
        if args.get("filter_owner"):
            where["owner"] = args["filter_owner"]

        # Search
        results = self.vector_store.search(query, n_results=limit, where=where if where else None)

        # Format results
        if not results:
            return [TextContent(type="text", text="No matching columns found.")]

        lines = [f"Found {len(results)} columns matching '{query}':\n"]
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            lines.append(f"\n{i}. **{metadata['full_name']}**")
            lines.append(f"   - Type: {metadata['node_type']}")

            if metadata.get('pii') == 'True':
                lines.append("   - ⚠️  Contains PII")

            if metadata.get('owner'):
                lines.append(f"   - Owner: {metadata['owner']}")

            # Show excerpt from document
            doc_lines = result['document'].split('\n')
            if len(doc_lines) > 1:
                lines.append(f"   - {doc_lines[1]}")  # Usually the description

        return [TextContent(type="text", text="\n".join(lines))]

    async def _trace_lineage_backward(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """Trace column lineage backward to sources"""
        table_name = args["table_name"]
        column_name = args["column_name"]

        try:
            sources = self.pipeline.trace_column_backward(table_name, column_name)

            if not sources:
                return [TextContent(
                    type="text",
                    text=f"No source columns found for {table_name}.{column_name}"
                )]

            lines = [f"Source columns for **{table_name}.{column_name}** ({len(sources)} total):\n"]
            for source in sources:
                lines.append(f"- {source.full_name}")
                if source.expression:
                    lines.append(f"  Expression: `{source.expression}`")
                if source.description:
                    lines.append(f"  Description: {source.description}")

            return [TextContent(type="text", text="\n".join(lines))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error tracing lineage: {str(e)}")]

    async def _trace_lineage_forward(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """Trace column lineage forward to impacts"""
        table_name = args["table_name"]
        column_name = args["column_name"]

        try:
            impacts = self.pipeline.trace_column_forward(table_name, column_name)

            if not impacts:
                return [TextContent(
                    type="text",
                    text=f"No downstream columns found for {table_name}.{column_name}"
                )]

            lines = [f"Downstream columns for **{table_name}.{column_name}** ({len(impacts)} total):\n"]
            for impact in impacts:
                lines.append(f"- {impact.full_name}")
                if impact.expression:
                    lines.append(f"  Expression: `{impact.expression}`")
                if impact.description:
                    lines.append(f"  Description: {impact.description}")

            return [TextContent(type="text", text="\n".join(lines))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error tracing lineage: {str(e)}")]

    async def _get_column_info(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """Get detailed information about a column"""
        table_name = args["table_name"]
        column_name = args["column_name"]
        full_name = f"{table_name}.{column_name}"

        column = self.pipeline.columns.get(full_name)
        if not column:
            return [TextContent(type="text", text=f"Column not found: {full_name}")]

        lines = [f"# {full_name}\n"]

        # Basic info
        lines.append(f"**Type:** {column.node_type}")
        if column.layer:
            lines.append(f"**Layer:** {column.layer}")

        # Description
        if column.description:
            lines.append(f"\n**Description:** {column.description}")

        # Expression
        if column.expression:
            lines.append(f"\n**Expression:** `{column.expression}`")

        # Operation
        if column.operation:
            lines.append(f"**Operation:** {column.operation}")

        # Metadata
        lines.append("\n**Metadata:**")
        lines.append(f"- PII: {'Yes ⚠️' if column.pii else 'No'}")
        if column.owner:
            lines.append(f"- Owner: {column.owner}")
        if column.tags:
            lines.append(f"- Tags: {', '.join(column.tags)}")

        # Lineage
        upstream = self.pipeline.column_graph.get_upstream(full_name)
        if upstream:
            lines.append(f"\n**Upstream columns ({len(upstream)}):**")
            for col in upstream[:5]:  # Limit to 5
                lines.append(f"- {col.full_name}")
            if len(upstream) > 5:
                lines.append(f"- ... and {len(upstream) - 5} more")

        downstream = self.pipeline.column_graph.get_downstream(full_name)
        if downstream:
            lines.append(f"\n**Downstream columns ({len(downstream)}):**")
            for col in downstream[:5]:
                lines.append(f"- {col.full_name}")
            if len(downstream) > 5:
                lines.append(f"- ... and {len(downstream) - 5} more")

        return [TextContent(type="text", text="\n".join(lines))]

    async def _get_pii_columns(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """Get all PII columns"""
        pii_columns = self.pipeline.get_pii_columns()

        if not pii_columns:
            return [TextContent(type="text", text="No PII columns found.")]

        lines = [f"Found {len(pii_columns)} PII columns:\n"]
        for col in pii_columns:
            lines.append(f"- **{col.full_name}**")
            if col.description:
                lines.append(f"  {col.description}")
            if col.owner:
                lines.append(f"  Owner: {col.owner}")

        return [TextContent(type="text", text="\n".join(lines))]

    async def _list_tables(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """List all tables"""
        tables = {}
        for full_name, col in self.pipeline.columns.items():
            table = col.table_name
            if table not in tables:
                tables[table] = []
            tables[table].append(col)

        lines = [f"Found {len(tables)} tables:\n"]
        for table, cols in sorted(tables.items()):
            pii_count = sum(1 for c in cols if c.pii)
            lines.append(f"\n**{table}**")
            lines.append(f"- Columns: {len(cols)}")
            if pii_count:
                lines.append(f"- PII columns: {pii_count} ⚠️")

        return [TextContent(type="text", text="\n".join(lines))]

    async def _get_table_lineage(self, args: Dict[str, Any]) -> Sequence[TextContent]:
        """Get table-level lineage"""
        table_name = args.get("table_name")

        if table_name:
            # Get dependencies for specific table
            table_node = self.pipeline.table_graph.tables.get(table_name)
            if not table_node:
                return [TextContent(type="text", text=f"Table not found: {table_name}")]

            lines = [f"# {table_name}\n"]

            deps = self.pipeline.table_graph.get_dependencies(table_name)
            if deps:
                lines.append(f"**Dependencies ({len(deps)}):**")
                for dep in deps:
                    lines.append(f"- {dep.table_name}")

            downstream = self.pipeline.table_graph.get_downstream(table_name)
            if downstream:
                lines.append(f"\n**Downstream ({len(downstream)}):**")
                for ds in downstream:
                    lines.append(f"- {ds.table_name}")

            return [TextContent(type="text", text="\n".join(lines))]
        else:
            # Show all table dependencies
            source_tables = self.pipeline.table_graph.get_source_tables()
            final_tables = self.pipeline.table_graph.get_final_tables()

            lines = ["# Table Lineage\n"]
            lines.append(f"**Source tables ({len(source_tables)}):**")
            for table in source_tables:
                lines.append(f"- {table.table_name}")

            lines.append(f"\n**Final tables ({len(final_tables)}):**")
            for table in final_tables:
                lines.append(f"- {table.table_name}")

            return [TextContent(type="text", text="\n".join(lines))]

    async def run(self):
        """Run the MCP server"""
        logger.info("Starting Lineage MCP Server...")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Lineage MCP Server")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file"
    )
    args = parser.parse_args()

    server = LineageMCPServer(config_path=args.config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
