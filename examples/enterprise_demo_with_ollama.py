#!/usr/bin/env python3
"""
Enterprise Demo with Ollama: End-to-End Example

This example demonstrates clgraph's full capabilities using the enterprise-demo
SQL pipeline with Ollama for LLM-powered features:

1. Load enterprise-demo SQL files (4-layer pipeline: raw → staging → analytics → marts)
2. Generate column descriptions using Ollama
3. Use the LineageAgent to answer natural language questions
4. Demonstrate text-to-SQL capabilities

Requirements:
- Install: uv pip install -e ".[dev]"
- Install Ollama: brew install ollama (or see https://ollama.ai)
- Pull model: ollama pull llama3.2 (or qwen2.5-coder:7b for better results)
- Start Ollama: ollama serve

Usage:
    python examples/enterprise_demo_with_ollama.py
    python examples/enterprise_demo_with_ollama.py --model qwen2.5-coder:7b
    python examples/enterprise_demo_with_ollama.py --skip-descriptions  # Skip LLM generation
"""

import argparse
import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clgraph import Pipeline
from clgraph.agent import LineageAgent
from clgraph.column import generate_description
from clgraph.export import JSONExporter
from clgraph.tools import (
    GetTableSchemaTool,
    ListTablesTool,
    SearchColumnsTool,
    TraceBackwardTool,
    TraceForwardTool,
)

# Path to ClickHouse example SQL files (within clgraph/examples/)
CLICKHOUSE_EXAMPLE_SQL_PATH = Path(__file__).parent / "clickhouse_example"


def load_enterprise_pipeline() -> Pipeline:
    """Load the ClickHouse example SQL pipeline."""
    if not CLICKHOUSE_EXAMPLE_SQL_PATH.exists():
        print(f"❌ ClickHouse example SQL files not found at: {CLICKHOUSE_EXAMPLE_SQL_PATH}")
        print("   Make sure you're running from the clgraph directory.")
        sys.exit(1)

    sql_files = sorted(CLICKHOUSE_EXAMPLE_SQL_PATH.glob("*.sql"))
    queries = []

    print("📂 Loading SQL files:")
    for sql_file in sql_files:
        # Skip init schema
        if sql_file.name.startswith("00_"):
            continue
        content = sql_file.read_text()
        query_id = sql_file.stem
        queries.append((query_id, content))
        print(f"   ✓ {sql_file.name}")

    print()
    print("🔧 Creating pipeline with template_context={'env': 'dev'}...")

    pipeline = Pipeline.from_tuples(
        queries,
        dialect="clickhouse",
        template_context={"env": "dev"},
    )

    print(
        f"✅ Pipeline created: {len(pipeline.table_graph.tables)} tables, {len(pipeline.columns)} columns"
    )
    return pipeline


def setup_ollama_llm(model: str = "llama3.2"):
    """Setup Ollama LLM for description generation."""
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=model,
            temperature=0.3,  # Lower for consistent descriptions
        )

        # Test connection
        print(f"🔌 Testing Ollama connection (model: {model})...")
        response = llm.invoke("Say 'OK' if you can hear me.")
        print(f"✅ Ollama connected: {response.content[:50]}...")
        return llm

    except ImportError:
        print("❌ langchain-ollama not installed. Install with: uv pip install langchain-ollama")
        return None
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("   Make sure Ollama is running: ollama serve")
        print(f"   And the model is pulled: ollama pull {model}")
        return None


def demo_description_generation(pipeline: Pipeline, llm):
    """Demonstrate LLM-powered description generation for all tables."""
    print()
    print("=" * 80)
    print("🤖 DESCRIPTION GENERATION (All Tables)")
    print("=" * 80)
    print()

    # Set LLM on pipeline
    pipeline.llm = llm

    # Get all output columns (final table columns, not intermediate)
    all_tables = sorted(pipeline.table_graph.tables.keys())

    # Count columns needing generation
    columns_with_desc = 0
    columns_without_desc = 0
    for col in pipeline.columns.values():
        if col.description:
            columns_with_desc += 1
        else:
            columns_without_desc += 1

    print("📊 Pipeline Statistics:")
    print(f"   • Tables: {len(all_tables)}")
    print(f"   • Total columns: {len(pipeline.columns)}")
    print(f"   • With descriptions: {columns_with_desc} (from SQL comments)")
    print(f"   • Need generation: {columns_without_desc}")
    print()

    # Generate descriptions for columns without them
    if columns_without_desc > 0:
        print(f"📝 Generating descriptions for {columns_without_desc} columns...")
        print("   (This may take a few minutes)")
        print()

        generated = 0
        errors = 0

        for table_name in all_tables:
            table_cols = [c for c in pipeline.columns.values() if c.table_name == table_name]
            cols_to_generate = [c for c in table_cols if c.description is None]

            if cols_to_generate:
                print(f"   📋 {table_name} ({len(cols_to_generate)} columns to generate)")

                for col in cols_to_generate:
                    print(f"      • {col.column_name}...", end=" ", flush=True)
                    try:
                        generate_description(col, llm, pipeline)
                        if col.description:
                            print("✓")
                            generated += 1
                        else:
                            print("✗ (no description returned)")
                    except Exception as e:
                        print(f"✗ Error: {str(e)[:50]}")
                        errors += 1

        print()
        print(f"✅ Generated: {generated}, Errors: {errors}")
    else:
        print("ℹ️  All columns already have descriptions from SQL comments!")
        print("   No LLM generation needed.")

    print()

    # Show descriptions by table
    print("📊 Column Descriptions by Table:")
    print("=" * 80)

    for table_name in all_tables:
        table_cols = [c for c in pipeline.columns.values() if c.table_name == table_name]
        if table_cols:
            print(f"\n📋 {table_name} ({len(table_cols)} columns)")
            print("-" * 70)
            for col in sorted(table_cols, key=lambda c: c.column_name):
                desc = col.description or "(no description)"
                source = col.description_source.value if col.description_source else "none"
                # Truncate long descriptions
                desc_display = desc[:45] + "..." if len(desc) > 45 else desc
                print(f"   {col.column_name:25} [{source:8}] {desc_display}")


def demo_lineage_agent(pipeline: Pipeline, llm):
    """Demonstrate the LineageAgent for natural language queries."""
    print()
    print("=" * 80)
    print("🤖 LINEAGE AGENT (Natural Language Interface)")
    print("=" * 80)
    print()

    agent = LineageAgent(pipeline, llm=llm, verbose=True)

    print("Available tools:")
    for i, tool in enumerate(agent.list_tools(), 1):
        print(f"   {i:2}. {tool}")
    print()

    # Comprehensive example questions demonstrating various capabilities
    questions = [
        # Schema exploration
        ("📋 Schema", "What tables are available?"),
        ("📋 Schema", "What columns does marts_dev.customer_360 have?"),
        # Lineage tracing
        ("🔙 Backward", "Where does analytics_dev.customer_metrics.lifetime_value come from?"),
        ("🔜 Forward", "What depends on raw_dev.orders.total_amount?"),
        # Search
        ("🔍 Search", "Find columns containing 'customer'"),
        ("🔍 Search", "Find columns containing 'revenue'"),
        # Data governance
        ("🔒 Governance", "Which columns contain PII?"),
        ("👤 Ownership", "Who owns the customer_360 table?"),
    ]

    for category, question in questions:
        print(f"{category} ❓ {question}")
        print("-" * 70)

        result = agent.query(question)

        print(f"   🔧 Tool: {result.tool_used}")
        # Show full answer for important queries
        answer = result.answer
        if len(answer) > 300:
            answer = answer[:300] + "..."
        print(f"   📝 Answer: {answer}")
        if result.error:
            print(f"   ❌ Error: {result.error}")
        print()


def demo_tools_directly(pipeline: Pipeline):
    """Demonstrate using tools directly without LLM."""
    print()
    print("=" * 80)
    print("🔧 DIRECT TOOL USAGE (No LLM Required)")
    print("=" * 80)
    print()

    # List tables
    print("📋 ListTablesTool:")
    print("-" * 60)
    tool = ListTablesTool(pipeline)
    result = tool.run()
    print(f"Found {len(result.data)} tables:")
    for table in result.data[:5]:
        print(f"  • {table['name']} ({table['column_count']} columns)")
    print()

    # Trace backward
    print("🔙 TraceBackwardTool:")
    print("-" * 60)
    tool = TraceBackwardTool(pipeline)
    result = tool.run(table="staging_dev.orders", column="amount")
    print("Sources for staging_dev.orders.amount:")
    print(result.message)
    print()

    # Trace forward
    print("🔜 TraceForwardTool:")
    print("-" * 60)
    tool = TraceForwardTool(pipeline)
    result = tool.run(table="raw_dev.orders", column="customer_id")
    print("Dependents of raw_dev.orders.customer_id:")
    print(result.message)
    print()

    # Search columns
    print("🔍 SearchColumnsTool:")
    print("-" * 60)
    tool = SearchColumnsTool(pipeline)
    result = tool.run(pattern="revenue")
    print("Columns matching 'revenue':")
    print(result.message)
    print()

    # Get table schema
    print("📊 GetTableSchemaTool:")
    print("-" * 60)
    tool = GetTableSchemaTool(pipeline)
    result = tool.run(table="analytics_dev.customer_metrics")
    print(result.message[:500])


def demo_export(pipeline: Pipeline):
    """Demonstrate JSON export for persistence."""
    print()
    print("=" * 80)
    print("💾 JSON EXPORT (For Persistence)")
    print("=" * 80)
    print()

    # Export to JSON
    data = JSONExporter.export(pipeline, include_metadata=True, include_queries=True)

    print("Exported pipeline:")
    print(f"  • Dialect: {data['dialect']}")
    print(f"  • Queries: {len(data['queries'])}")
    print(f"  • Tables: {len(data['tables'])}")
    print(f"  • Columns: {len(data['columns'])}")
    print(f"  • Edges: {len(data['edges'])}")
    print()

    # Save to file
    output_path = Path(__file__).parent / "enterprise_pipeline.json"
    JSONExporter.export_to_file(pipeline, str(output_path), include_queries=True)
    print(f"💾 Saved to: {output_path}")
    print()

    # Demonstrate round-trip
    print("🔄 Testing round-trip (load from JSON)...")
    restored = Pipeline.from_json_file(str(output_path))
    print(
        f"✅ Restored: {len(restored.table_graph.tables)} tables, {len(restored.columns)} columns"
    )


def demo_text_to_sql(pipeline: Pipeline, llm):
    """Demonstrate text-to-SQL capability with schema context."""
    print()
    print("=" * 80)
    print("💬 TEXT-TO-SQL (Schema-Aware Query Generation)")
    print("=" * 80)
    print()

    # Build comprehensive schema context with descriptions
    print("📋 Building schema context for LLM...")
    schema_info = []
    for table_name in sorted(pipeline.table_graph.tables.keys()):
        columns = list(pipeline.get_columns_by_table(table_name))
        col_details = []
        for col in columns:
            desc = f" -- {col.description}" if col.description else ""
            col_details.append(f"    {col.column_name}{desc}")
        schema_info.append(f"Table: {table_name}\nColumns:\n" + "\n".join(col_details))

    schema_context = "\n\n".join(schema_info)
    print(f"   ✓ Included {len(pipeline.table_graph.tables)} tables with descriptions")
    print()

    # Text-to-SQL prompt with rich schema context
    system_prompt = f"""You are a SQL expert for ClickHouse. Generate SQL queries based on natural language questions.

SCHEMA:
{schema_context}

RULES:
1. Use only the tables and columns from the schema above
2. Generate valid ClickHouse SQL syntax
3. Include appropriate JOINs when needed
4. Use table aliases for clarity
5. Return ONLY the SQL query, no explanations or markdown
"""

    # More comprehensive questions
    questions = [
        # Simple queries
        ("Simple", "Show me the top 10 customers by lifetime value"),
        ("Simple", "Find all customers who made more than 5 orders"),
        # Aggregation queries
        ("Aggregation", "What is the average order value by month?"),
        ("Aggregation", "Show total revenue by customer segment"),
        # Complex queries
        ("Complex", "Find platinum customers who are at risk of churning"),
        ("Complex", "What products have the highest revenue in the last quarter?"),
    ]

    for category, question in questions:
        print(f"[{category}] ❓ {question}")
        print("-" * 70)

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=question),
                ]
            )

            sql = response.content.strip()
            # Clean up markdown code blocks if present
            if sql.startswith("```"):
                sql = sql.split("```")[1]
                if sql.startswith("sql"):
                    sql = sql[3:]
                sql = sql.strip()

            print("📝 Generated SQL:")
            # Indent SQL for readability
            for line in sql.split("\n"):
                print(f"   {line}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Enterprise Demo with Ollama")
    parser.add_argument("--model", default="gpt-oss:20b", help="Ollama model to use")
    parser.add_argument(
        "--skip-descriptions", action="store_true", help="Skip LLM description generation"
    )
    parser.add_argument("--skip-agent", action="store_true", help="Skip LineageAgent demo")
    parser.add_argument("--skip-text-to-sql", action="store_true", help="Skip text-to-SQL demo")
    args = parser.parse_args()

    print()
    print("🚀 Enterprise Demo with Ollama")
    print("=" * 80)
    print()

    # Load pipeline
    pipeline = load_enterprise_pipeline()

    # Setup LLM
    llm = None
    if not (args.skip_descriptions and args.skip_agent and args.skip_text_to_sql):
        llm = setup_ollama_llm(args.model)

    # Demo: Direct tool usage (no LLM required)
    demo_tools_directly(pipeline)

    # Demo: Description generation (requires LLM)
    if llm and not args.skip_descriptions:
        demo_description_generation(pipeline, llm)

    # Demo: LineageAgent (requires LLM)
    if llm and not args.skip_agent:
        demo_lineage_agent(pipeline, llm)

    # Demo: Text-to-SQL (requires LLM)
    if llm and not args.skip_text_to_sql:
        demo_text_to_sql(pipeline, llm)

    # Demo: Export
    demo_export(pipeline)

    print()
    print("=" * 80)
    print("✅ Demo Complete!")
    print("=" * 80)
    print()
    print("💡 Tips:")
    print(f"   - For better results, try: python {__file__} --model qwen2.5-coder:7b")
    print(
        "   - Run without LLM: python {__file__} --skip-descriptions --skip-agent --skip-text-to-sql"
    )
    print("   - The exported JSON can be used to restore the pipeline later")
    print()


if __name__ == "__main__":
    main()
