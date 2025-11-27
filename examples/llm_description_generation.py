"""
Example: LLM-Powered Description Generation

This example demonstrates how to use LangChain LLMs to automatically generate
natural language descriptions for columns in a SQL pipeline.

Requirements:
- Install: uv pip install -e .
- For Ollama: Install Ollama and run: ollama pull llama3.2
- For OpenAI: Set OPENAI_API_KEY environment variable
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clpipe.multi_query import MultiQueryParser
from clpipe.pipeline import PipelineLineageBuilder


def example_with_ollama():
    """Example using local Ollama with qwen3-coder:30b (free, no API key needed)"""
    print("=" * 80)
    print("Example 1: Using Ollama with qwen3-coder:30b (Local LLM)")
    print("=" * 80)
    print()

    # Sample SQL pipeline
    sql_queries = [
        """
        CREATE OR REPLACE TABLE staging.user_orders AS
        SELECT
            user_id,
            order_id,
            order_date,
            amount,
            status
        FROM raw.orders
        """,
        """
        CREATE OR REPLACE TABLE analytics.user_metrics AS
        SELECT
            user_id,
            COUNT(*) as order_count,
            SUM(amount) as total_revenue,
            AVG(amount) as avg_order_value,
            MAX(order_date) as last_order_date
        FROM staging.user_orders
        WHERE status = 'completed'
        GROUP BY user_id
        """,
    ]

    # Parse the pipeline
    print("📊 Parsing SQL pipeline...")
    parser = MultiQueryParser()
    table_graph = parser.parse_queries(sql_queries)

    # Build lineage graph
    builder = PipelineLineageBuilder()
    lineage_graph = builder.build(table_graph)
    print(f"✅ Found {len(lineage_graph.columns)} columns")
    print()

    # IMPORTANT: Set source descriptions BEFORE generating descriptions
    # This allows the LLM to use source column context when generating
    print("📝 Setting source column descriptions...")
    for col in lineage_graph.columns.values():
        if col.table_name == "raw.orders":
            if col.column_name == "user_id":
                col.set_source_description("Unique identifier for the user")
            elif col.column_name == "order_id":
                col.set_source_description("Unique identifier for the order")
            elif col.column_name == "amount":
                col.set_source_description("Order amount in USD")
            elif col.column_name == "order_date":
                col.set_source_description("Date when order was placed")
            elif col.column_name == "status":
                col.set_source_description("Order status: pending, completed, cancelled")
    print()

    # Configure LLM (Ollama with qwen3-coder:30b)
    print("🤖 Configuring Ollama LLM...")
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model="qwen3-coder:30b",
            temperature=0.3,  # Lower temperature for more consistent descriptions
        )
        lineage_graph.llm = llm
        print("✅ Ollama configured (model: qwen3-coder:30b)")
    except Exception as e:
        print(f"❌ Failed to configure Ollama: {e}")
        print("💡 Make sure Ollama is installed and running:")
        print("   brew install ollama")
        print("   ollama pull qwen3-coder:30b")
        print("   ollama serve")
        return
    print()

    # Generate descriptions using LLM
    print("🔮 Generating descriptions using LLM...")
    print("(This may take 10-30 seconds depending on your machine)")
    print()

    try:
        lineage_graph.generate_all_descriptions(verbose=True)
        print()

        # Display results
        print("=" * 80)
        print("Generated Descriptions")
        print("=" * 80)
        print()

        # Group by table
        tables = {}
        for col in lineage_graph.columns.values():
            if col.table_name not in tables:
                tables[col.table_name] = []
            tables[col.table_name].append(col)

        for table_name in sorted(tables.keys()):
            print(f"📊 {table_name}")
            print("-" * 80)
            for col in sorted(tables[table_name], key=lambda c: c.column_name):
                source_marker = (
                    "👤 USER"
                    if col.description_source and col.description_source.value == "source"
                    else "🤖 LLM"
                )
                print(f"  {col.column_name:20} [{source_marker}] {col.description}")
            print()

    except Exception as e:
        print(f"❌ Failed to generate descriptions: {e}")
        import traceback

        traceback.print_exc()


def example_with_openai():
    """Example using OpenAI GPT-4 (requires API key)"""
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping OpenAI example - OPENAI_API_KEY not set")
        return

    print("=" * 80)
    print("Example 2: Using OpenAI GPT-4")
    print("=" * 80)
    print()

    # Sample SQL
    sql = """
    CREATE OR REPLACE TABLE user_engagement AS
    SELECT
        user_id,
        DATE_TRUNC('week', activity_date) as week,
        COUNT(DISTINCT session_id) as weekly_sessions,
        SUM(page_views) as total_page_views,
        AVG(session_duration_minutes) as avg_session_duration
    FROM user_activity
    WHERE activity_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY user_id, DATE_TRUNC('week', activity_date)
    """

    # Parse
    parser = MultiQueryParser()
    table_graph = parser.parse_queries([sql])
    builder = PipelineLineageBuilder()
    lineage_graph = builder.build(table_graph)

    # Configure OpenAI
    print("🤖 Configuring OpenAI GPT-4...")
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    lineage_graph.llm = llm
    print("✅ OpenAI configured")
    print()

    # Generate descriptions
    print("🔮 Generating descriptions using GPT-4...")
    lineage_graph.generate_all_descriptions(verbose=True)
    print()

    # Display results
    print("Generated Descriptions:")
    print("-" * 80)
    for col in lineage_graph.columns.values():
        if col.description:
            print(f"{col.full_name:40} {col.description}")


def example_fallback():
    """Example showing fallback when no LLM is available"""
    print("=" * 80)
    print("Example 3: Fallback Mode (No LLM)")
    print("=" * 80)
    print()

    sql = """
    SELECT
        user_id,
        total_revenue,
        order_count
    FROM analytics.user_metrics
    """

    parser = MultiQueryParser()
    table_graph = parser.parse_queries([sql])
    builder = PipelineLineageBuilder()
    lineage_graph = builder.build(table_graph)

    # Don't set an LLM - columns will use fallback generation
    print("⚠️  No LLM configured - columns will show expression-based descriptions")
    print()

    print("Column Information (without LLM):")
    print("-" * 80)
    for col in lineage_graph.columns.values():
        # Show column expression instead of trying to generate description
        expr = col.expression if col.expression else "N/A"
        print(f"{col.full_name:40} Expression: {expr}")
    print()


if __name__ == "__main__":
    print("\n")
    print("🚀 LLM Description Generation Examples")
    print("=" * 80)
    print()

    # Run examples
    example_with_ollama()
    print("\n" * 2)

    # Uncomment to run OpenAI example if you have an API key
    # example_with_openai()
    # print("\n" * 2)

    example_fallback()

    print("\n")
    print("✅ Examples complete!")
    print()
    print("💡 Tips:")
    print("   - Ollama is free and runs locally (recommended for development)")
    print("   - OpenAI provides higher quality descriptions but requires API key")
    print("   - Fallback mode works without any LLM but produces simple descriptions")
    print()
