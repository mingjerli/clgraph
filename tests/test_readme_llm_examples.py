"""
Test LLM-dependent README examples with Ollama.

These tests require Ollama to be running locally with a model available.
They are skipped by default and can be run with:

    pytest tests/test_readme_llm_examples.py -v --run-llm

Or run all tests including LLM:

    pytest tests/ --run-llm

Prerequisites:
    1. Install Ollama: brew install ollama
    2. Start Ollama: ollama serve
    3. Pull a model: ollama pull llama3.2 (or qwen2.5-coder:7b)
"""

import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def ollama_llm():
    """Create Ollama LLM instance."""
    try:
        from langchain_ollama import ChatOllama

        # Try models in order of preference
        models_to_try = ["gpt-oss:20b", "qwen3-coder:30b", "llama3.1:8b", "llama3.2"]

        for model in models_to_try:
            try:
                llm = ChatOllama(
                    model=model,
                    temperature=0.3,
                )
                # Test connection
                response = llm.invoke("Say 'OK'")
                if response.content:
                    print(f"\nUsing Ollama model: {model}")
                    return llm
            except Exception:
                continue

        pytest.skip("No Ollama models available")
    except ImportError:
        pytest.skip("langchain-ollama not installed")
    except Exception as e:
        pytest.skip(f"Ollama not available: {e}")


@pytest.fixture
def sample_pipeline():
    """Create a sample pipeline for testing."""
    from clgraph import Pipeline

    queries = [
        (
            "raw.orders",
            """
            CREATE TABLE raw.orders AS
            SELECT order_id, user_email, amount, order_date
            FROM source.orders
        """,
        ),
        (
            "analytics.revenue",
            """
            CREATE TABLE analytics.revenue AS
            SELECT user_email, SUM(amount) as total_revenue
            FROM raw.orders
            GROUP BY user_email
        """,
        ),
    ]
    return Pipeline(queries, dialect="bigquery")


@pytest.fixture
def enterprise_pipeline():
    """Create a larger pipeline for realistic testing."""
    from pathlib import Path

    from clgraph import Pipeline

    sql_dir = Path(__file__).parent.parent / "examples" / "clickhouse_example"
    if not sql_dir.exists():
        pytest.skip("ClickHouse example SQL files not found")

    queries = []
    for sql_file in sorted(sql_dir.glob("*.sql")):
        if sql_file.name.startswith("00_"):
            continue
        content = sql_file.read_text()
        query_id = sql_file.stem
        queries.append((query_id, content))

    return Pipeline.from_tuples(
        queries,
        dialect="clickhouse",
        template_context={"env": "dev"},
    )


# =============================================================================
# LLM Description Generation Tests
# =============================================================================


@pytest.mark.llm
class TestLLMDescriptionGeneration:
    """Test LLM-powered description generation."""

    def test_generate_single_description(self, ollama_llm, sample_pipeline):
        """Test generating a description for a single column."""
        from clgraph.column import generate_description

        # Get a column without description
        col = sample_pipeline.get_column("raw.orders", "amount")
        assert col is not None
        assert col.description is None

        # Generate description
        generate_description(col, ollama_llm, sample_pipeline)

        # Should have description now
        assert col.description is not None
        assert len(col.description) > 0
        print(f"Generated: {col.description}")

    def test_generate_all_descriptions(self, ollama_llm, sample_pipeline):
        """Test generating descriptions for all columns."""
        sample_pipeline.llm = ollama_llm

        # Count columns without descriptions
        initial_without = sum(1 for c in sample_pipeline.columns.values() if not c.description)
        assert initial_without > 0

        # Generate all descriptions
        sample_pipeline.generate_all_descriptions(verbose=True)

        # Should have more descriptions now
        final_with = sum(1 for c in sample_pipeline.columns.values() if c.description)
        print(f"Generated descriptions for {final_with} columns")
        assert final_with > 0


# =============================================================================
# Lineage Agent with LLM Tests
# =============================================================================


@pytest.mark.llm
class TestLineageAgentWithLLM:
    """Test LineageAgent with LLM for SQL generation."""

    def test_agent_sql_generation(self, ollama_llm, sample_pipeline):
        """Test agent generates SQL with LLM."""
        from clgraph.agent import LineageAgent, QuestionType

        agent = LineageAgent(sample_pipeline, llm=ollama_llm)

        # SQL generation question
        result = agent.query("Write SQL to get total revenue by email")

        assert result.question_type == QuestionType.SQL_GENERATE
        assert result.tool_used == "generate_sql"
        # With LLM, should have SQL in data
        if result.data and "sql" in result.data:
            print(f"Generated SQL:\n{result.data['sql']}")
        else:
            # May not have LLM configured properly
            print(f"Result: {result.answer}")

    def test_agent_general_question(self, ollama_llm, sample_pipeline):
        """Test agent handles general questions with LLM."""
        from clgraph.agent import LineageAgent, QuestionType

        agent = LineageAgent(sample_pipeline, llm=ollama_llm)

        # General question that needs LLM
        result = agent.query("What is this pipeline used for?")

        assert result.question_type == QuestionType.GENERAL
        print(f"Answer: {result.answer}")


# =============================================================================
# Text-to-SQL Tool Tests
# =============================================================================


@pytest.mark.llm
class TestTextToSQL:
    """Test GenerateSQLTool with LLM."""

    def test_simple_query_generation(self, ollama_llm, sample_pipeline):
        """Test generating a simple SQL query."""
        from clgraph.tools import GenerateSQLTool

        tool = GenerateSQLTool(sample_pipeline, llm=ollama_llm)

        result = tool.run(question="Get all orders with amount over 100")

        assert result.success
        assert result.data is not None
        assert "sql" in result.data
        print(f"Generated SQL:\n{result.data['sql']}")

    def test_aggregation_query(self, ollama_llm, sample_pipeline):
        """Test generating an aggregation query."""
        from clgraph.tools import GenerateSQLTool

        tool = GenerateSQLTool(sample_pipeline, llm=ollama_llm)

        result = tool.run(question="Calculate total revenue grouped by email")

        assert result.success
        assert "sql" in result.data
        sql = result.data["sql"].lower()
        # Should have aggregation keywords
        assert "sum" in sql or "count" in sql or "group" in sql
        print(f"Generated SQL:\n{result.data['sql']}")

    def test_enterprise_pipeline_query(self, ollama_llm, enterprise_pipeline):
        """Test generating SQL for enterprise pipeline."""
        from clgraph.tools import GenerateSQLTool

        tool = GenerateSQLTool(enterprise_pipeline, llm=ollama_llm)

        result = tool.run(question="Find top 10 customers by lifetime value")

        assert result.success
        print(f"Generated SQL:\n{result.data['sql']}")


# =============================================================================
# Explain Query Tool Tests
# =============================================================================


@pytest.mark.llm
class TestExplainQuery:
    """Test ExplainQueryTool with LLM."""

    def test_explain_simple_query(self, ollama_llm, sample_pipeline):
        """Test explaining a SQL query."""
        from clgraph.tools import ExplainQueryTool

        tool = ExplainQueryTool(sample_pipeline, llm=ollama_llm)

        sql = """
        SELECT user_email, SUM(amount) as total
        FROM raw.orders
        GROUP BY user_email
        HAVING SUM(amount) > 1000
        """

        result = tool.run(sql=sql)

        assert result.success
        assert result.message
        print(f"Explanation:\n{result.message}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--run-llm"])
