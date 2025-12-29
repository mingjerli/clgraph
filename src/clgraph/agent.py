"""
LineageAgent - Unified natural language interface to lineage.

Provides a conversational interface for querying lineage information,
routing questions to appropriate tools automatically.

Example:
    from clgraph import Pipeline
    from clgraph.agent import LineageAgent
    from langchain_openai import ChatOpenAI

    pipeline = Pipeline.from_sql_files("queries/")
    agent = LineageAgent(pipeline, llm=ChatOpenAI())

    # Ask any question
    result = agent.query("Where does the revenue column come from?")
    print(result.answer)

    result = agent.query("Write SQL to get monthly revenue")
    print(result.data["sql"])
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

from .tools import ToolResult, create_tool_registry

if TYPE_CHECKING:
    from .pipeline import Pipeline


class QuestionType(Enum):
    """Types of questions the agent can handle."""

    LINEAGE_BACKWARD = "lineage_backward"  # Where does X come from?
    LINEAGE_FORWARD = "lineage_forward"  # What depends on X?
    LINEAGE_PATH = "lineage_path"  # How does X relate to Y?
    SCHEMA_TABLES = "schema_tables"  # What tables exist?
    SCHEMA_COLUMNS = "schema_columns"  # What columns does X have?
    SCHEMA_SEARCH = "schema_search"  # Find columns matching X
    GOVERNANCE_PII = "governance_pii"  # Which columns are PII?
    GOVERNANCE_OWNER = "governance_owner"  # Who owns X?
    SQL_GENERATE = "sql_generate"  # Write SQL to...
    SQL_EXPLAIN = "sql_explain"  # What does this query do?
    GENERAL = "general"  # General question


@dataclass
class AgentResult:
    """Result from agent query."""

    answer: str
    """Natural language answer to the question."""

    question_type: QuestionType
    """Detected type of question."""

    tool_used: Optional[str] = None
    """Name of tool that was used."""

    tool_result: Optional[ToolResult] = None
    """Raw result from tool execution."""

    data: Any = None
    """Structured data from the tool."""

    error: Optional[str] = None
    """Error message if query failed."""

    def __repr__(self) -> str:
        if self.error:
            return f"AgentResult(error={self.error!r})"
        preview = self.answer[:50] + "..." if len(self.answer) > 50 else self.answer
        return f"AgentResult(answer={preview!r}, tool={self.tool_used})"


# Question patterns for classification
QUESTION_PATTERNS = {
    QuestionType.LINEAGE_BACKWARD: [
        r"where\s+does?\s+.+\s+come\s+from",
        r"source\s+of",
        r"what\s+feeds?\s+into",
        r"origin\s+of",
        r"derived\s+from",
        r"trace\s+back",
        r"upstream",
    ],
    QuestionType.LINEAGE_FORWARD: [
        r"what\s+depends?\s+on",
        r"impact\s+of",
        r"what\s+(uses?|consumes?)",
        r"downstream",
        r"affects?",
        r"if\s+i\s+change",
    ],
    QuestionType.SCHEMA_TABLES: [
        r"what\s+tables?\s+(are|exist|do we have)",
        r"list\s+(all\s+)?tables?",
        r"show\s+(me\s+)?(all\s+)?tables?",
        r"available\s+tables?",
    ],
    QuestionType.SCHEMA_COLUMNS: [
        r"what\s+columns?\s+(does|are in|has)",
        r"schema\s+(of|for)",
        r"describe\s+table",
        r"show\s+(me\s+)?columns?",
    ],
    QuestionType.SCHEMA_SEARCH: [
        r"find\s+columns?",
        r"search\s+for",
        r"columns?\s+named",
        r"columns?\s+like",
    ],
    QuestionType.GOVERNANCE_PII: [
        r"pii",
        r"personal\s+data",
        r"sensitive\s+(data|columns?)",
        r"privacy",
    ],
    QuestionType.GOVERNANCE_OWNER: [
        r"who\s+owns?",
        r"owner\s+of",
        r"ownership",
        r"responsible\s+for",
    ],
    QuestionType.SQL_GENERATE: [
        r"(write|generate|create)\s+(a\s+)?(sql|query)",
        r"sql\s+(to|for|that)",
        r"query\s+to",
        r"how\s+do\s+i\s+(query|select|get)",
    ],
    QuestionType.SQL_EXPLAIN: [
        r"(explain|what\s+does)\s+.*(sql|query)",
        r"what\s+does\s+this\s+(query|sql)",
    ],
}


class LineageAgent:
    """
    Unified natural language interface to lineage.

    Routes questions to appropriate tools and formats responses
    for human consumption.

    Example:
        agent = LineageAgent(pipeline, llm=my_llm)

        # Lineage questions
        result = agent.query("Where does mart.revenue.total come from?")

        # Schema questions
        result = agent.query("What tables are available?")

        # SQL generation (requires LLM)
        result = agent.query("Write SQL to get revenue by month")

        # Direct tool access
        result = agent.run_tool("trace_backward", table="mart", column="total")
    """

    def __init__(
        self,
        pipeline: "Pipeline",
        llm: Any = None,
        verbose: bool = False,
    ):
        """
        Initialize LineageAgent.

        Args:
            pipeline: The clgraph Pipeline to query.
            llm: Optional LLM for SQL generation and complex queries.
            verbose: Whether to print debug information.
        """
        self.pipeline = pipeline
        self.llm = llm
        self.verbose = verbose

        # Create tool registry
        self.registry = create_tool_registry(pipeline, llm)

    def query(self, question: str) -> AgentResult:
        """
        Answer a natural language question about lineage.

        Args:
            question: Natural language question.

        Returns:
            AgentResult with answer and structured data.

        Example:
            result = agent.query("Where does revenue come from?")
            print(result.answer)
        """
        # Classify question
        question_type = self._classify_question(question)

        if self.verbose:
            print(f"Question type: {question_type}")

        # Route to appropriate handler
        try:
            if question_type == QuestionType.LINEAGE_BACKWARD:
                return self._handle_lineage_backward(question)
            elif question_type == QuestionType.LINEAGE_FORWARD:
                return self._handle_lineage_forward(question)
            elif question_type == QuestionType.SCHEMA_TABLES:
                return self._handle_list_tables(question)
            elif question_type == QuestionType.SCHEMA_COLUMNS:
                return self._handle_schema_columns(question)
            elif question_type == QuestionType.SCHEMA_SEARCH:
                return self._handle_search_columns(question)
            elif question_type == QuestionType.GOVERNANCE_PII:
                return self._handle_pii(question)
            elif question_type == QuestionType.GOVERNANCE_OWNER:
                return self._handle_owners(question)
            elif question_type == QuestionType.SQL_GENERATE:
                return self._handle_sql_generate(question)
            elif question_type == QuestionType.SQL_EXPLAIN:
                return self._handle_sql_explain(question)
            else:
                return self._handle_general(question)
        except Exception as e:
            return AgentResult(
                answer=f"Sorry, I encountered an error: {e}",
                question_type=question_type,
                error=str(e),
            )

    def run_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Run a specific tool directly.

        Args:
            tool_name: Name of the tool to run.
            **kwargs: Tool parameters.

        Returns:
            ToolResult from the tool.
        """
        return self.registry.run(tool_name, **kwargs)

    def list_tools(self) -> List[str]:
        """List available tool names."""
        return self.registry.tool_names()

    # =========================================================================
    # Question Classification
    # =========================================================================

    def _classify_question(self, question: str) -> QuestionType:
        """Classify question into a type."""
        question_lower = question.lower()

        for qtype, patterns in QUESTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return qtype

        return QuestionType.GENERAL

    def _extract_table_column(self, question: str) -> tuple:
        """Extract table and column references from question."""
        # First, try to match against known table names (handles schema.table.column)
        # Sort by length descending to match longer table names first
        known_tables = sorted(self.pipeline.table_graph.tables.keys(), key=len, reverse=True)

        for table_name in known_tables:
            # Check for table.column pattern
            pattern = re.escape(table_name) + r"\.([a-zA-Z_][a-zA-Z0-9_]*)"
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return table_name, match.group(1)

        # Fallback: Pattern for schema.table.column or table.column
        pattern = r"([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)"
        match = re.search(pattern, question)
        if match:
            return match.group(1), match.group(2)

        # Simple pattern: table.column (no schema)
        pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)"
        match = re.search(pattern, question)
        if match:
            potential_table = match.group(1)
            column = match.group(2)
            # Check if this is a known table
            if potential_table in self.pipeline.table_graph.tables:
                return potential_table, column

        # Try to find just column name
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question)
        # Check if any word is a column name
        for word in words:
            for col in self.pipeline.columns.values():
                if col.column_name.lower() == word.lower():
                    return col.table_name, col.column_name

        return None, None

    def _extract_table(self, question: str) -> Optional[str]:
        """Extract table name from question."""
        # Check each word against known tables
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]*", question)
        for word in words:
            if word in self.pipeline.table_graph.tables:
                return word
            # Try partial match
            for table_name in self.pipeline.table_graph.tables:
                if word.lower() in table_name.lower():
                    return table_name
        return None

    # =========================================================================
    # Question Handlers
    # =========================================================================

    def _handle_lineage_backward(self, question: str) -> AgentResult:
        """Handle 'where does X come from' questions."""
        table, column = self._extract_table_column(question)

        if not table or not column:
            return AgentResult(
                answer="I couldn't identify which column you're asking about. "
                "Please specify as 'table.column' (e.g., 'analytics.revenue.total').",
                question_type=QuestionType.LINEAGE_BACKWARD,
            )

        result = self.registry.run("trace_backward", table=table, column=column)

        if not result.success:
            return AgentResult(
                answer=result.error or "Failed to trace lineage.",
                question_type=QuestionType.LINEAGE_BACKWARD,
                tool_used="trace_backward",
                error=result.error,
            )

        return AgentResult(
            answer=result.message,
            question_type=QuestionType.LINEAGE_BACKWARD,
            tool_used="trace_backward",
            tool_result=result,
            data=result.data,
        )

    def _handle_lineage_forward(self, question: str) -> AgentResult:
        """Handle 'what depends on X' questions."""
        table, column = self._extract_table_column(question)

        if not table or not column:
            return AgentResult(
                answer="I couldn't identify which column you're asking about. "
                "Please specify as 'table.column'.",
                question_type=QuestionType.LINEAGE_FORWARD,
            )

        result = self.registry.run("trace_forward", table=table, column=column)

        if not result.success:
            return AgentResult(
                answer=result.error or "Failed to trace impact.",
                question_type=QuestionType.LINEAGE_FORWARD,
                tool_used="trace_forward",
                error=result.error,
            )

        return AgentResult(
            answer=result.message,
            question_type=QuestionType.LINEAGE_FORWARD,
            tool_used="trace_forward",
            tool_result=result,
            data=result.data,
        )

    def _handle_list_tables(self, question: str) -> AgentResult:
        """Handle 'what tables exist' questions."""
        result = self.registry.run("list_tables")

        return AgentResult(
            answer=result.message,
            question_type=QuestionType.SCHEMA_TABLES,
            tool_used="list_tables",
            tool_result=result,
            data=result.data,
        )

    def _handle_schema_columns(self, question: str) -> AgentResult:
        """Handle 'what columns does X have' questions."""
        table = self._extract_table(question)

        if not table:
            return AgentResult(
                answer="I couldn't identify which table you're asking about. "
                "Please specify the table name.",
                question_type=QuestionType.SCHEMA_COLUMNS,
            )

        result = self.registry.run("get_table_schema", table=table)

        if not result.success:
            return AgentResult(
                answer=result.error or "Failed to get table schema.",
                question_type=QuestionType.SCHEMA_COLUMNS,
                tool_used="get_table_schema",
                error=result.error,
            )

        return AgentResult(
            answer=result.message,
            question_type=QuestionType.SCHEMA_COLUMNS,
            tool_used="get_table_schema",
            tool_result=result,
            data=result.data,
        )

    def _handle_search_columns(self, question: str) -> AgentResult:
        """Handle 'find columns matching X' questions."""
        # Extract search pattern from question
        patterns = re.findall(
            r"(?:named|like|matching|called|containing|with)\s+['\"]?(\w+)['\"]?", question.lower()
        )
        if patterns:
            pattern = patterns[0]
        else:
            # Use words that might be the search term
            words = question.lower().split()
            # Skip common words
            skip = {
                "find",
                "search",
                "columns",
                "column",
                "for",
                "named",
                "like",
                "the",
                "a",
                "an",
                "containing",
                "with",
                "matching",
                "called",
                "show",
                "me",
                "all",
                "that",
                "have",
                "contain",
                "include",
            }
            # Also strip quotes from words
            cleaned_words = [w.strip("'\"") for w in words]
            pattern = next((w for w in cleaned_words if w not in skip and len(w) > 1), "")

        if not pattern:
            return AgentResult(
                answer="Please specify what you're searching for.",
                question_type=QuestionType.SCHEMA_SEARCH,
            )

        result = self.registry.run("search_columns", pattern=pattern)

        return AgentResult(
            answer=result.message,
            question_type=QuestionType.SCHEMA_SEARCH,
            tool_used="search_columns",
            tool_result=result,
            data=result.data,
        )

    def _handle_pii(self, question: str) -> AgentResult:
        """Handle PII-related questions."""
        table = self._extract_table(question)

        result = self.registry.run("find_pii_columns", table=table)

        return AgentResult(
            answer=result.message,
            question_type=QuestionType.GOVERNANCE_PII,
            tool_used="find_pii_columns",
            tool_result=result,
            data=result.data,
        )

    def _handle_owners(self, question: str) -> AgentResult:
        """Handle ownership questions."""
        table = self._extract_table(question)

        result = self.registry.run("get_owners", table=table)

        return AgentResult(
            answer=result.message,
            question_type=QuestionType.GOVERNANCE_OWNER,
            tool_used="get_owners",
            tool_result=result,
            data=result.data,
        )

    def _handle_sql_generate(self, question: str) -> AgentResult:
        """Handle SQL generation requests."""
        if self.llm is None:
            return AgentResult(
                answer="SQL generation requires an LLM. Please initialize the agent with an LLM.",
                question_type=QuestionType.SQL_GENERATE,
                error="No LLM configured",
            )

        result = self.registry.run("generate_sql", question=question)

        if not result.success:
            return AgentResult(
                answer=result.error or "Failed to generate SQL.",
                question_type=QuestionType.SQL_GENERATE,
                tool_used="generate_sql",
                error=result.error,
            )

        sql = result.data.get("sql", "")
        explanation = result.data.get("explanation", "")

        if explanation:
            answer = f"{explanation}\n\n```sql\n{sql}\n```"
        else:
            answer = f"```sql\n{sql}\n```"

        return AgentResult(
            answer=answer,
            question_type=QuestionType.SQL_GENERATE,
            tool_used="generate_sql",
            tool_result=result,
            data=result.data,
        )

    def _handle_sql_explain(self, question: str) -> AgentResult:
        """Handle SQL explanation requests."""
        if self.llm is None:
            return AgentResult(
                answer="SQL explanation requires an LLM. Please initialize the agent with an LLM.",
                question_type=QuestionType.SQL_EXPLAIN,
                error="No LLM configured",
            )

        # Extract SQL from question
        sql_match = re.search(r"```sql?\s*(.*?)\s*```", question, re.DOTALL)
        if sql_match:
            sql = sql_match.group(1)
        else:
            # Assume everything after common phrases is SQL
            for phrase in ["explain", "what does", "this query"]:
                if phrase in question.lower():
                    idx = question.lower().find(phrase)
                    sql = question[idx + len(phrase) :].strip()
                    if sql.startswith(":"):
                        sql = sql[1:].strip()
                    break
            else:
                sql = question

        result = self.registry.run("explain_query", sql=sql)

        return AgentResult(
            answer=result.message if result.success else result.error,
            question_type=QuestionType.SQL_EXPLAIN,
            tool_used="explain_query",
            tool_result=result,
            data=result.data,
        )

    def _handle_general(self, question: str) -> AgentResult:
        """Handle general questions."""
        # Try to be helpful by suggesting what we can do
        available_actions = [
            "- Trace where a column comes from (e.g., 'Where does mart.revenue.total come from?')",
            "- Find what depends on a column (e.g., 'What depends on raw.orders.amount?')",
            "- List tables (e.g., 'What tables are available?')",
            "- Get table schema (e.g., 'What columns does analytics.revenue have?')",
            "- Find PII columns (e.g., 'Which columns contain PII?')",
        ]

        if self.llm:
            available_actions.append("- Generate SQL (e.g., 'Write SQL to get revenue by month')")

        answer = (
            f"I'm not sure how to answer '{question}'. Here's what I can help with:\n\n"
            + "\n".join(available_actions)
        )

        return AgentResult(
            answer=answer,
            question_type=QuestionType.GENERAL,
        )


__all__ = [
    "LineageAgent",
    "AgentResult",
    "QuestionType",
]
