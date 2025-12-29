"""
SQL generation tools.

Tools for generating SQL from natural language queries.
Requires an LLM for operation.
"""

import re
from typing import Dict, List, Optional, Tuple

from .base import LLMTool, ParameterSpec, ParameterType, ToolResult
from .context import ContextBuilder, ContextConfig

# =============================================================================
# Prompt Templates
# =============================================================================

GENERATE_SQL_PROMPT = """You are a SQL expert. Generate a SQL query to answer the user's question.

## Database Schema

{schema_context}

{relationship_section}

{notes_section}

## Question
{question}

## Instructions
- Generate ONLY the SQL query, no explanations unless asked
- Use {dialect} SQL syntax
- Use fully qualified table names when available
{extra_instructions}

## SQL Query
```sql
"""

GENERATE_SQL_WITH_EXPLANATION_PROMPT = """You are a SQL expert. Generate a SQL query to answer the user's question.

## Database Schema

{schema_context}

{relationship_section}

{notes_section}

## Question
{question}

## Instructions
- Use {dialect} SQL syntax
- Use fully qualified table names when available
- First provide a brief explanation of your approach
- Then provide the SQL query
{extra_instructions}

## Response Format
Explanation: <your explanation here>

```sql
<your SQL query here>
```
"""

TABLE_SELECTION_PROMPT = """Given the following database tables and a user question, identify which tables are needed to answer the question.

## Available Tables

{table_summaries}

## Question
{question}

## Instructions
- Return ONLY a JSON array of table names that are needed
- Include tables needed for joins even if not directly mentioned
- Be conservative - only include tables that are definitely needed

## Required Tables (JSON array)
"""


# =============================================================================
# SQL Generation Tool
# =============================================================================


class GenerateSQLTool(LLMTool):
    """
    Generate SQL from natural language.

    Uses LLM with pipeline schema context to convert natural language
    questions into SQL queries.

    Example:
        tool = GenerateSQLTool(pipeline, llm)
        result = tool.run(question="What is total revenue by month?")
        print(result.data["sql"])
    """

    name = "generate_sql"
    description = "Generate a SQL query from a natural language question"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "question": ParameterSpec(
                name="question",
                type=ParameterType.STRING,
                description="Natural language question to convert to SQL",
                required=True,
            ),
            "include_explanation": ParameterSpec(
                name="include_explanation",
                type=ParameterType.BOOLEAN,
                description="Whether to include an explanation of the query",
                required=False,
                default=True,
            ),
            "strategy": ParameterSpec(
                name="strategy",
                type=ParameterType.STRING,
                description="Strategy: 'direct' uses all tables, 'two_stage' selects tables first",
                required=False,
                default="direct",
                enum=["direct", "two_stage"],
            ),
        }

    def run(
        self,
        question: str,
        include_explanation: bool = True,
        strategy: str = "direct",
    ) -> ToolResult:
        try:
            if strategy == "two_stage":
                return self._generate_two_stage(question, include_explanation)
            else:
                return self._generate_direct(question, include_explanation)
        except Exception as e:
            return ToolResult.error_result(f"SQL generation failed: {e}")

    def _generate_direct(self, question: str, include_explanation: bool) -> ToolResult:
        """Generate SQL using all tables as context."""
        config = ContextConfig(
            include_descriptions=True,
            include_pii_flags=True,
            include_lineage=True,
        )
        builder = ContextBuilder(self.pipeline, config)

        # Build context
        schema_context = builder.build_schema_context()
        tables = builder.get_table_names()
        relationship_context = builder.build_relationship_context(tables)

        # Build notes
        notes = self._build_notes(tables)
        notes_section = self._format_notes(notes)

        # Choose prompt
        if include_explanation:
            prompt = GENERATE_SQL_WITH_EXPLANATION_PROMPT
        else:
            prompt = GENERATE_SQL_PROMPT

        prompt = prompt.format(
            schema_context=schema_context,
            relationship_section=relationship_context,
            notes_section=notes_section,
            question=question,
            dialect=self.pipeline.dialect,
            extra_instructions="",
        )

        # Call LLM
        response = self.call_llm(prompt)

        # Parse response
        sql, explanation = self._parse_response(response)

        return ToolResult.success_result(
            data={
                "sql": sql,
                "explanation": explanation,
                "tables_used": tables,
                "strategy": "direct",
            },
            message=f"Generated SQL query for: {question[:50]}..."
            if len(question) > 50
            else f"Generated SQL query for: {question}",
        )

    def _generate_two_stage(self, question: str, include_explanation: bool) -> ToolResult:
        """Generate SQL using two-stage approach (select tables first)."""
        config = ContextConfig(
            include_descriptions=True,
            include_pii_flags=True,
            include_lineage=True,
        )
        builder = ContextBuilder(self.pipeline, config)

        # Stage 1: Select tables
        selected_tables = self._select_tables(question, builder)

        # Expand with lineage
        expanded_tables = builder.expand_with_lineage(selected_tables)

        # Stage 2: Build context and generate
        schema_context = builder.build_context_for_tables(expanded_tables)
        lineage_context = builder.build_lineage_context(expanded_tables)

        # Build notes
        notes = self._build_notes(expanded_tables)
        notes_section = self._format_notes(notes)

        # Build prompt
        if include_explanation:
            prompt = GENERATE_SQL_WITH_EXPLANATION_PROMPT
        else:
            prompt = GENERATE_SQL_PROMPT

        prompt = prompt.format(
            schema_context=schema_context,
            relationship_section=lineage_context,
            notes_section=notes_section,
            question=question,
            dialect=self.pipeline.dialect,
            extra_instructions="- Use ONLY the tables listed above",
        )

        # Call LLM
        response = self.call_llm(prompt)

        # Parse response
        sql, explanation = self._parse_response(response)

        return ToolResult.success_result(
            data={
                "sql": sql,
                "explanation": explanation,
                "tables_used": expanded_tables,
                "strategy": "two_stage",
            },
            message=f"Generated SQL query using {len(expanded_tables)} tables",
        )

    def _select_tables(self, question: str, builder: ContextBuilder) -> List[str]:
        """Select relevant tables using LLM."""
        # Get table summaries
        summaries = builder.get_all_tables()
        summaries_text = self._format_table_summaries(summaries)

        prompt = TABLE_SELECTION_PROMPT.format(table_summaries=summaries_text, question=question)

        try:
            response = self.call_llm(prompt)
            selected = self._parse_table_list(response)

            if selected:
                # Validate tables exist
                valid = [t for t in selected if t in self.pipeline.table_graph.tables]
                if valid:
                    return valid
        except Exception:
            pass

        # Fallback to keyword selection
        return builder.select_tables_by_keywords(question)

    def _format_table_summaries(self, summaries) -> str:
        """Format table summaries for selection prompt."""
        lines = []
        for s in summaries:
            line = f"- **{s.table_name}**"
            if s.description:
                line += f": {s.description}"
            cols = s.columns[:10]
            line += f"\n  Columns: {', '.join(cols)}"
            if len(s.columns) > 10:
                line += f" (+{len(s.columns) - 10} more)"
            lines.append(line)
        return "\n\n".join(lines)

    def _parse_table_list(self, response: str) -> List[str]:
        """Parse table list from LLM response."""
        import json

        # Try to find JSON array
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []

    def _build_notes(self, tables: List[str]) -> List[str]:
        """Build notes about PII and other concerns."""
        notes = []

        # Find PII columns
        pii_columns = []
        for table in tables:
            for col in self.pipeline.get_columns_by_table(table):
                if col.pii:
                    pii_columns.append(f"{table}.{col.column_name}")

        if pii_columns:
            preview = pii_columns[:5]
            note = f"Columns marked [PII] contain sensitive data: {', '.join(preview)}"
            if len(pii_columns) > 5:
                note += f" (+{len(pii_columns) - 5} more)"
            notes.append(note)

        return notes

    def _format_notes(self, notes: List[str]) -> str:
        """Format notes section for prompt."""
        if not notes:
            return ""
        return "## Important Notes\n\n" + "\n".join(f"- {n}" for n in notes)

    def _parse_response(self, response: str) -> Tuple[str, Optional[str]]:
        """Parse SQL and explanation from LLM response."""
        # Try to extract SQL from code block
        sql_match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            sql = sql_match.group(1).strip()
        else:
            # Try generic code block
            code_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if code_match:
                sql = code_match.group(1).strip()
            else:
                # Assume whole response is SQL
                sql = response.strip()

        # Try to extract explanation
        explanation = None
        expl_match = re.search(r"[Ee]xplanation:\s*(.*?)(?:```|$)", response, re.DOTALL)
        if expl_match:
            explanation = expl_match.group(1).strip()

        return sql, explanation


class ExplainQueryTool(LLMTool):
    """
    Explain an existing SQL query.

    Analyzes a SQL query and provides a natural language explanation
    of what it does, with context from the pipeline schema.

    Example:
        tool = ExplainQueryTool(pipeline, llm)
        result = tool.run(sql="SELECT ... FROM ...")
    """

    name = "explain_query"
    description = "Explain what a SQL query does in natural language"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "sql": ParameterSpec(
                name="sql",
                type=ParameterType.STRING,
                description="SQL query to explain",
                required=True,
            ),
            "detail_level": ParameterSpec(
                name="detail_level",
                type=ParameterType.STRING,
                description="Level of detail: 'brief', 'normal', 'detailed'",
                required=False,
                default="normal",
                enum=["brief", "normal", "detailed"],
            ),
        }

    def run(self, sql: str, detail_level: str = "normal") -> ToolResult:
        # Extract table names from SQL
        tables = self._extract_tables(sql)

        # Build context for those tables
        config = ContextConfig(include_descriptions=True)
        builder = ContextBuilder(self.pipeline, config)

        valid_tables = [t for t in tables if t in self.pipeline.table_graph.tables]
        schema_context = builder.build_context_for_tables(valid_tables) if valid_tables else ""

        # Build prompt
        detail_instructions = {
            "brief": "Provide a one-sentence summary of what this query does.",
            "normal": "Explain what this query does in 2-3 sentences.",
            "detailed": "Provide a detailed explanation including: purpose, tables used, joins, filters, and output.",
        }

        prompt = f"""Explain the following SQL query.

## Schema Context
{schema_context if schema_context else "(No schema context available)"}

## SQL Query
```sql
{sql}
```

## Instructions
{detail_instructions[detail_level]}

## Explanation
"""

        try:
            explanation = self.call_llm(prompt).strip()

            return ToolResult.success_result(
                data={
                    "sql": sql,
                    "explanation": explanation,
                    "tables_referenced": valid_tables,
                },
                message=explanation[:200] + "..." if len(explanation) > 200 else explanation,
            )
        except Exception as e:
            return ToolResult.error_result(f"Failed to explain query: {e}")

    def _extract_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL (simple regex approach)."""
        # Match FROM and JOIN clauses
        pattern = r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_.]*)"
        matches = re.findall(pattern, sql, re.IGNORECASE)
        return list(set(matches))


__all__ = [
    "GenerateSQLTool",
    "ExplainQueryTool",
]
