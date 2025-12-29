"""
Shared context building utilities for lineage tools.

Provides ContextBuilder for creating rich context from Pipeline metadata,
used by SQL generation, schema tools, and other components.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from ..pipeline import Pipeline


@dataclass
class TableInfo:
    """Information about a table for context building."""

    table_name: str
    description: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    column_descriptions: Dict[str, str] = field(default_factory=dict)
    column_pii: Set[str] = field(default_factory=set)
    column_owners: Dict[str, str] = field(default_factory=dict)
    is_source: bool = False
    created_by: Optional[str] = None
    source_tables: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "table_name": self.table_name,
            "description": self.description,
            "columns": self.columns,
            "column_descriptions": self.column_descriptions,
            "is_source": self.is_source,
            "created_by": self.created_by,
            "source_tables": self.source_tables,
        }


@dataclass
class ContextConfig:
    """Configuration for context building."""

    include_descriptions: bool = True
    """Include table and column descriptions."""

    include_pii_flags: bool = True
    """Mark PII columns in output."""

    include_owners: bool = False
    """Include ownership information."""

    include_lineage: bool = True
    """Include lineage relationships."""

    include_source_tables: bool = True
    """Include source table information for derived tables."""

    max_tables: int = 50
    """Maximum tables to include in context."""

    max_columns_per_table: int = 100
    """Maximum columns per table."""

    max_description_length: int = 200
    """Truncate descriptions longer than this."""


class ContextBuilder:
    """
    Builds rich context from Pipeline metadata.

    Used by multiple tools to create consistent context for LLM prompts
    and structured output. Extracts schema, descriptions, lineage,
    and metadata from the Pipeline.

    Example:
        builder = ContextBuilder(pipeline)

        # Get all tables as structured data
        tables = builder.get_all_tables()

        # Build text context for LLM
        context = builder.build_schema_context()

        # Build context for specific tables
        context = builder.build_context_for_tables(["analytics.revenue"])

        # Expand tables with lineage
        tables = builder.expand_with_lineage(["analytics.revenue"])
    """

    def __init__(self, pipeline: "Pipeline", config: Optional[ContextConfig] = None):
        """
        Initialize ContextBuilder.

        Args:
            pipeline: The clgraph Pipeline to build context from.
            config: Optional configuration for context building.
        """
        self.pipeline = pipeline
        self.config = config or ContextConfig()

    # =========================================================================
    # Structured Data Methods
    # =========================================================================

    def get_all_tables(self) -> List[TableInfo]:
        """
        Get information about all tables in the pipeline.

        Returns:
            List of TableInfo objects for each table.
        """
        tables = []
        for table_name in self.pipeline.table_graph.tables:
            table_info = self.get_table_info(table_name)
            if table_info:
                tables.append(table_info)
        return tables

    def get_table_info(self, table_name: str) -> Optional[TableInfo]:
        """
        Get detailed information about a specific table.

        Args:
            table_name: Name of the table.

        Returns:
            TableInfo object or None if table not found.
        """
        table_node = self.pipeline.table_graph.tables.get(table_name)
        if not table_node:
            return None

        # Get columns for this table
        columns = self.pipeline.get_columns_by_table(table_name)

        # Filter to output columns to avoid duplicates
        output_columns = [c for c in columns if c.layer == "output"]
        if not output_columns:
            output_columns = columns

        # Limit columns if needed
        if len(output_columns) > self.config.max_columns_per_table:
            output_columns = output_columns[: self.config.max_columns_per_table]

        # Build column info
        column_names = [c.column_name for c in output_columns]
        column_descriptions = {}
        column_pii = set()
        column_owners = {}

        for col in output_columns:
            if col.description:
                column_descriptions[col.column_name] = col.description
            if col.pii:
                column_pii.add(col.column_name)
            if col.owner:
                column_owners[col.column_name] = col.owner

        # Get source tables
        source_tables = []
        if table_node.created_by:
            query = self.pipeline.table_graph.queries.get(table_node.created_by)
            if query:
                source_tables = list(query.source_tables)

        return TableInfo(
            table_name=table_name,
            description=table_node.description,
            columns=column_names,
            column_descriptions=column_descriptions,
            column_pii=column_pii,
            column_owners=column_owners,
            is_source=table_node.is_source,
            created_by=table_node.created_by,
            source_tables=source_tables,
        )

    def get_table_names(self, include_sources: bool = True) -> List[str]:
        """
        Get list of all table names.

        Args:
            include_sources: Whether to include source tables.

        Returns:
            List of table names.
        """
        tables = []
        for name, node in self.pipeline.table_graph.tables.items():
            if include_sources or not node.is_source:
                tables.append(name)
        return sorted(tables)

    def get_pii_columns(self, table_name: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get all PII-flagged columns.

        Args:
            table_name: Optional filter by table.

        Returns:
            List of dicts with table, column, description.
        """
        pii_columns = []
        for col in self.pipeline.columns.values():
            if col.pii:
                if table_name is None or col.table_name == table_name:
                    pii_columns.append(
                        {
                            "table": col.table_name,
                            "column": col.column_name,
                            "description": col.description,
                            "owner": col.owner,
                        }
                    )
        return pii_columns

    def get_columns_by_owner(self, owner: str) -> List[Dict[str, str]]:
        """
        Get all columns owned by a specific owner.

        Args:
            owner: Owner name to filter by.

        Returns:
            List of dicts with table, column info.
        """
        columns = []
        for col in self.pipeline.columns.values():
            if col.owner == owner:
                columns.append(
                    {
                        "table": col.table_name,
                        "column": col.column_name,
                        "description": col.description,
                        "pii": col.pii,
                    }
                )
        return columns

    def get_columns_by_tag(self, tag: str) -> List[Dict[str, str]]:
        """
        Get all columns with a specific tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of dicts with table, column info.
        """
        columns = []
        for col in self.pipeline.columns.values():
            if tag in col.tags:
                columns.append(
                    {
                        "table": col.table_name,
                        "column": col.column_name,
                        "description": col.description,
                        "tags": list(col.tags),
                    }
                )
        return columns

    # =========================================================================
    # Lineage Methods
    # =========================================================================

    def expand_with_lineage(self, tables: List[str]) -> List[str]:
        """
        Expand table list with lineage-related tables.

        For each table, adds its source tables to help with
        understanding join relationships.

        Args:
            tables: Initial list of table names.

        Returns:
            Expanded list including source tables.
        """
        if not self.config.include_lineage:
            return tables

        expanded = set(tables)

        for table_name in tables:
            table_node = self.pipeline.table_graph.tables.get(table_name)
            if not table_node:
                continue

            # Add source tables
            if table_node.created_by:
                query = self.pipeline.table_graph.queries.get(table_node.created_by)
                if query:
                    expanded.update(query.source_tables)

        return list(expanded)

    def get_table_relationships(self, tables: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get relationships between tables.

        Args:
            tables: Optional list to filter by. If None, all tables.

        Returns:
            List of relationship dicts with source, target, type.
        """
        relationships = []
        filter_set = set(tables) if tables else None

        for table_name, table_node in self.pipeline.table_graph.tables.items():
            if filter_set and table_name not in filter_set:
                continue

            if table_node.created_by:
                query = self.pipeline.table_graph.queries.get(table_node.created_by)
                if query:
                    for source_table in query.source_tables:
                        if filter_set is None or source_table in filter_set:
                            relationships.append(
                                {
                                    "source": source_table,
                                    "target": table_name,
                                    "type": "derives_from",
                                    "query_id": table_node.created_by,
                                }
                            )

        return relationships

    # =========================================================================
    # Text Context Methods (for LLM prompts)
    # =========================================================================

    def build_schema_context(self, tables: Optional[List[str]] = None) -> str:
        """
        Build text context describing the schema.

        Args:
            tables: Optional list of tables to include. If None, all tables.

        Returns:
            Formatted string describing the schema.
        """
        if tables is None:
            tables = self.get_table_names()

        # Apply max tables limit
        if len(tables) > self.config.max_tables:
            # Prioritize derived tables over source tables
            source_tables = [t for t in tables if self.pipeline.table_graph.tables[t].is_source]
            derived_tables = [t for t in tables if t not in source_tables]
            remaining = self.config.max_tables - len(derived_tables)
            tables = derived_tables + source_tables[: max(0, remaining)]

        return self.build_context_for_tables(tables)

    def build_context_for_tables(self, tables: List[str]) -> str:
        """
        Build text context for specific tables.

        Args:
            tables: List of table names to include.

        Returns:
            Formatted string describing the tables.
        """
        context_parts = []

        for table_name in sorted(tables):
            table_context = self._format_table_context(table_name)
            if table_context:
                context_parts.append(table_context)

        return "\n\n".join(context_parts)

    def _format_table_context(self, table_name: str) -> Optional[str]:
        """Format a single table for text context."""
        table_info = self.get_table_info(table_name)
        if not table_info:
            return None

        lines = [f"### {table_name}"]

        # Table description
        if self.config.include_descriptions and table_info.description:
            desc = self._truncate(table_info.description)
            lines.append(f"Description: {desc}")

        # Table type
        if table_info.is_source:
            lines.append("(Source table)")
        elif table_info.source_tables and self.config.include_source_tables:
            sources = ", ".join(table_info.source_tables[:3])
            if len(table_info.source_tables) > 3:
                sources += f" (+{len(table_info.source_tables) - 3} more)"
            lines.append(f"Sources: {sources}")

        lines.append("")
        lines.append("Columns:")

        # Columns
        for col_name in table_info.columns:
            col_line = f"  - {col_name}"

            if self.config.include_descriptions:
                desc = table_info.column_descriptions.get(col_name)
                if desc:
                    desc = self._truncate(desc, 100)
                    col_line += f": {desc}"

            if self.config.include_pii_flags and col_name in table_info.column_pii:
                col_line += " [PII]"

            if self.config.include_owners:
                owner = table_info.column_owners.get(col_name)
                if owner:
                    col_line += f" (owner: {owner})"

            lines.append(col_line)

        return "\n".join(lines)

    def build_relationship_context(self, tables: Optional[List[str]] = None) -> str:
        """
        Build text context describing table relationships.

        Args:
            tables: Optional list to filter by.

        Returns:
            Formatted string describing relationships.
        """
        relationships = self.get_table_relationships(tables)
        if not relationships:
            return ""

        lines = ["## Table Relationships", ""]
        for rel in relationships:
            lines.append(f"- {rel['target']} is derived from {rel['source']}")

        return "\n".join(lines)

    def build_lineage_context(self, tables: List[str]) -> str:
        """
        Build text context describing column lineage.

        Args:
            tables: Tables to include lineage for.

        Returns:
            Formatted string describing column lineage.
        """
        if not self.config.include_lineage:
            return ""

        lineage_info = []

        for table_name in tables:
            columns = self.pipeline.get_columns_by_table(table_name)
            output_columns = [c for c in columns if c.layer == "output"]

            for col in output_columns[:10]:  # Limit per table
                sources = self.pipeline.trace_column_backward(table_name, col.column_name)
                relevant_sources = [s for s in sources if s.table_name in tables]

                if relevant_sources and relevant_sources[0].table_name != table_name:
                    source_str = ", ".join(
                        f"{s.table_name}.{s.column_name}" for s in relevant_sources[:3]
                    )
                    lineage_info.append(f"- {table_name}.{col.column_name} <- {source_str}")

        if not lineage_info:
            return ""

        return "## Column Lineage\n\n" + "\n".join(lineage_info[:20])

    # =========================================================================
    # Table Selection (for two-stage approaches)
    # =========================================================================

    def select_tables_by_keywords(
        self, question: str, min_tables: int = 3, max_tables: int = 10
    ) -> List[str]:
        """
        Select relevant tables based on keyword matching.

        Simple heuristic for table selection without LLM.

        Args:
            question: Natural language question.
            min_tables: Minimum tables to return.
            max_tables: Maximum tables to return.

        Returns:
            List of relevant table names.
        """
        question_lower = question.lower()
        words = set(re.findall(r"\w+", question_lower))

        scored_tables = []

        for table_name in self.pipeline.table_graph.tables:
            score = 0

            # Check table name
            table_words = set(re.findall(r"\w+", table_name.lower()))
            score += len(words & table_words) * 2

            # Check column names and descriptions
            for col in self.pipeline.get_columns_by_table(table_name):
                col_words = set(re.findall(r"\w+", col.column_name.lower()))
                if words & col_words:
                    score += 1

                if col.description:
                    desc_words = set(re.findall(r"\w+", col.description.lower()))
                    if words & desc_words:
                        score += 0.5

            if score > 0:
                scored_tables.append((table_name, score))

        # Sort by score
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        selected = [t[0] for t in scored_tables[:max_tables]]

        # Ensure minimum
        if len(selected) < min_tables:
            all_tables = list(self.pipeline.table_graph.tables.keys())
            for table in all_tables:
                if table not in selected:
                    selected.append(table)
                    if len(selected) >= min_tables:
                        break

        return selected

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _truncate(self, text: str, max_length: Optional[int] = None) -> str:
        """Truncate text to max length."""
        max_len = max_length or self.config.max_description_length
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."


__all__ = [
    "TableInfo",
    "ContextConfig",
    "ContextBuilder",
]
