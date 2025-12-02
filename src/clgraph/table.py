"""
Table-level pipeline models and dependency graph.

Contains TableNode and TableDependencyGraph for tracking table-level lineage
across multi-query pipelines.
"""

from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from typing import Dict, List, Optional, Set

from .models import ParsedQuery


@dataclass
class TableNode:
    """
    Represents a table in the pipeline (source, intermediate, or final).
    Includes description field for table-level documentation.
    """

    table_name: str  # Fully qualified name (e.g., "raw.orders")
    is_source: bool  # True if external source (not created by any query)

    # Queries that reference this table
    created_by: Optional[str] = None  # Query ID that creates this table
    modified_by: List[str] = field(default_factory=list)  # Query IDs that modify this table
    read_by: List[str] = field(default_factory=list)  # Query IDs that read this table

    # Columns (if known)
    columns: Set[str] = field(default_factory=set)  # Column names in this table

    # Template support
    template_pattern: Optional[str] = (
        None  # Original template (e.g., "{{project}}.analytics.users")
    )
    resolved_name: Optional[str] = None  # Resolved name (e.g., "production.analytics.users")

    # Description
    description: Optional[str] = None

    @property
    def effective_name(self) -> str:
        """Return resolved name if available, otherwise template pattern"""
        return self.resolved_name or self.template_pattern or self.table_name

    def __str__(self) -> str:
        """Return table name as string representation"""
        return self.table_name

    def __repr__(self) -> str:
        """Return detailed representation"""
        return f"TableNode({self.table_name!r}, is_source={self.is_source})"

    def generate_description(self, llm, lineage_graph):
        """
        Generate table-level description using LLM.

        Args:
            llm: LangChain LLM instance (BaseChatModel)
            lineage_graph: The pipeline lineage graph for column lookup
        """
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            pass

        # Build prompt
        prompt = self._build_description_prompt(lineage_graph)

        # Call LLM
        try:
            from langchain_core.prompts import ChatPromptTemplate

            template = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a data documentation expert. Generate concise table descriptions.",
                    ),
                    ("user", prompt),
                ]
            )

            chain = template | llm
            response = chain.invoke({})

            self.description = response.content.strip()
        except Exception:
            # Fallback to simple rule-based description
            self._generate_fallback_description()

    def _build_description_prompt(self, lineage_graph) -> str:
        """Build LLM prompt for table description generation"""
        lines = [
            f"Table: {self.table_name}",
            "",
            "Columns:",
        ]

        # Get all columns for this table
        columns = self.get_columns(lineage_graph)
        for col in columns[:20]:  # Limit to first 20 columns
            col_info = f"- {col.column_name}"
            if col.description:
                col_info += f": {col.description}"
            lines.append(col_info)

        if len(columns) > 20:
            lines.append(f"- ... and {len(columns) - 20} more columns")

        lines.extend(
            [
                "",
                "Generate a table description that:",
                "- Is one sentence, max 20 words",
                "- Summarizes the purpose of this table",
                "- Uses natural language",
                "",
                "Return ONLY the description.",
            ]
        )

        return "\n".join(lines)

    def _generate_fallback_description(self):
        """Generate simple fallback description without LLM"""
        # Humanize table name
        parts = self.table_name.split(".")
        table_part = parts[-1]  # Get last part (actual table name)
        words = table_part.replace("_", " ").split()
        base_desc = " ".join(word.capitalize() for word in words) + " table"

        self.description = base_desc

    def get_columns(self, lineage_graph) -> List:
        """
        Get all columns for this table.

        Args:
            lineage_graph: The pipeline lineage graph

        Returns:
            List of ColumnNode objects for this table
        """

        return [col for col in lineage_graph.columns.values() if col.table_name == self.table_name]


@dataclass
class TableDependencyGraph:
    """
    DAG of tables and the queries that connect them.
    This is the table-level view of a multi-query pipeline.
    """

    tables: Dict[str, TableNode] = field(default_factory=dict)  # table_name -> TableNode
    queries: Dict[str, ParsedQuery] = field(default_factory=dict)  # query_id -> ParsedQuery

    def add_table(self, table_name: str, is_source: bool = False) -> TableNode:
        """Add a table to the graph"""
        if table_name not in self.tables:
            self.tables[table_name] = TableNode(table_name=table_name, is_source=is_source)
        return self.tables[table_name]

    def add_query(self, query: ParsedQuery):
        """Add a query and update table dependencies"""
        self.queries[query.query_id] = query

        # Register destination table
        if query.destination_table:
            table = self.add_table(query.destination_table, is_source=False)
            if query.is_ddl():
                table.created_by = query.query_id
                table.is_source = False  # Explicitly mark as not a source
            elif query.is_dml():
                table.modified_by.append(query.query_id)

        # Register source tables
        for source_table in query.source_tables:
            # Only add if not already in graph
            if source_table not in self.tables:
                table = self.add_table(source_table, is_source=True)
            else:
                table = self.tables[source_table]
            table.read_by.append(query.query_id)

    def _build_query_dependencies(self) -> Dict[str, Set[str]]:
        """
        Build dependency map: query_id -> set of query_ids it depends on.
        This is the core algorithm used by both topological_sort and get_execution_order.
        """
        deps = {}
        for query_id, query in self.queries.items():
            deps[query_id] = set()
            for source_table in query.source_tables:
                if source_table in self.tables:
                    table_node = self.tables[source_table]
                    # Depend on the query that creates this table
                    if table_node.created_by:
                        deps[query_id].add(table_node.created_by)
                    # Also depend on queries that modify it (if any)
                    deps[query_id].update(table_node.modified_by)
        return deps

    def _build_table_dependencies(self) -> Dict[str, Set[str]]:
        """
        Build dependency map: table_name -> set of table_names it depends on.
        This is the table-level equivalent of _build_query_dependencies.

        For each table, finds all upstream tables by looking at:
        - The query that creates this table (DDL: CREATE TABLE/VIEW)
        - Queries that modify this table (DML: INSERT, MERGE)

        Returns:
            Dict mapping table_name to set of upstream table_names
        """
        deps: Dict[str, Set[str]] = {}

        for table_name, table in self.tables.items():
            deps[table_name] = set()

            # Source tables have no dependencies
            if table.is_source and table.created_by is None:
                continue

            # Get dependencies from the query that creates this table (DDL)
            if table.created_by:
                query = self.queries.get(table.created_by)
                if query:
                    for source_table in query.source_tables:
                        if source_table in self.tables:
                            deps[table_name].add(source_table)

            # Also get dependencies from queries that modify this table (DML)
            for query_id in table.modified_by:
                query = self.queries.get(query_id)
                if query:
                    for source_table in query.source_tables:
                        if source_table in self.tables:
                            deps[table_name].add(source_table)

        return deps

    def topological_sort(self) -> List[str]:
        """
        Return query IDs in topological order (dependencies before dependents).
        Uses Python's graphlib.TopologicalSorter.
        """
        deps = self._build_query_dependencies()
        sorter = TopologicalSorter(deps)
        return list(sorter.static_order())

    def get_execution_order(self) -> List[TableNode]:
        """
        Return tables in execution order (dependencies before dependents).

        Returns:
            List of TableNode objects in topological order, with source tables first,
            followed by tables in the order they are created by queries.
        """
        result = []
        seen = set()

        # Add source tables first (no dependencies)
        for table in self.tables.values():
            if table.is_source and table.created_by is None:
                result.append(table)
                seen.add(table.table_name)

        # Add tables in query execution order
        for query_id in self.topological_sort():
            query = self.queries[query_id]
            if query.destination_table and query.destination_table not in seen:
                result.append(self.tables[query.destination_table])
                seen.add(query.destination_table)

        return result

    def get_source_tables(self) -> List[TableNode]:
        """Get all external source tables (not created by any query)"""
        return [t for t in self.tables.values() if t.is_source and t.created_by is None]

    def get_final_tables(self) -> List[TableNode]:
        """Get all final tables (not read by any query)"""
        return [t for t in self.tables.values() if len(t.read_by) == 0]

    def get_dependencies(self, table_name: str) -> List[TableNode]:
        """
        Get upstream tables that a table depends on.

        Args:
            table_name: Name of the table to get dependencies for

        Returns:
            List of TableNode objects that this table depends on (source tables
            used by queries that create or modify this table)
        """
        if table_name not in self.tables:
            return []

        table_deps = self._build_table_dependencies()
        upstream_names = table_deps.get(table_name, set())

        return [self.tables[name] for name in upstream_names if name in self.tables]

    def get_downstream(self, table_name: str) -> List[TableNode]:
        """
        Get downstream tables that depend on this table.

        Args:
            table_name: Name of the table to get downstream tables for

        Returns:
            List of TableNode objects that depend on this table (tables created
            by queries that read this table)
        """
        if table_name not in self.tables:
            return []

        # Build table dependencies and invert to find downstream
        table_deps = self._build_table_dependencies()

        # Find all tables that have table_name in their dependencies
        downstream = []
        for other_table, deps in table_deps.items():
            if table_name in deps:
                downstream.append(self.tables[other_table])

        return downstream


__all__ = [
    "TableNode",
    "TableDependencyGraph",
]
