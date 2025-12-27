"""
Pipeline lineage graph and metadata operations.

Contains PipelineLineageGraph for multi-query column lineage,
plus utility functions for description generation and metadata propagation.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Set

from .models import (
    ColumnEdge,
    ColumnNode,
    DescriptionSource,
    IssueSeverity,
    ValidationIssue,
)

# Logger for validation issues
logger = logging.getLogger("clgraph.validation")

if TYPE_CHECKING:
    from .pipeline import Pipeline


# ============================================================================
# Description Generation Utilities
# ============================================================================


def generate_description(column: ColumnNode, llm: Any, pipeline: "Pipeline"):
    """
    Generate description using LLM based on SQL expression and source columns.

    Args:
        column: The column node to generate description for
        llm: LangChain LLM instance (BaseChatModel)
        pipeline: The pipeline for source lookup
    """
    # Don't overwrite source descriptions
    if column.description_source == DescriptionSource.SOURCE:
        return

    # Build prompt
    prompt = _build_description_prompt(column, pipeline)

    # Call LLM
    try:
        from langchain_core.prompts import ChatPromptTemplate

        template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a data documentation expert. Generate concise column descriptions.",
                ),
                ("user", prompt),
            ]
        )

        chain = template | llm
        response = chain.invoke({})

        column.description = response.content.strip()
        column.description_source = DescriptionSource.GENERATED
    except Exception:
        # Fallback to simple rule-based description
        _generate_fallback_description(column)


def _build_description_prompt(column: ColumnNode, pipeline: "Pipeline") -> str:
    """Build LLM prompt for description generation"""
    lines = [
        f"Column: {column.column_name}",
        f"Table: {column.table_name}",
        f"SQL: {column.expression or column.column_name}",
    ]

    # Add source column descriptions
    source_descs = []
    incoming_edges = [e for e in pipeline.edges if e.to_node == column]

    for edge in incoming_edges:
        source_col = edge.from_node
        if source_col.description:
            source_descs.append(f"- {source_col.full_name}: {source_col.description}")

    if source_descs:
        lines.append("")
        lines.append("Source columns:")
        lines.extend(source_descs)

    lines.extend(
        [
            "",
            "Generate a description that:",
            "- Is one sentence, max 15 words",
            "- Uses natural language (no SQL jargon)",
            "- Mentions sources if derived",
            "- Includes 'per X' for aggregations",
            "",
            "Return ONLY the description.",
        ]
    )

    return "\n".join(lines)


def _generate_fallback_description(column: ColumnNode):
    """Generate simple fallback description without LLM"""
    # Humanize column name
    words = column.column_name.replace("_", " ").split()
    base_desc = " ".join(word.capitalize() for word in words)

    column.description = base_desc
    column.description_source = DescriptionSource.GENERATED


def propagate_metadata_backward(column: ColumnNode, pipeline: "Pipeline"):
    """
    Propagate metadata backward from output columns to their input layer columns.

    This is the first pass of metadata propagation. When a SQL comment sets
    metadata (like PII) on an output column, we need to propagate it backward
    to the input layer columns so that other output columns derived from the
    same source will also inherit the metadata.

    Example:
        SELECT email, UPPER(email) as email_upper FROM users
        -- If email has [pii: true], we propagate PII to users.email
        -- Then email_upper will inherit PII from users.email in forward pass

    Args:
        column: The output column with metadata to propagate backward
        pipeline: The pipeline for source lookup
    """
    # Only propagate from output columns that have metadata
    if column.layer != "output":
        return

    has_metadata = column.pii or column.owner or column.tags

    if not has_metadata:
        return

    # Get source columns via incoming edges
    incoming_edges = [e for e in pipeline.edges if e.to_node == column]
    if not incoming_edges:
        return

    # Propagate to input layer sources
    for edge in incoming_edges:
        source = edge.from_node
        if source.layer == "input":
            # Propagate PII backward
            if column.pii:
                source.pii = True

            # Propagate owner backward (only if source doesn't have one)
            if column.owner and not source.owner:
                source.owner = column.owner

            # Propagate tags backward
            if column.tags:
                source.tags.update(column.tags)


def propagate_metadata(column: ColumnNode, pipeline: "Pipeline"):
    """
    Propagate metadata from source columns to this column (forward propagation).

    Propagation rules:
    - Owner: Only propagate if not already set and all sources have the same owner
    - PII: Union - if any source is PII, this column is PII
    - Tags: Union - merge all source tags
    - Custom metadata: Not automatically propagated

    Note: Metadata propagation is independent of description source.
    Even user-authored descriptions should inherit PII flags and other metadata.

    Args:
        column: The column to propagate metadata to
        pipeline: The pipeline for source lookup
    """
    # Don't propagate to source columns
    if not column.is_computed():
        return

    # Get source columns via incoming edges
    incoming_edges = [e for e in pipeline.edges if e.to_node == column]
    if not incoming_edges:
        return

    source_columns = [e.from_node for e in incoming_edges]

    # Propagate owner (only if not already set and all sources agree)
    if column.owner is None:
        owners = {col.owner for col in source_columns if col.owner is not None}
        if len(owners) == 1:
            column.owner = owners.pop()
        # If multiple different owners, don't propagate (keep as None)

    # Propagate PII (union - any source is PII)
    # Always propagate PII to ensure compliance
    if any(col.pii for col in source_columns):
        column.pii = True

    # Propagate tags (union of all source tags)
    for col in source_columns:
        column.tags.update(col.tags)


# ============================================================================
# Pipeline Lineage Graph
# ============================================================================


@dataclass
class PipelineLineageGraph:
    """
    DAG of columns and the edges that connect them across a multi-query pipeline.
    This is the column-level view of lineage across the pipeline.

    Similar to TableDependencyGraph but for column-level lineage:
    - columns (nodes): Individual columns at specific locations in the pipeline
    - edges: Lineage relationships between columns (direct, transform, aggregate, etc.)

    Uses unified ColumnNode and ColumnEdge from models.py.
    """

    columns: Dict[str, ColumnNode] = field(default_factory=dict)  # full_name -> ColumnNode
    edges: List[ColumnEdge] = field(default_factory=list)
    issues: List[ValidationIssue] = field(default_factory=list)  # Validation issues

    def add_column(self, column: ColumnNode) -> ColumnNode:
        """Add a column node to the graph"""
        self.columns[column.full_name] = column
        return column

    def add_edge(self, edge: ColumnEdge):
        """Add a lineage edge"""
        self.edges.append(edge)

    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue and log it"""
        self.issues.append(issue)

        # Log the issue at appropriate level
        log_msg = f"[{issue.category.value}] {issue.message}"
        if issue.query_id:
            log_msg = f"Query '{issue.query_id}': {log_msg}"

        if issue.severity == IssueSeverity.ERROR:
            logger.error(log_msg)
        elif issue.severity == IssueSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def _build_column_dependencies(self) -> Dict[str, Set[str]]:
        """
        Build dependency map: column_full_name -> set of column_full_names it depends on.
        This is the column-level equivalent of TableDependencyGraph._build_table_dependencies.

        Returns:
            Dict mapping column full_name to set of upstream column full_names
        """
        deps: Dict[str, Set[str]] = {}

        for full_name in self.columns:
            deps[full_name] = set()

        # Build dependencies from edges
        for edge in self.edges:
            to_name = edge.to_node.full_name
            from_name = edge.from_node.full_name
            if to_name in deps:
                deps[to_name].add(from_name)

        return deps

    def get_upstream(self, full_name: str) -> List[ColumnNode]:
        """
        Get upstream columns that a column depends on (direct dependencies only).

        Args:
            full_name: Full name of the column (e.g., "analytics.users.total_revenue")

        Returns:
            List of ColumnNode objects that this column directly depends on
        """
        if full_name not in self.columns:
            return []

        column_deps = self._build_column_dependencies()
        upstream_names = column_deps.get(full_name, set())

        return [self.columns[name] for name in upstream_names if name in self.columns]

    def get_downstream(self, full_name: str) -> List[ColumnNode]:
        """
        Get downstream columns that depend on this column (direct dependents only).

        Args:
            full_name: Full name of the column (e.g., "raw.orders.amount")

        Returns:
            List of ColumnNode objects that directly depend on this column
        """
        if full_name not in self.columns:
            return []

        # Build column dependencies and invert to find downstream
        column_deps = self._build_column_dependencies()

        # Find all columns that have full_name in their dependencies
        downstream = []
        for other_column, deps in column_deps.items():
            if full_name in deps:
                downstream.append(self.columns[other_column])

        return downstream

    def get_source_columns(self) -> List[ColumnNode]:
        """
        Get all source columns (columns with no incoming edges).

        Returns:
            List of ColumnNode objects that have no upstream dependencies
        """
        column_deps = self._build_column_dependencies()
        return [self.columns[name] for name, deps in column_deps.items() if len(deps) == 0]

    def get_final_columns(self) -> List[ColumnNode]:
        """
        Get all final columns (columns with no outgoing edges).

        Returns:
            List of ColumnNode objects that have no downstream dependents
        """
        # Find columns that appear as from_node in any edge
        has_outgoing = {edge.from_node.full_name for edge in self.edges}

        return [col for col in self.columns.values() if col.full_name not in has_outgoing]

    def to_simplified(self) -> "PipelineLineageGraph":
        """
        Create a simplified version of the graph with only physical table columns.

        This removes query-internal structures (CTEs, subqueries) and creates
        direct edges between physical table columns.

        - Keeps: All physical table columns (raw.*, staging.*, analytics.*, etc.)
        - Removes: CTE columns (query_id:cte:*), subquery columns (query_id:subq:*),
                   star nodes (table.*)
        - Edges: Traces through CTEs/subqueries to create direct table-to-table edges

        Returns:
            A new PipelineLineageGraph with only physical table columns and direct edges.
        """
        simplified = PipelineLineageGraph()

        # 1. Identify physical table columns (not CTEs, subqueries)
        # Physical table columns don't have ":" in their full_name (e.g., "staging.orders.amount")
        # CTE/subquery columns have format like "query_id:cte:name.column" or "query_id:subq:id.column"
        #
        # For star nodes (table.*): only filter out if we have explicit columns for that table.
        # Keep stars for external tables with unknown schema (e.g., COUNT(*) FROM external.table)
        physical_columns = {name: col for name, col in self.columns.items() if ":" not in name}

        # Find tables that have explicit (non-star) columns
        tables_with_explicit_cols = {
            col.table_name
            for col in physical_columns.values()
            if not col.is_star and col.table_name
        }

        # Filter out star nodes only if we have explicit columns for that table
        table_columns = {
            name: col
            for name, col in physical_columns.items()
            if not col.is_star or col.table_name not in tables_with_explicit_cols
        }

        # Add physical table columns
        for col in table_columns.values():
            simplified.add_column(col)

        # 2. Build adjacency list for backward traversal
        upstream_map: Dict[str, List[str]] = {name: [] for name in self.columns}
        for edge in self.edges:
            upstream_map[edge.to_node.full_name].append(edge.from_node.full_name)

        # 3. For each table column, find upstream table columns (tracing through CTEs)
        table_col_names = set(table_columns.keys())

        for col_name, col in table_columns.items():
            # BFS backward to find all reachable table columns
            visited: Set[str] = set()
            queue = [col_name]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                # Get upstream nodes
                for upstream in upstream_map.get(current, []):
                    if upstream in visited:
                        continue

                    # Check if upstream is a table column
                    if upstream in table_col_names and upstream != col_name:
                        # Create direct edge from upstream table column to this table column
                        upstream_col = self.columns[upstream]
                        edge = ColumnEdge(
                            from_node=upstream_col,
                            to_node=col,
                            edge_type="simplified",
                            transformation="direct",
                            context="simplified",
                        )
                        simplified.add_edge(edge)
                    else:
                        # Not a table column (CTE/subquery), continue traversal
                        queue.append(upstream)

        # Copy issues
        simplified.issues = self.issues.copy()

        return simplified


__all__ = [
    "PipelineLineageGraph",
    "generate_description",
    "propagate_metadata",
]
