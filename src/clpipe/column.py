"""
Pipeline column models and metadata operations.

Contains PipelineColumnNode with description generation and metadata support.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from .models import DescriptionSource

if TYPE_CHECKING:
    from .pipeline import Pipeline


@dataclass
class PipelineColumnNode:
    """
    Represents a column at a specific location in the pipeline.

    Includes table and query context, metadata fields for description generation,
    and support for metadata propagation through lineage.
    """

    # Core identity
    column_name: str  # Column name (e.g., "amount", "total_revenue")
    table_name: Optional[str]  # Table name (e.g., "raw.orders", "analytics.user_metrics")
    query_id: Optional[str]  # Query that produces this column

    # Node type
    node_type: str  # "source", "intermediate", "output"

    # Lineage context
    full_name: str  # Fully qualified name (e.g., "raw.orders.amount")

    # Transformation info
    expression: Optional[str] = None  # SQL expression (if derived)
    operation: Optional[str] = None  # Operation type (e.g., "SUM", "CASE", "JOIN")

    # Description

    description: Optional[str] = None
    description_source: Optional[DescriptionSource] = None

    # Metadata
    owner: Optional[str] = None
    pii: bool = False
    tags: Set[str] = field(default_factory=set)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.full_name)

    def __eq__(self, other):
        if not isinstance(other, PipelineColumnNode):
            return False
        return self.full_name == other.full_name

    def set_source_description(self, description: str):
        """Set user-provided source description"""
        self.description = description
        self.description_source = DescriptionSource.SOURCE

    def is_computed(self) -> bool:
        """
        Check if this column is derived (not a true external source).

        A column is considered "computed" if it's created by a query,
        even if it's a direct pass-through (base_column).
        Only true external source columns (before any processing) are not computed.
        """
        # "source" means truly external data (not created by any query in pipeline)
        # "base_column" can be computed if it's in a derived table
        # For pipeline purposes, we check if this column has a query_id
        # If it has a query_id, it was created by a query, so it's computed
        if self.query_id:
            return True

        # Legacy behavior: node_type not in source/base_column
        return self.node_type not in ["source", "base_column"]

    def generate_description(self, llm: Any, pipeline: "Pipeline"):
        """
        Generate description using LLM based on SQL expression and source columns.

        Args:
            llm: LangChain LLM instance (BaseChatModel)
            pipeline: The pipeline for source lookup
        """
        # Don't overwrite source descriptions
        if self.description_source == DescriptionSource.SOURCE:
            return

        # Build prompt
        prompt = self._build_description_prompt(pipeline)

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

            self.description = response.content.strip()
            self.description_source = DescriptionSource.GENERATED
        except Exception:
            # Fallback to simple rule-based description
            self._generate_fallback_description()

    def _build_description_prompt(self, pipeline: "Pipeline") -> str:
        """Build LLM prompt for description generation"""
        lines = [
            f"Column: {self.column_name}",
            f"Table: {self.table_name}",
            f"SQL: {self.expression or self.column_name}",
        ]

        # Add source column descriptions
        source_descs = []
        incoming_edges = [e for e in pipeline.edges if e.to_column == self]

        for edge in incoming_edges:
            source_col = edge.from_column
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

    def _generate_fallback_description(self):
        """Generate simple fallback description without LLM"""
        # Humanize column name
        words = self.column_name.replace("_", " ").split()
        base_desc = " ".join(word.capitalize() for word in words)

        self.description = base_desc
        self.description_source = DescriptionSource.GENERATED

    def propagate_metadata(self, pipeline: "Pipeline"):
        """
        Propagate metadata from source columns to this column.

        Propagation rules:
        - Owner: Only propagate if not already set and all sources have the same owner
        - PII: Union - if any source is PII, this column is PII
        - Tags: Union - merge all source tags
        - Custom metadata: Not automatically propagated

        Note: Metadata propagation is independent of description source.
        Even user-authored descriptions should inherit PII flags and other metadata.

        Args:
            pipeline: The pipeline for source lookup
        """
        # Don't propagate to source columns
        if not self.is_computed():
            return

        # Get source columns via incoming edges
        incoming_edges = [e for e in pipeline.edges if e.to_column == self]
        if not incoming_edges:
            return

        source_columns = [e.from_column for e in incoming_edges]

        # Propagate owner (only if not already set and all sources agree)
        if self.owner is None:
            owners = {col.owner for col in source_columns if col.owner is not None}
            if len(owners) == 1:
                self.owner = owners.pop()
            # If multiple different owners, don't propagate (keep as None)

        # Propagate PII (union - any source is PII)
        # Always propagate PII to ensure compliance
        if any(col.pii for col in source_columns):
            self.pii = True

        # Propagate tags (union of all source tags)
        for col in source_columns:
            self.tags.update(col.tags)


@dataclass
class PipelineColumnEdge:
    """
    Represents a lineage relationship between two columns in a pipeline.
    Supports both intra-query and cross-query edges.
    """

    from_column: PipelineColumnNode
    to_column: PipelineColumnNode

    # Edge metadata
    edge_type: str  # "direct", "transform", "aggregate", "join", "cross_query"
    transformation: Optional[str] = None  # Description of transformation
    query_id: Optional[str] = None  # Query where this edge exists


@dataclass
class ColumnGraph:
    """
    DAG of columns and the edges that connect them.
    This is the column-level view of lineage across the pipeline.

    Similar to TableDependencyGraph but for column-level lineage:
    - columns (nodes): Individual columns at specific locations in the pipeline
    - edges: Lineage relationships between columns (direct, transform, aggregate, etc.)
    """

    columns: Dict[str, PipelineColumnNode] = field(default_factory=dict)  # full_name -> PipelineColumnNode
    edges: list = field(default_factory=list)  # List[PipelineColumnEdge]

    def add_column(self, column: PipelineColumnNode) -> PipelineColumnNode:
        """Add a column node to the graph"""
        self.columns[column.full_name] = column
        return column

    def add_edge(self, edge: PipelineColumnEdge):
        """Add a lineage edge"""
        self.edges.append(edge)

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
            to_name = edge.to_column.full_name
            from_name = edge.from_column.full_name
            if to_name in deps:
                deps[to_name].add(from_name)

        return deps

    def get_upstream(self, full_name: str) -> list:
        """
        Get upstream columns that a column depends on (direct dependencies only).

        Args:
            full_name: Full name of the column (e.g., "analytics.users.total_revenue")

        Returns:
            List of PipelineColumnNode objects that this column directly depends on
        """
        if full_name not in self.columns:
            return []

        column_deps = self._build_column_dependencies()
        upstream_names = column_deps.get(full_name, set())

        return [self.columns[name] for name in upstream_names if name in self.columns]

    def get_downstream(self, full_name: str) -> list:
        """
        Get downstream columns that depend on this column (direct dependents only).

        Args:
            full_name: Full name of the column (e.g., "raw.orders.amount")

        Returns:
            List of PipelineColumnNode objects that directly depend on this column
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

    def get_source_columns(self) -> list:
        """
        Get all source columns (columns with no incoming edges).

        Returns:
            List of PipelineColumnNode objects that have no upstream dependencies
        """
        column_deps = self._build_column_dependencies()
        return [
            self.columns[name]
            for name, deps in column_deps.items()
            if len(deps) == 0
        ]

    def get_final_columns(self) -> list:
        """
        Get all final columns (columns with no outgoing edges).

        Returns:
            List of PipelineColumnNode objects that have no downstream dependents
        """
        # Find columns that appear as from_column in any edge
        has_outgoing = {edge.from_column.full_name for edge in self.edges}

        return [
            col for col in self.columns.values()
            if col.full_name not in has_outgoing
        ]


__all__ = [
    "PipelineColumnNode",
    "PipelineColumnEdge",
    "ColumnGraph",
]
