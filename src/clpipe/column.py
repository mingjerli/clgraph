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


__all__ = [
    "PipelineColumnNode",
    "PipelineColumnEdge",
]
