"""
Pipeline diff functionality.

Contains PipelineDiff and ColumnDiff classes for detecting changes
between pipeline versions and enabling incremental updates.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from .pipeline import Pipeline


@dataclass
class ColumnDiff:
    """Represents a change to a single column field"""

    column_name: str
    table_name: str
    field_name: str  # What changed (e.g., "expression", "source_columns")
    old_value: Any
    new_value: Any

    @property
    def full_name(self) -> str:
        """Get fully qualified column name"""
        return f"{self.table_name}.{self.column_name}"


class PipelineDiff:
    """
    Represents differences between two pipeline versions.
    Focuses on SQL code changes (not metadata changes).
    """

    def __init__(self, new_graph: "Pipeline", old_graph: "Pipeline"):
        """
        Compare two pipelines and compute differences.

        Args:
            new_graph: The newer version of the pipeline
            old_graph: The older version of the pipeline
        """
        self.new_graph = new_graph
        self.old_graph = old_graph

        # Compute differences
        self.columns_added: List[str] = []
        self.columns_removed: List[str] = []
        self.columns_modified: List[ColumnDiff] = []

        self._compute_diff()

    def _compute_diff(self):
        """Compute the differences between old and new graphs"""
        old_columns = set(self.old_graph.columns.keys())
        new_columns = set(self.new_graph.columns.keys())

        # Detect additions and removals
        self.columns_added = list(new_columns - old_columns)
        self.columns_removed = list(old_columns - new_columns)

        # Detect modifications (columns that exist in both)
        common_columns = old_columns & new_columns

        for full_name in common_columns:
            old_col = self.old_graph.columns[full_name]
            new_col = self.new_graph.columns[full_name]

            # Check SQL expression changes
            if old_col.expression != new_col.expression:
                self.columns_modified.append(
                    ColumnDiff(
                        column_name=new_col.column_name,
                        table_name=new_col.table_name or "unknown",
                        field_name="expression",
                        old_value=old_col.expression,
                        new_value=new_col.expression,
                    )
                )

            # Check lineage changes (different source columns)
            old_sources = self._get_source_columns(self.old_graph, full_name)
            new_sources = self._get_source_columns(self.new_graph, full_name)

            if old_sources != new_sources:
                self.columns_modified.append(
                    ColumnDiff(
                        column_name=new_col.column_name,
                        table_name=new_col.table_name or "unknown",
                        field_name="source_columns",
                        old_value=sorted(old_sources),
                        new_value=sorted(new_sources),
                    )
                )

    def _get_source_columns(self, graph: "Pipeline", full_name: str) -> set:
        """Get set of source column full names for a given column"""
        if full_name not in graph.columns:
            return set()

        column = graph.columns[full_name]
        incoming = [e for e in graph.edges if e.to_node == column]

        return {e.from_node.full_name for e in incoming}

    def has_changes(self) -> bool:
        """Check if there are any changes"""
        return bool(self.columns_added or self.columns_removed or self.columns_modified)

    def summary(self) -> str:
        """Generate human-readable summary of changes"""
        lines = ["Pipeline Diff Summary:"]
        lines.append("=" * 50)

        if not self.has_changes():
            lines.append("No changes detected.")
            return "\n".join(lines)

        # Additions
        if self.columns_added:
            lines.append(f"\n✅ Columns Added: {len(self.columns_added)}")
            for full_name in self.columns_added[:5]:
                lines.append(f"   + {full_name}")
            if len(self.columns_added) > 5:
                lines.append(f"   ... and {len(self.columns_added) - 5} more")

        # Removals
        if self.columns_removed:
            lines.append(f"\n❌ Columns Removed: {len(self.columns_removed)}")
            for full_name in self.columns_removed[:5]:
                lines.append(f"   - {full_name}")
            if len(self.columns_removed) > 5:
                lines.append(f"   ... and {len(self.columns_removed) - 5} more")

        # Modifications
        if self.columns_modified:
            lines.append(f"\n🔄 Columns Modified: {len(self.columns_modified)}")

            # Group by change type
            sql_changes = [d for d in self.columns_modified if d.field_name == "expression"]
            lineage_changes = [d for d in self.columns_modified if d.field_name == "source_columns"]

            if sql_changes:
                lines.append(f"   SQL Expression Changes: {len(sql_changes)}")
                for diff in sql_changes[:3]:
                    lines.append(f"      • {diff.full_name}")
                if len(sql_changes) > 3:
                    lines.append(f"      ... and {len(sql_changes) - 3} more")

            if lineage_changes:
                lines.append(f"   Lineage Changes: {len(lineage_changes)}")
                for diff in lineage_changes[:3]:
                    lines.append(f"      • {diff.full_name}")
                if len(lineage_changes) > 3:
                    lines.append(f"      ... and {len(lineage_changes) - 3} more")

        # Columns needing update
        needing_update = self.get_columns_needing_update()
        lines.append(f"\n📊 Total Columns Needing Metadata Update: {len(needing_update)}")

        return "\n".join(lines)

    def get_sql_changes(self) -> List[ColumnDiff]:
        """Get only SQL expression changes"""
        return [d for d in self.columns_modified if d.field_name == "expression"]

    def get_lineage_changes(self) -> List[ColumnDiff]:
        """Get only lineage changes"""
        return [d for d in self.columns_modified if d.field_name == "source_columns"]

    def get_columns_needing_update(self) -> List[str]:
        """
        Get list of all columns that need metadata regeneration.

        Includes:
        - All added columns
        - All modified columns (SQL or lineage changes)

        Returns:
            List of column full names that need updates
        """
        needing_update = set(self.columns_added)

        # Add all modified columns
        for diff in self.columns_modified:
            needing_update.add(diff.full_name)

        return sorted(needing_update)


__all__ = [
    "ColumnDiff",
    "PipelineDiff",
]
