"""
Governance and compliance tools.

Tools for finding PII columns, checking ownership, and tag-based discovery.
"""

from typing import Dict, List, Optional

from .base import BaseTool, ParameterSpec, ParameterType, ToolResult
from .context import ContextBuilder


class FindPIIColumnsTool(BaseTool):
    """
    Find all columns marked as PII (Personally Identifiable Information).

    Returns columns flagged with pii=True, optionally filtered by table.

    Example:
        tool = FindPIIColumnsTool(pipeline)
        result = tool.run()  # All PII columns
        result = tool.run(table="customers")  # PII in specific table
    """

    name = "find_pii_columns"
    description = "Find all columns flagged as containing PII"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "table": ParameterSpec(
                name="table",
                type=ParameterType.STRING,
                description="Optional: filter to specific table",
                required=False,
            ),
            "include_lineage": ParameterSpec(
                name="include_lineage",
                type=ParameterType.BOOLEAN,
                description="Whether to include columns that inherit PII through lineage",
                required=False,
                default=False,
            ),
        }

    def run(self, table: Optional[str] = None, include_lineage: bool = False) -> ToolResult:
        if table and table not in self.pipeline.table_graph.tables:
            return ToolResult.error_result(f"Table '{table}' not found")

        builder = ContextBuilder(self.pipeline)
        pii_columns = builder.get_pii_columns(table)

        # If include_lineage, trace forward from PII columns to find derived columns
        if include_lineage:
            derived_pii = set()
            for pii_col in pii_columns:
                impacts = self.pipeline.trace_column_forward(pii_col["table"], pii_col["column"])
                for impact in impacts:
                    key = f"{impact.table_name}.{impact.column_name}"
                    if key not in derived_pii:
                        derived_pii.add(key)
                        pii_columns.append(
                            {
                                "table": impact.table_name,
                                "column": impact.column_name,
                                "description": impact.description,
                                "owner": impact.owner,
                                "derived_from_pii": True,
                            }
                        )

        if not pii_columns:
            msg = "No PII columns found"
            if table:
                msg += f" in table '{table}'"
            return ToolResult.success_result(data=[], message=msg)

        # Group by table for summary
        by_table: Dict[str, List[str]] = {}
        for col in pii_columns:
            tbl = col["table"]
            if tbl not in by_table:
                by_table[tbl] = []
            by_table[tbl].append(col["column"])

        msg = f"Found {len(pii_columns)} PII columns across {len(by_table)} tables"

        return ToolResult.success_result(data=pii_columns, message=msg)


class GetOwnersTool(BaseTool):
    """
    Get ownership information for columns or tables.

    Returns owner assignments for data governance.

    Example:
        tool = GetOwnersTool(pipeline)
        result = tool.run()  # All owners
        result = tool.run(owner="data-team")  # Columns owned by specific team
    """

    name = "get_owners"
    description = "Get column and table ownership information"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "owner": ParameterSpec(
                name="owner",
                type=ParameterType.STRING,
                description="Optional: filter to specific owner",
                required=False,
            ),
            "table": ParameterSpec(
                name="table",
                type=ParameterType.STRING,
                description="Optional: filter to specific table",
                required=False,
            ),
        }

    def run(self, owner: Optional[str] = None, table: Optional[str] = None) -> ToolResult:
        if table and table not in self.pipeline.table_graph.tables:
            return ToolResult.error_result(f"Table '{table}' not found")

        results = []
        owner_counts: Dict[str, int] = {}

        for col in self.pipeline.columns.values():
            if col.owner is None:
                continue

            # Apply filters
            if owner and col.owner != owner:
                continue
            if table and col.table_name != table:
                continue

            results.append(
                {
                    "table": col.table_name,
                    "column": col.column_name,
                    "owner": col.owner,
                    "description": col.description,
                    "pii": col.pii,
                }
            )

            # Count by owner
            if col.owner not in owner_counts:
                owner_counts[col.owner] = 0
            owner_counts[col.owner] += 1

        if not results:
            msg = "No ownership information found"
            if owner:
                msg += f" for owner '{owner}'"
            if table:
                msg += f" in table '{table}'"
            return ToolResult.success_result(data=[], message=msg)

        # Sort by owner, then table
        results.sort(key=lambda x: (x["owner"], x["table"], x["column"]))

        msg = f"Found {len(results)} columns with ownership across {len(owner_counts)} owners"

        return ToolResult.success_result(
            data=results, message=msg, metadata={"owner_counts": owner_counts}
        )


class GetColumnsByTagTool(BaseTool):
    """
    Find columns with a specific tag.

    Returns all columns that have been tagged with the given tag.

    Example:
        tool = GetColumnsByTagTool(pipeline)
        result = tool.run(tag="financial")
        result = tool.run(tag="sensitive")
    """

    name = "get_columns_by_tag"
    description = "Find all columns with a specific tag"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "tag": ParameterSpec(
                name="tag",
                type=ParameterType.STRING,
                description="The tag to search for",
                required=True,
            ),
        }

    def run(self, tag: str) -> ToolResult:
        builder = ContextBuilder(self.pipeline)
        columns = builder.get_columns_by_tag(tag)

        if not columns:
            return ToolResult.success_result(data=[], message=f"No columns found with tag '{tag}'")

        # Group by table
        by_table: Dict[str, int] = {}
        for col in columns:
            tbl = col["table"]
            by_table[tbl] = by_table.get(tbl, 0) + 1

        msg = f"Found {len(columns)} columns with tag '{tag}' across {len(by_table)} tables"

        return ToolResult.success_result(data=columns, message=msg)


class ListTagsTool(BaseTool):
    """
    List all tags used in the pipeline.

    Returns unique tags with counts of how many columns use each.

    Example:
        tool = ListTagsTool(pipeline)
        result = tool.run()
    """

    name = "list_tags"
    description = "List all tags used across columns"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {}

    def run(self) -> ToolResult:
        tag_counts: Dict[str, int] = {}

        for col in self.pipeline.columns.values():
            for tag in col.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        if not tag_counts:
            return ToolResult.success_result(data=[], message="No tags found in pipeline")

        # Format as list sorted by count
        tags = [
            {"tag": tag, "count": count}
            for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])
        ]

        msg = f"Found {len(tags)} unique tags across {sum(tag_counts.values())} column-tag assignments"

        return ToolResult.success_result(data=tags, message=msg)


class CheckDataQualityTool(BaseTool):
    """
    Check data quality indicators in the pipeline.

    Reports on metadata completeness, PII coverage, ownership gaps.

    Example:
        tool = CheckDataQualityTool(pipeline)
        result = tool.run()
    """

    name = "check_data_quality"
    description = "Check metadata quality and completeness"

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return {
            "table": ParameterSpec(
                name="table",
                type=ParameterType.STRING,
                description="Optional: check specific table only",
                required=False,
            ),
        }

    def run(self, table: Optional[str] = None) -> ToolResult:
        if table and table not in self.pipeline.table_graph.tables:
            return ToolResult.error_result(f"Table '{table}' not found")

        # Count columns
        total_columns = 0
        with_description = 0
        with_owner = 0
        pii_flagged = 0
        with_tags = 0

        for col in self.pipeline.columns.values():
            if table and col.table_name != table:
                continue

            total_columns += 1
            if col.description:
                with_description += 1
            if col.owner:
                with_owner += 1
            if col.pii:
                pii_flagged += 1
            if col.tags:
                with_tags += 1

        # Count tables
        total_tables = 0
        tables_with_description = 0

        for tbl_name, tbl_node in self.pipeline.table_graph.tables.items():
            if table and tbl_name != table:
                continue

            total_tables += 1
            if tbl_node.description:
                tables_with_description += 1

        quality_report = {
            "total_columns": total_columns,
            "columns_with_description": with_description,
            "columns_with_owner": with_owner,
            "columns_with_pii_flag": pii_flagged,
            "columns_with_tags": with_tags,
            "total_tables": total_tables,
            "tables_with_description": tables_with_description,
            "description_coverage_pct": round(100 * with_description / total_columns, 1)
            if total_columns > 0
            else 0,
            "ownership_coverage_pct": round(100 * with_owner / total_columns, 1)
            if total_columns > 0
            else 0,
        }

        # Build recommendations
        recommendations = []
        if quality_report["description_coverage_pct"] < 50:
            recommendations.append(
                f"Low description coverage ({quality_report['description_coverage_pct']}%). "
                "Consider using generate_all_descriptions() to add descriptions."
            )
        if quality_report["ownership_coverage_pct"] < 20:
            recommendations.append("Low ownership coverage. Assign owners to important columns.")
        if pii_flagged == 0:
            recommendations.append(
                "No PII columns flagged. Review columns like email, name, phone for PII."
            )

        quality_report["recommendations"] = recommendations

        scope = f"table '{table}'" if table else "pipeline"
        msg = (
            f"Quality report for {scope}: "
            f"{quality_report['description_coverage_pct']}% descriptions, "
            f"{quality_report['ownership_coverage_pct']}% ownership"
        )

        return ToolResult.success_result(data=quality_report, message=msg)


__all__ = [
    "FindPIIColumnsTool",
    "GetOwnersTool",
    "GetColumnsByTagTool",
    "ListTagsTool",
    "CheckDataQualityTool",
]
