"""
Pipeline validation component.

This module provides the PipelineValidator class which contains all validation
logic extracted from the Pipeline class.

The PipelineValidator collects and filters validation issues from:
- Individual query lineage graphs
- Pipeline-level lineage graph
"""

import logging
from typing import TYPE_CHECKING, List, Optional, Union

from .models import IssueCategory, IssueSeverity, ValidationIssue

if TYPE_CHECKING:
    from .pipeline import Pipeline

logger = logging.getLogger(__name__)


class PipelineValidator:
    """
    Validation logic for Pipeline.

    This class is extracted from Pipeline to follow the Single Responsibility
    Principle. It contains all validation methods that operate on the Pipeline's
    column graph and query graphs.

    The validator is lazily initialized by Pipeline when first needed.

    Example (via Pipeline - recommended):
        pipeline = Pipeline(queries, dialect="bigquery")
        issues = pipeline.get_all_issues()

    Example (direct usage - advanced):
        from clgraph.pipeline_validator import PipelineValidator

        validator = PipelineValidator(pipeline)
        issues = validator.get_all_issues()
    """

    def __init__(self, pipeline: "Pipeline"):
        """
        Initialize PipelineValidator with a Pipeline reference.

        Args:
            pipeline: The Pipeline instance to validate.
        """
        self._pipeline = pipeline

    def get_all_issues(self) -> List[ValidationIssue]:
        """
        Get all validation issues from all queries in the pipeline.

        Returns combined list of issues from:
        - Individual query lineage graphs
        - Pipeline-level lineage graph

        Returns:
            List of ValidationIssue objects
        """
        all_issues: List[ValidationIssue] = []

        # Collect issues from individual query lineage graphs
        for _query_id, query_lineage in self._pipeline.query_graphs.items():
            all_issues.extend(query_lineage.issues)

        # Add pipeline-level issues
        all_issues.extend(self._pipeline.column_graph.issues)

        return all_issues

    def get_issues(
        self,
        severity: Optional[Union[str, IssueSeverity]] = None,
        category: Optional[Union[str, IssueCategory]] = None,
        query_id: Optional[str] = None,
    ) -> List[ValidationIssue]:
        """
        Get filtered validation issues.

        Args:
            severity: Filter by severity ('error', 'warning', 'info' or IssueSeverity enum)
            category: Filter by category (string or IssueCategory enum)
            query_id: Filter by query ID

        Returns:
            Filtered list of ValidationIssue objects

        Example:
            # Get all errors (using string)
            errors = validator.get_issues(severity='error')

            # Get all errors (using enum)
            errors = validator.get_issues(severity=IssueSeverity.ERROR)

            # Get all star-related issues
            star_issues = validator.get_issues(
                category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES
            )

            # Get all issues from a specific query
            query_issues = validator.get_issues(query_id='query_1')
        """
        issues = self.get_all_issues()

        # Filter by severity
        if severity:
            severity_enum = (
                severity if isinstance(severity, IssueSeverity) else IssueSeverity(severity)
            )
            issues = [i for i in issues if i.severity == severity_enum]

        # Filter by category
        if category:
            category_enum = (
                category if isinstance(category, IssueCategory) else IssueCategory(category)
            )
            issues = [i for i in issues if i.category == category_enum]

        # Filter by query_id
        if query_id:
            issues = [i for i in issues if i.query_id == query_id]

        return issues

    def has_errors(self) -> bool:
        """Check if pipeline has any ERROR-level issues."""
        return any(i.severity.value == "error" for i in self.get_all_issues())

    def has_warnings(self) -> bool:
        """Check if pipeline has any WARNING-level issues."""
        return any(i.severity.value == "warning" for i in self.get_all_issues())

    def print_issues(self, severity: Optional[Union[str, IssueSeverity]] = None) -> None:
        """
        Print all validation issues in a human-readable format.

        Args:
            severity: Optional filter by severity ('error', 'warning', 'info'
                     or IssueSeverity enum)
        """
        from collections import defaultdict

        issues = self.get_issues(severity=severity) if severity else self.get_all_issues()

        if not issues:
            logger.info("No validation issues found")
            return

        # Group by severity
        by_severity = defaultdict(list)
        for issue in issues:
            by_severity[issue.severity.value].append(issue)

        # Log by severity (errors first, then warnings, then info)
        for sev in ["error", "warning", "info"]:
            if sev not in by_severity:
                continue

            issues_list = by_severity[sev]
            logger.info("%s (%d)", sev.upper(), len(issues_list))
            for issue in issues_list:
                logger.info("%s", issue)


__all__ = ["PipelineValidator"]
