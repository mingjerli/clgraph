"""
SQL Lineage - Column lineage and pipeline dependency analysis for SQL

A powerful library for tracing data lineage through SQL queries and multi-query pipelines.
"""

try:
    from importlib.metadata import version

    __version__ = version("clgraph")
except Exception:
    # Fallback when package metadata is not available (e.g., mounted as volume)
    __version__ = "dev"

# Import main public API from parser
# Import diff functionality
# Import lineage intelligence components
from .agent import AgentResult, LineageAgent, QuestionType
from .diff import ColumnDiff, PipelineDiff

# Import export functionality
from .export import CSVExporter, JSONExporter

# Import validation models
from .models import IssueCategory, IssueSeverity, ValidationIssue
from .parser import (
    ColumnEdge,
    ColumnLineageGraph,
    ColumnNode,
    DescriptionSource,
    # Multi-query pipeline lineage
    MultiQueryParser,
    # Pipeline structure
    ParsedQuery,
    Pipeline,
    PipelineLineageBuilder,
    PipelineLineageGraph,
    # Query structure
    QueryUnit,
    QueryUnitGraph,
    QueryUnitType,
    RecursiveLineageBuilder,
    RecursiveQueryParser,
    # Single query column lineage
    SQLColumnTracer,
    SQLOperation,
    TableDependencyGraph,
    TableNode,
    TemplateTokenizer,
)
from .tools import (
    ALL_TOOLS,
    BASIC_TOOLS,
    LLM_TOOLS,
    BaseTool,
    CheckDataQualityTool,
    ContextBuilder,
    ContextConfig,
    ExplainQueryTool,
    FindPIIColumnsTool,
    GenerateSQLTool,
    GetColumnsByTagTool,
    GetExecutionOrderTool,
    GetLineagePathTool,
    GetOwnersTool,
    GetRelationshipsTool,
    GetTableLineageTool,
    GetTableSchemaTool,
    ListTablesTool,
    ListTagsTool,
    LLMTool,
    ParameterSpec,
    ParameterType,
    SearchColumnsTool,
    ToolRegistry,
    ToolResult,
    TraceBackwardTool,
    TraceForwardTool,
    create_tool_registry,
)

# Import visualization functions
from .visualizations import (
    visualize_column_lineage,
    visualize_column_lineage_simple,
    visualize_column_path,
    visualize_lineage_path,
    visualize_pipeline_lineage,
    visualize_query_structure_from_lineage,
    visualize_query_units,
    visualize_table_dependencies,
    visualize_table_dependencies_with_levels,
)

__all__ = [
    # Version
    "__version__",
    # Main entry points
    "Pipeline",
    "SQLColumnTracer",
    "MultiQueryParser",
    "PipelineLineageBuilder",
    # Column lineage (unified types)
    "ColumnLineageGraph",
    "ColumnNode",
    "ColumnEdge",
    # Pipeline lineage
    "PipelineLineageGraph",
    # Query structure (advanced usage)
    "QueryUnit",
    "QueryUnitType",
    "QueryUnitGraph",
    "RecursiveQueryParser",
    "RecursiveLineageBuilder",
    # Pipeline structure (advanced usage)
    "ParsedQuery",
    "SQLOperation",
    "TableNode",
    "TableDependencyGraph",
    "TemplateTokenizer",
    # Metadata
    "DescriptionSource",
    "PipelineDiff",
    "ColumnDiff",
    # Validation
    "ValidationIssue",
    "IssueSeverity",
    "IssueCategory",
    # Export
    "JSONExporter",
    "CSVExporter",
    # Visualization functions
    "visualize_query_units",
    "visualize_query_structure_from_lineage",
    "visualize_column_lineage",
    "visualize_column_lineage_simple",
    "visualize_column_path",
    "visualize_lineage_path",
    "visualize_pipeline_lineage",
    "visualize_table_dependencies",
    "visualize_table_dependencies_with_levels",
    # Lineage Intelligence - Agent
    "LineageAgent",
    "AgentResult",
    "QuestionType",
    # Lineage Intelligence - Tools
    "BaseTool",
    "LLMTool",
    "ToolResult",
    "ToolRegistry",
    "ParameterSpec",
    "ParameterType",
    "ContextBuilder",
    "ContextConfig",
    "create_tool_registry",
    # Lineage tools
    "TraceBackwardTool",
    "TraceForwardTool",
    "GetLineagePathTool",
    "GetTableLineageTool",
    # Schema tools
    "ListTablesTool",
    "GetTableSchemaTool",
    "GetRelationshipsTool",
    "SearchColumnsTool",
    "GetExecutionOrderTool",
    # Governance tools
    "FindPIIColumnsTool",
    "GetOwnersTool",
    "GetColumnsByTagTool",
    "ListTagsTool",
    "CheckDataQualityTool",
    # SQL tools
    "GenerateSQLTool",
    "ExplainQueryTool",
    # Tool lists
    "BASIC_TOOLS",
    "LLM_TOOLS",
    "ALL_TOOLS",
]
