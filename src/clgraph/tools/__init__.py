"""
Lineage Tools - Operations on Pipeline data.

This module provides the core tools for interacting with lineage data.
Tools are the universal primitive - LineageAgent, MCP Server, and
direct Python API all use these same tools.

Tool Categories:
- Lineage: Trace column sources and impacts
- Schema: Explore tables and columns
- Governance: PII, ownership, tags
- SQL: Generate SQL from natural language (requires LLM)

Example:
    from clgraph import Pipeline
    from clgraph.tools import TraceBackwardTool, ListTablesTool

    pipeline = Pipeline.from_sql_files("queries/")

    # Use lineage tools
    trace = TraceBackwardTool(pipeline)
    result = trace.run(table="analytics.revenue", column="total")
    print(result.data)

    # Use schema tools
    tables = ListTablesTool(pipeline)
    result = tables.run()
    print(result.data)
"""

# Base classes
from .base import (
    BaseTool,
    LLMTool,
    ParameterSpec,
    ParameterType,
    ToolRegistry,
    ToolResult,
)

# Context builder
from .context import ContextBuilder, ContextConfig, TableInfo

# Governance tools
from .governance import (
    CheckDataQualityTool,
    FindPIIColumnsTool,
    GetColumnsByTagTool,
    GetOwnersTool,
    ListTagsTool,
)

# Lineage tools
from .lineage import (
    GetLineagePathTool,
    GetTableLineageTool,
    TraceBackwardTool,
    TraceForwardTool,
)

# Schema tools
from .schema import (
    GetExecutionOrderTool,
    GetRelationshipsTool,
    GetTableSchemaTool,
    ListTablesTool,
    SearchColumnsTool,
)

# SQL tools (require LLM)
from .sql import ExplainQueryTool, GenerateSQLTool

# All non-LLM tools
BASIC_TOOLS = [
    # Lineage
    TraceBackwardTool,
    TraceForwardTool,
    GetLineagePathTool,
    GetTableLineageTool,
    # Schema
    ListTablesTool,
    GetTableSchemaTool,
    GetRelationshipsTool,
    SearchColumnsTool,
    GetExecutionOrderTool,
    # Governance
    FindPIIColumnsTool,
    GetOwnersTool,
    GetColumnsByTagTool,
    ListTagsTool,
    CheckDataQualityTool,
]

# Tools that require LLM
LLM_TOOLS = [
    GenerateSQLTool,
    ExplainQueryTool,
]

# All tools
ALL_TOOLS = BASIC_TOOLS + LLM_TOOLS


def create_tool_registry(pipeline, llm=None) -> ToolRegistry:
    """
    Create a ToolRegistry with all available tools.

    Args:
        pipeline: The clgraph Pipeline
        llm: Optional LLM for SQL tools

    Returns:
        ToolRegistry with all tools registered

    Example:
        registry = create_tool_registry(pipeline, llm=my_llm)
        result = registry.run("trace_backward", table="X", column="Y")
    """
    registry = ToolRegistry(pipeline, llm)
    registry.register_all(ALL_TOOLS)
    return registry


__all__ = [
    # Base
    "BaseTool",
    "LLMTool",
    "ToolResult",
    "ToolRegistry",
    "ParameterSpec",
    "ParameterType",
    # Context
    "ContextBuilder",
    "ContextConfig",
    "TableInfo",
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
    # Registry factory
    "create_tool_registry",
    # Tool lists
    "BASIC_TOOLS",
    "LLM_TOOLS",
    "ALL_TOOLS",
]
