"""
SQL Lineage System - Main Entry Point

This module re-exports all components from the refactored modules for backward compatibility.

Architecture:
- models.py: Core dataclasses (QueryUnit, ColumnNode, ParsedQuery, etc.)
- query_parser.py: RecursiveQueryParser for query structure analysis
- lineage_builder.py: RecursiveLineageBuilder and SQLColumnTracer for column lineage
- multi_query.py: MultiQueryParser and TemplateTokenizer for pipeline parsing
- table.py: TableNode and TableDependencyGraph for table-level lineage
- column.py: PipelineColumnNode and PipelineColumnEdge for pipeline columns
- pipeline.py: Pipeline class for unified orchestration and lineage
"""

# Core models
from .column import PipelineColumnEdge, PipelineColumnNode
from .lineage_builder import RecursiveLineageBuilder, SQLColumnTracer
from .models import (
    ColumnEdge,
    ColumnLineageGraph,
    ColumnNode,
    DescriptionSource,
    ParsedQuery,
    QueryUnit,
    QueryUnitGraph,
    QueryUnitType,
    SQLOperation,
)

# Multi-query pipeline
from .multi_query import MultiQueryParser, TemplateTokenizer
from .pipeline import Pipeline, PipelineLineageBuilder

# Query parsing and lineage building
from .query_parser import RecursiveQueryParser
from .table import TableDependencyGraph, TableNode

__all__ = [
    # Query structure types
    "QueryUnit",
    "QueryUnitType",
    "QueryUnitGraph",
    "RecursiveQueryParser",
    # Lineage types
    "ColumnNode",
    "ColumnEdge",
    "ColumnLineageGraph",
    "RecursiveLineageBuilder",
    # High-level API
    "SQLColumnTracer",
    # Multi-query types
    "SQLOperation",
    "ParsedQuery",
    "TableNode",
    "TableDependencyGraph",
    "TemplateTokenizer",
    "MultiQueryParser",
    "Pipeline",
    "PipelineColumnNode",
    "PipelineColumnEdge",
    "PipelineLineageBuilder",
    # Metadata
    "DescriptionSource",
]
