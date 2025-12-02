"""
SQL Lineage - Column lineage and pipeline dependency analysis for SQL

A powerful library for tracing data lineage through SQL queries and multi-query pipelines.
"""

from importlib.metadata import version

__version__ = version("clgraph")

# Import main public API from parser
# Import diff functionality
from .diff import ColumnDiff, PipelineDiff

# Import export functionality
from .export import CSVExporter, GraphVizExporter, JSONExporter
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

# Import visualization functions
from .visualizations import (
    visualize_column_lineage,
    visualize_column_lineage_simple,
    visualize_column_path,
    visualize_query_structure_from_lineage,
    visualize_query_units,
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
    # Export
    "JSONExporter",
    "CSVExporter",
    "GraphVizExporter",
    # Visualization functions
    "visualize_query_units",
    "visualize_query_structure_from_lineage",
    "visualize_column_lineage",
    "visualize_column_lineage_simple",
    "visualize_column_path",
]
