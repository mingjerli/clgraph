"""
Core data models for SQL lineage system.

Contains all dataclass definitions for:
- Query structure models
- Column lineage models
- Multi-query pipeline models
- Metadata support models
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from sqlglot import exp

if TYPE_CHECKING:
    from .metadata_parser import ColumnMetadata

# ============================================================================
# Query Structure Models
# ============================================================================


class QueryUnitType(Enum):
    """Type of query unit"""

    MAIN_QUERY = "main_query"
    CTE = "cte"
    SUBQUERY_FROM = "subquery_from"  # Subquery in FROM clause
    SUBQUERY_SELECT = "subquery_select"  # Scalar subquery in SELECT
    SUBQUERY_WHERE = "subquery_where"  # Subquery in WHERE
    SUBQUERY_HAVING = "subquery_having"  # Subquery in HAVING
    DERIVED_TABLE = "derived_table"  # Inline view

    # Set operations
    UNION = "union"  # UNION or UNION ALL operation
    INTERSECT = "intersect"  # INTERSECT operation
    EXCEPT = "except"  # EXCEPT operation
    SUBQUERY_UNION = "subquery_union"  # SELECT branch in a UNION/INTERSECT/EXCEPT

    # Table transformations
    PIVOT = "pivot"  # PIVOT operation
    UNPIVOT = "unpivot"  # UNPIVOT operation
    SUBQUERY_PIVOT_SOURCE = "subquery_pivot_source"  # Source query for PIVOT/UNPIVOT


@dataclass
class QueryUnit:
    """
    Represents a single query unit in any context.

    Can be a SELECT statement, or a set operation (UNION/INTERSECT/EXCEPT),
    or a table transformation (PIVOT/UNPIVOT).
    This is the fundamental unit of lineage tracing.
    """

    unit_id: str  # Unique identifier (e.g., "main", "cte:monthly_sales", "subq:0")
    unit_type: QueryUnitType
    name: Optional[str]  # CTE name or alias
    select_node: Optional[exp.Select]  # The actual SELECT AST node (None for set operations)
    parent_unit: Optional["QueryUnit"]  # Parent query unit (None for main query)

    # Dependencies
    depends_on_units: List[str] = field(default_factory=list)  # Other QueryUnit IDs
    depends_on_tables: List[str] = field(default_factory=list)  # Base table names

    # Alias resolution: Maps alias -> (actual_name, is_unit)
    # e.g., {"b": ("base", True), "u": ("users", False)}
    alias_mapping: Dict[str, Tuple[str, bool]] = field(default_factory=dict)

    # Columns
    output_columns: List[Dict] = field(default_factory=list)  # What this unit produces

    # Set operations (UNION, INTERSECT, EXCEPT)
    set_operation_type: Optional[str] = None  # "union", "union_all", "intersect", "except"
    set_operation_branches: List[str] = field(default_factory=list)  # unit_ids of branches

    # PIVOT operations
    pivot_config: Optional[Dict[str, Any]] = None  # Configuration for PIVOT
    # Example: {'pivot_column': 'quarter', 'aggregations': ['SUM(revenue)'], 'value_columns': ['Q1', 'Q2', 'Q3', 'Q4']}

    # UNPIVOT operations
    unpivot_config: Optional[Dict[str, Any]] = None  # Configuration for UNPIVOT
    # Example: {'value_column': 'revenue', 'unpivot_columns': ['q1', 'q2', 'q3', 'q4'], 'name_column': 'quarter'}

    # Metadata
    depth: int = 0  # Nesting depth (0 = main query)
    order: int = 0  # Topological order for CTEs

    def __hash__(self):
        return hash(self.unit_id)

    def __eq__(self, other):
        if not isinstance(other, QueryUnit):
            return False
        return self.unit_id == other.unit_id

    def is_leaf(self) -> bool:
        """Check if this is a leaf unit (only depends on base tables)"""
        return len(self.depends_on_units) == 0

    def get_all_source_tables(self) -> Set[str]:
        """Get all base tables this unit ultimately depends on"""
        return set(self.depends_on_tables)


@dataclass
class QueryUnitGraph:
    """
    Graph of all query units in the SQL statement.
    Built before column lineage to understand query structure.
    """

    units: Dict[str, QueryUnit] = field(default_factory=dict)  # unit_id -> QueryUnit
    main_unit_id: Optional[str] = None

    def add_unit(self, unit: QueryUnit):
        """Add a query unit to the graph"""
        self.units[unit.unit_id] = unit
        # Set operations can also be top-level units
        if unit.unit_type in (
            QueryUnitType.MAIN_QUERY,
            QueryUnitType.UNION,
            QueryUnitType.INTERSECT,
            QueryUnitType.EXCEPT,
        ):
            self.main_unit_id = unit.unit_id

    def get_topological_order(self) -> List[QueryUnit]:
        """Get units in dependency order (leaves first)"""
        from graphlib import TopologicalSorter

        # Build dependency graph
        deps = {unit_id: unit.depends_on_units for unit_id, unit in self.units.items()}

        # Topological sort
        ts = TopologicalSorter(deps)
        ordered_ids = list(ts.static_order())

        return [self.units[uid] for uid in ordered_ids]

    def get_unit_by_name(self, name: str) -> Optional[QueryUnit]:
        """Find a query unit by its name (for CTE lookups)"""
        for unit in self.units.values():
            if unit.name == name:
                return unit
        return None


# ============================================================================
# Column Lineage Models
# ============================================================================


@dataclass
class ColumnNode:
    """
    Unified column node for SQL lineage analysis.

    Supports both single-query and multi-query (pipeline) analysis.
    Context fields (query_id, unit_id) form a hierarchy:
        pipeline > query > unit (CTE/subquery) > table > column

    Works for:
    - Single query analysis (unit_id identifies CTE/subquery)
    - Multi-query pipeline analysis (query_id identifies the query)
    - Both combined (full hierarchy)
    """

    # ─── Core Identity ───
    column_name: str  # "id", "total", "*"
    table_name: str  # "users", "monthly_sales", etc.
    full_name: str  # "users.id", "table1.*", "output.total"

    # ─── Hierarchical Context ───
    query_id: Optional[str] = None  # Which query in pipeline (for multi-query)
    unit_id: Optional[str] = None  # Which CTE/subquery within query

    # ─── Classification ───
    node_type: str = "intermediate"  # "source", "intermediate", "output", "base_column", "star", "aggregate", "expression"
    layer: Optional[str] = None  # "input", "cte", "subquery", "output" (for backward compatibility)

    # ─── Expression ───
    expression: Optional[str] = None  # Original SQL expression
    operation: Optional[str] = None  # Operation type (e.g., "SUM", "CASE", "JOIN")
    source_expression: Optional[exp.Expression] = None  # sqlglot AST node

    # ─── Star Expansion (for SQL parsing) ───
    is_star: bool = False
    star_source_table: Optional[str] = None
    except_columns: Set[str] = field(default_factory=set)
    replace_columns: Dict[str, str] = field(default_factory=dict)

    # ─── Metadata & Documentation ───
    description: Optional[str] = None
    description_source: Optional["DescriptionSource"] = None
    sql_metadata: Optional["ColumnMetadata"] = None  # From SQL comments

    # ─── Governance ───
    owner: Optional[str] = None
    pii: bool = False
    tags: Set[str] = field(default_factory=set)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    # ─── Validation ───
    warnings: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.full_name)

    def __eq__(self, other):
        if not isinstance(other, ColumnNode):
            return False
        return self.full_name == other.full_name

    def get_display_name(self) -> str:
        """Get human-readable display name for UI"""
        if not self.is_star:
            return self.full_name

        # Build star notation with modifiers
        base = f"{self.table_name}.*"

        modifiers = []
        if self.except_columns:
            modifiers.append(f"EXCEPT({', '.join(sorted(self.except_columns))})")
        if self.replace_columns:
            replacements = [f"{col} AS {expr}" for col, expr in self.replace_columns.items()]
            modifiers.append(f"REPLACE({', '.join(replacements)})")

        if modifiers:
            return f"{base} {' '.join(modifiers)}"
        return base

    def is_computed(self) -> bool:
        """
        Check if this column is derived (not a true external source).

        A column is considered "computed" if it's created by a query,
        even if it's a direct pass-through.
        """
        if self.query_id:
            return True
        return self.node_type not in ["source", "base_column"]

    def set_source_description(self, description: str):
        """Set user-provided source description"""
        self.description = description
        self.description_source = DescriptionSource.SOURCE


@dataclass
class ColumnEdge:
    """
    Unified edge representing lineage between columns.
    Works at any level: within query, across queries, or both.
    """

    from_node: ColumnNode
    to_node: ColumnNode

    # ─── Classification ───
    edge_type: str = (
        "direct"  # "direct", "transform", "aggregate", "join", "star_passthrough", "cross_query"
    )

    # ─── Context ───
    query_id: Optional[str] = None  # Query where this edge exists
    context: Optional[str] = None  # "SELECT", "CTE", "main_query", "cross_query"

    # ─── Details ───
    transformation: Optional[str] = None  # Description of transformation
    expression: Optional[str] = None  # SQL expression

    def __hash__(self):
        return hash((self.from_node.full_name, self.to_node.full_name))

    def __eq__(self, other):
        if not isinstance(other, ColumnEdge):
            return False
        return self.from_node == other.from_node and self.to_node == other.to_node


@dataclass
class ColumnLineageGraph:
    """
    Complete column lineage graph for a SQL query.
    Contains nodes for all columns at all layers and edges showing dependencies.
    """

    nodes: Dict[str, ColumnNode] = field(default_factory=dict)  # full_name -> ColumnNode
    edges: List[ColumnEdge] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)  # Validation warnings

    def add_node(self, node: ColumnNode):
        """Add a column node to the graph"""
        self.nodes[node.full_name] = node

    def add_edge(self, edge: ColumnEdge):
        """Add an edge to the graph"""
        # Ensure both nodes exist
        if edge.from_node.full_name not in self.nodes:
            self.add_node(edge.from_node)
        if edge.to_node.full_name not in self.nodes:
            self.add_node(edge.to_node)

        # Add edge if not duplicate
        if edge not in self.edges:
            self.edges.append(edge)

    def add_warning(self, warning: str):
        """Add a validation warning"""
        if warning not in self.warnings:
            self.warnings.append(warning)

    def get_input_nodes(self) -> List[ColumnNode]:
        """Get all input layer nodes"""
        return [n for n in self.nodes.values() if n.layer == "input"]

    def get_output_nodes(self) -> List[ColumnNode]:
        """Get all output layer nodes"""
        return [n for n in self.nodes.values() if n.layer == "output"]

    def get_edges_from(self, node: ColumnNode) -> List[ColumnEdge]:
        """Get all edges originating from a node"""
        return [e for e in self.edges if e.from_node == node]

    def get_edges_to(self, node: ColumnNode) -> List[ColumnEdge]:
        """Get all edges pointing to a node"""
        return [e for e in self.edges if e.to_node == node]


# ============================================================================
# Multi-Query Pipeline Models
# ============================================================================


class SQLOperation(Enum):
    """
    Type of SQL operation.

    DDL (Data Definition Language): Define/modify schema
    DML (Data Manipulation Language): Modify data
    DQL (Data Query Language): Query data
    """

    # DDL Operations
    CREATE_TABLE = "CREATE TABLE"
    CREATE_OR_REPLACE_TABLE = "CREATE OR REPLACE TABLE"
    CREATE_VIEW = "CREATE VIEW"
    CREATE_OR_REPLACE_VIEW = "CREATE OR REPLACE VIEW"

    # DML Operations
    INSERT = "INSERT"
    MERGE = "MERGE"
    DELETE_AND_INSERT = "DELETE+INSERT"  # Common pattern
    UPDATE = "UPDATE"

    # DQL Operations
    SELECT = "SELECT"  # Query-only, no table creation/modification

    UNKNOWN = "UNKNOWN"


@dataclass
class ParsedQuery:
    """
    Represents a single SQL query with metadata about table dependencies.
    Extends single-query analysis to support multi-query pipelines.
    """

    query_id: str  # Unique identifier (e.g., "query_0", "file:pipeline/staging.sql")
    sql: str  # Original SQL text
    ast: exp.Expression  # Parsed sqlglot AST

    # Table dependencies
    operation: SQLOperation  # What kind of operation is this?
    destination_table: Optional[str]  # Table being created/modified (None for SELECT-only)
    source_tables: Set[str]  # Tables being read

    # Query-level lineage
    query_lineage: Optional["ColumnLineageGraph"] = None  # Single-query lineage graph

    # Metadata
    file_path: Optional[str] = None  # Source file (if from file)
    order: int = 0  # Topological order in pipeline

    # Template metadata
    original_sql: Optional[str] = None  # SQL before template resolution
    template_variables: Dict[str, str] = field(default_factory=dict)  # Variables used
    is_templated: bool = False  # Was this query templated?

    def is_ddl(self) -> bool:
        """Check if this is a DDL (Data Definition Language) operation"""
        return self.operation in [
            SQLOperation.CREATE_TABLE,
            SQLOperation.CREATE_OR_REPLACE_TABLE,
            SQLOperation.CREATE_VIEW,
            SQLOperation.CREATE_OR_REPLACE_VIEW,
        ]

    def is_dml(self) -> bool:
        """Check if this is a DML (Data Manipulation Language) operation"""
        return self.operation in [
            SQLOperation.INSERT,
            SQLOperation.MERGE,
            SQLOperation.DELETE_AND_INSERT,
            SQLOperation.UPDATE,
        ]

    def is_dql(self) -> bool:
        """Check if this is a DQL (Data Query Language) operation"""
        return self.operation == SQLOperation.SELECT

    def has_destination(self) -> bool:
        """Check if this query writes to a table (DDL or DML)"""
        return self.destination_table is not None


# ============================================================================
# Metadata Models
# ============================================================================


class DescriptionSource(Enum):
    """Source of column description"""

    SOURCE = "source"  # User-provided
    GENERATED = "generated"  # LLM-generated
    PROPAGATED = "propagated"  # Inherited from source


__all__ = [
    # Query structure
    "QueryUnitType",
    "QueryUnit",
    "QueryUnitGraph",
    # Column lineage
    "ColumnNode",
    "ColumnEdge",
    "ColumnLineageGraph",
    # Multi-query pipeline
    "SQLOperation",
    "ParsedQuery",
    # Metadata
    "DescriptionSource",
]
