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
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

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


@dataclass
class QueryUnit:
    """
    Represents a single SELECT statement in any context.
    This is the fundamental unit of lineage tracing.
    """

    unit_id: str  # Unique identifier (e.g., "main", "cte:monthly_sales", "subq:0")
    unit_type: QueryUnitType
    name: Optional[str]  # CTE name or alias
    select_node: exp.Select  # The actual SELECT AST node
    parent_unit: Optional["QueryUnit"]  # Parent query unit (None for main query)

    # Dependencies
    depends_on_units: List[str] = field(default_factory=list)  # Other QueryUnit IDs
    depends_on_tables: List[str] = field(default_factory=list)  # Base table names

    # Alias resolution: Maps alias -> (actual_name, is_unit)
    # e.g., {"b": ("base", True), "u": ("users", False)}
    alias_mapping: Dict[str, Tuple[str, bool]] = field(default_factory=dict)

    # Columns
    output_columns: List[Dict] = field(default_factory=list)  # What this unit produces

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
        if unit.unit_type == QueryUnitType.MAIN_QUERY:
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
    """Represents a column or column group at any layer in the query"""

    layer: str  # "input", "cte", "subquery", "output"
    table_name: str  # "users", "monthly_sales", etc.
    column_name: str  # "id", "total", "*"
    full_name: str  # "users.id", "table1.*", "output.total"
    expression: str  # Original SQL expression
    node_type: str  # "base_column", "star", "aggregate", "expression", etc.
    source_expression: Optional[exp.Expression] = None  # sqlglot AST node

    # Query unit association
    unit_id: Optional[str] = None  # ID of the QueryUnit that owns this column

    # Star-specific fields
    is_star: bool = False
    star_source_table: Optional[str] = None
    except_columns: Set[str] = field(default_factory=set)
    replace_columns: Dict[str, str] = field(default_factory=dict)

    # Metadata from SQL comments
    sql_metadata: Optional["ColumnMetadata"] = None  # Extracted from inline comments

    # Validation warnings
    warnings: List[str] = field(default_factory=list)  # Warnings about ambiguous lineage

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


@dataclass
class ColumnEdge:
    """Represents a dependency between columns"""

    from_node: ColumnNode
    to_node: ColumnNode
    transformation: str  # "direct", "aggregate", "star_passthrough", etc.
    context: str  # "SELECT", "CTE", "main_query", etc.
    expression: Optional[str] = None

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
