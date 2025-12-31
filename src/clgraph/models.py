"""
Core data models for SQL lineage system.

Contains all dataclass definitions for:
- Query structure models
- Column lineage models
- Multi-query pipeline models
- Metadata support models
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from sqlglot import exp

# Logger for validation issues
logger = logging.getLogger("clgraph.validation")

if TYPE_CHECKING:
    from .metadata_parser import ColumnMetadata

# ============================================================================
# Validation Models
# ============================================================================


class IssueSeverity(Enum):
    """Severity level for validation issues"""

    ERROR = "error"  # Prevents lineage construction or causes incorrect results
    WARNING = "warning"  # May cause issues or is a bad practice
    INFO = "info"  # Informational, suggestions for improvement


class IssueCategory(Enum):
    """Category of validation issue"""

    # Column reference issues
    AMBIGUOUS_COLUMN = "ambiguous_column"
    UNQUALIFIED_COLUMN = "unqualified_column"
    MISSING_COLUMN = "missing_column"

    # Star notation issues
    UNQUALIFIED_STAR_MULTIPLE_TABLES = "unqualified_star_multiple_tables"
    STAR_WITHOUT_SCHEMA = "star_without_schema"

    # Table reference issues
    MISSING_TABLE = "missing_table"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    AMBIGUOUS_TABLE = "ambiguous_table"

    # Expression issues
    UNSUPPORTED_SYNTAX = "unsupported_syntax"
    COMPLEX_EXPRESSION = "complex_expression"
    PARSE_ERROR = "parse_error"

    # Schema issues
    MISSING_SCHEMA_INFO = "missing_schema_info"
    SCHEMA_MISMATCH = "schema_mismatch"
    TYPE_MISMATCH = "type_mismatch"

    # Best practices
    IMPLICIT_JOIN = "implicit_join"
    SELECT_STAR_DISCOURAGED = "select_star_discouraged"


@dataclass
class ValidationIssue:
    """
    Represents a validation issue found during SQL parsing or lineage construction.

    Issues are collected during parsing and lineage building to provide users
    with actionable feedback on SQL quality and correctness.
    """

    severity: IssueSeverity
    category: IssueCategory
    message: str
    query_id: Optional[str] = None
    location: Optional[str] = None  # e.g., "line 5, column 10", "SELECT clause"
    suggestion: Optional[str] = None  # How to fix the issue
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context

    def __str__(self) -> str:
        """Human-readable string representation"""
        parts = [f"[{self.severity.value.upper()}]"]
        if self.query_id:
            parts.append(f"Query {self.query_id}")
        if self.location:
            parts.append(f"at {self.location}")
        parts.append(f"- {self.message}")
        if self.suggestion:
            parts.append(f"\n  💡 Suggestion: {self.suggestion}")
        return " ".join(parts)


# ============================================================================
# Query Structure Models
# ============================================================================


class QueryUnitType(Enum):
    """Type of query unit"""

    MAIN_QUERY = "main_query"
    CTE = "cte"
    CTE_BASE = "cte_base"  # Base/anchor case of a recursive CTE
    CTE_RECURSIVE = "cte_recursive"  # Recursive case of a recursive CTE
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

    # MERGE/UPSERT operations
    MERGE = "merge"  # MERGE INTO statement
    MERGE_SOURCE = "merge_source"  # Source subquery in MERGE


@dataclass
class RecursiveCTEInfo:
    """Information about a recursive CTE."""

    cte_name: str
    is_recursive: bool = True

    # Column info
    base_columns: List[str] = field(default_factory=list)  # Columns from base case
    recursive_columns: List[str] = field(default_factory=list)  # Columns from recursive case

    # Union type between base and recursive
    union_type: str = "union_all"  # "union" or "union_all"

    # Self-reference info
    self_reference_alias: Optional[str] = (
        None  # Alias used for self-ref (e.g., "h" in "JOIN cte h")
    )
    join_condition: Optional[str] = None  # How recursive joins to self

    # Recursion control
    max_recursion: Optional[int] = None  # MAXRECURSION hint if present

    def __repr__(self):
        return f"RecursiveCTEInfo({self.cte_name}, alias={self.self_reference_alias})"


@dataclass
class ValuesInfo:
    """Information about a VALUES clause (inline table literal)."""

    alias: str  # Table alias (e.g., "t" in "VALUES (...) AS t(id, name)")
    column_names: List[str] = field(default_factory=list)  # Column aliases
    row_count: int = 0  # Number of rows

    # Inferred column types
    column_types: List[str] = field(default_factory=list)  # e.g., ["integer", "string"]

    # Sample data for debugging/display (first few rows)
    sample_values: List[List[Any]] = field(default_factory=list)

    def __repr__(self) -> str:
        cols = ", ".join(self.column_names) if self.column_names else "..."
        return f"ValuesInfo({self.alias}({cols}), {self.row_count} rows)"


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

    # UNNEST/Array expansion sources in FROM clause
    # Maps alias -> UnnestInfo dict
    # Example: {'item': {'source_table': 'orders', 'source_column': 'items', 'offset_alias': None, 'expansion_type': 'unnest'}}
    unnest_sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # LATERAL subquery metadata
    is_lateral: bool = False  # True if this is a LATERAL subquery
    lateral_parent: Optional[str] = None  # unit_id of the preceding table being correlated to
    correlated_columns: List[str] = field(
        default_factory=list
    )  # Columns from outer scope (e.g., ["orders.order_id"])
    # Maps alias -> LateralInfo dict
    # Example: {'t': {'correlated_columns': ['orders.order_id'], 'preceding_tables': ['orders']}}
    lateral_sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Table-Valued Functions in FROM clause
    # Maps alias -> TVFInfo
    # Example: {'t': TVFInfo(function_name='generate_series', tvf_type=TVFType.GENERATOR, alias='t', ...)}
    tvf_sources: Dict[str, "TVFInfo"] = field(default_factory=dict)

    # QUALIFY clause metadata
    # Stores info about QUALIFY clause used for row filtering based on window functions
    # Example: {'condition': 'ROW_NUMBER() OVER (...) = 1',
    #           'partition_columns': ['customer_id'],
    #           'order_columns': ['order_date'],
    #           'window_functions': ['ROW_NUMBER']}
    qualify_info: Optional[Dict[str, Any]] = None

    # GROUPING SETS/CUBE/ROLLUP metadata
    # Stores info about complex grouping operations
    # Example: {'grouping_type': 'cube',
    #           'grouping_columns': ['region', 'product'],
    #           'grouping_sets': [['region', 'product'], ['region'], ['product'], []]}
    grouping_config: Optional[Dict[str, Any]] = None

    # Window function metadata
    # Stores info about window functions used in this query unit
    # Example: {'windows': [{'output_column': 'rolling_sum', 'function': 'SUM', 'arguments': ['amount'],
    #                        'partition_by': ['customer_id'], 'order_by': [{'column': 'date', 'direction': 'asc'}],
    #                        'frame_type': 'rows', 'frame_start': '6 preceding', 'frame_end': 'current row'}]}
    window_info: Optional[Dict[str, Any]] = None

    # Named window definitions from WINDOW clause
    # Example: {'w': {'partition_by': ['c'], 'order_by': [{'column': 'd', 'direction': 'asc'}]}}
    window_definitions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Recursive CTE metadata
    recursive_cte_info: Optional["RecursiveCTEInfo"] = None  # Info for recursive CTEs
    is_recursive_reference: bool = False  # True if this unit references a recursive CTE

    # VALUES clause in FROM clause
    # Maps alias -> ValuesInfo
    # Example: {'t': ValuesInfo(alias='t', column_names=['id', 'name'], row_count=2)}
    values_sources: Dict[str, "ValuesInfo"] = field(default_factory=dict)

    # Metadata
    depth: int = 0  # Nesting depth (0 = main query)
    order: int = 0  # Topological order for CTEs

    def __hash__(self):
        return hash(self.unit_id)

    def __eq__(self, other):
        if not isinstance(other, QueryUnit):
            return False
        return self.unit_id == other.unit_id

    def __repr__(self):
        return f"{self.unit_type.value}:{self.unit_id}"

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

    def __repr__(self):
        """Show topologically sorted query units"""
        sorted_units = self.get_topological_order()
        unit_reprs = [repr(unit) for unit in sorted_units]
        units_str = ", ".join(unit_reprs)
        return f"QueryUnitGraph([{units_str}])"


# ============================================================================
# Complex Aggregate Models
# ============================================================================


class AggregateType(Enum):
    """Type of aggregate function output."""

    SCALAR = "scalar"  # SUM, COUNT, AVG -> single value
    ARRAY = "array"  # ARRAY_AGG -> array
    STRING = "string"  # STRING_AGG -> concatenated string
    OBJECT = "object"  # OBJECT_AGG -> JSON/map
    STATISTICAL = "statistical"  # PERCENTILE -> computed value


@dataclass
class OrderByColumn:
    """Column specification in ORDER BY clause."""

    column: str
    direction: str = "asc"  # "asc" or "desc"
    nulls: Optional[str] = None  # "first" or "last"


@dataclass
class AggregateSpec:
    """Specification for an aggregate function."""

    function_name: str
    aggregate_type: AggregateType
    return_type: str = "any"  # "array<int>", "string", "object", etc.

    # Input columns
    value_columns: List[str] = field(default_factory=list)  # Column(s) being aggregated
    key_columns: List[str] = field(default_factory=list)  # For OBJECT_AGG key column

    # Modifiers
    distinct: bool = False
    order_by: List[OrderByColumn] = field(default_factory=list)
    limit: Optional[int] = None  # For ARRAY_AGG LIMIT

    # Parameters
    separator: Optional[str] = None  # For STRING_AGG
    null_handling: str = "exclude"  # "exclude", "include", "respect"

    def __repr__(self) -> str:
        parts = [self.function_name]
        if self.distinct:
            parts.append("DISTINCT")
        if self.value_columns:
            parts.append(f"({', '.join(self.value_columns)})")
        return " ".join(parts)


# ============================================================================
# Table-Valued Function Models
# ============================================================================


class TVFType(Enum):
    """Type of Table-Valued Function."""

    GENERATOR = "generator"  # No column input, generates data (GENERATE_SERIES)
    COLUMN_INPUT = "column_input"  # Takes column(s) as input (UNNEST, EXPLODE)
    EXTERNAL = "external"  # External data source (READ_CSV, EXTERNAL_QUERY)
    SYSTEM = "system"  # System/metadata tables (INFORMATION_SCHEMA)


@dataclass
class TVFInfo:
    """Information about a Table-Valued Function."""

    function_name: str
    tvf_type: TVFType
    alias: str
    output_columns: List[str] = field(default_factory=list)

    # For GENERATOR type - function parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"start": 1, "end": 10}

    # For COLUMN_INPUT type - input column references
    input_columns: List[str] = field(default_factory=list)
    # e.g., ["orders.items"]

    # For EXTERNAL type - external source location
    external_source: Optional[str] = None
    # e.g., "s3://bucket/file.csv"

    def __repr__(self) -> str:
        return f"TVFInfo({self.function_name} AS {self.alias})"


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

    # ─── TVF/Synthetic Column ───
    is_synthetic: bool = False  # True for TVF-generated columns
    synthetic_source: Optional[str] = None  # TVF name that created this column
    tvf_parameters: Dict[str, Any] = field(default_factory=dict)  # TVF parameters

    # ─── VALUES/Literal Column ───
    is_literal: bool = False  # True for VALUES-generated columns
    literal_values: Optional[List[Any]] = None  # Sample values from VALUES clause
    literal_type: Optional[str] = None  # Inferred type (e.g., "integer", "string")

    # ─── Validation ───
    warnings: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.full_name)

    def __eq__(self, other):
        if not isinstance(other, ColumnNode):
            return False
        return self.full_name == other.full_name

    def __repr__(self):
        return f"ColumnNode('{self.full_name}')"

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

    # ─── JSON Extraction Metadata ───
    json_path: Optional[str] = None  # Normalized JSON path (e.g., "$.address.city")
    json_function: Optional[str] = None  # Original function name (e.g., "JSON_EXTRACT")

    # ─── Array Expansion Metadata ───
    is_array_expansion: bool = False  # True if this edge is from UNNEST/FLATTEN/EXPLODE
    expansion_type: Optional[str] = None  # "unnest", "flatten", "explode"
    offset_column: Optional[str] = None  # Position column name if WITH OFFSET

    # ─── Nested Access Metadata ───
    nested_path: Optional[str] = None  # Normalized path like "[0].field" or "['key']"
    access_type: Optional[str] = None  # "array", "map", "struct", or "mixed"

    # ─── LATERAL Correlation Metadata ───
    is_lateral_correlation: bool = False  # True if this edge is a LATERAL correlation reference
    lateral_alias: Optional[str] = (
        None  # Alias of the LATERAL subquery (e.g., "t" in "LATERAL (...) t")
    )

    # ─── MERGE Statement Metadata ───
    is_merge_operation: bool = False  # True if this edge is from a MERGE statement
    merge_action: Optional[str] = None  # "match", "update", "insert", "delete"
    merge_condition: Optional[str] = None  # Condition for conditional WHEN clauses

    # ─── QUALIFY Clause Metadata ───
    is_qualify_column: bool = False  # True if this column is used in QUALIFY clause
    qualify_context: Optional[str] = None  # "partition", "order", or "filter"
    qualify_function: Optional[str] = None  # Window function name (e.g., "ROW_NUMBER")

    # ─── GROUPING SETS/CUBE/ROLLUP Metadata ───
    is_grouping_column: bool = False  # True if column is used in GROUPING SETS/CUBE/ROLLUP
    grouping_type: Optional[str] = None  # "cube", "rollup", "grouping_sets"

    # ─── Window Function Metadata ───
    is_window_function: bool = False  # True if this edge involves a window function
    window_role: Optional[str] = None  # "aggregate", "partition", "order"
    window_function: Optional[str] = None  # Function name (e.g., "SUM", "ROW_NUMBER", "LAG")
    window_frame_type: Optional[str] = None  # "rows", "range", "groups"
    window_frame_start: Optional[str] = None  # "unbounded preceding", "3 preceding", etc.
    window_frame_end: Optional[str] = None  # "current row", "1 following", etc.
    window_order_direction: Optional[str] = None  # "asc" or "desc" (for order edges)
    window_order_nulls: Optional[str] = None  # "first" or "last" (for order edges)

    # ─── Complex Aggregate Metadata ───
    aggregate_spec: Optional["AggregateSpec"] = None  # Full aggregate specification

    # ─── Table-Valued Function Metadata ───
    tvf_info: Optional["TVFInfo"] = None  # Full TVF specification
    is_tvf_output: bool = False  # True if this edge is from a TVF output

    def __hash__(self):
        return hash((self.from_node.full_name, self.to_node.full_name, self.edge_type))

    def __eq__(self, other):
        if not isinstance(other, ColumnEdge):
            return False
        return (
            self.from_node == other.from_node
            and self.to_node == other.to_node
            and self.edge_type == other.edge_type
        )

    def __repr__(self):
        return f"ColumnEdge({self.from_node.full_name!r} -> {self.to_node.full_name!r})"


@dataclass
class ColumnLineageGraph:
    """
    Complete column lineage graph for a SQL query.
    Contains nodes for all columns at all layers and edges showing dependencies.
    """

    nodes: Dict[str, ColumnNode] = field(default_factory=dict)  # full_name -> ColumnNode
    edges: List[ColumnEdge] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)  # Legacy validation warnings (deprecated)
    issues: List[ValidationIssue] = field(default_factory=list)  # Structured validation issues

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
        """Add a validation warning (deprecated - use add_issue instead)"""
        if warning not in self.warnings:
            self.warnings.append(warning)

    def add_issue(self, issue: ValidationIssue):
        """Add a structured validation issue and log it"""
        self.issues.append(issue)

        # Log the issue at appropriate level
        log_msg = f"[{issue.category.value}] {issue.message}"
        if issue.query_id:
            log_msg = f"Query '{issue.query_id}': {log_msg}"

        if issue.severity == IssueSeverity.ERROR:
            logger.error(log_msg)
        elif issue.severity == IssueSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

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

    def to_simplified(self) -> "ColumnLineageGraph":
        """
        Create a simplified version of the graph with only input and output nodes.

        This collapses all intermediate layers (CTEs, subqueries) and creates
        direct edges from input nodes to output nodes based on the original
        lineage relationships.

        Returns:
            A new ColumnLineageGraph with only input/output nodes and direct edges.
        """
        simplified = ColumnLineageGraph()

        # 1. Add only input and output nodes
        input_nodes = self.get_input_nodes()
        output_nodes = self.get_output_nodes()

        for node in input_nodes + output_nodes:
            simplified.add_node(node)

        # 2. Build adjacency list for backward traversal
        # Map: node.full_name -> list of upstream node.full_names
        upstream_map: Dict[str, List[str]] = {n: [] for n in self.nodes}
        for edge in self.edges:
            upstream_map[edge.to_node.full_name].append(edge.from_node.full_name)

        # 3. For each output node, find all input nodes that are upstream
        input_full_names = {n.full_name for n in input_nodes}

        for output_node in output_nodes:
            # BFS/DFS backward to find all reachable input nodes
            visited: Set[str] = set()
            queue = [output_node.full_name]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                # Check if this is an input node
                if current in input_full_names and current != output_node.full_name:
                    # Create direct edge from input to output
                    input_node = self.nodes[current]
                    edge = ColumnEdge(
                        from_node=input_node,
                        to_node=output_node,
                        edge_type="simplified",
                        transformation="direct",
                        context="simplified",
                    )
                    simplified.add_edge(edge)

                # Add upstream nodes to queue
                for upstream in upstream_map.get(current, []):
                    if upstream not in visited:
                        queue.append(upstream)

        # Copy warnings and issues
        simplified.warnings = self.warnings.copy()
        simplified.issues = self.issues.copy()

        return simplified

    def __repr__(self) -> str:
        query_units = sorted({n.unit_id for n in self.nodes.values() if n.unit_id})
        node_lines = sorted(
            (
                f"{node.full_name} (layer={node.layer}, type={node.node_type})"
                for node in self.nodes.values()
            ),
            key=lambda s: s,
        )
        edges = sorted(
            (
                f"{edge.from_node.full_name} -> {edge.to_node.full_name} ({edge.edge_type})"
                for edge in self.edges
            ),
            key=lambda s: s,
        )
        query_units_display = ", ".join(query_units)
        nodes_display = "\n  ".join(node_lines)
        edges_display = "\n  ".join(edges)
        return (
            "ColumnLineageGraph(\n"
            f"  query_units=[{query_units_display}]\n"
            "  nodes=[\n"
            f"  {nodes_display}\n"
            "  ]\n"
            "  edges=[\n"
            f"  {edges_display}\n"
            "  ]\n"
            ")"
        )


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
    "RecursiveCTEInfo",
    "ValuesInfo",
    # Column lineage
    "ColumnNode",
    "ColumnEdge",
    "ColumnLineageGraph",
    # Complex aggregates
    "AggregateType",
    "AggregateSpec",
    "OrderByColumn",
    # Table-valued functions
    "TVFType",
    "TVFInfo",
    # Multi-query pipeline
    "SQLOperation",
    "ParsedQuery",
    # Metadata
    "DescriptionSource",
]
