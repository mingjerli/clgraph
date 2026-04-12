# Data Models Codemap

> Freshness: 2026-04-11 | Source: src/clgraph/models.py (935 LOC)

## Core Graph Nodes

### ColumnNode
Primary unit of column lineage.

```
ColumnNode
  identity:   column_name, table_name, full_name (unique key)
  context:    query_id?, unit_id?
  classify:   node_type (source|intermediate|output|aggregate), layer (input|cte|subquery|output)
  expression: expression?, operation?
  star:       is_star, except_columns: Set, replace_columns: Dict
  metadata:   description?, owner?, pii: bool, tags: Set[str]
  special:    is_synthetic (TVF), is_literal (VALUES)
```

### TableNode
Table-level representation with governance metadata.

```
TableNode
  identity:  table_name, schema?, database?
  metadata:  description?, owner?, tags: Set[str]
  stats:     column_count, row_count?
```

## Core Graph Edges

### ColumnEdge
Lineage relationship between two ColumnNodes.

```
ColumnEdge
  endpoints:    from_node: ColumnNode, to_node: ColumnNode
  classify:     edge_type (direct|transform|aggregate|join|cross_query)
  transform:    transformation?, expression?
  json:         json_path? ("$.address.city"), json_function?
  array:        is_array_expansion, expansion_type? (unnest|flatten|explode), offset_column?
  window:       is_window_function, window_function?, window_frame_{type,start,end}?
  merge:        is_merge_operation, merge_action? (match|update|insert)
  qualify:      is_qualify_column, qualify_function?
```

## Query Structure

### QueryUnit
Represents a single parseable SQL unit.

```
QueryUnit
  identity:    unit_id ("main"|"cte:name"|"subq:0"), unit_type: QueryUnitType, name?
  ast:         select_node: sqlglot.exp.Select?
  deps:        depends_on_units: List[str], depends_on_tables: List[str]
  aliases:     alias_mapping: Dict[str, Tuple[str, bool]]
  complex:     set_operation_type?, pivot_config?, unpivot_config?
  advanced:    window_info?, recursive_cte_info?, tvf_sources, values_sources, lateral_sources
```

### QueryUnitType (enum)
```
MAIN_QUERY | CTE | SUBQUERY_FROM | SUBQUERY_SELECT | SUBQUERY_WHERE
SUBQUERY_HAVING | UNION | INTERSECT | EXCEPT | PIVOT | UNPIVOT
MERGE | TVF | VALUES | LATERAL
```

## Composite Structures

### QueryUnitGraph
```
units: Dict[str, QueryUnit]     # All query units keyed by unit_id
root_unit_id: str               # Entry point unit
dependency_order: List[str]     # Topological sort
```

### ColumnLineageGraph
```
nodes: Dict[str, ColumnNode]    # All columns keyed by full_name
edges: List[ColumnEdge]         # All lineage edges
query_units: QueryUnitGraph     # Underlying query structure
validation_issues: List[ValidationIssue]
```

### Pipeline (composite)
```
dialect: str
table_graph: TableDependencyGraph
query_graphs: Dict[str, ColumnLineageGraph]   # Per-query lineage
columns: Dict[str, ColumnNode]                # Unified column index
edges: List[ColumnEdge]                       # All edges (including cross-query)
```

## Metadata Types

### ValidationIssue
```
level: "error" | "warning" | "info"
message: str
location?: str          # e.g., "cte:monthly_sales"
column?: str
```

### AggregateSpec
```
function: str           # "SUM", "ARRAY_AGG"
distinct: bool
order_by?: List[str]
separator?: str         # For STRING_AGG
```

### RecursiveCTEInfo
```
cte_name: str
base_case_unit_id: str
recursive_case_unit_id: str
anchor_columns: List[str]
```

### TVFInfo
```
function_name: str      # "GENERATE_SERIES", "UNNEST"
alias: str
output_columns: List[str]
```

## Metadata Inline Format
Parsed by MetadataExtractor from SQL comments:
```sql
-- Column description [pii: true, owner: data-team, tags: sensitive, metric]
SUM(amount) AS total /* [tags: finance] */
```
