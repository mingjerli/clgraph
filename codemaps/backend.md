# Backend Codemap

> Freshness: 2026-04-11 | Source: src/clgraph/ (22 modules, 18,205 LOC)

## Core Engine

### query_parser.py (2,354 LOC)
- `RecursiveQueryParser` - Parses SQL into QueryUnit graph
  - `.parse()` -> `QueryUnitGraph`
  - Handles: CTEs, subqueries, UNION/INTERSECT/EXCEPT, MERGE, PIVOT/UNPIVOT
  - Handles: Recursive CTEs, TVFs, VALUES, LATERAL, QUALIFY

### lineage_builder.py (3,419 LOC)
- `RecursiveLineageBuilder` - Traces column lineage through query units
  - `.build()` -> `ColumnLineageGraph`
  - Forward trace: expression -> source columns
  - Star expansion: SELECT *, EXCEPT, REPLACE
  - Special: JSON paths, window functions, aggregates, MERGE actions
- `SQLColumnTracer` - High-level wrapper (backward compat)

### multi_query.py (310 LOC)
- `TemplateTokenizer` - Handles `{{ var }}`, `{var}`, `{{ ref() }}`
- `MultiQueryParser` - Parses multiple SQL statements

## Pipeline Layer

### pipeline.py (2,795 LOC)
- `Pipeline` - Unified orchestration
  - `.from_sql_files()`, `.from_directory()`, `.from_database()`
  - Cross-query lineage, schema inference, metadata propagation
  - Topological sort for execution order
- `PipelineLineageBuilder` - Builds unified lineage graph

### column.py
- `PipelineLineageGraph` - Multi-query column lineage container

### table.py
- `TableNode` - Table with ownership, description, tags
- `TableDependencyGraph` - Table-level DAG

### execution.py (264 LOC)
- `PipelineExecutor` - Sync/async SQL execution with validation

## Intelligence Layer (tools/)

### tools/base.py
- `BaseTool` - Abstract tool with name, description, run()
- `LLMTool` - Tool requiring LLM (langchain)
- `ToolRegistry` - Tool discovery and execution

### tools/lineage.py
- `TraceBackwardTool`, `TraceForwardTool`, `GetLineagePathTool`, `GetTableLineageTool`

### tools/schema.py
- `ListTablesTool`, `GetTableSchemaTool`, `SearchColumnsTool`
- `GetRelationshipsTool`, `GetExecutionOrderTool`

### tools/governance.py
- `FindPIIColumnsTool`, `GetOwnersTool`, `GetColumnsByTagTool`, `ListTagsTool`

### tools/sql.py (LLM-powered)
- `GenerateSQLTool`, `ExplainQueryTool`

### tools/context.py
- `ContextBuilder`, `ContextConfig` - Build LLM context from lineage

## Agent Layer

### agent.py (608 LOC)
- `LineageAgent` - Natural language interface
  - `.query(question)` -> routes to tools via `QuestionType` classification
  - 11 question types: LINEAGE_*, SCHEMA_*, GOVERNANCE_*, SQL_*, GENERAL

## Integrations

### orchestrators/
- `BaseOrchestrator` (abstract) in base.py
- `AirflowOrchestrator` - Apache Airflow DAG generation (2.x/3.x)
- `DagsterOrchestrator` - Dagster asset graph generation
- `PrefectOrchestrator` - Prefect flow generation (2.x/3.x)
- `KestraOrchestrator` - Kestra YAML flow generation
- `MageOrchestrator` - Mage notebook pipeline generation

### mcp/server.py
- MCP server exposing lineage tools to Claude Desktop

## Support Modules

### metadata_parser.py
- `MetadataExtractor` - Parses `[pii: true, owner: team, tags: x]` from SQL comments

### diff.py
- `PipelineDiff`, `ColumnDiff` - Version comparison for lineage changes

### export.py
- `JSONExporter`, `CSVExporter` - Serialize lineage graphs

### visualizations.py
- `visualize_column_lineage()`, `visualize_table_dependencies()`
- `visualize_lineage_path()`, `visualize_query_structure_from_lineage()`
- Graphviz-based rendering
