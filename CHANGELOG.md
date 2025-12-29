# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3] - 2025-12-29

### Added

#### AI/LLM Agent Integration
- **Agent module** (`clgraph.agent`): Build lineage-aware AI agents with LangChain
- **Tools module** (`clgraph.tools`): LangChain-compatible tools for lineage queries
  - Lineage tools: trace columns backward/forward, find paths
  - Governance tools: PII detection, impact analysis
  - SQL tools: query generation, validation
  - Schema tools: table/column lookup
  - Context tools: pipeline summary, metadata
- **MCP Server** (`clgraph.mcp`): Model Context Protocol server for Claude Desktop integration

#### Core Features
- `to_simplified()` method for input/output only lineage graph (filters internal CTEs/subqueries)
- `build_subpipeline()` convenience method for extracting sub-pipelines
- JSON round-trip serialization: `Pipeline.from_json()` and `Pipeline.from_json_file()`
- Template variable support in Pipeline class with `template_context` parameter
- Validation framework with structured issue reporting (`ValidationIssue`, `add_issue()`)
- Logging for validation issues at library level (logger: `clgraph.validation`)
- Enhanced validation for unqualified columns in JOIN conditions
- `COUNT(*)` resolution to individual columns when schema is known
- Star (`*`) expansion for cross-query column lineage with EXCEPT/REPLACE support
- API validation mode with auto-generated API dictionary
- `__repr__` methods for QueryUnit, QueryUnitGraph, and Pipeline for better debugging

#### Visualization
- Consolidated visualization functions into library (`clgraph.visualizations`)
- `visualize_pipeline_lineage()`, `visualize_table_dependencies()`, `visualize_column_lineage()`

#### Examples
- ClickHouse example with enterprise data pipeline (raw → staging → analytics → marts)
- Enterprise demo with Ollama for local LLM integration

### Changed
- **Breaking**: Renamed package from `clpipe` to `clgraph`
- **Breaking**: Removed `GraphVizExporter` class (use `visualize_*` functions from `clgraph.visualizations` instead)
- Renamed `query_lineages` to `query_graphs` for clarity
- Unified column naming for cross-pipeline lineage
- Removed redundant `save_metadata`, `load_metadata`, `apply_metadata` methods (use `to_json`/`from_json` instead)
- Filter redundant input star nodes from lineage graph
- Updated minimum sqlglot version to `>=28.0.0`

### Fixed
- Pin Airflow to 2.x for API stability (3.x has breaking changes)
- Handle sqlglot 28.x breaking change in EXCEPT/REPLACE key names
- Exclude star nodes from simplified lineage view
- SELECT queries without destination now treated as virtual result tables (`{query_id}_result`)
- Sanitize Graphviz node IDs to avoid colon port syntax issues
- Handle Schema objects in multi_query with version fallback
- Fix metadata propagation with two-pass approach

### Documentation
- Revamped README with user-focused messaging and updated examples
- Added architecture diagram
- Added illustration and expanded introduction with use cases
- Comprehensive docstrings and output examples in README

## [0.0.2] - 2025-12-02

### Changed
- Refactored version management to use single source of truth (pyproject.toml)
- Version now read dynamically from package metadata via `importlib.metadata.version()`

### Fixed
- CI/CD pipeline improvements for PyPI publishing workflow

## [0.0.1] - 2025-12-02

### Added

#### Core Features
- **Single Query Column Lineage**: Perfect column-level lineage tracking for any SQL query
  - Recursive query parsing with arbitrary CTE and subquery nesting
  - Bottom-up lineage building with dependency-ordered processing
  - Star notation preservation with EXCEPT/REPLACE support
  - Forward and backward lineage tracing capabilities

- **Multi-Query Pipeline Analysis**: Cross-query lineage tracking
  - Table dependency graph construction
  - Pipeline-level column lineage across multiple queries
  - Template variable support ({{variable}} syntax)
  - Pipeline-wide impact analysis

- **Metadata Management System**
  - Column metadata tracking (descriptions, ownership, PII flags, custom tags)
  - Automatic metadata propagation through lineage
  - Inline SQL comment parsing (`-- description [pii: true]`)
  - LLM integration for description generation (Ollama, OpenAI, etc.)
  - Pipeline diff tracking between versions

#### Export Functionality
- JSON export for machine-readable integration
- CSV export for spreadsheet analysis
- GraphViz DOT format export for visualization

#### Supported SQL Dialects
- BigQuery, PostgreSQL, MySQL, Snowflake, Redshift, DuckDB, and more via sqlglot

#### API Components
- `SQLColumnTracer` - Single query lineage analysis
- `Pipeline` - Multi-query pipeline analysis
- `MultiQueryParser` - Query parsing and table dependency resolution
- `PipelineLineageBuilder` - Cross-query lineage construction
- `RecursiveQueryParser` - Query structure parsing
- `RecursiveLineageBuilder` - Single query lineage building
- Export classes: `JSONExporter`, `CSVExporter`, `GraphVizExporter`
- Diff classes: `PipelineDiff`, `ColumnDiff`

#### Developer Experience
- Comprehensive test suite with 16 test modules
- Example scripts demonstrating all major features
- GitHub Actions CI/CD with testing, linting, and formatting
- Development tooling: pytest, ruff, mypy
- Git pre-commit hook installation script

### Dependencies
- sqlglot >= 20.0.0 (SQL parsing)
- graphviz >= 0.20.0 (visualization)
- jinja2 >= 3.0.0 (templating)
- langchain >= 1.0.5 (LLM integration)
- langchain-core >= 1.0.4
- langchain-ollama >= 1.0.0
- cloudpickle >= 3.1.2

### Documentation
- Comprehensive README with quickstart examples
- QUICKSTART.md for rapid onboarding
- CONTRIBUTING.md for contributor guidelines
- Detailed API documentation in code
- Multiple working examples in `/examples` directory

[0.0.3]: https://github.com/mingjerli/clgraph/releases/tag/v0.0.3
[0.0.2]: https://github.com/mingjerli/clgraph/releases/tag/v0.0.2
[0.0.1]: https://github.com/mingjerli/clgraph/releases/tag/v0.0.1
