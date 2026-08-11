# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `DescriptionSource.FALLBACK` - rule-based placeholder descriptions are now
  distinguishable from model output and retried on the next
  `generate_all_descriptions()` run. Exports emit `"fallback"`; older clgraph
  versions will not recognize this value when importing such exports.
- `generate_all_descriptions(include_sources=True)` also describes
  source-table columns from forward usage context. New public
  `build_source_description_prompt()`.
- Public `Pipeline.get_incoming_edges()`. Bulk description/metadata passes now
  use the adjacency index instead of linear edge scans.
- Text2sql prompts now include column lineage in the default direct strategy,
  table role labels (source/intermediate/final) with a prefer-final-tables
  instruction, and a `## Join Hints` section (observed equi-joins from
  pipeline SQL plus identity-preserving candidate joins).
- `ContextConfig` fields `max_lineage_columns_per_table`, `max_lineage_lines`,
  `annotate_table_roles`, `lineage_expansion_depth`, `max_join_hints`.

### Changed

- `generate_all_descriptions()` now also describes computed columns of
  queries without a destination table (terminal SELECTs) - reruns may issue
  more LLM calls than before.
- `expand_with_lineage()` walks ancestors transitively (default depth 2,
  configurable); two-stage text2sql context may include more tables.
- `build_schema_context()`/`resolve_context_tables()` - explicit table
  selections now preserve caller order and truncate to `max_tables`
  (previously oversized explicit lists were reordered and could exceed the
  cap).

## [0.0.7] - 2026-08-02

### Fixed

- **`import clgraph` failed on a clean install.** `clgraph.orchestrators.kestra`
  imported `yaml` at module scope and `clgraph/orchestrators/__init__.py` imports
  every backend eagerly, so `import clgraph` raised
  `ModuleNotFoundError: No module named 'yaml'` for anyone who installed clgraph
  without extras. PyYAML was never a declared dependency - it only ever arrived
  transitively in development environments, which is why the full test suite and
  CI stayed green. Regression introduced in 0.0.5 and also present in 0.0.6;
  0.0.3 was unaffected.

  PyYAML is now imported at point of use. `KestraOrchestrator` imports and
  constructs without it; only `to_flow()`, `to_flow_with_triggers()` and
  `to_flow_dict()` need it, and they raise an `ImportError` naming the package
  and how to install it. The other orchestrators (Airflow, Dagster, Prefect,
  Mage) emit code as text and were never affected.

### Added

- `clgraph[kestra]` extra, which installs PyYAML.
- CI job `bare-install`, which installs the built wheel into a clean environment
  with no extras and imports it. Every other job installs `.[dev]`, so nothing
  in the pipeline would have caught this class of bug.
- Regression tests in `tests/test_optional_orchestrator_deps.py` that run in a
  subprocess with `yaml` made unimportable, since the development environment
  has PyYAML installed and the failure only reproduces without it.

### Compatibility

No API changes. Anyone who already has PyYAML installed sees identical behavior.

## [0.0.6] - 2026-08-01

### Added

- `build_description_prompt(column, pipeline)` is now public API, exported from
  the package root. It builds clgraph's lineage-aware column-description prompt
  (including the column's SQL expression and upstream sources) so callers who
  want to drive the LLM themselves - to control error handling, batching, or
  model choice - can reuse it instead of reimplementing it. The former private
  name `_build_description_prompt` remains as an alias.
- `generate_description(...)` gained two keyword-only parameters:
  - `overwrite=False` - by default a description that came from a SQL comment
    (`description_source` is `SOURCE`) is left alone and the LLM is not called.
    Pass `True` to describe the column anyway, which is what you want when
    capturing a model's opinion *alongside* the authored text.
  - `on_error="fallback"` - `"raise"` raises the new `DescriptionGenerationError`
    instead of silently writing a rule-based description derived from the column
    name. Use it when a silent fallback would be mistaken for real model output.
- `generate_description(...)` now returns `bool`: `True` only when the LLM
  produced the stored description, `False` when the column was skipped or a
  fallback was written. Previously it returned `None`, so a caller could not
  tell a successful generation from a fallback.
- `DescriptionGenerationError`, raised by `on_error="raise"`.
- `Pipeline.generate_all_descriptions()` and
  `MetadataManager.generate_all_descriptions()` accept and forward the same
  `overwrite` and `on_error` keyword-only parameters.
- `generate_description` and `DescriptionGenerationError` are now exported from
  the package root alongside `build_description_prompt`.

### Fixed

- Callers had no way to distinguish "the LLM wrote this description" from "the
  LLM call failed and clgraph substituted the humanized column name" - both left
  `description_source` set to `GENERATED`. Any tool attributing the result to a
  model could therefore label a rule-based fallback, or a column's own
  hand-authored SQL comment, as model-generated output. The new return value and
  `on_error="raise"` make both cases detectable.

### Compatibility

No breaking changes. All new parameters are keyword-only with defaults that
preserve the previous behavior exactly, the new return value replaces `None`
(falsy either way), and the private prompt-builder name still resolves.

## [0.0.5] - 2026-07-31

### Security

- `Pipeline.from_sql_files()` and `Pipeline.from_json_file()` now validate paths:
  directory traversal, disallowed extensions, and symbolic links are rejected.
- `Pipeline.from_dbt_models()` now validates model-file paths (symlink/traversal
  rejection, TOCTOU-safe reads), consistent with `from_sql_files()`.
- LLM prompts (column descriptions, SQL generation, SQL explanation) now sanitize
  and delimit user-controlled content, separate instructions from data, and
  validate generated SQL against destructive operations.
- Table-level LLM descriptions (TableNode.generate_description) now sanitize and
  delimit content and validate output, consistent with column descriptions.

### Changed

- **BREAKING:** `Pipeline.from_sql_files()` and `Pipeline.from_json_file()` reject
  symlinked paths, glob patterns that escape the directory, and files whose
  extension is not `.sql`/`.json`. Pass `allow_symlinks=True` to opt back into
  following symbolic links.

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
