"""
Pipeline orchestration with integrated lineage analysis.

Contains Pipeline class for unified SQL workflow orchestration with:
- Table and column lineage
- Metadata propagation
- LLM-powered documentation
- Pipeline execution (sync/async)
- Airflow DAG generation
"""

import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from .column import (
    PipelineLineageGraph,
)
from .models import (
    ColumnEdge,
    ColumnLineageGraph,
    ColumnNode,
    IssueCategory,
    IssueSeverity,
    ValidationIssue,
)
from .pipeline_lineage_builder import PipelineLineageBuilder
from .table import TableDependencyGraph

logger = logging.getLogger(__name__)


def _wrap_as_create_table(sql: str, target_table: str) -> str:
    """Wrap a SELECT-only statement with ``CREATE TABLE <target> AS``.

    Strips leading SQL line comments (``-- ...``) and block comments
    (``/* ... */``) before sniffing the first keyword, so dbt models that
    retain header comments after Jinja rendering are still detected.
    If ``sql`` already starts with a DDL/DML keyword, it is returned as-is
    and ``target_table`` is ignored.
    """
    stripped = sql.strip().rstrip(";").strip()
    body = stripped
    while True:
        if body.startswith("--"):
            nl = body.find("\n")
            body = body[nl + 1 :].lstrip() if nl >= 0 else ""
        elif body.startswith("/*"):
            end = body.find("*/")
            body = body[end + 2 :].lstrip() if end >= 0 else ""
        else:
            break

    head = body[:10].upper()
    if head.startswith("SELECT") or head.startswith("WITH"):
        return f"CREATE TABLE {target_table} AS {stripped}"
    return sql


class Pipeline:
    """
    Main pipeline class for SQL workflow orchestration with integrated lineage analysis.

    Provides:
    - Table and column lineage tracking
    - Metadata propagation (PII, owner, tags)
    - LLM-powered documentation generation
    - Pipeline execution (sync and async)
    - Airflow DAG generation (TaskFlow API)
    - Loading queries from SQL files or tuples

    Example:
        # Load from SQL files
        pipeline = Pipeline.from_sql_files("queries/", dialect="bigquery")

        # Or define inline
        queries = [
            ("staging", "CREATE TABLE staging AS SELECT * FROM raw"),
            ("final", "CREATE TABLE final AS SELECT * FROM staging"),
        ]
        pipeline = Pipeline(queries, dialect="bigquery")

        # Execute locally
        result = pipeline.run(executor=execute_sql)

        # Trace lineage
        sources = pipeline.trace_column_backward("final", "metric")

        # Propagate metadata
        pipeline.columns["raw.users.email"].pii = True
        pipeline.propagate_all_metadata()

        # Or generate Airflow DAG
        dag = pipeline.to_airflow_dag(executor=execute_sql, dag_id="my_pipeline")
    """

    def __init__(
        self,
        queries: "List[Union[Tuple[str, str], Tuple[str, str, str]]]",
        dialect: str = "bigquery",
        template_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Create pipeline from queries.

        Prefer using factory methods for clearer intent:
        - Pipeline.from_tuples() - from [(query_id, sql), ...]
        - Pipeline.from_dict() - from {query_id: sql, ...}
        - Pipeline.from_sql_list() - from [sql, ...] with auto-generated IDs
        - Pipeline.from_sql_string() - from semicolon-separated SQL string
        - Pipeline.from_sql_files() - from directory of .sql files

        Args:
            queries: List of (query_id, sql) tuples
            dialect: SQL dialect (bigquery, snowflake, etc.)
            template_context: Optional dictionary of template variables for Jinja2/variable substitution
                Example: {"env": "prod", "project": "my_project"}

        Example:
            # With template variables
            queries = [
                ("staging", "CREATE TABLE {{env}}_staging.orders AS SELECT * FROM {{env}}_raw.orders")
            ]
            pipeline = Pipeline(queries, dialect="bigquery", template_context={"env": "prod"})
        """
        from .multi_query import MultiQueryParser

        self.dialect = dialect
        self.template_context = template_context
        self.query_mapping: Dict[str, str] = {}  # Maps auto_id -> user_id

        # Column-level lineage graph
        self.column_graph: PipelineLineageGraph = PipelineLineageGraph()
        self.query_graphs: Dict[str, ColumnLineageGraph] = {}
        # Per-query unit graphs (populated during build), used for cross-query CTE edges
        self._unit_graphs: Dict[str, Any] = {}
        self.llm: Optional[Any] = None  # LangChain BaseChatModel

        # Lazy-initialized component instances
        self._tracer: Optional[Any] = None  # LineageTracer (lazy)
        self._validator: Optional[Any] = None  # PipelineValidator (lazy)
        self._metadata_mgr: Optional[Any] = None  # MetadataManager (lazy)
        self._subpipeline_builder: Optional[Any] = None  # SubpipelineBuilder (lazy)

        # Convert tuples to plain SQL strings for MultiQueryParser.
        # Tuples may be 2-tuples (id, sql) or 3-tuples (id, sql, target_table)
        # for dbt-style SELECT-only models.
        sql_list = []
        for query_tuple in queries:
            if len(query_tuple) == 3:
                user_query_id, sql, target_table = query_tuple
                sql = _wrap_as_create_table(sql, target_table)
            elif len(query_tuple) == 2:
                user_query_id, sql = query_tuple
            else:
                raise ValueError(
                    f"Pipeline query entries must be (id, sql) or (id, sql, target_table); "
                    f"got tuple of length {len(query_tuple)}"
                )
            sql_list.append(sql)
            auto_id = f"query_{len(sql_list) - 1}"
            self.query_mapping[auto_id] = user_query_id

        # Parse queries using current API with template support
        parser = MultiQueryParser(dialect=dialect)
        self.table_graph = parser.parse_queries(sql_list, template_context=template_context)

        # Remap auto-generated query IDs to user-provided IDs
        self._remap_query_ids()

        # Build lineage directly into this Pipeline instance
        builder = PipelineLineageBuilder()
        builder.build(self)

    @property
    def columns(self) -> Dict[str, ColumnNode]:
        """Access columns through column_graph for backward compatibility"""
        return self.column_graph.columns

    @property
    def edges(self) -> List[ColumnEdge]:
        """Access edges through column_graph for backward compatibility"""
        return self.column_graph.edges

    def _get_incoming_edges(self, full_name: str) -> List[ColumnEdge]:
        """Get incoming edges for a column using adjacency index."""
        return self.column_graph._incoming_index.get(full_name, [])

    def _get_outgoing_edges(self, full_name: str) -> List[ColumnEdge]:
        """Get outgoing edges for a column using adjacency index."""
        return self.column_graph._outgoing_index.get(full_name, [])

    def get_column(
        self, table_name: str, column_name: str, query_id: Optional[str] = None
    ) -> Optional[ColumnNode]:
        """
        Get a column by table and column name.

        Column keys now include query_id prefix (e.g., "query_1:table.column")
        for uniqueness. This method provides convenient lookup by table/column name.

        Args:
            table_name: The table name
            column_name: The column name
            query_id: Optional query_id to filter by

        Returns:
            The ColumnNode if found, None otherwise
        """
        for col in self.columns.values():
            if col.table_name == table_name and col.column_name == column_name:
                if query_id is None or col.query_id == query_id:
                    return col
        return None

    def get_columns_by_table(self, table_name: str) -> List[ColumnNode]:
        """
        Get all columns for a given table.

        Args:
            table_name: The table name to filter by

        Returns:
            List of ColumnNodes for the table
        """
        return [col for col in self.columns.values() if col.table_name == table_name]

    def get_simplified_column_graph(self) -> "PipelineLineageGraph":
        """
        Get a simplified version of the column lineage graph.

        This removes query-internal structures (CTEs, subqueries) and creates
        direct edges between physical table columns.

        - Keeps: All physical table columns (raw.*, staging.*, analytics.*, etc.)
        - Removes: CTE columns, subquery columns
        - Edges: Traces through CTEs/subqueries to create direct table-to-table edges

        Returns:
            A new PipelineLineageGraph with only physical table columns and direct edges.

        Example:
            pipeline = Pipeline(queries, dialect="bigquery")
            simplified = pipeline.get_simplified_column_graph()

            # Full graph has CTEs
            print(f"Full: {len(pipeline.columns)} columns")

            # Simplified has only table columns
            print(f"Simplified: {len(simplified.columns)} columns")
        """
        return self.column_graph.to_simplified()

    @classmethod
    def from_tuples(
        cls,
        queries: List[Tuple[str, str]],
        dialect: str = "bigquery",
        template_context: Optional[Dict[str, Any]] = None,
    ) -> "Pipeline":
        """
        Create pipeline from list of (query_id, sql) tuples.

        Args:
            queries: List of (query_id, sql) tuples
            dialect: SQL dialect (bigquery, snowflake, etc.)
            template_context: Optional dictionary of template variables

        Returns:
            Pipeline instance

        Example:
            pipeline = Pipeline.from_tuples([
                ("staging", "CREATE TABLE staging AS SELECT * FROM raw"),
                ("final", "CREATE TABLE final AS SELECT * FROM staging"),
            ])

            # With templates
            pipeline = Pipeline.from_tuples([
                ("staging", "CREATE TABLE {{env}}_staging.orders AS SELECT * FROM {{env}}_raw.orders"),
            ], template_context={"env": "prod"})
        """
        from .pipeline_factory import create_from_tuples

        return create_from_tuples(queries, dialect=dialect, template_context=template_context)

    @classmethod
    def from_dict(
        cls,
        queries: Dict[str, str],
        dialect: str = "bigquery",
        template_context: Optional[Dict[str, Any]] = None,
    ) -> "Pipeline":
        """
        Create pipeline from dictionary of {query_id: sql}.

        Args:
            queries: Dictionary mapping query_id to SQL string
            dialect: SQL dialect (bigquery, snowflake, etc.)
            template_context: Optional dictionary of template variables

        Returns:
            Pipeline instance

        Example:
            pipeline = Pipeline.from_dict({
                "staging": "CREATE TABLE staging AS SELECT * FROM raw",
                "final": "CREATE TABLE final AS SELECT * FROM staging",
            })

            # With templates
            pipeline = Pipeline.from_dict({
                "staging": "CREATE TABLE {{env}}_staging.orders AS SELECT * FROM raw.orders"
            }, template_context={"env": "prod"})
        """
        from .pipeline_factory import create_from_dict

        return create_from_dict(queries, dialect=dialect, template_context=template_context)

    @classmethod
    def from_sql_list(
        cls,
        queries: List[str],
        dialect: str = "bigquery",
        template_context: Optional[Dict[str, Any]] = None,
    ) -> "Pipeline":
        """
        Create pipeline from list of SQL strings (auto-generates query IDs).

        Query IDs are generated as: {operation}_{table_name}
        If duplicates exist, a number suffix is added: {operation}_{table_name}_2

        Args:
            queries: List of SQL query strings
            dialect: SQL dialect (bigquery, snowflake, etc.)
            template_context: Optional dictionary of template variables

        Returns:
            Pipeline instance

        Example:
            pipeline = Pipeline.from_sql_list([
                "CREATE TABLE staging AS SELECT * FROM raw",
                "INSERT INTO staging SELECT * FROM raw2",
                "CREATE TABLE final AS SELECT * FROM staging",
            ])
            # Query IDs will be: create_staging, insert_staging, create_final

            # With templates
            pipeline = Pipeline.from_sql_list([
                "CREATE TABLE {{env}}_staging.orders AS SELECT * FROM raw.orders"
            ], template_context={"env": "prod"})
        """
        from .pipeline_factory import create_from_sql_list

        return create_from_sql_list(queries, dialect=dialect, template_context=template_context)

    @staticmethod
    def _generate_query_id(sql: str, dialect: str, id_counts: Dict[str, int]) -> str:
        """
        Generate a meaningful query ID from SQL statement.

        Format priority:
        1. {operation}_{dest_table}
        2. {operation}_{dest_table}_from_{source_table} (if duplicate)
        3. {operation}_{dest_table}_from_{source_table}_2 (if still duplicate)

        Args:
            sql: SQL query string
            dialect: SQL dialect
            id_counts: Dictionary tracking ID usage counts

        Returns:
            Generated query ID
        """
        from .pipeline_factory import generate_query_id

        return generate_query_id(sql, dialect, id_counts)

    @classmethod
    def from_sql_string(
        cls,
        sql: str,
        dialect: str = "bigquery",
        template_context: Optional[Dict[str, Any]] = None,
    ) -> "Pipeline":
        """
        Create pipeline from single SQL string with semicolon-separated queries.

        Query IDs are generated as: {operation}_{table_name}
        If duplicates exist, a number suffix is added: {operation}_{table_name}_2

        Args:
            sql: SQL string with multiple queries separated by semicolons
            dialect: SQL dialect (bigquery, snowflake, etc.)
            template_context: Optional dictionary of template variables

        Returns:
            Pipeline instance

        Example:
            pipeline = Pipeline.from_sql_string('''
                CREATE TABLE staging AS SELECT * FROM raw;
                CREATE TABLE final AS SELECT * FROM staging
            ''')
            # Query IDs will be: create_staging, create_final

            # With templates
            pipeline = Pipeline.from_sql_string(
                "CREATE TABLE {{env}}_staging.orders AS SELECT * FROM raw.orders",
                template_context={"env": "prod"}
            )
        """
        from .pipeline_factory import create_from_sql_string

        return create_from_sql_string(sql, dialect=dialect, template_context=template_context)

    @classmethod
    def from_json(
        cls,
        data: Dict[str, Any],
        apply_metadata: bool = True,
    ) -> "Pipeline":
        """
        Create pipeline from JSON data exported by JSONExporter.

        This enables round-trip serialization: export a pipeline to JSON,
        store/transfer it, and recreate the same pipeline later.

        Args:
            data: JSON dictionary from JSONExporter.export() or Pipeline.to_json()
            apply_metadata: Whether to apply metadata (descriptions, PII, etc.)
                from the JSON to the reconstructed pipeline

        Returns:
            Pipeline instance

        Example:
            # Save pipeline to file
            data = pipeline.to_json()
            with open("pipeline.json", "w") as f:
                json.dump(data, f)

            # Later, reload the pipeline
            with open("pipeline.json") as f:
                data = json.load(f)
            pipeline = Pipeline.from_json(data)

            # Verify round-trip
            assert len(pipeline.columns) > 0
            assert len(pipeline.edges) > 0
        """
        from .pipeline_factory import create_from_json

        return create_from_json(data, apply_metadata=apply_metadata)

    @classmethod
    def from_json_file(cls, file_path: str, apply_metadata: bool = True) -> "Pipeline":
        """
        Create pipeline from JSON file exported by JSONExporter.

        Args:
            file_path: Path to JSON file
            apply_metadata: Whether to apply metadata from the JSON

        Returns:
            Pipeline instance

        Example:
            # Export pipeline
            JSONExporter.export_to_file(pipeline, "pipeline.json")

            # Later, reload it
            pipeline = Pipeline.from_json_file("pipeline.json")
        """
        from .pipeline_factory import create_from_json_file

        return create_from_json_file(file_path, apply_metadata=apply_metadata)

    @classmethod
    def _create_empty(cls, table_graph: "TableDependencyGraph") -> "Pipeline":
        """
        Create an empty Pipeline with just a table_graph (for testing).

        This bypasses SQL parsing and creates a minimal Pipeline that can be
        populated manually with columns and edges.

        Args:
            table_graph: Pre-built table dependency graph

        Returns:
            Empty Pipeline instance
        """
        from .pipeline_factory import create_empty

        return create_empty(table_graph)

    # === Lineage methods (from PipelineLineageGraph) ===

    def add_column(self, column: ColumnNode) -> ColumnNode:
        """Add a column node to the graph"""
        self.column_graph.add_column(column)

        # Also add to table's columns set if table exists
        if column.table_name and column.table_name in self.table_graph.tables:
            self.table_graph.tables[column.table_name].columns.add(column.column_name)

        return column

    def add_edge(self, edge: ColumnEdge):
        """Add a lineage edge"""
        self.column_graph.add_edge(edge)

    # === Lazy-initialized components ===

    @property
    def _lineage_tracer(self):
        """Lazily initialize and return the LineageTracer component."""
        if self._tracer is None:
            from .lineage_tracer import LineageTracer

            self._tracer = LineageTracer(self)
        return self._tracer

    @property
    def _pipeline_validator(self):
        """Lazily initialize and return the PipelineValidator component."""
        if self._validator is None:
            from .pipeline_validator import PipelineValidator

            self._validator = PipelineValidator(self)
        return self._validator

    @property
    def _metadata_manager(self):
        """Lazily initialize and return the MetadataManager component."""
        if self._metadata_mgr is None:
            from .metadata_manager import MetadataManager

            self._metadata_mgr = MetadataManager(self)
        return self._metadata_mgr

    @property
    def _subpipeline_builder_component(self):
        """Lazily initialize and return the SubpipelineBuilder component."""
        if self._subpipeline_builder is None:
            from .subpipeline_builder import SubpipelineBuilder

            self._subpipeline_builder = SubpipelineBuilder(self)
        return self._subpipeline_builder

    # === Lineage methods (delegate to LineageTracer) ===

    def trace_column_backward(self, table_name: str, column_name: str) -> List[ColumnNode]:
        """
        Trace a column backward to its ultimate sources.
        Returns list of source columns across all queries.

        For full lineage path with all intermediate nodes, use trace_column_backward_full().
        """
        from .lineage_tracer import trace_backward

        return trace_backward(
            self.columns,
            self.column_graph._incoming_index,
            table_name,
            column_name,
        )

    def trace_column_backward_full(
        self, table_name: str, column_name: str, include_ctes: bool = True
    ) -> Tuple[List[ColumnNode], List[ColumnEdge]]:
        """
        Trace a column backward with full transparency.

        Returns complete lineage path including all intermediate tables and CTEs.

        Args:
            table_name: The table containing the column to trace
            column_name: The column name to trace
            include_ctes: If True, include CTE columns; if False, only real tables

        Returns:
            Tuple of (nodes, edges) representing the complete lineage path.
            - nodes: All columns in the lineage, in BFS order from target to sources
            - edges: All edges connecting the columns

        Example:
            nodes, edges = pipeline.trace_column_backward_full("mart_customer_ltv", "lifetime_revenue")

            # Print the lineage path:
            for node in nodes:
                print(f"{node.table_name}.{node.column_name} (query={node.query_id})")

            # Print the edges:
            for edge in edges:
                print(f"{edge.from_node.table_name}.{edge.from_node.column_name} -> "
                      f"{edge.to_node.table_name}.{edge.to_node.column_name}")
        """
        from .lineage_tracer import trace_backward_full

        return trace_backward_full(
            self.columns,
            self.column_graph._incoming_index,
            table_name,
            column_name,
            include_ctes,
        )

    def get_table_lineage_path(
        self, table_name: str, column_name: str
    ) -> List[Tuple[str, str, str]]:
        """
        Get simplified table-level lineage path for a column.

        Returns list of (table_name, column_name, query_id) tuples representing
        the lineage through real tables only (skipping CTEs).

        This provides a clear view of how data flows between tables in your pipeline.

        Example:
            path = pipeline.get_table_lineage_path("mart_customer_ltv", "lifetime_revenue")
            # Returns:
            # [
            #   ("mart_customer_ltv", "lifetime_revenue", "07_mart_customer_ltv"),
            #   ("stg_orders_enriched", "total_amount", "05_stg_orders_enriched"),
            #   ("raw_orders", "total_amount", "01_raw_orders"),
            #   ("source_orders", "total_amount", "01_raw_orders"),
            # ]
        """
        from .lineage_tracer import get_table_lineage_path

        return get_table_lineage_path(
            self.columns,
            self.column_graph._incoming_index,
            table_name,
            column_name,
            self.table_graph.tables,
        )

    def trace_column_forward(self, table_name: str, column_name: str) -> List[ColumnNode]:
        """
        Trace a column forward to see what depends on it.
        Returns list of final downstream columns across all queries.

        For full impact path with all intermediate nodes, use trace_column_forward_full().
        """
        from .lineage_tracer import trace_forward

        return trace_forward(
            self.columns,
            self.column_graph._outgoing_index,
            table_name,
            column_name,
        )

    def trace_column_forward_full(
        self, table_name: str, column_name: str, include_ctes: bool = True
    ) -> Tuple[List[ColumnNode], List[ColumnEdge]]:
        """
        Trace a column forward with full transparency.

        Returns complete impact path including all intermediate tables and CTEs.

        Args:
            table_name: The table containing the column to trace
            column_name: The column name to trace
            include_ctes: If True, include CTE columns; if False, only real tables

        Returns:
            Tuple of (nodes, edges) representing the complete impact path.
            - nodes: All columns impacted, in BFS order from source to finals
            - edges: All edges connecting the columns

        Example:
            nodes, edges = pipeline.trace_column_forward_full("raw_orders", "total_amount")

            # Print the impact path:
            for node in nodes:
                print(f"{node.table_name}.{node.column_name} (query={node.query_id})")
        """
        from .lineage_tracer import trace_forward_full

        return trace_forward_full(
            self.columns,
            self.column_graph._outgoing_index,
            table_name,
            column_name,
            include_ctes,
        )

    def get_table_impact_path(
        self, table_name: str, column_name: str
    ) -> List[Tuple[str, str, str]]:
        """
        Get simplified table-level impact path for a column.

        Returns list of (table_name, column_name, query_id) tuples representing
        the downstream impact through real tables only (skipping CTEs).

        This provides a clear view of how a source column impacts downstream tables.

        Example:
            path = pipeline.get_table_impact_path("raw_orders", "total_amount")
            # Returns:
            # [
            #   ("raw_orders", "total_amount", "01_raw_orders"),
            #   ("stg_orders_enriched", "total_amount", "05_stg_orders_enriched"),
            #   ("mart_customer_ltv", "lifetime_revenue", "07_mart_customer_ltv"),
            #   ...
            # ]
        """
        from .lineage_tracer import get_table_impact_path

        return get_table_impact_path(
            self.columns,
            self.column_graph._outgoing_index,
            table_name,
            column_name,
            self.table_graph.tables,
        )

    def get_lineage_path(
        self, from_table: str, from_column: str, to_table: str, to_column: str
    ) -> List[ColumnEdge]:
        """
        Find the lineage path between two columns.
        Returns list of edges connecting them (if path exists).
        """
        from .lineage_tracer import get_lineage_path

        return get_lineage_path(
            self.columns,
            self.column_graph._outgoing_index,
            from_table,
            from_column,
            to_table,
            to_column,
        )

    def generate_all_descriptions(self, batch_size: int = 10, verbose: bool = True):
        """
        Generate descriptions for all columns using LLM.

        Processes columns in topological order (sources first).

        Args:
            batch_size: Number of columns per batch (currently processes sequentially)
            verbose: If True, print progress messages
        """
        return self._metadata_manager.generate_all_descriptions(batch_size, verbose)

    def propagate_all_metadata(self, verbose: bool = True):
        """
        Propagate metadata (owner, PII, tags) through lineage.

        Uses a two-pass approach:
        1. Backward pass: Propagate metadata from output columns (with SQL comment
           metadata) to their input layer sources. This ensures that if an output
           column has PII from a comment, the source column also gets PII.
        2. Forward pass: Propagate metadata from source columns to downstream
           columns in topological order.

        Args:
            verbose: If True, print progress messages
        """
        return self._metadata_manager.propagate_all_metadata(verbose)

    def get_pii_columns(self) -> List[ColumnNode]:
        """
        Get all columns marked as PII.

        Returns:
            List of columns where pii == True
        """
        return self._metadata_manager.get_pii_columns()

    def get_columns_by_owner(self, owner: str) -> List[ColumnNode]:
        """
        Get all columns with a specific owner.

        Args:
            owner: Owner name to filter by

        Returns:
            List of columns with matching owner
        """
        return self._metadata_manager.get_columns_by_owner(owner)

    def get_columns_by_tag(self, tag: str) -> List[ColumnNode]:
        """
        Get all columns containing a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of columns containing the tag
        """
        return self._metadata_manager.get_columns_by_tag(tag)

    def diff(self, other: "Pipeline"):
        """
        Compare this pipeline with another and return differences.

        Args:
            other: The other pipeline to compare with (typically older version)

        Returns:
            PipelineDiff object containing the differences
        """
        from .diff import PipelineDiff

        return PipelineDiff(new_graph=self, old_graph=other)

    def to_json(self, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Export pipeline to JSON-serializable dictionary.

        Convenience wrapper for JSONExporter.export().

        Args:
            include_metadata: Whether to include metadata (descriptions, PII, etc.)

        Returns:
            Dictionary with columns, edges, and tables

        Example:
            data = pipeline.to_json()
            with open("lineage.json", "w") as f:
                json.dump(data, f, indent=2)
        """
        from .export import JSONExporter

        return JSONExporter.export(self, include_metadata=include_metadata)

    @classmethod
    def from_sql_files(
        cls,
        sql_dir: str,
        dialect: str = "bigquery",
        pattern: str = "*.sql",
        query_id_from: str = "filename",
        template_context: Optional[Dict[str, Any]] = None,
    ) -> "Pipeline":
        """
        Create pipeline from SQL files in a directory.

        Args:
            sql_dir: Directory containing SQL files
            dialect: SQL dialect (bigquery, snowflake, etc.)
            pattern: Glob pattern for SQL files (default: "*.sql")
            query_id_from: How to determine query ID:
                - "filename": Use filename without extension (default)
                - "comment": Extract from first line comment (-- query_id: name)
            template_context: Optional dictionary of template variables

        Returns:
            Pipeline instance

        Example:
            # Query IDs from filenames
            pipeline = Pipeline.from_sql_files("queries/", dialect="bigquery")

            # Query IDs from comments
            pipeline = Pipeline.from_sql_files(
                "queries/",
                query_id_from="comment"
            )

            # With templates
            pipeline = Pipeline.from_sql_files(
                "queries/",
                template_context={"env": "prod", "project": "my_project"}
            )
        """
        from .pipeline_factory import create_from_sql_files

        return create_from_sql_files(
            sql_dir,
            dialect=dialect,
            pattern=pattern,
            query_id_from=query_id_from,
            template_context=template_context,
        )

    @classmethod
    def from_dbt_models(
        cls,
        project_dir: Any,
        schema_map: Optional[Dict[str, str]] = None,
        **pipeline_kwargs: Any,
    ) -> "Pipeline":
        """Build a Pipeline directly from a dbt project's model files.

        dbt models are SELECT-only; this helper wraps each model in
        ``CREATE TABLE <schema>.<model_name> AS ...`` so lineage links into
        the physical table graph.

        Args:
            project_dir: Path to the dbt project root (containing ``models/``).
            schema_map: Optional ordered mapping of ``models/<subdir>`` to the
                target schema. Defaults to ``{"staging": "staging", "marts": "marts"}``.
            **pipeline_kwargs: Forwarded to :class:`Pipeline` (``dialect``,
                ``template_context``, etc.).

        Returns:
            Fully-built Pipeline instance.
        """
        from .pipeline_factory import wrap_dbt_models

        queries = wrap_dbt_models(project_dir, schema_map=schema_map)
        return cls(queries, **pipeline_kwargs)

    def _remap_query_ids(self):
        """Remap auto-generated query IDs to user-provided IDs"""
        # Remap in queries dict
        new_queries = {}
        for auto_id, query in self.table_graph.queries.items():
            user_id = self.query_mapping.get(auto_id, auto_id)
            query.query_id = user_id
            new_queries[user_id] = query
        self.table_graph.queries = new_queries

        # Remap in table references
        for table in self.table_graph.tables.values():
            if table.created_by and table.created_by in self.query_mapping:
                table.created_by = self.query_mapping[table.created_by]
            table.read_by = [self.query_mapping.get(qid, qid) for qid in table.read_by]
            table.modified_by = [self.query_mapping.get(qid, qid) for qid in table.modified_by]

    def __repr__(self):
        """Show topologically sorted SQL statements with query units"""
        sorted_query_ids = self.table_graph.topological_sort()
        query_strs = []

        for query_id in sorted_query_ids:
            query = self.table_graph.queries[query_id]
            # Truncate SQL to first 60 chars for readability
            sql_preview = query.sql.strip().replace("\n", " ")
            if len(sql_preview) > 60:
                sql_preview = sql_preview[:57] + "..."

            query_str = f"{query_id}: {sql_preview}"

            # Add query units if lineage exists
            if query_id in self.query_graphs:
                query_lineage = self.query_graphs[query_id]
                # Extract unique unit_ids from nodes
                unit_ids = sorted({n.unit_id for n in query_lineage.nodes.values() if n.unit_id})
                if unit_ids:
                    # Format each unit on its own line with indentation
                    for unit_id in unit_ids:
                        query_str += f"\n    {unit_id}"

            query_strs.append(query_str)

        queries_display = "\n  ".join(query_strs)
        return f"Pipeline(\n  {queries_display}\n)"

    def build_subpipeline(self, target_table: str) -> "Pipeline":
        """
        Build a subpipeline containing only queries needed to build a specific table.

        This is a convenience wrapper around split() for building a single target.

        Args:
            target_table: The table to build (e.g., "analytics.revenue")

        Returns:
            A new Pipeline containing only the queries needed to build target_table

        Example:
            # Build only what's needed for analytics.revenue
            subpipeline = pipeline.build_subpipeline("analytics.revenue")

            print(f"Full pipeline: {len(pipeline.table_graph.queries)} queries")
            print(f"Subpipeline: {len(subpipeline.table_graph.queries)} queries")

            # Run just the subpipeline
            result = subpipeline.run(executor=execute_sql)
        """
        return self._subpipeline_builder_component.build_subpipeline(target_table)

    def split(self, sinks: List) -> List["Pipeline"]:
        """
        Split pipeline into non-overlapping subpipelines based on target tables.

        Each subpipeline contains all queries needed to build its sink tables,
        ensuring no query appears in multiple subpipelines.

        Args:
            sinks: List of sink specifications. Each element can be:
                   - A single table name (str)
                   - A list of table names (List[str])

        Returns:
            List of Pipeline instances, one per sink group

        Examples:
            # Split into 3 subpipelines
            subpipelines = pipeline.split(
                sinks=[
                    "final_table",           # Single table
                    ["metrics", "summary"],  # Multiple tables in one subpipeline
                    "aggregated_data"        # Another single table
                ]
            )

            # Each subpipeline can be run independently
            subpipelines[0].run(executor=execute_sql)  # Builds final_table
            subpipelines[1].run(executor=execute_sql)  # Builds metrics + summary
            subpipelines[2].run(executor=execute_sql)  # Builds aggregated_data
        """
        return self._subpipeline_builder_component.split(sinks)

    def _get_execution_levels(self) -> List[List[str]]:
        """
        Group queries into levels for concurrent execution.

        Level 0: Queries with no dependencies
        Level 1: Queries that depend only on Level 0
        Level 2: Queries that depend on Level 0 or 1
        etc.

        Queries in the same level can run concurrently.

        Returns:
            List of levels, where each level is a list of query IDs
        """
        from .execution import PipelineExecutor

        return PipelineExecutor(self).get_execution_levels()

    def to_airflow_dag(
        self,
        executor: Callable[[str], None],
        dag_id: str,
        schedule: str = "@daily",
        start_date: Optional[datetime] = None,
        default_args: Optional[dict] = None,
        airflow_version: Optional[str] = None,
        **dag_kwargs,
    ):
        """Create Airflow DAG from this pipeline. See AirflowOrchestrator for full documentation."""
        from .orchestrators import AirflowOrchestrator

        return AirflowOrchestrator(self).to_dag(
            executor=executor,
            dag_id=dag_id,
            schedule=schedule,
            start_date=start_date,
            default_args=default_args,
            airflow_version=airflow_version,
            **dag_kwargs,
        )

    def run(
        self,
        executor: Callable[[str], None],
        max_workers: int = 4,
        verbose: bool = True,
    ) -> dict:
        """Execute pipeline synchronously with concurrent execution. See PipelineExecutor for full documentation."""
        from .execution import PipelineExecutor

        return PipelineExecutor(self).run(
            executor=executor,
            max_workers=max_workers,
            verbose=verbose,
        )

    async def async_run(
        self,
        executor: Callable[[str], Awaitable[None]],
        max_workers: int = 4,
        verbose: bool = True,
    ) -> dict:
        """Execute pipeline asynchronously with concurrent execution. See PipelineExecutor for full documentation."""
        from .execution import PipelineExecutor

        return await PipelineExecutor(self).async_run(
            executor=executor,
            max_workers=max_workers,
            verbose=verbose,
        )

    # ========================================================================
    # Orchestrator Methods - Dagster
    # ========================================================================

    def to_dagster_assets(
        self,
        executor: Callable[[str], None],
        group_name: Optional[str] = None,
        key_prefix: Optional[Union[str, List[str]]] = None,
        compute_kind: str = "sql",
        **asset_kwargs,
    ) -> List:
        """Create Dagster Assets from this pipeline. See DagsterOrchestrator for full documentation."""
        from .orchestrators import DagsterOrchestrator

        return DagsterOrchestrator(self).to_assets(
            executor=executor,
            group_name=group_name,
            key_prefix=key_prefix,
            compute_kind=compute_kind,
            **asset_kwargs,
        )

    def to_dagster_job(
        self,
        executor: Callable[[str], None],
        job_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **job_kwargs,
    ):
        """Create Dagster Job from this pipeline using ops. See DagsterOrchestrator for full documentation."""
        from .orchestrators import DagsterOrchestrator

        return DagsterOrchestrator(self).to_job(
            executor=executor,
            job_name=job_name,
            description=description,
            tags=tags,
            **job_kwargs,
        )

    # ========================================================================
    # Orchestrator Methods - Prefect
    # ========================================================================

    def to_prefect_flow(
        self,
        executor: Callable[[str], None],
        flow_name: str,
        description: Optional[str] = None,
        retries: int = 2,
        retry_delay_seconds: int = 60,
        timeout_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        **flow_kwargs,
    ):
        """Create Prefect Flow from this pipeline. See PrefectOrchestrator for full documentation."""
        from .orchestrators import PrefectOrchestrator

        return PrefectOrchestrator(self).to_flow(
            executor=executor,
            flow_name=flow_name,
            description=description,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            timeout_seconds=timeout_seconds,
            tags=tags,
            **flow_kwargs,
        )

    def to_prefect_deployment(
        self,
        executor: Callable[[str], None],
        flow_name: str,
        deployment_name: str,
        cron: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        work_pool_name: Optional[str] = None,
        **kwargs,
    ):
        """Create Prefect Deployment from this pipeline for scheduled execution. See PrefectOrchestrator for full documentation."""
        from .orchestrators import PrefectOrchestrator

        return PrefectOrchestrator(self).to_deployment(
            executor=executor,
            flow_name=flow_name,
            deployment_name=deployment_name,
            cron=cron,
            interval_seconds=interval_seconds,
            work_pool_name=work_pool_name,
            **kwargs,
        )

    # ========================================================================
    # Orchestrator Methods - Kestra
    # ========================================================================

    def to_kestra_flow(
        self,
        flow_id: str,
        namespace: str,
        description: Optional[str] = None,
        connection_config: Optional[Dict[str, str]] = None,
        cron: Optional[str] = None,
        retry_attempts: int = 3,
        labels: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> str:
        """Generate Kestra flow YAML from this pipeline. See KestraOrchestrator for full documentation."""
        from .orchestrators import KestraOrchestrator

        orchestrator = KestraOrchestrator(self)

        if cron:
            return orchestrator.to_flow_with_triggers(
                flow_id=flow_id,
                namespace=namespace,
                description=description,
                connection_config=connection_config,
                cron=cron,
                retry_attempts=retry_attempts,
                labels=labels,
                **kwargs,
            )
        else:
            return orchestrator.to_flow(
                flow_id=flow_id,
                namespace=namespace,
                description=description,
                connection_config=connection_config,
                retry_attempts=retry_attempts,
                labels=labels,
                **kwargs,
            )

    # ========================================================================
    # Orchestrator Methods - Mage
    # ========================================================================

    def to_mage_pipeline(
        self,
        pipeline_name: str,
        description: Optional[str] = None,
        connection_name: str = "clickhouse_default",
        db_connector: str = "clickhouse",
    ) -> Dict[str, Any]:
        """Generate Mage pipeline files from this pipeline. See MageOrchestrator for full documentation."""
        from .orchestrators import MageOrchestrator

        return MageOrchestrator(self).to_pipeline_files(
            pipeline_name=pipeline_name,
            description=description,
            connection_name=connection_name,
            db_connector=db_connector,
        )

    # ========================================================================
    # Validation Methods (delegate to PipelineValidator)
    # ========================================================================

    def get_all_issues(self) -> List["ValidationIssue"]:
        """
        Get all validation issues from all queries in the pipeline.

        Returns combined list of issues from:
        - Individual query lineage graphs
        - Pipeline-level lineage graph

        Returns:
            List of ValidationIssue objects
        """
        return self._pipeline_validator.get_all_issues()

    def get_issues(
        self,
        severity: Optional[str | IssueSeverity] = None,
        category: Optional[str | IssueCategory] = None,
        query_id: Optional[str] = None,
    ) -> List["ValidationIssue"]:
        """
        Get filtered validation issues.

        Args:
            severity: Filter by severity ('error', 'warning', 'info' or IssueSeverity enum)
            category: Filter by category (string or IssueCategory enum)
            query_id: Filter by query ID

        Returns:
            Filtered list of ValidationIssue objects

        Example:
            # Get all errors (using string)
            errors = pipeline.get_issues(severity='error')

            # Get all errors (using enum)
            errors = pipeline.get_issues(severity=IssueSeverity.ERROR)

            # Get all star-related issues
            star_issues = pipeline.get_issues(category=IssueCategory.UNQUALIFIED_STAR_MULTIPLE_TABLES)

            # Get all issues from a specific query
            query_issues = pipeline.get_issues(query_id='query_1')
        """
        return self._pipeline_validator.get_issues(severity, category, query_id)

    def has_errors(self) -> bool:
        """Check if pipeline has any ERROR-level issues"""
        return self._pipeline_validator.has_errors()

    def has_warnings(self) -> bool:
        """Check if pipeline has any WARNING-level issues"""
        return self._pipeline_validator.has_warnings()

    def print_issues(self, severity: Optional[str | IssueSeverity] = None):
        """
        Print all validation issues in a human-readable format.

        Args:
            severity: Optional filter by severity ('error', 'warning', 'info' or IssueSeverity enum)
        """
        self._pipeline_validator.print_issues(severity)


__all__ = [
    "PipelineLineageBuilder",
    "Pipeline",
]
