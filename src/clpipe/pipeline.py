"""
Pipeline orchestration with integrated lineage analysis.

Contains Pipeline class for unified SQL workflow orchestration with:
- Table and column lineage
- Metadata propagation
- LLM-powered documentation
- Pipeline execution (sync/async)
- Airflow DAG generation
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sqlglot import exp

from .column import PipelineColumnEdge, PipelineColumnNode
from .lineage_builder import RecursiveLineageBuilder
from .models import ColumnLineageGraph, ColumnNode, DescriptionSource, ParsedQuery
from .table import TableDependencyGraph


class PipelineLineageBuilder:
    """
    Builds unified lineage graph from multiple queries.
    Combines single-query lineage with cross-query connections.
    """

    def build(self, pipeline_or_graph) -> "Pipeline":
        """
        Build unified pipeline lineage graph.

        Args:
            pipeline_or_graph: Either a Pipeline instance to populate,
                              or a TableDependencyGraph (for backward compatibility)

        Returns:
            The populated Pipeline instance

        Algorithm:
        1. Topologically sort queries
        2. For each query (bottom-up):
           a. Run single-query lineage (RecursiveLineageBuilder)
           b. Add columns to pipeline graph
           c. Add within-query edges
        3. Add cross-query edges (connect tables)
        """
        # Handle backward compatibility: accept TableDependencyGraph directly
        if isinstance(pipeline_or_graph, TableDependencyGraph):
            pipeline = Pipeline._create_empty(pipeline_or_graph)
        else:
            pipeline = pipeline_or_graph

        table_graph = pipeline.table_graph

        # Step 1: Topological sort
        sorted_query_ids = table_graph.topological_sort()

        # Step 2: Process each query
        for query_id in sorted_query_ids:
            query = table_graph.queries[query_id]

            # Step 2a: Run single-query lineage
            try:
                # Extract SELECT statement from DDL/DML if needed
                sql_for_lineage = self._extract_select_from_query(query)

                if sql_for_lineage:
                    # RecursiveLineageBuilder handles parsing internally
                    lineage_builder = RecursiveLineageBuilder(sql_for_lineage)
                    query_lineage = lineage_builder.build()

                    # Store query lineage
                    pipeline.query_lineages[query_id] = query_lineage
                    query.query_lineage = query_lineage

                    # Step 2b: Add columns to pipeline graph
                    self._add_query_columns(pipeline, query, query_lineage)

                    # Step 2c: Add within-query edges
                    self._add_query_edges(pipeline, query, query_lineage)
                else:
                    # No SELECT to analyze (e.g., UPDATE without SELECT)
                    print(f"Info: Skipping lineage for {query_id} (no SELECT statement)")
            except Exception as e:
                # If lineage fails, skip this query
                print(f"Warning: Failed to build lineage for {query_id}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Step 3: Add cross-query edges
        self._add_cross_query_edges(pipeline)

        return pipeline

    def _add_query_columns(
        self,
        pipeline: "Pipeline",
        query: ParsedQuery,
        query_lineage: ColumnLineageGraph,
    ):
        """
        Add all columns from a query to the pipeline graph.

        Note: We add all nodes (both input and output layers) to maintain full lineage.
        Input layer nodes represent source columns, output layer nodes represent derived columns.
        Both are needed for complete lineage tracing.
        """
        # Add columns with table context
        for node in query_lineage.nodes.values():
            # Extract metadata from SQL comments if available
            description = None
            description_source = None
            pii = False
            owner = None
            tags = set()
            custom_metadata = {}

            if node.sql_metadata is not None:
                metadata = node.sql_metadata
                description = metadata.description
                pii = metadata.pii or False
                owner = metadata.owner
                tags = metadata.tags
                custom_metadata = metadata.custom_metadata

                # Set description source if we have a description from SQL
                if description:
                    description_source = DescriptionSource.SOURCE

            column = PipelineColumnNode(
                column_name=node.column_name,
                table_name=self._infer_table_name(node, query),
                query_id=query.query_id,
                node_type=node.node_type,
                full_name=self._make_full_name(node, query),
                expression=node.expression,
                operation=node.node_type,  # Use node_type as operation for now
                description=description,
                description_source=description_source,
                pii=pii,
                owner=owner,
                tags=tags,
                custom_metadata=custom_metadata,
            )
            pipeline.add_column(column)

    def _add_query_edges(
        self,
        pipeline: "Pipeline",
        query: ParsedQuery,
        query_lineage: ColumnLineageGraph,
    ):
        """
        Add all edges from a query to the pipeline graph.
        """
        for edge in query_lineage.edges:
            from_full = self._make_full_name(edge.from_node, query)
            to_full = self._make_full_name(edge.to_node, query)

            if from_full in pipeline.columns and to_full in pipeline.columns:
                pipeline_edge = PipelineColumnEdge(
                    from_column=pipeline.columns[from_full],
                    to_column=pipeline.columns[to_full],
                    edge_type=edge.transformation,
                    transformation=edge.transformation,
                    query_id=query.query_id,
                )
                pipeline.add_edge(pipeline_edge)

    def _add_cross_query_edges(self, pipeline: "Pipeline"):
        """
        Add edges connecting queries via tables.

        Algorithm:
        For each table T:
          - Find query Q1 that creates/modifies T
          - Find queries [Q2, Q3, ...] that read from T
          - For each column C in Q1's output:
              - For each query Qi that reads T:
                  - If Qi references T.C, create edge: Q1.C -> Qi.C
        """
        for table_name, table_node in pipeline.table_graph.tables.items():
            # Find query that creates this table
            if not table_node.created_by:
                continue  # External source table

            creating_query_id = table_node.created_by

            # Find output columns from creating query
            output_columns = [
                col
                for col in pipeline.columns.values()
                if col.query_id == creating_query_id and col.table_name == table_name
            ]

            # Find queries that read this table
            for reading_query_id in table_node.read_by:
                # Match columns by name
                for output_col in output_columns:
                    # Find corresponding input column in reading query
                    # Search for this column in reading query's lineage
                    for col in pipeline.columns.values():
                        if (
                            col.query_id == reading_query_id
                            and col.table_name == table_name
                            and col.column_name == output_col.column_name
                        ):
                            # Create cross-query edge
                            edge = PipelineColumnEdge(
                                from_column=output_col,
                                to_column=col,
                                edge_type="cross_query",
                                transformation=f"{creating_query_id} -> {reading_query_id}",
                                query_id=None,  # Cross-query edge
                            )
                            pipeline.add_edge(edge)

    def _infer_table_name(self, node: ColumnNode, query: ParsedQuery) -> Optional[str]:
        """
        Infer which table this column belongs to.
        Maps table references (aliases) to fully qualified names.
        """
        # For output columns, use destination table
        if node.layer == "output":
            return query.destination_table

        # For input columns, map table_name to fully qualified name
        if node.table_name:
            # Single-query lineage uses short table names like "orders", "users"
            # Pipeline uses fully qualified names like "raw.orders", "staging.users"

            # Try exact match first (already qualified)
            if node.table_name in query.source_tables:
                return node.table_name

            # Try to find matching source table by suffix
            for source_table in query.source_tables:
                # Check if source_table ends with ".{node.table_name}"
                if source_table.endswith(f".{node.table_name}"):
                    return source_table
                # Or if they're the same (no schema prefix)
                if source_table == node.table_name:
                    return source_table

            # If only one source table, assume it's that one
            if len(query.source_tables) == 1:
                return list(query.source_tables)[0]

        # Fallback: if only one source table, use it
        if len(query.source_tables) == 1:
            return list(query.source_tables)[0]

        # Ambiguous - can't determine table
        return None

    def _make_full_name(self, node: ColumnNode, query: ParsedQuery) -> str:
        """
        Create fully qualified column name.
        """
        table_name = self._infer_table_name(node, query)
        if table_name:
            return f"{table_name}.{node.column_name}"
        else:
            return f"{query.query_id}.{node.column_name}"

    def _extract_select_from_query(self, query: ParsedQuery) -> Optional[str]:
        """
        Extract SELECT statement from DDL/DML queries.
        Single-query lineage only works on SELECT statements, so we need to extract
        the SELECT from CREATE TABLE AS SELECT, INSERT INTO ... SELECT, etc.
        """
        ast = query.ast

        # CREATE TABLE/VIEW AS SELECT
        if isinstance(ast, exp.Create):
            if ast.expression and isinstance(ast.expression, exp.Select):
                return ast.expression.sql()

        # INSERT INTO ... SELECT
        elif isinstance(ast, exp.Insert):
            if ast.expression and isinstance(ast.expression, exp.Select):
                return ast.expression.sql()

        # MERGE uses a USING clause which is a SELECT or table
        elif isinstance(ast, exp.Merge):
            # Merge is complex - for now skip lineage
            return None

        # Plain SELECT
        elif isinstance(ast, exp.Select):
            return query.sql

        # UPDATE, DELETE, etc. - no SELECT to extract
        return None


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

    def __init__(self, queries: List[Tuple[str, str]], dialect: str = "bigquery"):
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
        """
        from .multi_query import MultiQueryParser

        self.dialect = dialect
        self.query_mapping: Dict[str, str] = {}  # Maps auto_id -> user_id

        # Column-level lineage (from PipelineLineageGraph)
        self.columns: Dict[str, PipelineColumnNode] = {}  # full_name -> PipelineColumnNode
        self.edges: List[PipelineColumnEdge] = []
        self.query_lineages: Dict[str, ColumnLineageGraph] = {}
        self.llm: Optional[Any] = None  # LangChain BaseChatModel

        # Convert tuples to plain SQL strings for MultiQueryParser
        sql_list = []
        for user_query_id, sql in queries:
            sql_list.append(sql)
            auto_id = f"query_{len(sql_list) - 1}"
            self.query_mapping[auto_id] = user_query_id

        # Parse queries using current API
        parser = MultiQueryParser(dialect=dialect)
        self.table_graph = parser.parse_queries(sql_list)

        # Remap auto-generated query IDs to user-provided IDs
        self._remap_query_ids()

        # Build lineage directly into this Pipeline instance
        builder = PipelineLineageBuilder()
        builder.build(self)

    @classmethod
    def from_tuples(cls, queries: List[Tuple[str, str]], dialect: str = "bigquery") -> "Pipeline":
        """
        Create pipeline from list of (query_id, sql) tuples.

        Args:
            queries: List of (query_id, sql) tuples
            dialect: SQL dialect (bigquery, snowflake, etc.)

        Returns:
            Pipeline instance

        Example:
            pipeline = Pipeline.from_tuples([
                ("staging", "CREATE TABLE staging AS SELECT * FROM raw"),
                ("final", "CREATE TABLE final AS SELECT * FROM staging"),
            ])
        """
        return cls(queries, dialect=dialect)

    @classmethod
    def from_dict(cls, queries: Dict[str, str], dialect: str = "bigquery") -> "Pipeline":
        """
        Create pipeline from dictionary of {query_id: sql}.

        Args:
            queries: Dictionary mapping query_id to SQL string
            dialect: SQL dialect (bigquery, snowflake, etc.)

        Returns:
            Pipeline instance

        Example:
            pipeline = Pipeline.from_dict({
                "staging": "CREATE TABLE staging AS SELECT * FROM raw",
                "final": "CREATE TABLE final AS SELECT * FROM staging",
            })
        """
        query_list = list(queries.items())
        return cls(query_list, dialect=dialect)

    @classmethod
    def from_sql_list(cls, queries: List[str], dialect: str = "bigquery") -> "Pipeline":
        """
        Create pipeline from list of SQL strings (auto-generates query IDs).

        Query IDs are generated as: {operation}_{table_name}
        If duplicates exist, a number suffix is added: {operation}_{table_name}_2

        Args:
            queries: List of SQL query strings
            dialect: SQL dialect (bigquery, snowflake, etc.)

        Returns:
            Pipeline instance

        Example:
            pipeline = Pipeline.from_sql_list([
                "CREATE TABLE staging AS SELECT * FROM raw",
                "INSERT INTO staging SELECT * FROM raw2",
                "CREATE TABLE final AS SELECT * FROM staging",
            ])
            # Query IDs will be: create_staging, insert_staging, create_final
        """
        query_list = []
        id_counts: Dict[str, int] = {}

        for sql in queries:
            query_id = cls._generate_query_id(sql, dialect, id_counts)
            query_list.append((query_id, sql))

        return cls(query_list, dialect=dialect)

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
        import sqlglot
        from sqlglot import exp

        try:
            parsed = sqlglot.parse_one(sql, dialect=dialect)

            # Determine operation
            if isinstance(parsed, exp.Create):
                if parsed.kind == "VIEW":
                    operation = "create_view"
                else:
                    operation = "create"
            elif isinstance(parsed, exp.Insert):
                operation = "insert"
            elif isinstance(parsed, exp.Merge):
                operation = "merge"
            elif isinstance(parsed, exp.Update):
                operation = "update"
            elif isinstance(parsed, exp.Delete):
                operation = "delete"
            elif isinstance(parsed, exp.Select):
                operation = "select"
            else:
                operation = "query"

            # Determine destination table name
            dest_table = None
            if isinstance(parsed, exp.Create):
                table_expr = parsed.this
                if table_expr:
                    dest_table = table_expr.name
            elif isinstance(parsed, (exp.Insert, exp.Merge)):
                table_expr = parsed.this
                if table_expr:
                    dest_table = table_expr.name
            elif isinstance(parsed, (exp.Update, exp.Delete)):
                table_expr = parsed.this
                if table_expr:
                    dest_table = table_expr.name

            # Determine source tables
            source_tables = []
            for table in parsed.find_all(exp.Table):
                # Skip the destination table
                table_name = table.name
                if table_name and table_name != dest_table:
                    source_tables.append(table_name)

            # Build base ID
            if dest_table:
                base_id = f"{operation}_{dest_table}"
            else:
                base_id = operation

            # Try base_id first
            if base_id not in id_counts:
                id_counts[base_id] = 1
                return base_id

            # Try with source table
            if source_tables:
                # Use first source table
                id_with_source = f"{base_id}_from_{source_tables[0]}"
                if id_with_source not in id_counts:
                    id_counts[id_with_source] = 1
                    return id_with_source
                else:
                    # Still duplicate, use number
                    id_counts[id_with_source] += 1
                    return f"{id_with_source}_{id_counts[id_with_source]}"
            else:
                # No source table, use number
                id_counts[base_id] += 1
                return f"{base_id}_{id_counts[base_id]}"

        except Exception:
            # Fallback if parsing fails
            base_id = "query"
            if base_id not in id_counts:
                id_counts[base_id] = 1
                return base_id
            else:
                id_counts[base_id] += 1
                return f"{base_id}_{id_counts[base_id]}"

    @classmethod
    def from_sql_string(cls, sql: str, dialect: str = "bigquery") -> "Pipeline":
        """
        Create pipeline from single SQL string with semicolon-separated queries.

        Query IDs are generated as: {operation}_{table_name}
        If duplicates exist, a number suffix is added: {operation}_{table_name}_2

        Args:
            sql: SQL string with multiple queries separated by semicolons
            dialect: SQL dialect (bigquery, snowflake, etc.)

        Returns:
            Pipeline instance

        Example:
            pipeline = Pipeline.from_sql_string('''
                CREATE TABLE staging AS SELECT * FROM raw;
                CREATE TABLE final AS SELECT * FROM staging
            ''')
            # Query IDs will be: create_staging, create_final
        """
        # Split by semicolon and filter empty strings
        queries = [q.strip() for q in sql.split(";") if q.strip()]
        return cls.from_sql_list(queries, dialect=dialect)

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
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.dialect = "bigquery"
        instance.query_mapping = {}
        instance.columns = {}
        instance.edges = []
        instance.query_lineages = {}
        instance.llm = None
        instance.table_graph = table_graph
        return instance

    # === Lineage methods (from PipelineLineageGraph) ===

    def add_column(self, column: PipelineColumnNode) -> PipelineColumnNode:
        """Add a column node to the graph"""
        self.columns[column.full_name] = column

        # Also add to table's columns set if table exists
        if column.table_name and column.table_name in self.table_graph.tables:
            self.table_graph.tables[column.table_name].columns.add(column.column_name)

        return column

    def add_edge(self, edge: PipelineColumnEdge):
        """Add a lineage edge"""
        self.edges.append(edge)

    def trace_column_backward(self, table_name: str, column_name: str) -> List[PipelineColumnNode]:
        """
        Trace a column backward to its ultimate sources.
        Returns list of source columns across all queries.
        """
        # Start from the target column
        full_name = f"{table_name}.{column_name}"
        if full_name not in self.columns:
            return []

        # BFS backward through edges
        visited = set()
        queue = [self.columns[full_name]]
        sources = []

        while queue:
            current = queue.pop(0)
            if current.full_name in visited:
                continue
            visited.add(current.full_name)

            # Find incoming edges
            incoming = [e for e in self.edges if e.to_column.full_name == current.full_name]

            if not incoming:
                # No incoming edges = source column
                sources.append(current)
            else:
                for edge in incoming:
                    queue.append(edge.from_column)

        return sources

    def trace_column_forward(self, table_name: str, column_name: str) -> List[PipelineColumnNode]:
        """
        Trace a column forward to see what depends on it.
        Returns list of downstream columns across all queries.
        """
        # Start from the source column
        full_name = f"{table_name}.{column_name}"
        if full_name not in self.columns:
            return []

        # BFS forward through edges
        visited = set()
        queue = [self.columns[full_name]]
        descendants = []

        while queue:
            current = queue.pop(0)
            if current.full_name in visited:
                continue
            visited.add(current.full_name)

            # Find outgoing edges
            outgoing = [e for e in self.edges if e.from_column.full_name == current.full_name]

            if not outgoing:
                # No outgoing edges = final column
                descendants.append(current)
            else:
                for edge in outgoing:
                    queue.append(edge.to_column)

        return descendants

    def get_lineage_path(
        self, from_table: str, from_column: str, to_table: str, to_column: str
    ) -> List[PipelineColumnEdge]:
        """
        Find the lineage path between two columns.
        Returns list of edges connecting them (if path exists).
        """
        # BFS to find shortest path
        from_full = f"{from_table}.{from_column}"
        to_full = f"{to_table}.{to_column}"

        if from_full not in self.columns or to_full not in self.columns:
            return []

        # BFS with path tracking
        queue = [(self.columns[from_full], [])]
        visited = set()

        while queue:
            current, path = queue.pop(0)
            if current.full_name in visited:
                continue
            visited.add(current.full_name)

            if current.full_name == to_full:
                return path

            # Find outgoing edges
            for edge in self.edges:
                if edge.from_column.full_name == current.full_name:
                    queue.append((edge.to_column, path + [edge]))

        return []  # No path found

    def generate_all_descriptions(self, batch_size: int = 10, verbose: bool = True):
        """
        Generate descriptions for all columns using LLM.

        Processes columns in topological order (sources first).

        Args:
            batch_size: Number of columns per batch (currently processes sequentially)
            verbose: If True, print progress messages
        """
        if not self.llm:
            raise ValueError("LLM not configured. Set pipeline.llm before calling.")

        # Get columns in topological order
        sorted_query_ids = self.table_graph.topological_sort()

        columns_to_process = []
        for query_id in sorted_query_ids:
            query = self.table_graph.queries[query_id]
            if query.destination_table:
                for col in self.columns.values():
                    if (
                        col.table_name == query.destination_table
                        and not col.description
                        and col.is_computed()
                    ):
                        columns_to_process.append(col)

        if verbose:
            print(f"📊 Generating descriptions for {len(columns_to_process)} columns...")

        # Process columns
        for i, col in enumerate(columns_to_process):
            if verbose and (i + 1) % batch_size == 0:
                print(f"   Processed {i + 1}/{len(columns_to_process)} columns...")

            col.generate_description(self.llm, self)

        if verbose:
            print(f"✅ Done! Generated {len(columns_to_process)} descriptions")

    def propagate_all_metadata(self, verbose: bool = True):
        """
        Propagate metadata (owner, PII, tags) through lineage.

        Processes columns in topological order (sources first) to ensure
        metadata flows correctly through transformations.

        Args:
            verbose: If True, print progress messages
        """
        # Get columns in topological order
        sorted_query_ids = self.table_graph.topological_sort()

        columns_to_process = []
        for query_id in sorted_query_ids:
            query = self.table_graph.queries[query_id]
            if query.destination_table:
                for col in self.columns.values():
                    if col.table_name == query.destination_table and col.is_computed():
                        columns_to_process.append(col)

        if verbose:
            print(f"📊 Propagating metadata for {len(columns_to_process)} columns...")

        # Process columns
        for col in columns_to_process:
            col.propagate_metadata(self)

        if verbose:
            print(f"✅ Done! Propagated metadata for {len(columns_to_process)} columns")

    def save(self, file_path: str):
        """
        Save metadata to file using cloudpickle.

        Saves only metadata (descriptions, owners, PII, tags, custom_metadata),
        not the SQL code or lineage structure.

        Args:
            file_path: Path to save metadata file
        """
        import cloudpickle

        # Extract metadata from all columns
        column_metadata = {}
        for full_name, col in self.columns.items():
            column_metadata[full_name] = {
                "description": col.description,
                "description_source": col.description_source.value
                if col.description_source
                else None,
                "owner": col.owner,
                "pii": col.pii,
                "tags": list(col.tags),
                "custom_metadata": col.custom_metadata,
            }

        # Extract metadata from all tables
        table_metadata = {}
        for table_name, table in self.table_graph.tables.items():
            table_metadata[table_name] = {
                "description": table.description,
            }

        # Create metadata dict
        metadata = {
            "version": "1.0",
            "columns": column_metadata,
            "tables": table_metadata,
        }

        # Serialize with cloudpickle
        with open(file_path, "wb") as f:
            cloudpickle.dump(metadata, f)

    @staticmethod
    def load_metadata(file_path: str) -> Dict:
        """
        Load metadata from file.

        Args:
            file_path: Path to metadata file

        Returns:
            Metadata dictionary
        """
        import cloudpickle

        with open(file_path, "rb") as f:
            metadata = cloudpickle.load(f)

        return metadata

    def apply_metadata(self, metadata: Dict):
        """
        Apply loaded metadata to columns and tables.

        Args:
            metadata: Metadata dictionary from load_metadata()
        """
        from .models import DescriptionSource

        # Apply column metadata
        column_metadata = metadata.get("columns", {})
        for full_name, col_meta in column_metadata.items():
            if full_name in self.columns:
                col = self.columns[full_name]
                col.description = col_meta.get("description")

                # Convert description_source string back to enum
                desc_source_str = col_meta.get("description_source")
                if desc_source_str:
                    col.description_source = DescriptionSource(desc_source_str)

                col.owner = col_meta.get("owner")
                col.pii = col_meta.get("pii", False)
                col.tags = set(col_meta.get("tags", []))
                col.custom_metadata = col_meta.get("custom_metadata", {})

        # Apply table metadata
        table_metadata = metadata.get("tables", {})
        for table_name, table_meta in table_metadata.items():
            if table_name in self.table_graph.tables:
                table = self.table_graph.tables[table_name]
                table.description = table_meta.get("description")

    def get_pii_columns(self) -> List[PipelineColumnNode]:
        """
        Get all columns marked as PII.

        Returns:
            List of columns where pii == True
        """
        return [col for col in self.columns.values() if col.pii]

    def get_columns_by_owner(self, owner: str) -> List[PipelineColumnNode]:
        """
        Get all columns with a specific owner.

        Args:
            owner: Owner name to filter by

        Returns:
            List of columns with matching owner
        """
        return [col for col in self.columns.values() if col.owner == owner]

    def get_columns_by_tag(self, tag: str) -> List[PipelineColumnNode]:
        """
        Get all columns containing a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of columns containing the tag
        """
        return [col for col in self.columns.values() if tag in col.tags]

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
        """
        import re
        from pathlib import Path

        sql_path = Path(sql_dir)
        sql_files = sorted(sql_path.glob(pattern))

        if not sql_files:
            raise ValueError(f"No SQL files found in {sql_dir} matching {pattern}")

        queries = []
        for sql_file in sql_files:
            sql_content = sql_file.read_text()

            if query_id_from == "filename":
                query_id = sql_file.stem  # Filename without extension
            elif query_id_from == "comment":
                # Extract from first line comment: -- query_id: name
                match = re.match(r"--\s*query_id:\s*(\w+)", sql_content)
                if match:
                    query_id = match.group(1)
                else:
                    # Fallback to filename if no comment found
                    query_id = sql_file.stem
            else:
                raise ValueError(f"Invalid query_id_from: {query_id_from}")

            queries.append((query_id, sql_content))

        return cls(queries, dialect=dialect)

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
        # Normalize sinks to list of lists
        normalized_sinks: List[List[str]] = []
        for sink in sinks:
            if isinstance(sink, str):
                normalized_sinks.append([sink])
            elif isinstance(sink, list):
                normalized_sinks.append(sink)
            else:
                raise ValueError(f"Invalid sink type: {type(sink)}. Expected str or List[str]")

        # For each sink group, find all required queries
        subpipeline_queries: List[set] = []

        for sink_group in normalized_sinks:
            required_queries = set()

            # BFS backward from each sink to find all dependencies
            for sink_table in sink_group:
                if sink_table not in self.table_graph.tables:
                    raise ValueError(
                        f"Sink table '{sink_table}' not found in pipeline. "
                        f"Available tables: {list(self.table_graph.tables.keys())}"
                    )

                # Find all queries needed for this sink
                visited = set()
                queue = [sink_table]

                while queue:
                    current_table = queue.pop(0)
                    if current_table in visited:
                        continue
                    visited.add(current_table)

                    table_node = self.table_graph.tables.get(current_table)
                    if not table_node:
                        continue

                    # Add the query that creates this table
                    if table_node.created_by:
                        query_id = table_node.created_by
                        required_queries.add(query_id)

                        # Add source tables to queue
                        query = self.table_graph.queries[query_id]
                        for source_table in query.source_tables:
                            if source_table not in visited:
                                queue.append(source_table)

            subpipeline_queries.append(required_queries)

        # Ensure non-overlapping: assign each query to only one subpipeline
        # Strategy: Assign to the first subpipeline that needs it
        assigned_queries: dict = {}  # query_id -> subpipeline_index

        for idx, query_set in enumerate(subpipeline_queries):
            for query_id in query_set:
                if query_id not in assigned_queries:
                    assigned_queries[query_id] = idx

        # Build final non-overlapping query sets
        final_query_sets: List[set] = [set() for _ in normalized_sinks]
        for query_id, subpipeline_idx in assigned_queries.items():
            final_query_sets[subpipeline_idx].add(query_id)

        # Create Pipeline instances for each subpipeline
        subpipelines = []

        for query_ids in final_query_sets:
            if not query_ids:
                # Empty subpipeline - skip
                continue

            # Extract queries in order
            subpipeline_query_list = []
            for query_id in self.table_graph.topological_sort():
                if query_id in query_ids:
                    query = self.table_graph.queries[query_id]
                    subpipeline_query_list.append((query_id, query.sql))

            # Create new Pipeline instance
            subpipeline = Pipeline(subpipeline_query_list, dialect=self.dialect)
            subpipelines.append(subpipeline)

        return subpipelines

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
        levels = []
        completed = set()

        while len(completed) < len(self.table_graph.queries):
            current_level = []

            for query_id, query in self.table_graph.queries.items():
                if query_id in completed:
                    continue

                # Check if all dependencies are completed
                dependencies_met = True
                for source_table in query.source_tables:
                    # Find query that creates this table
                    table_node = self.table_graph.tables.get(source_table)
                    if table_node and table_node.created_by:
                        if table_node.created_by not in completed:
                            dependencies_met = False
                            break

                if dependencies_met:
                    current_level.append(query_id)

            if not current_level:
                # No progress - circular dependency
                raise RuntimeError("Circular dependency detected in pipeline")

            levels.append(current_level)
            completed.update(current_level)

        return levels

    def to_airflow_dag(
        self,
        executor: Callable[[str], None],
        dag_id: str,
        schedule: str = "@daily",
        start_date: Optional[datetime] = None,
        default_args: Optional[dict] = None,
        **dag_kwargs,
    ):
        """
        Create Airflow DAG from this pipeline using TaskFlow API.

        Supports all Airflow DAG parameters via **dag_kwargs for complete flexibility.

        Args:
            executor: Function that executes SQL (takes sql string)
            dag_id: Airflow DAG ID
            schedule: Schedule interval (default: "@daily")
            start_date: DAG start date (default: datetime(2024, 1, 1))
            default_args: Airflow default_args (default: owner='data_team', retries=2)
            **dag_kwargs: Additional DAG parameters (catchup, tags, max_active_runs,
                         description, max_active_tasks, dagrun_timeout, etc.)
                         See Airflow DAG documentation for all available parameters.

        Returns:
            Airflow DAG instance

        Examples:
            # Basic usage
            def execute_sql(sql: str):
                from google.cloud import bigquery
                client = bigquery.Client()
                client.query(sql).result()

            dag = pipeline.to_airflow_dag(
                executor=execute_sql,
                dag_id="my_pipeline"
            )

            # Advanced usage with all DAG parameters
            dag = pipeline.to_airflow_dag(
                executor=execute_sql,
                dag_id="my_pipeline",
                schedule="0 0 * * *",  # Daily at midnight
                description="Customer analytics pipeline",
                catchup=False,
                max_active_runs=3,
                max_active_tasks=10,
                tags=["analytics", "daily"],
                default_view="graph",
                orientation="LR",
            )
        """
        try:
            from airflow.decorators import dag, task  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "Airflow is required for DAG generation. "
                "Install it with: pip install apache-airflow"
            ) from e

        if start_date is None:
            start_date = datetime(2024, 1, 1)

        if default_args is None:
            default_args = {
                "owner": "data_team",
                "retries": 2,
                "retry_delay": timedelta(minutes=5),
            }

        # Build DAG parameters
        dag_params = {
            "dag_id": dag_id,
            "schedule": schedule,
            "start_date": start_date,
            "default_args": default_args,
            **dag_kwargs,  # Allow user to override any parameter
        }

        # Set default values only if not provided by user
        dag_params.setdefault("catchup", False)
        dag_params.setdefault("tags", ["clpipe"])

        table_graph = self.table_graph

        @dag(**dag_params)
        def pipeline_dag():
            """Generated pipeline DAG"""

            # Create task callables for each query
            task_callables = {}

            for query_id in table_graph.topological_sort():
                query = table_graph.queries[query_id]
                sql_to_execute = query.sql

                # Create task with unique function name using closure
                def make_task(qid, sql):
                    @task(task_id=qid.replace("-", "_"))
                    def execute_query():
                        """Execute SQL query"""
                        executor(sql)
                        return f"Completed: {qid}"

                    return execute_query

                task_callables[query_id] = make_task(query_id, sql_to_execute)

            # Instantiate all tasks once before wiring dependencies
            task_instances = {qid: callable() for qid, callable in task_callables.items()}

            # Set up dependencies based on table lineage
            for _table_name, table_node in table_graph.tables.items():
                if table_node.created_by:
                    upstream_id = table_node.created_by
                    for downstream_id in table_node.read_by:
                        if upstream_id in task_instances and downstream_id in task_instances:
                            # Airflow: downstream >> upstream means upstream runs first
                            task_instances[upstream_id] >> task_instances[downstream_id]

        return pipeline_dag()

    def run(
        self,
        executor: Callable[[str], None],
        max_workers: int = 4,
        verbose: bool = True,
    ) -> dict:
        """
        Execute pipeline synchronously with concurrent execution.

        Args:
            executor: Function that executes SQL (takes sql string)
            max_workers: Max concurrent workers (default: 4)
            verbose: Print progress (default: True)

        Returns:
            dict with execution results: {
                "completed": list of completed query IDs,
                "failed": list of (query_id, error) tuples,
                "elapsed_seconds": total execution time,
                "total_queries": total number of queries
            }

        Example:
            def execute_sql(sql: str):
                import duckdb
                conn = duckdb.connect()
                conn.execute(sql)

            result = pipeline.run(executor=execute_sql, max_workers=4)
            print(f"Completed {len(result['completed'])} queries")
        """
        if verbose:
            print(f"🚀 Starting pipeline execution ({len(self.table_graph.queries)} queries)")
            print()

        # Track completed queries
        completed = set()
        failed = []
        start_time = time.time()

        # Group queries by level for concurrent execution
        levels = self._get_execution_levels()

        # Execute level by level
        for level_num, level_queries in enumerate(levels, 1):
            if verbose:
                print(f"📊 Level {level_num}: {len(level_queries)} queries")

            # Execute queries in this level concurrently
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {}

                for query_id in level_queries:
                    query = self.table_graph.queries[query_id]
                    future = pool.submit(executor, query.sql)
                    futures[future] = query_id

                # Wait for completion
                for future in as_completed(futures):
                    query_id = futures[future]

                    try:
                        future.result()
                        completed.add(query_id)

                        if verbose:
                            print(f"  ✅ {query_id}")
                    except Exception as e:
                        failed.append((query_id, str(e)))

                        if verbose:
                            print(f"  ❌ {query_id}: {e}")

            if verbose:
                print()

        elapsed = time.time() - start_time

        # Summary
        if verbose:
            print("=" * 60)
            print(f"✅ Pipeline completed in {elapsed:.2f}s")
            print(f"   Successful: {len(completed)}")
            print(f"   Failed: {len(failed)}")
            if failed:
                print("\n⚠️  Failed queries:")
                for query_id, error in failed:
                    print(f"   - {query_id}: {error}")
            print("=" * 60)

        return {
            "completed": list(completed),
            "failed": failed,
            "elapsed_seconds": elapsed,
            "total_queries": len(self.table_graph.queries),
        }

    async def async_run(
        self,
        executor: Callable[[str], Awaitable[None]],
        max_workers: int = 4,
        verbose: bool = True,
    ) -> dict:
        """
        Execute pipeline asynchronously with concurrent execution.

        Args:
            executor: Async function that executes SQL (takes sql string)
            max_workers: Max concurrent workers (controls semaphore, default: 4)
            verbose: Print progress (default: True)

        Returns:
            dict with execution results: {
                "completed": list of completed query IDs,
                "failed": list of (query_id, error) tuples,
                "elapsed_seconds": total execution time,
                "total_queries": total number of queries
            }

        Example:
            async def execute_sql(sql: str):
                # Your async database connection
                await async_conn.execute(sql)

            result = await pipeline.async_run(executor=execute_sql, max_workers=4)
            print(f"Completed {len(result['completed'])} queries")
        """
        if verbose:
            print(f"🚀 Starting async pipeline execution ({len(self.table_graph.queries)} queries)")
            print()

        # Track completed queries
        completed = set()
        failed = []
        start_time = time.time()

        # Group queries by level for concurrent execution
        levels = self._get_execution_levels()

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_workers)

        # Execute level by level
        for level_num, level_queries in enumerate(levels, 1):
            if verbose:
                print(f"📊 Level {level_num}: {len(level_queries)} queries")

            async def execute_with_semaphore(query_id: str, sql: str):
                """Execute query with semaphore for concurrency control"""
                async with semaphore:
                    try:
                        await executor(sql)
                        completed.add(query_id)
                        if verbose:
                            print(f"  ✅ {query_id}")
                    except Exception as e:
                        failed.append((query_id, str(e)))
                        if verbose:
                            print(f"  ❌ {query_id}: {e}")

            # Execute queries in this level concurrently
            tasks = []
            for query_id in level_queries:
                query = self.table_graph.queries[query_id]
                task = execute_with_semaphore(query_id, query.sql)
                tasks.append(task)

            # Wait for all tasks in this level to complete
            await asyncio.gather(*tasks)

            if verbose:
                print()

        elapsed = time.time() - start_time

        # Summary
        if verbose:
            print("=" * 60)
            print(f"✅ Pipeline completed in {elapsed:.2f}s")
            print(f"   Successful: {len(completed)}")
            print(f"   Failed: {len(failed)}")
            if failed:
                print("\n⚠️  Failed queries:")
                for query_id, error in failed:
                    print(f"   - {query_id}: {error}")
            print("=" * 60)

        return {
            "completed": list(completed),
            "failed": failed,
            "elapsed_seconds": elapsed,
            "total_queries": len(self.table_graph.queries),
        }


__all__ = [
    "PipelineLineageBuilder",
    "Pipeline",
]
