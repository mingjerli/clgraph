"""
Pipeline factory functions.

Module-level functions extracted from Pipeline classmethods for
better separation of concerns and testability.

Each function mirrors the corresponding Pipeline classmethod signature
(minus `cls`) and returns a Pipeline instance. Pipeline is imported
locally inside each function to avoid circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .pipeline import Pipeline
    from .table import TableDependencyGraph


def create_from_tuples(
    queries: List[Tuple[str, str]],
    dialect: str = "bigquery",
    template_context: Optional[Dict[str, Any]] = None,
) -> "Pipeline":
    """
    Create Pipeline from list of (query_id, sql) tuples.

    Args:
        queries: List of (query_id, sql) tuples
        dialect: SQL dialect (bigquery, snowflake, etc.)
        template_context: Optional dictionary of template variables

    Returns:
        Pipeline instance
    """
    from .pipeline import Pipeline

    return Pipeline(queries, dialect=dialect, template_context=template_context)


def create_from_dict(
    queries: Dict[str, str],
    dialect: str = "bigquery",
    template_context: Optional[Dict[str, Any]] = None,
) -> "Pipeline":
    """
    Create Pipeline from dictionary of {query_id: sql}.

    Args:
        queries: Dictionary mapping query_id to SQL string
        dialect: SQL dialect (bigquery, snowflake, etc.)
        template_context: Optional dictionary of template variables

    Returns:
        Pipeline instance
    """
    query_list = list(queries.items())
    return create_from_tuples(query_list, dialect=dialect, template_context=template_context)


def generate_query_id(sql: str, dialect: str, id_counts: Dict[str, int]) -> str:
    """
    Generate a meaningful query ID from SQL statement.

    Format priority:
    1. {operation}_{dest_table}
    2. {operation}_{dest_table}_from_{source_table} (if duplicate)
    3. {operation}_{dest_table}_from_{source_table}_2 (if still duplicate)

    Args:
        sql: SQL query string
        dialect: SQL dialect
        id_counts: Dictionary tracking ID usage counts (mutated in place)

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
            id_with_source = f"{base_id}_from_{source_tables[0]}"
            if id_with_source not in id_counts:
                id_counts[id_with_source] = 1
                return id_with_source
            else:
                id_counts[id_with_source] += 1
                return f"{id_with_source}_{id_counts[id_with_source]}"
        else:
            id_counts[base_id] += 1
            return f"{base_id}_{id_counts[base_id]}"

    except (sqlglot.errors.SqlglotError, KeyError, AttributeError):
        base_id = "query"
        if base_id not in id_counts:
            id_counts[base_id] = 1
            return base_id
        else:
            id_counts[base_id] += 1
            return f"{base_id}_{id_counts[base_id]}"


def create_from_sql_list(
    queries: List[str],
    dialect: str = "bigquery",
    template_context: Optional[Dict[str, Any]] = None,
) -> "Pipeline":
    """
    Create Pipeline from list of SQL strings (auto-generates query IDs).

    Query IDs are generated as: {operation}_{table_name}
    If duplicates exist, a number suffix is added: {operation}_{table_name}_2

    Args:
        queries: List of SQL query strings
        dialect: SQL dialect (bigquery, snowflake, etc.)
        template_context: Optional dictionary of template variables

    Returns:
        Pipeline instance
    """
    query_list = []
    id_counts: Dict[str, int] = {}

    for sql in queries:
        query_id = generate_query_id(sql, dialect, id_counts)
        query_list.append((query_id, sql))

    return create_from_tuples(query_list, dialect=dialect, template_context=template_context)


def create_from_sql_string(
    sql: str,
    dialect: str = "bigquery",
    template_context: Optional[Dict[str, Any]] = None,
) -> "Pipeline":
    """
    Create Pipeline from single SQL string with semicolon-separated queries.

    Query IDs are generated as: {operation}_{table_name}
    If duplicates exist, a number suffix is added: {operation}_{table_name}_2

    Args:
        sql: SQL string with multiple queries separated by semicolons
        dialect: SQL dialect (bigquery, snowflake, etc.)
        template_context: Optional dictionary of template variables

    Returns:
        Pipeline instance
    """
    queries = [q.strip() for q in sql.split(";") if q.strip()]
    return create_from_sql_list(queries, dialect=dialect, template_context=template_context)


def create_from_json(
    data: Dict[str, Any],
    apply_metadata: bool = True,
) -> "Pipeline":
    """
    Create Pipeline from JSON data exported by JSONExporter.

    Args:
        data: JSON dictionary from JSONExporter.export() or Pipeline.to_json()
        apply_metadata: Whether to apply metadata (descriptions, PII, etc.)
            from the JSON to the reconstructed pipeline

    Returns:
        Pipeline instance
    """
    from .models import DescriptionSource

    # Validate required fields for round-trip
    if "queries" not in data:
        raise ValueError(
            "JSON data missing 'queries' field. "
            "Ensure JSONExporter.export() was called with include_queries=True"
        )

    if "dialect" not in data:
        raise ValueError("JSON data missing 'dialect' field")

    # Extract pipeline construction data
    dialect = data["dialect"]
    template_context = data.get("template_context")

    # Reconstruct queries list
    queries = [(q["query_id"], q["sql"]) for q in data["queries"]]

    # Create pipeline from queries
    pipeline = create_from_tuples(queries, dialect=dialect, template_context=template_context)

    # Apply metadata if requested
    if apply_metadata and "columns" in data:
        for col_data in data["columns"]:
            full_name = col_data.get("full_name")
            if full_name and full_name in pipeline.columns:
                col = pipeline.columns[full_name]

                if col_data.get("description"):
                    col.description = col_data["description"]
                if col_data.get("description_source"):
                    col.description_source = DescriptionSource(col_data["description_source"])
                if col_data.get("owner"):
                    col.owner = col_data["owner"]
                if col_data.get("pii"):
                    col.pii = col_data["pii"]
                if col_data.get("tags"):
                    col.tags = set(col_data["tags"])
                if col_data.get("custom_metadata"):
                    col.custom_metadata = col_data["custom_metadata"]

    return pipeline


def create_from_json_file(
    file_path: str,
    apply_metadata: bool = True,
) -> "Pipeline":
    """
    Create Pipeline from JSON file exported by JSONExporter.

    Args:
        file_path: Path to JSON file
        apply_metadata: Whether to apply metadata from the JSON

    Returns:
        Pipeline instance
    """
    import json
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(path) as f:
        data = json.load(f)

    return create_from_json(data, apply_metadata=apply_metadata)


def create_from_sql_files(
    sql_dir: str,
    dialect: str = "bigquery",
    pattern: str = "*.sql",
    query_id_from: str = "filename",
    template_context: Optional[Dict[str, Any]] = None,
) -> "Pipeline":
    """
    Create Pipeline from SQL files in a directory.

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
            query_id = sql_file.stem
        elif query_id_from == "comment":
            match = re.match(r"--\s*query_id:\s*(\w+)", sql_content)
            if match:
                query_id = match.group(1)
            else:
                query_id = sql_file.stem
        else:
            raise ValueError(f"Invalid query_id_from: {query_id_from}")

        queries.append((query_id, sql_content))

    return create_from_tuples(queries, dialect=dialect, template_context=template_context)


def create_empty(
    table_graph: "TableDependencyGraph",
) -> "Pipeline":
    """
    Create an empty Pipeline with just a table_graph (for testing).

    This bypasses SQL parsing and creates a minimal Pipeline that can be
    populated manually with columns and edges.

    Args:
        table_graph: Pre-built table dependency graph

    Returns:
        Empty Pipeline instance
    """
    from .column import PipelineLineageGraph
    from .pipeline import Pipeline

    instance = Pipeline.__new__(Pipeline)
    instance.dialect = "bigquery"
    instance.query_mapping = {}
    instance.column_graph = PipelineLineageGraph()
    instance.query_graphs = {}
    instance.llm = None
    instance.table_graph = table_graph
    instance._tracer = None
    instance._validator = None
    instance._metadata_mgr = None
    instance._subpipeline_builder = None
    return instance
