"""
Node and edge factory functions for column lineage graph construction.

Pure factory functions extracted from RecursiveLineageBuilder to enable
easier testing and reuse. Functions that previously used `self.lineage_graph`
accept an explicit `graph: ColumnLineageGraph` parameter instead.
"""

from typing import Any, Dict, Optional

from .metadata_parser import MetadataExtractor
from .models import (
    ColumnEdge,
    ColumnLineageGraph,
    ColumnNode,
    QueryUnit,
    QueryUnitType,
    TVFInfo,
    TVFType,
    ValuesInfo,
)


def get_layer_for_unit(unit: QueryUnit) -> str:
    """Determine layer name for a query unit."""
    if unit.unit_type == QueryUnitType.MAIN_QUERY:
        return "output"
    elif unit.unit_type == QueryUnitType.CTE:
        return "cte"
    else:
        return "subquery"


def get_node_key(unit: QueryUnit, col_info: Dict) -> str:
    """Get cache key for a column node."""
    is_main = unit.unit_type == QueryUnitType.MAIN_QUERY
    table_name = unit.name if unit.name else "__output__"
    col_name = col_info["name"]

    if is_main:
        return f"output.{col_name}"
    else:
        return f"{table_name}.{col_name}"


def find_or_create_star_node(
    graph: ColumnLineageGraph, unit: QueryUnit, source_table: str
) -> ColumnNode:
    """Find or create star node for a query unit."""
    node_key = f"{unit.name}.*"

    if node_key in graph.nodes:
        return graph.nodes[node_key]

    node = ColumnNode(
        layer=get_layer_for_unit(unit),
        table_name=unit.name or unit.unit_id,
        column_name="*",
        full_name=node_key,
        expression="*",
        node_type="star",
        source_expression=None,
        unit_id=unit.unit_id,
        is_star=True,
        star_source_table=source_table,
    )
    graph.add_node(node)

    return node


def resolve_external_table_name(
    external_table_columns: Dict, table_name: str
) -> Optional[str]:
    """
    Resolve a table name to its full qualified version in external_table_columns.

    The parser may return just 'events' while external_table_columns has 'staging.events'.
    This method finds the matching full qualified name.

    Args:
        external_table_columns: Dict mapping full table names to column lists
        table_name: Short or full table name (e.g., 'events' or 'staging.events')

    Returns:
        The full qualified table name if found in external_table_columns, None otherwise
    """
    if table_name in external_table_columns:
        return table_name

    for full_name in external_table_columns:
        if full_name.endswith(f".{table_name}"):
            return full_name

    return None


def find_or_create_table_star_node(
    graph: ColumnLineageGraph, table_name: str
) -> ColumnNode:
    """Find or create star node for a base table."""
    node_key = f"{table_name}.*"

    if node_key in graph.nodes:
        return graph.nodes[node_key]

    node = ColumnNode(
        layer="input",
        table_name=table_name,
        column_name="*",
        full_name=node_key,
        expression="*",
        node_type="star",
        source_expression=None,
        unit_id=None,
        is_star=True,
        star_source_table=table_name,
    )
    graph.add_node(node)

    return node


def find_or_create_table_column_node(
    graph: ColumnLineageGraph, table_name: str, col_name: str
) -> ColumnNode:
    """Find or create column node for a base table."""
    node_key = f"{table_name}.{col_name}"

    if node_key in graph.nodes:
        return graph.nodes[node_key]

    node = ColumnNode(
        layer="input",
        table_name=table_name,
        column_name=col_name,
        full_name=node_key,
        expression=col_name,
        node_type="base_column",
        source_expression=None,
        unit_id=None,
    )
    graph.add_node(node)

    return node


def create_unnest_edge(
    graph: ColumnLineageGraph,
    unit: QueryUnit,
    output_node: ColumnNode,
    col_info: Dict,
    unnest_info: Dict[str, Any],
    col_name: str,
):
    """
    Create an array expansion edge from UNNEST source to output column.

    Args:
        graph: The column lineage graph to add the edge to
        unit: The current query unit
        output_node: The output column node
        col_info: Column info dictionary
        unnest_info: UNNEST metadata from query parser
        col_name: The column name being referenced (may be field of struct)
    """
    source_table = unnest_info.get("source_table")
    source_column = unnest_info.get("source_column")
    expansion_type = unnest_info.get("expansion_type", "unnest")
    offset_alias = unnest_info.get("offset_alias")

    if unnest_info.get("is_offset"):
        actual_unnest_alias = unnest_info.get("unnest_alias")
        if actual_unnest_alias and actual_unnest_alias in unit.unnest_sources:
            actual_unnest_info = unit.unnest_sources[actual_unnest_alias]
            source_table = actual_unnest_info.get("source_table")
            source_column = actual_unnest_info.get("source_column")

    if not source_column:
        return

    actual_source_table = source_table
    if source_table and source_table in unit.alias_mapping:
        actual_source_table, _ = unit.alias_mapping[source_table]
    elif source_table is None:
        if unit.depends_on_tables:
            actual_source_table = unit.depends_on_tables[0]

    if not actual_source_table:
        return

    source_node = find_or_create_table_column_node(graph, actual_source_table, source_column)

    edge = ColumnEdge(
        from_node=source_node,
        to_node=output_node,
        edge_type="array_expansion",
        transformation="array_expansion",
        context=unit.unit_type.value,
        expression=col_info["expression"],
        is_array_expansion=True,
        expansion_type=expansion_type,
        offset_column=offset_alias if unnest_info.get("is_offset") else None,
    )
    graph.add_edge(edge)


def find_or_create_tvf_column_node(
    graph: ColumnLineageGraph, tvf_info: TVFInfo, col_name: str
) -> ColumnNode:
    """Find or create a synthetic column node for TVF output."""
    node_key = f"{tvf_info.alias}.{col_name}"

    if node_key in graph.nodes:
        return graph.nodes[node_key]

    node = ColumnNode(
        layer="input",
        table_name=tvf_info.alias,
        column_name=col_name,
        full_name=node_key,
        expression=f"{tvf_info.function_name}(...)",
        node_type="tvf_synthetic",
        source_expression=None,
        unit_id=None,
        is_synthetic=True,
        synthetic_source=tvf_info.function_name,
        tvf_parameters=tvf_info.parameters,
    )
    graph.add_node(node)

    return node


def create_tvf_edge(
    graph: ColumnLineageGraph,
    unit: QueryUnit,
    output_node: ColumnNode,
    col_info: Dict,
    tvf_info: TVFInfo,
    col_name: str,
):
    """
    Create an edge from TVF synthetic column to output column.

    Args:
        graph: The column lineage graph to add edges to
        unit: The current query unit
        output_node: The output column node
        col_info: Column info dictionary
        tvf_info: TVFInfo metadata from query parser
        col_name: The column name being referenced
    """
    source_node = find_or_create_tvf_column_node(graph, tvf_info, col_name)

    if tvf_info.tvf_type == TVFType.COLUMN_INPUT and tvf_info.input_columns:
        for input_col in tvf_info.input_columns:
            parts = input_col.split(".", 1)
            if len(parts) == 2:
                input_table, input_col_name = parts
                input_node = find_or_create_table_column_node(graph, input_table, input_col_name)
            else:
                input_node = find_or_create_table_column_node(graph, "_input", input_col)

            input_edge = ColumnEdge(
                from_node=input_node,
                to_node=source_node,
                edge_type="tvf_input",
                transformation="tvf_input",
                context=unit.unit_type.value,
                expression=f"{tvf_info.function_name}({input_col})",
                tvf_info=tvf_info,
                is_tvf_output=True,
            )
            graph.add_edge(input_edge)

    edge = ColumnEdge(
        from_node=source_node,
        to_node=output_node,
        edge_type="tvf_output",
        transformation="tvf_output",
        context=unit.unit_type.value,
        expression=col_info["expression"],
        tvf_info=tvf_info,
        is_tvf_output=True,
    )
    graph.add_edge(edge)


def find_or_create_values_column_node(
    graph: ColumnLineageGraph, values_info: ValuesInfo, col_name: str
) -> ColumnNode:
    """Find or create a literal column node for VALUES output."""
    node_key = f"{values_info.alias}.{col_name}"

    if node_key in graph.nodes:
        return graph.nodes[node_key]

    col_idx = (
        values_info.column_names.index(col_name) if col_name in values_info.column_names else -1
    )

    sample_values = None
    literal_type = None
    if col_idx >= 0:
        sample_values = [row[col_idx] for row in values_info.sample_values if col_idx < len(row)]
        if col_idx < len(values_info.column_types):
            literal_type = values_info.column_types[col_idx]

    node = ColumnNode(
        layer="input",
        table_name=values_info.alias,
        column_name=col_name,
        full_name=node_key,
        expression="VALUES(...)",
        node_type="literal",
        source_expression=None,
        unit_id=None,
        is_literal=True,
        literal_values=sample_values,
        literal_type=literal_type,
    )
    graph.add_node(node)

    return node


def create_values_edge(
    graph: ColumnLineageGraph,
    unit: QueryUnit,
    output_node: ColumnNode,
    col_info: Dict,
    values_info: ValuesInfo,
    col_name: str,
):
    """
    Create an edge from VALUES literal column to output column.

    Args:
        graph: The column lineage graph to add the edge to
        unit: The current query unit
        output_node: The output column node
        col_info: Column info dictionary
        values_info: ValuesInfo metadata from query parser
        col_name: The column name being referenced
    """
    source_node = find_or_create_values_column_node(graph, values_info, col_name)

    edge = ColumnEdge(
        from_node=source_node,
        to_node=output_node,
        edge_type="literal_source",
        transformation="literal_source",
        context=unit.unit_type.value,
        expression=col_info["expression"],
    )
    graph.add_edge(edge)


def create_column_node(
    unit: QueryUnit,
    col_info: Dict,
    metadata_extractor: MetadataExtractor,
    is_output: bool = False,
) -> ColumnNode:
    """Create a ColumnNode from column info."""
    layer = (
        "output"
        if is_output and unit.unit_type == QueryUnitType.MAIN_QUERY
        else get_layer_for_unit(unit)
    )
    table_name = unit.name if unit.name else "__output__"
    col_name = col_info["name"]

    full_name = f"{table_name}.{col_name}"
    if is_output and unit.unit_type == QueryUnitType.MAIN_QUERY:
        full_name = f"output.{col_name}"

    sql_metadata = None
    ast_node = col_info.get("ast_node")
    if ast_node is not None:
        sql_metadata = metadata_extractor.extract_from_expression(ast_node)
        if not (
            sql_metadata.description
            or sql_metadata.pii is not None
            or sql_metadata.owner
            or sql_metadata.tags
            or sql_metadata.custom_metadata
        ):
            sql_metadata = None

    node = ColumnNode(
        layer=layer,
        table_name=table_name,
        column_name=col_name,
        full_name=full_name,
        expression=col_info["expression"],
        node_type=col_info["type"],
        source_expression=ast_node,
        unit_id=unit.unit_id,
        is_star=col_info.get("is_star", False),
        star_source_table=col_info.get("source_table"),
        except_columns=col_info.get("except_columns", set()),
        replace_columns=col_info.get("replace_columns", {}),
        sql_metadata=sql_metadata,
    )

    return node
