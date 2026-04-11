"""
Trace dependency strategies for different SQL constructs.

Each function handles one branch of column dependency tracing.
Resolution methods are passed as callables to avoid coupling to the builder class.
"""

from typing import Callable, Dict, List

from .aggregate_parser import (
    parse_aggregate_spec,
    unit_has_fully_resolved_columns,
)
from .models import (
    ColumnEdge,
    ColumnLineageGraph,
    ColumnNode,
    QueryUnit,
    QueryUnitGraph,
)
from .node_factory import (
    create_tvf_edge,
    create_unnest_edge,
    create_values_edge,
    find_or_create_star_node,
    find_or_create_table_column_node,
    find_or_create_table_star_node,
    resolve_external_table_name,
)


def trace_star_passthrough(
    graph: ColumnLineageGraph,
    unit: QueryUnit,
    output_node: ColumnNode,
    col_info: Dict,
    unit_graph: QueryUnitGraph,
) -> None:
    """Handle star column passthrough (SELECT * FROM ...)."""
    source_unit_id = col_info.get("source_unit")
    source_table = col_info.get("source_table")

    if source_unit_id:
        # Star from another query unit (CTE or subquery)
        source_unit = unit_graph.units[source_unit_id]
        source_star_node = find_or_create_star_node(
            graph, source_unit, source_table or source_unit.name or source_unit.unit_id
        )

        edge = ColumnEdge(
            from_node=source_star_node,
            to_node=output_node,
            edge_type="star_passthrough",
            transformation="star_passthrough",
            context=unit.unit_type.value,
            expression=col_info["expression"],
        )
        graph.add_edge(edge)

    elif source_table:
        # Star from base table
        table_star_node = find_or_create_table_star_node(graph, source_table)

        edge = ColumnEdge(
            from_node=table_star_node,
            to_node=output_node,
            edge_type="star_passthrough",
            transformation="star_passthrough",
            context=unit.unit_type.value,
            expression=col_info["expression"],
        )
        graph.add_edge(edge)


def trace_aggregate_star(
    graph: ColumnLineageGraph,
    unit: QueryUnit,
    output_node: ColumnNode,
    col_info: Dict,
    unit_columns_cache: Dict[str, List[ColumnNode]],
    external_table_columns: Dict[str, List[str]],
    unit_graph: QueryUnitGraph,
) -> None:
    """Handle COUNT(*) and ambiguous aggregates."""
    # Add warning only if multiple tables are involved (ambiguous case)
    has_multiple_sources = len(unit.depends_on_tables) + len(unit.depends_on_units) > 1
    if has_multiple_sources:
        warning = (
            f"Ambiguous lineage: {col_info['expression']} uses * with multiple sources."
        )
        output_node.warnings.append(warning)

    # Link to ALL tables involved.
    # For base tables: use individual columns if schema is known, else use *
    for table_name in unit.depends_on_tables:
        resolved_table = resolve_external_table_name(external_table_columns, table_name)
        if resolved_table:
            # Schema is known — link to individual columns
            column_names = external_table_columns[resolved_table]
            for col_name in column_names:
                source_node = find_or_create_table_column_node(
                    graph, resolved_table, col_name
                )
                edge = ColumnEdge(
                    from_node=source_node,
                    to_node=output_node,
                    edge_type="aggregate",
                    transformation="ambiguous_aggregate",
                    context=unit.unit_type.value,
                    expression=col_info["expression"],
                )
                graph.add_edge(edge)
        else:
            # Schema unknown — use star node
            table_star_node = find_or_create_table_star_node(graph, table_name)
            edge = ColumnEdge(
                from_node=table_star_node,
                to_node=output_node,
                edge_type="aggregate",
                transformation="ambiguous_aggregate",
                context=unit.unit_type.value,
                expression=col_info["expression"],
            )
            graph.add_edge(edge)

    # For query units: link to all explicit columns if fully resolved, else use *
    for unit_id in unit.depends_on_units:
        dep_unit = unit_graph.units[unit_id]

        # Check if this unit has fully resolved columns (no stars in its SELECT)
        if unit_has_fully_resolved_columns(dep_unit):
            # Link to ALL explicit columns from this unit
            if unit_id in unit_columns_cache:
                for col_node in unit_columns_cache[unit_id]:
                    if not col_node.is_star:  # Skip any * nodes
                        edge = ColumnEdge(
                            from_node=col_node,
                            to_node=output_node,
                            edge_type="aggregate",
                            transformation="ambiguous_aggregate",
                            context=unit.unit_type.value,
                            expression=col_info["expression"],
                        )
                        graph.add_edge(edge)
        else:
            # Unit has unresolved columns (has * in SELECT), use * node
            unit_star_node = find_or_create_star_node(
                graph, dep_unit, dep_unit.name or dep_unit.unit_id
            )
            edge = ColumnEdge(
                from_node=unit_star_node,
                to_node=output_node,
                edge_type="aggregate",
                transformation="ambiguous_aggregate",
                context=unit.unit_type.value,
                expression=col_info["expression"],
            )
            graph.add_edge(edge)


def trace_set_operation(
    graph: ColumnLineageGraph,
    unit: QueryUnit,
    output_node: ColumnNode,
    col_info: Dict,
    unit_columns_cache: Dict[str, List[ColumnNode]],
) -> None:
    """Handle UNION/INTERSECT/EXCEPT columns."""
    source_branches = col_info["source_branches"]
    col_index = col_info.get("index", 0)

    # Create edges from each branch's corresponding column
    for branch_id in source_branches:
        if branch_id in unit_columns_cache:
            branch_cols = unit_columns_cache[branch_id]
            # Get the column at the same position in the branch
            if col_index < len(branch_cols):
                branch_col_node = branch_cols[col_index]
                edge = ColumnEdge(
                    from_node=branch_col_node,
                    to_node=output_node,
                    edge_type="union",
                    transformation=unit.set_operation_type or "union",
                    context=unit.unit_type.value,
                    expression=col_info["expression"],
                )
                graph.add_edge(edge)


def trace_merge_columns(
    graph: ColumnLineageGraph,
    unit: QueryUnit,
    output_node: ColumnNode,
    col_info: Dict,
    resolve_source_unit: Callable,
    resolve_base_table_name: Callable,
    find_column_in_unit: Callable,
) -> None:
    """Handle MERGE statement columns."""
    merge_action = col_info.get("merge_action", col_info.get("type"))
    merge_condition = col_info.get("merge_condition")
    source_refs = col_info.get("source_columns", [])

    for source_ref in source_refs:
        table_ref, col_name = source_ref[:2]

        # Try to resolve as a source unit or base table
        source_node = None
        source_unit = resolve_source_unit(unit, table_ref) if table_ref else None
        if source_unit:
            source_node = find_column_in_unit(source_unit, col_name)
        if not source_node:
            # Try as base table
            base_table = (
                resolve_base_table_name(unit, table_ref) if table_ref else None
            )
            if base_table:
                source_node = find_or_create_table_column_node(
                    graph, base_table, col_name
                )
            elif table_ref:
                # Fallback: use table_ref directly
                source_node = find_or_create_table_column_node(
                    graph, table_ref, col_name
                )

        if source_node:
            edge = ColumnEdge(
                from_node=source_node,
                to_node=output_node,
                edge_type=col_info["type"],
                transformation=col_info["type"],
                context=unit.unit_type.value,
                expression=col_info["expression"],
                is_merge_operation=True,
                merge_action=merge_action,
                merge_condition=merge_condition,
            )
            graph.add_edge(edge)


def trace_regular_columns(
    graph: ColumnLineageGraph,
    unit: QueryUnit,
    output_node: ColumnNode,
    col_info: Dict,
    source_columns: List,
    resolve_source_unit: Callable,
    resolve_base_table_name: Callable,
    find_column_in_unit: Callable,
    get_default_from_table: Callable,
) -> None:
    """Handle regular column references with UNNEST, TVF, VALUES sub-cases."""
    for source_ref in source_columns:
        # Unpack source reference (now includes JSON and nested access metadata)
        # Handle different tuple formats for backward compatibility
        if len(source_ref) >= 6:
            table_ref, col_name, json_path, json_function, nested_path, access_type = (
                source_ref
            )
        elif len(source_ref) == 4:
            table_ref, col_name, json_path, json_function = source_ref
            nested_path, access_type = None, None
        else:
            table_ref, col_name = source_ref[:2]
            json_path, json_function = None, None
            nested_path, access_type = None, None

        # Resolve table_ref to either a query unit or base table
        # If table_ref is None, try to infer the default source
        effective_table_ref = table_ref

        # Check if this is an UNNEST alias (check both table_ref and col_name)
        # Case 1: Qualified reference like unnest_alias.field (table_ref = unnest_alias)
        # Case 2: Unqualified reference where column name is the UNNEST alias itself
        unnest_info = None
        if table_ref and table_ref in unit.unnest_sources:
            unnest_info = unit.unnest_sources[table_ref]
        elif not table_ref and col_name in unit.unnest_sources:
            # Unqualified column name matches UNNEST alias
            unnest_info = unit.unnest_sources[col_name]

        if not table_ref:
            # No explicit table reference — infer from FROM clause
            default_table = get_default_from_table(unit)
            if default_table:
                effective_table_ref = default_table

        if unnest_info:
            # This is a reference to an UNNEST result
            create_unnest_edge(
                graph,
                unit=unit,
                output_node=output_node,
                col_info=col_info,
                unnest_info=unnest_info,
                col_name=col_name,
            )
            continue

        # Check if this is a TVF alias (Table-Valued Function)
        tvf_info = None
        if table_ref and table_ref in unit.tvf_sources:
            tvf_info = unit.tvf_sources[table_ref]
        elif not table_ref and col_name in unit.tvf_sources:
            # Unqualified reference where column name matches TVF alias
            tvf_info = unit.tvf_sources[col_name]
        elif not table_ref:
            # Check if unqualified column is a TVF output column
            for _alias, tvf in unit.tvf_sources.items():
                if col_name in tvf.output_columns:
                    tvf_info = tvf
                    break

        if tvf_info:
            # This is a reference to a TVF output — create synthetic edge
            create_tvf_edge(
                graph,
                unit=unit,
                output_node=output_node,
                col_info=col_info,
                tvf_info=tvf_info,
                col_name=col_name,
            )
            continue

        # Check if this is a VALUES alias (literal table)
        values_info = None
        if table_ref and table_ref in unit.values_sources:
            values_info = unit.values_sources[table_ref]
        elif not table_ref:
            # Check if unqualified column is a VALUES output column
            for _alias, vals in unit.values_sources.items():
                if col_name in vals.column_names:
                    values_info = vals
                    break

        if values_info:
            # This is a reference to a VALUES output — create literal edge
            create_values_edge(
                graph,
                unit=unit,
                output_node=output_node,
                col_info=col_info,
                values_info=values_info,
                col_name=col_name,
            )
            continue

        source_unit = (
            resolve_source_unit(unit, effective_table_ref)
            if effective_table_ref
            else None
        )

        # Parse aggregate spec if this is an aggregate edge
        aggregate_spec = None
        if col_info["type"] == "aggregate":
            aggregate_spec = parse_aggregate_spec(col_info.get("ast_node"))

        if source_unit:
            # Reference to another query unit (CTE or subquery)
            source_node = find_column_in_unit(source_unit, col_name)

            if source_node:
                edge = ColumnEdge(
                    from_node=source_node,
                    to_node=output_node,
                    edge_type=col_info["type"],
                    transformation=col_info["type"],
                    context=unit.unit_type.value,
                    expression=col_info["expression"],
                    json_path=json_path,
                    json_function=json_function,
                    nested_path=nested_path,
                    access_type=access_type,
                    aggregate_spec=aggregate_spec,
                )
                graph.add_edge(edge)

        else:
            # Reference to base table — resolve alias to actual table name
            base_table = (
                resolve_base_table_name(unit, effective_table_ref)
                if effective_table_ref
                else None
            )
            if base_table:
                source_node = find_or_create_table_column_node(graph, base_table, col_name)

                edge = ColumnEdge(
                    from_node=source_node,
                    to_node=output_node,
                    edge_type=col_info["type"],
                    transformation=col_info["type"],
                    context=unit.unit_type.value,
                    expression=col_info["expression"],
                    json_path=json_path,
                    json_function=json_function,
                    nested_path=nested_path,
                    access_type=access_type,
                    aggregate_spec=aggregate_spec,
                )
                graph.add_edge(edge)
