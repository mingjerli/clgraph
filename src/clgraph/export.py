"""
Export functionality for lineage graphs.

Supports exporting to various formats:
- JSON: Machine-readable format for integration
- CSV: Column and table metadata for spreadsheets
- GraphViz DOT: Visual graph representation
"""

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from .pipeline import Pipeline


class JSONExporter:
    """
    Export lineage graph to JSON format for serialization and round-trip loading.

    This is the primary exporter for:
    - Machine-readable integration (columns, edges, tables with full lineage)
    - Round-trip serialization (save pipeline to JSON, reload with Pipeline.from_json)
    - Caching analyzed pipelines for faster subsequent loads

    For human-readable spreadsheet export, see CSVExporter (one-way only).
    """

    @staticmethod
    def export(
        graph: "Pipeline",
        include_metadata: bool = True,
        include_queries: bool = True,
    ) -> Dict[str, Any]:
        """
        Export pipeline lineage graph to JSON-serializable dictionary.

        Args:
            graph: The pipeline lineage graph to export
            include_metadata: Whether to include metadata (descriptions, PII, etc.)
            include_queries: Whether to include original queries for round-trip loading

        Returns:
            Dictionary with columns, edges, tables, and optionally queries/dialect

        Example:
            # Export and reload
            data = JSONExporter.export(pipeline)
            with open("pipeline.json", "w") as f:
                json.dump(data, f)

            # Later, reload the pipeline
            with open("pipeline.json") as f:
                data = json.load(f)
            pipeline = Pipeline.from_json(data)
        """
        result: Dict[str, Any] = {"columns": [], "edges": [], "tables": []}

        # Include queries and dialect for round-trip serialization
        if include_queries:
            result["dialect"] = graph.dialect
            result["template_context"] = graph.template_context
            result["queries"] = []

            # Export queries in order
            for query_id in graph.table_graph.topological_sort():
                if query_id in graph.table_graph.queries:
                    parsed_query = graph.table_graph.queries[query_id]
                    query_dict = {
                        "query_id": query_id,
                        "sql": parsed_query.sql,
                    }
                    # Include original SQL if different (templated)
                    if parsed_query.original_sql and parsed_query.original_sql != parsed_query.sql:
                        query_dict["original_sql"] = parsed_query.original_sql
                        query_dict["template_variables"] = parsed_query.template_variables
                    result["queries"].append(query_dict)

        # Export columns
        for col in graph.columns.values():
            col_dict = {
                "full_name": col.full_name,
                "column_name": col.column_name,
                "table_name": col.table_name,
                "query_id": col.query_id,
                "node_type": col.node_type,
                "expression": col.expression,
                "operation": col.operation,
            }

            # Include synthetic column metadata if present
            if getattr(col, "is_synthetic", False):
                col_dict["is_synthetic"] = True
                col_dict["synthetic_source"] = getattr(col, "synthetic_source", None)
                tvf_params = getattr(col, "tvf_parameters", None)
                if tvf_params:
                    col_dict["tvf_parameters"] = tvf_params

            # Include literal column metadata if present (VALUES clause)
            if getattr(col, "is_literal", False):
                col_dict["is_literal"] = True
                col_dict["literal_type"] = getattr(col, "literal_type", None)
                literal_values = getattr(col, "literal_values", None)
                if literal_values:
                    col_dict["literal_values"] = literal_values

            if include_metadata:
                col_dict.update(
                    {
                        "description": col.description,
                        "description_source": col.description_source.value
                        if col.description_source
                        else None,
                        "owner": col.owner,
                        "pii": col.pii,
                        "tags": list(col.tags),
                        "custom_metadata": col.custom_metadata,
                    }
                )

            result["columns"].append(col_dict)

        # Export edges
        for edge in graph.edges:
            edge_dict = {
                "from_column": edge.from_node.full_name,
                "to_column": edge.to_node.full_name,
                "edge_type": edge.edge_type,
                "transformation": edge.transformation,
                "query_id": edge.query_id,
            }

            # Include JSON extraction metadata if present
            if getattr(edge, "json_path", None):
                edge_dict["json_path"] = edge.json_path
            if getattr(edge, "json_function", None):
                edge_dict["json_function"] = edge.json_function

            # Include array expansion metadata if present
            if getattr(edge, "is_array_expansion", False):
                edge_dict["is_array_expansion"] = True
                edge_dict["expansion_type"] = getattr(edge, "expansion_type", None)
                if getattr(edge, "offset_column", None):
                    edge_dict["offset_column"] = edge.offset_column

            # Include nested access metadata if present
            if getattr(edge, "nested_path", None):
                edge_dict["nested_path"] = edge.nested_path
            if getattr(edge, "access_type", None):
                edge_dict["access_type"] = edge.access_type

            # Include LATERAL correlation metadata if present
            if getattr(edge, "is_lateral_correlation", False):
                edge_dict["is_lateral_correlation"] = True
                edge_dict["lateral_alias"] = getattr(edge, "lateral_alias", None)

            # Include MERGE operation metadata if present
            if getattr(edge, "is_merge_operation", False):
                edge_dict["is_merge_operation"] = True
                edge_dict["merge_action"] = getattr(edge, "merge_action", None)
                edge_dict["merge_condition"] = getattr(edge, "merge_condition", None)

            # Include QUALIFY clause metadata if present
            if getattr(edge, "is_qualify_column", False):
                edge_dict["is_qualify_column"] = True
                edge_dict["qualify_context"] = getattr(edge, "qualify_context", None)
                edge_dict["qualify_function"] = getattr(edge, "qualify_function", None)

            # Include GROUPING SETS/CUBE/ROLLUP metadata if present
            if getattr(edge, "is_grouping_column", False):
                edge_dict["is_grouping_column"] = True
                edge_dict["grouping_type"] = getattr(edge, "grouping_type", None)

            # Include window function metadata if present
            if getattr(edge, "is_window_function", False):
                edge_dict["is_window_function"] = True
                edge_dict["window_role"] = getattr(edge, "window_role", None)
                edge_dict["window_function"] = getattr(edge, "window_function", None)
                edge_dict["window_frame_type"] = getattr(edge, "window_frame_type", None)
                edge_dict["window_frame_start"] = getattr(edge, "window_frame_start", None)
                edge_dict["window_frame_end"] = getattr(edge, "window_frame_end", None)
                if getattr(edge, "window_order_direction", None):
                    edge_dict["window_order_direction"] = edge.window_order_direction
                if getattr(edge, "window_order_nulls", None):
                    edge_dict["window_order_nulls"] = edge.window_order_nulls

            # Include complex aggregate specification if present
            agg_spec = getattr(edge, "aggregate_spec", None)
            if agg_spec is not None:
                edge_dict["aggregate_spec"] = {
                    "function_name": agg_spec.function_name,
                    "aggregate_type": agg_spec.aggregate_type.value,
                    "return_type": agg_spec.return_type,
                    "value_columns": agg_spec.value_columns,
                    "key_columns": agg_spec.key_columns,
                    "distinct": agg_spec.distinct,
                    "order_by": [
                        {
                            "column": o.column,
                            "direction": o.direction,
                            "nulls": o.nulls,
                        }
                        for o in agg_spec.order_by
                    ],
                    "separator": agg_spec.separator,
                }

            # Include TVF edge metadata if present
            if getattr(edge, "is_tvf_output", False):
                edge_dict["is_tvf_output"] = True
                tvf_info = getattr(edge, "tvf_info", None)
                if tvf_info:
                    edge_dict["tvf_info"] = {
                        "function_name": tvf_info.function_name,
                        "tvf_type": tvf_info.tvf_type.value,
                        "alias": tvf_info.alias,
                        "output_columns": tvf_info.output_columns,
                        "parameters": tvf_info.parameters,
                        "external_source": tvf_info.external_source,
                    }

            result["edges"].append(edge_dict)

        # Export tables
        for table in graph.table_graph.tables.values():
            table_dict = {
                "table_name": table.table_name,
                "is_source": table.is_source,
                "created_by": table.created_by,
                "modified_by": table.modified_by,
                "read_by": table.read_by,
                "columns": list(table.columns),
            }

            if include_metadata:
                table_dict["description"] = table.description

            result["tables"].append(table_dict)

        return result

    @staticmethod
    def export_to_file(
        graph: "Pipeline",
        file_path: str,
        include_metadata: bool = True,
        include_queries: bool = True,
        indent: int = 2,
    ):
        """
        Export pipeline lineage graph to JSON file.

        Args:
            graph: The pipeline lineage graph to export
            file_path: Path to output JSON file
            include_metadata: Whether to include metadata
            include_queries: Whether to include queries for round-trip loading
            indent: JSON indentation (default: 2)
        """
        data = JSONExporter.export(
            graph,
            include_metadata=include_metadata,
            include_queries=include_queries,
        )

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=indent)


class CSVExporter:
    """
    Export column and table metadata to CSV format (one-way export only).

    Use this for:
    - Opening metadata in Excel/Google Sheets for review
    - Sharing column inventory with non-technical stakeholders
    - Auditing PII flags and ownership across tables

    Note: This is a flat data dump - lineage relationships are not included.
    For machine-readable export with full lineage, use JSONExporter instead.
    For round-trip serialization (save/reload pipelines), use JSONExporter
    with include_queries=True and Pipeline.from_json().
    """

    @staticmethod
    def export_columns_to_file(graph: "Pipeline", file_path: str):
        """
        Export column metadata to CSV file.

        Args:
            graph: The pipeline lineage graph to export
            file_path: Path to output CSV file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "full_name",
                    "table_name",
                    "column_name",
                    "node_type",
                    "expression",
                    "description",
                    "description_source",
                    "owner",
                    "pii",
                    "tags",
                ]
            )

            # Rows
            for col in sorted(graph.columns.values(), key=lambda c: c.full_name):
                writer.writerow(
                    [
                        col.full_name,
                        col.table_name or "",
                        col.column_name,
                        col.node_type,
                        col.expression or "",
                        col.description or "",
                        col.description_source.value if col.description_source else "",
                        col.owner or "",
                        "Yes" if col.pii else "No",
                        ",".join(sorted(col.tags)) if col.tags else "",
                    ]
                )

    @staticmethod
    def export_tables_to_file(graph: "Pipeline", file_path: str):
        """
        Export table metadata to CSV file.

        Args:
            graph: The pipeline lineage graph to export
            file_path: Path to output CSV file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "table_name",
                    "is_source",
                    "created_by",
                    "column_count",
                    "description",
                ]
            )

            # Rows
            for table in sorted(graph.table_graph.tables.values(), key=lambda t: t.table_name):
                writer.writerow(
                    [
                        table.table_name,
                        "Yes" if table.is_source else "No",
                        table.created_by or "",
                        len(table.columns),
                        table.description or "",
                    ]
                )


__all__ = [
    "JSONExporter",
    "CSVExporter",
]
