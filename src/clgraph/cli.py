"""
clgraph CLI — Column lineage analysis from the terminal.

Usage:
    clgraph analyze ./sql/ --dialect bigquery
    clgraph analyze query.sql --format json
    clgraph analyze query.sql --format dot
    clgraph diff old/ new/ --dialect bigquery
    clgraph mcp --pipeline ./sql/
"""

import json
from pathlib import Path

import typer
from typing_extensions import Annotated

app = typer.Typer(
    name="clgraph",
    help="Column lineage and pipeline dependency analysis for SQL.",
    no_args_is_help=True,
)


def _load_pipeline(path: Path, dialect: str):
    """Load a Pipeline from a file or directory."""
    from clgraph import Pipeline

    if path.is_dir():
        return Pipeline.from_sql_files(str(path), dialect=dialect)
    elif path.suffix == ".sql":
        sql = path.read_text()
        return Pipeline.from_sql_string(sql, dialect=dialect)
    elif path.suffix == ".json":
        return Pipeline.from_json_file(str(path))
    else:
        typer.echo(f"Error: unsupported file type: {path.suffix}", err=True)
        raise typer.Exit(code=1)


def _print_table_summary(pipeline):
    """Print a Rich table summary of the pipeline."""
    import sys

    from rich.console import Console
    from rich.table import Table

    console = Console(file=sys.stdout)

    summary = Table(title="Pipeline Tables")
    summary.add_column("Table", style="cyan")
    summary.add_column("Type", style="green")
    summary.add_column("Columns", justify="right")
    summary.add_column("Upstream", justify="right")
    summary.add_column("Downstream", justify="right")

    for table_name, table_node in pipeline.table_graph.tables.items():
        col_count = len(list(pipeline.get_columns_by_table(table_name)))
        upstream = pipeline.table_graph.get_dependencies(table_name)
        downstream = pipeline.table_graph.get_downstream(table_name)
        kind = "source" if table_node.is_source else "derived"

        summary.add_row(
            table_name,
            kind,
            str(col_count),
            str(len(upstream)),
            str(len(downstream)),
        )

    console.print(summary)

    total_tables = len(pipeline.table_graph.tables)
    total_cols = len(pipeline.columns)
    total_edges = len(pipeline.edges)
    console.print(
        f"\n[bold]{total_tables}[/bold] tables, "
        f"[bold]{total_cols}[/bold] columns, "
        f"[bold]{total_edges}[/bold] lineage edges"
    )

    issues = pipeline.get_all_issues()
    if issues:
        errors = [i for i in issues if i.severity.value == "error"]
        warnings = [i for i in issues if i.severity.value == "warning"]
        if errors:
            console.print(f"[red]{len(errors)} errors[/red]")
        if warnings:
            console.print(f"[yellow]{len(warnings)} warnings[/yellow]")


def _print_json_summary(pipeline):
    """Print pipeline summary as JSON."""
    tables = []
    for table_name, table_node in pipeline.table_graph.tables.items():
        columns = [
            {
                "name": col.column_name,
                "type": col.node_type,
                "pii": col.pii,
            }
            for col in pipeline.get_columns_by_table(table_name)
        ]
        tables.append(
            {
                "name": table_name,
                "is_source": table_node.is_source,
                "columns": columns,
            }
        )

    output = {
        "dialect": pipeline.dialect,
        "tables": tables,
        "columns": len(pipeline.columns),
        "edges": len(pipeline.edges),
        "issues": len(pipeline.get_all_issues()),
    }
    typer.echo(json.dumps(output, indent=2, default=str))


def _print_dot(pipeline):
    """Print pipeline lineage as Graphviz DOT."""
    from clgraph.visualizations import visualize_table_dependencies

    dot = visualize_table_dependencies(pipeline.table_graph)
    typer.echo(dot.source)


@app.command()
def analyze(
    path: Annotated[
        Path,
        typer.Argument(help="Path to SQL file, directory, or JSON pipeline file"),
    ],
    dialect: Annotated[
        str,
        typer.Option(help="SQL dialect"),
    ] = "bigquery",
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: table, json, dot"),
    ] = "table",
):
    """Analyze SQL files and display column lineage summary."""
    if not path.exists():
        typer.echo(f"Error: path does not exist: {path}", err=True)
        raise typer.Exit(code=1)

    pipeline = _load_pipeline(path, dialect)

    if output_format == "json":
        _print_json_summary(pipeline)
    elif output_format == "dot":
        _print_dot(pipeline)
    else:
        _print_table_summary(pipeline)


@app.command()
def diff(
    old_path: Annotated[
        Path,
        typer.Argument(help="Path to old SQL file or directory"),
    ],
    new_path: Annotated[
        Path,
        typer.Argument(help="Path to new SQL file or directory"),
    ],
    dialect: Annotated[
        str,
        typer.Option(help="SQL dialect"),
    ] = "bigquery",
    output_format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: table, json"),
    ] = "table",
):
    """Compare lineage between two SQL pipeline versions."""
    if not old_path.exists():
        typer.echo(f"Error: path does not exist: {old_path}", err=True)
        raise typer.Exit(code=1)
    if not new_path.exists():
        typer.echo(f"Error: path does not exist: {new_path}", err=True)
        raise typer.Exit(code=1)

    old_pipeline = _load_pipeline(old_path, dialect)
    new_pipeline = _load_pipeline(new_path, dialect)

    diff_result = new_pipeline.diff(old_pipeline)

    if output_format == "json":
        output = {
            "columns_added": diff_result.columns_added,
            "columns_removed": diff_result.columns_removed,
            "columns_modified": [
                {
                    "column": cd.full_name,
                    "field": cd.field_name,
                    "old_value": cd.old_value,
                    "new_value": cd.new_value,
                }
                for cd in diff_result.columns_modified
            ],
            "has_changes": diff_result.has_changes(),
        }
        typer.echo(json.dumps(output, indent=2, default=str))
    else:
        _print_diff_summary(diff_result)


def _print_diff_summary(diff_result):
    """Print diff summary as a Rich table."""
    import sys

    from rich.console import Console

    console = Console(file=sys.stdout)

    added = diff_result.columns_added
    removed = diff_result.columns_removed
    modified = diff_result.columns_modified

    if not added and not removed and not modified:
        console.print("[green]No lineage changes detected.[/green]")
        return

    if added:
        console.print(f"\n[green]+{len(added)} columns added[/green]")
        for col in added:
            console.print(f"  [green]+ {col}[/green]")

    if removed:
        console.print(f"\n[red]-{len(removed)} columns removed[/red]")
        for col in removed:
            console.print(f"  [red]- {col}[/red]")

    if modified:
        console.print(f"\n[yellow]~{len(modified)} columns modified[/yellow]")
        for cd in modified:
            console.print(f"  [yellow]~ {cd.full_name} ({cd.field_name})[/yellow]")


@app.command()
def mcp(
    pipeline: Annotated[
        Path,
        typer.Option("--pipeline", "-p", help="Path to SQL files directory or JSON pipeline file"),
    ],
    dialect: Annotated[
        str,
        typer.Option(help="SQL dialect"),
    ] = "bigquery",
    transport: Annotated[
        str,
        typer.Option(help="Transport type: stdio, http"),
    ] = "stdio",
    no_llm_tools: Annotated[
        bool,
        typer.Option("--no-llm-tools", help="Exclude LLM-dependent tools"),
    ] = False,
):
    """Start MCP server for LLM integration (Claude Desktop, etc.).

    Requires the mcp extra: pip install clgraph[mcp]
    """
    if not pipeline.exists():
        typer.echo(f"Error: path does not exist: {pipeline}", err=True)
        raise typer.Exit(code=1)

    loaded = _load_pipeline(pipeline, dialect)

    try:
        from clgraph.mcp import run_mcp_server
    except ImportError as err:
        typer.echo(
            "Error: MCP dependencies not installed. Install with: pip install clgraph[mcp]",
            err=True,
        )
        raise typer.Exit(code=1) from err

    run_mcp_server(
        loaded,
        llm=None,
        include_llm_tools=not no_llm_tools,
        transport=transport,
    )


if __name__ == "__main__":
    app()
