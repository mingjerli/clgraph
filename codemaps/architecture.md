# Architecture Codemap

> Freshness: 2026-04-11 | Source: 18,205 LOC across 22 Python modules

## Project Identity

**clgraph** (Column-Lineage Graph) - Python 3.10+ library for SQL column lineage and pipeline dependency analysis. Static analysis only (no database required). Built on sqlglot.

## Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  PUBLIC API (__init__.py)                                    │
│  SQLColumnTracer, Pipeline, LineageAgent, Tools              │
├─────────────────────────────────────────────────────────────┤
│  ORCHESTRATION LAYER                                        │
│  pipeline.py (2,795L) | agent.py (608L) | execution.py     │
├─────────────────────────────────────────────────────────────┤
│  INTELLIGENCE LAYER                                         │
│  tools/{lineage,schema,governance,sql,context,base}.py      │
├─────────────────────────────────────────────────────────────┤
│  GRAPH LAYER                                                │
│  column.py | table.py | diff.py | export.py | visualizations│
├─────────────────────────────────────────────────────────────┤
│  CORE ENGINE                                                │
│  lineage_builder.py (3,419L) | query_parser.py (2,354L)    │
├─────────────────────────────────────────────────────────────┤
│  FOUNDATION                                                 │
│  models.py (935L) | multi_query.py (310L)                   │
│  metadata_parser.py                                         │
├─────────────────────────────────────────────────────────────┤
│  INTEGRATIONS                                               │
│  orchestrators/{airflow,dagster,prefect,kestra,mage}.py     │
│  mcp/server.py (Claude Desktop)                             │
└─────────────────────────────────────────────────────────────┘

External: sqlglot, graphviz, jinja2, langchain (opt), mcp (opt)
```

## Module Dependency Flow

```
models.py ← query_parser.py ← lineage_builder.py
                                      ↑
              multi_query.py ──────────┘
                                      ↑
    column.py, table.py, metadata_parser.py
                    ↑
              pipeline.py ← execution.py
                    ↑
    agent.py, tools/*, export.py, visualizations.py, diff.py
                    ↑
          orchestrators/*, mcp/server.py
                    ↑
            __init__.py (public API)
```

## Key Entry Points

| Entry Point | Module | Purpose |
|---|---|---|
| `SQLColumnTracer` | lineage_builder.py | Single-query column lineage |
| `Pipeline` | pipeline.py | Multi-query pipeline analysis |
| `LineageAgent` | agent.py | Natural language lineage queries |
| `create_tool_registry` | tools/__init__.py | Programmatic tool access |
| `python -m clgraph.mcp` | mcp/__main__.py | MCP server for Claude |

## Design Principles

- **Graph-based**: All lineage as ColumnNode/ColumnEdge graphs
- **Recursive descent**: Query units extracted recursively from SQL AST
- **Static analysis**: No database connection needed
- **Multi-dialect**: sqlglot handles dialect differences
- **Optional LLM**: Graceful degradation without langchain
- **Extensible tools**: Custom tools via BaseTool subclass
