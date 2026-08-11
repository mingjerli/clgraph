"""
Shared context building utilities for lineage tools.

Provides ContextBuilder for creating rich context from Pipeline metadata,
used by SQL generation, schema tools, and other components.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

if TYPE_CHECKING:
    from ..pipeline import Pipeline

# Edge types produced by the lineage builder that are ALWAYS identity-preserving
# regardless of the node's own expression text: "direct_column" (bare,
# unaliased column refs, e.g. `SELECT id FROM t`), "star_passthrough" (SELECT *
# expansion), and "cross_query" (column carried across queries unchanged).
# "expression" is deliberately excluded here: it's the lineage builder's
# catch-all for *any* aliased, non-aggregate projection, so it covers both
# harmless renames (`id AS user_id`) and genuine transforms
# (`UPPER(email) AS x`) alike — see `_is_identity_edge`, which disambiguates
# it by inspecting the destination node's own expression. Any other edge type
# (e.g. "aggregate", "join_predicate", "case", "arithmetic", "window_*",
# "merge_*") fails closed and disqualifies the whole path.
_IDENTITY_EDGE_TYPES = frozenset({"direct_column", "star_passthrough", "cross_query"})


@dataclass
class TableInfo:
    """Information about a table for context building."""

    table_name: str
    description: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    column_descriptions: Dict[str, str] = field(default_factory=dict)
    column_pii: Set[str] = field(default_factory=set)
    column_owners: Dict[str, str] = field(default_factory=dict)
    is_source: bool = False
    created_by: Optional[str] = None
    source_tables: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "table_name": self.table_name,
            "description": self.description,
            "columns": self.columns,
            "column_descriptions": self.column_descriptions,
            "is_source": self.is_source,
            "created_by": self.created_by,
            "source_tables": self.source_tables,
        }


@dataclass
class ContextConfig:
    """Configuration for context building."""

    include_descriptions: bool = True
    """Include table and column descriptions."""

    include_pii_flags: bool = True
    """Mark PII columns in output."""

    include_owners: bool = False
    """Include ownership information."""

    include_lineage: bool = True
    """Include lineage relationships."""

    include_source_tables: bool = True
    """Include source table information for derived tables."""

    max_tables: int = 50
    """Maximum tables to include in context."""

    max_columns_per_table: int = 100
    """Maximum columns per table."""

    max_description_length: int = 200
    """Truncate descriptions longer than this."""

    max_lineage_columns_per_table: int = 10
    """Maximum lineage lines contributed per table."""

    max_lineage_lines: int = 20
    """Maximum total lines in the column-lineage section."""

    annotate_table_roles: bool = True
    """Label tables as source/intermediate/final in schema context."""

    lineage_expansion_depth: int = 2
    """How many ancestor levels expand_with_lineage() walks."""

    max_join_hints: int = 15
    """Maximum lines in the join-hints section (observed joins first)."""


class ContextBuilder:
    """
    Builds rich context from Pipeline metadata.

    Used by multiple tools to create consistent context for LLM prompts
    and structured output. Extracts schema, descriptions, lineage,
    and metadata from the Pipeline.

    Example:
        builder = ContextBuilder(pipeline)

        # Get all tables as structured data
        tables = builder.get_all_tables()

        # Build text context for LLM
        context = builder.build_schema_context()

        # Build context for specific tables
        context = builder.build_context_for_tables(["analytics.revenue"])

        # Expand tables with lineage
        tables = builder.expand_with_lineage(["analytics.revenue"])
    """

    def __init__(self, pipeline: "Pipeline", config: Optional[ContextConfig] = None):
        """
        Initialize ContextBuilder.

        Args:
            pipeline: The clgraph Pipeline to build context from.
            config: Optional configuration for context building.
        """
        self.pipeline = pipeline
        self.config = config or ContextConfig()

    # =========================================================================
    # Structured Data Methods
    # =========================================================================

    def get_all_tables(self) -> List[TableInfo]:
        """
        Get information about all tables in the pipeline.

        Returns:
            List of TableInfo objects for each table.
        """
        tables = []
        for table_name in self.pipeline.table_graph.tables:
            table_info = self.get_table_info(table_name)
            if table_info:
                tables.append(table_info)
        return tables

    def get_table_info(self, table_name: str) -> Optional[TableInfo]:
        """
        Get detailed information about a specific table.

        Args:
            table_name: Name of the table.

        Returns:
            TableInfo object or None if table not found.
        """
        table_node = self.pipeline.table_graph.tables.get(table_name)
        if not table_node:
            return None

        # Get columns for this table
        columns = self.pipeline.get_columns_by_table(table_name)

        # Filter to output columns to avoid duplicates
        output_columns = [c for c in columns if c.layer == "output"]
        if not output_columns:
            output_columns = columns

        # Limit columns if needed
        if len(output_columns) > self.config.max_columns_per_table:
            output_columns = output_columns[: self.config.max_columns_per_table]

        # Build column info
        column_names = [c.column_name for c in output_columns]
        column_descriptions = {}
        column_pii = set()
        column_owners = {}

        for col in output_columns:
            if col.description:
                column_descriptions[col.column_name] = col.description
            if col.pii:
                column_pii.add(col.column_name)
            if col.owner:
                column_owners[col.column_name] = col.owner

        # Get source tables
        source_tables = []
        if table_node.created_by:
            query = self.pipeline.table_graph.queries.get(table_node.created_by)
            if query:
                source_tables = list(query.source_tables)

        return TableInfo(
            table_name=table_name,
            description=table_node.description,
            columns=column_names,
            column_descriptions=column_descriptions,
            column_pii=column_pii,
            column_owners=column_owners,
            is_source=table_node.is_source,
            created_by=table_node.created_by,
            source_tables=source_tables,
        )

    def get_table_names(self, include_sources: bool = True) -> List[str]:
        """
        Get list of all table names.

        Args:
            include_sources: Whether to include source tables.

        Returns:
            List of table names.
        """
        tables = []
        for name, node in self.pipeline.table_graph.tables.items():
            if include_sources or not node.is_source:
                tables.append(name)
        return sorted(tables)

    def table_role(self, table_name: str) -> str:
        """ "source", "final" (no downstream readers), or "intermediate"."""
        node = self.pipeline.table_graph.tables[table_name]
        if node.is_source:
            return "source"
        if len(node.read_by) == 0:
            return "final"
        return "intermediate"

    def get_pii_columns(self, table_name: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get all PII-flagged columns.

        Args:
            table_name: Optional filter by table.

        Returns:
            List of dicts with table, column, description.
        """
        pii_columns = []
        for col in self.pipeline.columns.values():
            if col.pii:
                if table_name is None or col.table_name == table_name:
                    pii_columns.append(
                        {
                            "table": col.table_name,
                            "column": col.column_name,
                            "description": col.description,
                            "owner": col.owner,
                        }
                    )
        return pii_columns

    def get_columns_by_owner(self, owner: str) -> List[Dict[str, str]]:
        """
        Get all columns owned by a specific owner.

        Args:
            owner: Owner name to filter by.

        Returns:
            List of dicts with table, column info.
        """
        columns = []
        for col in self.pipeline.columns.values():
            if col.owner == owner:
                columns.append(
                    {
                        "table": col.table_name,
                        "column": col.column_name,
                        "description": col.description,
                        "pii": col.pii,
                    }
                )
        return columns

    def get_columns_by_tag(self, tag: str) -> List[Dict[str, str]]:
        """
        Get all columns with a specific tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of dicts with table, column info.
        """
        columns = []
        for col in self.pipeline.columns.values():
            if tag in col.tags:
                columns.append(
                    {
                        "table": col.table_name,
                        "column": col.column_name,
                        "description": col.description,
                        "tags": list(col.tags),
                    }
                )
        return columns

    # =========================================================================
    # Lineage Methods
    # =========================================================================

    def expand_with_lineage(self, tables: List[str], depth: Optional[int] = None) -> List[str]:
        """Expand a table list with ancestors via BFS, shallow-first.

        ``depth`` bounds the number of ancestor levels; ``None`` reads
        ``config.lineage_expansion_depth``. Result order: the original
        selection, then depth-1 parents, then depth-2, ...
        """
        if not self.config.include_lineage:
            return list(tables)
        if depth is None:
            depth = self.config.lineage_expansion_depth

        ordered = list(tables)
        seen = set(tables)
        frontier = list(tables)
        for _ in range(depth):
            next_frontier = []
            for table_name in frontier:
                table_node = self.pipeline.table_graph.tables.get(table_name)
                if not table_node or not table_node.created_by:
                    continue
                query = self.pipeline.table_graph.queries.get(table_node.created_by)
                if not query:
                    continue
                for parent in sorted(query.source_tables):
                    if parent not in seen:
                        seen.add(parent)
                        ordered.append(parent)
                        next_frontier.append(parent)
            if not next_frontier:
                break
            frontier = next_frontier
        return ordered

    def get_table_relationships(self, tables: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get relationships between tables.

        Args:
            tables: Optional list to filter by. If None, all tables.

        Returns:
            List of relationship dicts with source, target, type.
        """
        relationships = []
        filter_set = set(tables) if tables else None

        for table_name, table_node in self.pipeline.table_graph.tables.items():
            if filter_set and table_name not in filter_set:
                continue

            if table_node.created_by:
                query = self.pipeline.table_graph.queries.get(table_node.created_by)
                if query:
                    for source_table in query.source_tables:
                        if filter_set is None or source_table in filter_set:
                            relationships.append(
                                {
                                    "source": source_table,
                                    "target": table_name,
                                    "type": "derives_from",
                                    "query_id": table_node.created_by,
                                }
                            )

        return relationships

    def _and_leaves(self, condition):
        """Flatten a boolean condition into AND-connected leaves."""
        if isinstance(condition, exp.And):
            return self._and_leaves(condition.left) + self._and_leaves(condition.right)
        return [condition]

    def _physical_name(self, table: exp.Table) -> str:
        parts = [table.args.get("catalog"), table.args.get("db"), table.this]
        return ".".join(p.name for p in parts if p is not None)

    def get_observed_joins(self, tables: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Equi-join pairs observed in the pipeline's SQL. Never fabricates:
        predicates that cannot be resolved to exactly two physical pipeline
        tables are skipped (and debug-logged)."""
        import logging

        log = logging.getLogger(__name__)
        known = set(self.pipeline.table_graph.tables)
        wanted = set(tables) if tables is not None else None
        results, seen = [], {}

        for query in self.pipeline.table_graph.queries.values():
            cte_names = {cte.alias_or_name for cte in query.ast.find_all(exp.CTE)}
            for select in query.ast.find_all(exp.Select):
                # sqlglot's Select stores its FROM clause under the "from" arg key
                # in most releases, but under "from_" in some (e.g. 30.x) to avoid
                # colliding with the Python keyword. Check both for robustness
                # across the pinned sqlglot range (>=28.0.0,<31.0.0).
                from_expr = select.args.get("from") or select.args.get("from_")
                from_table = from_expr.this if from_expr is not None else None

                # Scope the alias map to THIS select's direct relations only (its
                # FROM and each JOIN's target) — never find_all over the subtree,
                # which would recurse into nested subqueries and let a correlated
                # subquery's reused alias silently overwrite an outer table's
                # mapping (fabricating a join between unrelated tables).
                relations = []
                if from_table is not None:
                    relations.append(from_table)
                for j in select.args.get("joins") or []:
                    relations.append(j.this)
                alias_map = {}
                for rel in relations:
                    if isinstance(rel, exp.Table):
                        name = self._physical_name(rel)
                        alias_map[rel.alias_or_name] = name
                        alias_map[name] = name

                def resolve(alias, cte_names=cte_names, alias_map=alias_map):
                    if alias in cte_names:
                        return None
                    name = alias_map.get(alias)
                    return name if name in known else None

                joins = select.args.get("joins") or []
                for join_index, join in enumerate(joins):
                    pairs = []
                    on = join.args.get("on")
                    using = join.args.get("using")
                    right = join.this if isinstance(join.this, exp.Table) else None
                    if on is not None:
                        for leaf in self._and_leaves(on):
                            if (
                                isinstance(leaf, exp.EQ)
                                and isinstance(leaf.left, exp.Column)
                                and isinstance(leaf.right, exp.Column)
                            ):
                                lt = resolve(leaf.left.table)
                                rt = resolve(leaf.right.table)
                                if lt and rt and lt != rt:
                                    pairs.append((lt, leaf.left.name, rt, leaf.right.name))
                                else:
                                    log.debug(
                                        "skipping unresolvable join predicate: %s", leaf.sql()
                                    )
                    elif using and right is not None:
                        # USING is only safe when the left input is one physical table.
                        if join_index == 0 and isinstance(from_table, exp.Table):
                            lt = resolve(from_table.alias_or_name)
                            rt = resolve(right.alias_or_name)
                            if lt and rt:
                                for ident in using:
                                    pairs.append((lt, ident.name, rt, ident.name))
                        else:
                            log.debug("skipping USING join with composite left input")

                    for lt, lc, rt, rc in pairs:
                        if wanted is not None and (lt not in wanted or rt not in wanted):
                            continue
                        key = tuple(sorted([(lt, lc), (rt, rc)]))
                        existing = seen.get(key)
                        if existing is not None:
                            if query.query_id not in existing["query_ids"]:
                                existing["query_ids"].append(query.query_id)
                            continue
                        entry = {
                            "left_table": lt,
                            "left_column": lc,
                            "right_table": rt,
                            "right_column": rc,
                            "query_id": query.query_id,
                            "query_ids": [query.query_id],
                        }
                        seen[key] = entry
                        results.append(entry)
        return results

    def _ancestor_table_names(self, table_name: str) -> Set[str]:
        """All tables (any depth) that ``table_name`` transitively derives from."""
        ancestors: Set[str] = set()
        frontier = [table_name]
        while frontier:
            next_frontier = []
            for name in frontier:
                node = self.pipeline.table_graph.tables.get(name)
                if not node or not node.created_by:
                    continue
                query = self.pipeline.table_graph.queries.get(node.created_by)
                if not query:
                    continue
                for parent in query.source_tables:
                    if parent not in ancestors:
                        ancestors.add(parent)
                        next_frontier.append(parent)
            frontier = next_frontier
        return ancestors

    def _is_lineage_related(self, table_a: str, table_b: str) -> bool:
        """True if one table transitively derives from the other. Such pairs are
        excluded from candidates: the relationship is already visible via table
        lineage (derives_from), so it isn't a "hidden" join, and treating it as
        one would let the join-hints section reference tables outside the
        tables given to build_join_context."""
        return table_b in self._ancestor_table_names(
            table_a
        ) or table_a in self._ancestor_table_names(table_b)

    def _is_identity_edge(self, edge: Any) -> bool:
        """True if a single lineage edge preserves column identity.

        `direct_column` / `star_passthrough` / `cross_query` always qualify.
        `expression` is the lineage builder's catch-all for any aliased,
        non-aggregate projection — it covers both a harmless rename
        (`id AS user_id`) and a genuine transform (`UPPER(email) AS x`) alike,
        since the builder classifies by the outer alias wrapper rather than
        the inner expression. Disambiguate by parsing the destination node's
        own expression and requiring it reduce, after stripping any alias, to
        a bare column reference. Missing/unparsable expressions and any other
        edge type fail closed.
        """
        if edge.edge_type in _IDENTITY_EDGE_TYPES:
            return True
        if edge.edge_type != "expression":
            return False
        expr = edge.to_node.expression
        if not expr:
            return False
        try:
            parsed = sqlglot.parse_one(expr, dialect=self.pipeline.dialect)
        except Exception:
            return False
        return isinstance(parsed.unalias(), exp.Column)

    def _identity_join_candidates(self, tables: List[str]) -> List[Dict[str, str]]:
        """Join candidates from shared ultimate sources, restricted to columns
        whose entire backward path is identity-preserving. Unknown edge types
        fail closed."""
        by_source: Dict[str, List[Tuple[str, str]]] = {}
        source_tables: Dict[str, str] = {}
        for table_name in tables:
            for col in self.pipeline.get_columns_by_table(table_name):
                if col.layer != "output":
                    continue
                _nodes, edges = self.pipeline.trace_column_backward_full(
                    table_name, col.column_name
                )
                if not edges:
                    continue
                if any(not self._is_identity_edge(e) for e in edges):
                    continue
                for leaf in self.pipeline.trace_column_backward(table_name, col.column_name):
                    if leaf.table_name != table_name:
                        by_source.setdefault(leaf.full_name, []).append(
                            (table_name, col.column_name)
                        )
                        source_tables[leaf.full_name] = leaf.table_name

        candidates = []
        for source_name, endpoints in sorted(by_source.items()):
            per_table = {}
            for table_name, column_name in endpoints:
                per_table.setdefault(table_name, column_name)
            table_names = sorted(per_table)
            for i, left in enumerate(table_names):
                for right in table_names[i + 1 :]:
                    if self._is_lineage_related(left, right):
                        continue
                    candidates.append(
                        {
                            "left_table": left,
                            "left_column": per_table[left],
                            "right_table": right,
                            "right_column": per_table[right],
                            "source": source_name,
                            "source_table": source_tables[source_name],
                        }
                    )
        return candidates

    def build_join_context(self, tables: List[str]) -> str:
        """Join-hints prompt section: observed joins first, then candidates."""
        observed = self.get_observed_joins(tables)[: self.config.max_join_hints]
        observed_keys = {
            tuple(
                sorted([(j["left_table"], j["left_column"]), (j["right_table"], j["right_column"])])
            )
            for j in observed
        }
        lines = []
        for j in observed:
            observed_in = ", ".join(j["query_ids"][:3])
            lines.append(
                f"- {j['left_table']}.{j['left_column']} = "
                f"{j['right_table']}.{j['right_column']} (observed in {observed_in})"
            )
        for c in self._identity_join_candidates(tables):
            if len(lines) >= self.config.max_join_hints:
                break
            key = tuple(
                sorted([(c["left_table"], c["left_column"]), (c["right_table"], c["right_column"])])
            )
            if key in observed_keys:
                continue
            if c["source_table"] in tables:
                explanation = f"both derive from {c['source']}"
            else:
                explanation = "shared upstream key"
            lines.append(
                f"- candidate: {c['left_table']}.{c['left_column']} = "
                f"{c['right_table']}.{c['right_column']} ({explanation})"
            )
        if not lines:
            return ""
        return "\n".join(["## Join Hints", ""] + lines)

    # =========================================================================
    # Text Context Methods (for LLM prompts)
    # =========================================================================

    def resolve_context_tables(self, tables: Optional[List[str]] = None) -> List[str]:
        """Ordered, capped table list — the single source of truth for a prompt.

        With no explicit selection, all tables are considered and ranked by role
        (final > intermediate > source) when trimming. An explicit selection
        keeps its order and is truncated to ``max_tables``.
        """
        if tables is None:
            tables = self.get_table_names()
            priority = {"final": 0, "intermediate": 1, "source": 2}
            tables = sorted(tables, key=lambda t: (priority[self.table_role(t)], t))
        return list(tables)[: self.config.max_tables]

    def build_schema_context(self, tables: Optional[List[str]] = None) -> str:
        return self.build_context_for_tables(self.resolve_context_tables(tables))

    def build_context_for_tables(self, tables: List[str]) -> str:
        """
        Build text context for specific tables.

        Args:
            tables: List of table names to include.

        Returns:
            Formatted string describing the tables.
        """
        context_parts = []

        for table_name in sorted(tables):
            table_context = self._format_table_context(table_name)
            if table_context:
                context_parts.append(table_context)

        return "\n\n".join(context_parts)

    def _format_table_context(self, table_name: str) -> Optional[str]:
        """Format a single table for text context."""
        table_info = self.get_table_info(table_name)
        if not table_info:
            return None

        lines = [f"### {table_name}"]

        # Table description
        if self.config.include_descriptions and table_info.description:
            desc = self._truncate(table_info.description)
            lines.append(f"Description: {desc}")

        # Table type
        if self.config.annotate_table_roles:
            role = self.table_role(table_name)
            lines.append(
                {"source": "(Source table)", "final": "(Final table)"}.get(
                    role, "(Intermediate table)"
                )
            )
        if (
            not table_info.is_source
            and table_info.source_tables
            and self.config.include_source_tables
        ):
            sources = ", ".join(table_info.source_tables[:3])
            if len(table_info.source_tables) > 3:
                sources += f" (+{len(table_info.source_tables) - 3} more)"
            lines.append(f"Sources: {sources}")

        lines.append("")
        lines.append("Columns:")

        # Columns
        for col_name in table_info.columns:
            col_line = f"  - {col_name}"

            if self.config.include_descriptions:
                desc = table_info.column_descriptions.get(col_name)
                if desc:
                    desc = self._truncate(desc, 100)
                    col_line += f": {desc}"

            if self.config.include_pii_flags and col_name in table_info.column_pii:
                col_line += " [PII]"

            if self.config.include_owners:
                owner = table_info.column_owners.get(col_name)
                if owner:
                    col_line += f" (owner: {owner})"

            lines.append(col_line)

        return "\n".join(lines)

    def build_relationship_context(self, tables: Optional[List[str]] = None) -> str:
        """
        Build text context describing table relationships.

        Args:
            tables: Optional list to filter by.

        Returns:
            Formatted string describing relationships.
        """
        relationships = self.get_table_relationships(tables)
        if not relationships:
            return ""

        lines = ["## Table Relationships", ""]
        for rel in relationships:
            lines.append(f"- {rel['target']} is derived from {rel['source']}")

        return "\n".join(lines)

    def build_lineage_context(self, tables: List[str]) -> str:
        """
        Build text context describing column lineage.

        Args:
            tables: Tables to include lineage for.

        Returns:
            Formatted string describing column lineage.
        """
        if not self.config.include_lineage:
            return ""

        lineage_info = []

        for table_name in tables:
            columns = self.pipeline.get_columns_by_table(table_name)
            output_columns = [c for c in columns if c.layer == "output"]

            for col in output_columns[: self.config.max_lineage_columns_per_table]:
                sources = self.pipeline.trace_column_backward(table_name, col.column_name)
                relevant_sources = [s for s in sources if s.table_name in tables]

                if relevant_sources and relevant_sources[0].table_name != table_name:
                    source_str = ", ".join(
                        f"{s.table_name}.{s.column_name}" for s in relevant_sources[:3]
                    )
                    lineage_info.append(f"- {table_name}.{col.column_name} <- {source_str}")

        if not lineage_info:
            return ""

        return "## Column Lineage\n\n" + "\n".join(lineage_info[: self.config.max_lineage_lines])

    # =========================================================================
    # Table Selection (for two-stage approaches)
    # =========================================================================

    def select_tables_by_keywords(
        self, question: str, min_tables: int = 3, max_tables: int = 10
    ) -> List[str]:
        """
        Select relevant tables based on keyword matching.

        Simple heuristic for table selection without LLM.

        Args:
            question: Natural language question.
            min_tables: Minimum tables to return.
            max_tables: Maximum tables to return.

        Returns:
            List of relevant table names.
        """
        question_lower = question.lower()
        words = set(re.findall(r"\w+", question_lower))

        scored_tables = []

        for table_name in self.pipeline.table_graph.tables:
            score = 0

            # Check table name
            table_words = set(re.findall(r"\w+", table_name.lower()))
            score += len(words & table_words) * 2

            # Check column names and descriptions
            for col in self.pipeline.get_columns_by_table(table_name):
                col_words = set(re.findall(r"\w+", col.column_name.lower()))
                if words & col_words:
                    score += 1

                if col.description:
                    desc_words = set(re.findall(r"\w+", col.description.lower()))
                    if words & desc_words:
                        score += 0.5

            if score > 0:
                scored_tables.append((table_name, score))

        # Sort by score
        scored_tables.sort(key=lambda x: x[1], reverse=True)
        selected = [t[0] for t in scored_tables[:max_tables]]

        # Ensure minimum
        if len(selected) < min_tables:
            all_tables = list(self.pipeline.table_graph.tables.keys())
            for table in all_tables:
                if table not in selected:
                    selected.append(table)
                    if len(selected) >= min_tables:
                        break

        return selected

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _truncate(self, text: str, max_length: Optional[int] = None) -> str:
        """Truncate text to max length."""
        max_len = max_length or self.config.max_description_length
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."


__all__ = [
    "TableInfo",
    "ContextConfig",
    "ContextBuilder",
]
