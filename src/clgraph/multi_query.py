"""
Multi-query pipeline parser.

Parses multiple SQL queries and builds table dependency graphs.
Includes template tokenization for Jinja2/f-string support.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

from .models import ParsedQuery, SQLOperation
from .table import TableDependencyGraph


class TemplateTokenizer:
    """
    Tokenizes template variables in SQL for parsing, then restores them.

    Strategy:
    1. Replace template expressions ({{ var }}, {var}, etc.) with tokens (__TMPL_N__)
    2. Parse SQL normally (sqlglot sees valid identifiers)
    3. Restore templates in table names
    4. Optionally resolve when context provided

    Supports:
    - Jinja2: {{ var }}, {{ func() }}, {{ obj.prop }}
    - Python f-strings: {var}, {obj.prop}
    - Airflow: {{ ds }}, {{ ds_nodash }}
    - dbt: {{ ref('table') }}, {{ source('schema', 'table') }}
    """

    def __init__(self):
        self.templates: Dict[str, str] = {}  # Token -> Original template
        self.token_counter = 0
        self.context: Optional[Dict] = None

        # Template patterns (ordered by specificity)
        self.patterns = [
            # Jinja2 function calls: {{ ref('table') }}
            (r"\{\{\s*(\w+)\s*\([^)]*\)\s*\}\}", "jinja_func"),
            # Jinja2 nested access: {{ config.project }}
            (r"\{\{\s*([\w.]+)\s*\}\}", "jinja_var"),
            # Python f-string nested access: {obj.prop}
            (r"\{([\w.]+)\}", "f_string"),
        ]

    def tokenize_sql(self, sql: str) -> str:
        """
        Replace all template expressions with tokens.
        Returns tokenized SQL ready for parsing.
        """
        result = sql

        for pattern, _pattern_type in self.patterns:

            def replacer(match):
                template = match.group(0)
                token = f"__TMPL_{self.token_counter}__"
                self.templates[token] = template
                self.token_counter += 1
                return token

            result = re.sub(pattern, replacer, result)

        return result

    def restore_templates(self, text: str) -> str:
        """
        Restore original templates from tokens in text.
        """
        result = text
        for token, template in self.templates.items():
            result = result.replace(token, template)
        return result

    def resolve(self, context: Dict):
        """
        Set context for template resolution.
        Call this before restore_templates to get resolved values.
        """
        self.context = context

        # Resolve all templates
        if context:
            resolved_templates = {}
            for token, template in self.templates.items():
                resolved = self._resolve_template(template, context)
                resolved_templates[token] = resolved
            self.templates = resolved_templates

    def _resolve_template(self, template: str, context: Dict) -> str:
        """Resolve a single template with context"""
        # Try Jinja2 resolution
        try:
            from jinja2 import Template as JinjaTemplate  # type: ignore[import-untyped]

            jinja_template = JinjaTemplate(template)
            return jinja_template.render(**context)
        except Exception:
            # If Jinja2 fails, try f-string style
            try:
                # Simple variable substitution
                for key, value in context.items():
                    template = template.replace(f"{{{key}}}", str(value))
                    template = template.replace(f"{{{{ {key} }}}}", str(value))
                return template
            except Exception:
                # If all fails, return original
                return template


class MultiQueryParser:
    """
    Parses multiple SQL queries and builds table dependency graph.
    Single-query case is just a pipeline with 1 query.
    """

    def __init__(self, dialect: str = "bigquery"):
        self.dialect = dialect

    def parse_queries(
        self,
        queries: List[str],
        template_context: Optional[Dict] = None,
        template_engine: str = "jinja2",
    ) -> TableDependencyGraph:
        """
        Parse multiple SQL queries and build table dependency graph.

        Args:
            queries: List of SQL query strings (may contain templates)
            template_context: Variables for template resolution (optional)
            template_engine: "jinja2", "f-string", or "airflow"

        Returns:
            TableDependencyGraph with all queries and table dependencies
        """
        graph = TableDependencyGraph()

        for i, sql in enumerate(queries):
            # Handle templates
            tokenizer = TemplateTokenizer()
            original_sql = sql
            is_templated = False

            # Check if SQL contains templates
            if "{{" in sql or ("{" in sql and "}" in sql):
                is_templated = True
                # Tokenize templates
                sql = tokenizer.tokenize_sql(sql)

                # Resolve if context provided
                if template_context:
                    tokenizer.resolve(template_context)

            query_id = f"query_{i}"
            parsed = self._parse_single_query(
                query_id, sql, tokenizer, is_templated, original_sql, template_context
            )
            graph.add_query(parsed)

        return graph

    def _parse_single_query(
        self,
        query_id: str,
        sql: str,
        tokenizer: TemplateTokenizer,
        is_templated: bool,
        original_sql: str,
        template_context: Optional[Dict] = None,
    ) -> ParsedQuery:
        """
        Parse a single SQL query and extract metadata.
        """
        ast = sqlglot.parse_one(sql, dialect=self.dialect)

        # Determine operation type and destination table
        operation, destination = self._extract_operation_and_destination(ast, tokenizer)

        # Extract source tables
        sources = self._extract_source_tables(ast, tokenizer)

        # Restore templates in SQL for lineage building
        # Only restore if templates were resolved (template_context was provided)
        # Otherwise keep tokenized SQL to avoid sqlglot parse errors
        if is_templated and template_context:
            # Templates were resolved, safe to restore
            sql_for_lineage = tokenizer.restore_templates(sql)
        else:
            # Either not templated, or templates not resolved - use tokenized SQL
            sql_for_lineage = sql

        return ParsedQuery(
            query_id=query_id,
            sql=sql_for_lineage,
            ast=ast,
            operation=operation,
            destination_table=destination,
            source_tables=sources,
            original_sql=original_sql if is_templated else None,
            is_templated=is_templated,
        )

    def _extract_operation_and_destination(
        self, ast: exp.Expression, tokenizer: TemplateTokenizer
    ) -> Tuple[SQLOperation, Optional[str]]:
        """
        Determine SQL operation type (DDL/DML/DQL) and destination table.
        """
        # DDL: CREATE TABLE / CREATE VIEW
        if isinstance(ast, exp.Create):
            if ast.kind == "TABLE":
                if ast.args.get("replace"):
                    operation = SQLOperation.CREATE_OR_REPLACE_TABLE
                else:
                    operation = SQLOperation.CREATE_TABLE
            else:  # VIEW
                if ast.args.get("replace"):
                    operation = SQLOperation.CREATE_OR_REPLACE_VIEW
                else:
                    operation = SQLOperation.CREATE_VIEW

            # Get table name
            table_node = ast.this
            destination = self._get_table_name(table_node, tokenizer)

        # DML: INSERT / MERGE / UPDATE
        elif isinstance(ast, exp.Insert):
            operation = SQLOperation.INSERT
            destination = self._get_table_name(ast.this, tokenizer)

        elif isinstance(ast, exp.Merge):
            operation = SQLOperation.MERGE
            destination = self._get_table_name(ast.this, tokenizer)

        elif isinstance(ast, exp.Update):
            operation = SQLOperation.UPDATE
            destination = self._get_table_name(ast.this, tokenizer)

        # DQL: SELECT (query-only)
        elif isinstance(ast, exp.Select):
            operation = SQLOperation.SELECT
            destination = None  # No destination table

        else:
            operation = SQLOperation.UNKNOWN
            destination = None

        return operation, destination

    def _extract_source_tables(self, ast: exp.Expression, tokenizer: TemplateTokenizer) -> Set[str]:
        """
        Extract all source tables referenced in the query (excluding destination table).
        """
        tables = set()

        # For CREATE/INSERT/MERGE/UPDATE, the destination table is in ast.this
        # We need to exclude it from source tables
        destination_table = None
        if isinstance(ast, (exp.Create, exp.Insert, exp.Merge, exp.Update)):
            if ast.this:
                destination_table = self._get_table_name(ast.this, tokenizer)

        # Find all Table nodes in the AST
        for table_node in ast.find_all(exp.Table):
            table_name = self._get_table_name(table_node, tokenizer)
            if table_name and table_name != destination_table:
                tables.add(table_name)

        return tables

    def _get_table_name(self, table_node: exp.Table, tokenizer: TemplateTokenizer) -> str:
        """
        Get fully qualified table name from Table node.
        Restores templates if present.

        Note: For CREATE TABLE statements, sqlglot returns a Schema object.
        We extract the Table object from schema.this.
        """
        # Handle Schema objects (from CREATE TABLE)
        if isinstance(table_node, exp.Schema):
            if hasattr(table_node, "this") and table_node.this:
                table_node = table_node.this
            else:
                # Fallback: use the schema's name directly
                return tokenizer.restore_templates(str(table_node.name))

        # Extract table name parts from Table object
        parts = []
        if hasattr(table_node, "catalog") and table_node.catalog:
            parts.append(str(table_node.catalog))
        if hasattr(table_node, "db") and table_node.db:
            parts.append(str(table_node.db))
        if hasattr(table_node, "name"):
            parts.append(str(table_node.name))

        # Join parts and restore templates
        table_name = ".".join(parts) if parts else str(table_node)
        return tokenizer.restore_templates(table_name)


__all__ = ["TemplateTokenizer", "MultiQueryParser"]
