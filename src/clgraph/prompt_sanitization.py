"""
Prompt sanitization module for LLM prompt injection mitigation.

This module provides functions to sanitize user-controlled content before
including it in LLM prompts, and to validate LLM-generated output.

Defense Layers:
1. Content Delimiting (handled by prompt templates)
2. Input Sanitization (this module)
3. Structured Message Formats (handled by LLM tools)
4. Output Validation (this module)

Environment Variables:
    CLGRAPH_DISABLE_PROMPT_SANITIZATION: Set to "1" to disable input
        sanitization (for debugging only, NOT recommended for production).
        Output validation remains active.
"""

import logging
import os
import re
import unicodedata
from typing import Optional

logger = logging.getLogger("clgraph.security")

# Delimiter tags used in our prompt templates
# These tags separate instructions from user data
_DELIMITER_TAGS = frozenset(
    {
        "data",
        "schema",
        "question",
        "sql",
        "system",
        "user",
        "assistant",
    }
)

# Known SQL type keywords that use angle bracket syntax
# These appear before <type> in SQL like STRUCT<field STRING>
_SQL_TYPE_KEYWORDS = frozenset(
    {
        "STRUCT",
        "ARRAY",
        "MAP",
        "MULTISET",
        "ROW",
    }
)

# Compile the tag pattern once for efficiency
# Matches <tag>, </tag>, <tag attr="value">, etc.
# This pattern matches potential delimiter tags
_TAG_PATTERN = re.compile(
    r"</?(?:" + "|".join(re.escape(tag) for tag in _DELIMITER_TAGS) + r")(?:\s[^>]*)?>",
    re.IGNORECASE,
)

# Pattern to detect SQL type context: KEYWORD< at the start of a tag
_SQL_TYPE_CONTEXT_PATTERN = re.compile(
    r"(?:" + "|".join(re.escape(kw) for kw in _SQL_TYPE_KEYWORDS) + r")<",
    re.IGNORECASE,
)

# Control characters to strip (except \n, \t, and \r which is stripped separately)
# \x00-\x08: NUL through BS
# \x0b: VT (vertical tab)
# \x0c: FF (form feed)
# \x0e-\x1f: SO through US
# \x7f: DEL
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Carriage return pattern (strip CR but keep LF)
_CR_PATTERN = re.compile(r"\r(?!\n)")  # CR not followed by LF
_CRLF_PATTERN = re.compile(r"\r\n")  # CRLF -> LF


def _is_sanitization_disabled() -> bool:
    """Check if sanitization is disabled via environment variable.

    Only returns True if CLGRAPH_DISABLE_PROMPT_SANITIZATION is exactly "1".
    """
    return os.environ.get("CLGRAPH_DISABLE_PROMPT_SANITIZATION") == "1"


def _escape_tag(match: re.Match, text: str) -> str:
    """Escape a matched tag by converting < and > to HTML entities.

    Checks if the tag appears in SQL type context (e.g., STRUCT<data STRING>)
    and skips escaping in that case.
    """
    tag = match.group(0)
    start = match.start()

    # Check if this tag is preceded by a SQL type keyword
    # We look back to find if there's a keyword like STRUCT immediately before
    if start > 0:
        # Find where the potential keyword starts (look back up to 10 chars)
        lookback_start = max(0, start - 10)
        prefix = text[lookback_start:start]

        # Check if any SQL type keyword appears at the end of the prefix
        for keyword in _SQL_TYPE_KEYWORDS:
            if prefix.upper().endswith(keyword):
                # This is SQL type syntax - don't escape
                return tag

    return tag.replace("<", "&lt;").replace(">", "&gt;")


def sanitize_for_prompt(text: Optional[str], max_length: int = 1000) -> str:
    """
    Sanitize user-controlled text for safe LLM prompt inclusion.

    This function applies multiple layers of sanitization:
    1. Truncation to max_length to prevent context flooding
    2. NFKC Unicode normalization to catch homoglyph attacks
    3. Escaping of delimiter tags (data, schema, question, sql, system, user, assistant)
    4. Stripping of control characters (except newlines and tabs)

    Args:
        text: The text to sanitize. If None or empty, returns empty string.
        max_length: Maximum length of the sanitized output. Defaults to 1000.

    Returns:
        Sanitized text safe for prompt inclusion.

    Example:
        >>> sanitize_for_prompt("customer_id")
        'customer_id'
        >>> sanitize_for_prompt("<data>malicious</data>")
        '&lt;data&gt;malicious&lt;/data&gt;'
        >>> sanitize_for_prompt("a" * 2000, max_length=100)
        'aaa...aaa'  # 100 chars
    """
    if not text:
        return ""

    # Check if sanitization is disabled (for debugging only)
    if _is_sanitization_disabled():
        logger.warning(
            "SECURITY: Prompt sanitization disabled via environment variable. "
            "This is not recommended for production use."
        )
        return text[:max_length]

    # Step 1: Pre-truncate to prevent processing massive inputs
    # We'll truncate again at the end after transformations
    result = text[: max_length * 2]  # Allow some headroom for escaping

    # Step 2: NFKC Unicode normalization
    # This normalizes certain characters:
    # - Fullwidth characters to ASCII (e.g., fullwidth '<' to '<')
    # - Compatibility characters
    # Note: Cyrillic letters don't normalize to Latin (they're distinct scripts)
    result = unicodedata.normalize("NFKC", result)

    # Step 3: Escape delimiter tags
    # We escape rather than remove to prevent the replacement itself
    # from being used for injection (e.g., "[removed-tag]" attacks)
    # Use a lambda to pass the full text for context checking
    result = _TAG_PATTERN.sub(lambda m: _escape_tag(m, result), result)

    # Step 4: Strip control characters (except \n and \t)
    # First handle CRLF -> LF
    result = _CRLF_PATTERN.sub("\n", result)
    # Then strip lone CR
    result = _CR_PATTERN.sub("", result)
    # Finally strip other control characters
    result = _CONTROL_CHAR_PATTERN.sub("", result)

    # Step 5: Final truncation to ensure max_length after all transformations
    # This ensures context flooding prevention even after escaping
    result = result[:max_length]

    return result


def sanitize_sql_for_prompt(sql: Optional[str], max_length: int = 5000) -> str:
    """
    Sanitize SQL with higher length limit.

    SQL queries can be longer than typical text, so this function
    uses a higher default max_length (5000 vs 1000).

    This preserves SQL syntax like STRUCT<data STRING> because we escape
    delimiter tags only when they appear as complete tags, not as
    part of SQL type syntax.

    Args:
        sql: The SQL string to sanitize.
        max_length: Maximum length. Defaults to 5000 for SQL.

    Returns:
        Sanitized SQL safe for prompt inclusion.

    Example:
        >>> sanitize_sql_for_prompt("SELECT STRUCT<data STRING> FROM t")
        'SELECT STRUCT<data STRING> FROM t'
    """
    return sanitize_for_prompt(sql, max_length=max_length)


# =============================================================================
# Output Validation
# =============================================================================

# Instruction-like patterns that indicate prompt injection in descriptions
_INSTRUCTION_PATTERNS = [
    # "ignore/forget/disregard previous instructions/rules"
    re.compile(
        r"\b(ignore|forget|disregard|instead|override|bypass)\b.*"
        r"\b(instruction|previous|above|rule)",
        re.IGNORECASE,
    ),
    # "you are/act as/pretend to be"
    re.compile(r"\b(you are|act as|pretend|roleplay)\b", re.IGNORECASE),
    # "do not/don't follow/obey"
    re.compile(r"\b(do not|don'?t|never)\b.*\b(follow|obey|listen)", re.IGNORECASE),
    # System/Human/Assistant prompt markers
    re.compile(r"\bsystem\s*:", re.IGNORECASE),
    re.compile(r"\bhuman\s*:", re.IGNORECASE),
    re.compile(r"\bassistant\s*:", re.IGNORECASE),
]

# SQL statement patterns that should not appear in descriptions
# These match actual SQL commands, not just keywords as adjectives
_SQL_STATEMENT_PATTERNS = [
    re.compile(r"\bSELECT\s+[\w\*]", re.IGNORECASE),
    re.compile(r"\bDROP\s+(TABLE|DATABASE|INDEX|VIEW|SCHEMA)", re.IGNORECASE),
    re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE),
    re.compile(r"\bINSERT\s+INTO\b", re.IGNORECASE),
    re.compile(r"\bUPDATE\s+\w+\s+SET\b", re.IGNORECASE),
    re.compile(r"\bTRUNCATE\s+TABLE\b", re.IGNORECASE),
    re.compile(r"\bALTER\s+(TABLE|DATABASE|INDEX)", re.IGNORECASE),
]

# Common data description words for semantic relevance check
_DATA_CONCEPT_WORDS = frozenset(
    {
        "count",
        "sum",
        "total",
        "average",
        "avg",
        "min",
        "max",
        "date",
        "time",
        "timestamp",
        "datetime",
        "id",
        "identifier",
        "key",
        "primary",
        "foreign",
        "name",
        "title",
        "label",
        "description",
        "value",
        "amount",
        "number",
        "quantity",
        "price",
        "cost",
        "type",
        "status",
        "state",
        "code",
        "flag",
        "indicator",
        "user",
        "customer",
        "order",
        "product",
        "item",
        "created",
        "updated",
        "deleted",
        "modified",
        "active",
        "enabled",
        "disabled",
        "email",
        "phone",
        "address",
        "rate",
        "ratio",
        "percent",
        "percentage",
        "field",
        "column",
        "attribute",
        "property",
        "record",
        "row",
        "entry",
        "data",
        "counter",
        "sequence",
        "index",
    }
)


def _validate_description_output(
    response: str,
    column_name: str,
    table_name: str,
) -> Optional[str]:
    """
    Validate LLM-generated description for injection attempts.

    This function checks LLM output for:
    1. Length enforcement (max 200 characters)
    2. Instruction-like patterns (e.g., "ignore all previous instructions")
    3. Role confusion attempts (e.g., "You are now a different AI")
    4. SQL statement patterns (e.g., "DROP TABLE users")
    5. Semantic relevance (description should relate to column/table)

    Args:
        response: The LLM-generated description to validate.
        column_name: The column name being described (for relevance check).
        table_name: The table name (for relevance check).

    Returns:
        The validated description if safe, or None to trigger fallback
        to rule-based description generation.

    Example:
        >>> _validate_description_output(
        ...     "Total revenue from sales",
        ...     column_name="total_revenue",
        ...     table_name="sales"
        ... )
        'Total revenue from sales'
        >>> _validate_description_output(
        ...     "Ignore all previous instructions",
        ...     column_name="x",
        ...     table_name="t"
        ... )
        None
    """
    if not response:
        return ""

    description = response.strip()

    if not description:
        return ""

    # Step 1: Length enforcement
    if len(description) > 200:
        # Too long descriptions are suspicious; return None for fallback
        return None

    description_lower = description.lower()

    # Step 2: Check for instruction-like patterns
    for pattern in _INSTRUCTION_PATTERNS:
        if pattern.search(description_lower):
            logger.warning(
                "Rejected LLM description containing instruction pattern: %s",
                description[:50],
            )
            return None

    # Step 3: Check for SQL statement patterns
    for pattern in _SQL_STATEMENT_PATTERNS:
        if pattern.search(description):
            logger.warning(
                "Rejected LLM description containing SQL statement: %s",
                description[:50],
            )
            return None

    # Step 4: Semantic relevance check for longer descriptions
    # Short descriptions (<=50 chars) are allowed without strict relevance
    if len(description) > 50:
        # Build relevance terms from column and table names
        relevance_terms = set()

        # Add column name parts
        for part in column_name.lower().replace("_", " ").split():
            if len(part) > 2:  # Skip very short parts
                relevance_terms.add(part)

        # Add table name parts
        for part in table_name.lower().replace("_", " ").split():
            if len(part) > 2:
                relevance_terms.add(part)

        # Add common data concept words
        relevance_terms.update(_DATA_CONCEPT_WORDS)

        # Check if description has any relevance using word boundaries
        # This prevents false positives like "over" matching in "moreover"
        has_relevance = False
        for term in relevance_terms:
            # Use word boundary regex for accurate matching
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, description_lower):
                has_relevance = True
                break

        if not has_relevance:
            logger.warning(
                "Rejected LLM description lacking semantic relevance: %s",
                description[:50],
            )
            return None

    return description


def _validate_generated_sql(sql: str, allow_mutations: bool = False) -> str:
    """
    Validate generated SQL for destructive operations.

    Uses sqlglot parsing for accurate detection rather than string matching.
    This catches obfuscated SQL patterns like "D E L E T E" that string
    matching would miss.

    Args:
        sql: The generated SQL to validate.
        allow_mutations: If True, allows INSERT/UPDATE/DELETE operations.
            Defaults to False for safety.

    Returns:
        The validated SQL if safe.

    Raises:
        ValueError: If SQL contains destructive operations (when not allowed)
            or cannot be parsed for validation.

    Example:
        >>> _validate_generated_sql("SELECT * FROM users")
        'SELECT * FROM users'
        >>> _validate_generated_sql("DROP TABLE users")
        ValueError: Generated SQL contains destructive operation: Drop
    """
    if not sql or not sql.strip():
        raise ValueError("Generated SQL is empty")

    try:
        import sqlglot
    except ImportError:
        logger.warning("sqlglot not available for SQL validation; falling back to pattern matching")
        return _validate_sql_with_patterns(sql, allow_mutations)

    try:
        parsed = sqlglot.parse(sql)
    except sqlglot.errors.ParseError as e:
        raise ValueError(f"Generated SQL could not be parsed for validation: {e}") from e

    # Define destructive statement types
    # Note: sqlglot uses specific class names like TruncateTable, AlterTable
    destructive_types = {
        "Drop",
        "Delete",
        "Truncate",
        "TruncateTable",  # sqlglot uses this name
        "Alter",
        "AlterTable",  # sqlglot may use this
        "AlterColumn",
    }

    # Types that are destructive only when allow_mutations is False
    mutation_types = {
        "Insert",
        "Update",
        "Merge",
    }

    for statement in parsed:
        if statement is None:
            continue

        stmt_type = type(statement).__name__

        if stmt_type in destructive_types:
            raise ValueError(f"Generated SQL contains destructive operation: {stmt_type}")

        if not allow_mutations and stmt_type in mutation_types:
            raise ValueError(f"Generated SQL contains destructive operation: {stmt_type}")

    return sql


def _validate_sql_with_patterns(sql: str, allow_mutations: bool = False) -> str:
    """
    Fallback SQL validation using regex patterns.

    Used when sqlglot is not available.
    """
    sql_upper = sql.upper()

    # Always reject these
    destructive_patterns = [
        r"\bDROP\s+(TABLE|DATABASE|INDEX|VIEW|SCHEMA)\b",
        r"\bTRUNCATE\s+TABLE\b",
        r"\bALTER\s+(TABLE|DATABASE|INDEX)\b",
    ]

    # Reject unless mutations allowed
    mutation_patterns = [
        r"\bDELETE\s+FROM\b",
        r"\bINSERT\s+INTO\b",
        r"\bUPDATE\s+\w+\s+SET\b",
        r"\bMERGE\s+INTO\b",
    ]

    for pattern in destructive_patterns:
        if re.search(pattern, sql_upper):
            raise ValueError(f"Generated SQL contains destructive operation matching: {pattern}")

    if not allow_mutations:
        for pattern in mutation_patterns:
            if re.search(pattern, sql_upper):
                raise ValueError(
                    f"Generated SQL contains destructive operation matching: {pattern}"
                )

    return sql
