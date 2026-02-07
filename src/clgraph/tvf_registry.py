"""
Table-Valued Function (TVF) Registry.

Contains constants for known TVF expressions, names, and default output columns.
Used by RecursiveQueryParser for TVF detection and handling.

Extracted from query_parser.py to improve module organization.
"""

from typing import Dict, List

from sqlglot import exp

from .models import TVFType

# ============================================================================
# Table-Valued Functions (TVF) Registry
# ============================================================================

# Known TVF expressions mapped to their types
KNOWN_TVF_EXPRESSIONS: Dict[type, TVFType] = {
    # Generator TVFs
    exp.ExplodingGenerateSeries: TVFType.GENERATOR,
    exp.GenerateSeries: TVFType.GENERATOR,
    exp.GenerateDateArray: TVFType.GENERATOR,
    # External data TVFs
    exp.ReadCSV: TVFType.EXTERNAL,
}

# Known TVF function names (for Anonymous function calls)
KNOWN_TVF_NAMES: Dict[str, TVFType] = {
    # Generator TVFs
    "generate_series": TVFType.GENERATOR,
    "generate_date_array": TVFType.GENERATOR,
    "generate_timestamp_array": TVFType.GENERATOR,
    "sequence": TVFType.GENERATOR,
    "generator": TVFType.GENERATOR,
    "range": TVFType.GENERATOR,
    # Column-input TVFs (UNNEST/EXPLODE handled separately)
    "flatten": TVFType.COLUMN_INPUT,
    "explode": TVFType.COLUMN_INPUT,
    "posexplode": TVFType.COLUMN_INPUT,
    # External data TVFs
    "read_csv": TVFType.EXTERNAL,
    "read_parquet": TVFType.EXTERNAL,
    "read_json": TVFType.EXTERNAL,
    "read_ndjson": TVFType.EXTERNAL,
    "external_query": TVFType.EXTERNAL,
    # System TVFs
    "table": TVFType.SYSTEM,
    "result_scan": TVFType.SYSTEM,
}

# Default output column names for known TVFs
TVF_DEFAULT_COLUMNS: Dict[str, List[str]] = {
    "generate_series": ["generate_series"],
    "generate_date_array": ["date"],
    "generate_timestamp_array": ["timestamp"],
    "sequence": ["value"],
    "generator": ["seq4"],
    "range": ["range"],
    "flatten": ["value", "index", "key", "path", "this"],
    "explode": ["col"],
    "posexplode": ["pos", "col"],
}


__all__ = [
    "KNOWN_TVF_EXPRESSIONS",
    "KNOWN_TVF_NAMES",
    "TVF_DEFAULT_COLUMNS",
]
