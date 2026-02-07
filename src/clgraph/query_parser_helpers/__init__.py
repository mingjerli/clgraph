"""Query parser helpers - composition-based decomposition.

This package contains helper classes extracted from RecursiveQueryParser
using the composition pattern. Each helper receives a reference to the
parent parser in its constructor for accessing shared state.

Helper Dependencies:
- FromClauseParser depends on: JoinHandler, SpecialSourcesHandler
- All helpers depend on: RecursiveQueryParser (via constructor injection)

Usage:
    from clgraph.query_parser_helpers import FromClauseParser, WindowFunctionsParser
"""

from .from_clause import FromClauseParser
from .grouping import GroupingParser
from .join_handler import JoinHandler
from .merge import MergeParser
from .pivot_unpivot import PivotUnpivotParser
from .recursive_cte import RecursiveCTEParser
from .set_operations import SetOperationsParser
from .special_sources import SpecialSourcesHandler
from .subqueries import SubqueryParser
from .window_functions import WindowFunctionsParser

__all__ = [
    "FromClauseParser",
    "JoinHandler",
    "SpecialSourcesHandler",
    "SetOperationsParser",
    "PivotUnpivotParser",
    "MergeParser",
    "WindowFunctionsParser",
    "GroupingParser",
    "RecursiveCTEParser",
    "SubqueryParser",
]
