#!/usr/bin/env python3
"""
Test Python code examples in markdown files.

This script extracts Python code blocks from markdown files and executes them
to verify they run without errors.

Features:
- Isolated mode: Each block runs in its own namespace (default)
- Sequential mode: Blocks share namespace (--sequential)
- Preamble support: Auto-inject common imports and sample pipeline (--preamble)
- Skip markers: <!-- skip-test --> to skip specific blocks
- API validation: Check if methods/attributes exist without running (--validate-api)
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

# Default preamble that sets up common imports and a sample pipeline
DEFAULT_PREAMBLE = '''
from pathlib import Path
from clgraph import Pipeline

# Sample pipeline for documentation examples
_sample_queries = [
    ("01_raw", """
        CREATE TABLE raw.orders AS
        SELECT order_id, customer_id, amount, status FROM external.orders
    """),
    ("02_staging", """
        CREATE TABLE staging.orders AS
        SELECT
            customer_id,
            SUM(amount) as total_amount,
            COUNT(*) as order_count
        FROM raw.orders
        WHERE status = 'completed'
        GROUP BY customer_id
    """),
    ("03_analytics", """
        CREATE TABLE analytics.customer_metrics AS
        SELECT
            customer_id,
            total_amount,
            order_count,
            total_amount / order_count as avg_order_value
        FROM staging.orders
    """)
]

pipeline = Pipeline.from_tuples(_sample_queries, dialect="bigquery")
table_graph = pipeline.table_graph
'''

# Extended preamble with pre-loaded pipeline and mock LLM for agent/tool examples
EXTENDED_PREAMBLE = '''
import sys
from pathlib import Path

# Add fixtures to path for mock imports
_fixtures_path = Path(__file__).parent / "fixtures" if "__file__" in dir() else Path("tests/fixtures")
if not _fixtures_path.exists():
    _fixtures_path = Path("tests/fixtures")
sys.path.insert(0, str(_fixtures_path))

from clgraph import Pipeline
from clgraph.agent import LineageAgent
from clgraph.tools import (
    TraceBackwardTool,
    TraceForwardTool,
    ListTablesTool,
    GetTableSchemaTool,
    SearchColumnsTool,
    FindPIIColumnsTool,
    GenerateSQLTool,
)

# Load pre-built pipeline from fixture (realistic multi-layer pipeline)
_pipeline_json_path = _fixtures_path / "sql_files_pipeline.json"
if _pipeline_json_path.exists():
    pipeline = Pipeline.from_json_file(str(_pipeline_json_path))
else:
    # Fallback to simple pipeline if fixture not found
    _sample_queries = [
        ("staging.orders", """
            CREATE TABLE staging.orders AS
            SELECT order_id, customer_email, amount FROM raw.orders
        """),
        ("analytics.revenue", """
            CREATE TABLE analytics.revenue AS
            SELECT customer_email, SUM(amount) as total FROM staging.orders GROUP BY 1
        """),
    ]
    pipeline = Pipeline.from_tuples(_sample_queries, dialect="bigquery")

table_graph = pipeline.table_graph

# Mock LLM for testing (provides predictable responses)
from mock_llm import MockLLM
llm = MockLLM(model="mock-model", temperature=0.3)
'''


class CodeBlock:
    """Represents a Python code block extracted from markdown."""

    def __init__(self, code: str, line_number: int, block_number: int, skip: bool = False):
        self.code = code
        self.line_number = line_number
        self.block_number = block_number
        self.skip = skip


def _get_public_members(cls) -> Set[str]:
    """Get public methods, properties, and instance attributes from a class."""
    import inspect

    members = set()

    # Get methods and properties from class
    for name, _ in inspect.getmembers(cls):
        if not name.startswith("_"):
            members.add(name)

    # Also get instance attributes from __init__ using AST
    try:
        import inspect as insp
        import textwrap

        source = insp.getsource(cls.__init__)
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            # Handle regular assignments: self.foo = bar
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if (
                            isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and not target.attr.startswith("_")
                        ):
                            members.add(target.attr)
            # Handle annotated assignments: self.foo: Type = bar
            elif isinstance(node, ast.AnnAssign):
                target = node.target
                if isinstance(target, ast.Attribute):
                    if (
                        isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and not target.attr.startswith("_")
                    ):
                        members.add(target.attr)
    except (OSError, TypeError, SyntaxError):
        pass  # Source not available or parse error

    return members


def generate_clgraph_api() -> dict:
    """
    Auto-generate API dictionary from actual clgraph classes.

    Returns:
        Dictionary mapping class names to their public methods/attributes.
    """
    try:
        from clgraph import Pipeline
        from clgraph.column import PipelineLineageGraph
        from clgraph.export import CSVExporter, JSONExporter
        from clgraph.models import ColumnLineageGraph
        from clgraph.table import TableDependencyGraph

        return {
            "Pipeline": _get_public_members(Pipeline),
            "TableDependencyGraph": _get_public_members(TableDependencyGraph),
            "PipelineLineageGraph": _get_public_members(PipelineLineageGraph),
            "ColumnLineageGraph": _get_public_members(ColumnLineageGraph),
            "CSVExporter": _get_public_members(CSVExporter),
            "JSONExporter": _get_public_members(JSONExporter),
        }
    except ImportError:
        # Fallback to static definition if imports fail
        return _STATIC_CLGRAPH_API


# Static fallback API definition (used if clgraph not importable)
_STATIC_CLGRAPH_API = {
    "Pipeline": {
        "columns",
        "edges",
        "table_graph",
        "column_graph",
        "query_graphs",
        "llm",
        "from_sql_files",
        "from_sql_string",
        "from_sql_list",
        "from_tuples",
        "from_dict",
        "trace_column_backward",
        "trace_column_backward_full",
        "trace_column_forward",
        "trace_column_forward_full",
        "get_lineage_path",
        "get_simplified_column_graph",
        "get_column",
        "get_columns_by_table",
        "get_columns_by_owner",
        "get_columns_by_tag",
        "propagate_all_metadata",
        "get_pii_columns",
        "generate_all_descriptions",
        "to_json",
        "to_diff",
        "run",
        "async_run",
        "to_airflow_dag",
    },
    "TableDependencyGraph": {
        "tables",
        "queries",
        "topological_sort",
        "get_execution_levels",
        "get_table_dependencies",
        "get_all_sources",
        "get_all_finals",
    },
    "PipelineLineageGraph": {
        "columns",
        "edges",
        "to_simplified",
        "add_column",
        "add_edge",
        "get_upstream",
        "get_downstream",
        "get_final_columns",
    },
    "ColumnLineageGraph": {
        "nodes",
        "edges",
        "query_units",
        "get_input_nodes",
        "get_output_nodes",
        "get_cte_nodes",
    },
    "CSVExporter": {
        "export_columns_to_file",
        "export_edges_to_file",
    },
    "JSONExporter": {
        "export_to_file",
    },
}

# Auto-generate API from actual classes (with static fallback)
CLGRAPH_API = generate_clgraph_api()


class APIExtractor(ast.NodeVisitor):
    """Extract method/attribute accesses from Python AST."""

    def __init__(self):
        self.accesses: List[Tuple[str, str, int]] = []  # (var_name, attr_name, line)
        self.var_types: dict = {}  # Track variable types from assignments

    def visit_Assign(self, node: ast.Assign):
        """Track variable assignments to infer types."""
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute):
                # e.g., pipeline = Pipeline.from_sql_files(...)
                class_name = self._get_call_class(node.value.func)
                if class_name:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.var_types[target.id] = class_name
            elif isinstance(node.value.func, ast.Name):
                # e.g., pipeline = Pipeline(...)
                class_name = node.value.func.id
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.var_types[target.id] = class_name
        self.generic_visit(node)

    def _get_call_class(self, node: ast.Attribute) -> Optional[str]:
        """Get the class name from a method call like Pipeline.from_sql_files."""
        if isinstance(node.value, ast.Name):
            return node.value.id
        return None

    def visit_Attribute(self, node: ast.Attribute):
        """Record attribute accesses."""
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            attr_name = node.attr
            self.accesses.append((var_name, attr_name, node.lineno))
        elif isinstance(node.value, ast.Attribute):
            # Handle chained attributes like pipeline.table_graph.tables
            self._handle_chained(node)
        self.generic_visit(node)

    def _handle_chained(self, node: ast.Attribute):
        """Handle chained attribute access."""
        if isinstance(node.value, ast.Attribute):
            # Get the innermost name
            inner = node.value
            while isinstance(inner.value, ast.Attribute):
                inner = inner.value
            if isinstance(inner.value, ast.Name):
                # Record both the intermediate and final attributes
                self.accesses.append((inner.value.id, inner.attr, inner.lineno))


def validate_api_usage(code: str, base_line: int = 0) -> Tuple[List[str], List[str]]:
    """
    Validate that API calls in code reference existing methods/attributes.

    Args:
        code: Python code to validate
        base_line: Line offset for error reporting

    Returns:
        Tuple of (valid_accesses, invalid_accesses)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [], [f"Syntax error at line {e.lineno}: {e.msg}"]

    extractor = APIExtractor()
    extractor.visit(tree)

    valid = []
    invalid = []

    for var_name, attr_name, line in extractor.accesses:
        # Check if variable has known type
        var_type = extractor.var_types.get(var_name)

        # Also check common naming patterns
        if var_type is None:
            if var_name == "pipeline" or var_name.endswith("_pipeline"):
                var_type = "Pipeline"
            elif var_name == "table_graph" or var_name.endswith("_graph"):
                var_type = "TableDependencyGraph"
            elif var_name == "column_graph":
                var_type = "PipelineLineageGraph"

        if var_type and var_type in CLGRAPH_API:
            valid_attrs = CLGRAPH_API[var_type]
            if attr_name in valid_attrs:
                valid.append(f"{var_type}.{attr_name}")
            else:
                invalid.append(
                    f"Line {base_line + line}: {var_type}.{attr_name} - "
                    f"'{attr_name}' not found in {var_type}"
                )

        # Check class method calls like Pipeline.from_sql_files
        if var_name in CLGRAPH_API:
            if attr_name in CLGRAPH_API[var_name]:
                valid.append(f"{var_name}.{attr_name}")
            else:
                invalid.append(
                    f"Line {base_line + line}: {var_name}.{attr_name} - "
                    f"'{attr_name}' not found in {var_name}"
                )

    return valid, invalid


def validate_markdown_api(
    filepath: Path,
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Validate API usage in all Python code blocks.

    Args:
        filepath: Path to the markdown file
        verbose: If True, print detailed output

    Returns:
        Tuple of (valid_count, invalid_count)
    """
    print(f"Validating API usage in: {filepath}")
    print("-" * 60)

    content = filepath.read_text()
    blocks, _ = extract_python_blocks(content)

    if not blocks:
        print("No Python code blocks found.")
        return 0, 0

    print(f"Found {len(blocks)} Python code blocks\n")

    total_valid = 0
    total_invalid = 0

    for block in blocks:
        block_id = f"Block {block.block_number + 1} (line {block.line_number})"

        if block.skip:
            print(f"{block_id}: ⏭️  SKIP (still validating API)")

        valid, invalid = validate_api_usage(block.code, block.line_number)

        if verbose and valid:
            print(f"\n{block_id} - Valid API calls:")
            for v in valid:
                print(f"  ✓ {v}")

        if invalid:
            print(f"\n{block_id} - Invalid API calls:")
            for inv in invalid:
                print(f"  ✗ {inv}")
            total_invalid += len(invalid)
        else:
            print(f"{block_id}: ✅ API OK ({len(valid)} calls validated)")

        total_valid += len(valid)

    print("\n" + "=" * 60)
    print(f"API Validation: {total_valid} valid, {total_invalid} invalid")

    return total_valid, total_invalid


def extract_python_blocks(markdown_content: str) -> Tuple[List[CodeBlock], bool]:
    """
    Extract Python code blocks from markdown content.

    Supports markers:
    - <!-- skip-test --> : Skip the next code block
    - <!-- use-preamble --> : Enable preamble for this file (at file level)

    Args:
        markdown_content: The markdown file content

    Returns:
        Tuple of (List of CodeBlock objects, use_preamble flag)
    """
    blocks = []
    lines = markdown_content.split("\n")
    in_python_block = False
    current_block = []
    block_start_line = 0
    block_count = 0
    skip_next_block = False
    use_preamble = False

    for i, line in enumerate(lines, start=1):
        # Check for preamble marker (file-level setting)
        if "<!-- use-preamble -->" in line:
            use_preamble = True
            continue

        # Check for skip marker before code block
        if "<!-- skip-test -->" in line or "<!-- doctest: skip -->" in line:
            skip_next_block = True
            continue

        # Check for code block start
        if line.strip().startswith("```python"):
            in_python_block = True
            block_start_line = i + 1
            current_block = []
        # Check for code block end
        elif line.strip().startswith("```") and in_python_block:
            in_python_block = False
            if current_block:
                code = "\n".join(current_block)
                # Check if first line has skip marker
                skip = skip_next_block or (
                    current_block
                    and current_block[0].strip() in ["# skip-test", "# doctest: skip", "# noqa"]
                )
                blocks.append(CodeBlock(code, block_start_line, block_count, skip))
                block_count += 1
                skip_next_block = False
        # Collect code lines
        elif in_python_block:
            current_block.append(line)

    return blocks, use_preamble


def create_preamble_namespace(preamble: str = DEFAULT_PREAMBLE) -> dict:
    """
    Execute preamble code and return the namespace.

    Args:
        preamble: Python code to execute as preamble

    Returns:
        Dictionary namespace with preamble variables
    """
    namespace = {"__name__": "__main__"}
    try:
        exec(preamble, namespace)
    except Exception as e:
        print(f"Warning: Preamble execution failed: {e}")
    return namespace


def execute_code_block(
    block: CodeBlock,
    isolated: bool = True,
    shared_globals: Optional[dict] = None,
    preamble: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Execute a code block and return success status and error message.

    Args:
        block: The CodeBlock to execute
        isolated: If True, execute in isolated namespace. If False, use shared_globals
        shared_globals: Shared global namespace for sequential execution
        preamble: Optional preamble code to run before each isolated block

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        if isolated:
            # Execute in isolated namespace, optionally with preamble
            if preamble:
                namespace = create_preamble_namespace(preamble)
            else:
                namespace = {"__name__": "__main__"}
            exec(block.code, namespace)
        else:
            # Execute in shared namespace for sequential execution
            if shared_globals is None:
                shared_globals = {"__name__": "__main__"}
            exec(block.code, shared_globals)
        return True, ""
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return False, error_msg


def test_markdown_file(
    filepath: Path,
    isolated: bool = True,
    verbose: bool = False,
    use_preamble: bool = False,
    preamble: str = DEFAULT_PREAMBLE,
) -> Tuple[int, int]:
    """
    Test all Python code blocks in a markdown file.

    Args:
        filepath: Path to the markdown file
        isolated: If True, run each block in isolation. If False, run sequentially
        verbose: If True, print detailed output
        use_preamble: If True, inject preamble before each isolated block
        preamble: Custom preamble code (uses DEFAULT_PREAMBLE if not specified)

    Returns:
        Tuple of (passed_count, total_count)
    """
    print(f"Testing Python code blocks in: {filepath}")
    mode_parts = []
    if isolated:
        mode_parts.append("isolated")
    else:
        mode_parts.append("sequential")
    if use_preamble:
        mode_parts.append("with preamble")
    print(f"Execution mode: {', '.join(mode_parts)}")
    print("-" * 60)

    # Read markdown file
    content = filepath.read_text()

    # Extract code blocks and check for file-level preamble marker
    blocks, file_uses_preamble = extract_python_blocks(content)

    # Use preamble if enabled via CLI or file marker
    effective_use_preamble = use_preamble or file_uses_preamble

    if file_uses_preamble and not use_preamble:
        print("Note: File has <!-- use-preamble --> marker, enabling preamble")

    if not blocks:
        print("No Python code blocks found.")
        return 0, 0

    print(f"Found {len(blocks)} Python code blocks\n")

    # For sequential mode with preamble, run preamble first
    if not isolated and effective_use_preamble:
        shared_globals = create_preamble_namespace(preamble)
    elif not isolated:
        shared_globals = {"__name__": "__main__"}
    else:
        shared_globals = None

    # Execute blocks
    passed = 0
    failed = 0
    skipped = 0

    for block in blocks:
        block_id = f"Block {block.block_number + 1} (line {block.line_number})"

        if verbose:
            print(f"\n{block_id}:")
            print("```python")
            print(block.code)
            print("```")

        if block.skip:
            skipped += 1
            status = "⏭️  SKIP"
        else:
            # For isolated mode with preamble, pass preamble to each block
            block_preamble = preamble if (isolated and effective_use_preamble) else None
            success, error = execute_code_block(block, isolated, shared_globals, block_preamble)

            if success:
                passed += 1
                status = "✅ PASS"
            else:
                failed += 1
                status = f"❌ FAIL: {error}"

        print(f"{block_id}: {status}")

    # Print summary
    print("\n" + "=" * 60)
    tested = passed + failed
    print(f"Results: {passed}/{tested} passed, {failed}/{tested} failed, {skipped} skipped")
    print(f"Total blocks: {len(blocks)}")

    return passed, tested


def main():
    parser = argparse.ArgumentParser(
        description="Test Python code examples in markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test in isolated mode (default)
  python markdown_examples.py docs/example.md

  # Test with shared state between blocks
  python markdown_examples.py docs/tutorial.md --sequential

  # Test with default preamble (imports + sample pipeline)
  python markdown_examples.py docs/api.md --preamble

  # Combine sequential + preamble for tutorials
  python markdown_examples.py docs/tutorial.md --sequential --preamble

Markers in markdown:
  <!-- skip-test -->     Skip the next code block
  <!-- use-preamble -->  Enable preamble for this file (add at top of file)
        """,
    )
    parser.add_argument("filepath", type=Path, help="Path to the markdown file")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run code blocks sequentially with shared state (default: isolated)",
    )
    parser.add_argument(
        "--preamble",
        action="store_true",
        help="Inject default preamble (imports + sample pipeline) before execution",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print code blocks being tested"
    )
    parser.add_argument(
        "--validate-api",
        action="store_true",
        help="Only validate API usage (check methods/attributes exist) without running code",
    )

    args = parser.parse_args()

    if not args.filepath.exists():
        print(f"Error: File not found: {args.filepath}", file=sys.stderr)
        sys.exit(1)

    if not args.filepath.suffix == ".md":
        print(f"Warning: File does not have .md extension: {args.filepath}")

    if args.validate_api:
        # API validation mode - check methods/attributes without running
        valid, invalid = validate_markdown_api(
            args.filepath,
            verbose=args.verbose,
        )
        sys.exit(1 if invalid > 0 else 0)
    else:
        # Execution mode - run the code blocks
        passed, total = test_markdown_file(
            args.filepath,
            isolated=not args.sequential,
            verbose=args.verbose,
            use_preamble=args.preamble,
        )

        # Exit with error code if any tests failed
        if passed < total:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
