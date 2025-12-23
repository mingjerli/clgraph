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
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Default preamble that sets up common imports and a sample pipeline
DEFAULT_PREAMBLE = '''
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


class CodeBlock:
    """Represents a Python code block extracted from markdown."""

    def __init__(self, code: str, line_number: int, block_number: int, skip: bool = False):
        self.code = code
        self.line_number = line_number
        self.block_number = block_number
        self.skip = skip


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

    args = parser.parse_args()

    if not args.filepath.exists():
        print(f"Error: File not found: {args.filepath}", file=sys.stderr)
        sys.exit(1)

    if not args.filepath.suffix == ".md":
        print(f"Warning: File does not have .md extension: {args.filepath}")

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
