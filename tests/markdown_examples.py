#!/usr/bin/env python3
"""
Test Python code examples in markdown files.

This script extracts Python code blocks from markdown files and executes them
to verify they run without errors.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


class CodeBlock:
    """Represents a Python code block extracted from markdown."""

    def __init__(self, code: str, line_number: int, block_number: int, skip: bool = False):
        self.code = code
        self.line_number = line_number
        self.block_number = block_number
        self.skip = skip


def extract_python_blocks(markdown_content: str) -> List[CodeBlock]:
    """
    Extract Python code blocks from markdown content.

    Supports skip markers:
    - HTML comment before code block: <!-- skip-test -->
    - First line of code: # skip-test

    Args:
        markdown_content: The markdown file content

    Returns:
        List of CodeBlock objects containing the code and metadata
    """
    blocks = []
    lines = markdown_content.split("\n")
    in_python_block = False
    current_block = []
    block_start_line = 0
    block_count = 0
    skip_next_block = False

    for i, line in enumerate(lines, start=1):
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

    return blocks


def execute_code_block(
    block: CodeBlock, isolated: bool = True, shared_globals: dict = None
) -> Tuple[bool, str]:
    """
    Execute a code block and return success status and error message.

    Args:
        block: The CodeBlock to execute
        isolated: If True, execute in isolated namespace. If False, use shared_globals
        shared_globals: Shared global namespace for sequential execution

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        if isolated:
            # Execute in isolated namespace
            exec(block.code, {"__name__": "__main__"})
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
    filepath: Path, isolated: bool = True, verbose: bool = False
) -> Tuple[int, int]:
    """
    Test all Python code blocks in a markdown file.

    Args:
        filepath: Path to the markdown file
        isolated: If True, run each block in isolation. If False, run sequentially
        verbose: If True, print detailed output

    Returns:
        Tuple of (passed_count, total_count)
    """
    print(f"Testing Python code blocks in: {filepath}")
    print(f"Execution mode: {'isolated' if isolated else 'sequential'}")
    print("-" * 60)

    # Read markdown file
    content = filepath.read_text()

    # Extract code blocks
    blocks = extract_python_blocks(content)

    if not blocks:
        print("No Python code blocks found.")
        return 0, 0

    print(f"Found {len(blocks)} Python code blocks\n")

    # Execute blocks
    passed = 0
    failed = 0
    skipped = 0
    shared_globals = {"__name__": "__main__"} if not isolated else None

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
            success, error = execute_code_block(block, isolated, shared_globals)

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

    return passed, tested  # Return tested count, not total blocks


def main():
    parser = argparse.ArgumentParser(description="Test Python code examples in markdown files")
    parser.add_argument("filepath", type=Path, help="Path to the markdown file")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run code blocks sequentially with shared state (default: isolated)",
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
        args.filepath, isolated=not args.sequential, verbose=args.verbose
    )

    # Exit with error code if any tests failed
    if passed < total:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
