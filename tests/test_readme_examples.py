"""
Test that code examples in README.md work correctly.

This test uses markdown_examples to extract and run Python code blocks
from the README to ensure documentation stays accurate.
"""

import contextlib
import io
import sys
from pathlib import Path

import pytest

# Import the markdown testing module
sys.path.insert(0, str(Path(__file__).parent))
from markdown_examples import execute_code_block, extract_python_blocks


@pytest.fixture(scope="module")
def readme_path():
    """Get path to README.md."""
    return Path(__file__).parent.parent / "README.md"


@pytest.fixture(scope="module")
def readme_content(readme_path):
    """Get README content."""
    return readme_path.read_text()


@pytest.fixture(scope="module")
def all_blocks(readme_content):
    """Extract all Python code blocks from README."""
    blocks, _use_preamble = extract_python_blocks(readme_content)
    return blocks


@pytest.fixture(scope="module")
def runnable_blocks(all_blocks):
    """Get only runnable (non-skipped) blocks."""
    return [block for block in all_blocks if not block.skip]


def get_section_for_block(content: str, block_line: int) -> str:
    """
    Get the markdown section (heading) that a code block belongs to.

    Args:
        content: Full markdown content
        block_line: Line number where the code block starts

    Returns:
        The section heading (e.g., "Single Query Column Lineage")
    """
    lines = content.split("\n")
    current_section = "Unknown Section"

    for _i, line in enumerate(lines[:block_line], start=1):
        if line.startswith("#"):
            # Strip all # and whitespace to get clean section name
            current_section = line.lstrip("#").strip()

    return current_section


def execute_block_with_output(block):
    """Execute a code block and capture its output."""
    output_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(output_buffer):
            success, error = execute_code_block(block, isolated=True)
    except Exception as e:
        success = False
        error = str(e)

    return success, error, output_buffer.getvalue()


# ============================================================================
# Programmatic Tests - No Hard-Coded Block Numbers!
# ============================================================================


def test_readme_exists(readme_path):
    """Verify README.md exists."""
    assert readme_path.exists(), f"README not found at {readme_path}"


def test_readme_has_python_examples(all_blocks):
    """Verify README contains Python code examples."""
    assert len(all_blocks) > 0, "README should contain Python code examples"


def test_readme_has_runnable_examples(runnable_blocks):
    """Verify README has at least some runnable (non-skipped) examples."""
    assert len(runnable_blocks) > 0, "README should have runnable code examples"


def pytest_generate_tests(metafunc):
    """Dynamically generate tests for each README code block."""
    if "code_block" in metafunc.fixturenames:
        # Get the README content and extract blocks
        readme_path = Path(metafunc.config.rootdir) / "README.md"
        content = readme_path.read_text()
        blocks, _use_preamble = extract_python_blocks(content)

        # Filter to runnable blocks only
        runnable = [b for b in blocks if not b.skip]

        # Generate test IDs based on section names
        ids = []
        for block in runnable:
            section = get_section_for_block(content, block.line_number)
            # Create clean test ID
            test_id = f"block_{block.block_number + 1}_{section.replace(' ', '_').lower()}"
            ids.append(test_id)

        # Parametrize the test
        metafunc.parametrize("code_block", runnable, ids=ids)


def test_each_readme_example(code_block, readme_content):
    """
    Test a single README code example.

    This test is dynamically generated for each runnable code block.
    The test ID includes the block number and section name for easy identification.
    """
    section = get_section_for_block(readme_content, code_block.line_number)

    print(f"\n{'=' * 70}")
    print(f"Testing: {section}")
    print(f"Block {code_block.block_number + 1} (line {code_block.line_number})")
    print(f"{'=' * 70}")

    # Execute and capture output
    success, error, output = execute_block_with_output(code_block)

    # Display output
    if output:
        print("Output:")
        print(output)

    # Assert success
    if not success:
        print(f"❌ FAILED: {error}")
        pytest.fail(f"Example in section '{section}' failed: {error}")

    print("✅ PASSED")
    print(f"{'=' * 70}")


def test_skip_markers_are_working(all_blocks):
    """Verify that skip markers are being respected."""
    skipped = [b for b in all_blocks if b.skip]
    runnable = [b for b in all_blocks if not b.skip]

    print(f"\nTotal blocks: {len(all_blocks)}")
    print(f"Runnable: {len(runnable)}")
    print(f"Skipped: {len(skipped)}")

    # Should have both runnable and skipped blocks
    assert len(runnable) > 0, "Should have some runnable examples"
    assert len(skipped) > 0, "Should have some skipped blocks (API references)"


def test_all_examples_summary(runnable_blocks, readme_content):
    """
    Run all examples and provide a summary.

    This is a comprehensive test that runs all blocks and reports results.
    """
    print("\n" + "=" * 70)
    print("README Examples Summary")
    print("=" * 70)

    results = []

    for block in runnable_blocks:
        section = get_section_for_block(readme_content, block.line_number)
        success, error, output = execute_block_with_output(block)

        results.append(
            {
                "block_num": block.block_number + 1,
                "section": section,
                "success": success,
                "error": error,
                "output": output,
            }
        )

    # Print summary table
    print(f"\n{'Block':<8} {'Section':<40} {'Status':<10}")
    print("-" * 70)

    for r in results:
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        print(f"{r['block_num']:<8} {r['section'][:40]:<40} {status:<10}")

    print("-" * 70)

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    print(f"\nResults: {passed}/{total} passed")
    print("=" * 70)

    # Collect failures
    failures = [r for r in results if not r["success"]]

    if failures:
        error_msg = f"\n{len(failures)} example(s) failed:\n\n"
        for f in failures:
            error_msg += f"Block {f['block_num']} - {f['section']}:\n"
            error_msg += f"  Error: {f['error']}\n\n"
        pytest.fail(error_msg)

    assert passed == total, f"Expected all {total} examples to pass, but {total - passed} failed"


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "-s"])
