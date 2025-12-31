#!/usr/bin/env python3
"""
Convert Python examples to Jupyter notebooks with markdown explanations.

This script:
1. Parses each Python file to extract docstrings and code structure
2. Creates a Jupyter notebook with markdown cells explaining each section
3. Breaks code into logical cells (imports, setup, each major section)

Usage:
    python convert_to_notebooks.py [--dry-run] [--verbose] [file.py ...]

    Without arguments, converts all .py files in examples/ directory.
    With file arguments, converts only the specified files.
"""

import ast
import json
import re
import sys
import uuid
from pathlib import Path


def create_notebook_cell(cell_type: str, source: list[str], **kwargs) -> dict:
    """Create a notebook cell dictionary."""
    cell = {
        "cell_type": cell_type,
        "id": str(uuid.uuid4())[:8],  # Short unique ID
        "metadata": kwargs.get("metadata", {}),
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def extract_module_docstring(source_code: str) -> tuple[str, str]:
    """Extract module docstring and return (docstring, rest_of_code)."""
    try:
        tree = ast.parse(source_code)
        docstring = ast.get_docstring(tree)
        if docstring:
            lines = source_code.split("\n")
            # Find end of docstring
            in_docstring = False
            docstring_end = 0
            for i, line in enumerate(lines):
                if '"""' in line or "'''" in line:
                    if not in_docstring:
                        in_docstring = True
                        count = line.count('"""') + line.count("'''")
                        if count >= 2:
                            docstring_end = i + 1
                            break
                    else:
                        docstring_end = i + 1
                        break
            rest = "\n".join(lines[docstring_end:]).strip()
            return docstring, rest
        return "", source_code
    except SyntaxError:
        return "", source_code


def extract_main_function_body(source_code: str) -> tuple[str, str]:
    """Extract the body of main() function and imports.

    Returns (imports_and_globals, main_body)
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return source_code, ""

    lines = source_code.split("\n")

    # Find main function
    main_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            main_func = node
            break

    if not main_func:
        # No main function, return code as-is
        return source_code, ""

    # Get imports and code before main
    imports_end = main_func.lineno - 1

    # Get the body of main (dedented)
    main_start = main_func.lineno
    main_end = main_func.end_lineno if hasattr(main_func, "end_lineno") else len(lines)

    # Extract main body lines (skip def main():)
    main_body_lines = []
    in_main = False
    base_indent = None

    for i in range(main_start - 1, main_end):
        line = lines[i]
        if "def main" in line:
            in_main = True
            continue
        if in_main:
            if line.strip() and base_indent is None:
                base_indent = len(line) - len(line.lstrip())
            if base_indent is not None:
                # Dedent
                if line.strip():
                    dedented = line[base_indent:] if len(line) > base_indent else line.lstrip()
                    main_body_lines.append(dedented)
                else:
                    main_body_lines.append("")

    imports_section = "\n".join(lines[:imports_end]).strip()
    main_body = "\n".join(main_body_lines).strip()

    # Remove if __name__ block from imports
    imports_lines = []
    for line in imports_section.split("\n"):
        if "__name__" not in line and "__main__" not in line:
            imports_lines.append(line)

    return "\n".join(imports_lines).strip(), main_body


def split_by_section_markers(code: str) -> list[tuple[str, str]]:
    """Split code by section markers (comments like # ==== or # Step N).

    Returns list of (title, code) tuples.
    """
    lines = code.split("\n")
    sections = []

    current_title = ""
    current_lines = []

    # Patterns that indicate a new section
    separator_pattern = re.compile(r"^# [=\-]{10,}$")
    step_pattern = re.compile(r"^# (Step \d+|Example \d+|\d+\.)")

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for separator line
        if separator_pattern.match(stripped):
            # Save current section
            if current_lines or current_title:
                sections.append((current_title, "\n".join(current_lines).strip()))
                current_lines = []

            # Look for title in next lines
            title_found = False
            j = i + 1
            while j < len(lines) and j < i + 3:
                next_line = lines[j].strip()
                if separator_pattern.match(next_line):
                    j += 1
                    continue
                if next_line.startswith("#") and not separator_pattern.match(next_line):
                    current_title = next_line.lstrip("# ").strip()
                    i = j
                    title_found = True
                    break
                j += 1

            if not title_found:
                current_title = "Section"
            i += 1
            continue

        # Check for step/example markers
        match = step_pattern.match(stripped)
        if match:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
                current_lines = []
            current_title = stripped.lstrip("# ").strip()
            i += 1
            continue

        # Regular line
        current_lines.append(line)
        i += 1

    # Add remaining
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    # Clean up
    cleaned = []
    for title, code in sections:
        # Remove leading/trailing separators from code
        code_lines = code.split("\n")
        while code_lines and separator_pattern.match(code_lines[0].strip()):
            code_lines.pop(0)
        while code_lines and separator_pattern.match(code_lines[-1].strip()):
            code_lines.pop()

        code = "\n".join(code_lines).strip()
        if code:
            # Extract title from first comment if no title
            if not title:
                first_line = code_lines[0].strip() if code_lines else ""
                if first_line.startswith("#") and not separator_pattern.match(first_line):
                    title = first_line.lstrip("# ").strip()

            cleaned.append((title or "Code", code))

    return cleaned


def clean_print_separators(code: str) -> str:
    """Remove print statements that are just separators."""
    lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        # Skip print("=" * N) or print("-" * N) separators
        if re.match(r'print\(["\'][=\-]+["\']\)', stripped):
            continue
        if re.match(r'print\(["\'].*["\'] \* \d+\)', stripped):
            continue
        lines.append(line)
    return "\n".join(lines)


def fix_file_references(code: str) -> str:
    """Replace __file__ references with Path('.') for Jupyter compatibility."""
    # Replace Path(__file__).parent with Path('.')
    code = re.sub(r"Path\(__file__\)\.parent", "Path('.')", code)
    # Replace other __file__ patterns
    code = code.replace("__file__", "'notebook'")
    return code


def generate_markdown_title(title: str, code: str) -> str:
    """Generate a markdown cell based on the title and code content."""
    if not title or title == "Code":
        # Try to infer from code
        first_line = code.strip().split("\n")[0] if code.strip() else ""
        if first_line.startswith("#"):
            title = first_line.lstrip("# ").strip()
        else:
            title = "Code"

    # Clean up title
    title = title.rstrip(":").strip()

    # Look for descriptive comment at start of code
    description = ""
    lines = code.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and not re.match(r"^# [=\-]+$", stripped):
            desc = stripped.lstrip("# ").strip()
            if desc and desc != title:
                description = desc
                break
        elif stripped and not stripped.startswith("#"):
            break

    md = f"### {title}"
    if description and description.lower() != title.lower():
        md += f"\n\n{description}"

    return md


def convert_py_to_notebook(py_file: Path, verbose: bool = False) -> dict:
    """Convert a Python file to a Jupyter notebook structure."""
    source_code = py_file.read_text()
    example_name = py_file.stem

    # Extract module docstring
    docstring, rest_of_code = extract_module_docstring(source_code)

    cells = []

    # Title cell from docstring
    title_parts = [f"# {example_name.replace('_', ' ').title()}"]
    if docstring:
        # Clean up docstring - remove "Example demonstrating..." type phrases
        doc_lines = docstring.strip().split("\n")
        # Use first line as subtitle if it's not too long
        if doc_lines and len(doc_lines[0]) < 100:
            title_parts.append(f"\n**{doc_lines[0].strip()}**")
            if len(doc_lines) > 1:
                rest_doc = "\n".join(doc_lines[1:]).strip()
                if rest_doc:
                    title_parts.append(f"\n\n{rest_doc}")
        else:
            title_parts.append(f"\n\n{docstring}")

    cells.append(create_notebook_cell("markdown", ["\n".join(title_parts)]))

    # Extract imports and main body
    imports, main_body = extract_main_function_body(rest_of_code)

    # Add imports cell
    if imports.strip():
        # Clean imports - remove sys.path manipulation
        import_lines = []
        skip_next = False
        for line in imports.split("\n"):
            if "sys.path" in line:
                continue
            if skip_next:
                skip_next = False
                continue
            # Skip standalone sys import if only used for sys.path
            if line.strip() == "import sys":
                # Check if sys is used elsewhere
                if "sys." not in main_body:
                    continue
            import_lines.append(line)

        cleaned_imports = "\n".join(import_lines).strip()
        # Fix __file__ references in imports too
        cleaned_imports = fix_file_references(cleaned_imports)
        if cleaned_imports:
            cells.append(create_notebook_cell("markdown", ["### Imports"]))
            cells.append(create_notebook_cell("code", [cleaned_imports]))

    # If no main body, use the whole code
    if not main_body:
        main_body = rest_of_code

    # Split main body into sections
    sections = split_by_section_markers(main_body)

    if verbose:
        print(f"  Found {len(sections)} sections:")
        for title, _ in sections:
            print(f"    - {title[:50]}...")

    # Process each section
    for title, code in sections:
        code = code.strip()
        if not code:
            continue

        # Skip if __name__ blocks
        if "__name__" in code and "__main__" in code:
            continue

        # Clean up the code
        code = clean_print_separators(code)
        code = fix_file_references(code)

        if not code.strip():
            continue

        # Add markdown cell
        md = generate_markdown_title(title, code)
        cells.append(create_notebook_cell("markdown", [md]))

        # Add code cell
        cells.append(create_notebook_cell("code", [code]))

    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    return notebook


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    dry_run = "--dry-run" in sys.argv

    # Get files to convert
    examples_dir = Path(__file__).parent / "examples"

    # Filter out flags from arguments
    file_args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    if file_args:
        py_files = [Path(f) for f in file_args]
    else:
        py_files = sorted(examples_dir.glob("*.py"))

    print(f"Converting {len(py_files)} Python files to Jupyter notebooks...")
    print()

    converted = 0
    skipped = 0

    for py_file in py_files:
        print(f"Processing: {py_file.name}")

        try:
            notebook = convert_py_to_notebook(py_file, verbose=verbose)
            nb_file = py_file.with_suffix(".ipynb")

            if dry_run:
                print(f"  Would create: {nb_file.name}")
                print(f"  Cells: {len(notebook['cells'])}")
            else:
                with open(nb_file, "w") as f:
                    json.dump(notebook, f, indent=2)
                print(f"  Created: {nb_file.name} ({len(notebook['cells'])} cells)")

            converted += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            if verbose:
                traceback.print_exc()
            skipped += 1

        print()

    print("=" * 60)
    print(f"Converted: {converted}")
    print(f"Skipped: {skipped}")
    if dry_run:
        print("(Dry run - no files were written)")


if __name__ == "__main__":
    main()
