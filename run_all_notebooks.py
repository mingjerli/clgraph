#!/usr/bin/env python3
"""
Script to run all Jupyter notebooks and verify they work correctly.

This script runs all notebook files in the examples/ directory and reports
which ones succeed or fail.

Usage:
    python run_all_notebooks.py [--verbose] [--skip-llm] [--save-outputs]

Options:
    --verbose       Show full output from each notebook
    --skip-llm      Skip notebooks that require LLM API keys
    --save-outputs  Save execution outputs back to notebook files
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# ANSI color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


class NotebookRunner:
    """Runs all notebook files and tracks results."""

    def __init__(self, verbose: bool = False, skip_llm: bool = False, save_outputs: bool = False):
        self.verbose = verbose
        self.skip_llm = skip_llm
        self.save_outputs = save_outputs
        self.examples_dir = Path(__file__).parent / "examples"
        self.results: list[tuple[str, bool, str]] = []

    # Notebooks that require LLM (Ollama or API keys)
    LLM_NOTEBOOKS = {
        "llm_description_generation.ipynb",
        "enterprise_demo_with_ollama.ipynb",
    }

    def get_notebooks(self) -> list[Path]:
        """Get all notebook files to run, including subdirectories."""
        # Get notebooks from main examples directory
        notebooks = list(self.examples_dir.glob("*.ipynb"))

        # Get notebooks from sql_files subdirectory
        sql_files_dir = self.examples_dir / "sql_files"
        if sql_files_dir.exists():
            notebooks.extend(sql_files_dir.glob("*.ipynb"))

        notebooks = sorted(notebooks)

        # Filter out notebooks that require LLM if requested
        if self.skip_llm:
            notebooks = [nb for nb in notebooks if nb.name not in self.LLM_NOTEBOOKS]

        return notebooks

    def run_notebook(self, notebook_path: Path) -> tuple[bool, str]:
        """
        Run a single notebook file.

        Returns:
            Tuple of (success, output/error message)
        """
        # LLM notebooks need longer timeout (5 minutes) due to model inference
        is_llm_notebook = notebook_path.name in self.LLM_NOTEBOOKS
        timeout = 300 if is_llm_notebook else 60

        # Set working directory to notebook's parent directory
        # This allows notebooks in subdirectories (like sql_files/) to work correctly
        working_dir = str(notebook_path.parent)

        nb = None
        try:
            nb = nbformat.read(notebook_path, as_version=4)
            client = NotebookClient(
                nb,
                timeout=timeout,
                kernel_name="python3",
                resources={"metadata": {"path": working_dir}},
            )
            client.execute()

            # Save outputs back to the notebook if requested
            if self.save_outputs:
                nbformat.write(nb, notebook_path)

            return True, "Executed successfully"

        except CellExecutionError as e:
            # Save the notebook with error outputs so the error is visible in the file
            if self.save_outputs and nb is not None:
                nbformat.write(nb, notebook_path)
            # Include traceback for better debugging
            error_msg = f"Cell execution error: {e.ename}: {e.evalue}"
            if hasattr(e, "traceback") and e.traceback:
                # Get last few lines of traceback for context
                tb_lines = str(e.traceback).split("\n")[-10:]
                error_msg += "\n    Traceback:\n    " + "\n    ".join(tb_lines)
            return False, error_msg
        except TimeoutError:
            # Save partial outputs on timeout
            if self.save_outputs and nb is not None:
                nbformat.write(nb, notebook_path)
            return False, f"Timeout after {timeout} seconds"
        except Exception as e:
            # Save partial outputs on other errors
            if self.save_outputs and nb is not None:
                nbformat.write(nb, notebook_path)
            return False, f"Error: {str(e)}"

    def print_header(self):
        """Print script header."""
        print()
        print("=" * 80)
        print(f"{BOLD}Running All clgraph Notebooks{RESET}")
        print("=" * 80)
        print()

    def print_notebook_start(self, notebook_name: str, idx: int, total: int):
        """Print when starting a notebook."""
        print(f"{BLUE}[{idx}/{total}]{RESET} {notebook_name}...", end=" ", flush=True)

    def print_notebook_result(self, success: bool, output: str):
        """Print the result of running a notebook."""
        if success:
            print(f"{GREEN}✓ PASS{RESET}")
        else:
            print(f"{RED}✗ FAIL{RESET}")

        if self.verbose or not success:
            print()
            print("-" * 80)
            print(output)
            print("-" * 80)
            print()

    def print_summary(self):
        """Print final summary of results."""
        print()
        print("=" * 80)
        print(f"{BOLD}Summary{RESET}")
        print("=" * 80)
        print()

        passed = sum(1 for _, success, _ in self.results if success)
        failed = len(self.results) - passed

        print(f"Total notebooks: {len(self.results)}")
        print(f"{GREEN}Passed: {passed}{RESET}")
        print(f"{RED}Failed: {failed}{RESET}")
        print()

        if failed > 0:
            print(f"{RED}{BOLD}Failed notebooks:{RESET}")
            for name, success, msg in self.results:
                if not success:
                    print(f"  {RED}✗{RESET} {name}")
                    print(f"      {msg}")
            print()

        print("=" * 80)
        print()

    def run_all(self) -> bool:
        """
        Run all notebooks and return whether all passed.

        Returns:
            True if all notebooks passed, False otherwise
        """
        notebooks = self.get_notebooks()

        if not notebooks:
            print(f"{YELLOW}No notebooks found in {self.examples_dir}{RESET}")
            return False

        self.print_header()

        if self.skip_llm:
            print(
                f"{YELLOW}⚠️  Skipping LLM notebooks (use without --skip-llm to include them){RESET}"
            )
            print()

        if self.save_outputs:
            print(f"{BLUE}📝 Saving outputs to notebook files{RESET}")
            print()

        for idx, notebook_path in enumerate(notebooks, 1):
            notebook_name = notebook_path.name
            self.print_notebook_start(notebook_name, idx, len(notebooks))

            success, output = self.run_notebook(notebook_path)
            self.results.append((notebook_name, success, output))

            self.print_notebook_result(success, output)

        self.print_summary()

        # Return True if all passed
        return all(success for _, success, _ in self.results)


def main():
    """Main entry point."""
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    skip_llm = "--skip-llm" in sys.argv
    save_outputs = "--save-outputs" in sys.argv

    runner = NotebookRunner(verbose=verbose, skip_llm=skip_llm, save_outputs=save_outputs)
    all_passed = runner.run_all()

    # Exit with non-zero code if any notebooks failed
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
