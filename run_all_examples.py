#!/usr/bin/env python3
"""
Script to run all examples and verify they work correctly.

This script runs all example files in the examples/ directory and reports
which ones succeed or fail. It's useful for:
- Validating that examples work before releases
- Catching documentation drift
- Ensuring good user experience

This is separate from automated tests because:
- Examples might be slow or require external dependencies (e.g., LLM API keys)
- Examples focus on "does this still work?" vs. unit tests that verify correctness
- You can run this manually or in optional CI workflows

Usage:
    python run_all_examples.py [--verbose] [--skip-llm]

Options:
    --verbose    Show full output from each example
    --skip-llm   Skip examples that require LLM API keys
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# ANSI color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


class ExampleRunner:
    """Runs all example files and tracks results."""

    def __init__(self, verbose: bool = False, skip_llm: bool = False):
        self.verbose = verbose
        self.skip_llm = skip_llm
        self.examples_dir = Path(__file__).parent / "examples"
        self.results: List[Tuple[str, bool, str]] = []

    def get_examples(self) -> List[Path]:
        """Get all example Python files to run."""
        examples = sorted(self.examples_dir.glob("*.py"))

        # Filter out examples that require LLM API keys if requested
        if self.skip_llm:
            examples = [ex for ex in examples if "llm_description_generation" not in ex.name]

        return examples

    def run_example(self, example_path: Path) -> Tuple[bool, str]:
        """
        Run a single example file.

        Returns:
            Tuple of (success, output)
        """
        try:
            result = subprocess.run(
                ["python", str(example_path)],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout per example
                cwd=self.examples_dir.parent,  # Run from clgraph/ directory
            )

            output = result.stdout + result.stderr
            success = result.returncode == 0

            return success, output

        except subprocess.TimeoutExpired:
            return False, "ERROR: Timeout after 30 seconds"
        except Exception as e:
            return False, f"ERROR: {str(e)}"

    def print_header(self):
        """Print script header."""
        print()
        print("=" * 80)
        print(f"{BOLD}Running All clgraph Examples{RESET}")
        print("=" * 80)
        print()

    def print_example_start(self, example_name: str, idx: int, total: int):
        """Print when starting an example."""
        print(f"{BLUE}[{idx}/{total}]{RESET} {example_name}...", end=" ", flush=True)

    def print_example_result(self, success: bool, output: str):
        """Print the result of running an example."""
        if success:
            print(f"{GREEN}✓ PASS{RESET}")
        else:
            print(f"{RED}✗ FAIL{RESET}")

        if self.verbose or not success:
            # Show output if verbose or if it failed
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

        print(f"Total examples: {len(self.results)}")
        print(f"{GREEN}Passed: {passed}{RESET}")
        print(f"{RED}Failed: {failed}{RESET}")
        print()

        if failed > 0:
            print(f"{RED}{BOLD}Failed examples:{RESET}")
            for name, success, _ in self.results:
                if not success:
                    print(f"  {RED}✗{RESET} {name}")
            print()

        print("=" * 80)
        print()

    def run_all(self) -> bool:
        """
        Run all examples and return whether all passed.

        Returns:
            True if all examples passed, False otherwise
        """
        examples = self.get_examples()

        if not examples:
            print(f"{YELLOW}No examples found in {self.examples_dir}{RESET}")
            return False

        self.print_header()

        if self.skip_llm:
            print(
                f"{YELLOW}⚠️  Skipping LLM examples (use without --skip-llm to include them){RESET}"
            )
            print()

        for idx, example_path in enumerate(examples, 1):
            example_name = example_path.name
            self.print_example_start(example_name, idx, len(examples))

            success, output = self.run_example(example_path)
            self.results.append((example_name, success, output))

            self.print_example_result(success, output)

        self.print_summary()

        # Return True if all passed
        return all(success for _, success, _ in self.results)


def main():
    """Main entry point."""
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    skip_llm = "--skip-llm" in sys.argv

    runner = ExampleRunner(verbose=verbose, skip_llm=skip_llm)
    all_passed = runner.run_all()

    # Exit with non-zero code if any examples failed
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
