.PHONY: help setup test test-verbose test-coverage test-docs test-docs-verbose test-llm clean lint lint-fix lint-fix-unsafe format format-check type-check check check-examples-verbose pre-commit install install-all install-light install-hooks build publish-test publish docs rebuild-notebooks rebuild-notebooks-llm

# Default target
help:
	@echo "Available targets:"
	@echo "  make setup          - Setup Python environment with uv"
	@echo "  make test           - Run tests locally"
	@echo "  make test-verbose   - Run tests with verbose output"
	@echo "  make test-coverage  - Run tests with coverage report"
	@echo "  make test-docs      - Test README documentation examples"
	@echo "  make test-docs-verbose - Test README examples with detailed output"
	@echo "  make test-llm       - Test LLM examples with Ollama (requires Ollama running)"
	@echo "  make lint           - Run linting checks (ruff)"
	@echo "  make lint-fix       - Auto-fix linting issues"
	@echo "  make lint-fix-unsafe- Auto-fix linting issues (including unsafe fixes)"
	@echo "  make format         - Format code with ruff"
	@echo "  make format-check   - Check formatting without modifying files"
	@echo "  make type-check     - Run type checking with mypy"
	@echo "  make check          - Run all checks (format, lint, type-check, test, test-docs)"
	@echo "  make check-examples - Run all examples (skips LLM examples)"
	@echo "  make check-examples-verbose - Run examples with verbose output (skips LLM)"
	@echo "  make check-examples-llm - Run ALL examples including LLM (requires Ollama)"
	@echo "  make rebuild-notebooks - Run all notebooks and save outputs (skips LLM)"
	@echo "  make rebuild-notebooks-llm - Rebuild ALL notebooks including LLM (requires Ollama)"
	@echo "  make pre-commit     - Run pre-commit hook (ruff + ty)"
	@echo "  make install        - Install package in editable mode (dev deps)"
	@echo "  make install-all    - Install package with ALL optional deps (llm, mcp, airflow)"
	@echo "  make install-light  - Install package without dev deps (minimal)"
	@echo "  make install-hooks  - Install git pre-commit hooks"
	@echo "  make build          - Build distribution packages"
	@echo "  make publish-test   - Publish to TestPyPI"
	@echo "  make publish        - Publish to PyPI"
	@echo "  make clean          - Remove build artifacts and cache files"
	@echo "  make docs           - Generate documentation"

# Setup Python environment with uv
setup:
	@echo "Setting up Python environment with uv..."
	uv venv
	@echo "Installing all dependencies..."
	uv sync --extra dev --extra llm --extra mcp --extra airflow
	@echo "Installing git hooks..."
	@bash scripts/install-hooks.sh
	@echo ""
	@echo "✓ Setup complete!"
	@echo "Activate the environment with: source .venv/bin/activate"

# Install package in editable mode (dev dependencies)
install:
	@echo "Installing package in editable mode (dev deps)..."
	uv sync --extra dev
	@echo "✓ Installation complete!"

# Install package with ALL optional dependencies
install-all:
	@echo "Installing package with ALL optional dependencies..."
	uv sync --extra dev --extra llm --extra mcp --extra airflow
	@echo "✓ Installation complete! Includes: dev, llm, mcp, airflow"

# Install package without dev deps (minimal)
install-light:
	@echo "Installing package in editable mode (minimal)..."
	uv sync
	@echo "✓ Installation complete!"

# Install git hooks
install-hooks:
	@bash scripts/install-hooks.sh

# Run tests locally
test:
	@echo "Running tests..."
	uv run python -m pytest tests/ -v
	@echo "✓ Tests complete!"

# Run tests with verbose output
test-verbose:
	@echo "Running tests with verbose output..."
	uv run python -m pytest tests/ -vv
	@echo "✓ Tests complete!"

# Run tests with coverage report
test-coverage:
	@echo "Running tests with coverage..."
	uv pip install pytest-cov
	uv run python -m pytest tests/ -v --cov=clpipe --cov-report=term-missing --cov-report=html
	@echo "✓ Coverage report generated in htmlcov/"

# Test README documentation examples
test-docs:
	@echo "Testing README documentation examples..."
	uv run pytest tests/test_readme_examples.py -v -s
	@echo "✓ Documentation tests complete!"

# Test README with standalone script (detailed output)
test-docs-verbose:
	@echo "Testing README examples with detailed output..."
	uv run python tests/markdown_examples.py README.md -v
	@echo "✓ Documentation tests complete!"

# Test LLM-dependent examples (requires Ollama running locally)
test-llm:
	@echo "Testing LLM-dependent examples (requires Ollama)..."
	@echo "Prerequisites: ollama serve && ollama pull llama3.1:8b"
	uv run pytest tests/test_readme_llm_examples.py -v -s --run-llm
	@echo "✓ LLM tests complete!"

# Run linting checks
lint:
	@echo "Running linting checks..."
	uv run ruff check src/ tests/
	@echo "✓ Linting complete!"

# Auto-fix linting issues
lint-fix:
	@echo "Auto-fixing linting issues..."
	uv run ruff check --fix src/ tests/
	@echo "✓ Auto-fix complete!"

# Auto-fix with unsafe fixes
lint-fix-unsafe:
	@echo "Auto-fixing linting issues (including unsafe fixes)..."
	uv run ruff check --fix --unsafe-fixes src/ tests/
	@echo "✓ Auto-fix complete!"

# Format code with ruff
format:
	@echo "Formatting code with ruff..."
	uv run ruff format src/ tests/ examples/
	@echo "✓ Formatting complete!"

# Check formatting without modifying files
format-check:
	@echo "Checking code formatting..."
	uv run ruff format --check src/ tests/ examples/
	@echo "✓ Format check complete!"

# Run type checking
type-check:
	@echo "Running type checks..."
	-uv run ty check src/
	@echo "✓ Type checking complete!"

# Run all checks
check: format lint type-check test test-docs
	@echo ""
	@echo "✓ All checks passed!"

# Run all notebooks (skips LLM notebooks by default)
check-examples:
	@echo "Running all notebooks (skipping LLM notebooks)..."
	@uv run python run_all_notebooks.py --skip-llm

# Run all notebooks with verbose output
check-examples-verbose:
	@echo "Running all notebooks (verbose, skipping LLM notebooks)..."
	@uv run python run_all_notebooks.py --verbose --skip-llm

# Run ALL notebooks including LLM (requires Ollama)
check-examples-llm:
	@echo "Running ALL notebooks including LLM (requires Ollama)..."
	@uv run python run_all_notebooks.py --verbose

# Rebuild all notebooks (run and save outputs)
rebuild-notebooks:
	@echo "Rebuilding all notebooks (running and saving outputs)..."
	@uv run python run_all_notebooks.py --skip-llm --save-outputs
	@echo "✓ Notebooks rebuilt with outputs!"

# Rebuild ALL notebooks including LLM (requires Ollama running)
rebuild-notebooks-llm:
	@echo "Rebuilding ALL notebooks including LLM (requires Ollama)..."
	@uv run python run_all_notebooks.py --save-outputs
	@echo "✓ All notebooks rebuilt with outputs!"

# Pre-commit: format code, fix lint issues, and run the git pre-commit hook
pre-commit:
	@echo "Formatting code with ruff..."
	@uv run ruff format src/ tests/ examples/
	@echo "Fixing lint issues with ruff..."
	@uv run ruff check --fix src/ tests/ examples/
	@if [ -f ../.git/hooks/pre-commit ]; then \
		../.git/hooks/pre-commit; \
	else \
		echo "❌ Pre-commit hook not found at ../.git/hooks/pre-commit"; \
		exit 1; \
	fi

# Clean build artifacts and cache
clean:
	@echo "Cleaning build artifacts and cache..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✓ Clean complete!"

# Build distribution packages
build: clean
	@echo "Building distribution packages..."
	uv pip install build
	uv run python -m build
	@echo "✓ Build complete! Check dist/ directory"

# Publish to TestPyPI
publish-test: build
	@echo "Publishing to TestPyPI..."
	uv pip install twine
	uv run twine check dist/*
	uv run twine upload --repository testpypi dist/*
	@echo "✓ Published to TestPyPI!"

# Publish to PyPI
publish: build
	@echo "Publishing to PyPI..."
	@echo "WARNING: This will publish to production PyPI!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		uv pip install twine; \
		uv run twine check dist/*; \
		uv run twine upload dist/*; \
		echo "✓ Published to PyPI!"; \
	else \
		echo "Cancelled."; \
	fi

# Generate documentation (placeholder for future)
docs:
	@echo "Documentation generation not yet implemented"
	@echo "See README.md for current documentation"

# Development workflow shortcuts
dev-setup: setup
	@echo "Installing pre-commit hooks (if available)..."
	@if [ -f .git/hooks/pre-commit ]; then \
		echo "Pre-commit hooks already configured"; \
	else \
		echo "No pre-commit hooks found"; \
	fi

# Quick test run (just run without verbose)
quick-test:
	@uv run pytest tests/ -q

# Run a specific test file
test-lineage:
	@echo "Running lineage tests..."
	@uv run pytest tests/test_lineage.py -v

test-multi-query:
	@echo "Running multi-query tests..."
	@uv run pytest tests/test_multi_query.py -v

# Run examples
example-simple:
	@echo "Running simple example..."
	@uv run python examples/simple_example.py

example-pipeline:
	@echo "Running pipeline example..."
	@uv run python examples/pipeline_example.py

# Show project info
info:
	@echo "Project: clpipe"
	@echo "Version: 0.1.0"
	@echo "Python: $$(python --version 2>&1 || echo 'Not activated')"
	@echo "Location: $$(pwd)"
	@echo ""
	@echo "Virtual environment:"
	@if [ -d .venv ]; then \
		echo "  ✓ .venv exists"; \
	else \
		echo "  ✗ .venv not found (run 'make setup')"; \
	fi
