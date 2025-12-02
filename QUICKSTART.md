# Quick Start Guide

## For Library Developers

### Setup Development Environment

```bash
# Navigate to the library
cd clgraph

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

### Run Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_lineage.py

# Run specific test class
pytest tests/test_lineage.py::TestQueryUnitBasics

# Run specific test
pytest tests/test_lineage.py::TestQueryUnitBasics::test_query_unit_creation
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

### Building the Package

```bash
# Build distribution packages
python -m build

# This creates:
# - dist/clgraph-0.1.0.tar.gz (source distribution)
# - dist/clgraph-0.1.0-py3-none-any.whl (wheel)
```

### Publishing to PyPI

```bash
# Install twine
pip install twine

# Check the distribution
twine check dist/*

# Upload to Test PyPI (for testing)
twine upload --repository testpypi dist/*

# Upload to PyPI (for production)
twine upload dist/*
```

## For Library Users

### Installation

```bash
# From PyPI (once published)
pip install clgraph

# From source
pip install git+https://github.com/yourusername/clgraph.git

# For development
git clone https://github.com/yourusername/clgraph.git
cd clgraph
pip install -e .
```

### Basic Usage

```python
from clgraph import SQLColumnTracer

sql = """
SELECT
  u.name,
  o.total_amount
FROM users u
JOIN orders o ON u.id = o.user_id
"""

# Create tracer
tracer = SQLColumnTracer(sql, dialect="bigquery")

# Build lineage graph
lineage = tracer.build_column_lineage_graph()

# Query the lineage
# Note: The API is still being finalized
# Check the examples/ directory for current usage
```

### Running Examples

```bash
# Simple column lineage example
python examples/simple_example.py

# Multi-query pipeline example
python examples/pipeline_example.py
```

## For CI/CD

### GitHub Actions

The repository includes a GitHub Actions workflow (`.github/workflows/test.yml`) that:
- Runs tests on Python 3.9, 3.10, 3.11, 3.12
- Tests on Ubuntu, macOS, and Windows
- Runs linting and formatting checks
- Tests examples

The workflow runs automatically on:
- Pull requests to `main`
- Pushes to `main`

### Local Pre-commit Checks

```bash
# Run all checks before committing
black src/ tests/
ruff check src/ tests/
mypy src/
pytest
```

## Troubleshooting

### Import Errors

If you get import errors after installation:
```bash
# Make sure you're in the right directory
cd clgraph

# Reinstall in editable mode
uv pip install -e .

# Verify installation
python -c "from clgraph import SQLColumnTracer; print('Success!')"
```

### Test Failures

If tests fail:
```bash
# Check if dependencies are installed
uv pip list

# Reinstall dependencies
uv pip install -e ".[dev]"

# Run tests with verbose output to see details
pytest -vv
```

### Virtual Environment Issues

```bash
# Remove and recreate virtual environment
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Next Steps

1. Check out the [README.md](README.md) for detailed documentation
2. Read the [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
3. Review the [DEPLOYMENT_PLAN.md](DEPLOYMENT_PLAN.md) for the overall project strategy
4. Explore the code in `src/clgraph/parser.py`
5. Run the examples in `examples/`
6. Write your own queries and test them!
