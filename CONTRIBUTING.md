# Contributing to clgraph

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/mingjerli/clgraph.git
cd clgraph
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install in editable mode with dev dependencies:
```bash
pip install -e ".[dev]"
```

## Running Tests

Run the full test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=clgraph --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_lineage.py
```

## Code Style

We use:
- **ruff** for code formatting and linting

Format code:
```bash
ruff format src/ tests/
```

Lint:
```bash
ruff check src/ tests/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Format and lint your code
7. Commit with clear messages
8. Push to your fork
9. Create a Pull Request

## Commit Messages

Use clear, descriptive commit messages:
- `feat: add support for window functions`
- `fix: correct handling of nested CTEs`
- `docs: update README with examples`
- `test: add tests for star notation`

## Testing Guidelines

- Write tests for all new features
- Maintain or improve code coverage
- Test edge cases and error conditions
- Use descriptive test names

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Include usage examples for new features

## Questions?

Open an issue for questions or discussions about contributions.
