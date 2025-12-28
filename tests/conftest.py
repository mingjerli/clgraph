"""
Pytest configuration for clgraph tests.

Provides:
- --run-llm flag to enable LLM-dependent tests
- LLM marker for tests requiring Ollama
"""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-llm",
        action="store_true",
        default=False,
        help="Run LLM-dependent tests (requires Ollama running locally)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "llm: mark test as requiring LLM (Ollama)")


def pytest_collection_modifyitems(config, items):
    """Skip LLM tests unless --run-llm is specified."""
    if config.getoption("--run-llm"):
        return

    skip_llm = pytest.mark.skip(reason="Need --run-llm option to run")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip_llm)
