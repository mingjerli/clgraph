#!/bin/bash
# Test build script for PyPI release verification
# This script tests the build process, package validation, and installation

set -e  # Exit on error

echo "=========================================="
echo "ClPipe PyPI Build Test Script"
echo "=========================================="

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Step 1: Clean up old builds
echo ""
echo "Step 1: Cleaning up old build artifacts..."
rm -rf dist/ build/ *.egg-info src/*.egg-info

# Step 2: Build the package
echo ""
echo "Step 2: Building the package..."
uv run python -m build

# Step 3: Check the distribution with twine
echo ""
echo "Step 3: Validating distribution files with twine..."
uv run twine check dist/*

# Step 4: Test installation in a clean virtual environment
echo ""
echo "Step 4: Testing installation in clean virtual environment..."

# Create temp directory for venv
TEMP_VENV=$(mktemp -d)/test_clpipe_venv

# Create venv with uv
uv venv "$TEMP_VENV"

# Activate venv
source "$TEMP_VENV/bin/activate"

echo "Installing from wheel..."
uv pip install dist/*.whl

echo "Testing import and version..."
python -c "
import clpipe
print(f'✓ Successfully imported clpipe')
print(f'✓ Version: {clpipe.__version__}')

# Test basic functionality
from clpipe import Pipeline, SQLColumnTracer
print('✓ Successfully imported Pipeline and SQLColumnTracer')

# Test that examples would work
try:
    tracer = SQLColumnTracer('SELECT 1 as col')
    print('✓ SQLColumnTracer instantiation works')
except Exception as e:
    print(f'✗ SQLColumnTracer failed: {e}')
    exit(1)
"

# Deactivate and cleanup
deactivate
rm -rf "$TEMP_VENV"

echo ""
echo "=========================================="
echo "✓ All build tests passed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the distribution files in dist/"
echo "2. Test examples work after installation"
echo "3. Set up PyPI account with 2FA"
echo "4. Generate API token for publishing"
echo "5. Publish to TestPyPI first: twine upload --repository testpypi dist/*"
echo "6. Publish to PyPI: twine upload dist/*"
