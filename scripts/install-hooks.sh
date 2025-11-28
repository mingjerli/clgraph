#!/bin/bash
# Install git hooks for the clpipe repository

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "📦 Installing git hooks for clpipe repository..."

# Check if we're in the right directory
if [ ! -f "$REPO_ROOT/pyproject.toml" ] || [ ! -d "$REPO_ROOT/src/clpipe" ]; then
    echo "❌ Error: Not in clpipe repository root"
    exit 1
fi

# Create pre-commit hook
HOOK_PATH="$REPO_ROOT/.git/hooks/pre-commit"

cat > "$HOOK_PATH" << 'EOF'
#!/bin/bash
# Pre-commit hook that runs ruff for formatting and linting, and ty for type checking

set -e

echo "🔍 Running pre-commit checks..."

# Run ruff format check
echo ""
echo "[1/3] Running ruff format check..."
if ! uv run ruff format --check src/ tests/ examples/; then
    echo "❌ Ruff formatting check failed"
    echo "💡 Run 'uv run ruff format src/ tests/ examples/' to auto-format"
    exit 1
fi

# Run ruff for linting
echo ""
echo "[2/3] Running ruff lint..."
if ! uv run ruff check src/ tests/ examples/; then
    echo "❌ Ruff linting failed"
    echo "💡 Run 'uv run ruff check --fix src/ tests/ examples/' to auto-fix"
    exit 1
fi

# Run ty for type checking
echo ""
echo "[3/3] Running ty type checker..."
if ! uv run ty check src/; then
    echo "⚠️  ty type checking found issues (not blocking)"
    # Don't exit 1 here since we want to allow commits with warnings
fi

echo ""
echo "✅ All pre-commit checks passed!"
exit 0
EOF

chmod +x "$HOOK_PATH"

echo "✅ Git hooks installed successfully!"
echo ""
echo "The pre-commit hook will now run:"
echo "  - Ruff format check"
echo "  - Ruff linting"
echo "  - Type checking (warnings only)"
echo ""
echo "To bypass the hook (not recommended), use: git commit --no-verify"
