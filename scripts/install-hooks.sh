#!/bin/bash
# Install git hooks for the clgraph repository

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "📦 Installing git hooks for clgraph repository..."

# Check if we're in the right directory
if [ ! -f "$REPO_ROOT/pyproject.toml" ] || [ ! -d "$REPO_ROOT/src/clgraph" ]; then
    echo "❌ Error: Not in clgraph repository root"
    exit 1
fi

# Find the git hooks directory (handle submodule case)
if [ -d "$REPO_ROOT/.git/hooks" ]; then
    HOOKS_DIR="$REPO_ROOT/.git/hooks"
elif [ -f "$REPO_ROOT/.git" ]; then
    # Submodule case: .git is a file pointing to parent's git dir
    PARENT_GIT_DIR="$(cd "$REPO_ROOT" && git rev-parse --git-dir)"
    HOOKS_DIR="$PARENT_GIT_DIR/hooks"
else
    echo "⚠️  No .git directory found - skipping hook installation"
    echo "   (This is expected when running in a submodule context)"
    exit 0
fi

PRE_COMMIT="$HOOKS_DIR/pre-commit"
PRE_PUSH="$HOOKS_DIR/pre-push"

cat > "$PRE_COMMIT" << 'EOF'
#!/bin/bash
# Pre-commit hook that runs ruff for formatting and linting, and ty for type checking

set -e

echo "🔍 Running pre-commit checks..."

# Run ruff format to auto-format files
echo ""
echo "[1/3] Running ruff format..."
uv run ruff format .

# Stage any formatting changes
git add -u

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

chmod +x "$PRE_COMMIT"

cat > "$PRE_PUSH" << 'EOF'
#!/bin/bash
# Pre-push hook that mirrors CI lint + format checks.
# Catches issues BEFORE they reach GitHub Actions.
#
# Bypass (not recommended): git push --no-verify

set -e

echo "🔍 Running pre-push CI-equivalent checks..."

# Run the same commands as .github/workflows/ci.yml "Lint & Format" job
echo ""
echo "[1/2] ruff check . (lint, same as CI)"
if ! uv run ruff check .; then
    echo "❌ Lint failed. Fix with: uv run ruff check --fix ."
    exit 1
fi

echo ""
echo "[2/2] ruff format --check . (formatting, same as CI)"
if ! uv run ruff format --check .; then
    echo "❌ Formatting issues. Fix with: uv run ruff format ."
    exit 1
fi

echo ""
echo "✅ All pre-push checks passed!"
exit 0
EOF

chmod +x "$PRE_PUSH"

echo "✅ Git hooks installed successfully!"
echo ""
echo "pre-commit will run on every commit:"
echo "  - Ruff auto-format (automatically formats and stages changes)"
echo "  - Ruff linting"
echo "  - Type checking (warnings only)"
echo ""
echo "pre-push will run before every push (mirrors CI):"
echo "  - Ruff check ."
echo "  - Ruff format --check ."
echo ""
echo "To bypass: git commit --no-verify / git push --no-verify (not recommended)"
