#!/bin/bash
# Setup linting tools for the project

set -e

echo "🔧 Setting up linting tools..."

# Install pre-commit and linting tools
echo "📦 Installing linting dependencies..."
pip3 install --user pre-commit black flake8 isort

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Run initial check on all files
echo "🔍 Running initial lint check on all files..."
pre-commit run --all-files || true

echo "✅ Linting setup complete!"
echo ""
echo "Usage:"
echo "  - Hooks run automatically on git commit"
echo "  - Manual run: pre-commit run --all-files"
echo "  - Format Python: black ."
echo "  - Check Python: flake8"
echo "  - Skip hooks: git commit --no-verify"
