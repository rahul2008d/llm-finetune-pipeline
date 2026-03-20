#!/usr/bin/env bash
# Local development setup script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== LLM Fine-Tuning Pipeline - Local Setup ==="

cd "$PROJECT_ROOT"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]"

# Set up pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Copy .env.example if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "WARNING: Update .env with your actual values before running the pipeline."
fi

echo ""
echo "=== Setup Complete ==="
echo "Activate your environment: source .venv/bin/activate"
echo "Run tests: make test"
echo "Run linter: make lint"
