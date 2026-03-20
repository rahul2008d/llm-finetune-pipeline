#!/usr/bin/env bash
# Run the training pipeline locally (for development/testing)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

CONFIG_FILE="${1:-$PROJECT_ROOT/configs/training/default.yaml}"

echo "=== Local Training Run ==="
echo "Config: $CONFIG_FILE"
echo ""

cd "$PROJECT_ROOT"
python -m training.runner --config "$CONFIG_FILE"
