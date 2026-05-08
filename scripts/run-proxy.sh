#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
  echo "Missing .env. Create one in the project root with:"
  echo "  ANTHROPIC_API_KEY=sk-ant-..."
  echo "  ANTHROPIC_MODEL=claude-sonnet-4-6"
  echo "  ANTHROPIC_BASE_URL=https://api.anthropic.com"
  echo "  PROXY_ADDR=127.0.0.1:8180"
  exit 1
fi

echo "Starting rap"
echo "  config: $PROJECT_ROOT/.env"

cd "$PROJECT_ROOT"
exec go run ./cmd/rap
