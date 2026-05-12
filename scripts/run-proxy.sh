#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$PROJECT_ROOT/.env" && ! -f "$PROJECT_ROOT/rap.config.json" ]]; then
  echo "Missing configuration. Create .env in the project root with:"
  echo "  ANTHROPIC_API_KEY=sk-ant-..."
  echo "  RAP_API_KEY=local-client-key"
  echo "  ANTHROPIC_MODEL=claude-sonnet-4-6"
  echo "  ANTHROPIC_BASE_URL=https://api.anthropic.com"
  echo "  PROXY_ADDR=127.0.0.1:8180"
  echo ""
  echo "Or create rap.config.json for upstream, service, model mappings, and listen_addr."
  exit 1
fi

echo "Starting rap"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  echo "  env config: $PROJECT_ROOT/.env"
fi
if [[ -f "$PROJECT_ROOT/rap.config.json" ]]; then
  echo "  json config: $PROJECT_ROOT/rap.config.json"
fi

cd "$PROJECT_ROOT"
exec go run ./cmd/rap
