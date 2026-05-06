#!/usr/bin/env bash
set -euo pipefail

# Edit these values for local runs, or override them from the shell:
#   ANTHROPIC_API_KEY=... ANTHROPIC_MODEL=... ./scripts/run-proxy.sh
: "${ANTHROPIC_API_KEY:=sk-e3f24b4040bb4c118e4813e02fb900fd282025df400c83bcd0d6cb675b8559f6}"
: "${ANTHROPIC_MODEL:=claude-sonnet-4-6}"
: "${ANTHROPIC_BASE_URL:=http://127.0.0.1:8080}"
: "${PROXY_ADDR:=127.0.0.1:8180}"

if [[ -z "$ANTHROPIC_API_KEY" ]]; then
  echo "ANTHROPIC_API_KEY is required."
  echo "Set it near the top of scripts/run-proxy.sh or pass it when running:"
  echo "  ANTHROPIC_API_KEY=sk-ant-... ./scripts/run-proxy.sh"
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

export ANTHROPIC_API_KEY
export ANTHROPIC_MODEL
export ANTHROPIC_BASE_URL
export PROXY_ADDR

echo "Starting uni-api proxy"
echo "  listen: http://$PROXY_ADDR/v1"
echo "  upstream: $ANTHROPIC_BASE_URL"
echo "  model: $ANTHROPIC_MODEL"

cd "$PROJECT_ROOT"
exec go run ./cmd/proxy
