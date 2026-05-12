#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/dist/rap}"

mkdir -p "$(dirname -- "$OUTPUT")"

echo "Building rap binary"
echo "  output: $OUTPUT"

cd "$PROJECT_ROOT"
CGO_ENABLED="${CGO_ENABLED:-0}" GOCACHE="${GOCACHE:-$PROJECT_ROOT/.gocache}" go build -trimpath -o "$OUTPUT" ./cmd/rap

echo "Done"
