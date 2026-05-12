#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

mkdir -p "$tmp_dir/scripts"
cp "$project_root/scripts/run-proxy.sh" "$tmp_dir/scripts/run-proxy.sh"
mkdir -p "$tmp_dir/cmd/rap"
cat > "$tmp_dir/rap.config.json" <<'JSON'
{
  "upstream": {
    "api_key": "sk-ant-test"
  },
  "service": {
    "api_key": "rap-test-key",
    "listen_addr": "127.0.0.1:8180"
  }
}
JSON

cat > "$tmp_dir/go" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" > "$GO_STUB_ARGS_FILE"
SH
chmod +x "$tmp_dir/go"

set +e
output="$(
  cd "$tmp_dir"
  PATH="$tmp_dir:$PATH" GO_STUB_ARGS_FILE="$tmp_dir/go.args" ./scripts/run-proxy.sh
)"
status=$?
set -e

if [[ "$status" -ne 0 ]] || grep -q "Missing .env" <<<"$output"; then
	echo "run-proxy.sh rejected config-only startup"
	exit 1
fi

if [[ "$(cat "$tmp_dir/go.args")" != "run ./cmd/rap" ]]; then
  echo "run-proxy.sh did not invoke go run ./cmd/rap"
  exit 1
fi
