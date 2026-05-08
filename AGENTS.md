# Repository Guidelines

## Project Structure & Module Organization

This repository is a Go HTTP proxy exposing an OpenAI Responses-compatible API backed by Anthropic Messages.

- `cmd/rap/`: executable entry point and environment wiring.
- `internal/openai/`: OpenAI Responses request, response, tool, and error types.
- `internal/anthropic/`: Anthropic Messages types and upstream HTTP client.
- `internal/convert/`: bidirectional OpenAI/Anthropic conversion logic.
- `internal/server/`: HTTP routing and `/v1/responses` handler.
- `internal/stream/`: SSE bridge from Anthropic stream events to Responses stream events.
- `internal/state/`: in-memory `previous_response_id` transcript store and tool-call ID index.
- `docs/api/`: local API reference notes.
- `scripts/`: developer scripts such as `scripts/run-proxy.sh`.

Tests live beside their package as `*_test.go`.

## Build, Test, and Development Commands

- `go test ./...`: run all tests.
- `GOCACHE="$PWD/.gocache" go test -count=1 ./...`: run fresh tests with a repo-local cache.
- `go run ./cmd/rap`: start the proxy directly.
- `ANTHROPIC_API_KEY=sk-ant-... ./scripts/run-proxy.sh`: start with editable defaults from the script.
- `gofmt -w <files>`: format Go files before committing.

The proxy defaults to `127.0.0.1:8180`; clients should use `http://127.0.0.1:8180/v1`.

## Coding Style & Naming Conventions

Use standard Go style: tabs via `gofmt`, short package names, exported identifiers only when used across packages, and clear structs for API JSON shapes. Keep packages focused by boundary: transport in `server`/`anthropic`, conversion in `convert`, and state in `state`.

Preserve unknown or loosely typed API fields with `json.RawMessage` where the upstream schema may evolve.

Tool-call compatibility is a core boundary. OpenAI-facing `function_call.call_id` and `function_call_output.call_id` must be resolved through `state.Store` before Anthropic requests are created. Do not pass OpenAI call IDs directly as Anthropic `tool_result.tool_use_id` unless the resolver explicitly maps them to the same value.

When changing upstream error handling, keep detailed diagnostic context in logs for compatibility debugging, but never log API keys or request authorization headers.

## Testing Guidelines

Use Go’s standard `testing` package. Prefer focused tests per behavior, for example `TestCreateResponseToMessageConvertsCoreFieldsAndTools`. Avoid real network listeners; use custom `http.RoundTripper` mocks for upstream calls. Add stream tests for SSE event translation and conversion tests for tool-call loops.

For tool-call changes, cover non-streaming and streaming loops, `previous_response_id` continuation, unique call-ID recovery without `previous_response_id`, ambiguous call IDs, duplicate tool results, and local 400 errors that must not call upstream.

Run `GOCACHE="$PWD/.gocache" go test -count=1 ./...` before reporting work complete.

## Commit & Pull Request Guidelines

Commits follow the current history style:

```text
✨ feat: add Anthropic-backed Responses proxy
```

Use an emoji, a conventional type (`feat`, `fix`, `docs`, `test`, `chore`, etc.), an imperative summary, and a body for non-trivial changes. After committing, inspect `git log -1 --pretty=full` to ensure body line breaks render correctly.

Pull requests should include a short summary, test results, linked issues if any, and notes for configuration or behavior changes. Include screenshots only for UI work; this project currently has no UI.

## Security & Configuration Tips

Never commit API keys. Use `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `ANTHROPIC_BASE_URL`, and `PROXY_ADDR` from the environment or local script edits. Treat the proxy as local/trusted-network software; client `Authorization` is accepted loosely by design.
