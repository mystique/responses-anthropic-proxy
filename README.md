# rap

Local OpenAI Responses API compatibility proxy backed by Anthropic Messages.

## Run

Create either a local `.env` file or a local `rap.config.json` file in the project root.

For `.env`:

```sh
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6
ANTHROPIC_BASE_URL=https://api.anthropic.com
PROXY_ADDR=127.0.0.1:8180
```

For `rap.config.json`:

```json
{
  "upstream": {
    "base_url": "https://api.anthropic.com",
    "api_key": "sk-ant-..."
  },
  "service": {
    "api_key": "local-client-key",
    "listen_addr": "127.0.0.1:8180"
  },
  "models": {
    "gpt-5": "claude-sonnet-4-6",
    "gpt-5-mini": "claude-haiku-4-6"
  },
  "config_password": "local-config-password",
  "default_model": "claude-sonnet-4-6"
}
```

The service loads `.env` and `rap.config.json` automatically on startup. Set `RAP_CONFIG=/path/to/rap.config.json` to use a config file outside the project tree. Existing shell environment variables take precedence over values in both local files. `config_password` is only used for the local configuration page and is never shown or edited by that page.

Environment overrides:

- `ANTHROPIC_API_KEY` overrides `upstream.api_key`
- `ANTHROPIC_BASE_URL` overrides `upstream.base_url`
- `ANTHROPIC_MODEL` overrides `default_model`
- `RAP_API_KEY` overrides `service.api_key`
- `PROXY_ADDR` overrides `service.listen_addr`

Start directly:

```sh
go run ./cmd/rap
```

Or use the helper script:

```sh
./scripts/run-proxy.sh
```

The `.env` file is ignored by git and should not be committed.

Open the local configuration page after startup:

```text
http://127.0.0.1:8180/config
```

If `config_password` is set in `rap.config.json`, the page requires that password before showing settings. If it is empty or omitted, the page opens without login. Saving writes `rap.config.json`; restart `rap` for saved changes to affect proxy behavior.

Build the short command binary:

```sh
go build -o rap ./cmd/rap
```

Defaults:

- `PROXY_ADDR=127.0.0.1:8180`
- `ANTHROPIC_MODEL=claude-sonnet-4-6`
- `ANTHROPIC_BASE_URL=https://api.anthropic.com`

`ANTHROPIC_API_KEY` is required.

When `models` is configured, the OpenAI request `model` must match one of the configured keys. The mapped value is sent upstream as the Anthropic model. Unknown models return `400 unsupported_model` locally.

Point Codex or another Responses API client at:

```text
http://127.0.0.1:8180/v1
```

Clients must send `Authorization: Bearer <service.api_key>`, matching the configured local service key exactly. If no local service key is configured, the proxy falls back to `ANTHROPIC_API_KEY` for backward compatibility. The proxy authenticates upstream with `x-api-key` plus `anthropic-version: 2023-06-01`.

## Scope

Implemented:

- `POST /v1/responses`
- `GET /v1/responses/{response_id}` from local in-memory state
- `DELETE /v1/responses/{response_id}` from local in-memory state
- `POST /v1/responses/{response_id}/cancel` for queued/in-progress local records
- Non-streaming and SSE streaming responses
- Function tools and `tool_use` / `function_call_output` loops
- Web search tools via Anthropic server-side web search
- In-memory `previous_response_id` transcript continuation with 24 hour TTL
- Tool-call ID compatibility between OpenAI Responses and Anthropic Messages
- Detailed upstream error diagnostics for compatibility debugging

Limitations:

- Response retrieval, deletion, cancellation, and `previous_response_id` only work for records created in the current process.
- Cancellation is local state mutation; completed Anthropic requests cannot be cancelled after the fact.
- Unknown endpoints return OpenAI-style `404 not_found`; wrong methods return `405 method_not_allowed`.

## Implementation Details

### Request Flow

`POST /v1/responses` is handled in `internal/server`. The handler:

1. Decodes the OpenAI Responses request.
2. Restores the previous Anthropic transcript from `previous_response_id`, when present.
3. Detects `function_call_output` items and resolves their OpenAI `call_id` values through `internal/state`.
4. Converts the OpenAI request to an Anthropic Messages request in `internal/convert`.
5. Sends the converted request to Anthropic through `internal/anthropic`.
6. Converts the Anthropic response or stream events back to OpenAI Responses output.
7. Saves the full transcript and response in the in-memory store.

The store is process-local. Restarting the proxy clears response history, transcript history, and tool-call indexes.

### Tool-Call ID Mapping

OpenAI and Anthropic use similar but distinct tool-result references:

- OpenAI clients receive `function_call.call_id` and later send `function_call_output.call_id`.
- Anthropic expects `tool_result.tool_use_id` to match the previous assistant `tool_use.id`.

The proxy keeps these concepts separate. When an Anthropic `tool_use` is converted to an OpenAI `function_call`, the response is saved with a `ToolCallRecord` containing:

- `OpenAICallID`
- `AnthropicToolUseID`
- `ResponseID`
- tool name and arguments
- output index
- creation and resolved timestamps

Today the exposed OpenAI call ID usually matches the Anthropic tool-use ID, but the code does not rely on that. `function_call_output` conversion uses a resolver supplied by the server. If the resolver cannot map the OpenAI call ID to an Anthropic tool-use ID, the proxy returns a local OpenAI-style `400 invalid_request_error` instead of sending a bad request upstream.

### Tool-Result Continuation Rules

For requests with `function_call_output`:

- With `previous_response_id`, the proxy looks up the tool call within that response's stored transcript.
- Without `previous_response_id`, the proxy searches the in-memory tool-call index by `call_id`.
- If exactly one matching response is found, the proxy restores that response's transcript automatically.
- If no match is found, it returns `400` with code `tool_call_not_found`.
- If multiple responses contain the same call ID, it returns `400` with code `ambiguous_tool_call_id`; the client must send `previous_response_id`.
- If the tool call was already resolved, it returns `400` with code `tool_call_already_resolved`.

Converted Anthropic user messages place all `tool_result` blocks before text or image blocks, which matches Anthropic's required tool-result ordering.

### Tool-Result Content

`function_call_output.output` supports:

- a string, forwarded as Anthropic `tool_result.content`
- a content list containing supported text and image blocks

Unsupported tool-result content types return a local `400 invalid_request_error`. This keeps schema incompatibilities visible at the proxy boundary instead of surfacing as less actionable Anthropic errors.

### Streaming

Streaming uses `internal/stream` to translate Anthropic SSE events into Responses stream events. The bridge also reconstructs the assistant transcript from streamed text and `tool_use` content blocks. After the stream completes, the server saves:

- the OpenAI response object
- the full Anthropic transcript
- any tool-call mappings discovered in streamed `tool_use` blocks

Tool-result follow-up requests work for both non-streaming and streaming responses.

### Upstream Error Diagnostics

When an upstream Anthropic request fails, the proxy returns an OpenAI-style error to the client and logs a structured JSON diagnostic record with event name `upstream_error_context`.

The log includes:

- local response ID and creation time
- request method and path
- original OpenAI request body
- converted Anthropic request body
- `previous_response_id` and resolved response ID
- tool-call IDs from the request
- resolved OpenAI call ID to Anthropic tool-use ID records
- previous and newly added message counts
- whether the request was streaming
- upstream HTTP status, error type, message, and raw body when available

The log intentionally does not include client `Authorization` headers or the Anthropic API key.

Anthropic `invalid_request_error` responses are returned as local `400 invalid_request_error`. Network errors and upstream 5xx-style failures remain `502 upstream_error`.

## Testing

Run the full suite with a repo-local cache:

```sh
GOCACHE="$PWD/.gocache" go test -count=1 ./...
```

The tests cover request conversion, response conversion, state cloning, response retrieval/deletion/cancellation, stream bridging, tool-call ID resolution, duplicate tool-output rejection, unique call-ID recovery, ambiguity handling, and upstream error diagnostic logging.
