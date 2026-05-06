# uni-api

Local OpenAI Responses API compatibility proxy backed by Anthropic Messages.

## Run

```sh
ANTHROPIC_API_KEY=sk-ant-... go run ./cmd/proxy
```

Or use the editable script:

```sh
ANTHROPIC_API_KEY=sk-ant-... ./scripts/run-proxy.sh
```

Runtime parameters are at the top of [scripts/run-proxy.sh](scripts/run-proxy.sh).

Defaults:

- `PROXY_ADDR=127.0.0.1:8180`
- `ANTHROPIC_MODEL=claude-sonnet-4-6`
- `ANTHROPIC_BASE_URL=https://api.anthropic.com`

Point Codex or another Responses API client at:

```text
http://127.0.0.1:8180/v1
```

The proxy accepts any client `Authorization` header and always authenticates upstream with `x-api-key` plus `anthropic-version: 2023-06-01`.

## Scope

Implemented:

- `POST /v1/responses`
- Non-streaming and SSE streaming responses
- Function tools and `tool_use` / `function_call_output` loops
- In-memory `previous_response_id` transcript continuation with 24 hour TTL

Other Responses API endpoints return OpenAI-style `501 not_implemented` JSON.
