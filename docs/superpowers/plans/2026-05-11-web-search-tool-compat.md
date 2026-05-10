# Web Search Tool Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make OpenAI Responses requests using `tools:[{"type":"web_search"}]` or `tools:[{"type":"web_search_preview"}]` work through this Anthropic-backed proxy instead of failing locally with `unsupported_tool_type`.

**Architecture:** Add a typed mapping from OpenAI web search tool declarations to Anthropic's server tool `web_search_20250305`. Preserve Anthropic web search result blocks in stored transcript history, and expose search activity in OpenAI-compatible response output and SSE events using `web_search_call` items plus text annotations when citations are present.

**Tech Stack:** Go standard library, repository-local `internal/openai`, `internal/anthropic`, `internal/convert`, `internal/stream`, `internal/server`, standard `testing`, `gofmt`, `go test`.

---

## Reference Facts

Anthropic web search tool request shape:

```json
{
  "type": "web_search_20250305",
  "name": "web_search",
  "max_uses": 5,
  "allowed_domains": ["example.com"],
  "blocked_domains": ["example.org"],
  "user_location": {
    "type": "approximate",
    "city": "San Francisco",
    "region": "California",
    "country": "US",
    "timezone": "America/Los_Angeles"
  }
}
```

Anthropic non-stream response content can include:

```json
{
  "type": "server_tool_use",
  "id": "srvtoolu_123",
  "name": "web_search",
  "input": {"query": "latest OpenAI news"}
}
```

```json
{
  "type": "web_search_tool_result",
  "tool_use_id": "srvtoolu_123",
  "content": [
    {
      "type": "web_search_result",
      "title": "Result title",
      "url": "https://example.com",
      "encrypted_content": "..."
    }
  ]
}
```

Anthropic text blocks may include citations:

```json
{
  "type": "text",
  "text": "Answer with source.",
  "citations": [
    {
      "type": "web_search_result_location",
      "url": "https://example.com",
      "title": "Result title",
      "cited_text": "quoted snippet"
    }
  ]
}
```

OpenAI request compatibility target:

```json
{
  "model": "gpt-5.5",
  "input": "Search the web",
  "tools": [{"type": "web_search"}],
  "stream": true
}
```

Also accept `web_search_preview` because current OpenAI Responses clients may emit that alias.

## File Structure

- Modify `internal/openai/types.go`
  - Extend `OutputItem` / content structs only if needed for `web_search_call` and URL annotations.
  - Keep `Tool.Raw` as the source for web search-specific settings so unknown OpenAI fields remain forward-compatible.

- Modify `internal/anthropic/types.go`
  - Extend `Tool` with `MaxUses`, `AllowedDomains`, `BlockedDomains`, and `UserLocation`.
  - Extend `ContentBlock` with `Citations` and fields needed for `server_tool_use` and `web_search_tool_result` transcript preservation.
  - Extend stream event structs if they currently cannot unmarshal `server_tool_use`, `web_search_tool_result`, or citations.

- Modify `internal/convert/convert.go`
  - Add `convertWebSearchTool`.
  - Update `convertTools` to accept `web_search` and `web_search_preview`.
  - Update Anthropic-to-OpenAI response conversion to emit `web_search_call` for `server_tool_use` and preserve `web_search_tool_result` in transcript.
  - Attach web citations to output text annotations when citation data exists.

- Modify `internal/stream/bridge.go`
  - Track `server_tool_use` blocks separately from client `tool_use`.
  - Emit OpenAI-compatible `response.output_item.added` / `response.output_item.done` events for `web_search_call`.
  - Preserve `server_tool_use` and `web_search_tool_result` blocks in the returned Anthropic transcript.
  - Include citation annotations on streamed text part completion when Anthropic provides them in content block metadata.

- Modify tests:
  - `internal/convert/convert_test.go`
  - `internal/server/handler_test.go`
  - `internal/stream/bridge_test.go`
  - `internal/anthropic/client.go` tests are not required unless header behavior changes.

- Modify docs:
  - `docs/api/anthropic-messages-api.md`
  - `docs/api/openai-responses-api.md`
  - `README.md` only if it has a feature support list.

## Compatibility Decisions

- OpenAI `web_search` and `web_search_preview` both map to Anthropic `type:"web_search_20250305", name:"web_search"`.
- If the OpenAI tool contains `max_uses`, pass it through when positive.
- If the OpenAI tool contains `filters.allowed_domains` or top-level `allowed_domains`, pass to Anthropic `allowed_domains`.
- If the OpenAI tool contains `filters.blocked_domains` or top-level `blocked_domains`, pass to Anthropic `blocked_domains`.
- If the OpenAI tool contains `user_location`, pass only the common approximate location fields: `type`, `city`, `region`, `country`, `timezone`.
- Ignore OpenAI-only knobs such as `search_context_size` unless Anthropic has an equivalent. Do not reject requests because of ignored unknown fields.
- Do not add an Anthropic beta header for web search unless a current official doc requires one. The existing `Betas` mechanism remains available if verification shows one is required.
- For `tool_choice: {"type":"web_search"}` or `{"type":"web_search_preview"}`, force Anthropic `tool_choice: {"type":"tool","name":"web_search"}`.
- Continue rejecting `file_search_preview` locally until a separate file search implementation exists.

## Task 1: Add Anthropic and OpenAI Web Search Types

**Files:**
- Modify: `internal/anthropic/types.go`
- Modify: `internal/openai/types.go`

- [ ] **Step 1: Write the failing compile-oriented test**

Add this test to `internal/convert/convert_test.go`:

```go
func TestCreateResponseToMessageMapsWebSearchToolFields(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`"Search the web"`),
		Tools: []openai.Tool{{
			Type: "web_search",
			Raw: json.RawMessage(`{
				"type":"web_search",
				"max_uses":3,
				"filters":{"allowed_domains":["example.com"],"blocked_domains":["blocked.example"]},
				"user_location":{"type":"approximate","city":"San Francisco","region":"California","country":"US","timezone":"America/Los_Angeles"},
				"search_context_size":"medium"
			}`),
		}},
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}
	if len(got.Tools) != 1 {
		t.Fatalf("expected one tool, got %+v", got.Tools)
	}
	tool := got.Tools[0]
	if tool.Type != "web_search_20250305" || tool.Name != "web_search" {
		t.Fatalf("web search tool not mapped: %+v", tool)
	}
	if tool.MaxUses != 3 {
		t.Fatalf("max_uses not mapped: %+v", tool)
	}
	if len(tool.AllowedDomains) != 1 || tool.AllowedDomains[0] != "example.com" {
		t.Fatalf("allowed_domains not mapped: %+v", tool)
	}
	if len(tool.BlockedDomains) != 1 || tool.BlockedDomains[0] != "blocked.example" {
		t.Fatalf("blocked_domains not mapped: %+v", tool)
	}
	if tool.UserLocation == nil || tool.UserLocation.City != "San Francisco" || tool.UserLocation.Country != "US" {
		t.Fatalf("user_location not mapped: %+v", tool.UserLocation)
	}
}
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./internal/convert -run TestCreateResponseToMessageMapsWebSearchToolFields
```

Expected: compile failure for missing Anthropic `Tool` fields or runtime failure with `unsupported_tool_type`.

- [ ] **Step 3: Extend Anthropic structs**

In `internal/anthropic/types.go`, update `Tool` and add `UserLocation`:

```go
type Tool struct {
	Type            string        `json:"type,omitempty"`
	Name            string        `json:"name"`
	Description     string        `json:"description,omitempty"`
	InputSchema     json.RawMessage `json:"input_schema,omitempty"`
	DisplayWidthPx  int           `json:"display_width_px,omitempty"`
	DisplayHeightPx int           `json:"display_height_px,omitempty"`
	DisplayNumber   int           `json:"display_number,omitempty"`
	MaxUses         int           `json:"max_uses,omitempty"`
	AllowedDomains  []string      `json:"allowed_domains,omitempty"`
	BlockedDomains  []string      `json:"blocked_domains,omitempty"`
	UserLocation    *UserLocation `json:"user_location,omitempty"`
}

type UserLocation struct {
	Type     string `json:"type,omitempty"`
	City     string `json:"city,omitempty"`
	Region   string `json:"region,omitempty"`
	Country  string `json:"country,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}
```

Keep alignment/gofmt exact; the snippet is the intended field set.

- [ ] **Step 4: Run the focused test again**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./internal/convert -run TestCreateResponseToMessageMapsWebSearchToolFields
```

Expected: still FAIL with `unsupported_tool_type`.

## Task 2: Map OpenAI Web Search Tool Declarations

**Files:**
- Modify: `internal/convert/convert.go`
- Test: `internal/convert/convert_test.go`

- [ ] **Step 1: Add alias and defaults tests**

Add these tests to `internal/convert/convert_test.go`:

```go
func TestCreateResponseToMessageMapsWebSearchPreviewAlias(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`"Search the web"`),
		Tools: []openai.Tool{{
			Type: "web_search_preview",
			Raw:  json.RawMessage(`{"type":"web_search_preview"}`),
		}},
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}
	if len(got.Tools) != 1 || got.Tools[0].Type != "web_search_20250305" || got.Tools[0].Name != "web_search" {
		t.Fatalf("web_search_preview alias not mapped: %+v", got.Tools)
	}
}

func TestCreateResponseToMessageMapsTopLevelWebSearchDomains(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`"Search the web"`),
		Tools: []openai.Tool{{
			Type: "web_search",
			Raw:  json.RawMessage(`{"type":"web_search","allowed_domains":["example.com"],"blocked_domains":["blocked.example"]}`),
		}},
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}
	if len(got.Tools[0].AllowedDomains) != 1 || got.Tools[0].AllowedDomains[0] != "example.com" {
		t.Fatalf("top-level allowed_domains not mapped: %+v", got.Tools[0])
	}
	if len(got.Tools[0].BlockedDomains) != 1 || got.Tools[0].BlockedDomains[0] != "blocked.example" {
		t.Fatalf("top-level blocked_domains not mapped: %+v", got.Tools[0])
	}
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./internal/convert -run 'TestCreateResponseToMessageMapsWebSearch'
```

Expected: FAIL with unsupported tool type.

- [ ] **Step 3: Implement `convertWebSearchTool`**

In `internal/convert/convert.go`, add:

```go
func convertWebSearchTool(tool openai.Tool) (anthropic.Tool, error) {
	var spec struct {
		MaxUses        int `json:"max_uses"`
		AllowedDomains []string `json:"allowed_domains"`
		BlockedDomains []string `json:"blocked_domains"`
		Filters struct {
			AllowedDomains []string `json:"allowed_domains"`
			BlockedDomains []string `json:"blocked_domains"`
		} `json:"filters"`
		UserLocation *anthropic.UserLocation `json:"user_location"`
	}
	if len(tool.Raw) != 0 {
		if err := json.Unmarshal(tool.Raw, &spec); err != nil {
			return anthropic.Tool{}, &InputError{Message: "invalid web_search tool", Code: "invalid_tool"}
		}
	}
	allowed := spec.AllowedDomains
	if len(spec.Filters.AllowedDomains) > 0 {
		allowed = spec.Filters.AllowedDomains
	}
	blocked := spec.BlockedDomains
	if len(spec.Filters.BlockedDomains) > 0 {
		blocked = spec.Filters.BlockedDomains
	}
	out := anthropic.Tool{
		Type:           "web_search_20250305",
		Name:           "web_search",
		MaxUses:        spec.MaxUses,
		AllowedDomains: allowed,
		BlockedDomains: blocked,
		UserLocation:   spec.UserLocation,
	}
	if out.UserLocation != nil && out.UserLocation.Type == "" {
		out.UserLocation.Type = "approximate"
	}
	return out, nil
}
```

Then update `convertTools` before the unsupported type check:

```go
if tool.Type == "web_search" || tool.Type == "web_search_preview" {
	webTool, err := convertWebSearchTool(tool)
	if err != nil {
		return nil, nil, err
	}
	out = append(out, webTool)
	continue
}
```

- [ ] **Step 4: Run focused convert tests**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./internal/convert -run 'TestCreateResponseToMessageMapsWebSearch'
```

Expected: PASS.

## Task 3: Map Web Search Tool Choice

**Files:**
- Modify: `internal/convert/convert.go`
- Test: `internal/convert/convert_test.go`

- [ ] **Step 1: Add failing tool choice tests**

Add:

```go
func TestCreateResponseToMessageMapsWebSearchToolChoice(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input:      openai.RawJSON(`"Search the web"`),
		ToolChoice: openai.RawJSON(`{"type":"web_search"}`),
		Tools: []openai.Tool{{
			Type: "web_search",
			Raw:  json.RawMessage(`{"type":"web_search"}`),
		}},
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}
	if got.ToolChoice == nil || got.ToolChoice.Type != "tool" || got.ToolChoice.Name != "web_search" {
		t.Fatalf("web_search tool choice not mapped: %+v", got.ToolChoice)
	}
}

func TestCreateResponseToMessageMapsWebSearchPreviewToolChoice(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input:      openai.RawJSON(`"Search the web"`),
		ToolChoice: openai.RawJSON(`{"type":"web_search_preview"}`),
		Tools: []openai.Tool{{
			Type: "web_search_preview",
			Raw:  json.RawMessage(`{"type":"web_search_preview"}`),
		}},
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}
	if got.ToolChoice == nil || got.ToolChoice.Type != "tool" || got.ToolChoice.Name != "web_search" {
		t.Fatalf("web_search_preview tool choice not mapped: %+v", got.ToolChoice)
	}
}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./internal/convert -run 'TestCreateResponseToMessageMapsWebSearch.*ToolChoice'
```

Expected: FAIL because current `convertToolChoice` returns auto.

- [ ] **Step 3: Implement tool choice mapping**

In `convertToolChoice`, after object unmarshal and before function/custom handling, add:

```go
if obj.Type == "web_search" || obj.Type == "web_search_preview" {
	return &anthropic.ToolChoice{Type: "tool", Name: "web_search", DisableParallelToolUse: disableParallel}
}
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./internal/convert -run 'TestCreateResponseToMessageMapsWebSearch.*ToolChoice'
```

Expected: PASS.

## Task 4: Convert Anthropic Web Search Response Blocks

**Files:**
- Modify: `internal/anthropic/types.go`
- Modify: `internal/openai/types.go`
- Modify: `internal/convert/convert.go`
- Test: `internal/convert/convert_test.go`

- [ ] **Step 1: Inspect current output item structs**

Run:

```bash
sed -n '80,180p' internal/openai/types.go
sed -n '80,220p' internal/convert/convert.go
```

Confirm where `OutputItem` and `MessageToResponse` switch on content block type.

- [ ] **Step 2: Add failing non-stream response conversion test**

Add to `internal/convert/convert_test.go`:

```go
func TestMessageToResponseConvertsWebSearchBlocks(t *testing.T) {
	msg := anthropic.MessageResponse{
		ID:         "msg_123",
		Model:      "claude-test",
		Role:       "assistant",
		StopReason: "end_turn",
		Content: []anthropic.ContentBlock{{
			Type:  "server_tool_use",
			ID:    "srvtoolu_123",
			Name:  "web_search",
			Input: json.RawMessage(`{"query":"OpenAI news"}`),
		}, {
			Type:      "web_search_tool_result",
			ToolUseID: "srvtoolu_123",
			Content: []any{map[string]any{
				"type":  "web_search_result",
				"title": "OpenAI News",
				"url":   "https://example.com/openai",
			}},
		}, {
			Type: "text",
			Text: "OpenAI announced news.",
			Citations: []anthropic.Citation{{
				Type:      "web_search_result_location",
				URL:       "https://example.com/openai",
				Title:     "OpenAI News",
				CitedText: "OpenAI announced news.",
			}},
		}},
	}

	got, transcript, err := convert.MessageToResponse(msg, "resp_123", 111)
	if err != nil {
		t.Fatalf("MessageToResponse returned error: %v", err)
	}
	if len(got.Output) != 2 {
		t.Fatalf("expected web_search_call and message output, got %+v", got.Output)
	}
	if got.Output[0].Type != "web_search_call" || got.Output[0].ID != "srvtoolu_123" || got.Output[0].Status != "completed" {
		t.Fatalf("server_tool_use not converted to web_search_call: %+v", got.Output[0])
	}
	if len(got.Output[1].Content) != 1 || len(got.Output[1].Content[0].Annotations) != 1 {
		t.Fatalf("citation annotations not mapped: %+v", got.Output[1])
	}
	if got.Output[1].Content[0].Annotations[0].URL != "https://example.com/openai" {
		t.Fatalf("citation URL not mapped: %+v", got.Output[1].Content[0].Annotations[0])
	}
	if len(transcript) != 1 || len(transcript[0].Content) != 3 || transcript[0].Content[1].Type != "web_search_tool_result" {
		t.Fatalf("web search blocks not preserved in transcript: %+v", transcript)
	}
}
```

- [ ] **Step 3: Run the test and verify failure**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./internal/convert -run TestMessageToResponseConvertsWebSearchBlocks
```

Expected: compile failure for missing `Citation` / `Annotations` fields or runtime failure ignoring unknown block types.

- [ ] **Step 4: Add response data types**

In `internal/anthropic/types.go`, add:

```go
type Citation struct {
	Type      string `json:"type,omitempty"`
	URL       string `json:"url,omitempty"`
	Title     string `json:"title,omitempty"`
	CitedText string `json:"cited_text,omitempty"`
}
```

Add to `ContentBlock`:

```go
Citations []Citation `json:"citations,omitempty"`
```

In `internal/openai/types.go`, extend content/annotation support. Use the existing content item type names in the file; add this shape if no annotation struct exists:

```go
type Annotation struct {
	Type  string `json:"type,omitempty"`
	URL   string `json:"url,omitempty"`
	Title string `json:"title,omitempty"`
	Text  string `json:"text,omitempty"`
}
```

Add:

```go
Annotations []Annotation `json:"annotations,omitempty"`
```

to the output text content struct.

Ensure `OutputItem` can marshal:

```go
Type   string `json:"type"`
ID     string `json:"id,omitempty"`
Status string `json:"status,omitempty"`
```

for `web_search_call`. Reuse existing fields where possible; do not create a separate OpenAI type unless current structs make that cleaner.

- [ ] **Step 5: Implement conversion helpers**

In `internal/convert/convert.go`, add:

```go
func convertCitations(citations []anthropic.Citation) []openai.Annotation {
	if len(citations) == 0 {
		return nil
	}
	out := make([]openai.Annotation, 0, len(citations))
	for _, citation := range citations {
		if citation.URL == "" && citation.Title == "" && citation.CitedText == "" {
			continue
		}
		out = append(out, openai.Annotation{
			Type:  "url_citation",
			URL:   citation.URL,
			Title: citation.Title,
			Text:  citation.CitedText,
		})
	}
	return out
}
```

Update the text block conversion to set annotations from `convertCitations(block.Citations)`.

Update the block switch:

```go
case "server_tool_use":
	if block.Name == "web_search" {
		output = append(output, openai.OutputItem{
			ID:     block.ID,
			Type:   "web_search_call",
			Status: "completed",
		})
		transcriptBlocks = append(transcriptBlocks, normalizedBlock)
		continue
	}
case "web_search_tool_result":
	transcriptBlocks = append(transcriptBlocks, normalizedBlock)
	continue
```

Use the local variable names already present in `MessageToResponse`; this snippet describes the intended behavior, not exact surrounding code.

- [ ] **Step 6: Run focused conversion test**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./internal/convert -run TestMessageToResponseConvertsWebSearchBlocks
```

Expected: PASS.

## Task 5: Forward Web Search Tools Through HTTP Handler

**Files:**
- Modify: `internal/server/handler_test.go`
- Implementation should already be in `internal/convert/convert.go` and `internal/anthropic/types.go`

- [ ] **Step 1: Add failing handler forwarding test**

Add to `internal/server/handler_test.go`:

```go
func TestResponsesHandlerForwardsWebSearchTool(t *testing.T) {
	var upstreamBody map[string]any
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if err := json.NewDecoder(r.Body).Decode(&upstreamBody); err != nil {
			t.Fatal(err)
		}
		return jsonResponse(http.StatusOK, `{
			"id":"msg_123",
			"type":"message",
			"role":"assistant",
			"model":"claude-test",
			"content":[{"type":"text","text":"done"}],
			"stop_reason":"end_turn",
			"usage":{"input_tokens":1,"output_tokens":1}
		}`), nil
	})}
	h := server.New(server.Config{
		AnthropicAPIKey:  "anthropic-key",
		AnthropicModel:   "claude-test",
		AnthropicBaseURL: "http://anthropic.test",
	}, state.NewStore(24*time.Hour), httpClient)

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"input":"Search",
		"tools":[{"type":"web_search","max_uses":2}]
	}`))
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("unexpected status %d: %s", rec.Code, rec.Body.String())
	}
	tools, ok := upstreamBody["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("upstream tools missing: %+v", upstreamBody)
	}
	tool := tools[0].(map[string]any)
	if tool["type"] != "web_search_20250305" || tool["name"] != "web_search" || tool["max_uses"].(float64) != 2 {
		t.Fatalf("web search tool not forwarded: %+v", tool)
	}
}
```

- [ ] **Step 2: Run the focused handler test**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./internal/server -run TestResponsesHandlerForwardsWebSearchTool
```

Expected: PASS after Tasks 1-2. If it fails, fix only the forwarding path.

## Task 6: Stream Anthropic Web Search Blocks as Responses Events

**Files:**
- Modify: `internal/anthropic/types.go`
- Modify: `internal/stream/bridge.go`
- Test: `internal/stream/bridge_test.go`

- [ ] **Step 1: Inspect stream event structs**

Run:

```bash
sed -n '80,180p' internal/anthropic/types.go
sed -n '1,180p' internal/stream/bridge_test.go
```

Find `StreamEvent` and existing stream test style.

- [ ] **Step 2: Add failing stream test**

Add to `internal/stream/bridge_test.go`:

```go
func TestBridgeWebSearchStream(t *testing.T) {
	input := strings.Join([]string{
		`data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null}}`,
		``,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_123","name":"web_search","input":{"query":"OpenAI news"}}}`,
		``,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_123","content":[{"type":"web_search_result","title":"OpenAI News","url":"https://example.com/openai"}]}}`,
		``,
		`data: {"type":"content_block_stop","index":1}`,
		``,
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":"","citations":[{"type":"web_search_result_location","url":"https://example.com/openai","title":"OpenAI News","cited_text":"OpenAI news"}]}}`,
		``,
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"OpenAI news"}}`,
		``,
		`data: {"type":"content_block_stop","index":2}`,
		``,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	var out bytes.Buffer
	transcript, err := stream.BridgeWithResult(strings.NewReader(input), &out, "resp_123", 111)
	if err != nil {
		t.Fatalf("BridgeWithResult returned error: %v", err)
	}
	body := out.String()
	if !strings.Contains(body, `"type":"web_search_call"`) {
		t.Fatalf("web_search_call event missing:\n%s", body)
	}
	if !strings.Contains(body, `"url":"https://example.com/openai"`) {
		t.Fatalf("citation annotation missing:\n%s", body)
	}
	if len(transcript.Content) != 3 || transcript.Content[0].Type != "server_tool_use" || transcript.Content[1].Type != "web_search_tool_result" {
		t.Fatalf("web search transcript blocks not preserved: %+v", transcript)
	}
}
```

- [ ] **Step 3: Run stream test and verify failure**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./internal/stream -run TestBridgeWebSearchStream
```

Expected: FAIL because `server_tool_use` and `web_search_tool_result` are ignored.

- [ ] **Step 4: Extend stream tracking state**

In `internal/stream/bridge.go`, add:

```go
type serverToolState struct {
	ID    string
	Name  string
	Input json.RawMessage
}

type blockState struct {
	Block anthropic.ContentBlock
}
```

Update `BridgeWithResult` to initialize:

```go
serverTools := map[int]*serverToolState{}
blockResults := map[int]*blockState{}
```

Update `handleSSEData` signature to receive those maps. Update all call sites in the file.

- [ ] **Step 5: Handle content block starts**

In `content_block_start`, add cases:

```go
if event.ContentBlock != nil && event.ContentBlock.Type == "server_tool_use" && event.ContentBlock.Name == "web_search" {
	id := event.ContentBlock.ID
	if id == "" {
		id = fmt.Sprintf("%s_web_search_%d", responseID, event.Index)
	}
	serverTools[event.Index] = &serverToolState{ID: id, Name: "web_search", Input: event.ContentBlock.Input}
	return blocks, writeEvent(w, map[string]any{
		"type":         "response.output_item.added",
		"response_id":  responseID,
		"output_index": event.Index,
		"item": map[string]any{
			"id":     id,
			"type":   "web_search_call",
			"status": "in_progress",
		},
	})
}
if event.ContentBlock != nil && event.ContentBlock.Type == "web_search_tool_result" {
	blockResults[event.Index] = &blockState{Block: *event.ContentBlock}
	return blocks, nil
}
```

For text start, preserve `event.ContentBlock.Citations` in `textState`; add `Citations []anthropic.Citation` to `textState`.

- [ ] **Step 6: Handle content block stops**

In `content_block_stop`, before normal tool/text handling:

```go
if state := serverTools[event.Index]; state != nil {
	blocks = append(blocks, anthropic.ContentBlock{Type: "server_tool_use", ID: state.ID, Name: state.Name, Input: state.Input})
	return blocks, writeEvent(w, map[string]any{
		"type":         "response.output_item.done",
		"response_id":  responseID,
		"output_index": event.Index,
		"item": map[string]any{
			"id":     state.ID,
			"type":   "web_search_call",
			"status": "completed",
		},
	})
}
if state := blockResults[event.Index]; state != nil {
	blocks = append(blocks, state.Block)
	return blocks, nil
}
```

When finalizing text content, include annotations:

```go
annotations := convertCitationsForStream(state.Citations)
```

Use `annotations` in `response.content_part.done` and `response.output_item.done`. Add a stream-local helper returning `[]any` maps:

```go
func convertCitationsForStream(citations []anthropic.Citation) []any {
	out := make([]any, 0, len(citations))
	for _, citation := range citations {
		if citation.URL == "" && citation.Title == "" && citation.CitedText == "" {
			continue
		}
		out = append(out, map[string]any{
			"type":  "url_citation",
			"url":   citation.URL,
			"title": citation.Title,
			"text":  citation.CitedText,
		})
	}
	if len(out) == 0 {
		return []any{}
	}
	return out
}
```

- [ ] **Step 7: Run focused stream test**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./internal/stream -run TestBridgeWebSearchStream
```

Expected: PASS.

## Task 7: Store Web Search Transcript Blocks Across `previous_response_id`

**Files:**
- Modify: `internal/state/store_test.go` only if store filtering drops unknown blocks.
- Modify: `internal/state/store.go` only if needed.
- Test: `internal/server/handler_test.go`

- [ ] **Step 1: Add handler continuation test**

Add to `internal/server/handler_test.go`:

```go
func TestResponsesHandlerStoresWebSearchTranscriptForContinuation(t *testing.T) {
	var upstreamBodies []map[string]any
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		upstreamBodies = append(upstreamBodies, body)
		if len(upstreamBodies) == 1 {
			return jsonResponse(http.StatusOK, `{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"model":"claude-test",
				"content":[
					{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"OpenAI news"}},
					{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","title":"OpenAI News","url":"https://example.com/openai"}]},
					{"type":"text","text":"OpenAI news"}
				],
				"stop_reason":"end_turn",
				"usage":{"input_tokens":1,"output_tokens":1}
			}`), nil
		}
		return jsonResponse(http.StatusOK, `{
			"id":"msg_2",
			"type":"message",
			"role":"assistant",
			"model":"claude-test",
			"content":[{"type":"text","text":"continued"}],
			"stop_reason":"end_turn",
			"usage":{"input_tokens":1,"output_tokens":1}
		}`), nil
	})}
	store := state.NewStore(24 * time.Hour)
	h := server.New(server.Config{
		AnthropicAPIKey:  "anthropic-key",
		AnthropicModel:   "claude-test",
		AnthropicBaseURL: "http://anthropic.test",
	}, store, httpClient)

	first := httptest.NewRecorder()
	h.ServeHTTP(first, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"input":"Search",
		"tools":[{"type":"web_search"}]
	}`)))
	if first.Code != http.StatusOK {
		t.Fatalf("first status %d: %s", first.Code, first.Body.String())
	}
	var firstResp openai.Response
	if err := json.Unmarshal(first.Body.Bytes(), &firstResp); err != nil {
		t.Fatal(err)
	}

	second := httptest.NewRecorder()
	h.ServeHTTP(second, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"previous_response_id":"`+firstResp.ID+`",
		"input":"Continue"
	}`)))
	if second.Code != http.StatusOK {
		t.Fatalf("second status %d: %s", second.Code, second.Body.String())
	}
	if len(upstreamBodies) != 2 {
		t.Fatalf("expected two upstream calls, got %d", len(upstreamBodies))
	}
	messages := upstreamBodies[1]["messages"].([]any)
	assistant := messages[1].(map[string]any)
	content := assistant["content"].([]any)
	if len(content) != 3 || content[0].(map[string]any)["type"] != "server_tool_use" || content[1].(map[string]any)["type"] != "web_search_tool_result" {
		t.Fatalf("web search transcript not restored: %+v", upstreamBodies[1])
	}
}
```

- [ ] **Step 2: Run the continuation test**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./internal/server -run TestResponsesHandlerStoresWebSearchTranscriptForContinuation
```

Expected: PASS after Task 4 if transcript preservation is correct. If FAIL, inspect `internal/state/store.go` and update clone/index logic to preserve unknown server tool blocks without indexing them as client tool calls.

## Task 8: Keep Unsupported Tools Rejected Locally

**Files:**
- Modify: `internal/convert/convert_test.go`
- Modify: `internal/server/handler_test.go`

- [ ] **Step 1: Add regression assertions**

Update existing unsupported tool tests to cover both:

```go
Tools: []openai.Tool{{Type: "file_search_preview"}}
```

and:

```go
Tools: []openai.Tool{{Type: "unknown_preview"}}
```

Expected code remains:

```json
"code":"unsupported_tool_type"
```

- [ ] **Step 2: Run focused unsupported tests**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./internal/convert ./internal/server -run 'UnsupportedToolType|RejectsUnsupportedToolType'
```

Expected: PASS.

## Task 9: Update Local API Documentation

**Files:**
- Modify: `docs/api/anthropic-messages-api.md`
- Modify: `docs/api/openai-responses-api.md`
- Modify: `README.md` if it lists supported tool types.

- [ ] **Step 1: Document OpenAI accepted request shapes**

In `docs/api/openai-responses-api.md`, add a web search subsection near tool docs:

```md
### Web search tools

The proxy accepts `tools` entries with `type:"web_search"` and `type:"web_search_preview"`.
Both are mapped to Anthropic server-side web search.

Supported pass-through fields:

- `max_uses`
- `allowed_domains`
- `blocked_domains`
- `filters.allowed_domains`
- `filters.blocked_domains`
- `user_location.type`
- `user_location.city`
- `user_location.region`
- `user_location.country`
- `user_location.timezone`

OpenAI-only fields without an Anthropic equivalent, such as `search_context_size`, are ignored.
```

- [ ] **Step 2: Document Anthropic mapping**

In `docs/api/anthropic-messages-api.md`, add:

```md
### Web search server tool

OpenAI `web_search` and `web_search_preview` map to:

```json
{"type":"web_search_20250305","name":"web_search"}
```

Anthropic response blocks `server_tool_use` and `web_search_tool_result` are preserved in stored transcript history so `previous_response_id` continuation can send valid Messages history back upstream.
```

- [ ] **Step 3: Run docs grep**

Run:

```bash
rg -n "web_search|web_search_preview|web_search_20250305" README.md docs internal
```

Expected: docs and tests mention the new mapping; `file_search_preview` remains unsupported.

## Task 10: Full Verification

**Files:**
- All changed Go files.

- [ ] **Step 1: Format Go files**

Run:

```bash
gofmt -w internal/openai/types.go internal/anthropic/types.go internal/convert/convert.go internal/convert/convert_test.go internal/server/handler_test.go internal/stream/bridge.go internal/stream/bridge_test.go internal/state/store.go internal/state/store_test.go
```

If some listed files were not modified, `gofmt` still safely formats them.

- [ ] **Step 2: Run package tests**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./internal/convert ./internal/stream ./internal/server ./internal/state ./internal/openai ./internal/anthropic
```

Expected: PASS.

- [ ] **Step 3: Run all tests**

Run:

```bash
GOCACHE="$PWD/.gocache" go test -count=1 ./...
```

Expected: PASS.

- [ ] **Step 4: Optional local proxy smoke test**

Start the proxy only if an Anthropic-compatible upstream is available:

```bash
GOCACHE="$PWD/.gocache" ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" ANTHROPIC_MODEL="$ANTHROPIC_MODEL" ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL" PROXY_ADDR="127.0.0.1:8180" go run ./cmd/rap
```

Then POST:

```bash
curl -sS http://127.0.0.1:8180/v1/responses \
  -H 'content-type: application/json' \
  -d '{"model":"gpt-5.5","input":"Search for current OpenAI news","tools":[{"type":"web_search"}],"stream":false}'
```

Expected: no local `unsupported_tool_type` error. If upstream rejects due to account/model/tool availability, the error should be an upstream error, not a local compatibility error.

## Task 11: Commit

**Files:**
- All modified implementation, test, and docs files.

- [ ] **Step 1: Inspect status**

Run:

```bash
git status --short
```

Expected: only intended files are modified. Preserve unrelated pre-existing user changes.

- [ ] **Step 2: Commit**

Run:

```bash
git add internal/openai/types.go internal/anthropic/types.go internal/convert/convert.go internal/convert/convert_test.go internal/server/handler_test.go internal/stream/bridge.go internal/stream/bridge_test.go docs/api/anthropic-messages-api.md docs/api/openai-responses-api.md README.md
git commit -m "✨ feat: support Responses web search tools"
```

If `README.md` was not changed, omit it from `git add`.

- [ ] **Step 3: Inspect commit**

Run:

```bash
git log -1 --pretty=full
```

Expected: subject renders as `✨ feat: support Responses web search tools`.

## Self-Review

- Spec coverage: The plan covers request tool mapping, tool choice, non-stream response conversion, streaming conversion, transcript continuation, unsupported tool regression, docs, formatting, tests, and commit.
- Placeholder scan: No task says TBD/TODO/implement later. Code-bearing steps include concrete snippets or exact expected behavior tied to existing local functions.
- Type consistency: `anthropic.UserLocation`, `anthropic.Citation`, `openai.Annotation`, `web_search_call`, `server_tool_use`, and `web_search_tool_result` are introduced before later tasks depend on them.
