package server_test

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"responses-anthropic-proxy/internal/anthropic"
	"responses-anthropic-proxy/internal/openai"
	"responses-anthropic-proxy/internal/server"
	"responses-anthropic-proxy/internal/state"
)

func TestResponsesHandlerProxiesNonStreamingRequest(t *testing.T) {
	var upstreamBody map[string]any
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/v1/messages" {
			t.Fatalf("unexpected upstream path: %s", r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "anthropic-key" || r.Header.Get("anthropic-version") != "2023-06-01" {
			t.Fatalf("missing Anthropic headers: %+v", r.Header)
		}
		if err := json.NewDecoder(r.Body).Decode(&upstreamBody); err != nil {
			t.Fatal(err)
		}
		return jsonResponse(http.StatusOK, `{
			"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
			"content":[{"type":"text","text":"hello"}],
			"stop_reason":"end_turn",
			"usage":{"input_tokens":3,"output_tokens":2}
		}`), nil
	})}

	h := server.New(server.Config{
		AnthropicAPIKey:  "anthropic-key",
		AnthropicModel:   "claude-test",
		AnthropicBaseURL: "http://anthropic.test",
	}, state.NewStore(24*time.Hour), httpClient)

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"gpt","instructions":"sys","input":"hi"}`))
	req.Header.Set("authorization", "Bearer anything")
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("unexpected status %d: %s", rec.Code, rec.Body.String())
	}
	if upstreamBody["model"] != "claude-test" || upstreamBody["system"] != "sys" {
		t.Fatalf("unexpected upstream body: %+v", upstreamBody)
	}
	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp["object"] != "response" || resp["status"] != "completed" || resp["output_text"] != "hello" {
		t.Fatalf("unexpected proxy response: %+v", resp)
	}
	if _, ok := resp["id"].(string); !ok {
		t.Fatalf("missing response id: %+v", resp)
	}
}

func TestResponsesHandlerUsesPreviousResponseIDTranscript(t *testing.T) {
	var requestBodies []map[string]any
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		requestBodies = append(requestBodies, body)
		return jsonResponse(http.StatusOK, `{
			"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
			"content":[{"type":"text","text":"answer"}],
			"stop_reason":"end_turn"
		}`), nil
	})}

	store := state.NewStore(24 * time.Hour)
	h := server.New(server.Config{
		AnthropicAPIKey:  "anthropic-key",
		AnthropicModel:   "claude-test",
		AnthropicBaseURL: "http://anthropic.test",
	}, store, httpClient)

	first := httptest.NewRecorder()
	h.ServeHTTP(first, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"input":"first"}`)))
	var firstResp struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(first.Body.Bytes(), &firstResp); err != nil {
		t.Fatal(err)
	}

	second := httptest.NewRecorder()
	h.ServeHTTP(second, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"previous_response_id":"`+firstResp.ID+`","input":"second"}`)))

	if second.Code != http.StatusOK {
		t.Fatalf("unexpected second status %d: %s", second.Code, second.Body.String())
	}
	messages, ok := requestBodies[1]["messages"].([]any)
	if !ok {
		t.Fatalf("messages missing in upstream body: %+v", requestBodies[1])
	}
	if len(messages) != 3 {
		t.Fatalf("expected previous user+assistant plus new user messages, got %+v", messages)
	}
}

func TestResponsesHandlerSendsToolResultWithResolvedToolUseID(t *testing.T) {
	var requestBodies []map[string]any
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		requestBodies = append(requestBodies, body)
		if len(requestBodies) == 1 {
			return jsonResponse(http.StatusOK, `{
				"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
				"content":[{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"abc"}}],
				"stop_reason":"tool_use"
			}`), nil
		}
		return jsonResponse(http.StatusOK, `{
			"id":"msg_2","type":"message","role":"assistant","model":"claude-test",
			"content":[{"type":"text","text":"done"}],
			"stop_reason":"end_turn"
		}`), nil
	})}
	store := state.NewStore(24 * time.Hour)
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "claude-test", AnthropicBaseURL: "http://anthropic.test"}, store, httpClient)

	first := httptest.NewRecorder()
	h.ServeHTTP(first, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"input":"first"}`)))
	var firstResp openai.Response
	if err := json.Unmarshal(first.Body.Bytes(), &firstResp); err != nil {
		t.Fatal(err)
	}

	second := httptest.NewRecorder()
	h.ServeHTTP(second, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"previous_response_id":"`+firstResp.ID+`",
		"input":[{"type":"function_call_output","call_id":"`+firstResp.Output[0].CallID+`","output":"{\"ok\":true}"}]
	}`)))

	if second.Code != http.StatusOK {
		t.Fatalf("unexpected second status %d: %s", second.Code, second.Body.String())
	}
	messages := requestBodies[1]["messages"].([]any)
	assistant := messages[len(messages)-2].(map[string]any)
	toolResultUser := messages[len(messages)-1].(map[string]any)
	if assistant["role"] != "assistant" || toolResultUser["role"] != "user" {
		t.Fatalf("unexpected message order: %+v", messages)
	}
	content := toolResultUser["content"].([]any)
	block := content[0].(map[string]any)
	if block["type"] != "tool_result" || block["tool_use_id"] != "toolu_1" {
		t.Fatalf("tool result did not use Anthropic tool_use id: %+v", block)
	}
}

func TestResponsesHandlerRestoresHistoryForUniqueCallIDWithoutPreviousResponseID(t *testing.T) {
	var requestBodies []map[string]any
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		requestBodies = append(requestBodies, body)
		if len(requestBodies) == 1 {
			return jsonResponse(http.StatusOK, `{
				"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
				"content":[{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}],
				"stop_reason":"tool_use"
			}`), nil
		}
		return jsonResponse(http.StatusOK, `{"id":"msg_2","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn"}`), nil
	})}
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "claude-test", AnthropicBaseURL: "http://anthropic.test"}, state.NewStore(24*time.Hour), httpClient)

	first := httptest.NewRecorder()
	h.ServeHTTP(first, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"input":"first"}`)))
	var firstResp openai.Response
	if err := json.Unmarshal(first.Body.Bytes(), &firstResp); err != nil {
		t.Fatal(err)
	}

	second := httptest.NewRecorder()
	h.ServeHTTP(second, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"input":[{"type":"function_call_output","call_id":"`+firstResp.Output[0].CallID+`","output":"ok"}]
	}`)))

	if second.Code != http.StatusOK {
		t.Fatalf("unexpected second status %d: %s", second.Code, second.Body.String())
	}
	if messages := requestBodies[1]["messages"].([]any); len(messages) != 3 {
		t.Fatalf("expected restored user+assistant plus tool result, got %+v", messages)
	}
}

func TestResponsesHandlerAcceptsInlineFunctionCallOutputWithoutPreviousResponseID(t *testing.T) {
	var upstreamBody map[string]any
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if err := json.NewDecoder(r.Body).Decode(&upstreamBody); err != nil {
			t.Fatal(err)
		}
		return jsonResponse(http.StatusOK, `{"id":"msg_2","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn"}`), nil
	})}
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "claude-test", AnthropicBaseURL: "http://anthropic.test"}, state.NewStore(24*time.Hour), httpClient)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"input":[
			{"type":"function_call","call_id":"call_123","name":"exec_command","arguments":"{\"cmd\":\"pwd\"}"},
			{"type":"function_call_output","call_id":"call_123","output":"ok"}
		]
	}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("unexpected status %d: %s", rec.Code, rec.Body.String())
	}
	messages := upstreamBody["messages"].([]any)
	if len(messages) != 2 {
		t.Fatalf("expected inline assistant tool_use plus user tool_result, got %+v", messages)
	}
	assistant := messages[0].(map[string]any)
	user := messages[1].(map[string]any)
	if assistant["role"] != "assistant" || user["role"] != "user" {
		t.Fatalf("unexpected message roles: %+v", messages)
	}
	toolUse := assistant["content"].([]any)[0].(map[string]any)
	toolResult := user["content"].([]any)[0].(map[string]any)
	if toolUse["type"] != "tool_use" || toolUse["id"] != "call_123" || toolResult["tool_use_id"] != "call_123" {
		t.Fatalf("unexpected tool loop conversion: %+v %+v", toolUse, toolResult)
	}
}

func TestResponsesHandlerKeepsMultipleInlineFunctionCallsAdjacentToResults(t *testing.T) {
	var upstreamBody map[string]any
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if err := json.NewDecoder(r.Body).Decode(&upstreamBody); err != nil {
			t.Fatal(err)
		}
		return jsonResponse(http.StatusOK, `{"id":"msg_2","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn"}`), nil
	})}
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "claude-test", AnthropicBaseURL: "http://anthropic.test"}, state.NewStore(24*time.Hour), httpClient)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"input":[
			{"type":"function_call","call_id":"call_1","name":"exec_command","arguments":"{\"cmd\":\"pwd\"}"},
			{"type":"function_call","call_id":"call_2","name":"exec_command","arguments":"{\"cmd\":\"ls\"}"},
			{"type":"function_call_output","call_id":"call_1","output":"pwd ok"},
			{"type":"function_call_output","call_id":"call_2","output":"ls ok"}
		]
	}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("unexpected status %d: %s", rec.Code, rec.Body.String())
	}
	messages := upstreamBody["messages"].([]any)
	if len(messages) != 2 {
		t.Fatalf("expected one assistant tool_use message plus one user tool_result message, got %+v", messages)
	}
	assistant := messages[0].(map[string]any)
	user := messages[1].(map[string]any)
	toolUses := assistant["content"].([]any)
	toolResults := user["content"].([]any)
	if assistant["role"] != "assistant" || len(toolUses) != 2 {
		t.Fatalf("expected two tool uses in one assistant message, got %+v", assistant)
	}
	if user["role"] != "user" || len(toolResults) != 2 {
		t.Fatalf("expected two tool results in one user message, got %+v", user)
	}
	firstUse := toolUses[0].(map[string]any)
	secondUse := toolUses[1].(map[string]any)
	firstResult := toolResults[0].(map[string]any)
	secondResult := toolResults[1].(map[string]any)
	if firstUse["id"] != "call_1" || secondUse["id"] != "call_2" || firstResult["tool_use_id"] != "call_1" || secondResult["tool_use_id"] != "call_2" {
		t.Fatalf("unexpected tool call/result pairing: %+v %+v", assistant, user)
	}
}

func TestResponsesHandlerReturnsLocalErrorForMissingToolCall(t *testing.T) {
	var logs bytes.Buffer
	origWriter := log.Writer()
	origFlags := log.Flags()
	log.SetOutput(&logs)
	log.SetFlags(0)
	defer func() {
		log.SetOutput(origWriter)
		log.SetFlags(origFlags)
	}()
	called := false
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		called = true
		return jsonResponse(http.StatusOK, `{}`), nil
	})}
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "claude-test", AnthropicBaseURL: "http://anthropic.test"}, state.NewStore(24*time.Hour), httpClient)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"input":[{"type":"function_call_output","call_id":"missing","output":"ok"}]
	}`)))

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status %d: %s", rec.Code, rec.Body.String())
	}
	if called {
		t.Fatal("unexpected upstream call")
	}
	if !strings.Contains(rec.Body.String(), `"code":"tool_call_not_found"`) {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
	gotLogs := logs.String()
	for _, want := range []string{
		"local_compatibility_error_context",
		`"code":"tool_call_not_found"`,
		`"openai_request"`,
		`"input":[{"type":"function_call_output","call_id":"missing","output":"ok"}]`,
		`"local_response"`,
		`"store_snapshot"`,
	} {
		if !strings.Contains(gotLogs, want) {
			t.Fatalf("expected log to contain %s, got:\n%s", want, gotLogs)
		}
	}
}

func TestResponsesHandlerReturnsAmbiguousToolCallWithoutPreviousResponseID(t *testing.T) {
	store := state.NewStore(24 * time.Hour)
	for _, id := range []string{"resp_1", "resp_2"} {
		store.Save(state.ResponseRecord{
			ID:       id,
			Response: openai.Response{ID: id, Output: []openai.OutputItem{{Type: "function_call", CallID: "call_same"}}},
			Transcript: []anthropic.MessageParam{{
				Role:    "assistant",
				Content: []anthropic.ContentBlock{{Type: "tool_use", ID: "toolu_" + id}},
			}},
			CreatedAt: 123,
		})
	}
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "claude-test", AnthropicBaseURL: "http://anthropic.test"}, store, http.DefaultClient)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"input":[{"type":"function_call_output","call_id":"call_same","output":"ok"}]
	}`)))

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"code":"ambiguous_tool_call_id"`) {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
}

func TestResponsesHandlerRejectsDuplicateToolCallOutput(t *testing.T) {
	var calls int
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		calls++
		if calls == 1 {
			return jsonResponse(http.StatusOK, `{
				"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
				"content":[{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}],
				"stop_reason":"tool_use"
			}`), nil
		}
		return jsonResponse(http.StatusOK, `{"id":"msg_2","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn"}`), nil
	})}
	store := state.NewStore(24 * time.Hour)
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "claude-test", AnthropicBaseURL: "http://anthropic.test"}, store, httpClient)

	first := httptest.NewRecorder()
	h.ServeHTTP(first, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"input":"first"}`)))
	var firstResp openai.Response
	if err := json.Unmarshal(first.Body.Bytes(), &firstResp); err != nil {
		t.Fatal(err)
	}
	body := `{"previous_response_id":"` + firstResp.ID + `","input":[{"type":"function_call_output","call_id":"` + firstResp.Output[0].CallID + `","output":"ok"}]}`
	okRec := httptest.NewRecorder()
	h.ServeHTTP(okRec, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))
	if okRec.Code != http.StatusOK {
		t.Fatalf("unexpected status %d: %s", okRec.Code, okRec.Body.String())
	}

	dup := httptest.NewRecorder()
	h.ServeHTTP(dup, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))
	if dup.Code != http.StatusBadRequest {
		t.Fatalf("unexpected duplicate status %d: %s", dup.Code, dup.Body.String())
	}
	if !strings.Contains(dup.Body.String(), `"code":"tool_call_already_resolved"`) {
		t.Fatalf("unexpected duplicate body: %s", dup.Body.String())
	}
}

func TestResponsesHandlerMapsAnthropicInvalidRequestTo400(t *testing.T) {
	var logs bytes.Buffer
	origWriter := log.Writer()
	origFlags := log.Flags()
	log.SetOutput(&logs)
	log.SetFlags(0)
	defer func() {
		log.SetOutput(origWriter)
		log.SetFlags(origFlags)
	}()
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return jsonResponse(http.StatusBadRequest, `{"type":"error","error":{"type":"invalid_request_error","message":"bad tool result"}}`), nil
	})}
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "claude-test", AnthropicBaseURL: "http://anthropic.test"}, state.NewStore(24*time.Hour), httpClient)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"input":"hi"}`)))

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("unexpected status %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"message":"bad tool result"`) || !strings.Contains(rec.Body.String(), `"type":"invalid_request_error"`) {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
	gotLogs := logs.String()
	for _, want := range []string{
		"upstream_error_context",
		`"upstream_status":400`,
		`"upstream_type":"invalid_request_error"`,
		`"upstream_body":"{\"type\":\"error\"`,
		`"anthropic_request"`,
		`"messages":[{"role":"user","content":[{"type":"text","text":"hi"`,
		`"openai_request"`,
		`"input":"hi"`,
	} {
		if !strings.Contains(gotLogs, want) {
			t.Fatalf("expected log to contain %s, got:\n%s", want, gotLogs)
		}
	}
}

func TestResponsesHandlerRetrievesStoredResponse(t *testing.T) {
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return jsonResponse(http.StatusOK, `{
			"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
			"content":[{"type":"text","text":"hello"}],
			"stop_reason":"end_turn"
		}`), nil
	})}
	h := server.New(server.Config{
		AnthropicAPIKey:  "anthropic-key",
		AnthropicModel:   "claude-test",
		AnthropicBaseURL: "http://anthropic.test",
	}, state.NewStore(24*time.Hour), httpClient)

	create := httptest.NewRecorder()
	h.ServeHTTP(create, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"input":"hi"}`)))
	var created struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(create.Body.Bytes(), &created); err != nil {
		t.Fatal(err)
	}

	retrieve := httptest.NewRecorder()
	h.ServeHTTP(retrieve, httptest.NewRequest(http.MethodGet, "/v1/responses/"+created.ID, nil))

	if retrieve.Code != http.StatusOK {
		t.Fatalf("unexpected status %d: %s", retrieve.Code, retrieve.Body.String())
	}
	var got map[string]any
	if err := json.Unmarshal(retrieve.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got["id"] != created.ID || got["output_text"] != "hello" || got["status"] != "completed" {
		t.Fatalf("unexpected retrieved response: %+v", got)
	}
}

func TestResponsesHandlerDeletesStoredResponse(t *testing.T) {
	store := state.NewStore(24 * time.Hour)
	resp := openai.NewBaseResponse("resp_1", "claude-test", "completed", 123)
	store.Save(state.ResponseRecord{ID: "resp_1", Response: resp, Status: "completed", CreatedAt: 123})
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "m", AnthropicBaseURL: "http://example.test"}, store, http.DefaultClient)

	del := httptest.NewRecorder()
	h.ServeHTTP(del, httptest.NewRequest(http.MethodDelete, "/v1/responses/resp_1", nil))

	if del.Code != http.StatusOK {
		t.Fatalf("unexpected status %d: %s", del.Code, del.Body.String())
	}
	if !strings.Contains(del.Body.String(), `"id":"resp_1"`) || !strings.Contains(del.Body.String(), `"deleted":true`) {
		t.Fatalf("unexpected delete body: %s", del.Body.String())
	}

	get := httptest.NewRecorder()
	h.ServeHTTP(get, httptest.NewRequest(http.MethodGet, "/v1/responses/resp_1", nil))
	if get.Code != http.StatusNotFound {
		t.Fatalf("expected deleted response to be missing, got %d: %s", get.Code, get.Body.String())
	}
}

func TestResponsesHandlerDeleteMissingResponseReturns404(t *testing.T) {
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "m", AnthropicBaseURL: "http://example.test"}, state.NewStore(time.Hour), http.DefaultClient)
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, httptest.NewRequest(http.MethodDelete, "/v1/responses/resp_missing", nil))

	if rec.Code != http.StatusNotFound {
		t.Fatalf("unexpected status %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"code":"response_not_found"`) {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
}

func TestResponsesHandlerRetrievesStreamedResponse(t *testing.T) {
	streamBody := strings.Join([]string{
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"content-type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader(streamBody)),
		}, nil
	})}
	h := server.New(server.Config{
		AnthropicAPIKey:  "anthropic-key",
		AnthropicModel:   "claude-test",
		AnthropicBaseURL: "http://anthropic.test",
	}, state.NewStore(24*time.Hour), httpClient)

	create := httptest.NewRecorder()
	h.ServeHTTP(create, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"input":"hi","stream":true}`)))
	var responseID string
	for _, line := range strings.Split(create.Body.String(), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var payload struct {
			Response struct {
				ID string `json:"id"`
			} `json:"response"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &payload); err == nil && payload.Response.ID != "" {
			responseID = payload.Response.ID
			break
		}
	}
	if responseID == "" {
		t.Fatalf("missing response id in stream:\n%s", create.Body.String())
	}

	retrieve := httptest.NewRecorder()
	h.ServeHTTP(retrieve, httptest.NewRequest(http.MethodGet, "/v1/responses/"+responseID, nil))

	if retrieve.Code != http.StatusOK {
		t.Fatalf("unexpected status %d: %s", retrieve.Code, retrieve.Body.String())
	}
	if !strings.Contains(retrieve.Body.String(), `"output_text":"Hi"`) {
		t.Fatalf("unexpected retrieved response: %s", retrieve.Body.String())
	}
}

func TestResponsesHandlerStreamsToolResultFollowupAndMarksResolved(t *testing.T) {
	var calls int
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		calls++
		if calls == 1 {
			return jsonResponse(http.StatusOK, `{
				"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
				"content":[{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}],
				"stop_reason":"tool_use"
			}`), nil
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"content-type": []string{"text/event-stream"}},
			Body: io.NopCloser(strings.NewReader(strings.Join([]string{
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
				``,
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"done"}}`,
				``,
				`data: {"type":"content_block_stop","index":0}`,
				``,
				`data: {"type":"message_stop"}`,
				``,
			}, "\n"))),
		}, nil
	})}
	store := state.NewStore(24 * time.Hour)
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "claude-test", AnthropicBaseURL: "http://anthropic.test"}, store, httpClient)

	first := httptest.NewRecorder()
	h.ServeHTTP(first, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"input":"first"}`)))
	var firstResp openai.Response
	if err := json.Unmarshal(first.Body.Bytes(), &firstResp); err != nil {
		t.Fatal(err)
	}
	body := `{"previous_response_id":"` + firstResp.ID + `","stream":true,"input":[{"type":"function_call_output","call_id":"` + firstResp.Output[0].CallID + `","output":"ok"}]}`
	streamRec := httptest.NewRecorder()
	h.ServeHTTP(streamRec, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))
	if streamRec.Code != http.StatusOK {
		t.Fatalf("unexpected stream status %d: %s", streamRec.Code, streamRec.Body.String())
	}

	dup := httptest.NewRecorder()
	h.ServeHTTP(dup, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))
	if dup.Code != http.StatusBadRequest {
		t.Fatalf("unexpected duplicate status %d: %s", dup.Code, dup.Body.String())
	}
	if !strings.Contains(dup.Body.String(), `"code":"tool_call_already_resolved"`) {
		t.Fatalf("unexpected duplicate body: %s", dup.Body.String())
	}
}

func TestResponsesHandlerCancelsInProgressResponse(t *testing.T) {
	store := state.NewStore(24 * time.Hour)
	resp := openai.NewBaseResponse("resp_1", "claude-test", "in_progress", 123)
	store.Save(state.ResponseRecord{ID: "resp_1", Response: resp, Status: "in_progress", CreatedAt: 123})
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "m", AnthropicBaseURL: "http://example.test"}, store, http.DefaultClient)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses/resp_1/cancel", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("unexpected status %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"status":"cancelled"`) {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
}

func TestResponsesHandlerCancelCompletedResponseReturns409(t *testing.T) {
	store := state.NewStore(24 * time.Hour)
	resp := openai.NewBaseResponse("resp_1", "claude-test", "completed", 123)
	store.Save(state.ResponseRecord{ID: "resp_1", Response: resp, Status: "completed", CreatedAt: 123})
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "m", AnthropicBaseURL: "http://example.test"}, store, http.DefaultClient)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses/resp_1/cancel", nil))

	if rec.Code != http.StatusConflict {
		t.Fatalf("unexpected status %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"code":"response_not_cancellable"`) {
		t.Fatalf("unexpected body: %s", rec.Body.String())
	}
}

func TestUnknownEndpointReturnsOpenAIStyle404(t *testing.T) {
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "m", AnthropicBaseURL: "http://example.test"}, state.NewStore(time.Hour), http.DefaultClient)
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/unknown", nil))

	if rec.Code != http.StatusNotFound {
		t.Fatalf("unexpected status %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"code":"not_found"`) {
		t.Fatalf("unexpected error body: %s", rec.Body.String())
	}
}

func TestResponsesCollectionWrongMethodReturns405(t *testing.T) {
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "m", AnthropicBaseURL: "http://example.test"}, state.NewStore(time.Hour), http.DefaultClient)
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/responses", nil))

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("unexpected status %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"code":"method_not_allowed"`) {
		t.Fatalf("unexpected error body: %s", rec.Body.String())
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func jsonResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"content-type": []string{"application/json"}},
		Body:       io.NopCloser(bytes.NewBufferString(body)),
	}
}
