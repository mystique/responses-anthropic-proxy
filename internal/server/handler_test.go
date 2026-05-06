package server_test

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"uni-api/internal/server"
	"uni-api/internal/state"
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

func TestUnimplementedEndpointReturnsOpenAIStyle501(t *testing.T) {
	h := server.New(server.Config{AnthropicAPIKey: "k", AnthropicModel: "m", AnthropicBaseURL: "http://example.test"}, state.NewStore(time.Hour), http.DefaultClient)
	rec := httptest.NewRecorder()

	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/responses/resp_1", nil))

	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("unexpected status %d: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"type":"not_implemented"`) {
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
