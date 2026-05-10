package openai_test

import (
	"encoding/json"
	"testing"

	"rap/internal/openai"
)

func TestCreateResponseRequestPreservesCodexCompatibilityFields(t *testing.T) {
	var req openai.CreateResponseRequest
	if err := json.Unmarshal([]byte(`{
		"model":"gpt-test",
		"input":"hi",
		"reasoning":{"effort":"high"},
		"truncation":"auto",
		"include":["reasoning.encrypted_content"],
		"metadata":{"session":"abc"},
		"store":false,
		"user":"user_123",
		"stop":["END"],
		"text":{"format":{"type":"json_object"}}
	}`), &req); err != nil {
		t.Fatal(err)
	}

	if string(req.Reasoning) != `{"effort":"high"}` {
		t.Fatalf("reasoning not preserved: %s", req.Reasoning)
	}
	if req.Truncation != "auto" {
		t.Fatalf("truncation not decoded: %q", req.Truncation)
	}
	if len(req.Include) != 1 || req.Include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include not decoded: %+v", req.Include)
	}
	if req.Metadata == nil || string(req.Metadata["session"]) != `"abc"` {
		t.Fatalf("metadata not preserved: %+v", req.Metadata)
	}
	if req.Store == nil || *req.Store {
		t.Fatalf("store not decoded: %+v", req.Store)
	}
	if req.User != "user_123" {
		t.Fatalf("user not decoded: %q", req.User)
	}
	if string(req.Stop) != `["END"]` {
		t.Fatalf("stop not preserved: %s", req.Stop)
	}
	if string(req.Text) != `{"format":{"type":"json_object"}}` {
		t.Fatalf("text not preserved: %s", req.Text)
	}
}
