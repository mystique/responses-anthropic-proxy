package state_test

import (
	"testing"
	"time"

	"rap/internal/anthropic"
	"rap/internal/openai"
	"rap/internal/state"
)

func TestStoreSavesAndRetrievesTranscript(t *testing.T) {
	store := state.NewStore(24 * time.Hour)
	transcript := []anthropic.MessageParam{{
		Role:    "user",
		Content: []anthropic.ContentBlock{{Type: "text", Text: "hello"}},
	}}

	store.Save(state.ResponseRecord{
		ID:         "resp_1",
		Transcript: transcript,
		Status:     "completed",
		CreatedAt:  123,
	})

	got, ok := store.Get("resp_1")
	if !ok {
		t.Fatal("expected record to exist")
	}
	if got.Transcript[0].Content[0].Text != "hello" {
		t.Fatalf("unexpected transcript: %+v", got.Transcript)
	}

	got.Transcript[0].Content[0].Text = "mutated"
	again, _ := store.Get("resp_1")
	if again.Transcript[0].Content[0].Text != "hello" {
		t.Fatalf("store returned mutable transcript reference: %+v", again.Transcript)
	}
}

func TestStoreExpiresOldRecords(t *testing.T) {
	store := state.NewStore(time.Nanosecond)
	store.Save(state.ResponseRecord{ID: "resp_old", CreatedAt: time.Now().Unix() - 10})
	store.Cleanup(time.Now())
	if _, ok := store.Get("resp_old"); ok {
		t.Fatal("expected expired record to be removed")
	}
}

func TestStoreSavesAndRetrievesResponse(t *testing.T) {
	store := state.NewStore(24 * time.Hour)
	resp := openai.NewBaseResponse("resp_1", "claude-test", "completed", 123)
	resp.OutputText = "hello"

	store.Save(state.ResponseRecord{
		ID:        "resp_1",
		Response:  resp,
		Status:    "completed",
		CreatedAt: 123,
	})

	got, ok := store.Get("resp_1")
	if !ok {
		t.Fatal("expected record to exist")
	}
	if got.Response.ID != "resp_1" || got.Response.OutputText != "hello" {
		t.Fatalf("unexpected response: %+v", got.Response)
	}

	got.Response.OutputText = "mutated"
	again, _ := store.Get("resp_1")
	if again.Response.OutputText != "hello" {
		t.Fatalf("store returned mutable response reference: %+v", again.Response)
	}
}

func TestStoreDeletesRecord(t *testing.T) {
	store := state.NewStore(24 * time.Hour)
	store.Save(state.ResponseRecord{ID: "resp_1", Response: openai.NewBaseResponse("resp_1", "m", "completed", 123), CreatedAt: 123})

	if !store.Delete("resp_1") {
		t.Fatal("expected delete to return true")
	}
	if _, ok := store.Get("resp_1"); ok {
		t.Fatal("expected record to be removed")
	}
	if store.Delete("resp_1") {
		t.Fatal("expected second delete to return false")
	}
}

func TestStoreUpdatesResponseStatus(t *testing.T) {
	store := state.NewStore(24 * time.Hour)
	store.Save(state.ResponseRecord{ID: "resp_1", Response: openai.NewBaseResponse("resp_1", "m", "in_progress", 123), Status: "in_progress", CreatedAt: 123})

	got, ok := store.UpdateStatus("resp_1", "cancelled")
	if !ok {
		t.Fatal("expected record to exist")
	}
	if got.Status != "cancelled" || got.Response.Status != "cancelled" {
		t.Fatalf("status not updated: %+v", got)
	}
}

func TestStoreIndexesToolCallsByResponseAndCallID(t *testing.T) {
	store := state.NewStore(24 * time.Hour)
	store.Save(state.ResponseRecord{
		ID: "resp_1",
		Response: openai.Response{ID: "resp_1", Output: []openai.OutputItem{{
			Type:      "function_call",
			CallID:    "call_1",
			Name:      "lookup",
			Arguments: `{"q":"abc"}`,
		}}},
		Transcript: []anthropic.MessageParam{{
			Role: "assistant",
			Content: []anthropic.ContentBlock{{
				Type:  "tool_use",
				ID:    "toolu_1",
				Name:  "lookup",
				Input: []byte(`{"q":"abc"}`),
			}},
		}},
		CreatedAt: 123,
	})

	got, ok := store.FindToolCall("resp_1", "call_1")
	if !ok {
		t.Fatal("expected tool call to be indexed")
	}
	if got.OpenAICallID != "call_1" || got.AnthropicToolUseID != "toolu_1" || got.ResponseID != "resp_1" {
		t.Fatalf("unexpected tool call record: %+v", got)
	}
}

func TestStoreFindUniqueToolCallWithoutResponseID(t *testing.T) {
	store := state.NewStore(24 * time.Hour)
	store.Save(state.ResponseRecord{
		ID: "resp_1",
		Response: openai.Response{ID: "resp_1", Output: []openai.OutputItem{{
			Type:   "function_call",
			CallID: "call_1",
		}}},
		Transcript: []anthropic.MessageParam{{Role: "assistant", Content: []anthropic.ContentBlock{{Type: "tool_use", ID: "toolu_1"}}}},
		CreatedAt:  123,
	})

	got, matches := store.FindToolCallByCallID("call_1")
	if matches != 1 {
		t.Fatalf("expected one match, got %d", matches)
	}
	if got.ResponseID != "resp_1" || got.AnthropicToolUseID != "toolu_1" {
		t.Fatalf("unexpected match: %+v", got)
	}
}

func TestStoreFindToolCallByCallIDReportsAmbiguous(t *testing.T) {
	store := state.NewStore(24 * time.Hour)
	for _, id := range []string{"resp_1", "resp_2"} {
		store.Save(state.ResponseRecord{
			ID: id,
			Response: openai.Response{ID: id, Output: []openai.OutputItem{{
				Type:   "function_call",
				CallID: "call_1",
			}}},
			Transcript: []anthropic.MessageParam{{Role: "assistant", Content: []anthropic.ContentBlock{{Type: "tool_use", ID: "toolu_" + id}}}},
			CreatedAt:  123,
		})
	}

	_, matches := store.FindToolCallByCallID("call_1")
	if matches != 2 {
		t.Fatalf("expected two matches, got %d", matches)
	}
}

func TestStoreMarkToolCallResolvedPreventsDuplicate(t *testing.T) {
	store := state.NewStore(24 * time.Hour)
	store.Save(state.ResponseRecord{
		ID:       "resp_1",
		Response: openai.Response{ID: "resp_1", Output: []openai.OutputItem{{Type: "function_call", CallID: "call_1"}}},
		Transcript: []anthropic.MessageParam{{
			Role:    "assistant",
			Content: []anthropic.ContentBlock{{Type: "tool_use", ID: "toolu_1"}},
		}},
		CreatedAt: 123,
	})

	if ok := store.MarkToolCallResolved("resp_1", "call_1", 456); !ok {
		t.Fatal("expected first resolve to succeed")
	}
	record, ok := store.FindToolCall("resp_1", "call_1")
	if !ok || record.ResolvedAt == 0 {
		t.Fatalf("expected resolved record, got %+v", record)
	}
	if ok := store.MarkToolCallResolved("resp_1", "call_1", 789); ok {
		t.Fatal("expected duplicate resolve to fail")
	}
}

func TestStoreDeletesAndExpiresToolCallIndex(t *testing.T) {
	store := state.NewStore(time.Nanosecond)
	store.Save(state.ResponseRecord{
		ID:       "resp_delete",
		Response: openai.Response{ID: "resp_delete", Output: []openai.OutputItem{{Type: "function_call", CallID: "call_delete"}}},
		Transcript: []anthropic.MessageParam{{
			Role:    "assistant",
			Content: []anthropic.ContentBlock{{Type: "tool_use", ID: "toolu_delete"}},
		}},
		CreatedAt: time.Now().Unix(),
	})
	store.Delete("resp_delete")
	if _, ok := store.FindToolCall("resp_delete", "call_delete"); ok {
		t.Fatal("expected delete to remove tool call index")
	}

	store.Save(state.ResponseRecord{
		ID:       "resp_old",
		Response: openai.Response{ID: "resp_old", Output: []openai.OutputItem{{Type: "function_call", CallID: "call_old"}}},
		Transcript: []anthropic.MessageParam{{
			Role:    "assistant",
			Content: []anthropic.ContentBlock{{Type: "tool_use", ID: "toolu_old"}},
		}},
		CreatedAt: time.Now().Unix() - 10,
	})
	store.Cleanup(time.Now())
	if _, ok := store.FindToolCall("resp_old", "call_old"); ok {
		t.Fatal("expected cleanup to remove tool call index")
	}
}
