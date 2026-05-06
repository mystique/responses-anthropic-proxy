package state_test

import (
	"testing"
	"time"

	"uni-api/internal/anthropic"
	"uni-api/internal/state"
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
