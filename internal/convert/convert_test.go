package convert_test

import (
	"encoding/json"
	"errors"
	"testing"

	"uni-api/internal/anthropic"
	"uni-api/internal/convert"
	"uni-api/internal/openai"
)

func TestCreateResponseToMessageConvertsCoreFieldsAndTools(t *testing.T) {
	req := openai.CreateResponseRequest{
		Model:             "ignored-model",
		Instructions:      "You are concise.",
		Input:             openai.RawJSON(`"Hello"`),
		MaxOutputTokens:   intPtr(123),
		Temperature:       floatPtr(0.2),
		TopP:              floatPtr(0.9),
		ParallelToolCalls: boolPtr(false),
		Tools: []openai.Tool{{
			Type:        "function",
			Name:        "lookup",
			Description: "Lookup a thing",
			Parameters:  openai.RawJSON(`{"type":"object","properties":{"q":{"type":"string"}}}`),
		}},
		ToolChoice: openai.RawJSON(`{"type":"function","name":"lookup"}`),
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	if got.Model != "claude-test" || got.MaxTokens != 123 || got.System != "You are concise." {
		t.Fatalf("unexpected core fields: %+v", got)
	}
	if got.Temperature == nil || *got.Temperature != 0.2 || got.TopP == nil || *got.TopP != 0.9 {
		t.Fatalf("sampling fields not mapped: %+v", got)
	}
	if len(got.Messages) != 1 || got.Messages[0].Role != "user" {
		t.Fatalf("unexpected messages: %+v", got.Messages)
	}
	if got.Messages[0].Content[0].Type != "text" || got.Messages[0].Content[0].Text != "Hello" {
		t.Fatalf("string input not converted: %+v", got.Messages[0].Content)
	}
	if len(got.Tools) != 1 || got.Tools[0].Name != "lookup" {
		t.Fatalf("function tool not converted: %+v", got.Tools)
	}
	if got.ToolChoice == nil || got.ToolChoice.Type != "tool" || got.ToolChoice.Name != "lookup" {
		t.Fatalf("tool choice not mapped: %+v", got.ToolChoice)
	}
	if got.DisableParallelToolUse == nil || *got.DisableParallelToolUse != true {
		t.Fatalf("parallel_tool_calls=false should disable Anthropic parallel tools: %+v", got.DisableParallelToolUse)
	}
}

func TestFunctionCallOutputConvertsToToolResult(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`[
			{"type":"function_call_output","call_id":"call_123","output":"{\"ok\":true}"},
			{"type":"input_text","text":"next"}
		]`),
	}

	got, err := convert.CreateResponseToMessageWithContext(req, nil, "claude-test", convert.Context{
		ToolResolver: func(callID string) (string, bool) {
			if callID == "call_123" {
				return "toolu_123", true
			}
			return "", false
		},
	})
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	if len(got.Messages) != 1 || got.Messages[0].Role != "user" {
		t.Fatalf("unexpected messages: %+v", got.Messages)
	}
	block := got.Messages[0].Content[0]
	if block.Type != "tool_result" || block.ToolUseID != "toolu_123" || block.Content != "{\"ok\":true}" {
		t.Fatalf("function_call_output not converted to tool_result: %+v", block)
	}
	if got.Messages[0].Content[1].Type != "text" {
		t.Fatalf("tool_result should be ordered before text: %+v", got.Messages[0].Content)
	}
}

func TestFunctionCallOutputResolverMissReturnsToolCallNotFound(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`[{"type":"function_call_output","call_id":"call_missing","output":"{}"}]`),
	}

	_, err := convert.CreateResponseToMessageWithContext(req, nil, "claude-test", convert.Context{
		ToolResolver: func(string) (string, bool) { return "", false },
	})

	var inputErr *convert.InputError
	if err == nil || !errors.As(err, &inputErr) {
		t.Fatalf("expected InputError, got %T %v", err, err)
	}
	if inputErr.Code != "tool_call_not_found" {
		t.Fatalf("unexpected error code: %q", inputErr.Code)
	}
}

func TestFunctionCallOutputContentListConvertsSupportedBlocks(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`[
			{"type":"function_call_output","call_id":"call_123","output":[
				{"type":"output_text","text":"ok"},
				{"type":"input_image","image_url":"https://example.test/a.png"}
			]}
		]`),
	}

	got, err := convert.CreateResponseToMessageWithContext(req, nil, "claude-test", convert.Context{
		ToolResolver: func(string) (string, bool) { return "toolu_123", true },
	})
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	content, ok := got.Messages[0].Content[0].Content.([]anthropic.ContentBlock)
	if !ok {
		t.Fatalf("expected content block list, got %T", got.Messages[0].Content[0].Content)
	}
	if len(content) != 2 || content[0].Type != "text" || content[0].Text != "ok" || content[1].Type != "image" {
		t.Fatalf("unexpected tool_result content: %+v", content)
	}
}

func TestMessageToResponseConvertsTextToolUseAndStatus(t *testing.T) {
	msg := anthropic.MessageResponse{
		ID:         "msg_123",
		Model:      "claude-test",
		Role:       "assistant",
		StopReason: "tool_use",
		Content: []anthropic.ContentBlock{{
			Type: "text",
			Text: "Hi",
		}, {
			Type:  "tool_use",
			ID:    "toolu_123",
			Name:  "lookup",
			Input: json.RawMessage(`{"q":"abc"}`),
		}},
		Usage: anthropic.Usage{InputTokens: 10, OutputTokens: 5},
	}

	got, transcript, err := convert.MessageToResponse(msg, "resp_123", 111)
	if err != nil {
		t.Fatalf("MessageToResponse returned error: %v", err)
	}

	if got.ID != "resp_123" || got.Status != "completed" || got.OutputText != "Hi" {
		t.Fatalf("unexpected response: %+v", got)
	}
	if len(got.Output) != 2 {
		t.Fatalf("expected two output items: %+v", got.Output)
	}
	if got.Output[1].Type != "function_call" || got.Output[1].CallID != "toolu_123" || got.Output[1].Arguments != `{"q":"abc"}` {
		t.Fatalf("tool_use not converted: %+v", got.Output[1])
	}
	if len(transcript) != 1 || transcript[0].Role != "assistant" || len(transcript[0].Content) != 2 {
		t.Fatalf("unexpected transcript: %+v", transcript)
	}
}

func TestMessageToResponseMapsMaxTokensToIncomplete(t *testing.T) {
	msg := anthropic.MessageResponse{
		ID:         "msg_123",
		Model:      "claude-test",
		Role:       "assistant",
		StopReason: "max_tokens",
		Content:    []anthropic.ContentBlock{{Type: "text", Text: "Partial"}},
	}

	got, _, err := convert.MessageToResponse(msg, "resp_123", 111)
	if err != nil {
		t.Fatalf("MessageToResponse returned error: %v", err)
	}
	if got.Status != "incomplete" {
		t.Fatalf("expected incomplete, got %q", got.Status)
	}
}

func intPtr(v int) *int           { return &v }
func floatPtr(v float64) *float64 { return &v }
func boolPtr(v bool) *bool        { return &v }
