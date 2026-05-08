package convert_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"

	"responses-anthropic-proxy/internal/anthropic"
	"responses-anthropic-proxy/internal/convert"
	"responses-anthropic-proxy/internal/openai"
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
	body, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var encoded map[string]any
	if err := json.Unmarshal(body, &encoded); err != nil {
		t.Fatal(err)
	}
	if _, ok := encoded["disable_parallel_tool_use"]; ok {
		t.Fatalf("disable_parallel_tool_use must not be a top-level field, got %s", body)
	}
	if !bytes.Contains(body, []byte(`"tool_choice":{"type":"tool","name":"lookup","disable_parallel_tool_use":true}`)) {
		t.Fatalf("parallel_tool_calls=false should be nested under forced tool_choice, got %s", body)
	}
}

func TestCreateResponseToMessageNestsParallelToolDisableInAutoToolChoice(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input:             openai.RawJSON(`"Hello"`),
		ParallelToolCalls: boolPtr(false),
		Tools: []openai.Tool{{
			Type: "function",
			Name: "lookup",
		}},
		ToolChoice: openai.RawJSON(`"auto"`),
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	body, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(body, []byte(`"disable_parallel_tool_use":true,`)) {
		t.Fatalf("disable_parallel_tool_use must not be a top-level field, got %s", body)
	}
	if !bytes.Contains(body, []byte(`"tool_choice":{"type":"auto","disable_parallel_tool_use":true}`)) {
		t.Fatalf("parallel_tool_calls=false should be nested in tool_choice, got %s", body)
	}
}

func TestCreateResponseToMessageOmitsParallelToolDisableWhenParallelAllowed(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input:             openai.RawJSON(`"Hello"`),
		ParallelToolCalls: boolPtr(true),
		Tools: []openai.Tool{{
			Type: "function",
			Name: "lookup",
		}},
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	body, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(body, []byte(`"disable_parallel_tool_use"`)) {
		t.Fatalf("parallel_tool_calls=true should not send Anthropic disable flag, got %s", body)
	}
	if got.ToolChoice != nil {
		t.Fatalf("parallel_tool_calls=true alone should not add tool_choice, got %+v", got.ToolChoice)
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

func TestInlineFunctionCallAndOutputConvertToAssistantToolUseThenToolResult(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`[
			{"type":"function_call","call_id":"call_123","name":"exec_command","arguments":"{\"cmd\":\"pwd\"}"},
			{"type":"function_call_output","call_id":"call_123","output":"ok"}
		]`),
	}

	got, err := convert.CreateResponseToMessageWithContext(req, nil, "claude-test", convert.Context{
		ToolResolver: func(callID string) (string, bool) {
			if callID == "call_123" {
				return "call_123", true
			}
			return "", false
		},
	})
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	if len(got.Messages) != 2 {
		t.Fatalf("expected assistant tool_use and user tool_result messages, got %+v", got.Messages)
	}
	toolUse := got.Messages[0].Content[0]
	if got.Messages[0].Role != "assistant" || toolUse.Type != "tool_use" || toolUse.ID != "call_123" || toolUse.Name != "exec_command" || string(toolUse.Input) != `{"cmd":"pwd"}` {
		t.Fatalf("unexpected inline tool_use conversion: %+v", got.Messages[0])
	}
	toolResult := got.Messages[1].Content[0]
	if got.Messages[1].Role != "user" || toolResult.Type != "tool_result" || toolResult.ToolUseID != "call_123" || toolResult.Content != "ok" {
		t.Fatalf("unexpected inline tool_result conversion: %+v", got.Messages[1])
	}
}

func TestMultipleInlineFunctionCallsStayInOneAssistantMessage(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`[
			{"type":"function_call","call_id":"call_1","name":"exec_command","arguments":"{\"cmd\":\"pwd\"}"},
			{"type":"function_call","call_id":"call_2","name":"exec_command","arguments":"{\"cmd\":\"ls\"}"},
			{"type":"function_call_output","call_id":"call_1","output":"pwd ok"},
			{"type":"function_call_output","call_id":"call_2","output":"ls ok"}
		]`),
	}

	got, err := convert.CreateResponseToMessageWithContext(req, nil, "claude-test", convert.Context{
		ToolResolver: func(callID string) (string, bool) {
			return callID, callID == "call_1" || callID == "call_2"
		},
	})
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	if len(got.Messages) != 2 {
		t.Fatalf("expected one assistant message followed by one user message, got %+v", got.Messages)
	}
	if got.Messages[0].Role != "assistant" || len(got.Messages[0].Content) != 2 {
		t.Fatalf("expected both tool calls in one assistant message, got %+v", got.Messages[0])
	}
	if got.Messages[0].Content[0].ID != "call_1" || got.Messages[0].Content[1].ID != "call_2" {
		t.Fatalf("tool call order changed: %+v", got.Messages[0].Content)
	}
	if got.Messages[1].Role != "user" || len(got.Messages[1].Content) != 2 {
		t.Fatalf("expected both tool results in one user message, got %+v", got.Messages[1])
	}
	if got.Messages[1].Content[0].ToolUseID != "call_1" || got.Messages[1].Content[1].ToolUseID != "call_2" {
		t.Fatalf("tool result order changed: %+v", got.Messages[1].Content)
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

func TestMessageToResponseSynthesizesMissingToolUseID(t *testing.T) {
	msg := anthropic.MessageResponse{
		ID:         "msg_123",
		Model:      "claude-test",
		Role:       "assistant",
		StopReason: "tool_use",
		Content: []anthropic.ContentBlock{{
			Type:  "tool_use",
			Name:  "lookup",
			Input: json.RawMessage(`{"q":"abc"}`),
		}},
	}

	got, transcript, err := convert.MessageToResponse(msg, "resp_123", 111)
	if err != nil {
		t.Fatalf("MessageToResponse returned error: %v", err)
	}

	if len(got.Output) != 1 || got.Output[0].CallID != "resp_123_tool_0" {
		t.Fatalf("expected synthesized call id, got %+v", got.Output)
	}
	if transcript[0].Content[0].ID != "resp_123_tool_0" {
		t.Fatalf("expected transcript to use synthesized id: %+v", transcript)
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
