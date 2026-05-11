package convert_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"rap/internal/anthropic"
	"rap/internal/convert"
	"rap/internal/openai"
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

func TestCreateResponseToMessageMapsRequiredToolChoiceToAny(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input:      openai.RawJSON(`"Hello"`),
		ToolChoice: openai.RawJSON(`"required"`),
		Tools: []openai.Tool{{
			Type: "function",
			Name: "lookup",
		}},
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	if got.ToolChoice == nil || got.ToolChoice.Type != "any" {
		t.Fatalf("required tool choice not mapped to any: %+v", got.ToolChoice)
	}
}

func TestCreateResponseToMessageMapsStopAndTextFormat(t *testing.T) {
	req := openai.CreateResponseRequest{
		Instructions: "You are concise.",
		Input:        openai.RawJSON(`"Hello"`),
		Stop:         openai.RawJSON(`["END","STOP"]`),
		Text:         openai.RawJSON(`{"format":{"type":"json_object"}}`),
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	if len(got.StopSequences) != 2 || got.StopSequences[0] != "END" || got.StopSequences[1] != "STOP" {
		t.Fatalf("stop sequences not mapped: %+v", got.StopSequences)
	}
	if !strings.Contains(got.System, "You are concise.") || !strings.Contains(got.System, "valid JSON object") {
		t.Fatalf("text.format not added to system guidance: %q", got.System)
	}
}

func TestCreateResponseToMessageRejectsUnsupportedToolType(t *testing.T) {
	for _, toolType := range []string{"file_search_preview", "unknown_preview"} {
		t.Run(toolType, func(t *testing.T) {
			req := openai.CreateResponseRequest{
				Input: openai.RawJSON(`"Hello"`),
				Tools: []openai.Tool{{
					Type: toolType,
				}},
			}

			_, err := convert.CreateResponseToMessage(req, nil, "claude-test")

			var inputErr *convert.InputError
			if err == nil || !errors.As(err, &inputErr) {
				t.Fatalf("expected InputError, got %T %v", err, err)
			}
			if inputErr.Code != "unsupported_tool_type" {
				t.Fatalf("unexpected error code: %q", inputErr.Code)
			}
			if !strings.Contains(inputErr.Message, toolType) {
				t.Fatalf("error should name unsupported tool type, got %q", inputErr.Message)
			}
		})
	}
}

func TestCreateResponseToMessageMapsCustomToolAsFunctionTool(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`"Use apply_patch"`),
		Tools: []openai.Tool{{
			Type:        "custom",
			Name:        "apply_patch",
			Description: "Apply a patch to files",
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
	if tool.Name != "apply_patch" || tool.Description != "Apply a patch to files" {
		t.Fatalf("custom tool metadata not mapped: %+v", tool)
	}
	if string(tool.InputSchema) != `{"type":"object","properties":{}}` {
		t.Fatalf("custom tool should default empty input schema, got %s", tool.InputSchema)
	}
}

func TestCreateResponseToMessageMapsCustomToolChoiceToForcedTool(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input:      openai.RawJSON(`"Use apply_patch"`),
		ToolChoice: openai.RawJSON(`{"type":"custom","name":"apply_patch"}`),
		Tools: []openai.Tool{{
			Type: "custom",
			Name: "apply_patch",
		}},
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	if got.ToolChoice == nil || got.ToolChoice.Type != "tool" || got.ToolChoice.Name != "apply_patch" {
		t.Fatalf("custom tool choice not mapped to forced Anthropic tool: %+v", got.ToolChoice)
	}
}

func TestCreateResponseToMessageIgnoresNamespaceTools(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`"List files"`),
		Tools: []openai.Tool{
			{
				Type:        "function",
				Name:        "exec_command",
				Description: "Runs a shell command",
				Parameters:  openai.RawJSON(`{"type":"object","properties":{"cmd":{"type":"string"}}}`),
			},
			{
				Type: "namespace",
				Name: "mcp__lightpanda__",
			},
		},
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	if len(got.Tools) != 1 || got.Tools[0].Name != "exec_command" {
		t.Fatalf("namespace tool should be skipped without dropping callable tools: %+v", got.Tools)
	}
}

func TestCreateResponseToMessageMapsComputerUsePreviewTool(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`"Use the browser"`),
		Tools: []openai.Tool{{
			Type: "computer_use_preview",
			Raw:  json.RawMessage(`{"type":"computer_use_preview","display_width":1024,"display_height":768,"environment":"browser"}`),
		}},
		Truncation: "auto",
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	if len(got.Tools) != 1 {
		t.Fatalf("expected one computer tool, got %+v", got.Tools)
	}
	tool := got.Tools[0]
	if tool.Type != "computer_20250124" || tool.Name != "computer" || tool.DisplayWidthPx != 1024 || tool.DisplayHeightPx != 768 {
		t.Fatalf("computer tool not mapped: %+v", tool)
	}
	if len(got.Betas) != 1 || got.Betas[0] != "computer-use-2025-01-24" {
		t.Fatalf("computer beta not set: %+v", got.Betas)
	}
}

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

func TestCreateResponseToMessageMapsReasoningEffortToThinking(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input:     openai.RawJSON(`"Hello"`),
		Reasoning: openai.RawJSON(`{"effort":"high"}`),
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	if got.Thinking == nil || got.Thinking.Type != "enabled" || got.Thinking.BudgetTokens != 8192 {
		t.Fatalf("reasoning effort not mapped to thinking: %+v", got.Thinking)
	}
	if got.MaxTokens <= got.Thinking.BudgetTokens {
		t.Fatalf("max_tokens must exceed thinking budget, got max=%d budget=%d", got.MaxTokens, got.Thinking.BudgetTokens)
	}
}

func TestCreateResponseToMessageRejectsForcedToolChoiceWithReasoning(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input:      openai.RawJSON(`"Hello"`),
		Reasoning:  openai.RawJSON(`{"effort":"medium"}`),
		ToolChoice: openai.RawJSON(`{"type":"function","name":"lookup"}`),
		Tools: []openai.Tool{{
			Type: "function",
			Name: "lookup",
		}},
	}

	_, err := convert.CreateResponseToMessage(req, nil, "claude-test")

	var inputErr *convert.InputError
	if err == nil || !errors.As(err, &inputErr) {
		t.Fatalf("expected InputError, got %T %v", err, err)
	}
	if inputErr.Code != "invalid_reasoning_tool_choice" {
		t.Fatalf("unexpected error code: %q", inputErr.Code)
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

func TestComputerCallOutputConvertsToToolResultWithScreenshot(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`[
			{"type":"computer_call_output","call_id":"call_123","output":{"type":"computer_screenshot","image_url":"data:image/png;base64,abc"}}
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

	block := got.Messages[0].Content[0]
	if block.Type != "tool_result" || block.ToolUseID != "toolu_123" {
		t.Fatalf("computer_call_output not converted to tool_result: %+v", block)
	}
	content, ok := block.Content.([]anthropic.ContentBlock)
	if !ok || len(content) != 1 || content[0].Type != "image" || content[0].Source.Type != "base64" || content[0].Source.Data != "abc" {
		t.Fatalf("computer screenshot not converted to image content: %#v", block.Content)
	}
}

func TestCreateResponseToMessageConvertsFileAndDocumentInputsToText(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`[
			{"type":"input_file","file_id":"file_123","filename":"main.go"},
			{"type":"document","title":"Design note","source":{"type":"text","media_type":"text/plain","data":"hello"}}
		]`),
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	if len(got.Messages) != 1 || got.Messages[0].Role != "user" || len(got.Messages[0].Content) != 2 {
		t.Fatalf("unexpected messages: %+v", got.Messages)
	}
	fileText := got.Messages[0].Content[0].Text
	if !strings.Contains(fileText, "input_file") || !strings.Contains(fileText, "file_123") || !strings.Contains(fileText, "main.go") {
		t.Fatalf("file placeholder lost context: %q", fileText)
	}
	docText := got.Messages[0].Content[1].Text
	if !strings.Contains(docText, "document") || !strings.Contains(docText, "Design note") {
		t.Fatalf("document placeholder lost context: %q", docText)
	}
}

func TestCreateResponseToMessageRejectsUnsupportedInputType(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`[{"type":"codex_future_item","value":true}]`),
	}

	_, err := convert.CreateResponseToMessage(req, nil, "claude-test")

	var inputErr *convert.InputError
	if err == nil || !errors.As(err, &inputErr) {
		t.Fatalf("expected InputError, got %T %v", err, err)
	}
	if inputErr.Code != "unsupported_input_type" {
		t.Fatalf("unexpected error code: %q", inputErr.Code)
	}
	if !strings.Contains(inputErr.Message, "codex_future_item") {
		t.Fatalf("error should name unsupported input type, got %q", inputErr.Message)
	}
}

func TestCreateResponseToMessageIgnoresReasoningInputItems(t *testing.T) {
	req := openai.CreateResponseRequest{
		Input: openai.RawJSON(`[
			{
				"type":"reasoning",
				"summary":[{"type":"summary_text","text":"considering options"}],
				"content":[{"type":"reasoning_text","text":"considering options"}],
				"encrypted_content":"sig_123"
			},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"Continue"}]}
		]`),
	}

	got, err := convert.CreateResponseToMessage(req, nil, "claude-test")
	if err != nil {
		t.Fatalf("CreateResponseToMessage returned error: %v", err)
	}

	if len(got.Messages) != 1 {
		t.Fatalf("expected only the user message to be converted, got %+v", got.Messages)
	}
	if got.Messages[0].Role != "user" || len(got.Messages[0].Content) != 1 || got.Messages[0].Content[0].Text != "Continue" {
		t.Fatalf("unexpected converted messages: %+v", got.Messages)
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

func TestMessageToResponseMapsPauseTurnAndRefusal(t *testing.T) {
	pause := anthropic.MessageResponse{
		ID:         "msg_pause",
		Model:      "claude-test",
		Role:       "assistant",
		StopReason: "pause_turn",
		Content:    []anthropic.ContentBlock{{Type: "text", Text: "Paused"}},
	}
	got, _, err := convert.MessageToResponse(pause, "resp_pause", 111)
	if err != nil {
		t.Fatalf("MessageToResponse returned error: %v", err)
	}
	if got.Status != "incomplete" {
		t.Fatalf("pause_turn should map to incomplete, got %q", got.Status)
	}

	refusal := anthropic.MessageResponse{
		ID:         "msg_refusal",
		Model:      "claude-test",
		Role:       "assistant",
		StopReason: "refusal",
		Content:    []anthropic.ContentBlock{{Type: "text", Text: "I cannot help with that."}},
	}
	got, _, err = convert.MessageToResponse(refusal, "resp_refusal", 111)
	if err != nil {
		t.Fatalf("MessageToResponse returned error: %v", err)
	}
	if got.Status != "failed" || got.Error == nil || got.Error.Code != "refusal" {
		t.Fatalf("refusal should map to failed refusal error, got %+v", got)
	}
}

func TestMessageToResponseMapsUsageDetails(t *testing.T) {
	msg := anthropic.MessageResponse{
		ID:         "msg_123",
		Model:      "claude-test",
		Role:       "assistant",
		StopReason: "end_turn",
		Content:    []anthropic.ContentBlock{{Type: "text", Text: "Hi"}},
		Usage: anthropic.Usage{
			InputTokens:              10,
			OutputTokens:             5,
			CacheCreationInputTokens: 2,
			CacheReadInputTokens:     3,
		},
	}

	got, _, err := convert.MessageToResponse(msg, "resp_123", 111)
	if err != nil {
		t.Fatalf("MessageToResponse returned error: %v", err)
	}

	if got.Usage == nil || got.Usage.InputTokensDetails == nil || got.Usage.InputTokensDetails.CachedTokens != 3 || got.Usage.InputTokensDetails.CacheCreationTokens != 2 {
		t.Fatalf("usage details not mapped: %+v", got.Usage)
	}
}

func TestMessageToResponseConvertsThinkingBlocksToReasoningItems(t *testing.T) {
	msg := anthropic.MessageResponse{
		ID:         "msg_123",
		Model:      "claude-test",
		Role:       "assistant",
		StopReason: "end_turn",
		Content: []anthropic.ContentBlock{{
			Type:      "thinking",
			Thinking:  "considering options",
			Signature: "sig_123",
		}, {
			Type: "redacted_thinking",
			Data: "encrypted_123",
		}, {
			Type: "text",
			Text: "Answer",
		}},
	}

	got, transcript, err := convert.MessageToResponse(msg, "resp_123", 111)
	if err != nil {
		t.Fatalf("MessageToResponse returned error: %v", err)
	}

	if len(got.Output) != 3 {
		t.Fatalf("expected reasoning, redacted reasoning, and text output: %+v", got.Output)
	}
	if got.Output[0].Type != "reasoning" || len(got.Output[0].Summary) != 1 || got.Output[0].Summary[0].Text != "considering options" {
		t.Fatalf("thinking block not converted to reasoning summary: %+v", got.Output[0])
	}
	if len(got.Output[0].Content) != 1 || got.Output[0].Content[0].Type != "reasoning_text" || got.Output[0].Content[0].Text != "considering options" {
		t.Fatalf("thinking block not converted to reasoning content: %+v", got.Output[0])
	}
	if got.Output[0].EncryptedContent != "sig_123" {
		t.Fatalf("thinking signature not preserved: %+v", got.Output[0])
	}
	if got.Output[1].Type != "reasoning" || got.Output[1].EncryptedContent != "encrypted_123" {
		t.Fatalf("redacted thinking not converted to encrypted reasoning: %+v", got.Output[1])
	}
	if len(transcript) != 1 || len(transcript[0].Content) != 3 || transcript[0].Content[0].Type != "thinking" || transcript[0].Content[1].Type != "redacted_thinking" {
		t.Fatalf("thinking blocks not preserved in transcript: %+v", transcript)
	}
}

func TestMessageToResponseWithIncludeControlsEncryptedReasoning(t *testing.T) {
	msg := anthropic.MessageResponse{
		ID:         "msg_123",
		Model:      "claude-test",
		Role:       "assistant",
		StopReason: "end_turn",
		Content: []anthropic.ContentBlock{{
			Type:      "thinking",
			Thinking:  "hidden",
			Signature: "sig_123",
		}},
	}

	got, _, err := convert.MessageToResponseWithOptions(msg, "resp_123", 111, convert.ResponseOptions{})
	if err != nil {
		t.Fatalf("MessageToResponse returned error: %v", err)
	}
	if got.Output[0].EncryptedContent != "" {
		t.Fatalf("encrypted reasoning should be omitted by default: %+v", got.Output[0])
	}

	got, _, err = convert.MessageToResponseWithOptions(msg, "resp_123", 111, convert.ResponseOptions{Include: []string{"reasoning.encrypted_content"}})
	if err != nil {
		t.Fatalf("MessageToResponse returned error: %v", err)
	}
	if got.Output[0].EncryptedContent != "sig_123" {
		t.Fatalf("encrypted reasoning should be included when requested: %+v", got.Output[0])
	}
}

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
	annotation, ok := got.Output[1].Content[0].Annotations[0].(openai.Annotation)
	if !ok {
		t.Fatalf("citation annotation has unexpected type: %+v", got.Output[1].Content[0].Annotations[0])
	}
	if annotation.URL != "https://example.com/openai" {
		t.Fatalf("citation URL not mapped: %+v", annotation)
	}
	if len(transcript) != 1 || len(transcript[0].Content) != 3 || transcript[0].Content[1].Type != "web_search_tool_result" {
		t.Fatalf("web search blocks not preserved in transcript: %+v", transcript)
	}
}

func TestMessageToResponseConvertsComputerToolUseToComputerCall(t *testing.T) {
	msg := anthropic.MessageResponse{
		ID:         "msg_123",
		Model:      "claude-test",
		Role:       "assistant",
		StopReason: "tool_use",
		Content: []anthropic.ContentBlock{{
			Type:  "tool_use",
			ID:    "toolu_123",
			Name:  "computer",
			Input: json.RawMessage(`{"action":"left_click","coordinate":[10,20]}`),
		}},
	}

	got, transcript, err := convert.MessageToResponse(msg, "resp_123", 111)
	if err != nil {
		t.Fatalf("MessageToResponse returned error: %v", err)
	}

	if len(got.Output) != 1 || got.Output[0].Type != "computer_call" || got.Output[0].CallID != "toolu_123" {
		t.Fatalf("computer tool_use not converted to computer_call: %+v", got.Output)
	}
	if string(got.Output[0].Action) != `{"type":"click","button":"left","x":10,"y":20}` {
		t.Fatalf("computer action not converted: %s", got.Output[0].Action)
	}
	if len(transcript) != 1 || transcript[0].Content[0].Name != "computer" {
		t.Fatalf("computer tool_use not preserved in transcript: %+v", transcript)
	}
}

func TestMessageToResponseConvertsMoreComputerActions(t *testing.T) {
	cases := []struct {
		name string
		in   json.RawMessage
		want string
	}{{
		name: "mouse_move",
		in:   json.RawMessage(`{"action":"mouse_move","coordinate":[30,40]}`),
		want: `{"type":"move","x":30,"y":40}`,
	}, {
		name: "scroll",
		in:   json.RawMessage(`{"action":"scroll","scroll_direction":"down","scroll_amount":5,"coordinate":[1,2]}`),
		want: `{"type":"scroll","x":1,"y":2,"scroll_x":0,"scroll_y":5}`,
	}, {
		name: "key",
		in:   json.RawMessage(`{"action":"key","text":"ENTER"}`),
		want: `{"type":"keypress","keys":["ENTER"]}`,
	}, {
		name: "drag",
		in:   json.RawMessage(`{"action":"left_click_drag","coordinate":[10,20]}`),
		want: `{"type":"drag","x":10,"y":20}`,
	}}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			msg := anthropic.MessageResponse{
				ID:         "msg_123",
				Model:      "claude-test",
				Role:       "assistant",
				StopReason: "tool_use",
				Content: []anthropic.ContentBlock{{
					Type:  "tool_use",
					ID:    "toolu_123",
					Name:  "computer",
					Input: tc.in,
				}},
			}

			got, _, err := convert.MessageToResponse(msg, "resp_123", 111)
			if err != nil {
				t.Fatalf("MessageToResponse returned error: %v", err)
			}
			if string(got.Output[0].Action) != tc.want {
				t.Fatalf("unexpected action: got %s want %s", got.Output[0].Action, tc.want)
			}
		})
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
