package convert

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"rap/internal/anthropic"
	"rap/internal/openai"
)

const defaultMaxTokens = 4096

type Context struct {
	ToolResolver func(callID string) (anthropicToolUseID string, ok bool)
}

type InputError struct {
	Message string
	Code    string
}

func (e *InputError) Error() string {
	return e.Message
}

func CreateResponseToMessage(req openai.CreateResponseRequest, previous []anthropic.MessageParam, model string) (anthropic.CreateMessageRequest, error) {
	return CreateResponseToMessageWithContext(req, previous, model, Context{})
}

func CreateResponseToMessageWithContext(req openai.CreateResponseRequest, previous []anthropic.MessageParam, model string, ctx Context) (anthropic.CreateMessageRequest, error) {
	maxTokens := defaultMaxTokens
	if req.MaxOutputTokens != nil && *req.MaxOutputTokens > 0 {
		maxTokens = *req.MaxOutputTokens
	}
	tools, betas, err := convertTools(req.Tools)
	if err != nil {
		return anthropic.CreateMessageRequest{}, err
	}
	thinking, err := convertReasoning(req.Reasoning)
	if err != nil {
		return anthropic.CreateMessageRequest{}, err
	}
	stopSequences, err := convertStop(req.Stop)
	if err != nil {
		return anthropic.CreateMessageRequest{}, err
	}
	system, err := convertSystem(req.Instructions, req.Text)
	if err != nil {
		return anthropic.CreateMessageRequest{}, err
	}
	out := anthropic.CreateMessageRequest{
		Model:         model,
		MaxTokens:     maxTokens,
		Messages:      cloneMessages(previous),
		System:        system,
		StopSequences: stopSequences,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		Tools:         tools,
		Stream:        req.WantsStream(),
		Thinking:      thinking,
		Betas:         betas,
	}
	if out.Thinking != nil && out.MaxTokens <= out.Thinking.BudgetTokens {
		out.MaxTokens = out.Thinking.BudgetTokens + 1
	}
	out.ToolChoice = convertToolChoice(req.ToolChoice, req.ParallelToolCalls, len(out.Tools) > 0)
	if out.Thinking != nil && out.ToolChoice != nil && (out.ToolChoice.Type == "any" || out.ToolChoice.Type == "tool") {
		return anthropic.CreateMessageRequest{}, &InputError{Message: "reasoning is incompatible with forced tool_choice", Code: "invalid_reasoning_tool_choice"}
	}
	inputMessages, err := convertInput(req.Input, ctx)
	if err != nil {
		return out, err
	}
	out.Messages = append(out.Messages, inputMessages...)
	return out, nil
}

func MessageToResponse(msg anthropic.MessageResponse, responseID string, createdAt int64) (openai.Response, []anthropic.MessageParam, error) {
	return MessageToResponseWithOptions(msg, responseID, createdAt, ResponseOptions{Include: []string{"reasoning.encrypted_content"}})
}

type ResponseOptions struct {
	Include []string
}

func MessageToResponseWithOptions(msg anthropic.MessageResponse, responseID string, createdAt int64, opts ResponseOptions) (openai.Response, []anthropic.MessageParam, error) {
	status := "completed"
	if msg.StopReason == "max_tokens" || msg.StopReason == "pause_turn" {
		status = "incomplete"
	} else if msg.StopReason == "refusal" {
		status = "failed"
	}
	resp := openai.NewBaseResponse(responseID, msg.Model, status, createdAt)
	if msg.StopReason == "refusal" {
		resp.Error = &openai.ErrorObject{Message: "model refused the request", Type: "invalid_request_error", Code: "refusal"}
	}
	includeEncryptedReasoning := includes(opts.Include, "reasoning.encrypted_content")
	var text strings.Builder
	for i, block := range msg.Content {
		switch block.Type {
		case "text":
			text.WriteString(block.Text)
			annotations := annotationsAny(convertCitations(block.Citations))
			resp.Output = append(resp.Output, openai.OutputItem{
				ID:      fmt.Sprintf("%s_msg_%d", responseID, i),
				Type:    "message",
				Status:  "completed",
				Role:    "assistant",
				Content: []openai.ContentItem{{Type: "output_text", Text: block.Text, Annotations: annotations}},
			})
		case "thinking":
			resp.Output = append(resp.Output, openai.OutputItem{
				ID:      fmt.Sprintf("%s_reasoning_%d", responseID, i),
				Type:    "reasoning",
				Status:  "completed",
				Summary: []openai.ReasoningSummaryItem{{Type: "summary_text", Text: block.Thinking}},
				Content: []openai.ContentItem{{Type: "reasoning_text", Text: block.Thinking}},
			})
			if includeEncryptedReasoning {
				resp.Output[len(resp.Output)-1].EncryptedContent = block.Signature
			}
		case "redacted_thinking":
			resp.Output = append(resp.Output, openai.OutputItem{
				ID:      fmt.Sprintf("%s_reasoning_%d", responseID, i),
				Type:    "reasoning",
				Status:  "completed",
				Summary: []openai.ReasoningSummaryItem{},
			})
			if includeEncryptedReasoning {
				resp.Output[len(resp.Output)-1].EncryptedContent = block.Data
			}
		case "tool_use":
			toolUseID := block.ID
			if toolUseID == "" {
				toolUseID = fmt.Sprintf("%s_tool_%d", responseID, i)
				msg.Content[i].ID = toolUseID
			}
			args := string(block.Input)
			if args == "" {
				args = "{}"
			}
			if block.Name == "computer" {
				action, err := convertAnthropicComputerAction(block.Input)
				if err != nil {
					return resp, nil, err
				}
				resp.Output = append(resp.Output, openai.OutputItem{
					ID:                  toolUseID,
					Type:                "computer_call",
					Status:              "completed",
					CallID:              toolUseID,
					Action:              action,
					PendingSafetyChecks: []any{},
				})
				continue
			}
			resp.Output = append(resp.Output, openai.OutputItem{
				ID:        toolUseID,
				Type:      "function_call",
				Status:    "completed",
				CallID:    toolUseID,
				Name:      block.Name,
				Arguments: args,
			})
		case "server_tool_use":
			if block.Name == "web_search" {
				toolUseID := block.ID
				if toolUseID == "" {
					toolUseID = fmt.Sprintf("%s_web_search_%d", responseID, i)
					msg.Content[i].ID = toolUseID
				}
				resp.Output = append(resp.Output, openai.OutputItem{
					ID:     toolUseID,
					Type:   "web_search_call",
					Status: "completed",
				})
			}
		case "web_search_tool_result":
			continue
		}
	}
	resp.OutputText = text.String()
	if msg.Usage.InputTokens != 0 || msg.Usage.OutputTokens != 0 {
		resp.Usage = &openai.Usage{
			InputTokens:  msg.Usage.InputTokens,
			OutputTokens: msg.Usage.OutputTokens,
			TotalTokens:  msg.Usage.InputTokens + msg.Usage.OutputTokens,
		}
		if msg.Usage.CacheReadInputTokens != 0 || msg.Usage.CacheCreationInputTokens != 0 {
			resp.Usage.InputTokensDetails = &openai.InputTokensDetails{
				CachedTokens:        msg.Usage.CacheReadInputTokens,
				CacheCreationTokens: msg.Usage.CacheCreationInputTokens,
			}
		}
	}
	transcript := []anthropic.MessageParam{{
		Role:    "assistant",
		Content: cloneBlocks(msg.Content),
	}}
	return resp, transcript, nil
}

func FailureResponse(responseID string, createdAt int64, message string) openai.Response {
	resp := openai.NewBaseResponse(responseID, "", "failed", createdAt)
	resp.Error = &openai.ErrorObject{Message: message, Type: "upstream_error", Code: "upstream_error"}
	return resp
}

func convertInput(raw openai.RawJSON, ctx Context) ([]anthropic.MessageParam, error) {
	if raw.IsZero() {
		return nil, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []anthropic.MessageParam{{Role: "user", Content: []anthropic.ContentBlock{{Type: "text", Text: s}}}}, nil
	}
	var items []inputItem
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("unsupported input: %w", err)
	}
	var messages []anthropic.MessageParam
	var pendingAssistant []anthropic.ContentBlock
	var pendingUser []anthropic.ContentBlock
	flushAssistant := func() {
		if len(pendingAssistant) > 0 {
			messages = append(messages, anthropic.MessageParam{Role: "assistant", Content: pendingAssistant})
			pendingAssistant = nil
		}
	}
	flushUser := func() {
		if len(pendingUser) > 0 {
			pendingUser = orderToolResultsFirst(pendingUser)
			messages = append(messages, anthropic.MessageParam{Role: "user", Content: pendingUser})
			pendingUser = nil
		}
	}
	for _, item := range items {
		switch item.Type {
		case "message":
			flushAssistant()
			flushUser()
			blocks, err := convertContentItems(item.Content)
			if err != nil {
				return nil, err
			}
			role := item.Role
			if role != "assistant" {
				role = "user"
			}
			messages = append(messages, anthropic.MessageParam{Role: role, Content: blocks})
		case "function_call":
			flushUser()
			block, err := convertFunctionCall(item)
			if err != nil {
				return nil, err
			}
			pendingAssistant = append(pendingAssistant, block)
		case "input_text":
			flushAssistant()
			pendingUser = append(pendingUser, anthropic.ContentBlock{Type: "text", Text: item.Text})
		case "input_image":
			flushAssistant()
			pendingUser = append(pendingUser, convertImage(item))
		case "input_file", "file", "document":
			flushAssistant()
			pendingUser = append(pendingUser, convertFileOrDocument(item))
		case "function_call_output":
			flushAssistant()
			block, err := convertFunctionCallOutput(item, ctx)
			if err != nil {
				return nil, err
			}
			pendingUser = append(pendingUser, block)
		case "computer_call_output":
			flushAssistant()
			block, err := convertComputerCallOutput(item, ctx)
			if err != nil {
				return nil, err
			}
			pendingUser = append(pendingUser, block)
		default:
			return nil, &InputError{Message: "unsupported input type: " + displayType(item.Type), Code: "unsupported_input_type"}
		}
	}
	flushAssistant()
	flushUser()
	return messages, nil
}

func convertReasoning(raw openai.RawJSON) (*anthropic.Thinking, error) {
	if raw.IsZero() {
		return nil, nil
	}
	var req struct {
		Effort string `json:"effort"`
	}
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, &InputError{Message: "reasoning must be an object", Code: "invalid_reasoning"}
	}
	budget := thinkingBudgetForEffort(req.Effort)
	if budget == 0 {
		return nil, nil
	}
	return &anthropic.Thinking{Type: "enabled", BudgetTokens: budget}, nil
}

func convertStop(raw openai.RawJSON) ([]string, error) {
	if raw.IsZero() {
		return nil, nil
	}
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		if one == "" {
			return nil, nil
		}
		return []string{one}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err != nil {
		return nil, &InputError{Message: "stop must be a string or string array", Code: "invalid_stop"}
	}
	return many, nil
}

func convertSystem(instructions string, text openai.RawJSON) (string, error) {
	guidance, err := textFormatGuidance(text)
	if err != nil {
		return "", err
	}
	if guidance == "" {
		return instructions, nil
	}
	if strings.TrimSpace(instructions) == "" {
		return guidance, nil
	}
	return instructions + "\n\n" + guidance, nil
}

func textFormatGuidance(raw openai.RawJSON) (string, error) {
	if raw.IsZero() {
		return "", nil
	}
	var req struct {
		Format struct {
			Type   string          `json:"type"`
			Name   string          `json:"name"`
			Schema json.RawMessage `json:"schema"`
		} `json:"format"`
	}
	if err := json.Unmarshal(raw, &req); err != nil {
		return "", &InputError{Message: "text must be an object", Code: "invalid_text_format"}
	}
	switch req.Format.Type {
	case "", "text":
		return "", nil
	case "json_object":
		return "Return only a valid JSON object. Do not include markdown fences or explanatory text.", nil
	case "json_schema":
		if len(req.Format.Schema) == 0 {
			return "Return only JSON matching the requested schema. Do not include markdown fences or explanatory text.", nil
		}
		return "Return only JSON matching this JSON Schema. Do not include markdown fences or explanatory text.\nSchema: " + string(req.Format.Schema), nil
	default:
		return "", &InputError{Message: "unsupported text.format type: " + displayType(req.Format.Type), Code: "unsupported_text_format"}
	}
}

func thinkingBudgetForEffort(effort string) int {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "minimal", "low":
		return 1024
	case "", "medium":
		return 4096
	case "high":
		return 8192
	default:
		return 0
	}
}

type inputItem struct {
	Type      string         `json:"type"`
	Role      string         `json:"role,omitempty"`
	Content   []contentItem  `json:"content,omitempty"`
	Text      string         `json:"text,omitempty"`
	ImageURL  string         `json:"image_url,omitempty"`
	CallID    string         `json:"call_id,omitempty"`
	Name      string         `json:"name,omitempty"`
	Arguments string         `json:"arguments,omitempty"`
	Output    any            `json:"output,omitempty"`
	FileID    string         `json:"file_id,omitempty"`
	Filename  string         `json:"filename,omitempty"`
	Title     string         `json:"title,omitempty"`
	Source    openai.RawJSON `json:"source,omitempty"`
}

type contentItem struct {
	Type     string         `json:"type"`
	Text     string         `json:"text,omitempty"`
	ImageURL string         `json:"image_url,omitempty"`
	FileID   string         `json:"file_id,omitempty"`
	Filename string         `json:"filename,omitempty"`
	Title    string         `json:"title,omitempty"`
	Source   openai.RawJSON `json:"source,omitempty"`
}

func convertContentItems(items []contentItem) ([]anthropic.ContentBlock, error) {
	blocks := make([]anthropic.ContentBlock, 0, len(items))
	for _, item := range items {
		switch item.Type {
		case "input_text", "output_text", "text":
			blocks = append(blocks, anthropic.ContentBlock{Type: "text", Text: item.Text})
		case "input_image":
			blocks = append(blocks, convertImage(inputItem{ImageURL: item.ImageURL}))
		case "input_file", "file", "document":
			blocks = append(blocks, convertFileOrDocument(inputItem{
				Type:     item.Type,
				FileID:   item.FileID,
				Filename: item.Filename,
				Title:    item.Title,
				Source:   item.Source,
			}))
		default:
			return nil, &InputError{Message: "unsupported content type: " + displayType(item.Type), Code: "unsupported_content_type"}
		}
	}
	return blocks, nil
}

func convertFunctionCallOutput(item inputItem, ctx Context) (anthropic.ContentBlock, error) {
	toolUseID := item.CallID
	if ctx.ToolResolver != nil {
		var ok bool
		toolUseID, ok = ctx.ToolResolver(item.CallID)
		if !ok {
			return anthropic.ContentBlock{}, &InputError{Message: "tool call not found: " + item.CallID, Code: "tool_call_not_found"}
		}
	}
	content, err := convertToolResultContent(item.Output)
	if err != nil {
		return anthropic.ContentBlock{}, err
	}
	return anthropic.ContentBlock{Type: "tool_result", ToolUseID: toolUseID, Content: content}, nil
}

func convertComputerCallOutput(item inputItem, ctx Context) (anthropic.ContentBlock, error) {
	toolUseID := item.CallID
	if ctx.ToolResolver != nil {
		var ok bool
		toolUseID, ok = ctx.ToolResolver(item.CallID)
		if !ok {
			return anthropic.ContentBlock{}, &InputError{Message: "tool call not found: " + item.CallID, Code: "tool_call_not_found"}
		}
	}
	content, err := convertComputerResultContent(item.Output)
	if err != nil {
		return anthropic.ContentBlock{}, err
	}
	return anthropic.ContentBlock{Type: "tool_result", ToolUseID: toolUseID, Content: content}, nil
}

func convertFunctionCall(item inputItem) (anthropic.ContentBlock, error) {
	if item.CallID == "" {
		return anthropic.ContentBlock{}, &InputError{Message: "function_call missing call_id", Code: "tool_call_not_found"}
	}
	if item.Name == "" {
		return anthropic.ContentBlock{}, &InputError{Message: "function_call missing name", Code: "invalid_tool_call"}
	}
	args := strings.TrimSpace(item.Arguments)
	if args == "" {
		args = "{}"
	}
	if !json.Valid([]byte(args)) {
		return anthropic.ContentBlock{}, &InputError{Message: "function_call arguments must be valid JSON", Code: "invalid_tool_call_arguments"}
	}
	return anthropic.ContentBlock{Type: "tool_use", ID: item.CallID, Name: item.Name, Input: json.RawMessage(args)}, nil
}

func convertComputerResultContent(raw any) (any, error) {
	obj, ok := raw.(map[string]any)
	if !ok {
		return nil, &InputError{Message: fmt.Sprintf("unsupported computer_call_output output type: %T", raw), Code: "unsupported_tool_result_output"}
	}
	typ, _ := obj["type"].(string)
	if typ != "computer_screenshot" && typ != "input_image" {
		return nil, &InputError{Message: "unsupported computer_call_output output type: " + displayType(typ), Code: "unsupported_tool_result_content"}
	}
	imageURL, _ := obj["image_url"].(string)
	if imageURL == "" {
		fileID, _ := obj["file_id"].(string)
		return []anthropic.ContentBlock{{Type: "text", Text: "[computer screenshot file_id=" + fileID + "]"}}, nil
	}
	return []anthropic.ContentBlock{convertImage(inputItem{ImageURL: imageURL})}, nil
}

func convertAnthropicComputerAction(raw json.RawMessage) (json.RawMessage, error) {
	var in struct {
		Action          string `json:"action"`
		Coordinate      []int  `json:"coordinate"`
		Text            string `json:"text"`
		ScrollDirection string `json:"scroll_direction"`
		ScrollAmount    int    `json:"scroll_amount"`
	}
	if err := json.Unmarshal(raw, &in); err != nil {
		return nil, err
	}
	switch in.Action {
	case "left_click", "right_click", "middle_click", "double_click":
		button := strings.TrimSuffix(in.Action, "_click")
		if in.Action == "double_click" {
			button = "left"
		}
		out := struct {
			Type   string `json:"type"`
			Button string `json:"button,omitempty"`
			X      int    `json:"x,omitempty"`
			Y      int    `json:"y,omitempty"`
		}{Type: "click", Button: button}
		if in.Action == "double_click" {
			out.Type = "double_click"
		}
		if len(in.Coordinate) >= 2 {
			out.X = in.Coordinate[0]
			out.Y = in.Coordinate[1]
		}
		return json.Marshal(out)
	case "screenshot":
		return json.RawMessage(`{"type":"screenshot"}`), nil
	case "type":
		out := struct {
			Type string `json:"type"`
			Text string `json:"text,omitempty"`
		}{Type: "type", Text: in.Text}
		return json.Marshal(out)
	case "mouse_move":
		out := struct {
			Type string `json:"type"`
			X    int    `json:"x,omitempty"`
			Y    int    `json:"y,omitempty"`
		}{Type: "move"}
		if len(in.Coordinate) >= 2 {
			out.X = in.Coordinate[0]
			out.Y = in.Coordinate[1]
		}
		return json.Marshal(out)
	case "left_click_drag":
		out := struct {
			Type string `json:"type"`
			X    int    `json:"x,omitempty"`
			Y    int    `json:"y,omitempty"`
		}{Type: "drag"}
		if len(in.Coordinate) >= 2 {
			out.X = in.Coordinate[0]
			out.Y = in.Coordinate[1]
		}
		return json.Marshal(out)
	case "scroll":
		amount := in.ScrollAmount
		if amount == 0 {
			amount = 1
		}
		scrollX, scrollY := 0, 0
		switch in.ScrollDirection {
		case "up":
			scrollY = -amount
		case "down", "":
			scrollY = amount
		case "left":
			scrollX = -amount
		case "right":
			scrollX = amount
		}
		out := struct {
			Type    string `json:"type"`
			X       int    `json:"x,omitempty"`
			Y       int    `json:"y,omitempty"`
			ScrollX int    `json:"scroll_x"`
			ScrollY int    `json:"scroll_y"`
		}{Type: "scroll", ScrollX: scrollX, ScrollY: scrollY}
		if len(in.Coordinate) >= 2 {
			out.X = in.Coordinate[0]
			out.Y = in.Coordinate[1]
		}
		return json.Marshal(out)
	case "key":
		out := struct {
			Type string   `json:"type"`
			Keys []string `json:"keys"`
		}{Type: "keypress", Keys: []string{in.Text}}
		return json.Marshal(out)
	default:
		return raw, nil
	}
}

func convertToolResultContent(raw any) (any, error) {
	switch v := raw.(type) {
	case nil:
		return "", nil
	case string:
		return v, nil
	case []any:
		blocks := make([]anthropic.ContentBlock, 0, len(v))
		for _, elem := range v {
			b, err := json.Marshal(elem)
			if err != nil {
				return nil, err
			}
			var item contentItem
			if err := json.Unmarshal(b, &item); err != nil {
				return nil, err
			}
			switch item.Type {
			case "input_text", "output_text", "text":
				blocks = append(blocks, anthropic.ContentBlock{Type: "text", Text: item.Text})
			case "input_image":
				blocks = append(blocks, convertImage(inputItem{ImageURL: item.ImageURL}))
			default:
				return nil, &InputError{Message: "unsupported function_call_output content type: " + item.Type, Code: "unsupported_tool_result_content"}
			}
		}
		return blocks, nil
	default:
		return nil, &InputError{Message: fmt.Sprintf("unsupported function_call_output output type: %T", raw), Code: "unsupported_tool_result_output"}
	}
}

func orderToolResultsFirst(blocks []anthropic.ContentBlock) []anthropic.ContentBlock {
	if len(blocks) < 2 {
		return blocks
	}
	out := make([]anthropic.ContentBlock, 0, len(blocks))
	for _, block := range blocks {
		if block.Type == "tool_result" {
			out = append(out, block)
		}
	}
	for _, block := range blocks {
		if block.Type != "tool_result" {
			out = append(out, block)
		}
	}
	return out
}

func convertImage(item inputItem) anthropic.ContentBlock {
	source := &anthropic.ImageSource{Type: "url", URL: item.ImageURL}
	if strings.HasPrefix(item.ImageURL, "data:") {
		source.Type = "base64"
		parts := strings.SplitN(strings.TrimPrefix(item.ImageURL, "data:"), ";base64,", 2)
		if len(parts) == 2 {
			source.MediaType = parts[0]
			source.Data = parts[1]
			source.URL = ""
		}
	}
	return anthropic.ContentBlock{Type: "image", Source: source}
}

func convertFileOrDocument(item inputItem) anthropic.ContentBlock {
	parts := []string{"OpenAI " + item.Type + " input was provided but is not directly transferable to Anthropic Messages in this proxy stage."}
	if item.FileID != "" {
		parts = append(parts, "file_id="+item.FileID)
	}
	if item.Filename != "" {
		parts = append(parts, "filename="+item.Filename)
	}
	if item.Title != "" {
		parts = append(parts, "title="+item.Title)
	}
	if !item.Source.IsZero() {
		parts = append(parts, "source=present")
	}
	return anthropic.ContentBlock{Type: "text", Text: "[" + strings.Join(parts, " ") + "]"}
}

func displayType(typ string) string {
	if typ == "" {
		return "<missing>"
	}
	return typ
}

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

func annotationsAny(annotations []openai.Annotation) []any {
	if annotations == nil {
		return []any{}
	}
	out := make([]any, 0, len(annotations))
	for _, annotation := range annotations {
		out = append(out, annotation)
	}
	return out
}

func convertTools(tools []openai.Tool) ([]anthropic.Tool, []string, error) {
	out := make([]anthropic.Tool, 0, len(tools))
	var betas []string
	for _, tool := range tools {
		if tool.Type == "computer_use_preview" {
			computerTool, err := convertComputerTool(tool)
			if err != nil {
				return nil, nil, err
			}
			out = append(out, computerTool)
			betas = appendBeta(betas, "computer-use-2025-01-24")
			continue
		}
		if tool.Type == "web_search" || tool.Type == "web_search_preview" {
			webTool, err := convertWebSearchTool(tool)
			if err != nil {
				return nil, nil, err
			}
			out = append(out, webTool)
			continue
		}
		if tool.Type != "function" && tool.Type != "custom" && !(tool.Type == "" && tool.Function != nil) {
			toolType := tool.Type
			if toolType == "" {
				toolType = "<missing>"
			}
			return nil, nil, &InputError{Message: "unsupported tool type: " + toolType, Code: "unsupported_tool_type"}
		}
		name := tool.Name
		description := tool.Description
		params := tool.Parameters
		if tool.Function != nil {
			name = tool.Function.Name
			description = tool.Function.Description
			params = tool.Function.Parameters
		}
		if name == "" {
			return nil, nil, &InputError{Message: "function tool missing name", Code: "invalid_tool"}
		}
		if params.IsZero() {
			params = openai.RawJSON(`{"type":"object","properties":{}}`)
		}
		out = append(out, anthropic.Tool{Name: name, Description: description, InputSchema: json.RawMessage(params)})
	}
	return out, betas, nil
}

func convertWebSearchTool(tool openai.Tool) (anthropic.Tool, error) {
	var spec struct {
		MaxUses        int      `json:"max_uses"`
		AllowedDomains []string `json:"allowed_domains"`
		BlockedDomains []string `json:"blocked_domains"`
		Filters        struct {
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

func convertComputerTool(tool openai.Tool) (anthropic.Tool, error) {
	var spec struct {
		DisplayWidth  int `json:"display_width"`
		DisplayHeight int `json:"display_height"`
		DisplayNumber int `json:"display_number"`
	}
	if len(tool.Raw) != 0 {
		if err := json.Unmarshal(tool.Raw, &spec); err != nil {
			return anthropic.Tool{}, &InputError{Message: "invalid computer_use_preview tool", Code: "invalid_tool"}
		}
	}
	if spec.DisplayWidth <= 0 {
		spec.DisplayWidth = 1024
	}
	if spec.DisplayHeight <= 0 {
		spec.DisplayHeight = 768
	}
	return anthropic.Tool{
		Type:            "computer_20250124",
		Name:            "computer",
		DisplayWidthPx:  spec.DisplayWidth,
		DisplayHeightPx: spec.DisplayHeight,
		DisplayNumber:   spec.DisplayNumber,
	}, nil
}

func appendBeta(betas []string, beta string) []string {
	for _, existing := range betas {
		if existing == beta {
			return betas
		}
	}
	return append(betas, beta)
}

func convertToolChoice(raw openai.RawJSON, parallelToolCalls *bool, hasTools bool) *anthropic.ToolChoice {
	disableParallel := disableParallel(parallelToolCalls)
	if raw.IsZero() {
		if disableParallel != nil && hasTools {
			return &anthropic.ToolChoice{Type: "auto", DisableParallelToolUse: disableParallel}
		}
		return nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		switch s {
		case "auto", "any":
			return &anthropic.ToolChoice{Type: s, DisableParallelToolUse: disableParallel}
		case "required":
			return &anthropic.ToolChoice{Type: "any", DisableParallelToolUse: disableParallel}
		case "none":
			return &anthropic.ToolChoice{Type: "auto", DisableParallelToolUse: disableParallel}
		default:
			return &anthropic.ToolChoice{Type: "auto", DisableParallelToolUse: disableParallel}
		}
	}
	var obj struct {
		Type string `json:"type"`
		Name string `json:"name"`
		Mode string `json:"mode"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil && (obj.Type == "web_search" || obj.Type == "web_search_preview") {
		return &anthropic.ToolChoice{Type: "tool", Name: "web_search", DisableParallelToolUse: disableParallel}
	}
	if err := json.Unmarshal(raw, &obj); err == nil && (obj.Type == "function" || obj.Type == "custom") && obj.Name != "" {
		return &anthropic.ToolChoice{Type: "tool", Name: obj.Name, DisableParallelToolUse: disableParallel}
	}
	if obj.Mode == "required" {
		return &anthropic.ToolChoice{Type: "any", DisableParallelToolUse: disableParallel}
	}
	if obj.Mode == "auto" {
		return &anthropic.ToolChoice{Type: "auto", DisableParallelToolUse: disableParallel}
	}
	return &anthropic.ToolChoice{Type: "auto", DisableParallelToolUse: disableParallel}
}

func includes(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func disableParallel(v *bool) *bool {
	if v == nil || *v {
		return nil
	}
	out := true
	return &out
}

func cloneMessages(in []anthropic.MessageParam) []anthropic.MessageParam {
	out := make([]anthropic.MessageParam, len(in))
	for i, msg := range in {
		out[i] = anthropic.MessageParam{Role: msg.Role, Content: cloneBlocks(msg.Content)}
	}
	return out
}

func cloneBlocks(in []anthropic.ContentBlock) []anthropic.ContentBlock {
	out := make([]anthropic.ContentBlock, len(in))
	copy(out, in)
	return out
}

func NowUnix() int64 {
	return time.Now().Unix()
}
