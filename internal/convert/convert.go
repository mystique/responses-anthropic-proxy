package convert

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"uni-api/internal/anthropic"
	"uni-api/internal/openai"
)

const defaultMaxTokens = 4096

func CreateResponseToMessage(req openai.CreateResponseRequest, previous []anthropic.MessageParam, model string) (anthropic.CreateMessageRequest, error) {
	maxTokens := defaultMaxTokens
	if req.MaxOutputTokens != nil && *req.MaxOutputTokens > 0 {
		maxTokens = *req.MaxOutputTokens
	}
	out := anthropic.CreateMessageRequest{
		Model:                  model,
		MaxTokens:              maxTokens,
		Messages:               cloneMessages(previous),
		System:                 req.Instructions,
		Temperature:            req.Temperature,
		TopP:                   req.TopP,
		Tools:                  convertTools(req.Tools),
		ToolChoice:             convertToolChoice(req.ToolChoice),
		DisableParallelToolUse: disableParallel(req.ParallelToolCalls),
		Stream:                 req.WantsStream(),
	}
	inputMessages, err := convertInput(req.Input)
	if err != nil {
		return out, err
	}
	out.Messages = append(out.Messages, inputMessages...)
	return out, nil
}

func MessageToResponse(msg anthropic.MessageResponse, responseID string, createdAt int64) (openai.Response, []anthropic.MessageParam, error) {
	status := "completed"
	if msg.StopReason == "max_tokens" {
		status = "incomplete"
	}
	resp := openai.NewBaseResponse(responseID, msg.Model, status, createdAt)
	var text strings.Builder
	for i, block := range msg.Content {
		switch block.Type {
		case "text":
			text.WriteString(block.Text)
			resp.Output = append(resp.Output, openai.OutputItem{
				ID:      fmt.Sprintf("%s_msg_%d", responseID, i),
				Type:    "message",
				Status:  "completed",
				Role:    "assistant",
				Content: []openai.ContentItem{{Type: "output_text", Text: block.Text, Annotations: []any{}}},
			})
		case "tool_use":
			args := string(block.Input)
			if args == "" {
				args = "{}"
			}
			resp.Output = append(resp.Output, openai.OutputItem{
				ID:        block.ID,
				Type:      "function_call",
				Status:    "completed",
				CallID:    block.ID,
				Name:      block.Name,
				Arguments: args,
			})
		}
	}
	resp.OutputText = text.String()
	if msg.Usage.InputTokens != 0 || msg.Usage.OutputTokens != 0 {
		resp.Usage = &openai.Usage{
			InputTokens:  msg.Usage.InputTokens,
			OutputTokens: msg.Usage.OutputTokens,
			TotalTokens:  msg.Usage.InputTokens + msg.Usage.OutputTokens,
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

func convertInput(raw openai.RawJSON) ([]anthropic.MessageParam, error) {
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
	var pendingUser []anthropic.ContentBlock
	flushUser := func() {
		if len(pendingUser) > 0 {
			messages = append(messages, anthropic.MessageParam{Role: "user", Content: pendingUser})
			pendingUser = nil
		}
	}
	for _, item := range items {
		switch item.Type {
		case "message":
			flushUser()
			blocks := convertContentItems(item.Content)
			role := item.Role
			if role != "assistant" {
				role = "user"
			}
			messages = append(messages, anthropic.MessageParam{Role: role, Content: blocks})
		case "input_text":
			pendingUser = append(pendingUser, anthropic.ContentBlock{Type: "text", Text: item.Text})
		case "input_image":
			pendingUser = append(pendingUser, convertImage(item))
		case "function_call_output":
			pendingUser = append(pendingUser, anthropic.ContentBlock{Type: "tool_result", ToolUseID: item.CallID, Content: item.Output})
		}
	}
	flushUser()
	return messages, nil
}

type inputItem struct {
	Type     string        `json:"type"`
	Role     string        `json:"role,omitempty"`
	Content  []contentItem `json:"content,omitempty"`
	Text     string        `json:"text,omitempty"`
	ImageURL string        `json:"image_url,omitempty"`
	CallID   string        `json:"call_id,omitempty"`
	Output   string        `json:"output,omitempty"`
}

type contentItem struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

func convertContentItems(items []contentItem) []anthropic.ContentBlock {
	blocks := make([]anthropic.ContentBlock, 0, len(items))
	for _, item := range items {
		switch item.Type {
		case "input_text", "output_text", "text":
			blocks = append(blocks, anthropic.ContentBlock{Type: "text", Text: item.Text})
		case "input_image":
			blocks = append(blocks, convertImage(inputItem{ImageURL: item.ImageURL}))
		}
	}
	return blocks
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

func convertTools(tools []openai.Tool) []anthropic.Tool {
	out := make([]anthropic.Tool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != "function" {
			continue
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
			continue
		}
		if params.IsZero() {
			params = openai.RawJSON(`{"type":"object","properties":{}}`)
		}
		out = append(out, anthropic.Tool{Name: name, Description: description, InputSchema: json.RawMessage(params)})
	}
	return out
}

func convertToolChoice(raw openai.RawJSON) *anthropic.ToolChoice {
	if raw.IsZero() {
		return nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		switch s {
		case "auto", "any":
			return &anthropic.ToolChoice{Type: s}
		case "none":
			return &anthropic.ToolChoice{Type: "auto"}
		default:
			return &anthropic.ToolChoice{Type: "auto"}
		}
	}
	var obj struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil && obj.Type == "function" && obj.Name != "" {
		return &anthropic.ToolChoice{Type: "tool", Name: obj.Name}
	}
	return &anthropic.ToolChoice{Type: "auto"}
}

func disableParallel(v *bool) *bool {
	if v == nil {
		return nil
	}
	out := !*v
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
