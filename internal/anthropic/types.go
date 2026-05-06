package anthropic

import "encoding/json"

type CreateMessageRequest struct {
	Model                  string         `json:"model"`
	MaxTokens              int            `json:"max_tokens"`
	Messages               []MessageParam `json:"messages"`
	System                 string         `json:"system,omitempty"`
	Stream                 bool           `json:"stream,omitempty"`
	Temperature            *float64       `json:"temperature,omitempty"`
	TopP                   *float64       `json:"top_p,omitempty"`
	Tools                  []Tool         `json:"tools,omitempty"`
	ToolChoice             *ToolChoice    `json:"tool_choice,omitempty"`
	DisableParallelToolUse *bool          `json:"disable_parallel_tool_use,omitempty"`
}

type MessageParam struct {
	Role    string         `json:"role"`
	Content []ContentBlock `json:"content"`
}

type ContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	Source    *ImageSource    `json:"source,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
}

type ImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

type ToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

type MessageResponse struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Model        string         `json:"model"`
	Content      []ContentBlock `json:"content"`
	StopReason   string         `json:"stop_reason"`
	StopSequence string         `json:"stop_sequence,omitempty"`
	Usage        Usage          `json:"usage,omitempty"`
}

type Usage struct {
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
}

type ErrorResponse struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

type StreamEvent struct {
	Type         string           `json:"type"`
	Index        int              `json:"index,omitempty"`
	Message      *MessageResponse `json:"message,omitempty"`
	ContentBlock *ContentBlock    `json:"content_block,omitempty"`
	Delta        *StreamDelta     `json:"delta,omitempty"`
	Error        *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type StreamDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	StopReason  string `json:"stop_reason,omitempty"`
}
