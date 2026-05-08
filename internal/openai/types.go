package openai

import (
	"encoding/json"
	"time"
)

type RawJSON json.RawMessage

func (r RawJSON) MarshalJSON() ([]byte, error) {
	if len(r) == 0 {
		return []byte("null"), nil
	}
	return r, nil
}

func (r *RawJSON) UnmarshalJSON(data []byte) error {
	*r = append((*r)[:0], data...)
	return nil
}

func (r RawJSON) IsZero() bool {
	return len(r) == 0 || string(r) == "null"
}

type CreateResponseRequest struct {
	Model              string   `json:"model,omitempty"`
	Input              RawJSON  `json:"input,omitempty"`
	Instructions       string   `json:"instructions,omitempty"`
	PreviousResponseID string   `json:"previous_response_id,omitempty"`
	MaxOutputTokens    *int     `json:"max_output_tokens,omitempty"`
	Temperature        *float64 `json:"temperature,omitempty"`
	TopP               *float64 `json:"top_p,omitempty"`
	ParallelToolCalls  *bool    `json:"parallel_tool_calls,omitempty"`
	Stream             *bool    `json:"stream,omitempty"`
	Tools              []Tool   `json:"tools,omitempty"`
	ToolChoice         RawJSON  `json:"tool_choice,omitempty"`
}

func (r CreateResponseRequest) WantsStream() bool {
	return r.Stream != nil && *r.Stream
}

type Tool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Parameters  RawJSON         `json:"parameters,omitempty"`
	Function    *FunctionTool   `json:"function,omitempty"`
	Raw         json.RawMessage `json:"-"`
}

type FunctionTool struct {
	Name        string  `json:"name"`
	Description string  `json:"description,omitempty"`
	Parameters  RawJSON `json:"parameters,omitempty"`
}

func (t *Tool) UnmarshalJSON(data []byte) error {
	type alias Tool
	var out alias
	if err := json.Unmarshal(data, &out); err != nil {
		return err
	}
	out.Raw = append(out.Raw[:0], data...)
	*t = Tool(out)
	return nil
}

type Response struct {
	ID         string       `json:"id"`
	Object     string       `json:"object"`
	CreatedAt  int64        `json:"created_at"`
	Status     string       `json:"status"`
	Model      string       `json:"model,omitempty"`
	Output     []OutputItem `json:"output"`
	OutputText string       `json:"output_text,omitempty"`
	Usage      *Usage       `json:"usage,omitempty"`
	Error      *ErrorObject `json:"error,omitempty"`
}

type OutputItem struct {
	ID        string        `json:"id,omitempty"`
	Type      string        `json:"type"`
	Status    string        `json:"status,omitempty"`
	Role      string        `json:"role,omitempty"`
	Content   []ContentItem `json:"content,omitempty"`
	CallID    string        `json:"call_id,omitempty"`
	Name      string        `json:"name,omitempty"`
	Arguments string        `json:"arguments,omitempty"`
}

type ContentItem struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	Annotations []any  `json:"annotations,omitempty"`
}

type Usage struct {
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
	TotalTokens  int `json:"total_tokens,omitempty"`
}

type ErrorResponse struct {
	Error ErrorObject `json:"error"`
}

type DeleteResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Deleted bool   `json:"deleted"`
}

type ErrorObject struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param,omitempty"`
	Code    string `json:"code,omitempty"`
}

func NewErrorResponse(message, typ, code string) ErrorResponse {
	return ErrorResponse{Error: ErrorObject{Message: message, Type: typ, Code: code}}
}

func NewBaseResponse(id, model, status string, createdAt int64) Response {
	if createdAt == 0 {
		createdAt = time.Now().Unix()
	}
	return Response{
		ID:        id,
		Object:    "response",
		CreatedAt: createdAt,
		Status:    status,
		Model:     model,
		Output:    []OutputItem{},
	}
}
