package stream

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"rap/internal/anthropic"
)

type toolState struct {
	ID        string
	Name      string
	Arguments strings.Builder
}

type textState struct {
	ID   string
	Text strings.Builder
}

type thinkingState struct {
	ID        string
	Thinking  strings.Builder
	Signature string
}

func Bridge(r io.Reader, w io.Writer, responseID string, createdAt int64) error {
	_, err := BridgeWithResult(r, w, responseID, createdAt)
	return err
}

func BridgeWithResult(r io.Reader, w io.Writer, responseID string, createdAt int64) (anthropic.MessageParam, error) {
	if err := writeEvent(w, map[string]any{"type": "response.created", "response": baseResponse(responseID, createdAt, "queued")}); err != nil {
		return anthropic.MessageParam{}, err
	}
	if err := writeEvent(w, map[string]any{"type": "response.in_progress", "response": baseResponse(responseID, createdAt, "in_progress")}); err != nil {
		return anthropic.MessageParam{}, err
	}
	tools := map[int]*toolState{}
	text := map[int]*textState{}
	thinking := map[int]*thinkingState{}
	var blocks []anthropic.ContentBlock
	scanner := bufio.NewScanner(r)
	var dataLines []string
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			var err error
			blocks, err = handleSSEData(w, responseID, createdAt, strings.Join(dataLines, "\n"), tools, text, thinking, blocks)
			if err != nil {
				return anthropic.MessageParam{}, err
			}
			dataLines = nil
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if len(dataLines) > 0 {
		var err error
		blocks, err = handleSSEData(w, responseID, createdAt, strings.Join(dataLines, "\n"), tools, text, thinking, blocks)
		if err != nil {
			return anthropic.MessageParam{}, err
		}
	}
	if err := scanner.Err(); err != nil {
		return anthropic.MessageParam{}, err
	}
	return anthropic.MessageParam{Role: "assistant", Content: blocks}, nil
}

func handleSSEData(w io.Writer, responseID string, createdAt int64, data string, tools map[int]*toolState, text map[int]*textState, thinking map[int]*thinkingState, blocks []anthropic.ContentBlock) ([]anthropic.ContentBlock, error) {
	if strings.TrimSpace(data) == "" || strings.TrimSpace(data) == "[DONE]" {
		return blocks, nil
	}
	var event anthropic.StreamEvent
	if err := json.Unmarshal([]byte(data), &event); err != nil {
		return blocks, err
	}
	switch event.Type {
	case "content_block_start":
		if event.ContentBlock != nil && event.ContentBlock.Type == "text" {
			state := &textState{ID: fmt.Sprintf("%s_msg_%d", responseID, event.Index)}
			text[event.Index] = state
			if err := writeEvent(w, map[string]any{
				"type":         "response.output_item.added",
				"response_id":  responseID,
				"output_index": event.Index,
				"item": map[string]any{
					"id":      state.ID,
					"type":    "message",
					"status":  "in_progress",
					"role":    "assistant",
					"content": []any{},
				},
			}); err != nil {
				return blocks, err
			}
			if err := writeEvent(w, map[string]any{
				"type":          "response.content_part.added",
				"response_id":   responseID,
				"item_id":       state.ID,
				"output_index":  event.Index,
				"content_index": 0,
				"part": map[string]any{
					"type":        "output_text",
					"text":        "",
					"annotations": []any{},
				},
			}); err != nil {
				return blocks, err
			}
		}
		if event.ContentBlock != nil && event.ContentBlock.Type == "tool_use" {
			id := event.ContentBlock.ID
			if id == "" {
				id = fmt.Sprintf("%s_tool_%d", responseID, event.Index)
			}
			state := &toolState{ID: id, Name: event.ContentBlock.Name}
			tools[event.Index] = state
			itemType := "function_call"
			item := map[string]any{
				"id":        state.ID,
				"type":      itemType,
				"status":    "in_progress",
				"call_id":   state.ID,
				"name":      state.Name,
				"arguments": "",
			}
			if state.Name == "computer" {
				itemType = "computer_call"
				item = map[string]any{
					"id":                    state.ID,
					"type":                  itemType,
					"status":                "in_progress",
					"call_id":               state.ID,
					"pending_safety_checks": []any{},
				}
			}
			return blocks, writeEvent(w, map[string]any{
				"type":         "response.output_item.added",
				"response_id":  responseID,
				"output_index": event.Index,
				"item":         item,
			})
		}
		if event.ContentBlock != nil && event.ContentBlock.Type == "thinking" {
			state := &thinkingState{ID: fmt.Sprintf("%s_reasoning_%d", responseID, event.Index)}
			thinking[event.Index] = state
			return blocks, writeEvent(w, map[string]any{
				"type":         "response.output_item.added",
				"response_id":  responseID,
				"output_index": event.Index,
				"item": map[string]any{
					"id":      state.ID,
					"type":    "reasoning",
					"status":  "in_progress",
					"summary": []any{},
					"content": []any{},
				},
			})
		}
		if event.ContentBlock != nil && event.ContentBlock.Type == "redacted_thinking" {
			id := fmt.Sprintf("%s_reasoning_%d", responseID, event.Index)
			blocks = append(blocks, anthropic.ContentBlock{Type: "redacted_thinking", Data: event.ContentBlock.Data})
			if err := writeEvent(w, map[string]any{
				"type":         "response.output_item.added",
				"response_id":  responseID,
				"output_index": event.Index,
				"item": map[string]any{
					"id":                id,
					"type":              "reasoning",
					"status":            "in_progress",
					"summary":           []any{},
					"encrypted_content": event.ContentBlock.Data,
				},
			}); err != nil {
				return blocks, err
			}
			return blocks, writeEvent(w, map[string]any{
				"type":         "response.output_item.done",
				"response_id":  responseID,
				"output_index": event.Index,
				"item": map[string]any{
					"id":                id,
					"type":              "reasoning",
					"status":            "completed",
					"summary":           []any{},
					"encrypted_content": event.ContentBlock.Data,
				},
			})
		}
	case "content_block_delta":
		if event.Delta == nil {
			return blocks, nil
		}
		switch event.Delta.Type {
		case "text_delta":
			if text[event.Index] == nil {
				text[event.Index] = &textState{ID: fmt.Sprintf("%s_msg_%d", responseID, event.Index)}
			}
			state := text[event.Index]
			state.Text.WriteString(event.Delta.Text)
			return blocks, writeEvent(w, map[string]any{
				"type":          "response.output_text.delta",
				"response_id":   responseID,
				"item_id":       state.ID,
				"output_index":  event.Index,
				"content_index": 0,
				"delta":         event.Delta.Text,
			})
		case "input_json_delta":
			if state := tools[event.Index]; state != nil {
				state.Arguments.WriteString(event.Delta.PartialJSON)
				return blocks, writeEvent(w, map[string]any{
					"type":         "response.function_call_arguments.delta",
					"response_id":  responseID,
					"item_id":      state.ID,
					"output_index": event.Index,
					"delta":        event.Delta.PartialJSON,
				})
			}
		case "thinking_delta":
			if thinking[event.Index] == nil {
				thinking[event.Index] = &thinkingState{ID: fmt.Sprintf("%s_reasoning_%d", responseID, event.Index)}
			}
			state := thinking[event.Index]
			state.Thinking.WriteString(event.Delta.Thinking)
			return blocks, writeEvent(w, map[string]any{
				"type":          "response.reasoning_summary_text.delta",
				"response_id":   responseID,
				"item_id":       state.ID,
				"output_index":  event.Index,
				"summary_index": 0,
				"delta":         event.Delta.Thinking,
			})
		case "signature_delta":
			if thinking[event.Index] == nil {
				thinking[event.Index] = &thinkingState{ID: fmt.Sprintf("%s_reasoning_%d", responseID, event.Index)}
			}
			thinking[event.Index].Signature += event.Delta.Signature
		}
	case "content_block_stop":
		if state := tools[event.Index]; state != nil {
			args := state.Arguments.String()
			if args == "" {
				args = "{}"
			}
			blocks = append(blocks, anthropic.ContentBlock{Type: "tool_use", ID: state.ID, Name: state.Name, Input: json.RawMessage(args)})
			if state.Name == "computer" {
				action, err := convertComputerAction(json.RawMessage(args))
				if err != nil {
					return blocks, err
				}
				return blocks, writeEvent(w, map[string]any{
					"type":         "response.output_item.done",
					"response_id":  responseID,
					"output_index": event.Index,
					"item": map[string]any{
						"id":                    state.ID,
						"type":                  "computer_call",
						"status":                "completed",
						"call_id":               state.ID,
						"action":                action,
						"pending_safety_checks": []any{},
					},
				})
			}
			if err := writeEvent(w, map[string]any{
				"type":         "response.function_call_arguments.done",
				"response_id":  responseID,
				"item_id":      state.ID,
				"output_index": event.Index,
				"arguments":    args,
			}); err != nil {
				return blocks, err
			}
			return blocks, writeEvent(w, map[string]any{
				"type":         "response.output_item.done",
				"response_id":  responseID,
				"output_index": event.Index,
				"item": map[string]any{
					"id":        state.ID,
					"type":      "function_call",
					"status":    "completed",
					"call_id":   state.ID,
					"name":      state.Name,
					"arguments": args,
				},
			})
		}
		if state := text[event.Index]; state != nil {
			finalText := state.Text.String()
			blocks = append(blocks, anthropic.ContentBlock{Type: "text", Text: finalText})
			if err := writeEvent(w, map[string]any{
				"type":          "response.output_text.done",
				"response_id":   responseID,
				"item_id":       state.ID,
				"output_index":  event.Index,
				"content_index": 0,
				"text":          finalText,
			}); err != nil {
				return blocks, err
			}
			if err := writeEvent(w, map[string]any{
				"type":          "response.content_part.done",
				"response_id":   responseID,
				"item_id":       state.ID,
				"output_index":  event.Index,
				"content_index": 0,
				"part": map[string]any{
					"type":        "output_text",
					"text":        finalText,
					"annotations": []any{},
				},
			}); err != nil {
				return blocks, err
			}
			return blocks, writeEvent(w, map[string]any{
				"type":         "response.output_item.done",
				"response_id":  responseID,
				"output_index": event.Index,
				"item": map[string]any{
					"id":     state.ID,
					"type":   "message",
					"status": "completed",
					"role":   "assistant",
					"content": []any{map[string]any{
						"type":        "output_text",
						"text":        finalText,
						"annotations": []any{},
					}},
				},
			})
		}
		if state := thinking[event.Index]; state != nil {
			finalThinking := state.Thinking.String()
			blocks = append(blocks, anthropic.ContentBlock{Type: "thinking", Thinking: finalThinking, Signature: state.Signature})
			if err := writeEvent(w, map[string]any{
				"type":          "response.reasoning_summary_text.done",
				"response_id":   responseID,
				"item_id":       state.ID,
				"output_index":  event.Index,
				"summary_index": 0,
				"text":          finalThinking,
			}); err != nil {
				return blocks, err
			}
			return blocks, writeEvent(w, map[string]any{
				"type":         "response.output_item.done",
				"response_id":  responseID,
				"output_index": event.Index,
				"item": map[string]any{
					"id":                state.ID,
					"type":              "reasoning",
					"status":            "completed",
					"summary":           []any{map[string]any{"type": "summary_text", "text": finalThinking}},
					"content":           []any{map[string]any{"type": "reasoning_text", "text": finalThinking}},
					"encrypted_content": state.Signature,
				},
			})
		}
	case "message_stop":
		return blocks, writeEvent(w, map[string]any{"type": "response.completed", "response": baseResponse(responseID, createdAt, "completed")})
	case "error":
		msg := "upstream stream error"
		if event.Error != nil && event.Error.Message != "" {
			msg = event.Error.Message
		}
		return blocks, WriteFailed(w, responseID, createdAt, msg)
	}
	return blocks, nil
}

func WriteFailed(w io.Writer, responseID string, createdAt int64, message string) error {
	return writeEvent(w, map[string]any{
		"type": "response.failed",
		"response": map[string]any{
			"id":         responseID,
			"object":     "response",
			"created_at": createdAt,
			"status":     "failed",
			"error": map[string]any{
				"type":    "upstream_error",
				"message": message,
			},
		},
	})
}

func baseResponse(id string, createdAt int64, status string) map[string]any {
	return map[string]any{"id": id, "object": "response", "created_at": createdAt, "status": status}
}

func convertComputerAction(raw json.RawMessage) (any, error) {
	var in struct {
		Action     string `json:"action"`
		Coordinate []int  `json:"coordinate"`
		Text       string `json:"text"`
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
		out := map[string]any{"type": "click", "button": button}
		if in.Action == "double_click" {
			out["type"] = "double_click"
		}
		if len(in.Coordinate) >= 2 {
			out["x"] = in.Coordinate[0]
			out["y"] = in.Coordinate[1]
		}
		return out, nil
	case "screenshot":
		return map[string]any{"type": "screenshot"}, nil
	case "type":
		return map[string]any{"type": "type", "text": in.Text}, nil
	default:
		var out any
		if err := json.Unmarshal(raw, &out); err != nil {
			return nil, err
		}
		return out, nil
	}
}

func writeEvent(w io.Writer, payload map[string]any) error {
	b, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
		return err
	}
	if flusher, ok := w.(interface{ Flush() }); ok {
		flusher.Flush()
	}
	return nil
}
