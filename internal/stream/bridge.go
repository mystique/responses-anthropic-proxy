package stream

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"uni-api/internal/anthropic"
)

type toolState struct {
	ID        string
	Name      string
	Arguments strings.Builder
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
	text := map[int]*strings.Builder{}
	var blocks []anthropic.ContentBlock
	scanner := bufio.NewScanner(r)
	var dataLines []string
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			var err error
			blocks, err = handleSSEData(w, responseID, createdAt, strings.Join(dataLines, "\n"), tools, text, blocks)
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
		blocks, err = handleSSEData(w, responseID, createdAt, strings.Join(dataLines, "\n"), tools, text, blocks)
		if err != nil {
			return anthropic.MessageParam{}, err
		}
	}
	if err := scanner.Err(); err != nil {
		return anthropic.MessageParam{}, err
	}
	return anthropic.MessageParam{Role: "assistant", Content: blocks}, nil
}

func handleSSEData(w io.Writer, responseID string, createdAt int64, data string, tools map[int]*toolState, text map[int]*strings.Builder, blocks []anthropic.ContentBlock) ([]anthropic.ContentBlock, error) {
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
			text[event.Index] = &strings.Builder{}
		}
		if event.ContentBlock != nil && event.ContentBlock.Type == "tool_use" {
			tools[event.Index] = &toolState{ID: event.ContentBlock.ID, Name: event.ContentBlock.Name}
		}
	case "content_block_delta":
		if event.Delta == nil {
			return blocks, nil
		}
		switch event.Delta.Type {
		case "text_delta":
			if text[event.Index] == nil {
				text[event.Index] = &strings.Builder{}
			}
			text[event.Index].WriteString(event.Delta.Text)
			return blocks, writeEvent(w, map[string]any{
				"type":          "response.output_text.delta",
				"response_id":   responseID,
				"output_index":  event.Index,
				"content_index": 0,
				"delta":         event.Delta.Text,
			})
		case "input_json_delta":
			if state := tools[event.Index]; state != nil {
				state.Arguments.WriteString(event.Delta.PartialJSON)
			}
		}
	case "content_block_stop":
		if state := tools[event.Index]; state != nil {
			args := state.Arguments.String()
			if args == "" {
				args = "{}"
			}
			blocks = append(blocks, anthropic.ContentBlock{Type: "tool_use", ID: state.ID, Name: state.Name, Input: json.RawMessage(args)})
			return blocks, writeEvent(w, map[string]any{
				"type":         "response.function_call_arguments.done",
				"response_id":  responseID,
				"item_id":      state.ID,
				"output_index": event.Index,
				"arguments":    args,
			})
		}
		if text[event.Index] != nil {
			blocks = append(blocks, anthropic.ContentBlock{Type: "text", Text: text[event.Index].String()})
		}
		return blocks, writeEvent(w, map[string]any{
			"type":          "response.output_text.done",
			"response_id":   responseID,
			"output_index":  event.Index,
			"content_index": 0,
		})
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
