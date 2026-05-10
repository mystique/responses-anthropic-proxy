package stream_test

import (
	"bufio"
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"rap/internal/stream"
)

func TestBridgeTextStream(t *testing.T) {
	input := strings.NewReader(strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude-test","role":"assistant","content":[]}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n"))
	var output bytes.Buffer

	if err := stream.Bridge(input, &output, "resp_1", 111); err != nil {
		t.Fatalf("Bridge returned error: %v", err)
	}

	events := collectEvents(t, output.String())
	assertHasEvent(t, events, "response.created")
	assertHasEvent(t, events, "response.in_progress")
	assertHasEvent(t, events, "response.output_item.added")
	assertHasEvent(t, events, "response.content_part.added")
	assertHasEvent(t, events, "response.output_text.delta")
	assertHasEvent(t, events, "response.output_text.done")
	assertHasEvent(t, events, "response.content_part.done")
	assertHasEvent(t, events, "response.output_item.done")
	assertHasEvent(t, events, "response.completed")
	if !strings.Contains(output.String(), `"delta":"Hel"`) || !strings.Contains(output.String(), `"delta":"lo"`) {
		t.Fatalf("missing text deltas in output:\n%s", output.String())
	}
}

func TestBridgeTextStreamOpensOutputItemBeforeTextDelta(t *testing.T) {
	input := strings.NewReader(strings.Join([]string{
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n"))
	var output bytes.Buffer

	if err := stream.Bridge(input, &output, "resp_1", 111); err != nil {
		t.Fatalf("Bridge returned error: %v", err)
	}

	payloads := collectPayloads(t, output.String())
	added := indexOfEvent(payloads, "response.output_item.added")
	partAdded := indexOfEvent(payloads, "response.content_part.added")
	delta := indexOfEvent(payloads, "response.output_text.delta")
	textDone := indexOfEvent(payloads, "response.output_text.done")
	partDone := indexOfEvent(payloads, "response.content_part.done")
	itemDone := indexOfEvent(payloads, "response.output_item.done")
	if added == -1 || partAdded == -1 || delta == -1 || textDone == -1 || partDone == -1 || itemDone == -1 {
		t.Fatalf("missing stream lifecycle events: %+v", eventTypes(payloads))
	}
	if !(added < partAdded && partAdded < delta && delta < textDone && textDone < partDone && partDone < itemDone) {
		t.Fatalf("unexpected stream event order: %+v", eventTypes(payloads))
	}
	if payloads[delta]["item_id"] == "" || payloads[textDone]["text"] != "Hi" {
		t.Fatalf("text events missing item id or final text: %+v %+v", payloads[delta], payloads[textDone])
	}
}

func TestBridgeToolUseStream(t *testing.T) {
	input := strings.NewReader(strings.Join([]string{
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":":\"abc\"}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n"))
	var output bytes.Buffer

	if err := stream.Bridge(input, &output, "resp_1", 111); err != nil {
		t.Fatalf("Bridge returned error: %v", err)
	}

	events := collectEvents(t, output.String())
	assertHasEvent(t, events, "response.output_item.added")
	assertHasEvent(t, events, "response.function_call_arguments.delta")
	assertHasEvent(t, events, "response.function_call_arguments.done")
	assertHasEvent(t, events, "response.output_item.done")
	if !strings.Contains(output.String(), `"arguments":"{\"q\":\"abc\"}"`) {
		t.Fatalf("missing tool arguments in output:\n%s", output.String())
	}
}

func TestBridgeToolUseStreamSynthesizesMissingToolUseID(t *testing.T) {
	input := strings.NewReader(strings.Join([]string{
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","name":"lookup","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"abc\"}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n"))
	var output bytes.Buffer

	got, err := stream.BridgeWithResult(input, &output, "resp_1", 111)
	if err != nil {
		t.Fatalf("BridgeWithResult returned error: %v", err)
	}

	if len(got.Content) != 1 || got.Content[0].ID == "" {
		t.Fatalf("expected synthesized tool_use id, got %+v", got.Content)
	}
	if got.Content[0].ID != "resp_1_tool_1" {
		t.Fatalf("unexpected synthesized tool_use id: %q", got.Content[0].ID)
	}
	if !strings.Contains(output.String(), `"call_id":"resp_1_tool_1"`) {
		t.Fatalf("stream output did not use synthesized call id:\n%s", output.String())
	}
}

func TestBridgeWithResultReturnsAssistantTranscript(t *testing.T) {
	input := strings.NewReader(strings.Join([]string{
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"abc\"}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":1}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n"))
	var output bytes.Buffer

	got, err := stream.BridgeWithResult(input, &output, "resp_1", 111)
	if err != nil {
		t.Fatalf("BridgeWithResult returned error: %v", err)
	}
	if got.Role != "assistant" || len(got.Content) != 2 {
		t.Fatalf("unexpected transcript: %+v", got)
	}
	if got.Content[0].Type != "text" || got.Content[0].Text != "Hi" {
		t.Fatalf("missing text block: %+v", got.Content[0])
	}
	if got.Content[1].Type != "tool_use" || got.Content[1].ID != "toolu_1" || string(got.Content[1].Input) != `{"q":"abc"}` {
		t.Fatalf("missing tool block: %+v", got.Content[1])
	}
}

func TestBridgeUpstreamErrorEmitsResponseFailed(t *testing.T) {
	input := strings.NewReader("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"bad_request\",\"message\":\"nope\"}}\n\n")
	var output bytes.Buffer

	if err := stream.Bridge(input, &output, "resp_1", 111); err != nil {
		t.Fatalf("Bridge returned error: %v", err)
	}
	assertHasEvent(t, collectEvents(t, output.String()), "response.failed")
}

func collectEvents(t *testing.T, s string) []string {
	t.Helper()
	payloads := collectPayloads(t, s)
	var events []string
	for _, payload := range payloads {
		if typ, ok := payload["type"].(string); ok {
			events = append(events, typ)
		}
	}
	return events
}

func collectPayloads(t *testing.T, s string) []map[string]any {
	t.Helper()
	var payloads []map[string]any
	scanner := bufio.NewScanner(strings.NewReader(s))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			var payload map[string]any
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &payload); err != nil {
				t.Fatalf("invalid json payload %q: %v", line, err)
			}
			payloads = append(payloads, payload)
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return payloads
}

func indexOfEvent(payloads []map[string]any, want string) int {
	for i, payload := range payloads {
		if payload["type"] == want {
			return i
		}
	}
	return -1
}

func eventTypes(payloads []map[string]any) []string {
	types := make([]string, 0, len(payloads))
	for _, payload := range payloads {
		if typ, ok := payload["type"].(string); ok {
			types = append(types, typ)
		}
	}
	return types
}

func assertHasEvent(t *testing.T, events []string, want string) {
	t.Helper()
	for _, event := range events {
		if event == want {
			return
		}
	}
	t.Fatalf("missing event %q in %+v", want, events)
}
