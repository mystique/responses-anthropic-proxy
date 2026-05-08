package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"responses-anthropic-proxy/internal/anthropic"
	"responses-anthropic-proxy/internal/convert"
	"responses-anthropic-proxy/internal/openai"
	"responses-anthropic-proxy/internal/state"
	"responses-anthropic-proxy/internal/stream"
)

type Config struct {
	AnthropicAPIKey  string
	AnthropicModel   string
	AnthropicBaseURL string
}

type Handler struct {
	cfg    Config
	store  *state.Store
	client *anthropic.Client
}

type upstreamErrorContext struct {
	ResponseID         string                          `json:"response_id"`
	CreatedAt          int64                           `json:"created_at"`
	RequestPath        string                          `json:"request_path"`
	RequestMethod      string                          `json:"request_method"`
	OpenAIRequest      openai.CreateResponseRequest    `json:"openai_request"`
	AnthropicRequest   anthropic.CreateMessageRequest  `json:"anthropic_request"`
	PreviousResponseID string                          `json:"previous_response_id,omitempty"`
	ResolvedResponseID string                          `json:"resolved_response_id,omitempty"`
	ToolCallIDs        []string                        `json:"tool_call_ids,omitempty"`
	ResolvedToolCalls  map[string]state.ToolCallRecord `json:"resolved_tool_calls,omitempty"`
	PreviousMessages   int                             `json:"previous_messages"`
	NewMessages        int                             `json:"new_messages"`
	Stream             bool                            `json:"stream"`
}

type localErrorContext struct {
	RequestPath        string                          `json:"request_path"`
	RequestMethod      string                          `json:"request_method"`
	OpenAIRequest      openai.CreateResponseRequest    `json:"openai_request"`
	LocalResponse      openai.ErrorResponse            `json:"local_response"`
	PreviousResponseID string                          `json:"previous_response_id,omitempty"`
	ResolvedResponseID string                          `json:"resolved_response_id,omitempty"`
	ToolCallIDs        []string                        `json:"tool_call_ids,omitempty"`
	ResolvedToolCalls  map[string]state.ToolCallRecord `json:"resolved_tool_calls,omitempty"`
	PreviousMessages   int                             `json:"previous_messages"`
	StoreSnapshot      state.Snapshot                  `json:"store_snapshot"`
}

var forwardedMetadataHeaders = map[string]struct{}{
	"user-agent":        {},
	"x-forwarded-for":   {},
	"x-forwarded-host":  {},
	"x-forwarded-port":  {},
	"x-forwarded-proto": {},
	"x-real-ip":         {},
	"x-request-id":      {},
	"x-correlation-id":  {},
	"cf-connecting-ip":  {},
	"cf-ray":            {},
	"true-client-ip":    {},
	"forwarded":         {},
	"traceparent":       {},
	"tracestate":        {},
	"baggage":           {},
}

func New(cfg Config, store *state.Store, httpClient *http.Client) http.Handler {
	return &Handler{cfg: cfg, store: store, client: anthropic.NewClient(cfg.AnthropicBaseURL, cfg.AnthropicAPIKey, httpClient)}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/v1")
	if path == "/responses" {
		if r.Method == http.MethodPost {
			h.createResponse(w, r)
			return
		}
		writeMethodNotAllowed(w)
		return
	}
	if strings.HasPrefix(path, "/responses/") {
		h.handleStoredResponse(w, r, strings.TrimPrefix(path, "/responses/"))
		return
	}
	writeJSON(w, http.StatusNotFound, openai.NewErrorResponse("endpoint not found", "invalid_request_error", "not_found"))
}

func (h *Handler) createResponse(w http.ResponseWriter, r *http.Request) {
	var req openai.CreateResponseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, openai.NewErrorResponse("invalid JSON request body", "invalid_request_error", "invalid_json"))
		return
	}
	var previous []anthropic.MessageParam
	resolvedResponseID := req.PreviousResponseID
	toolCallIDs, err := functionCallOutputIDs(req.Input)
	if err != nil {
		h.writeLocalError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "invalid_input", localErrorContext{
			RequestPath:   r.URL.Path,
			RequestMethod: r.Method,
			OpenAIRequest: req,
			StoreSnapshot: h.store.Snapshot(),
		})
		return
	}
	inlineToolCallIDs := inlineFunctionCallIDs(req.Input)
	if req.PreviousResponseID != "" {
		record, ok := h.store.Get(req.PreviousResponseID)
		if !ok {
			h.writeLocalError(w, http.StatusBadRequest, "previous_response_id not found", "invalid_request_error", "previous_response_not_found", localErrorContext{
				RequestPath:        r.URL.Path,
				RequestMethod:      r.Method,
				OpenAIRequest:      req,
				PreviousResponseID: req.PreviousResponseID,
				ToolCallIDs:        toolCallIDs,
				StoreSnapshot:      h.store.Snapshot(),
			})
			return
		}
		previous = record.Transcript
	} else if len(toolCallIDs) > 0 && !allToolCallsInline(toolCallIDs, inlineToolCallIDs) {
		record, code, ok := h.restorePreviousFromToolCalls(toolCallIDs)
		if !ok {
			h.writeLocalError(w, http.StatusBadRequest, toolCallErrorMessage(code), "invalid_request_error", code, localErrorContext{
				RequestPath:   r.URL.Path,
				RequestMethod: r.Method,
				OpenAIRequest: req,
				ToolCallIDs:   toolCallIDs,
				StoreSnapshot: h.store.Snapshot(),
			})
			return
		}
		resolvedResponseID = record.ID
		previous = record.Transcript
	}
	resolvedToolCalls := map[string]state.ToolCallRecord{}
	for _, callID := range toolCallIDs {
		if inlineToolCallIDs[callID] {
			resolvedToolCalls[callID] = state.ToolCallRecord{
				OpenAICallID:       callID,
				AnthropicToolUseID: callID,
				ResponseID:         resolvedResponseID,
			}
			continue
		}
		record, ok := h.store.FindToolCall(resolvedResponseID, callID)
		if !ok {
			h.writeLocalError(w, http.StatusBadRequest, "tool call not found", "invalid_request_error", "tool_call_not_found", localErrorContext{
				RequestPath:        r.URL.Path,
				RequestMethod:      r.Method,
				OpenAIRequest:      req,
				PreviousResponseID: req.PreviousResponseID,
				ResolvedResponseID: resolvedResponseID,
				ToolCallIDs:        toolCallIDs,
				ResolvedToolCalls:  resolvedToolCalls,
				PreviousMessages:   len(previous),
				StoreSnapshot:      h.store.Snapshot(),
			})
			return
		}
		if record.ResolvedAt != 0 {
			h.writeLocalError(w, http.StatusBadRequest, "tool call already resolved", "invalid_request_error", "tool_call_already_resolved", localErrorContext{
				RequestPath:        r.URL.Path,
				RequestMethod:      r.Method,
				OpenAIRequest:      req,
				PreviousResponseID: req.PreviousResponseID,
				ResolvedResponseID: resolvedResponseID,
				ToolCallIDs:        toolCallIDs,
				ResolvedToolCalls:  resolvedToolCalls,
				PreviousMessages:   len(previous),
				StoreSnapshot:      h.store.Snapshot(),
			})
			return
		}
		resolvedToolCalls[callID] = record
	}
	msgReq, err := convert.CreateResponseToMessageWithContext(req, previous, h.cfg.AnthropicModel, convert.Context{
		ToolResolver: func(callID string) (string, bool) {
			record, ok := resolvedToolCalls[callID]
			if !ok {
				return "", false
			}
			return record.AnthropicToolUseID, true
		},
	})
	if err != nil {
		code := "invalid_input"
		var inputErr *convert.InputError
		if errors.As(err, &inputErr) && inputErr.Code != "" {
			code = inputErr.Code
		}
		h.writeLocalError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", code, localErrorContext{
			RequestPath:        r.URL.Path,
			RequestMethod:      r.Method,
			OpenAIRequest:      req,
			PreviousResponseID: req.PreviousResponseID,
			ResolvedResponseID: resolvedResponseID,
			ToolCallIDs:        toolCallIDs,
			ResolvedToolCalls:  resolvedToolCalls,
			PreviousMessages:   len(previous),
			StoreSnapshot:      h.store.Snapshot(),
		})
		return
	}
	responseID := newResponseID()
	createdAt := time.Now().Unix()
	fullTranscript := append(append([]anthropic.MessageParam{}, previous...), msgReq.Messages[len(previous):]...)
	if req.WantsStream() {
		h.createResponseStream(w, r, req, msgReq, responseID, createdAt, fullTranscript, previous, toolCallIDs, resolvedResponseID, resolvedToolCalls)
		return
	}
	metadataHeaders := upstreamMetadataHeaders(r.Header)
	msg, err := h.client.CreateMessage(r.Context(), msgReq, metadataHeaders)
	if err != nil {
		h.logUpstreamError(err, upstreamErrorContext{
			ResponseID:         responseID,
			CreatedAt:          createdAt,
			RequestPath:        r.URL.Path,
			RequestMethod:      r.Method,
			OpenAIRequest:      req,
			AnthropicRequest:   msgReq,
			PreviousResponseID: req.PreviousResponseID,
			ResolvedResponseID: resolvedResponseID,
			ToolCallIDs:        toolCallIDs,
			ResolvedToolCalls:  resolvedToolCalls,
			PreviousMessages:   len(previous),
			NewMessages:        len(msgReq.Messages) - len(previous),
			Stream:             false,
		})
		writeUpstreamError(w, err)
		return
	}
	resp, assistantTranscript, err := convert.MessageToResponse(msg, responseID, createdAt)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, openai.NewErrorResponse(err.Error(), "upstream_error", "conversion_error"))
		return
	}
	fullTranscript = append(fullTranscript, assistantTranscript...)
	for callID := range resolvedToolCalls {
		h.store.MarkToolCallResolved(resolvedResponseID, callID, createdAt)
	}
	h.store.Save(state.ResponseRecord{ID: responseID, Response: resp, Transcript: fullTranscript, Status: resp.Status, CreatedAt: createdAt})
	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) handleStoredResponse(w http.ResponseWriter, r *http.Request, suffix string) {
	if strings.HasSuffix(suffix, "/cancel") {
		id := strings.TrimSuffix(suffix, "/cancel")
		if id == "" || strings.Contains(id, "/") {
			writeJSON(w, http.StatusNotFound, openai.NewErrorResponse("endpoint not found", "invalid_request_error", "not_found"))
			return
		}
		if r.Method != http.MethodPost {
			writeMethodNotAllowed(w)
			return
		}
		h.cancelStoredResponse(w, id)
		return
	}
	if suffix == "" || strings.Contains(suffix, "/") {
		writeJSON(w, http.StatusNotFound, openai.NewErrorResponse("endpoint not found", "invalid_request_error", "not_found"))
		return
	}
	switch r.Method {
	case http.MethodGet:
		record, ok := h.store.Get(suffix)
		if !ok {
			writeJSON(w, http.StatusNotFound, openai.NewErrorResponse("response not found", "invalid_request_error", "response_not_found"))
			return
		}
		writeJSON(w, http.StatusOK, record.Response)
	case http.MethodDelete:
		h.deleteStoredResponse(w, suffix)
	default:
		writeMethodNotAllowed(w)
	}
}

func (h *Handler) deleteStoredResponse(w http.ResponseWriter, id string) {
	if !h.store.Delete(id) {
		writeJSON(w, http.StatusNotFound, openai.NewErrorResponse("response not found", "invalid_request_error", "response_not_found"))
		return
	}
	writeJSON(w, http.StatusOK, openai.DeleteResponse{ID: id, Object: "response.deleted", Deleted: true})
}

func (h *Handler) cancelStoredResponse(w http.ResponseWriter, id string) {
	record, ok := h.store.Get(id)
	if !ok {
		writeJSON(w, http.StatusNotFound, openai.NewErrorResponse("response not found", "invalid_request_error", "response_not_found"))
		return
	}
	if record.Response.Status != "queued" && record.Response.Status != "in_progress" {
		writeJSON(w, http.StatusConflict, openai.NewErrorResponse("response is not cancellable", "invalid_request_error", "response_not_cancellable"))
		return
	}
	updated, ok := h.store.UpdateStatus(id, "cancelled")
	if !ok {
		writeJSON(w, http.StatusNotFound, openai.NewErrorResponse("response not found", "invalid_request_error", "response_not_found"))
		return
	}
	writeJSON(w, http.StatusOK, updated.Response)
}

func (h *Handler) createResponseStream(w http.ResponseWriter, r *http.Request, req openai.CreateResponseRequest, msgReq anthropic.CreateMessageRequest, responseID string, createdAt int64, fullTranscript []anthropic.MessageParam, previous []anthropic.MessageParam, toolCallIDs []string, resolvedResponseID string, resolvedToolCalls map[string]state.ToolCallRecord) {
	body, err := h.client.CreateMessageStream(r.Context(), msgReq, upstreamMetadataHeaders(r.Header))
	if err != nil {
		h.logUpstreamError(err, upstreamErrorContext{
			ResponseID:         responseID,
			CreatedAt:          createdAt,
			RequestPath:        r.URL.Path,
			RequestMethod:      r.Method,
			OpenAIRequest:      req,
			AnthropicRequest:   msgReq,
			PreviousResponseID: req.PreviousResponseID,
			ResolvedResponseID: resolvedResponseID,
			ToolCallIDs:        toolCallIDs,
			ResolvedToolCalls:  resolvedToolCalls,
			PreviousMessages:   len(previous),
			NewMessages:        len(msgReq.Messages) - len(previous),
			Stream:             true,
		})
		writeUpstreamError(w, err)
		return
	}
	defer body.Close()
	w.Header().Set("content-type", "text/event-stream")
	w.Header().Set("cache-control", "no-cache")
	w.WriteHeader(http.StatusOK)
	assistantTranscript, err := stream.BridgeWithResult(body, w, responseID, createdAt)
	status := "completed"
	if err != nil {
		status = "failed"
		_ = stream.WriteFailed(w, responseID, createdAt, err.Error())
	}
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
	resp := openai.NewBaseResponse(responseID, h.cfg.AnthropicModel, status, createdAt)
	if len(assistantTranscript.Content) > 0 {
		fullTranscript = append(fullTranscript, assistantTranscript)
		msg := anthropic.MessageResponse{
			Model:      h.cfg.AnthropicModel,
			Role:       "assistant",
			Content:    assistantTranscript.Content,
			StopReason: "end_turn",
		}
		converted, _, convErr := convert.MessageToResponse(msg, responseID, createdAt)
		if convErr == nil {
			resp = converted
			resp.Status = status
		}
	}
	for callID := range resolvedToolCalls {
		h.store.MarkToolCallResolved(resolvedResponseID, callID, createdAt)
	}
	h.store.Save(state.ResponseRecord{ID: responseID, Response: resp, Transcript: fullTranscript, Status: resp.Status, CreatedAt: createdAt})
}

func upstreamMetadataHeaders(src http.Header) http.Header {
	dst := http.Header{}
	for name, values := range src {
		if _, ok := forwardedMetadataHeaders[strings.ToLower(name)]; !ok {
			continue
		}
		for _, value := range values {
			dst.Add(name, value)
		}
	}
	return dst
}

func (h *Handler) restorePreviousFromToolCalls(callIDs []string) (state.ResponseRecord, string, bool) {
	responseID := ""
	for _, callID := range callIDs {
		record, matches := h.store.FindToolCallByCallID(callID)
		switch matches {
		case 0:
			return state.ResponseRecord{}, "tool_call_not_found", false
		case 1:
			if responseID == "" {
				responseID = record.ResponseID
				continue
			}
			if responseID != record.ResponseID {
				return state.ResponseRecord{}, "ambiguous_tool_call_id", false
			}
		default:
			return state.ResponseRecord{}, "ambiguous_tool_call_id", false
		}
	}
	record, ok := h.store.Get(responseID)
	if !ok {
		return state.ResponseRecord{}, "tool_call_not_found", false
	}
	return record, "", true
}

func functionCallOutputIDs(raw openai.RawJSON) ([]string, error) {
	if raw.IsZero() {
		return nil, nil
	}
	var items []struct {
		Type   string `json:"type"`
		CallID string `json:"call_id"`
	}
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, nil
	}
	var ids []string
	seen := map[string]bool{}
	for _, item := range items {
		if item.Type != "function_call_output" {
			continue
		}
		if item.CallID == "" {
			return nil, fmt.Errorf("function_call_output missing call_id")
		}
		if !seen[item.CallID] {
			ids = append(ids, item.CallID)
			seen[item.CallID] = true
		}
	}
	return ids, nil
}

func inlineFunctionCallIDs(raw openai.RawJSON) map[string]bool {
	ids := map[string]bool{}
	if raw.IsZero() {
		return ids
	}
	var items []struct {
		Type   string `json:"type"`
		CallID string `json:"call_id"`
	}
	if err := json.Unmarshal(raw, &items); err != nil {
		return ids
	}
	for _, item := range items {
		if item.Type == "function_call" && item.CallID != "" {
			ids[item.CallID] = true
		}
	}
	return ids
}

func allToolCallsInline(callIDs []string, inline map[string]bool) bool {
	if len(callIDs) == 0 {
		return false
	}
	for _, callID := range callIDs {
		if !inline[callID] {
			return false
		}
	}
	return true
}

func toolCallErrorMessage(code string) string {
	switch code {
	case "ambiguous_tool_call_id":
		return "ambiguous tool call id; provide previous_response_id"
	case "tool_call_already_resolved":
		return "tool call already resolved"
	default:
		return "tool call not found"
	}
}

func (h *Handler) logUpstreamError(err error, ctx upstreamErrorContext) {
	payload := struct {
		Event           string `json:"event"`
		UpstreamStatus  int    `json:"upstream_status,omitempty"`
		UpstreamType    string `json:"upstream_type,omitempty"`
		UpstreamMessage string `json:"upstream_message,omitempty"`
		UpstreamBody    string `json:"upstream_body,omitempty"`
		Error           string `json:"error"`
		upstreamErrorContext
	}{
		Event:                "upstream_error_context",
		Error:                err.Error(),
		upstreamErrorContext: ctx,
	}
	var apiErr *anthropic.APIError
	if errors.As(err, &apiErr) {
		payload.UpstreamStatus = apiErr.StatusCode
		payload.UpstreamType = apiErr.Type
		payload.UpstreamMessage = apiErr.Message
		payload.UpstreamBody = apiErr.Body
	}
	b, marshalErr := json.Marshal(payload)
	if marshalErr != nil {
		log.Printf("upstream_error_context marshal_error=%q error=%q response_id=%q", marshalErr.Error(), err.Error(), ctx.ResponseID)
		return
	}
	log.Printf("%s", b)
}

func (h *Handler) writeLocalError(w http.ResponseWriter, status int, message, typ, code string, ctx localErrorContext) {
	resp := openai.NewErrorResponse(message, typ, code)
	ctx.LocalResponse = resp
	ctx.StoreSnapshot = h.store.Snapshot()
	h.logLocalError(ctx)
	writeJSON(w, status, resp)
}

func (h *Handler) logLocalError(ctx localErrorContext) {
	payload := struct {
		Event string `json:"event"`
		localErrorContext
	}{
		Event:             "local_compatibility_error_context",
		localErrorContext: ctx,
	}
	b, marshalErr := json.Marshal(payload)
	if marshalErr != nil {
		log.Printf("local_compatibility_error_context marshal_error=%q code=%q", marshalErr.Error(), ctx.LocalResponse.Error.Code)
		return
	}
	log.Printf("%s", b)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("content-type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeUpstreamError(w http.ResponseWriter, err error) {
	var apiErr *anthropic.APIError
	if errors.As(err, &apiErr) && apiErr.Type == "invalid_request_error" {
		writeJSON(w, http.StatusBadRequest, openai.NewErrorResponse(apiErr.Message, apiErr.Type, apiErr.Type))
		return
	}
	writeJSON(w, http.StatusBadGateway, openai.NewErrorResponse(err.Error(), "upstream_error", "upstream_error"))
}

func writeMethodNotAllowed(w http.ResponseWriter) {
	writeJSON(w, http.StatusMethodNotAllowed, openai.NewErrorResponse("method not allowed", "invalid_request_error", "method_not_allowed"))
}

func newResponseID() string {
	return "resp_" + strings.ReplaceAll(fmt.Sprintf("%d", time.Now().UnixNano()), "-", "")
}
