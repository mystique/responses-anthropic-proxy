package server

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"rap/internal/anthropic"
	"rap/internal/convert"
	"rap/internal/openai"
	"rap/internal/state"
	"rap/internal/stream"
)

type Config struct {
	AnthropicAPIKey  string
	ServiceAPIKey    string
	AnthropicModel   string
	AnthropicBaseURL string
	ListenAddr       string
	ModelMap         map[string]string
	ConfigPath       string
	ConfigPassword   string
}

type Handler struct {
	cfg            Config
	store          *state.Store
	client         *anthropic.Client
	configSessions map[string]struct{}
	configMu       sync.Mutex
}

type runtimeConfigFile struct {
	Upstream struct {
		BaseURL string `json:"base_url"`
		APIKey  string `json:"api_key"`
	} `json:"upstream"`
	Service struct {
		APIKey     string `json:"api_key"`
		ListenAddr string `json:"listen_addr"`
	} `json:"service"`
	Models         map[string]string `json:"models,omitempty"`
	ConfigPassword string            `json:"config_password,omitempty"`
	DefaultModel   string            `json:"default_model"`
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
	return &Handler{
		cfg:            cfg,
		store:          store,
		client:         anthropic.NewClient(cfg.AnthropicBaseURL, cfg.AnthropicAPIKey, httpClient),
		configSessions: map[string]struct{}{},
	}
}

func (h *Handler) authorized(header string) bool {
	const bearerPrefix = "Bearer "
	key := h.cfg.ServiceAPIKey
	if key == "" {
		key = h.cfg.AnthropicAPIKey
	}
	if strings.HasPrefix(header, bearerPrefix) {
		return strings.TrimPrefix(header, bearerPrefix) == key
	}
	return header == key
}

func (h *Handler) resolveModel(requested string) (string, bool) {
	if len(h.cfg.ModelMap) == 0 {
		return h.cfg.AnthropicModel, true
	}
	if requested == "" {
		if h.cfg.AnthropicModel != "" {
			return h.cfg.AnthropicModel, true
		}
		return "", false
	}
	model, ok := h.cfg.ModelMap[requested]
	return model, ok
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if strings.HasPrefix(r.URL.Path, "/config") {
		h.handleConfig(w, r)
		return
	}
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

func (h *Handler) handleConfig(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/config", "/config/":
		if r.Method != http.MethodGet {
			writeMethodNotAllowed(w)
			return
		}
		if !h.configAuthorized(r) {
			h.renderConfigLogin(w, http.StatusOK, "")
			return
		}
		h.renderConfigPage(w)
	case "/config/login":
		if r.Method != http.MethodPost {
			writeMethodNotAllowed(w)
			return
		}
		h.handleConfigLogin(w, r)
	case "/config/logout":
		if r.Method != http.MethodPost {
			writeMethodNotAllowed(w)
			return
		}
		h.handleConfigLogout(w, r)
	case "/config/api":
		if !h.configAuthorized(r) {
			writeJSON(w, http.StatusUnauthorized, openai.NewErrorResponse("config login required", "invalid_request_error", "config_login_required"))
			return
		}
		h.handleConfigAPI(w, r)
	default:
		writeJSON(w, http.StatusNotFound, openai.NewErrorResponse("endpoint not found", "invalid_request_error", "not_found"))
	}
}

func (h *Handler) configAuthorized(r *http.Request) bool {
	if h.cfg.ConfigPassword == "" {
		return true
	}
	cookie, err := r.Cookie("rap_config_session")
	if err != nil {
		return false
	}
	h.configMu.Lock()
	defer h.configMu.Unlock()
	_, ok := h.configSessions[cookie.Value]
	return ok
}

func (h *Handler) handleConfigLogin(w http.ResponseWriter, r *http.Request) {
	if h.cfg.ConfigPassword == "" {
		http.Redirect(w, r, "/config", http.StatusSeeOther)
		return
	}
	if err := r.ParseForm(); err != nil {
		h.renderConfigLogin(w, http.StatusBadRequest, "Unable to read password.")
		return
	}
	if subtle.ConstantTimeCompare([]byte(r.Form.Get("password")), []byte(h.cfg.ConfigPassword)) != 1 {
		h.renderConfigLogin(w, http.StatusUnauthorized, "Incorrect password.")
		return
	}
	token, err := newConfigSessionToken()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, openai.NewErrorResponse("could not create config session", "server_error", "config_session_error"))
		return
	}
	h.configMu.Lock()
	h.configSessions[token] = struct{}{}
	h.configMu.Unlock()
	http.SetCookie(w, &http.Cookie{
		Name:     "rap_config_session",
		Value:    token,
		Path:     "/config",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
	})
	http.Redirect(w, r, "/config", http.StatusSeeOther)
}

func (h *Handler) handleConfigLogout(w http.ResponseWriter, r *http.Request) {
	if cookie, err := r.Cookie("rap_config_session"); err == nil {
		h.configMu.Lock()
		delete(h.configSessions, cookie.Value)
		h.configMu.Unlock()
	}
	http.SetCookie(w, &http.Cookie{Name: "rap_config_session", Value: "", Path: "/config", MaxAge: -1, HttpOnly: true, SameSite: http.SameSiteLaxMode})
	http.Redirect(w, r, "/config", http.StatusSeeOther)
}

func (h *Handler) handleConfigAPI(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		cfg, err := h.loadEditableConfig()
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, openai.NewErrorResponse(err.Error(), "server_error", "config_read_error"))
			return
		}
		cfg.ConfigPassword = ""
		writeJSON(w, http.StatusOK, cfg)
	case http.MethodPost:
		var next runtimeConfigFile
		if err := json.NewDecoder(r.Body).Decode(&next); err != nil {
			writeJSON(w, http.StatusBadRequest, openai.NewErrorResponse("invalid JSON request body", "invalid_request_error", "invalid_json"))
			return
		}
		current, err := h.loadEditableConfig()
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, openai.NewErrorResponse(err.Error(), "server_error", "config_read_error"))
			return
		}
		next.ConfigPassword = current.ConfigPassword
		if err := h.saveEditableConfig(next); err != nil {
			writeJSON(w, http.StatusInternalServerError, openai.NewErrorResponse(err.Error(), "server_error", "config_write_error"))
			return
		}
		next.ConfigPassword = ""
		writeJSON(w, http.StatusOK, next)
	default:
		writeMethodNotAllowed(w)
	}
}

func (h *Handler) createResponse(w http.ResponseWriter, r *http.Request) {
	if !h.authorized(r.Header.Get("authorization")) {
		writeJSON(w, http.StatusUnauthorized, openai.NewErrorResponse("invalid API key", "invalid_request_error", "invalid_api_key"))
		return
	}
	var req openai.CreateResponseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, openai.NewErrorResponse("invalid JSON request body", "invalid_request_error", "invalid_json"))
		return
	}
	model, ok := h.resolveModel(req.Model)
	if !ok {
		h.writeLocalError(w, http.StatusBadRequest, "unsupported model", "invalid_request_error", "unsupported_model", localErrorContext{
			RequestPath:   r.URL.Path,
			RequestMethod: r.Method,
			OpenAIRequest: req,
			StoreSnapshot: h.store.Snapshot(),
		})
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
	msgReq, err := convert.CreateResponseToMessageWithContext(req, previous, model, convert.Context{
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
		h.createResponseStream(w, r, req, msgReq, responseID, createdAt, fullTranscript, previous, toolCallIDs, resolvedResponseID, resolvedToolCalls, model)
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
	resp, assistantTranscript, err := convert.MessageToResponseWithOptions(msg, responseID, createdAt, convert.ResponseOptions{Include: req.Include})
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

func (h *Handler) createResponseStream(w http.ResponseWriter, r *http.Request, req openai.CreateResponseRequest, msgReq anthropic.CreateMessageRequest, responseID string, createdAt int64, fullTranscript []anthropic.MessageParam, previous []anthropic.MessageParam, toolCallIDs []string, resolvedResponseID string, resolvedToolCalls map[string]state.ToolCallRecord, model string) {
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
	resp := openai.NewBaseResponse(responseID, model, status, createdAt)
	if len(assistantTranscript.Content) > 0 {
		fullTranscript = append(fullTranscript, assistantTranscript)
		msg := anthropic.MessageResponse{
			Model:      model,
			Role:       "assistant",
			Content:    assistantTranscript.Content,
			StopReason: "end_turn",
		}
		converted, _, convErr := convert.MessageToResponseWithOptions(msg, responseID, createdAt, convert.ResponseOptions{Include: req.Include})
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
		if item.Type != "function_call_output" && item.Type != "computer_call_output" {
			continue
		}
		if item.CallID == "" {
			return nil, fmt.Errorf("%s missing call_id", item.Type)
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
		if (item.Type == "function_call" || item.Type == "computer_call") && item.CallID != "" {
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

func (h *Handler) loadEditableConfig() (runtimeConfigFile, error) {
	if h.cfg.ConfigPath == "" {
		return runtimeConfigFromServerConfig(h.cfg), nil
	}
	file, err := os.Open(h.cfg.ConfigPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return runtimeConfigFromServerConfig(h.cfg), nil
		}
		return runtimeConfigFile{}, err
	}
	defer file.Close()
	var cfg runtimeConfigFile
	if err := json.NewDecoder(file).Decode(&cfg); err != nil {
		return runtimeConfigFile{}, err
	}
	return cfg, nil
}

func (h *Handler) saveEditableConfig(cfg runtimeConfigFile) error {
	if h.cfg.ConfigPath == "" {
		return errors.New("RAP_CONFIG or rap.config.json is required before saving configuration")
	}
	body, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	body = append(body, '\n')
	return os.WriteFile(h.cfg.ConfigPath, body, 0o600)
}

func runtimeConfigFromServerConfig(cfg Config) runtimeConfigFile {
	fileCfg := runtimeConfigFile{
		Models:         cfg.ModelMap,
		DefaultModel:   cfg.AnthropicModel,
		ConfigPassword: cfg.ConfigPassword,
	}
	fileCfg.Upstream.BaseURL = cfg.AnthropicBaseURL
	fileCfg.Upstream.APIKey = cfg.AnthropicAPIKey
	fileCfg.Service.APIKey = cfg.ServiceAPIKey
	fileCfg.Service.ListenAddr = cfg.ListenAddr
	return fileCfg
}

func newConfigSessionToken() (string, error) {
	var b [32]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(b[:]), nil
}

func (h *Handler) renderConfigLogin(w http.ResponseWriter, status int, message string) {
	w.Header().Set("content-type", "text/html; charset=utf-8")
	w.WriteHeader(status)
	_ = configLoginTemplate.Execute(w, struct {
		Message string
	}{Message: message})
}

func (h *Handler) renderConfigPage(w http.ResponseWriter) {
	w.Header().Set("content-type", "text/html; charset=utf-8")
	_ = configPageTemplate.Execute(w, struct {
		HasConfigFile bool
		ConfigPath    string
	}{
		HasConfigFile: h.cfg.ConfigPath != "",
		ConfigPath:    h.cfg.ConfigPath,
	})
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

var configLoginTemplate = template.Must(template.New("config-login").Parse(`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rap configuration</title>
<style>
:root{color-scheme:light dark;--bg:#f7f7f5;--panel:#fff;--text:#111;--muted:#6b6b6b;--line:#d8d8d0;--accent:#10a37f;--danger:#b42318}
@media (prefers-color-scheme:dark){:root{--bg:#171717;--panel:#202020;--text:#f4f4f4;--muted:#a3a3a3;--line:#3a3a3a}}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{min-height:100vh;display:grid;place-items:center;padding:24px}
.card{width:min(420px,100%);background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:28px;box-shadow:0 24px 60px rgba(0,0,0,.08)}
h1{font-size:22px;line-height:1.2;margin:0 0 8px;font-weight:650;letter-spacing:0}
p{margin:0 0 22px;color:var(--muted)}
label{display:block;font-weight:550;margin-bottom:8px}
input{box-sizing:border-box;width:100%;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--text);padding:10px 12px;font:inherit}
button{margin-top:16px;border:0;border-radius:6px;background:var(--accent);color:white;padding:10px 14px;font:inherit;font-weight:600;cursor:pointer}
.error{color:var(--danger);margin-bottom:14px}
</style>
</head>
<body>
<main>
<form class="card" method="post" action="/config/login">
<h1>rap configuration</h1>
<p>Enter the configured password to manage local proxy settings.</p>
{{if .Message}}<div class="error">{{.Message}}</div>{{end}}
<label for="password">Password</label>
<input id="password" name="password" type="password" autocomplete="current-password" autofocus>
<button type="submit">Continue</button>
</form>
</main>
</body>
</html>`))

var configPageTemplate = template.Must(template.New("config-page").Parse(`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rap configuration</title>
<style>
:root{color-scheme:light dark;--bg:#f7f7f5;--panel:#fff;--text:#111;--muted:#6b6b6b;--line:#d8d8d0;--soft:#efefeb;--accent:#10a37f;--accent-press:#0d8f70;--danger:#b42318}
@media (prefers-color-scheme:dark){:root{--bg:#171717;--panel:#202020;--text:#f4f4f4;--muted:#a3a3a3;--line:#3a3a3a;--soft:#292929}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{border-bottom:1px solid var(--line);background:color-mix(in srgb,var(--panel) 84%,transparent);position:sticky;top:0;backdrop-filter:blur(14px)}
.bar{max-width:1120px;margin:0 auto;padding:14px 24px;display:flex;align-items:center;justify-content:space-between;gap:16px}
.brand{font-weight:650;letter-spacing:0}
.path{color:var(--muted);font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
main{max-width:1120px;margin:0 auto;padding:30px 24px 48px}
.intro{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:22px}
h1{font-size:28px;line-height:1.15;margin:0 0 8px;font-weight:650;letter-spacing:0}
p{margin:0;color:var(--muted)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.section{padding:22px;border-top:1px solid var(--line)}
.section:first-child{border-top:0}
h2{font-size:15px;margin:0 0 16px;font-weight:650;letter-spacing:0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
label{display:block;font-size:13px;font-weight:550;margin-bottom:7px}
input{width:100%;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--text);padding:10px 12px;font:inherit}
.actions{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-top:18px}
.buttons{display:flex;gap:10px;align-items:center}
button{border:1px solid var(--line);border-radius:6px;background:var(--soft);color:var(--text);padding:9px 13px;font:inherit;font-weight:600;cursor:pointer}
button.primary{border-color:var(--accent);background:var(--accent);color:white}
button.primary:active{background:var(--accent-press)}
button.danger{border-color:transparent;background:transparent;color:var(--danger);font-size:22px;line-height:1;padding:7px 8px}
.status{color:var(--muted);font-size:13px}
.status.error{color:var(--danger)}
.note{border:1px solid var(--line);border-radius:8px;padding:12px 14px;background:var(--soft);color:var(--muted);font-size:13px;margin-bottom:18px}
.mapping-list{display:grid;gap:12px}
.mapping-row{display:grid;grid-template-columns:minmax(0,1fr) 28px minmax(0,1fr) 40px;gap:12px;align-items:center}
.arrow{color:#9aa1ad;text-align:center;font-size:24px}
.add-mapping{width:100%;border:2px dashed #cfd5df;background:transparent;color:#4b5563;padding:12px}
@media (max-width:760px){.grid,.intro{display:block}.grid{display:grid;grid-template-columns:1fr}.buttons{width:100%;justify-content:flex-start}.actions{align-items:flex-start;flex-direction:column}.mapping-row{grid-template-columns:1fr}.arrow{text-align:left}.danger{justify-self:start}}
</style>
</head>
<body>
<header>
<div class="bar">
<div class="brand">rap configuration</div>
<div class="path">{{if .HasConfigFile}}{{.ConfigPath}}{{else}}No config file loaded{{end}}</div>
</div>
</header>
<main>
<div class="intro">
<div>
<h1>Settings</h1>
<p>Changes are written to rap.config.json and take effect after restarting the service.</p>
</div>
<form method="post" action="/config/logout"><button type="submit">Sign out</button></form>
</div>
{{if not .HasConfigFile}}<div class="note">Saving requires RAP_CONFIG or an existing rap.config.json discovered at startup.</div>{{end}}
<form id="config-form" class="panel">
<div class="section">
<h2>Upstream</h2>
<div class="grid">
<div><label for="upstream-base-url">Anthropic base URL</label><input id="upstream-base-url" name="upstream.base_url" autocomplete="off"></div>
<div><label for="upstream-api-key">Anthropic API key</label><input id="upstream-api-key" name="upstream.api_key" autocomplete="off"></div>
</div>
</div>
<div class="section">
<h2>Service</h2>
<div class="grid">
<div><label for="service-api-key">Client API key</label><input id="service-api-key" name="service.api_key" autocomplete="off"></div>
<div><label for="listen-addr">Listen address</label><input id="listen-addr" name="service.listen_addr" autocomplete="off"></div>
</div>
</div>
<div class="section">
<h2>Models</h2>
<div><label for="default-model">Default Anthropic model</label><input id="default-model" name="default_model" autocomplete="off"></div>
<div style="height:16px"></div>
<div id="model-mappings" class="mapping-list" aria-label="OpenAI to Anthropic model mappings"></div>
<button type="button" id="add-model-mapping" class="add-mapping">+ Add mapping</button>
<div class="actions">
<div id="status" class="status">Loading configuration...</div>
<div class="buttons"><button type="button" id="reload">Reload</button><button class="primary" type="submit">Save changes</button></div>
</div>
</div>
</form>
</main>
<script>
const fields = {
  baseURL: document.querySelector('[name="upstream.base_url"]'),
  upstreamKey: document.querySelector('[name="upstream.api_key"]'),
  serviceKey: document.querySelector('[name="service.api_key"]'),
  listenAddr: document.querySelector('[name="service.listen_addr"]'),
  defaultModel: document.querySelector('[name="default_model"]')
};
const mappingsEl = document.getElementById('model-mappings');
const statusEl = document.getElementById('status');
function setStatus(text, error=false){ statusEl.textContent = text; statusEl.className = error ? 'status error' : 'status'; }
function addMappingRow(openaiModel='', anthropicModel=''){
  const row = document.createElement('div');
  row.className = 'mapping-row';
  row.innerHTML = '<input name="model.openai" autocomplete="off" placeholder="gpt-5.5"><div class="arrow">→</div><input name="model.anthropic" autocomplete="off" placeholder="claude-sonnet-4-6"><button type="button" class="danger" aria-label="Remove mapping">⌫</button>';
  row.querySelector('[name="model.openai"]').value = openaiModel;
  row.querySelector('[name="model.anthropic"]').value = anthropicModel;
  row.querySelector('button').addEventListener('click', () => row.remove());
  mappingsEl.appendChild(row);
}
function fillMappings(models){
  mappingsEl.replaceChildren();
  const entries = Object.entries(models || {});
  if(entries.length === 0){ addMappingRow(); return; }
  entries.forEach(([openaiModel, anthropicModel]) => addMappingRow(openaiModel, anthropicModel));
}
function readMappings(){
  const models = {};
  for(const row of mappingsEl.querySelectorAll('.mapping-row')){
    const openaiModel = row.querySelector('[name="model.openai"]').value.trim();
    const anthropicModel = row.querySelector('[name="model.anthropic"]').value.trim();
    if(openaiModel || anthropicModel){
      if(!openaiModel || !anthropicModel){ throw new Error('Each model mapping needs both sides.'); }
      models[openaiModel] = anthropicModel;
    }
  }
  return models;
}
function fill(cfg){
  fields.baseURL.value = cfg.upstream?.base_url || '';
  fields.upstreamKey.value = cfg.upstream?.api_key || '';
  fields.serviceKey.value = cfg.service?.api_key || '';
  fields.listenAddr.value = cfg.service?.listen_addr || '';
  fields.defaultModel.value = cfg.default_model || '';
  fillMappings(cfg.models || {});
}
async function loadConfig(){
  setStatus('Loading configuration...');
  const res = await fetch('/config/api');
  if(!res.ok){ setStatus('Unable to load configuration.', true); return; }
  fill(await res.json());
  setStatus('Loaded. Save changes, then restart rap.');
}
document.getElementById('reload').addEventListener('click', loadConfig);
document.getElementById('add-model-mapping').addEventListener('click', () => addMappingRow());
document.getElementById('config-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  let models;
  try { models = readMappings(); }
  catch (error) { setStatus(error.message, true); return; }
  const payload = {
    upstream: { base_url: fields.baseURL.value.trim(), api_key: fields.upstreamKey.value.trim() },
    service: { api_key: fields.serviceKey.value.trim(), listen_addr: fields.listenAddr.value.trim() },
    models,
    default_model: fields.defaultModel.value.trim()
  };
  const res = await fetch('/config/api', { method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(payload) });
  if(!res.ok){ setStatus('Save failed: ' + await res.text(), true); return; }
  fill(await res.json());
  setStatus('Saved. Restart rap for changes to take effect.');
});
loadConfig();
</script>
</body>
</html>`))
