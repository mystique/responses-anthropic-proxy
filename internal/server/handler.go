package server

import (
	"crypto/rand"
	"crypto/subtle"
	"embed"
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

//go:embed templates/*.html
var configTemplatesFS embed.FS

var (
	configLoginTemplate = template.Must(template.ParseFS(configTemplatesFS, "templates/config_login.html"))
	configPageTemplate  = template.Must(template.ParseFS(configTemplatesFS, "templates/config_page.html"))
)
