package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"uni-api/internal/anthropic"
	"uni-api/internal/convert"
	"uni-api/internal/openai"
	"uni-api/internal/state"
	"uni-api/internal/stream"
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

func New(cfg Config, store *state.Store, httpClient *http.Client) http.Handler {
	return &Handler{cfg: cfg, store: store, client: anthropic.NewClient(cfg.AnthropicBaseURL, cfg.AnthropicAPIKey, httpClient)}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/v1")
	if r.Method == http.MethodPost && path == "/responses" {
		h.createResponse(w, r)
		return
	}
	writeJSON(w, http.StatusNotImplemented, openai.NewErrorResponse("endpoint is not implemented by this proxy", "not_implemented", "not_implemented"))
}

func (h *Handler) createResponse(w http.ResponseWriter, r *http.Request) {
	var req openai.CreateResponseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, openai.NewErrorResponse("invalid JSON request body", "invalid_request_error", "invalid_json"))
		return
	}
	var previous []anthropic.MessageParam
	if req.PreviousResponseID != "" {
		record, ok := h.store.Get(req.PreviousResponseID)
		if !ok {
			writeJSON(w, http.StatusBadRequest, openai.NewErrorResponse("previous_response_id not found", "invalid_request_error", "previous_response_not_found"))
			return
		}
		previous = record.Transcript
	}
	msgReq, err := convert.CreateResponseToMessage(req, previous, h.cfg.AnthropicModel)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, openai.NewErrorResponse(err.Error(), "invalid_request_error", "invalid_input"))
		return
	}
	responseID := newResponseID()
	createdAt := time.Now().Unix()
	fullTranscript := append(append([]anthropic.MessageParam{}, previous...), msgReq.Messages[len(previous):]...)
	if req.WantsStream() {
		h.createResponseStream(w, r, msgReq, responseID, createdAt, fullTranscript)
		return
	}
	msg, err := h.client.CreateMessage(r.Context(), msgReq)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, openai.NewErrorResponse(err.Error(), "upstream_error", "upstream_error"))
		return
	}
	resp, assistantTranscript, err := convert.MessageToResponse(msg, responseID, createdAt)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, openai.NewErrorResponse(err.Error(), "upstream_error", "conversion_error"))
		return
	}
	fullTranscript = append(fullTranscript, assistantTranscript...)
	h.store.Save(state.ResponseRecord{ID: responseID, Transcript: fullTranscript, Status: resp.Status, CreatedAt: createdAt})
	writeJSON(w, http.StatusOK, resp)
}

func (h *Handler) createResponseStream(w http.ResponseWriter, r *http.Request, msgReq anthropic.CreateMessageRequest, responseID string, createdAt int64, fullTranscript []anthropic.MessageParam) {
	body, err := h.client.CreateMessageStream(r.Context(), msgReq)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, openai.NewErrorResponse(err.Error(), "upstream_error", "upstream_error"))
		return
	}
	defer body.Close()
	w.Header().Set("content-type", "text/event-stream")
	w.Header().Set("cache-control", "no-cache")
	w.WriteHeader(http.StatusOK)
	assistantTranscript, err := stream.BridgeWithResult(body, w, responseID, createdAt)
	if err != nil {
		_ = stream.WriteFailed(w, responseID, createdAt, err.Error())
	}
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
	if len(assistantTranscript.Content) > 0 {
		fullTranscript = append(fullTranscript, assistantTranscript)
	}
	h.store.Save(state.ResponseRecord{ID: responseID, Transcript: fullTranscript, Status: "completed", CreatedAt: createdAt})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("content-type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func newResponseID() string {
	return "resp_" + strings.ReplaceAll(fmt.Sprintf("%d", time.Now().UnixNano()), "-", "")
}
