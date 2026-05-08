package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type Client struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

type APIError struct {
	StatusCode int
	Type       string
	Message    string
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("anthropic returned %d: %s", e.StatusCode, e.Body)
}

func NewClient(baseURL, apiKey string, httpClient *http.Client) *Client {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &Client{baseURL: strings.TrimRight(baseURL, "/"), apiKey: apiKey, http: httpClient}
}

func (c *Client) CreateMessage(ctx context.Context, req CreateMessageRequest) (MessageResponse, error) {
	var out MessageResponse
	body, err := json.Marshal(req)
	if err != nil {
		return out, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return out, err
	}
	c.setHeaders(httpReq)
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return out, parseAPIError(resp.StatusCode, b)
	}
	return out, json.NewDecoder(resp.Body).Decode(&out)
}

func (c *Client) CreateMessageStream(ctx context.Context, req CreateMessageRequest) (io.ReadCloser, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	c.setHeaders(httpReq)
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		return nil, parseAPIError(resp.StatusCode, b)
	}
	return resp.Body, nil
}

func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("content-type", "application/json")
	req.Header.Set("accept", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
}

func parseAPIError(status int, body []byte) error {
	apiErr := &APIError{StatusCode: status, Body: string(body)}
	var parsed ErrorResponse
	if err := json.Unmarshal(body, &parsed); err == nil {
		apiErr.Type = parsed.Error.Type
		apiErr.Message = parsed.Error.Message
	}
	if apiErr.Message == "" {
		apiErr.Message = string(body)
	}
	return apiErr
}
