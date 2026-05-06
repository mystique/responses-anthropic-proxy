package main

import (
	"log"
	"net/http"
	"os"
	"time"

	"uni-api/internal/server"
	"uni-api/internal/state"
)

func main() {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		log.Fatal("ANTHROPIC_API_KEY is required")
	}
	model := envOrDefault("ANTHROPIC_MODEL", "claude-sonnet-4-6")
	baseURL := envOrDefault("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
	addr := envOrDefault("PROXY_ADDR", "127.0.0.1:8180")

	store := state.NewStore(24 * time.Hour)
	go func() {
		ticker := time.NewTicker(time.Hour)
		defer ticker.Stop()
		for now := range ticker.C {
			store.Cleanup(now)
		}
	}()

	handler := server.New(server.Config{
		AnthropicAPIKey:  apiKey,
		AnthropicModel:   model,
		AnthropicBaseURL: baseURL,
	}, store, http.DefaultClient)

	log.Printf("listening on http://%s", addr)
	if err := http.ListenAndServe(addr, handler); err != nil {
		log.Fatal(err)
	}
}

func envOrDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
