package main

import (
	"bufio"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"responses-anthropic-proxy/internal/server"
	"responses-anthropic-proxy/internal/state"
)

func main() {
	loadProjectDotEnv()

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

func loadProjectDotEnv() {
	wd, err := os.Getwd()
	if err != nil {
		log.Printf("could not determine working directory for .env loading: %v", err)
		return
	}
	path, ok := findDotEnv(wd)
	if !ok {
		return
	}
	if err := loadDotEnv(path); err != nil {
		log.Printf("could not load %s: %v", path, err)
	}
}

func findDotEnv(start string) (string, bool) {
	dir := filepath.Clean(start)
	for {
		path := filepath.Join(dir, ".env")
		if info, err := os.Stat(path); err == nil && !info.IsDir() {
			return path, true
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

func loadDotEnv(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		if key == "" || os.Getenv(key) != "" {
			continue
		}
		value = strings.TrimSpace(value)
		value = strings.Trim(value, `"'`)
		if err := os.Setenv(key, value); err != nil {
			return err
		}
	}
	return scanner.Err()
}
