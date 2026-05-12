package main

import (
	"bufio"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"rap/internal/server"
	"rap/internal/state"
)

func main() {
	loadProjectDotEnv()

	configPath, _ := findRuntimeConfig(".")
	cfg, err := loadRuntimeConfig(configPath)
	if err != nil {
		log.Fatal(err)
	}
	if cfg.AnthropicAPIKey == "" {
		log.Fatal("ANTHROPIC_API_KEY is required")
	}
	store := state.NewStore(24 * time.Hour)
	go func() {
		ticker := time.NewTicker(time.Hour)
		defer ticker.Stop()
		for now := range ticker.C {
			store.Cleanup(now)
		}
	}()

	handler := server.New(cfg, store, http.DefaultClient)

	log.Printf("listening on http://%s", cfg.ListenAddr)
	if err := http.ListenAndServe(cfg.ListenAddr, handler); err != nil {
		log.Fatal(err)
	}
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
	Models         map[string]string `json:"models"`
	ConfigPassword string            `json:"config_password"`
	DefaultModel   string            `json:"default_model"`
}

func loadRuntimeConfig(path string) (server.Config, error) {
	fileCfg := runtimeConfigFile{}
	configPath := path
	if configPath == "" {
		configPath = "rap.config.json"
	}
	if path != "" {
		file, err := os.Open(path)
		if err != nil {
			return server.Config{}, err
		}
		defer file.Close()
		if err := json.NewDecoder(file).Decode(&fileCfg); err != nil {
			return server.Config{}, err
		}
	}

	cfg := server.Config{
		AnthropicAPIKey:  firstNonEmpty(os.Getenv("ANTHROPIC_API_KEY"), fileCfg.Upstream.APIKey),
		ServiceAPIKey:    firstNonEmpty(os.Getenv("RAP_API_KEY"), fileCfg.Service.APIKey),
		AnthropicModel:   firstNonEmpty(os.Getenv("ANTHROPIC_MODEL"), fileCfg.DefaultModel, "claude-sonnet-4-6"),
		AnthropicBaseURL: firstNonEmpty(os.Getenv("ANTHROPIC_BASE_URL"), fileCfg.Upstream.BaseURL, "https://api.anthropic.com"),
		ListenAddr:       firstNonEmpty(os.Getenv("PROXY_ADDR"), fileCfg.Service.ListenAddr, "127.0.0.1:8180"),
		ModelMap:         fileCfg.Models,
		ConfigPath:       configPath,
		ConfigPassword:   fileCfg.ConfigPassword,
	}
	return cfg, nil
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func findRuntimeConfig(start string) (string, bool) {
	if path := os.Getenv("RAP_CONFIG"); path != "" {
		return path, true
	}
	return findFileUpwards(start, "rap.config.json")
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
	return findFileUpwards(start, ".env")
}

func findFileUpwards(start, name string) (string, bool) {
	dir := filepath.Clean(start)
	for {
		path := filepath.Join(dir, name)
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
