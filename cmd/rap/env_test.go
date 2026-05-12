package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadRuntimeConfigFromJSONAndEnvironment(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "from-env")
	t.Setenv("ANTHROPIC_BASE_URL", "http://env.example")
	t.Setenv("ANTHROPIC_MODEL", "claude-env-default")

	dir := t.TempDir()
	configPath := filepath.Join(dir, "rap.config.json")
	err := os.WriteFile(configPath, []byte(`{
		"upstream": {
			"base_url": "http://config.example",
			"api_key": "from-config"
		},
		"service": {
			"api_key": "service-from-config",
			"listen_addr": "127.0.0.1:9292"
		},
		"models": {
			"gpt-5": "claude-sonnet-4-6",
			"gpt-5-mini": "claude-haiku-4-6"
		},
		"config_password": "admin-password",
		"default_model": "claude-config-default"
	}`), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	cfg, err := loadRuntimeConfig(configPath)
	if err != nil {
		t.Fatalf("loadRuntimeConfig returned error: %v", err)
	}

	if cfg.AnthropicAPIKey != "from-env" {
		t.Fatalf("AnthropicAPIKey = %q, want env override", cfg.AnthropicAPIKey)
	}
	if cfg.ServiceAPIKey != "service-from-config" {
		t.Fatalf("ServiceAPIKey = %q, want config value", cfg.ServiceAPIKey)
	}
	if cfg.ListenAddr != "127.0.0.1:9292" {
		t.Fatalf("ListenAddr = %q, want config value", cfg.ListenAddr)
	}
	if cfg.AnthropicBaseURL != "http://env.example" {
		t.Fatalf("AnthropicBaseURL = %q, want env override", cfg.AnthropicBaseURL)
	}
	if cfg.AnthropicModel != "claude-env-default" {
		t.Fatalf("AnthropicModel = %q, want env override", cfg.AnthropicModel)
	}
	if cfg.ModelMap["gpt-5"] != "claude-sonnet-4-6" || cfg.ModelMap["gpt-5-mini"] != "claude-haiku-4-6" {
		t.Fatalf("unexpected model map: %+v", cfg.ModelMap)
	}
	if cfg.ConfigPath != configPath {
		t.Fatalf("ConfigPath = %q, want %q", cfg.ConfigPath, configPath)
	}
	if cfg.ConfigPassword != "admin-password" {
		t.Fatalf("ConfigPassword = %q, want config value", cfg.ConfigPassword)
	}
}

func TestLoadRuntimeConfigLetsEnvironmentOverrideListenAddr(t *testing.T) {
	t.Setenv("PROXY_ADDR", "127.0.0.1:9393")

	dir := t.TempDir()
	configPath := filepath.Join(dir, "rap.config.json")
	err := os.WriteFile(configPath, []byte(`{
		"upstream": {"api_key": "from-config"},
		"service": {"listen_addr": "127.0.0.1:9292"}
	}`), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	cfg, err := loadRuntimeConfig(configPath)
	if err != nil {
		t.Fatalf("loadRuntimeConfig returned error: %v", err)
	}

	if cfg.ListenAddr != "127.0.0.1:9393" {
		t.Fatalf("ListenAddr = %q, want env override", cfg.ListenAddr)
	}
}

func TestLoadRuntimeConfigUsesDefaultConfigPathWhenMissing(t *testing.T) {
	cfg, err := loadRuntimeConfig("")
	if err != nil {
		t.Fatalf("loadRuntimeConfig returned error: %v", err)
	}
	if cfg.ConfigPath != "rap.config.json" {
		t.Fatalf("ConfigPath = %q, want default rap.config.json", cfg.ConfigPath)
	}
}

func TestFindRuntimeConfigUsesExplicitPath(t *testing.T) {
	t.Setenv("RAP_CONFIG", "/tmp/custom-rap.config.json")

	path, ok := findRuntimeConfig(t.TempDir())
	if !ok {
		t.Fatal("findRuntimeConfig did not return explicit path")
	}
	if path != "/tmp/custom-rap.config.json" {
		t.Fatalf("findRuntimeConfig = %q, want explicit path", path)
	}
}

func TestLoadDotEnvSetsUnsetVariables(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_MODEL", "")
	t.Setenv("ANTHROPIC_BASE_URL", "")
	t.Setenv("PROXY_ADDR", "")

	dir := t.TempDir()
	dotEnv := filepath.Join(dir, ".env")
	err := os.WriteFile(dotEnv, []byte(`
# local proxy configuration
ANTHROPIC_API_KEY=sk-ant-test
ANTHROPIC_MODEL="claude-test"
ANTHROPIC_BASE_URL='http://127.0.0.1:8080'
PROXY_ADDR=127.0.0.1:9191
`), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	if err := loadDotEnv(dotEnv); err != nil {
		t.Fatalf("loadDotEnv returned error: %v", err)
	}

	assertEnv(t, "ANTHROPIC_API_KEY", "sk-ant-test")
	assertEnv(t, "ANTHROPIC_MODEL", "claude-test")
	assertEnv(t, "ANTHROPIC_BASE_URL", "http://127.0.0.1:8080")
	assertEnv(t, "PROXY_ADDR", "127.0.0.1:9191")
}

func TestLoadDotEnvPreservesExistingEnvironment(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "from-shell")

	dir := t.TempDir()
	dotEnv := filepath.Join(dir, ".env")
	err := os.WriteFile(dotEnv, []byte("ANTHROPIC_API_KEY=from-dotenv\nANTHROPIC_MODEL=claude-test\n"), 0o600)
	if err != nil {
		t.Fatal(err)
	}

	if err := loadDotEnv(dotEnv); err != nil {
		t.Fatalf("loadDotEnv returned error: %v", err)
	}

	assertEnv(t, "ANTHROPIC_API_KEY", "from-shell")
	assertEnv(t, "ANTHROPIC_MODEL", "claude-test")
}

func TestFindDotEnvWalksUpFromWorkingDirectory(t *testing.T) {
	root := t.TempDir()
	nested := filepath.Join(root, "cmd", "rap")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	dotEnv := filepath.Join(root, ".env")
	if err := os.WriteFile(dotEnv, []byte("ANTHROPIC_API_KEY=sk-ant-test\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	found, ok := findDotEnv(nested)
	if !ok {
		t.Fatal("findDotEnv did not find parent .env")
	}
	if found != dotEnv {
		t.Fatalf("findDotEnv = %q, want %q", found, dotEnv)
	}
}

func assertEnv(t *testing.T, key, want string) {
	t.Helper()
	if got := os.Getenv(key); got != want {
		t.Fatalf("%s = %q, want %q", key, got, want)
	}
}
