package main

import (
	"os"
	"path/filepath"
	"testing"
)

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
