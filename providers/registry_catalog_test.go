package providers

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	core "github.com/kayiik/llmgw-core"
)

// registryCatalogs are the /models answers the catalogs of the registry's
// OpenAI-compatible hosts and local runtimes are tested against, with the
// model ids each lists. Where a provider lists its models without a key,
// the fixture is its public answer, cut to its first rows; where listing
// takes a key, it is the answer the provider's documentation shows, or the
// fields it documents.
var registryCatalogs = map[string]struct {
	fixture string
	ids     []string
}{
	"deepseek":       {"deepseek.json", []string{"deepseek-flash", "deepseek-v4-pro"}},
	"moonshot":       {"moonshot.json", []string{"kimi-k3", "moonshot-v1-8k"}},
	"moonshot_cn":    {"moonshot.json", []string{"kimi-k3", "moonshot-v1-8k"}},
	"siliconflow":    {"siliconflow.json", []string{"deepseek-ai/DeepSeek-V4-Flash"}},
	"siliconflow_cn": {"siliconflow.json", []string{"deepseek-ai/DeepSeek-V4-Flash"}},
	"nvidia_nim":     {"nvidia_nim.json", []string{"01-ai/yi-large", "adept/fuyu-8b"}},
	"venice":         {"venice.json", []string{"gemini-3-6-flash", "gemini-3-7-flash"}},
	"near_ai":        {"near_ai.json", []string{"anthropic/claude-fable-5", "anthropic/claude-fable-5-1"}},
	"minimax":        {"minimax.json", []string{"MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.5"}},
	"minimax_cn":     {"minimax.json", []string{"MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.5"}},
	"longcat":        {"longcat.json", []string{"LongCat-2.5-Preview", "LongCat-2.0"}},
	"modelscope":     {"modelscope.json", []string{"deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro-0813"}},
	"modelscope_cn":  {"modelscope_cn.json", []string{"deepseek-ai/DeepSeek-V4-Flash-0731", "deepseek-ai/DeepSeek-V4-Pro"}},
	"novita":         {"novita.json", []string{"zai-org/glm-5.3-flash", "deepseek/deepseek-v4-pro-0813-p"}},
	"avian":          {"avian.json", []string{"deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4-flash"}},
	"shengsuanyun":   {"shengsuanyun.json", []string{"bigmodel/glm-5.3-flash", "anthropic/claude-fable-5.1"}},
	"litellm":        {"litellm.json", []string{"gpt-5.6-terra"}},
	"vllm":           {"vllm.json", []string{"Qwen/Qwen3-8B"}},
	"lmstudio":       {"lmstudio.json", []string{"qwen3-8b", "text-embedding-nomic-embed-text-v1.5"}},
}

// registryUndiscoverable are the entries whose provider documents no model
// list, so a product supplies their models.
var registryUndiscoverable = []string{
	"zhipu", "zai", "dashscope", "dashscope_cn", "dashscope_us", "alibaba_coding_plan", "alibaba_coding_plan_anthropic",
	"alibaba_coding_plan_cn", "alibaba_coding_plan_cn_anthropic", "volcengine_ark", "vivgrid",
}

// Each host's catalog is read from its /models answer at the path its
// default base URL names, with the instance's key, through the provider
// Build makes for the entry.
func TestRegistryHostsListTheirCatalogs(t *testing.T) {
	t.Parallel()
	for id, catalog := range registryCatalogs {
		t.Run(id, func(t *testing.T) {
			t.Parallel()
			entry, ok := DefaultRegistry().ByID(id)
			if !ok || !entry.SupportsModelDiscovery {
				t.Fatalf("entry = %+v, want one that supports model discovery", entry)
			}
			fixture, err := os.ReadFile(filepath.Join("testdata", "catalogs", catalog.fixture))
			if err != nil {
				t.Fatal(err)
			}
			base, err := url.Parse(entry.DefaultBaseURL)
			if err != nil {
				t.Fatal(err)
			}
			var mu sync.Mutex
			var seen []string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				seen = append(seen, r.Method+" "+r.URL.Path+" "+r.Header.Get("Authorization"))
				mu.Unlock()
				w.Header().Set("Content-Type", core.ContentTypeJSON)
				_, _ = w.Write(fixture)
			}))
			t.Cleanup(server.Close)
			provider, err := Build(entry, Instance{BaseURL: server.URL + base.Path, CatalogClient: server.Client()})
			if err != nil {
				t.Fatal(err)
			}
			models, err := provider.ListModels(context.Background(), &core.Credential{APIKey: "fixture-key"})
			if err != nil {
				t.Fatal(err)
			}
			if ids := modelIDsOf(models); !slices.Equal(ids, catalog.ids) {
				t.Errorf("models = %v, want %v", ids, catalog.ids)
			}
			mu.Lock()
			defer mu.Unlock()
			if want := []string{"GET " + base.Path + "/models Bearer fixture-key"}; !slices.Equal(seen, want) {
				t.Errorf("upstream = %q, want %q", seen, want)
			}
		})
	}
}

// An entry whose provider documents no model list declares no discovery,
// and still builds, at its default base URL.
func TestRegistryHostsWithoutAModelList(t *testing.T) {
	t.Parallel()
	for _, id := range registryUndiscoverable {
		entry, ok := DefaultRegistry().ByID(id)
		if !ok || entry.SupportsModelDiscovery || entry.DefaultBaseURL == "" {
			t.Errorf("%s: entry = %+v, want one without discovery and with a default base URL", id, entry)
			continue
		}
		if _, err := Build(entry, Instance{}); err != nil {
			t.Errorf("%s: %v", id, err)
		}
	}
	// A recorded answer only proves a catalog the entry claims.
	for id := range registryCatalogs {
		if slices.Contains(registryUndiscoverable, id) {
			t.Errorf("%s is both discoverable and not", id)
		}
	}
}

// The registry's anthropic-wire hosts serve Messages at their base URL.
func TestRegistryAnthropicHostsServeMessages(t *testing.T) {
	t.Parallel()
	var mu sync.Mutex
	var paths []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		paths = append(paths, r.URL.Path+" "+r.Header.Get("x-api-key"))
		mu.Unlock()
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		_, _ = io.WriteString(w, anthropicFixtureMessage)
	}))
	t.Cleanup(server.Close)
	for _, id := range []string{"alibaba_coding_plan_anthropic", "alibaba_coding_plan_cn_anthropic"} {
		entry, _ := DefaultRegistry().ByID(id)
		base, err := url.Parse(entry.DefaultBaseURL)
		if err != nil {
			t.Fatal(err)
		}
		provider, err := Build(entry, Instance{BaseURL: server.URL + base.Path, Client: server.Client()})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := provider.Invoke(context.Background(), anthropicMessagesRequest(`{"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`, anthropicAPIKey("fixture-key"))); err != nil {
			t.Fatalf("%s: %v", id, err)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if want := []string{"/apps/anthropic/v1/messages fixture-key", "/apps/anthropic/v1/messages fixture-key"}; !slices.Equal(paths, want) {
		t.Fatalf("upstream = %q", paths)
	}
}
