package runtime_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/providers"
	coreruntime "github.com/kayiik/llmgw-core/runtime"
)

// A product that describes its instances by registry entry gets each
// provider from core: an OpenAI-compatible entry serves Chat at the base
// URL the instance names, and Messages too when the instance asks for
// translation. An entry the registry does not hold, or one core does not
// build, fails before anything is sent, naming the entry, and lets a route
// fail over.
func TestRegistryProvidersBuildEachInstanceFromItsEntry(t *testing.T) {
	t.Parallel()
	var mu sync.Mutex
	var paths []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		paths = append(paths, r.URL.Path)
		mu.Unlock()
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		_, _ = io.WriteString(w, `{"id":"chatcmpl-fixture","object":"chat.completion","model":"fixture-model",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(server.Close)
	instances := map[string]coreruntime.RegistryInstance{
		"hosted":       {Entry: "custom_openai", Instance: providers.Instance{BaseURL: server.URL + "/v1", Client: server.Client()}, Translate: true},
		"untranslated": {Entry: "custom_openai", Instance: providers.Instance{BaseURL: server.URL + "/v1", Client: server.Client()}},
		"missing":      {Entry: "no_such_entry"},
		"zen":          {Entry: "opencode_zen"},
	}
	describe := func(settings map[string]coreruntime.RegistryInstance, instance string) (coreruntime.RegistryInstance, error) {
		described, ok := settings[instance]
		if !ok {
			return coreruntime.RegistryInstance{}, core.NewConfigurationError("instance "+instance+" is not configured", nil)
		}
		return described, nil
	}
	runtime, err := coreruntime.New(coreruntime.Options[map[string]coreruntime.RegistryInstance]{
		Settings:  coreruntime.NewMemorySettings(instances),
		Providers: coreruntime.RegistryProviders(nil, describe),
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, caller := context.Background(), core.LocalCaller()
	chat := core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "fixture-model", ContentType: core.ContentTypeJSON,
		Body: []byte(`{"messages":[{"role":"user","content":"hi"}]}`)}
	messages := core.Request{Surface: core.ModelSurfaceMessages, Model: "fixture-model", ContentType: core.ContentTypeJSON,
		Body: []byte(`{"max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)}
	if _, err := runtime.Invoke(ctx, caller, "hosted", chat); err != nil {
		t.Fatal(err)
	}
	response, err := runtime.Invoke(ctx, caller, "hosted", messages)
	if err != nil || !strings.Contains(string(response.Body), `"type":"message"`) {
		t.Fatalf("messages over chat = %s, %v", response.Body, err)
	}
	var surface *core.SurfaceError
	if _, err := runtime.Invoke(ctx, caller, "untranslated", messages); !errors.As(err, &surface) {
		t.Fatalf("messages without translation: err = %v, want a surface error", err)
	}
	mu.Lock()
	if got := strings.Join(paths, ","); got != "/v1/chat/completions,/v1/chat/completions" {
		t.Fatalf("upstream = %s", got)
	}
	mu.Unlock()
	for instance, reason := range map[string]string{
		"missing": "the registry holds no such entry",
		"zen":     "OpenCode Zen serves its models over several wires, which the product's own provider chooses between",
	} {
		_, err := runtime.Invoke(ctx, caller, instance, chat)
		var unsupported *providers.UnsupportedEntryError
		if !errors.As(err, &unsupported) || unsupported.Reason != reason || core.ClassifyError(err).Disposition() != core.DispositionFailover {
			t.Fatalf("%s: err = %#v, want the refusal %q", instance, err, reason)
		}
	}
	if _, err := runtime.Invoke(ctx, caller, "unconfigured", chat); err == nil || err.Error() != "instance unconfigured is not configured" {
		t.Fatalf("unconfigured: err = %v, want describe's error", err)
	}
}
