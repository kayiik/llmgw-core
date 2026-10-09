package providers

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	core "github.com/kayiik/llmgw-core"
)

// buildUnsupported is every default registry entry Build makes no provider
// for, with the reason it gives. A new entry Build cannot make fails the
// test until it is listed here, and so does a listed entry the registry no
// longer holds.
var buildUnsupported = map[string]string{
	"claude_code":        "the entry is client only, not available",
	"openai_codex":       "OpenAI Codex signs in with a ChatGPT account and speaks Responses to its own backend, which an extension daemon serves",
	"opencode_zen":       "OpenCode Zen serves its models over several wires, which the product's own provider chooses between",
	"github_copilot":     "core implements no provider for this runtime type; an extension daemon serves it",
	"google_antigravity": "core implements no provider for this runtime type; an extension daemon serves it",
	"edge_tts":           "core implements no provider for this runtime type; an extension daemon serves it",
}

// buildBases are the base URLs of the entries that require one and have no
// default.
var buildBases = map[string]string{
	"custom_openai": "https://upstream.invalid/v1", "custom_anthropic": "https://upstream.invalid",
	"azure_openai": "https://fixture.openai.azure.com",
}

// Every available entry of the default registry builds through Build into
// the provider of its runtime type, or is refused with its reason.
func TestBuildMakesTheProviderOfEveryEntryCoreServes(t *testing.T) {
	t.Parallel()
	registry := DefaultRegistry()
	for id := range buildUnsupported {
		if _, ok := registry.ByID(id); !ok {
			t.Errorf("%s is listed as unsupported, but the registry holds no such entry", id)
		}
	}
	for _, entry := range registry.Entries() {
		provider, err := Build(entry, Instance{BaseURL: buildBases[entry.ID], Project: "fixture-project"})
		if reason, refused := buildUnsupported[entry.ID]; refused {
			var unsupported *UnsupportedEntryError
			var failure *core.ProviderError
			if provider != nil || !errors.As(err, &unsupported) || unsupported.Entry != entry.ID || unsupported.RuntimeType != entry.RuntimeType ||
				unsupported.Reason != reason || !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration ||
				core.ClassifyError(err).Disposition() != core.DispositionFailover {
				t.Errorf("%s: provider = %v, err = %#v, want the refusal %q", entry.ID, provider, err, reason)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", entry.ID, err)
			continue
		}
		var built bool
		switch entry.RuntimeType {
		case "openai_compatible":
			p, ok := provider.(*OpenAICompatible)
			built = ok && p.registryID == entry.ID && p.label == "OpenAI-compatible"
		case "bedrock":
			p, ok := provider.(*OpenAICompatible)
			built = ok && p.label == "Bedrock" && p.baseURL == "https://bedrock-runtime.us-east-1.amazonaws.com/v1"
		case "anthropic":
			_, built = provider.(*Anthropic)
		case "ai_studio", "vertex_ai":
			p, ok := provider.(*Google)
			built = ok && string(p.deployment) == entry.RuntimeType
		case "azure_openai":
			_, built = provider.(*AzureOpenAI)
		case "ollama":
			_, built = provider.(*Ollama)
		case "elevenlabs":
			_, built = provider.(*ElevenLabs)
		case "mimo":
			_, built = provider.(*MiMo)
		}
		if !built {
			t.Errorf("%s: built %T for runtime type %s", entry.ID, provider, entry.RuntimeType)
		}
	}
}

// Every runtime type SupportedRuntimeTypes lists builds, so the list and
// Build cannot drift apart.
func TestSupportedRuntimeTypesAllBuild(t *testing.T) {
	t.Parallel()
	bases := map[string]string{"openai_compatible": "https://upstream.invalid/v1", "azure_openai": "https://fixture.openai.azure.com"}
	for _, runtimeType := range SupportedRuntimeTypes() {
		entry := RegistryEntry{ID: "fixture_" + runtimeType, RuntimeType: runtimeType, Availability: ProviderAvailable}
		if provider, err := Build(entry, Instance{BaseURL: bases[runtimeType], Project: "fixture-project"}); err != nil || provider == nil {
			t.Errorf("%s: provider = %v, err = %v", runtimeType, provider, err)
		}
	}
}

// Build refuses, before any request, an entry it cannot make and settings
// its provider refuses, each as a configuration error that permits
// failover. A runtime type only a product knows names itself.
func TestBuildRefusesWhatItCannotMake(t *testing.T) {
	t.Parallel()
	entry := func(id string) RegistryEntry {
		t.Helper()
		found, ok := DefaultRegistry().ByID(id)
		if !ok {
			t.Fatalf("the registry holds no %s", id)
		}
		return found
	}
	product := RegistryEntry{ID: "fixture_product", RuntimeType: "fixture_runtime", Availability: ProviderAvailable}
	planned := entry("openai")
	planned.Availability = ProviderPlanned
	for name, testCase := range map[string]struct {
		entry    RegistryEntry
		instance Instance
		message  string
		reason   string
	}{
		"a product's runtime type": {
			entry: product, reason: "core implements no provider for this runtime type",
			message: `registry entry "fixture_product", runtime type "fixture_runtime": core implements no provider for this runtime type`,
		},
		"a planned entry": {entry: planned, reason: "the entry is planned, not available"},
		"no base URL":     {entry: entry("custom_openai"), message: `registry entry "custom_openai" needs a base URL`},
		"no base URL, where the provider has a default": {
			entry: entry("custom_anthropic"), message: `registry entry "custom_anthropic" needs a base URL`,
		},
		"a Vertex AI location that is not a path segment": {
			entry: entry("vertex_ai"), instance: Instance{Location: "us/central1"},
			message: "Vertex AI project and location must be single URL path segments",
		},
		"a Bedrock region that is not a region": {entry: entry("bedrock"), instance: Instance{Region: "US East"}, message: `"us east" is not an AWS region name`},
		"Ollama's OpenAI-compatible URL": {
			entry: entry("ollama"), instance: Instance{BaseURL: "http://127.0.0.1:11434/v1"},
			message: "Ollama uses its native daemon root, not the OpenAI-compatible /v1 URL; remove /v1.",
		},
		"a header that cannot be sent": {
			entry: entry("ai_studio"), instance: Instance{Headers: map[string]string{"X Fixture": "v"}},
			message: `Google: the configured header name "X Fixture" is not an HTTP token`,
		},
	} {
		provider, err := Build(testCase.entry, testCase.instance)
		var failure *core.ProviderError
		var unsupported *UnsupportedEntryError
		switch {
		case provider != nil || !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration ||
			core.ClassifyError(err).Disposition() != core.DispositionFailover:
			t.Errorf("%s: provider = %v, err = %#v, want a configuration error", name, provider, err)
		case testCase.message != "" && err.Error() != testCase.message:
			t.Errorf("%s: error = %q, want %q", name, err, testCase.message)
		case (testCase.reason != "") != errors.As(err, &unsupported):
			t.Errorf("%s: error = %#v, want an unsupported entry: %v", name, err, testCase.reason != "")
		case testCase.reason != "" && unsupported.Reason != testCase.reason:
			t.Errorf("%s: reason = %q, want %q", name, unsupported.Reason, testCase.reason)
		}
	}
}

// The instance's settings reach the provider, and the entry fills what they
// leave out: its default base URL, region and location, and its id, which
// tells an OpenAI-compatible provider the wire facts it keys on the entry.
// An entry a product's overlay adds builds as any other.
func TestBuildPassesTheInstanceAndTheEntryDefaults(t *testing.T) {
	t.Parallel()
	registry := DefaultRegistry()
	build := func(id string, instance Instance) core.Provider {
		t.Helper()
		entry, ok := registry.ByID(id)
		if !ok {
			t.Fatalf("the registry holds no %s", id)
		}
		provider, err := Build(entry, instance)
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		return provider
	}
	if p := build("bedrock", Instance{Region: "eu-west-1"}).(*OpenAICompatible); p.baseURL != "https://bedrock-runtime.eu-west-1.amazonaws.com/v1" {
		t.Errorf("Bedrock base = %s", p.baseURL)
	}
	if p := build("vertex_ai", Instance{Project: "fixture-project", RequestType: "paygo"}).(*Google); p.location != "global" || p.project != "fixture-project" ||
		p.requestType != "shared" || p.baseURL != "" {
		t.Errorf("Vertex AI = %+v", p)
	}
	if p := build("anthropic", Instance{}).(*Anthropic); p.baseURL != "https://api.anthropic.com" {
		t.Errorf("Anthropic base = %s", p.baseURL)
	}
	if p := build("ollama", Instance{}).(*Ollama); p.chatURL != "http://127.0.0.1:11434/api/chat" {
		t.Errorf("Ollama = %s", p.chatURL)
	}
	if surfaces := build("openai", Instance{}).NativeSurfaces("fixture-model"); !slices.Contains(surfaces, core.ModelSurfaceResponses) {
		t.Errorf("the openai entry's surfaces = %v, want Responses native", surfaces)
	}
	if surfaces := build("elevenlabs", Instance{SpeechModels: []string{"fixture-voice-model"}}).NativeSurfaces("fixture-voice-model"); !slices.Equal(surfaces, []core.ModelSurface{core.ModelSurfaceAudioSpeech}) {
		t.Errorf("ElevenLabs surfaces = %v", surfaces)
	}

	// The settings reach the upstream.
	var mu sync.Mutex
	var seen []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen = append(seen, r.URL.Path+" "+r.Header.Get("X-Fixture")+" "+r.Header.Get("x-goog-api-key"))
		mu.Unlock()
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		_, _ = io.WriteString(w, googleHelloAnswer)
	}))
	t.Cleanup(server.Close)
	studio := build("ai_studio", Instance{BaseURL: server.URL, Client: server.Client(), Headers: map[string]string{"X-Fixture": "configured"}})
	if _, err := studio.Invoke(context.Background(), googleChatRequest("gemini-fixture", `{"messages":[{"role":"user","content":"hi"}]}`, googleKey("k"))); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if !slices.Equal(seen, []string{"/models/gemini-fixture:generateContent configured k"}) {
		t.Errorf("upstream = %q", seen)
	}

	// An overlay's own entry builds by its runtime type, with its id.
	overlaid, err := registry.WithOverlay(Overlay{Add: []RegistryEntry{{
		ID: "fixture_hosted", Label: "Fixture hosted", Description: "A product's own OpenAI-compatible host.", RuntimeType: "openai_compatible",
		Protocol: "openai", Availability: ProviderAvailable, AuthMethods: []string{AuthAPIKey}, ConnectionScope: ConnectionScopeSystemOrPersonal,
		DefaultProviderID: "fixture_hosted", DefaultBaseURL: "https://fixture-hosted.invalid/v1", RiskLevel: "green",
	}}}, ValidationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	entry, _ := overlaid.Lookup("fixture_hosted")
	provider, err := Build(entry, Instance{})
	if p, ok := provider.(*OpenAICompatible); err != nil || !ok || p.registryID != "fixture_hosted" || !strings.HasPrefix(p.baseURL, "https://fixture-hosted.invalid") {
		t.Errorf("overlay entry: provider = %#v, err = %v", provider, err)
	}
}

// An OpenAI-compatible provider's RegistryID may be an alias, which keys
// the same wire facts as its entry's id, or an entry a product's overlay
// adds, when its Registry is the product's. An id no registry holds is
// kept as given.
func TestOpenAICompatibleResolvesItsRegistryID(t *testing.T) {
	t.Parallel()
	overlaid, err := DefaultRegistry().WithOverlay(Overlay{Add: []RegistryEntry{{
		ID: "fixture_hosted", Aliases: []string{"fixture-alias"}, Label: "Fixture hosted", Description: "A product's own host.",
		RuntimeType: "openai_compatible", Protocol: "openai", Availability: ProviderAvailable, AuthMethods: []string{AuthAPIKey},
		ConnectionScope: ConnectionScopeSystemOrPersonal, DefaultProviderID: "fixture-hosted", RiskLevel: RiskGreen,
	}}}, ValidationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for _, testCase := range []struct {
		registry      *Registry
		id, canonical string
	}{
		{nil, "pollinations.ai", "pollinations"},
		{nil, " LLM7.IO ", "llm7"},
		{nil, "kimi", "moonshot"},
		{overlaid, "fixture-alias", "fixture_hosted"},
		{overlaid, "Fixture_Hosted", "fixture_hosted"},
		{nil, "fixture-alias", "fixture-alias"},
		{nil, " Product_Only ", "product_only"},
	} {
		provider, err := NewOpenAICompatible(OpenAICompatibleConfig{BaseURL: "https://upstream.invalid/v1", RegistryID: testCase.id, Registry: testCase.registry})
		if err != nil || provider.registryID != testCase.canonical {
			t.Errorf("RegistryID %q: provider = %+v, err = %v, want %q", testCase.id, provider, err, testCase.canonical)
		}
	}
	// The alias keys Pollinations' own Chat path.
	provider, _ := NewOpenAICompatible(OpenAICompatibleConfig{BaseURL: "https://upstream.invalid", RegistryID: "pollinations.ai"})
	if path := provider.chatPath(); path != "/v1/chat/completions" {
		t.Errorf("chat path = %s", path)
	}
}
