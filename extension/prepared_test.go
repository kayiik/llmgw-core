package extension_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"slices"
	"strings"
	"testing"

	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/extension"
)

func TestExtensionPreparedNativeDeclarationFreezesRequest(t *testing.T) {
	const body = `{"model":"fixture","messages":[{"role":"user","content":"hi"}],"unreviewed":{"integer":9007199254740993}}`
	const answer = `{"id":"chatcmpl-native","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},"vendor_native":true}`
	client, calls := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/extension/v1/native/invoke" || r.Header.Get(extension.HeaderSurface) != "chat_completions" || r.Header.Get(extension.HeaderModel) != "fixture" {
			t.Errorf("wrong frozen operation: %s %v", r.URL.Path, r.Header)
		}
		got, _ := io.ReadAll(r.Body)
		if string(got) != body {
			t.Errorf("native request fields changed: %s", got)
		}
		if r.Header.Get(extension.HeaderCredentialToken) != "fresh-fixture" {
			t.Error("execution credential was not rebound")
		}
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		_, _ = io.WriteString(w, answer)
	})
	p := extension.NewProvider(client, extension.ProviderInfo{ID: "native", Surfaces: []core.ModelSurface{core.ModelSurfaceChatCompletions}})
	req := core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "fixture", Body: []byte(body), ContentType: core.ContentTypeJSON}
	prepared, err := core.PrepareCall(context.Background(), p, req, core.CallPreparationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	d := prepared.Description()
	if d.UpstreamAPI != core.UpstreamAPIChatCompletions || len(d.Conversions) != 0 || d.WirePreserved == core.SupportSupported {
		t.Fatalf("native API versus wire facts=%+v", d)
	}
	if _, err := core.PlanPreparedCall(d, core.CallPolicy{}); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 0 {
		t.Fatal("preparation performed network I/O")
	}
	req.Body[0] = '!'
	response, err := prepared.Invoke(context.Background(), &core.Credential{Token: "fresh-fixture"})
	if err != nil || string(response.Body) != answer || calls.Load() != 1 {
		t.Fatalf("native operation changed: body=%s err=%v calls=%d", response.Body, err, calls.Load())
	}
}

func TestExtensionPreparedModelOverrideNativeStream(t *testing.T) {
	const wire = "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\"}}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"usage\":{\"input_tokens\":9007199254740993,\"output_tokens\":0,\"total_tokens\":9007199254740993}}}\n\n"
	client, calls := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get(extension.HeaderSurface) != "responses" || r.URL.Path != "/extension/v1/native/stream" {
			t.Errorf("model override was not native Responses: %s %v", r.URL.Path, r.Header)
		}
		w.Header().Set("Content-Type", core.ContentTypeEventStream)
		_, _ = io.WriteString(w, wire)
	})
	row := core.ModelInfo{ID: "fixture", SupportedAPIs: []string{"/v1/responses"}}
	lookups := 0
	p := extension.NewProviderWithModelLookup(client, extension.ProviderInfo{ID: "native", Surfaces: []core.ModelSurface{core.ModelSurfaceChatCompletions}}, func(model string) (core.ModelInfo, bool) { lookups++; return row, model == "fixture" })
	req := core.Request{Surface: core.ModelSurfaceResponses, Model: "fixture", Body: []byte(`{"input":"hi","vendor":false}`), ContentType: core.ContentTypeJSON}
	prepared, err := core.PrepareCall(context.Background(), p, req, core.CallPreparationOptions{Mode: core.CallModeStream})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := core.PlanPreparedCall(prepared.Description(), core.CallPolicy{RequireIncremental: true}); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 0 || lookups != 1 {
		t.Fatalf("prep I/O or repeated lookup: network=%d lookup=%d", calls.Load(), lookups)
	}
	row.SupportedAPIs[0] = "/v1/chat/completions"
	stream, err := prepared.Stream(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var result strings.Builder
	for {
		frame, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		result.Write(frame)
	}
	if result.String() != wire || calls.Load() != 1 || lookups != 1 {
		t.Fatalf("stream/plan changed: bytes=%q calls=%d lookup=%d", result.String(), calls.Load(), lookups)
	}
	observation, available := core.SourceUsageOf(stream)
	if !available || observation.API != core.UpstreamAPIResponses || observation.Usage.InputTokens.Value != 9007199254740993 || observation.Usage.InputTokens.Provenance != "reported" {
		t.Fatalf("native source usage lost/converted: %+v available=%v", observation, available)
	}
	if !slices.Equal(p.NativeSurfaces("fixture"), []core.ModelSurface{core.ModelSurfaceChatCompletions}) {
		t.Fatal("legacy NativeSurfaces contract changed")
	}
}

func TestExtensionPreparedDefaultRefusalAndAllowedExistingMapping(t *testing.T) {
	client, calls := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get(extension.HeaderSurface) != "responses" {
			t.Errorf("hidden daemon Chat conversion used: %s", r.Header.Get(extension.HeaderSurface))
		}
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		_, _ = io.WriteString(w, `{"id":"r1","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"native answer"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
	})
	p := extension.NewProviderWithModelLookup(client, extension.ProviderInfo{ID: "native", Surfaces: []core.ModelSurface{core.ModelSurfaceChatCompletions}}, func(string) (core.ModelInfo, bool) {
		return core.ModelInfo{ID: "fixture", SupportedAPIs: []string{"/responses"}}, true
	})
	req := core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "fixture", Body: []byte(`{"messages":[{"role":"user","content":"hi"}]}`), ContentType: core.ContentTypeJSON}
	if denied, err := core.PrepareCall(context.Background(), p, req, core.CallPreparationOptions{}); err == nil {
		if _, err = core.PlanPreparedCall(denied.Description(), core.CallPolicy{}); err == nil {
			t.Fatal("default cross-API request admitted")
		}
	}
	if calls.Load() != 0 {
		t.Fatal("default refusal performed inference/discovery")
	}
	prepared, err := core.PrepareCall(context.Background(), p, req, core.CallPreparationOptions{AllowAdaptation: true})
	if err != nil {
		t.Fatal(err)
	}
	d := prepared.Description()
	if d.UpstreamAPI != core.UpstreamAPIResponses || len(d.Conversions) != 1 || d.Conversions[0].ID != "chat-to-responses" {
		t.Fatalf("wrong named mapping: %+v", d)
	}
	if _, err := core.PlanPreparedCall(d, core.CallPolicy{AllowAdaptation: true}); err != nil {
		t.Fatal(err)
	}
	collector := core.NewSourceUsageCollector()
	response, err := prepared.Invoke(core.WithSourceUsageCollector(context.Background(), collector), nil)
	if err != nil || !strings.Contains(string(response.Body), "native answer") || calls.Load() != 1 {
		t.Fatalf("mapping failed: %s err=%v calls=%d", response.Body, err, calls.Load())
	}
	history := collector.SourceUsageHistory()
	if len(history.Attempts) != 1 || history.Attempts[0].API != core.UpstreamAPIResponses || history.Attempts[0].Usage.Dialect != "responses" || history.Attempts[0].Usage.InputTokens.Value != 1 {
		t.Fatalf("usage was observed after local conversion: %+v", history)
	}
}

func TestExtensionPreparedChatFacadeDoesNotInventOriginalUsage(t *testing.T) {
	client, _ := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		_, _ = io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}`)
	})
	p := extension.NewProvider(client, extension.ProviderInfo{ID: "native", Surfaces: []core.ModelSurface{core.ModelSurfaceChatCompletions}})
	prepared, err := core.PrepareCall(context.Background(), p, core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "fixture", ContentType: core.ContentTypeJSON, Body: []byte(`{"messages":[]}`)}, core.CallPreparationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	collector := core.NewSourceUsageCollector()
	response, err := prepared.Invoke(core.WithSourceUsageCollector(context.Background(), collector), nil)
	if err != nil || !strings.Contains(string(response.Body), `"prompt_tokens":7`) {
		t.Fatalf("Chat usage body changed: %s %v", response.Body, err)
	}
	if history := collector.SourceUsageHistory(); len(history.Attempts) != 0 {
		t.Fatalf("possibly daemon-converted Chat was labelled original native: %+v", history)
	}
}

func TestExtensionPreparedDeclaredMediaRetainsBinary(t *testing.T) {
	client, calls := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get(extension.HeaderSurface) != "audio_speech" {
			t.Error("media surface changed")
		}
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write([]byte{0, 1, 255, 2})
	})
	p := extension.NewProvider(client, extension.ProviderInfo{ID: "native", Surfaces: []core.ModelSurface{core.ModelSurfaceAudioSpeech}})
	prepared, err := core.PrepareCall(context.Background(), p, core.Request{Surface: core.ModelSurfaceAudioSpeech, Model: "voice", ContentType: core.ContentTypeJSON, Body: []byte(`{"input":"hello","voice":"fixture"}`)}, core.CallPreparationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := core.PlanPreparedCall(prepared.Description(), core.CallPolicy{}); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 0 {
		t.Fatal("media preparation did I/O")
	}
	response, err := prepared.Invoke(context.Background(), nil)
	if err != nil || !slices.Equal(response.Body, []byte{0, 1, 255, 2}) || response.ContentType != "audio/mpeg" {
		t.Fatalf("native media damaged: %v %s %v", response.Body, response.ContentType, err)
	}
}

func TestExtensionPreparedNativeUsageFailureQualityPreservesBody(t *testing.T) {
	const body = `{"id":"r1","status":"failed","error":{"code":"synthetic","message":"fixture"},"output":[],"usage":{"input_tokens":3,"output_tokens":0,"total_tokens":3}}`
	client, _ := daemon(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		_, _ = io.WriteString(w, body)
	})
	p := extension.NewProvider(client, extension.ProviderInfo{ID: "native", Surfaces: []core.ModelSurface{core.ModelSurfaceResponses}})
	prepared, err := core.PrepareCall(context.Background(), p, core.Request{Surface: core.ModelSurfaceResponses, Model: "fixture", ContentType: core.ContentTypeJSON, Body: []byte(`{"input":"hi"}`)}, core.CallPreparationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	collector := core.NewSourceUsageCollector()
	response, err := prepared.Invoke(core.WithSourceUsageCollector(context.Background(), collector), nil)
	if err != nil || string(response.Body) != body {
		t.Fatalf("native failure wire changed: %s %v", response.Body, err)
	}
	observed, available := collector.SourceUsage()
	if !available || observed.API != core.UpstreamAPIResponses || observed.Usage.Quality != "failed" || observed.Usage.InputTokens.Value != 3 {
		t.Fatalf("failure usage lost or called successful: %+v", observed)
	}
}

func TestExtensionPreparedUnknownAndCapabilityOverride(t *testing.T) {
	client, calls := daemon(t, func(http.ResponseWriter, *http.Request) { t.Error("unknown/denied path called daemon") })
	req := core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "fixture", Body: []byte(`{"messages":[{"role":"user","content":"hi"}]}`), ContentType: core.ContentTypeJSON}
	for _, tc := range []struct {
		info   extension.ProviderInfo
		lookup func(string) (core.ModelInfo, bool)
	}{
		{extension.ProviderInfo{ID: "native"}, nil},
		{extension.ProviderInfo{ID: "native", Surfaces: []core.ModelSurface{core.ModelSurfaceChatCompletions}}, nil},
		{extension.ProviderInfo{ID: "native", Surfaces: []core.ModelSurface{core.ModelSurfaceChatCompletions}}, func(string) (core.ModelInfo, bool) { return core.ModelInfo{ID: "fixture"}, true }},
		{extension.ProviderInfo{ID: "native", Surfaces: []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses}}, func(string) (core.ModelInfo, bool) {
			return core.ModelInfo{ID: "fixture", Capabilities: &core.ModelCapabilities{SchemaVersion: core.ModelCapabilitiesSchemaVersion, Operations: core.ModelOperationCapabilities{Chat: core.SupportSupported}}}, true
		}},
		{extension.ProviderInfo{ID: "native", Surfaces: []core.ModelSurface{core.ModelSurfaceChatCompletions}}, func(string) (core.ModelInfo, bool) { return core.ModelInfo{}, false }},
		{extension.ProviderInfo{ID: "native", Surfaces: []core.ModelSurface{core.ModelSurfaceChatCompletions}}, func(string) (core.ModelInfo, bool) {
			return core.ModelInfo{ID: "fixture", SupportedAPIs: []string{"ws:/responses"}}, true
		}},
	} {
		p := extension.NewProviderWithModelLookup(client, tc.info, tc.lookup)
		prepared, err := core.PrepareCall(context.Background(), p, req, core.CallPreparationOptions{AllowAdaptation: true})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := core.PlanPreparedCall(prepared.Description(), core.CallPolicy{AllowAdaptation: true}); err == nil {
			t.Fatalf("unknown declaration invented native API: %+v", prepared.Description())
		}
		if _, err := prepared.Invoke(context.Background(), nil); err == nil {
			t.Fatal("unknown prepared handle executed")
		}
	}
	row := core.ModelInfo{ID: "fixture", Capabilities: &core.ModelCapabilities{SchemaVersion: core.ModelCapabilitiesSchemaVersion, Surfaces: core.ModelSurfaceCapabilities{ChatCompletions: core.SupportUnsupported, Responses: core.SupportSupported}}}
	p := extension.NewProviderWithModelLookup(client, extension.ProviderInfo{ID: "native", Surfaces: []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses}}, func(string) (core.ModelInfo, bool) { return row, true })
	prepared, err := core.PrepareCall(context.Background(), p, req, core.CallPreparationOptions{AllowAdaptation: true})
	if err != nil {
		t.Fatal(err)
	}
	if prepared.Description().UpstreamAPI != core.UpstreamAPIResponses {
		t.Fatalf("model capability override ignored: %+v", prepared.Description())
	}
	if calls.Load() != 0 {
		t.Fatal("local capability assessment made network calls")
	}
	_, err = p.Invoke(context.Background(), core.Request{Surface: core.ModelSurfaceImages, Model: "fixture"})
	var unsupported *core.SurfaceError
	if !errors.As(err, &unsupported) {
		t.Fatalf("legacy surface refusal changed: %v", err)
	}
}
