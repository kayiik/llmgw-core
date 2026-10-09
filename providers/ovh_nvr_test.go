package providers

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

const ovhFixtureBase = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"

// nvrCall is one request a fixture NVR host received.
type nvrCall struct {
	url, contentType, accept, authorization, body string
}

// nvrTransport answers every request with answer, as an OVHcloud host
// would, and records it.
type nvrTransport struct {
	mu     sync.Mutex
	calls  []nvrCall
	answer func(w http.ResponseWriter)
}

func (n *nvrTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	var body []byte
	if r.Body != nil {
		body, _ = io.ReadAll(r.Body)
	}
	n.mu.Lock()
	n.calls = append(n.calls, nvrCall{
		url: r.URL.String(), contentType: r.Header.Get("Content-Type"), accept: r.Header.Get("Accept"),
		authorization: r.Header.Get("Authorization"), body: string(body),
	})
	n.mu.Unlock()
	recorder := httptest.NewRecorder()
	n.answer(recorder)
	response := recorder.Result()
	response.Request = r
	return response, nil
}

func (n *nvrTransport) take() []nvrCall {
	n.mu.Lock()
	defer n.mu.Unlock()
	calls := n.calls
	n.calls = nil
	return calls
}

// newNVRTest is an OVHcloud AI Endpoints instance whose every request is
// answered by answer.
func newNVRTest(t *testing.T, base string, answer func(w http.ResponseWriter)) (*OpenAICompatible, *nvrTransport) {
	t.Helper()
	transport := &nvrTransport{answer: answer}
	provider, err := NewOpenAICompatible(OpenAICompatibleConfig{
		BaseURL: base, RegistryID: "ovh_ai_endpoints", Client: &http.Client{Transport: transport}, CatalogClient: &http.Client{Transport: transport},
	})
	if err != nil {
		t.Fatal(err)
	}
	return provider, transport
}

// nvrFixtureAnswer answers with a recorded fixture's response.
func nvrFixtureAnswer(t *testing.T, fixture audioWireFixture) func(w http.ResponseWriter) {
	return func(w http.ResponseWriter) { writeAudioFixture(t, w, fixture) }
}

func nvrSpeech(model, body string, credential *core.Credential) core.Request {
	return core.Request{Surface: core.ModelSurfaceAudioSpeech, Model: model, Body: []byte(body), ContentType: core.ContentTypeJSON, Credential: credential}
}

func compactJSON(t *testing.T, raw []byte) string {
	t.Helper()
	var buffer bytes.Buffer
	if err := json.Compact(&buffer, raw); err != nil {
		t.Fatal(err)
	}
	return buffer.String()
}

// An OpenAI speech request to an NVR voice reaches the model's own host as
// Riva's synthesis request, recorded from OVHcloud: the input, the language
// its name holds, linear PCM at 16 kHz, and the voice. An OpenAI voice name
// is no voice of the language, so its base voice speaks and the swap is
// reported. The WAV answer comes back as audio/wav.
func TestOVHNVRSpeechWireFixture(t *testing.T) {
	t.Parallel()
	fixture := loadAudioFixture(t, "ovh_ai_endpoints", "nvr-speech-success")
	provider, transport := newNVRTest(t, ovhFixtureBase, nvrFixtureAnswer(t, fixture))
	audio, err := base64.StdEncoding.DecodeString(fixture.Response.BodyBase64)
	if err != nil {
		t.Fatal(err)
	}
	for voice, losses := range map[string][]core.Loss{
		"alloy": {{Path: "voice", Class: translate.LossDropped, Severity: translate.LossAdvisory,
			Detail: "the voice is not one of the model's language, so its base voice speaks"}},
		"English-US.Female-1": nil,
	} {
		response, err := provider.Invoke(context.Background(), nvrSpeech(fixture.Request.Model,
			`{"model":"nvr-tts-en-us","input":"Hi.","voice":"`+voice+`","speed":1}`, &core.Credential{APIKey: "fixture-key"}))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(response.Body, audio) || response.ContentType != "audio/wav" || !slicesEqualLosses(response.Losses, losses) {
			t.Fatalf("voice %s: response = %d bytes of %s, losses %+v", voice, len(response.Body), response.ContentType, response.Losses)
		}
		want := nvrCall{
			url:         "https://nvr-tts-en-us.endpoints.kepler.ai.cloud.ovh.net" + fixture.Request.Path,
			contentType: fixture.Request.ContentType, accept: "application/octet-stream", authorization: "Bearer fixture-key",
			body: compactJSON(t, fixture.Request.Body),
		}
		if calls := transport.take(); len(calls) != 1 || calls[0] != want {
			t.Fatalf("voice %s: upstream = %+v\nwant %+v", voice, calls, want)
		}
	}
}

func slicesEqualLosses(got, want []core.Loss) bool {
	return len(got) == len(want) && (len(got) == 0 || slices.Equal(got, want))
}

// OVHcloud's refusal of a voice its model lacks, recorded, ends the request
// with its status; a rate limit may be repeated after the wait it names.
func TestOVHNVRSpeechRefusals(t *testing.T) {
	t.Parallel()
	fixture := loadAudioFixture(t, "ovh_ai_endpoints", "nvr-speech-error-400")
	provider, transport := newNVRTest(t, ovhFixtureBase, nvrFixtureAnswer(t, fixture))
	_, err := provider.Invoke(context.Background(), nvrSpeech(fixture.Request.Model,
		`{"model":"nvr-tts-de-de","input":"Hallo.","voice":"German-DE-Female-1"}`, nil))
	var failure *core.ProviderError
	var cause *InvocationError
	if !errors.As(err, &failure) || err.Error() != "OVHcloud AI Endpoints returned HTTP 400" || core.ClassifyError(err).Disposition() != core.DispositionTerminal ||
		!errors.As(err, &cause) || cause.Status != http.StatusBadRequest || !strings.Contains(cause.Msg, "Voice German-DE-Female-1 for language de-DE not found") {
		t.Fatalf("error = %#v", err)
	}
	if calls := transport.take(); len(calls) != 1 || calls[0].body != compactJSON(t, fixture.Request.Body) || calls[0].authorization != "" {
		t.Fatalf("upstream = %+v", calls)
	}

	limited, _ := newNVRTest(t, ovhFixtureBase, func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Retry-After", "37")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"message":"API rate limit exceeded"}`)
	})
	_, err = limited.Invoke(context.Background(), nvrSpeech("nvr-tts-en-us", `{"input":"Hi.","voice":"alloy"}`, nil))
	if classification := core.ClassifyError(err); classification.StatusCode != http.StatusTooManyRequests || !classification.Retryable ||
		classification.RetryAfter != 37*time.Second {
		t.Fatalf("rate limit: err = %v, classification = %+v", err, classification)
	}
}

// A keyless instance sends no Authorization, "public" included, as its
// anonymous tier wants. A voice of the model's language is sent as named,
// and a language with no documented voices needs one.
func TestOVHNVRSpeechVoicesAndKeylessAccess(t *testing.T) {
	t.Parallel()
	fixture := loadAudioFixture(t, "ovh_ai_endpoints", "nvr-speech-success")
	provider, transport := newNVRTest(t, ovhFixtureBase, nvrFixtureAnswer(t, fixture))
	for _, testCase := range []struct {
		model, voice, sent string
		credential         *core.Credential
	}{
		{"nvr-tts-es-es", "Spanish-ES-Male-1", "Spanish-ES-Male-1", nil},
		{"nvr-tts-es-es", "English-US.Male-1", "Spanish-ES-Female-1", &core.Credential{APIKey: "public"}},
		{"nvr-tts-de-de", "", "German-DE-Male-1", nil},
		{"nvr-tts-it-it", "", "Italian-IT-Female-1", nil},
		{"nvr-tts-fr-fr", "French-FR-Female-1", "French-FR-Female-1", nil},
	} {
		if _, err := provider.Invoke(context.Background(), nvrSpeech(testCase.model, `{"input":"Hi.","voice":"`+testCase.voice+`"}`, testCase.credential)); err != nil {
			t.Fatalf("%s %q: %v", testCase.model, testCase.voice, err)
		}
		calls := transport.take()
		if len(calls) != 1 || calls[0].authorization != "" || !strings.Contains(calls[0].body, `"voice_name":"`+testCase.sent+`"`) ||
			!strings.HasPrefix(calls[0].url, "https://"+testCase.model+".endpoints.kepler.ai.cloud.ovh.net/") {
			t.Fatalf("%s %q: upstream = %+v, want voice %s", testCase.model, testCase.voice, calls, testCase.sent)
		}
	}
}

// What an NVR voice cannot serve is refused before anything is sent: a
// format other than WAV, no input, a language with no documented voice and
// none named, and an instance whose base URL is not OVHcloud's.
func TestOVHNVRSpeechRefusesBeforeSending(t *testing.T) {
	t.Parallel()
	provider, transport := newNVRTest(t, ovhFixtureBase, func(w http.ResponseWriter) { w.WriteHeader(http.StatusInternalServerError) })
	for body, class := range map[string]core.ProviderErrorClass{
		`{"input":"Hi.","response_format":"mp3"}`: core.ProviderErrorInvalidRequest,
		`{"input":" ","voice":"alloy"}`:           core.ProviderErrorInvalidRequest,
	} {
		_, err := provider.Invoke(context.Background(), nvrSpeech("nvr-tts-en-us", body, nil))
		var failure *core.ProviderError
		if !errors.As(err, &failure) || failure.Class != class {
			t.Errorf("%s: error = %v, want %s", body, err, class)
		}
	}
	if _, err := provider.Invoke(context.Background(), nvrSpeech("nvr-tts-fr-fr", `{"input":"Salut."}`, nil)); err == nil ||
		err.Error() != "an OVHcloud NVR model of fr-FR needs its voice named" {
		t.Errorf("unknown language without a voice: %v", err)
	}
	proxied, proxiedTransport := newNVRTest(t, "https://proxy.invalid/v1", func(w http.ResponseWriter) {})
	var failure *core.ProviderError
	if _, err := proxied.Invoke(context.Background(), nvrSpeech("nvr-tts-en-us", `{"input":"Hi."}`, nil)); !errors.As(err, &failure) ||
		failure.Class != core.ProviderErrorConfiguration {
		t.Errorf("another base URL: %v, want a configuration error", err)
	}
	if calls := append(transport.take(), proxiedTransport.take()...); len(calls) != 0 {
		t.Fatalf("upstream = %+v", calls)
	}
}

// A 200 that is not WAV audio cannot be used.
func TestOVHNVRSpeechAnswerMustBeWAV(t *testing.T) {
	t.Parallel()
	provider, _ := newNVRTest(t, ovhFixtureBase, func(w http.ResponseWriter) { _, _ = io.WriteString(w, `{"status":"ok"}`) })
	_, err := provider.Invoke(context.Background(), nvrSpeech("nvr-tts-en-us", `{"input":"Hi."}`, nil))
	if classification := core.ClassifyError(err); err == nil || !classification.FailoverEligible || !classification.CircuitFailure {
		t.Fatalf("error = %v, classification = %+v", err, classification)
	}
}

// An NVR voice serves speech alone: other surfaces are surface errors and
// speech does not stream. An instance of another entry serves the model on
// the OpenAI wire as any other.
func TestOVHNVRModelsServeSpeechAlone(t *testing.T) {
	t.Parallel()
	provider, transport := newNVRTest(t, ovhFixtureBase, func(w http.ResponseWriter) {})
	if surfaces := provider.NativeSurfaces("nvr-tts-en-us"); !slices.Equal(surfaces, []core.ModelSurface{core.ModelSurfaceAudioSpeech}) {
		t.Fatalf("surfaces = %v", surfaces)
	}
	if core.PreservesWire(provider, "nvr-tts-en-us", core.ModelSurfaceAudioSpeech) {
		t.Fatal("NVR speech was declared preserved")
	}
	var surface *core.SurfaceError
	chat := core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "nvr-tts-en-us", Body: []byte(`{}`), ContentType: core.ContentTypeJSON}
	if _, err := provider.Invoke(context.Background(), chat); !errors.As(err, &surface) {
		t.Fatalf("chat: %v", err)
	}
	if _, err := provider.Stream(context.Background(), chat); !errors.As(err, &surface) {
		t.Fatalf("chat stream: %v", err)
	}
	if _, err := provider.Stream(context.Background(), nvrSpeech("nvr-tts-en-us", `{"input":"Hi."}`, nil)); core.ClassifyError(err).Disposition() != core.DispositionFailover {
		t.Fatalf("speech stream: %v", err)
	}
	if calls := transport.take(); len(calls) != 0 {
		t.Fatalf("upstream = %+v", calls)
	}
	other, err := NewOpenAICompatible(OpenAICompatibleConfig{BaseURL: ovhFixtureBase})
	if err != nil {
		t.Fatal(err)
	}
	if surfaces := other.NativeSurfaces("nvr-tts-en-us"); !slices.Contains(surfaces, core.ModelSurfaceChatCompletions) {
		t.Fatalf("another entry's surfaces = %v", surfaces)
	}
}

// With a key, OVHcloud's recorded catalog lists its NVR voices as speech
// models alone.
func TestOVHCatalogDeclaresNVRVoicesSpeechOnly(t *testing.T) {
	t.Parallel()
	catalog, err := os.ReadFile("testdata/ovh_ai_endpoints/models-2026-09-28.json")
	if err != nil {
		t.Fatal(err)
	}
	provider, _ := newNVRTest(t, ovhFixtureBase, func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		_, _ = w.Write(catalog)
	})
	models, err := provider.ListModels(context.Background(), &core.Credential{APIKey: "fixture-key"})
	if err != nil {
		t.Fatal(err)
	}
	var voices []string
	for _, model := range models {
		speech := slices.Equal(model.SupportedAPIs, []string{"/v1/audio/speech"}) && model.LegacyCapabilities["tts"] == true
		if strings.HasPrefix(model.ID, "nvr-tts-") != speech {
			t.Errorf("%s: endpoints %v, capabilities %v", model.ID, model.SupportedAPIs, model.LegacyCapabilities)
		}
		if speech {
			voices = append(voices, model.ID)
		}
	}
	if !slices.Equal(voices, []string{"nvr-tts-it-it", "nvr-tts-en-us", "nvr-tts-de-de", "nvr-tts-es-es"}) {
		t.Fatalf("voices = %v", voices)
	}
}
