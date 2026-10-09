package providers

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
)

// headerUpstream is a synthetic upstream that records the headers of every
// request and answers each with answer.
type headerUpstream struct {
	mu      sync.Mutex
	paths   []string
	headers []http.Header
	answer  func(w http.ResponseWriter, r *http.Request, body []byte)
}

func newHeaderUpstream(t *testing.T, answer func(http.ResponseWriter, *http.Request, []byte)) (*headerUpstream, *httptest.Server) {
	t.Helper()
	upstream := &headerUpstream{answer: answer}
	server := httptest.NewServer(upstream)
	t.Cleanup(server.Close)
	return upstream, server
}

func (u *headerUpstream) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	u.mu.Lock()
	u.paths = append(u.paths, r.Method+" "+r.URL.Path)
	u.headers = append(u.headers, r.Header.Clone())
	u.mu.Unlock()
	u.answer(w, r, body)
}

func (u *headerUpstream) take() ([]string, []http.Header) {
	u.mu.Lock()
	defer u.mu.Unlock()
	paths, headers := u.paths, u.headers
	u.paths, u.headers = nil, nil
	return paths, headers
}

const headerFixtureCompletion = `{"id":"chatcmpl-fixture","object":"chat.completion","model":"fixture-model","choices":[{"index":0,` +
	`"message":{"role":"assistant","content":"ok","audio":{"data":"UklGRg==","format":"wav"}},"finish_reason":"stop"}]}`

// answerHeaderFixture answers every operation the providers under test
// perform with the smallest answer each accepts.
func answerHeaderFixture(w http.ResponseWriter, r *http.Request, body []byte) {
	stream := strings.Contains(string(body), `"stream":true`)
	path := r.URL.Path
	switch {
	case strings.HasPrefix(path, "/v1/messages") || path == "/v1/models":
		answerAnthropic(w, r, body)
	case strings.HasPrefix(path, "/openai/"):
		answerAzure(w, r, body)
	case path == "/api/chat" && stream:
		_, _ = io.WriteString(w, `{"message":{"role":"assistant","content":"ok"},"done":false}`+"\n"+`{"done":true}`+"\n")
	case path == "/api/chat":
		_, _ = io.WriteString(w, `{"model":"fixture-model","message":{"role":"assistant","content":"ok"},"done":true}`)
	case path == "/api/tags":
		_, _ = io.WriteString(w, `{"models":[{"name":"fixture-model"}]}`)
	case strings.HasSuffix(path, ":generateContent"):
		_, _ = io.WriteString(w, googleHelloAnswer)
	case strings.HasSuffix(path, ":embedContent"):
		_, _ = io.WriteString(w, `{"embedding":{"values":[0.5]}}`)
	case path == "/models" && r.URL.Query().Has("pageSize"):
		_, _ = io.WriteString(w, `{"models":[{"name":"models/gemini-fixture","supportedGenerationMethods":["generateContent"]}]}`)
	case strings.HasSuffix(path, "/models"):
		_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"fixture-model","object":"model"}]}`)
	case strings.HasSuffix(path, "/chat/completions") && stream:
		w.Header().Set("Content-Type", core.ContentTypeEventStream)
		_, _ = io.WriteString(w, `data: {"id":"chatcmpl-fixture","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"ok"}}]}`+"\n\ndata: [DONE]\n\n")
	case strings.HasSuffix(path, "/chat/completions"):
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		_, _ = io.WriteString(w, headerFixtureCompletion)
	case path == "/speech-to-text":
		_, _ = io.WriteString(w, `{"text":"ok"}`)
	case strings.HasPrefix(path, "/text-to-speech/"):
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = io.WriteString(w, "ID3")
	default:
		http.NotFound(w, r)
	}
}

// drainStream reads a stream to its end.
func drainStream(stream core.StreamIter, err error) error {
	if err != nil {
		return err
	}
	defer stream.Close()
	for {
		if _, err := stream.Next(); err == io.EOF {
			return nil
		} else if err != nil {
			return err
		}
	}
}

func chatRequest(model string, credential *core.Credential) core.Request {
	return core.Request{
		Surface: core.ModelSurfaceChatCompletions, Model: model, ContentType: core.ContentTypeJSON,
		Body: []byte(`{"messages":[{"role":"user","content":"hi"}]}`), Credential: credential,
	}
}

// extraHeaderProvider builds one provider with headers against base and
// performs every operation that sends a request, each with credential.
type extraHeaderProvider func(t *testing.T, base string, client *http.Client, headers map[string]string, credential *core.Credential) error

var extraHeaderProviders = map[string]extraHeaderProvider{
	"openai_compatible": func(t *testing.T, base string, client *http.Client, headers map[string]string, credential *core.Credential) error {
		provider, err := NewOpenAICompatible(OpenAICompatibleConfig{BaseURL: base + "/v1", Client: client, CatalogClient: client, Headers: headers})
		if err != nil {
			return err
		}
		ctx := t.Context()
		if _, err := provider.Invoke(ctx, chatRequest("fixture-model", credential)); err != nil {
			return err
		}
		if err := drainStream(provider.Stream(ctx, chatRequest("fixture-model", credential))); err != nil {
			return err
		}
		_, err = provider.ListModels(ctx, credential)
		return err
	},
	"bedrock": func(t *testing.T, base string, client *http.Client, headers map[string]string, credential *core.Credential) error {
		provider, err := NewBedrock("", base+"/v1", OpenAICompatibleConfig{Client: client, Headers: headers})
		if err != nil {
			return err
		}
		_, err = provider.Invoke(t.Context(), chatRequest("fixture-model", credential))
		return err
	},
	"anthropic": func(t *testing.T, base string, client *http.Client, headers map[string]string, credential *core.Credential) error {
		provider, err := NewAnthropic(AnthropicConfig{BaseURL: base, Client: client, CatalogClient: client, Headers: headers})
		if err != nil {
			return err
		}
		ctx, messages := t.Context(), anthropicMessagesRequest(`{"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`, credential)
		if _, err := provider.Invoke(ctx, messages); err != nil {
			return err
		}
		if err := drainStream(provider.Stream(ctx, messages)); err != nil {
			return err
		}
		if _, err := provider.CountTokens(ctx, core.TokenCountRequest{Request: messages}); err != nil {
			return err
		}
		_, err = provider.ListModels(ctx, credential)
		return err
	},
	"azure_openai": func(t *testing.T, base string, client *http.Client, headers map[string]string, credential *core.Credential) error {
		provider, err := NewAzureOpenAI(AzureOpenAIConfig{BaseURL: base, Client: client, CatalogClient: client, Headers: headers})
		if err != nil {
			return err
		}
		ctx := t.Context()
		if _, err := provider.Invoke(ctx, azureChatRequest(`{"messages":[{"role":"user","content":"hi"}]}`, credential)); err != nil {
			return err
		}
		if err := drainStream(provider.Stream(ctx, azureChatRequest(`{"messages":[{"role":"user","content":"hi"}]}`, credential))); err != nil {
			return err
		}
		_, err = provider.ListModels(ctx, credential)
		return err
	},
	"ai_studio": func(t *testing.T, base string, client *http.Client, headers map[string]string, credential *core.Credential) error {
		provider, err := NewGoogle(GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base, Client: client, Headers: headers})
		if err != nil {
			return err
		}
		ctx := t.Context()
		if _, err := provider.Invoke(ctx, chatRequest("gemini-fixture", credential)); err != nil {
			return err
		}
		embeddings := core.Request{
			Surface: core.ModelSurfaceEmbeddings, Model: "gemini-embedding-fixture", ContentType: core.ContentTypeJSON,
			Body: []byte(`{"input":"hi"}`), Credential: credential,
		}
		if _, err := provider.Invoke(ctx, embeddings); err != nil {
			return err
		}
		if credential == nil {
			return nil // the catalog needs a key
		}
		_, err = provider.ListModels(ctx, credential)
		return err
	},
	"vertex_ai": func(t *testing.T, base string, client *http.Client, headers map[string]string, credential *core.Credential) error {
		provider, err := NewGoogle(GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Project: "fixture-project", Client: client, Headers: headers})
		if err != nil {
			return err
		}
		_, err = provider.Invoke(t.Context(), chatRequest("gemini-fixture", credential))
		return err
	},
	"ollama": func(t *testing.T, base string, client *http.Client, headers map[string]string, credential *core.Credential) error {
		provider, err := NewOllama(OllamaConfig{BaseURL: base, Client: client, CatalogClient: client, Headers: headers})
		if err != nil {
			return err
		}
		ctx := t.Context()
		if _, err := provider.Invoke(ctx, chatRequest("fixture-model", credential)); err != nil {
			return err
		}
		if err := drainStream(provider.Stream(ctx, chatRequest("fixture-model", credential))); err != nil {
			return err
		}
		_, err = provider.ListModels(ctx, credential)
		return err
	},
	"elevenlabs": func(t *testing.T, base string, client *http.Client, headers map[string]string, credential *core.Credential) error {
		provider, err := NewElevenLabs(ElevenLabsConfig{
			BaseURL: base, Client: client, Headers: headers,
			TranscriptionModels: []string{"scribe-fixture"}, SpeechModels: []string{"speech-fixture"},
		})
		if err != nil {
			return err
		}
		speech := speechRequest("speech-fixture", `{"input":"hi","voice":"fixture-voice"}`)
		speech.Credential = credential
		if _, err := provider.Invoke(t.Context(), speech); err != nil {
			return err
		}
		transcription := transcriptionRequest(t, "scribe-fixture", "fixture.wav", []byte("RIFF"), nil)
		transcription.Credential = credential
		_, err = provider.Invoke(t.Context(), transcription)
		return err
	},
	"mimo": func(t *testing.T, base string, client *http.Client, headers map[string]string, credential *core.Credential) error {
		provider, err := NewMiMo(MiMoConfig{BaseURL: base, Client: client, Headers: headers, SpeechModels: []string{"speech-fixture"}})
		if err != nil {
			return err
		}
		speech := speechRequest("speech-fixture", `{"input":"hi"}`)
		speech.Credential = credential
		_, err = provider.Invoke(t.Context(), speech)
		return err
	},
}

// Every provider sends its configured headers with every request it makes,
// and keeps its own: none replaces a header the provider sets, and with a
// credential no configured header carries another, neither Authorization
// nor the provider's own key header.
func TestExtraHeadersReachTheUpstreamWithoutReplacingTheProvidersOwn(t *testing.T) {
	t.Parallel()
	for name, check := range map[string]struct {
		credential *core.Credential
		// key is the header the provider authenticates with and its value.
		key, value string
		// others are further credential headers that must not be sent.
		others []string
		// requests is how many requests the provider's operations send.
		requests int
	}{
		"openai_compatible": {credential: &core.Credential{APIKey: "fixture-key"}, key: "Authorization", value: "Bearer fixture-key", requests: 3},
		"bedrock":           {credential: &core.Credential{APIKey: "fixture-key"}, key: "Authorization", value: "Bearer fixture-key", requests: 1},
		"anthropic":         {credential: anthropicAPIKey("fixture-key"), key: "x-api-key", value: "fixture-key", requests: 4},
		"azure_openai":      {credential: azureKey("fixture-key"), key: "api-key", value: "fixture-key", requests: 3},
		"ai_studio":         {credential: googleKey("fixture-key"), key: "x-goog-api-key", value: "fixture-key", requests: 3},
		"vertex_ai": {
			credential: googleBearer("fixture-token"), key: "Authorization", value: "Bearer fixture-token",
			others: []string{"x-goog-api-key"}, requests: 1,
		},
		"elevenlabs": {credential: &core.Credential{APIKey: "fixture-key"}, key: "xi-api-key", value: "fixture-key", requests: 2},
		"mimo":       {credential: &core.Credential{APIKey: "fixture-key"}, key: "Api-Key", value: "fixture-key", requests: 1},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			upstream, server := newHeaderUpstream(t, answerHeaderFixture)
			configured := map[string]string{
				"X-Fixture-Tag": "configured", "content-type": "text/plain",
				"Authorization": "Bearer configured", check.key: "configured",
			}
			for _, other := range check.others {
				configured[other] = "configured"
			}
			if err := extraHeaderProviders[name](t, server.URL, server.Client(), configured, check.credential); err != nil {
				t.Fatal(err)
			}
			paths, headers := upstream.take()
			if len(headers) != check.requests {
				t.Fatalf("the upstream got %d requests (%v), want %d", len(headers), paths, check.requests)
			}
			for index, header := range headers {
				if got := header.Get("X-Fixture-Tag"); got != "configured" {
					t.Errorf("%s: X-Fixture-Tag = %q, want the configured header", paths[index], got)
				}
				// A catalog's GET has no body, and some providers set no
				// Content-Type on it to replace.
				if got := header.Values("Content-Type"); strings.HasPrefix(paths[index], http.MethodPost) &&
					(len(got) != 1 || strings.Contains(got[0], "text/plain")) {
					t.Errorf("%s: Content-Type = %q, want the provider's own", paths[index], got)
				}
				if got := header.Values(check.key); len(got) != 1 || got[0] != check.value {
					t.Errorf("%s: %s = %q, want the credential's %q alone", paths[index], check.key, got, check.value)
				}
				if !strings.EqualFold(check.key, "Authorization") && header.Get("Authorization") != "" {
					t.Errorf("%s: a configured Authorization was sent beside %s", paths[index], check.key)
				}
				for _, other := range check.others {
					if header.Get(other) != "" {
						t.Errorf("%s: a configured %s was sent beside %s", paths[index], other, check.key)
					}
				}
			}
		})
	}
}

// A request a provider sends without a credential carries the configured
// headers as configured, Authorization included, for an upstream behind a
// proxy that authenticates by header. Ollama never authenticates.
func TestExtraHeadersAuthenticateARequestWithoutACredential(t *testing.T) {
	t.Parallel()
	for name, requests := range map[string]int{"openai_compatible": 3, "anthropic": 4, "ai_studio": 2, "ollama": 3} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			upstream, server := newHeaderUpstream(t, answerHeaderFixture)
			configured := map[string]string{"X-Fixture-Tag": "configured", "Authorization": "Bearer proxy-fixture"}
			if err := extraHeaderProviders[name](t, server.URL, server.Client(), configured, nil); err != nil {
				t.Fatal(err)
			}
			paths, headers := upstream.take()
			if len(headers) != requests {
				t.Fatalf("the upstream got %d requests (%v), want %d", len(headers), paths, requests)
			}
			for index, header := range headers {
				if header.Get("X-Fixture-Tag") != "configured" || header.Get("Authorization") != "Bearer proxy-fixture" {
					t.Errorf("%s: headers = %v, want the configured ones", paths[index], header)
				}
			}
		})
	}
}

// A header Go's transport would refuse to send fails every request as if
// the upstream were out of reach, so every constructor refuses it as a
// configuration error, without quoting its value.
func TestProviderConstructorsRefuseHeadersTheTransportCannotSend(t *testing.T) {
	t.Parallel()
	constructors := map[string]func(map[string]string) error{
		"openai_compatible": func(headers map[string]string) error {
			_, err := NewOpenAICompatible(OpenAICompatibleConfig{BaseURL: "https://upstream.invalid/v1", Headers: headers})
			return err
		},
		"bedrock": func(headers map[string]string) error {
			_, err := NewBedrock("us-east-1", "", OpenAICompatibleConfig{Headers: headers})
			return err
		},
		"anthropic": func(headers map[string]string) error {
			_, err := NewAnthropic(AnthropicConfig{Headers: headers})
			return err
		},
		"azure_openai": func(headers map[string]string) error {
			_, err := NewAzureOpenAI(AzureOpenAIConfig{BaseURL: "https://fixture.openai.azure.com", Headers: headers})
			return err
		},
		"google": func(headers map[string]string) error {
			_, err := NewGoogle(GoogleConfig{Deployment: GoogleAIStudio, Headers: headers})
			return err
		},
		"ollama": func(headers map[string]string) error {
			_, err := NewOllama(OllamaConfig{Headers: headers})
			return err
		},
		"elevenlabs": func(headers map[string]string) error {
			_, err := NewElevenLabs(ElevenLabsConfig{Headers: headers})
			return err
		},
		"mimo": func(headers map[string]string) error {
			_, err := NewMiMo(MiMoConfig{Headers: headers})
			return err
		},
	}
	refused := map[string]map[string]string{
		"blank name":            {"": "fixture-secret"},
		"space in name":         {"X Fixture": "fixture-secret"},
		"colon in name":         {"X-Fixture:": "fixture-secret"},
		"line break in value":   {"X-Fixture": "fixture-secret\r\nInjected: yes"},
		"NUL in value":          {"X-Fixture": "fixture-secret\x00"},
		"DEL in value":          {"X-Fixture": "fixture-secret\x7f"},
		"case-folded conflicts": {"x-fixture": "fixture-secret", "X-Fixture": "fixture-secret-2"},
	}
	accepted := map[string]map[string]string{
		"none":                {},
		"tab and non-ASCII":   {"X-Fixture": "tab\there, ünïcode"},
		"case-folded repeats": {"x-fixture": "same", "X-Fixture": "same"},
		"token punctuation":   {"X-Fixture_!#$%&'*+.^`|~": "value"},
	}
	for provider, construct := range constructors {
		for name, headers := range refused {
			err := construct(headers)
			var failure *core.ProviderError
			if !errors.As(err, &failure) || failure.Class != core.ProviderErrorConfiguration {
				t.Errorf("%s, %s: error = %v, want a configuration error", provider, name, err)
				continue
			}
			if strings.Contains(err.Error(), "fixture-secret") {
				t.Errorf("%s, %s: the error quotes the value: %v", provider, name, err)
			}
		}
		for name, headers := range accepted {
			if err := construct(headers); err != nil {
				t.Errorf("%s, %s: %v", provider, name, err)
			}
		}
	}
}

// A provider keeps its own copy of the configured headers, so a product
// that reuses its map cannot change what an existing provider sends.
func TestExtraHeadersAreCopiedWhenTheProviderIsBuilt(t *testing.T) {
	t.Parallel()
	upstream, server := newHeaderUpstream(t, answerHeaderFixture)
	headers := map[string]string{"X-Fixture-Tag": "configured"}
	provider, err := NewOllama(OllamaConfig{BaseURL: server.URL, Client: server.Client(), Headers: headers})
	if err != nil {
		t.Fatal(err)
	}
	headers["X-Fixture-Tag"], headers["X-Later"] = "changed", "added"
	if _, err := provider.Invoke(context.Background(), chatRequest("fixture-model", nil)); err != nil {
		t.Fatal(err)
	}
	_, sent := upstream.take()
	if len(sent) != 1 || sent[0].Get("X-Fixture-Tag") != "configured" || sent[0].Get("X-Later") != "" {
		t.Fatalf("headers = %v, want the ones the provider was built with", sent)
	}
}
