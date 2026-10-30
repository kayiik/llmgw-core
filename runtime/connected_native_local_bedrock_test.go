package runtime_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/providers"
)

// These are Gateway-observed responses, not original upstream HTTP captures.
// Request matching proves the reconstructed native adapter contract only.
func TestConnectedNativeLocalAndBedrockCapture(t *testing.T) {
	raw, err := os.ReadFile("testdata/connected_native_local_bedrock.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Provenance string
		Cases      []struct {
			Name, Model     string
			GatewayRequest  json.RawMessage `json:"gateway_request"`
			ProviderRequest json.RawMessage `json:"provider_request"`
			Response        json.RawMessage
		}
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) != 2 || !strings.Contains(fixture.Provenance, "not upstream HTTP capture") {
		t.Fatal("missing bounded cases or provenance")
	}
	for _, recorded := range fixture.Cases {
		t.Run(recorded.Name, func(t *testing.T) {
			surface, api := core.ModelSurfaceMessages, core.UpstreamAPIMessages
			path, authHeader, authValue := "/api/v1/messages", "X-Api-Key", "synthetic-capture-key"
			if recorded.Name == "bedrock-chat" {
				surface, api = core.ModelSurfaceChatCompletions, core.UpstreamAPIChatCompletions
				path, authHeader, authValue = "/api/v1/chat/completions", "Authorization", "Bearer synthetic-capture-key"
			} else if recorded.Name != "local-messages" {
				t.Fatal("unexpected recorded case")
			}
			var observed struct{ Model string }
			if err := json.Unmarshal(recorded.GatewayRequest, &observed); err != nil ||
				!strings.HasSuffix(observed.Model, "/"+recorded.Model) {
				t.Fatal("Gateway selector no longer associates with native response model")
			}
			replay := &recordedChatReplay{exchanges: []recordedChatExchange{{
				Method: http.MethodPost, Path: path, Request: recorded.ProviderRequest,
				Response: recorded.Response, Status: http.StatusOK, ContentType: core.ContentTypeJSON,
			}}}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get(authHeader) != authValue {
					t.Error("native adapter did not use the synthetic credential")
					w.WriteHeader(http.StatusUnauthorized)
					return
				}
				replay.ServeHTTP(w, r)
			}))
			t.Cleanup(server.Close)
			client := server.Client()
			transport := &recordedChatLoopback{host: server.Listener.Addr().String(), next: client.Transport}
			client.Transport, client.Timeout = transport, 5*time.Second
			client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
			var provider core.Provider
			var err error
			if surface == core.ModelSurfaceMessages {
				provider, err = providers.NewAnthropic(providers.AnthropicConfig{
					BaseURL: server.URL + "/api", Client: client, CatalogClient: client,
				})
			} else {
				provider, err = providers.NewBedrock("us-east-1", server.URL+"/api/v1", providers.OpenAICompatibleConfig{
					Client: client, CatalogClient: client, ForwardAllFields: true,
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			collector := core.NewSourceUsageCollectorFor(core.Target{Provider: recorded.Name, Model: recorded.Model})
			ctx := core.WithSourceUsageCollector(t.Context(), collector)
			// Source usage is recorded by the prepared native operation, not the
			// legacy Provider.Invoke entry point. Keep this on the Gateway path.
			prepared, err := core.PrepareCall(ctx, provider, core.Request{
				Surface: surface, Model: recorded.Model, Body: recorded.GatewayRequest,
				ContentType: core.ContentTypeJSON, Credential: &core.Credential{APIKey: "synthetic-capture-key"},
			}, core.CallPreparationOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if description := prepared.Description(); description.UpstreamAPI != api || len(description.Conversions) != 0 {
				t.Fatalf("expected native prepared operation: %+v", description)
			}
			response, err := prepared.Invoke(ctx, &core.Credential{APIKey: "synthetic-capture-key"})
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(response.Body, recorded.Response) || response.ContentType != core.ContentTypeJSON || len(response.Losses) != 0 {
				t.Fatal("native response changed: text, reasoning, finish/stop fields, nulls, IDs or exact usage")
			}
			observation, ok := collector.SourceUsage()
			if !ok || observation.API != api || observation.Instance != recorded.Name || observation.Model != recorded.Model ||
				observation.Attempt != 1 || observation.Cancelled || observation.Usage.Quality != translate.CompletionComplete {
				t.Fatalf("native source association/completion changed: %+v", observation)
			}
			assertCount := func(name string, count translate.CountObservation, want int64) {
				t.Helper()
				if count.Presence != translate.CountPresent || count.Provenance != translate.UsageReported || count.Value != want {
					t.Fatalf("%s count changed: %+v", name, count)
				}
			}
			if surface == core.ModelSurfaceMessages {
				assertCount("base input", observation.Usage.BaseInputTokens, 34)
				assertCount("output", observation.Usage.OutputTokens, 3)
				assertCount("cache read explicit zero", observation.Usage.CacheReadTokens, 0)
			} else {
				assertCount("input", observation.Usage.InputTokens, 72)
				assertCount("output", observation.Usage.OutputTokens, 31)
				assertCount("total", observation.Usage.TotalTokens, 103)
				if observation.Usage.CacheReadTokens.Presence != translate.CountAbsent || observation.Usage.ReasoningTokens.Presence != translate.CountAbsent {
					t.Fatal("absent Bedrock cache/reasoning token counts must not become zero")
				}
			}
			if calls, accepted, rejected := replay.counts(); calls != 1 || accepted != 1 || rejected != 0 || transport.blocked.Load() != 0 {
				t.Fatalf("expected one exact loopback request; got %d/%d/%d, blocked %d", calls, accepted, rejected, transport.blocked.Load())
			}
		})
	}
}
