package providers

import (
	"context"
	"errors"
	"io"
	"net/http"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

func TestPreparedActualProviderDescriptions(t *testing.T) {
	t.Parallel()
	_, base := newGoogleFake(t, googleAnswer(200, googleHelloAnswer))
	studio := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	vertex := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, Project: "fixture-project", BaseURL: base})
	ollama, _ := ollamaDaemon(t, 200, `{"message":{"content":"ok"},"done":true}`)
	_, azureServer := newAzureBackend(t, answerAzure)
	azure := newTestAzure(t, azureServer)
	_, anthropicServer := newAnthropicBackend(t, answerAnthropic)
	anthropic := newTestAnthropic(t, anthropicServer)
	compatible, err := NewOpenAICompatible(OpenAICompatibleConfig{BaseURL: base})
	if err != nil {
		t.Fatal(err)
	}
	bedrock, err := NewBedrock("", base, OpenAICompatibleConfig{})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name    string
		p       core.Provider
		surface core.ModelSurface
		api     core.UpstreamAPI
		wire    core.Support
		adapted bool
	}{
		{"studio", studio, core.ModelSurfaceChatCompletions, core.UpstreamAPIGoogleGenerateContent, core.SupportUnsupported, true},
		{"vertex", vertex, core.ModelSurfaceChatCompletions, core.UpstreamAPIGoogleGenerateContent, core.SupportUnsupported, true},
		{"ollama", ollama, core.ModelSurfaceChatCompletions, core.UpstreamAPIOllamaChat, core.SupportUnsupported, true},
		{"azure", azure, core.ModelSurfaceChatCompletions, core.UpstreamAPIChatCompletions, core.SupportUnsupported, false},
		{"anthropic", anthropic, core.ModelSurfaceMessages, core.UpstreamAPIMessages, core.SupportSupported, false},
		{"compatible", compatible, core.ModelSurfaceChatCompletions, core.UpstreamAPIChatCompletions, core.SupportSupported, false},
		{"bedrock", bedrock, core.ModelSurfaceChatCompletions, core.UpstreamAPIChatCompletions, core.SupportSupported, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := googleChatRequest("m", `{"messages":[{"role":"user","content":"hello"}],"max_tokens":32}`, googleKey("fixture"))
			r.Surface = tc.surface
			call, err := core.PrepareCall(t.Context(), tc.p, r, core.CallPreparationOptions{Mode: core.CallModeInvoke, AllowAdaptation: true})
			if err != nil {
				t.Fatal(err)
			}
			d := call.Description()
			if d.UpstreamAPI != tc.api || d.WirePreserved != tc.wire {
				t.Fatalf("actual contract=%+v", d)
			}
			_, err = core.PlanPreparedCall(d, core.CallPolicy{})
			if tc.adapted != (err != nil) {
				t.Fatalf("native policy err=%v adapted=%t", err, tc.adapted)
			}
			if _, err := core.PlanPreparedCall(d, core.CallPolicy{AllowAdaptation: true}); err != nil {
				t.Fatalf("ordinary text unassessed: %v (%+v)", err, d)
			}
		})
	}
}

func TestPreparedCompatibleExecutesFrozenNativeFieldsAndSelection(t *testing.T) {
	backend, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	})
	provider, err := NewOpenAICompatible(OpenAICompatibleConfig{BaseURL: server.URL, ForwardAllFields: true, ForceAPISupport: true, Models: openAIResponsesOnly})
	if err != nil {
		t.Fatal(err)
	}
	request := googleChatRequest("responses-fixture", `{"messages":[{"role":"user","content":"hi"}],"vendor_number":12345678901234567890,"force_api_support":true}`, nil)
	prepared, err := core.PrepareCall(t.Context(), provider, request, core.CallPreparationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(backend.take()) != 0 {
		t.Fatal("preparation called upstream")
	}
	if d := prepared.Description(); d.UpstreamAPI != core.UpstreamAPIChatCompletions || len(d.Conversions) != 0 {
		t.Fatalf("force defeated native-only preparation: %+v", d)
	}
	for i := range request.Body {
		request.Body[i] = 'x'
	}
	if _, err := prepared.Invoke(t.Context(), nil); err != nil {
		t.Fatal(err)
	}
	requests := backend.take()
	if len(requests) != 1 || requests[0].path != "/chat/completions" || !strings.Contains(requests[0].body, `"vendor_number":12345678901234567890`) {
		t.Fatalf("frozen native request changed: %v", requests)
	}
}

func TestPreparedCompatibleDoesNotStripRejectedControls(t *testing.T) {
	backend, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		w.WriteHeader(400)
		_, _ = io.WriteString(w, `{"error":{"message":"temperature unsupported"}}`)
	})
	provider, err := NewOpenAICompatible(OpenAICompatibleConfig{BaseURL: server.URL, ForwardAllFields: true, ForceAPISupport: true, Models: openAIResponsesOnly})
	if err != nil {
		t.Fatal(err)
	}
	request := googleChatRequest("responses-fixture", `{"messages":[{"role":"user","content":"hi"}],"temperature":0.5}`, nil)
	prepared, err := core.PrepareCall(t.Context(), provider, request, core.CallPreparationOptions{AllowAdaptation: true})
	if err != nil {
		t.Fatal(err)
	}
	if d := prepared.Description(); d.UpstreamAPI != core.UpstreamAPIResponses {
		t.Fatalf("missing chosen response API: %+v", d)
	}
	if _, err := core.PlanPreparedCall(prepared.Description(), core.CallPolicy{AllowAdaptation: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := prepared.Invoke(t.Context(), nil); err == nil {
		t.Fatal("rejected temperature silently repaired")
	}
	if calls := backend.take(); len(calls) != 1 {
		t.Fatalf("hidden retry: %d calls", len(calls))
	}
}

func TestPreparedGoogleRejectsUnassessedNestedToolFields(t *testing.T) {
	for _, tc := range []struct{ body, path string }{
		{`{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"},"unreviewed_control":true}}]}`, "tools.0.function.unreviewed_control"},
		{`{"messages":[{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}","unreviewed_control":true}}]}]}`, "messages.0.tool_calls.0.function.unreviewed_control"},
		{`{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","name description":true}}]}`, "tools.0.function.name description"},
	} {
		fake, base := newGoogleFake(t, googleAnswer(200, googleHelloAnswer))
		p := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
		call, err := core.PrepareCall(t.Context(), p, googleChatRequest("m", tc.body, googleKey("fixture")), core.CallPreparationOptions{AllowAdaptation: true})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := core.PlanPreparedCall(call.Description(), core.CallPolicy{AllowAdaptation: true}); err == nil {
			t.Fatalf("unassessed %s admitted: %+v", tc.path, call.Description())
		}
		if !slices.Contains(call.Description().Assessment.UnknownPaths, tc.path) {
			t.Fatalf("missing unknown path %s: %+v", tc.path, call.Description())
		}
		if _, err := call.Invoke(t.Context(), googleKey("fixture")); err == nil {
			t.Fatal("direct prepared execution bypassed unsupported scope")
		}
		if calls := fake.take(); len(calls) != 0 {
			t.Fatal("unassessed controls reached inference")
		}
	}
}

func TestPreparedGoogleKeepsSupportedSignaturesAndSchemaPropertyData(t *testing.T) {
	fake, base := newGoogleFake(t, googleAnswer(200, googleHelloAnswer))
	p := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	for _, signature := range []string{
		`"extra_content":{"google":{"thought_signature":"fixture"}},"function":{"name":"f","arguments":"{}"}`,
		`"function":{"name":"f","arguments":"{}","thought_signature":"fixture"}`,
		`"thought_signature":"fixture","function":{"name":"f","arguments":"{}"}`,
	} {
		body := `{"messages":[{"role":"assistant","tool_calls":[{"id":"c1","type":"function",` + signature + `}]},{"role":"tool","tool_call_id":"c1","content":"ok"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"name description":{"type":"string"},"unreviewed_control":{"type":"boolean"}}}}}]}`
		call, err := core.PrepareCall(t.Context(), p, googleChatRequest("m", body, googleKey("fixture")), core.CallPreparationOptions{AllowAdaptation: true})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := core.PlanPreparedCall(call.Description(), core.CallPolicy{AllowAdaptation: true}); err != nil {
			t.Fatalf("valid signed data rejected: %v", err)
		}
		if _, err := call.Invoke(t.Context(), googleKey("fixture")); err != nil {
			t.Fatal(err)
		}
		calls := fake.take()
		if len(calls) != 1 || !strings.Contains(calls[0].body, `"thoughtSignature":"fixture"`) || !strings.Contains(calls[0].body, `"name description":{"type":"string"}`) {
			t.Fatalf("mapped body=%+v", calls)
		}
	}
}

func TestPreparedResponsesRefusalDoesNotFallback(t *testing.T) {
	for _, status := range []int{404, 405} {
		backend, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
			w.WriteHeader(status)
			_, _ = io.WriteString(w, `{"error":"not found"}`)
		})
		p := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) { config.RegistryID = "openai" })
		for _, mode := range []core.CallMode{core.CallModeInvoke, core.CallModeStream} {
			r := openAIRequest(core.ModelSurfaceResponses, "m", `{"input":"hello"}`, nil)
			call, err := core.PrepareCall(t.Context(), p, r, core.CallPreparationOptions{Mode: mode, AllowAdaptation: true})
			if err != nil {
				t.Fatal(err)
			}
			if mode == core.CallModeStream {
				_, err = call.Stream(t.Context(), nil)
			} else {
				_, err = call.Invoke(t.Context(), nil)
			}
			var surface *core.SurfaceError
			if !errors.As(err, &surface) {
				t.Fatalf("404/405 lost native refusal: %v", err)
			}
			if got := core.ClassifyError(err).StatusCode; got != status {
				t.Fatalf("refusal status=%d want=%d", got, status)
			}
			calls := backend.take()
			if len(calls) != 1 || calls[0].path != "/v1/responses" {
				t.Fatalf("hidden fallback=%+v", calls)
			}
		}
	}
}

func TestPreparedBufferedResponseStreamIsNotIncremental(t *testing.T) {
	backend, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		_, _ = io.WriteString(w, `{"id":"resp_fixture","object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`)
	})
	p := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) {
		config.ForceAPISupport = true
		config.Models = func(model string) (core.ModelInfo, bool) {
			row, known := openAIResponsesOnly(model)
			row.Capabilities = &core.ModelCapabilities{SchemaVersion: core.ModelCapabilitiesSchemaVersion, Streaming: core.SupportUnsupported}
			return row, known
		}
	})
	call, err := core.PrepareCall(t.Context(), p, googleChatRequest("responses-fixture", `{"messages":[{"role":"user","content":"hello"}]}`, nil), core.CallPreparationOptions{Mode: core.CallModeStream, AllowAdaptation: true})
	if err != nil {
		t.Fatal(err)
	}
	d := call.Description()
	if d.UpstreamTransport != core.CallTransportJSON || d.OutputTransport != core.CallTransportBufferedSSE {
		t.Fatalf("transport description=%+v", d)
	}
	if _, err := core.PlanPreparedCall(d, core.CallPolicy{AllowAdaptation: true, RequireIncremental: true}); err == nil {
		t.Fatal("buffered replay claimed incremental")
	}
	if len(backend.take()) != 0 {
		t.Fatal("planning opened inference")
	}
	stream, err := call.Stream(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var frames strings.Builder
	for {
		frame, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		frames.Write(frame)
	}
	if !strings.Contains(frames.String(), "ok") || !strings.Contains(frames.String(), "[DONE]") {
		t.Fatalf("bad synthetic frames=%s", frames.String())
	}
	if calls := backend.take(); len(calls) != 1 || strings.Contains(calls[0].body, `"stream":true`) {
		t.Fatalf("unexpected upstream streaming=%+v", calls)
	}
}

func TestPreparedGoogleRejectsUnrepresentableResponseState(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		wire := `{"candidates":[{"content":{"parts":[{"text":"affected","thoughtSignature":"opaque-fixture"}]}}]}`
		var p *Google
		var callCount func() int
		if streaming {
			take, base := newGoogleStreamFake(t, 200, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hello\"}]}}]}\n\ndata: "+wire+"\n\ndata: {\"candidates\":[{\"finishReason\":\"STOP\"}]}\n\n")
			p = newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
			callCount = func() int { return len(take()) }
		} else {
			fake, base := newGoogleFake(t, googleAnswer(200, wire))
			p = newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
			callCount = func() int { return len(fake.take()) }
		}
		options := core.CallPreparationOptions{AllowAdaptation: true}
		if streaming {
			options.Mode = core.CallModeStream
		}
		call, err := core.PrepareCall(t.Context(), p, googleChatRequest("m", `{"messages":[{"role":"user","content":"hi"}]}`, googleKey("fixture")), options)
		if err != nil {
			t.Fatal(err)
		}
		if streaming {
			stream, err := call.Stream(t.Context(), googleKey("fixture"))
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			var delivered strings.Builder
			for {
				frame, nextErr := stream.Next()
				if nextErr != nil {
					err = nextErr
					break
				}
				delivered.Write(frame)
			}
			if err == nil || err == io.EOF || strings.Contains(delivered.String(), "affected") || strings.Contains(delivered.String(), "[DONE]") {
				t.Fatalf("signed state silently dropped: %s err=%v", delivered.String(), err)
			}
			if len(core.StreamLosses(stream)) == 0 {
				t.Fatal("stream state refusal lost evidence")
			}
		} else {
			response, err := call.Invoke(t.Context(), googleKey("fixture"))
			if err == nil || len(response.Losses) == 0 {
				t.Fatalf("signed response silently mapped: %s %v", response.Body, err)
			}
		}
		if callCount() != 1 {
			t.Fatal("response refusal replayed inference")
		}
	}
}

func TestPreparedGoogleResponseKeepsExactToolArgumentsAndThinkingUsage(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		wire := `{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{"n":9007199254740993}},"thoughtSignature":"fixture"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3,"thoughtsTokenCount":29,"totalTokenCount":39}}`
		var p *Google
		if streaming {
			_, base := newGoogleStreamFake(t, 200, "data: "+wire+"\n\n")
			p = newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
		} else {
			_, base := newGoogleFake(t, googleAnswer(200, wire))
			p = newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
		}
		options := core.CallPreparationOptions{AllowAdaptation: true}
		if streaming {
			options.Mode = core.CallModeStream
		}
		body := `{"messages":[{"role":"user","content":"hi"}]}`
		if streaming {
			body = `{"messages":[{"role":"user","content":"hi"}],"stream_options":{"include_usage":true}}`
		}
		call, err := core.PrepareCall(t.Context(), p, googleChatRequest("m", body, googleKey("fixture")), options)
		if err != nil {
			t.Fatal(err)
		}
		var delivered strings.Builder
		if streaming {
			stream, err := call.Stream(t.Context(), googleKey("fixture"))
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			for {
				frame, err := stream.Next()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatal(err)
				}
				delivered.Write(frame)
			}
		} else {
			response, err := call.Invoke(t.Context(), googleKey("fixture"))
			if err != nil {
				t.Fatal(err)
			}
			delivered.Write(response.Body)
		}
		if !strings.Contains(delivered.String(), "9007199254740993") || !strings.Contains(delivered.String(), `"completion_tokens":32`) {
			t.Fatalf("precision/thinking accounting changed: %s", delivered.String())
		}
	}
}

func TestPreparedAnthropicFreezesPreambleProtocolAndNativeNumbers(t *testing.T) {
	backend, server := newAnthropicBackend(t, answerAnthropic)
	var prepares atomic.Int32
	preamble := "initial"
	p := newTestAnthropic(t, server, func(config *AnthropicConfig) {
		config.Preamble = func(context.Context) string { prepares.Add(1); return preamble }
	})
	request := anthropicMessagesRequest(`{"messages":[{"role":"user","content":"hello"}],"vendor_number":9007199254740993}`, googleKey("fixture"))
	request.Header = http.Header{}
	request.Header.Set("anthropic-version", "2025-01-01")
	call, err := core.PrepareCall(t.Context(), p, request, core.CallPreparationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if call.Description().APIVersion != "2025-01-01" {
		t.Fatalf("protocol description=%+v", call.Description())
	}
	preamble = "changed"
	for index := range request.Body {
		request.Body[index] = 'x'
	}
	request.Header.Set("anthropic-version", "2026-01-01")
	if _, err := call.Invoke(t.Context(), googleKey("fixture")); err != nil {
		t.Fatal(err)
	}
	calls := backend.take()
	if len(calls) != 1 || prepares.Load() != 1 || calls[0].header.Get("anthropic-version") != "2025-01-01" || !strings.Contains(calls[0].body, `"vendor_number":9007199254740993`) || !strings.Contains(calls[0].body, `"system":"initial"`) {
		t.Fatalf("native preparation re-evaluated: %+v prepares=%d", calls, prepares.Load())
	}
}

func TestPreparedCompatibleSelectedSurfaceIgnoresExpiredDynamicDenial(t *testing.T) {
	now := time.Unix(200, 0)
	past, future := now.Add(-time.Second), now.Add(time.Second)
	for _, tc := range []struct {
		name    string
		source  core.ModelCapabilitySource
		expires *time.Time
		allowed bool
	}{
		{"expired dynamic", core.ModelCapabilitySourceUpstreamReported, &past, true},
		{"fresh dynamic", core.ModelCapabilitySourceUpstreamReported, &future, false},
		{"operator declaration", core.ModelCapabilitySourceOperatorOverride, &past, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend, server := newOpenAIBackend(t, answerOpenAIResponses)
			provider := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) {
				config.Now = func() time.Time { return now }
				config.Models = func(model string) (core.ModelInfo, bool) {
					return core.ModelInfo{ID: model, SupportedAPIs: []string{"/responses"}, Capabilities: &core.ModelCapabilities{SchemaVersion: core.ModelCapabilitiesSchemaVersion, Surfaces: core.ModelSurfaceCapabilities{Responses: core.SupportUnsupported}, Provenance: core.ModelCapabilityProvenance{Source: tc.source}, Freshness: core.ModelCapabilityFreshness{ExpiresAt: tc.expires}}}, true
				}
			})
			request := openAIRequest(core.ModelSurfaceResponses, "fixture", `{"input":"hello"}`, nil)
			call, err := core.PrepareCall(t.Context(), provider, request, core.CallPreparationOptions{})
			if err != nil {
				t.Fatal(err)
			}
			_, err = core.PlanPreparedCall(call.Description(), core.CallPolicy{})
			if (err == nil) != tc.allowed {
				t.Fatalf("allowed=%t err=%v", tc.allowed, err)
			}
			if calls := backend.take(); len(calls) != 0 {
				t.Fatal("expiry assessment performed inference")
			}
		})
	}
}

func TestPreparedGoogleSchemaLabelsAreAdvisoryWithoutWeakeningConstraints(t *testing.T) {
	fake, base := newGoogleFake(t, googleAnswer(200, googleHelloAnswer))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	for _, strict := range []string{"", `,"strict":true`, `,"strict":"true"`} {
		body := `{"messages":[{"role":"user","content":"JSON"}],"response_format":{"type":"json_schema","json_schema":{"name":"verdict","description":"label","schema":{"type":"object","properties":{"ok":{"type":"boolean"}}}` + strict + `}}}`
		call, err := core.PrepareCall(t.Context(), provider, googleChatRequest("m", body, googleKey("fixture")), core.CallPreparationOptions{AllowAdaptation: true})
		if err != nil {
			t.Fatal(err)
		}
		_, err = core.PlanPreparedCall(call.Description(), core.CallPolicy{AllowAdaptation: true})
		if strict != "" {
			if err == nil || !strings.Contains(err.Error(), "response_format.json_schema.strict") {
				t.Fatalf("strict guarantee/type silently weakened: %v", err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("valid schema label rejected: %v", err)
		}
		response, err := call.Invoke(t.Context(), googleKey("fixture"))
		if err != nil {
			t.Fatal(err)
		}
		if len(response.Losses) != 2 || response.Losses[0].Severity != translate.LossAdvisory || response.Losses[1].Severity != translate.LossAdvisory {
			t.Fatalf("label observations lost: %v", response.Losses)
		}
		calls := fake.take()
		if len(calls) != 1 || !strings.Contains(calls[0].body, `"responseJsonSchema":{"properties":{"ok":{"type":"boolean"}},"type":"object"}`) {
			t.Fatalf("schema constraints changed: %+v", calls)
		}
	}
}

func TestPreparedGoogleInvalidToolChoiceDiagnosticRetainsSafePath(t *testing.T) {
	_, base := newGoogleFake(t, googleAnswer(200, googleHelloAnswer))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	call, err := core.PrepareCall(t.Context(), provider, googleChatRequest("m", `{"messages":[{"role":"user","content":"hi"}],"tool_choice":"required"}`, googleKey("fixture")), core.CallPreparationOptions{AllowAdaptation: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := core.PlanPreparedCall(call.Description(), core.CallPolicy{AllowAdaptation: true}); err == nil || !strings.Contains(err.Error(), "tool_choice") {
		t.Fatalf("diagnostic omitted path: %v", err)
	}
}

func TestPreparedChatToResponsesRetainsExplicitNonStrictToolDefault(t *testing.T) {
	backend, server := newOpenAIBackend(t, func(w http.ResponseWriter, _ *http.Request, _ int) {
		_, _ = io.WriteString(w, `{"id":"r1","object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`)
	})
	p := newTestOpenAICompatible(t, server, func(config *OpenAICompatibleConfig) {
		config.ForceAPISupport = true
		config.Models = openAIResponsesOnly
	})
	body := `{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"x":{"type":"string"}}}}}]}`
	call, err := core.PrepareCall(t.Context(), p, googleChatRequest("responses-fixture", body, nil), core.CallPreparationOptions{AllowAdaptation: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := core.PlanPreparedCall(call.Description(), core.CallPolicy{AllowAdaptation: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := call.Invoke(t.Context(), nil); err != nil {
		t.Fatal(err)
	}
	calls := backend.take()
	if len(calls) != 1 || !strings.Contains(calls[0].body, `"strict":false`) || !strings.Contains(calls[0].body, `"properties":{"x":{"type":"string"}}`) {
		t.Fatalf("tool default/schema changed: %+v", calls)
	}
}
