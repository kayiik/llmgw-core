package translation_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/providers"
	"github.com/kayiik/llmgw-core/translation"
)

func TestPreparedChatMessagesNoopControlsPreserveCompletionLimit(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Error(err)
		}
		if r.URL.Path != "/v1/messages" || payload["max_tokens"] != float64(64) {
			t.Errorf("native endpoint/limit mismatch: %s %+v", r.URL.Path, payload)
		}
		for _, key := range []string{"n", "logprobs", "response_format", "parallel_tool_calls", "max_completion_tokens"} {
			if _, exists := payload[key]; exists {
				t.Errorf("Chat-only control leaked into Messages: %s", key)
			}
		}
		_, _ = io.WriteString(w, `{"id":"m1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ordinary text"}],"stop_reason":"end_turn","usage":{"input_tokens":11,"output_tokens":7}}`)
	}))
	defer server.Close()
	provider, err := providers.NewAnthropic(providers.AnthropicConfig{BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	body := `{"messages":[{"role":"user","content":"hi"}],"n":1,"logprobs":false,"response_format":{"type":"text"},"parallel_tool_calls":true,"max_completion_tokens":64}`
	request := jsonRequest(core.ModelSurfaceChatCompletions, body)
	collector := core.NewSourceUsageCollector()
	ctx := core.WithSourceUsageCollector(t.Context(), collector)
	call, err := core.PrepareCall(ctx, translation.Adapter{Provider: provider}, request, core.CallPreparationOptions{AllowAdaptation: true})
	if err != nil {
		t.Fatal(err)
	}
	d := call.Description()
	if d.Ingress != core.ModelSurfaceChatCompletions || d.UpstreamAPI != core.UpstreamAPIMessages || len(d.Conversions) != 1 || d.Conversions[0].ID != "chat-to-messages" {
		t.Fatalf("source conversion semantics changed: %+v", d)
	}
	if _, err := core.PlanPreparedCall(d, core.CallPolicy{}); err == nil {
		t.Fatal("native-only default admitted conversion")
	}
	if _, err := core.PlanPreparedCall(d, core.CallPolicy{AllowAdaptation: true}); err != nil {
		t.Fatal(err)
	}
	response, err := call.Invoke(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(response.Body), "ordinary text") || calls.Load() != 1 || string(request.Body) != body {
		t.Fatalf("result/replay/source mutation: %s calls=%d", response.Body, calls.Load())
	}
	o, ok := collector.SourceUsage()
	if !ok || o.Usage.BaseInputTokens.Value != 11 || o.Usage.OutputTokens.Value != 7 {
		t.Fatalf("native usage lost: %+v", o)
	}
}

func TestPreparedChatMessagesNoopNormalizationKeepsRealControlsRefused(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { calls.Add(1); w.WriteHeader(500) }))
	defer server.Close()
	provider, err := providers.NewAnthropic(providers.AnthropicConfig{BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	for _, control := range []string{`"n":2`, `"logprobs":true`, `"response_format":{"type":"json_object"}`, `"response_format":{"type":"json_schema","json_schema":{"name":"f","schema":{"type":"object"},"strict":true}}`, `"response_format":{"type":"text","unreviewed":true}`, `"seed":42`, `"logit_bias":{"1":1}`, `"store":true`, `"service_tier":"priority"`} {
		t.Run(control, func(t *testing.T) {
			call, err := core.PrepareCall(t.Context(), translation.Adapter{Provider: provider}, jsonRequest(core.ModelSurfaceChatCompletions, `{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":64,`+control+`}`), core.CallPreparationOptions{AllowAdaptation: true})
			if err == nil {
				_, err = core.PlanPreparedCall(call.Description(), core.CallPolicy{AllowAdaptation: true})
			}
			if err == nil {
				t.Fatalf("real control silently normalized: %s", control)
			}
		})
	}
	if calls.Load() != 0 {
		t.Fatalf("preflight performed %d inference calls", calls.Load())
	}
}

func TestPreparedNativeChatKeepsNoopControlsOnWire(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Error(err)
		}
		if payload["n"] != float64(1) || payload["logprobs"] != false || payload["parallel_tool_calls"] != true || payload["max_completion_tokens"] != float64(64) || stringMustJSON(t, payload["response_format"]) != `{"type":"text"}` {
			t.Errorf("native controls changed: %+v", payload)
		}
		_, _ = io.WriteString(w, chatCompletion)
	}))
	defer server.Close()
	provider, err := providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{BaseURL: server.URL, ForwardAllFields: true})
	if err != nil {
		t.Fatal(err)
	}
	call, err := core.PrepareCall(t.Context(), translation.Adapter{Provider: provider}, jsonRequest(core.ModelSurfaceChatCompletions, `{"messages":[{"role":"user","content":"hi"}],"n":1,"logprobs":false,"response_format":{"type":"text"},"parallel_tool_calls":true,"max_completion_tokens":64}`), core.CallPreparationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(call.Description().Conversions) != 0 {
		t.Fatal("native call adapted")
	}
	if _, err := call.Invoke(t.Context(), nil); err != nil {
		t.Fatal(err)
	}
}

func TestPreparedResponsesThroughChatToNativeMessagesInvoke(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Error(err)
		}
		if r.URL.Path != "/v1/messages" || payload["max_tokens"] != float64(17) || !strings.Contains(stringMustJSON(t, payload["messages"]), "baseline input") {
			t.Errorf("native endpoint/limits/text mismatch: %s %+v", r.URL.Path, payload)
		}
		_, _ = io.WriteString(w, `{"id":"m1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"baseline output"}],"stop_reason":"end_turn","usage":{"input_tokens":11,"output_tokens":7}}`)
	}))
	defer server.Close()
	provider, err := providers.NewAnthropic(providers.AnthropicConfig{BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	collector := core.NewSourceUsageCollector()
	ctx := core.WithSourceUsageCollector(t.Context(), collector)
	call, err := core.PrepareCall(ctx, translation.Adapter{Provider: provider}, jsonRequest(core.ModelSurfaceResponses, `{"input":"baseline input","max_output_tokens":17,"text":{"format":{"type":"text"}}}`), core.CallPreparationOptions{AllowAdaptation: true})
	if err != nil {
		t.Fatal(err)
	}
	d := call.Description()
	if len(d.Conversions) != 2 || d.Conversions[0].ID != "responses-to-chat" || d.Conversions[1].ID != "chat-to-messages" || d.UpstreamAPI != core.UpstreamAPIMessages {
		t.Fatalf("not the named baseline chain: %+v", d)
	}
	if _, err := core.PlanPreparedCall(d, core.CallPolicy{}); err == nil {
		t.Fatal("native-only default admitted conversion")
	}
	if _, err := core.PlanPreparedCall(d, core.CallPolicy{AllowAdaptation: true}); err != nil {
		t.Fatal(err)
	}
	response, err := call.Invoke(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	var output map[string]any
	if err := json.Unmarshal(response.Body, &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(response.Body), "baseline output") || stringMustJSON(t, output["text"]) != `{"format":{"type":"text"}}` || calls.Load() != 1 {
		t.Fatalf("bad Responses result: %s calls=%d", response.Body, calls.Load())
	}
	observation, ok := collector.SourceUsage()
	if !ok || observation.API != core.UpstreamAPIMessages || observation.Usage.BaseInputTokens.Value != 11 || observation.Usage.OutputTokens.Value != 7 || observation.Usage.Quality != translate.CompletionComplete {
		t.Fatalf("lost original usage: %+v", observation)
	}
}

func stringMustJSON(t *testing.T, value any) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

func TestPreparedResponsesNativeMessagesUnsupportedPreflight(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { calls.Add(1); w.WriteHeader(500) }))
	defer server.Close()
	provider, err := providers.NewAnthropic(providers.AnthropicConfig{BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	for _, extra := range []string{`,"previous_response_id":"opaque"`, `,"tools":[{"type":"web_search_preview"}]`, `,"text":{"format":{"type":"json_schema","name":"result","strict":true,"schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}}}`} {
		call, err := core.PrepareCall(t.Context(), translation.Adapter{Provider: provider}, jsonRequest(core.ModelSurfaceResponses, `{"input":"hi","max_output_tokens":17`+extra+`}`), core.CallPreparationOptions{AllowAdaptation: true})
		if err == nil {
			_, err = core.PlanPreparedCall(call.Description(), core.CallPolicy{AllowAdaptation: true})
		}
		if err == nil {
			t.Fatalf("unsupported request admitted: %s", extra)
		}
	}
	if calls.Load() != 0 {
		t.Fatalf("preflight made %d inference calls", calls.Load())
	}
}

func TestPreparedAdapterComposesActualAPIAndInnerLosses(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`)
	}))
	defer server.Close()
	google, err := providers.NewGoogle(providers.GoogleConfig{Deployment: providers.GoogleAIStudio, BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	adapter := translation.Adapter{Provider: google}
	for _, extra := range []string{"", `,"top_p":0.5`} {
		r := jsonRequest(core.ModelSurfaceMessages, `{"messages":[{"role":"user","content":"hello"}],"max_tokens":32`+extra+`}`)
		call, err := core.PrepareCall(t.Context(), adapter, r, core.CallPreparationOptions{AllowAdaptation: true})
		if err != nil {
			t.Fatal(err)
		}
		d := call.Description()
		if d.UpstreamAPI != core.UpstreamAPIGoogleGenerateContent || len(d.Conversions) != 2 || d.Conversions[0].ID != "messages-to-chat" || d.Conversions[1].ID != "chat-to-google-generate-content" {
			t.Fatalf("composed description=%+v", d)
		}
		if extra != "" {
			if _, err := core.PlanPreparedCall(d, core.CallPolicy{AllowAdaptation: true}); err == nil {
				t.Fatal("clean outer edge hid inner input loss")
			}
			if len(d.Conversions[1].Assessment.Losses) == 0 {
				t.Fatal("inner loss attribution disappeared")
			}
			continue
		}
		if _, err := core.PlanPreparedCall(d, core.CallPolicy{AllowAdaptation: true}); err != nil {
			t.Fatal(err)
		}
		if _, err := call.Invoke(t.Context(), nil); err != nil {
			t.Fatal(err)
		}
	}
	if calls.Load() != 1 {
		t.Fatalf("inference calls=%d", calls.Load())
	}
}

func TestPreparedAdapterKeepsUnknownCustomAPIUnknown(t *testing.T) {
	provider := &fakeProvider{native: core.ModelSurfaceChatCompletions, response: chatCompletion}
	adapter := translation.Adapter{Provider: provider}
	call, err := core.PrepareCall(t.Context(), adapter, jsonRequest(core.ModelSurfaceMessages, `{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}`), core.CallPreparationOptions{AllowAdaptation: true})
	if err != nil {
		t.Fatal(err)
	}
	if call.Description().UpstreamAPI != core.UpstreamAPIUnknown {
		t.Fatal("adapter promoted legacy dispatch into known actual API")
	}
	if _, err := core.PlanPreparedCall(call.Description(), core.CallPolicy{AllowAdaptation: true}); err == nil {
		t.Fatal("unknown custom provider admitted")
	}
	if provider.calls.Load() != 0 {
		t.Fatal("preparation called legacy provider")
	}
}

func TestPreparedAdapterStreamDoesNotCompleteAfterOpaqueStateLoss(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"role\":\"assistant\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n"+
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n"+
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"opaque-fixture\"}}\n\n"+
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	defer server.Close()
	anthropic, err := providers.NewAnthropic(providers.AnthropicConfig{BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	call, err := core.PrepareCall(t.Context(), translation.Adapter{Provider: anthropic}, jsonRequest(core.ModelSurfaceChatCompletions, `{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}`), core.CallPreparationOptions{Mode: core.CallModeStream, AllowAdaptation: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := core.PlanPreparedCall(call.Description(), core.CallPolicy{AllowAdaptation: true}); err != nil {
		t.Fatal(err)
	}
	stream, err := call.Stream(t.Context(), nil)
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
	if err == nil || err == io.EOF {
		t.Fatalf("opaque state loss completed normally: %s", delivered.String())
	}
	if !strings.Contains(delivered.String(), "hello") || strings.Contains(delivered.String(), "[DONE]") || strings.Contains(delivered.String(), `"finish_reason":"stop"`) {
		t.Fatalf("dishonest termination: %s", delivered.String())
	}
	if len(core.StreamLosses(stream)) == 0 || calls.Load() != 1 {
		t.Fatalf("losses=%v inference=%d", core.StreamLosses(stream), calls.Load())
	}
}

func TestPreparedAdapterStreamStopsOnUnassessedResponseFields(t *testing.T) {
	for _, tc := range []struct {
		name       string
		anthropic  bool
		wire, path string
	}{
		{"Messages citations", true,
			"data: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"role\":\"assistant\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
				"data: {\"type\":\"content_block_start\",\"index\":3,\"content_block\":{\"type\":\"text\",\"text\":\"\",\"citations\":[{\"type\":\"unreviewed\"}]}}\n\n" +
				"data: {\"type\":\"content_block_delta\",\"index\":3,\"delta\":{\"type\":\"text_delta\",\"text\":\"affected\"}}\n\n" +
				"data: {\"type\":\"message_stop\"}\n\n", "content.3.citations"},
		{"Chat unreviewed control", false,
			"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"affected\",\"unreviewed_control\":true}}]}\n\ndata: [DONE]\n\n", "choices.0.delta.unreviewed_control"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, tc.wire)
			}))
			defer server.Close()
			var provider core.Provider
			var err error
			surface := core.ModelSurfaceMessages
			if tc.anthropic {
				provider, err = providers.NewAnthropic(providers.AnthropicConfig{BaseURL: server.URL})
				surface = core.ModelSurfaceChatCompletions
			} else {
				provider, err = providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{BaseURL: server.URL, ForwardAllFields: true})
			}
			if err != nil {
				t.Fatal(err)
			}
			call, err := core.PrepareCall(t.Context(), translation.Adapter{Provider: provider}, jsonRequest(surface, `{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}`), core.CallPreparationOptions{Mode: core.CallModeStream, AllowAdaptation: true})
			if err != nil {
				t.Fatal(err)
			}
			stream, err := call.Stream(t.Context(), nil)
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
			if err == io.EOF || err == nil {
				t.Fatalf("unassessed field completed: %s", delivered.String())
			}
			if strings.Contains(delivered.String(), "affected") || strings.Contains(delivered.String(), "[DONE]") || strings.Contains(delivered.String(), "message_stop") {
				t.Fatalf("affected output escaped assessment: %s", delivered.String())
			}
			found := false
			for _, loss := range core.StreamLosses(stream) {
				found = found || loss.Path == tc.path
			}
			if !found {
				t.Fatalf("missing unassessed path %s: %v", tc.path, core.StreamLosses(stream))
			}
			if calls.Load() != 1 {
				t.Fatalf("inference replayed %d times", calls.Load())
			}
		})
	}
}

func TestPreparedResponsesToolDefaultRemainsUnassessedUnlessExplicit(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { calls.Add(1); _, _ = io.WriteString(w, chatCompletion) }))
	defer server.Close()
	p, err := providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{BaseURL: server.URL, ForwardAllFields: true})
	if err != nil {
		t.Fatal(err)
	}
	for _, strict := range []string{"", `,"strict":null`, `,"strict":false`, `,"strict":true`} {
		body := `{"input":"hi","tools":[{"type":"function","name":"f","parameters":{"type":"object"}` + strict + `}]}`
		call, err := core.PrepareCall(t.Context(), translation.Adapter{Provider: p}, jsonRequest(core.ModelSurfaceResponses, body), core.CallPreparationOptions{AllowAdaptation: true})
		if err != nil {
			t.Fatal(err)
		}
		_, err = core.PlanPreparedCall(call.Description(), core.CallPolicy{AllowAdaptation: true})
		if strict == "" || strings.Contains(strict, "null") {
			if err == nil || !strings.Contains(err.Error(), "tools.0.strict") {
				t.Fatalf("implicit Responses default promoted: %v %+v", err, call.Description())
			}
		} else if err != nil {
			t.Fatalf("explicit tool strictness rejected: %v", err)
		}
	}
	if calls.Load() != 0 {
		t.Fatal("tool assessment performed inference")
	}
}

func TestPreparedResponsesEchoesOriginalAssessedFormatNotMappedChatEnvelope(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("wrong target %s", r.URL.Path)
		}
		_, _ = io.WriteString(w, `{"id":"c1","model":"reported-model","choices":[{"message":{"content":"{\"ok\":true}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer server.Close()
	provider, err := providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{BaseURL: server.URL, ForwardAllFields: true})
	if err != nil {
		t.Fatal(err)
	}
	request := jsonRequest(core.ModelSurfaceResponses, `{"input":"JSON","text":{"format":{"type":"json_schema","name":"verdict","strict":true,"schema":{"type":"object","properties":{"ok":{"type":"boolean"}}}}}}`)
	call, err := core.PrepareCall(t.Context(), translation.Adapter{Provider: provider}, request, core.CallPreparationOptions{AllowAdaptation: true})
	if err != nil {
		t.Fatal(err)
	}
	response, err := call.Invoke(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	value := decode(t, response.Body)
	text, _ := value["text"].(map[string]any)
	format, _ := text["format"].(map[string]any)
	if format["type"] != "json_schema" || format["name"] != "verdict" || format["strict"] != true {
		t.Fatalf("original Responses contract replaced by Chat defaults: %s", response.Body)
	}
}
