package translation_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/providers"
	"github.com/kayiik/llmgw-core/translation"
)

func TestPreparedResponsesThroughChatToNativeMessagesIncremental(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	release := make(chan struct{})
	var once sync.Once
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Error(err)
		}
		if r.URL.Path != "/v1/messages" || payload["stream"] != true || payload["max_tokens"] != float64(17) || !strings.Contains(stringMustJSON(t, payload["messages"]), "baseline input") {
			t.Errorf("native stream request mismatch: %s %+v", r.URL.Path, payload)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"usage\":{\"input_tokens\":11,\"output_tokens\":0}}}\n\n"+
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"+
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"baseline output\"}}\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-ctx.Done():
			return
		}
		_, _ = io.WriteString(w, "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"+
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}\n\n"+
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	t.Cleanup(server.Close)
	t.Cleanup(func() { once.Do(func() { close(release) }) })
	provider, err := providers.NewAnthropic(providers.AnthropicConfig{BaseURL: server.URL})
	if err != nil {
		t.Fatal(err)
	}
	collector := core.NewSourceUsageCollector()
	ctx = core.WithSourceUsageCollector(ctx, collector)
	call, err := core.PrepareCall(ctx, translation.Adapter{Provider: provider}, jsonRequest(core.ModelSurfaceResponses, `{"input":"baseline input","max_output_tokens":17,"stream":true,"text":{"format":{"type":"text"}}}`), core.CallPreparationOptions{Mode: core.CallModeStream, AllowAdaptation: true, Requirements: core.CallRequirements{Incremental: true}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := core.PlanPreparedCall(call.Description(), core.CallPolicy{AllowAdaptation: true, RequireIncremental: true}); err != nil {
		t.Fatal(err)
	}
	stream, err := call.Stream(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var delivered strings.Builder
	for !strings.Contains(delivered.String(), "baseline output") {
		frame, err := stream.Next()
		if err != nil {
			t.Fatalf("text waited for native terminal: %v; %s", err, delivered.String())
		}
		delivered.Write(frame)
	}
	if strings.Contains(delivered.String(), "response.completed") {
		t.Fatal("completed before native terminal")
	}
	once.Do(func() { close(release) })
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
	if !strings.Contains(delivered.String(), "response.completed") || !strings.Contains(delivered.String(), `"text":{"format":{"type":"text"}}`) || calls.Load() != 1 {
		t.Fatalf("terminal/replay mismatch: %s calls=%d", delivered.String(), calls.Load())
	}
	observation, ok := core.SourceUsageOf(stream)
	if !ok || observation.API != core.UpstreamAPIMessages || observation.Usage.BaseInputTokens.Value != 11 || observation.Usage.OutputTokens.Value != 7 || observation.Usage.Quality != translate.CompletionComplete {
		t.Fatalf("native usage lost without include_usage: %+v", observation)
	}
	history := core.SourceUsageHistoryOf(stream)
	if len(history.Attempts) != 1 || history.Truncated {
		t.Fatalf("conversion legs became native attempts: %+v", history)
	}
}

func TestPreparedChatResponsesStreamsEmitBeforeTerminal(t *testing.T) {
	for _, source := range []core.ModelSurface{core.ModelSurfaceResponses, core.ModelSurfaceChatCompletions} {
		t.Run(string(source), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			release := make(chan struct{})
			var once sync.Once
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				w.Header().Set("Content-Type", "text/event-stream")
				if source == core.ModelSurfaceResponses {
					_, _ = io.WriteString(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"early\"}}]}\n\n")
				} else {
					_, _ = io.WriteString(w, "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_fixture\",\"object\":\"response\",\"status\":\"in_progress\",\"output\":[]}}\n\n"+
						"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_fixture\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n"+
						"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_fixture\",\"output_index\":0,\"content_index\":0,\"delta\":\"early\"}\n\n")
				}
				w.(http.Flusher).Flush()
				select {
				case <-release:
				case <-ctx.Done():
					return
				}
				if source == core.ModelSurfaceResponses {
					_, _ = io.WriteString(w, "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
				} else {
					_, _ = io.WriteString(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_fixture\",\"object\":\"response\",\"status\":\"completed\",\"output\":[{\"id\":\"msg_fixture\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"early\",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n")
				}
			}))
			t.Cleanup(server.Close)
			t.Cleanup(func() { once.Do(func() { close(release) }) })
			config := providers.OpenAICompatibleConfig{BaseURL: server.URL, ForwardAllFields: true}
			body := `{"input":"hello","stream":true}`
			if source == core.ModelSurfaceChatCompletions {
				config.ForceAPISupport = true
				config.Models = func(model string) (core.ModelInfo, bool) {
					return core.ModelInfo{ID: model, SupportedAPIs: []string{"/responses"}}, true
				}
				body = `{"messages":[{"role":"user","content":"hello"}],"stream":true,"stream_options":{"include_usage":true}}`
			}
			provider, err := providers.NewOpenAICompatible(config)
			if err != nil {
				t.Fatal(err)
			}
			call, err := core.PrepareCall(ctx, translation.Adapter{Provider: provider}, jsonRequest(source, body), core.CallPreparationOptions{Mode: core.CallModeStream, AllowAdaptation: true, Requirements: core.CallRequirements{Incremental: true}})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := core.PlanPreparedCall(call.Description(), core.CallPolicy{AllowAdaptation: true, RequireIncremental: true}); err != nil {
				t.Fatal(err)
			}
			stream, err := call.Stream(ctx, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			var delivered strings.Builder
			for !strings.Contains(delivered.String(), "early") {
				frame, err := stream.Next()
				if err != nil {
					t.Fatalf("stream gated on terminal: %v; %s", err, delivered.String())
				}
				delivered.Write(frame)
			}
			once.Do(func() { close(release) })
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
			terminal := "response.completed"
			if source == core.ModelSurfaceChatCompletions {
				terminal = "[DONE]"
			}
			if !strings.Contains(delivered.String(), terminal) || calls.Load() != 1 {
				t.Fatalf("terminal/calls mismatch: %s calls=%d", delivered.String(), calls.Load())
			}
		})
	}
}
