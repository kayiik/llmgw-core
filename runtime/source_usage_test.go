package runtime_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/providers"
	coreruntime "github.com/kayiik/llmgw-core/runtime"
	"github.com/kayiik/llmgw-core/translation"
)

const sourceChatUsageEvents = "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"}}]}\n\n" +
	"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7,\"total_tokens\":18}}\n\ndata: [DONE]\n\n"
const sourceMessagesUsageEvents = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"role\":\"assistant\",\"content\":[],\"usage\":{\"input_tokens\":11,\"output_tokens\":0}}}\n\n" +
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n" +
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
	"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}\n\n" +
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

func TestPreparedSourceUsageSurvivesConversionIndependentlyOfWireIncludeUsage(t *testing.T) {
	for _, native := range []core.UpstreamAPI{core.UpstreamAPIChatCompletions, core.UpstreamAPIMessages} {
		for _, include := range []string{"absent", "false", "true"} {
			t.Run(string(native)+"/"+include, func(t *testing.T) {
				wire := sourceChatUsageEvents
				if native == core.UpstreamAPIMessages {
					wire = sourceMessagesUsageEvents
				}
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = io.WriteString(w, wire)
				}))
				defer server.Close()
				var provider core.Provider
				var err error
				if native == core.UpstreamAPIMessages {
					provider, err = providers.NewAnthropic(providers.AnthropicConfig{BaseURL: server.URL})
				} else {
					provider, err = providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{BaseURL: server.URL, ForwardAllFields: true})
				}
				if err != nil {
					t.Fatal(err)
				}
				r, err := coreruntime.New(coreruntime.Options[struct{}]{Settings: coreruntime.NewMemorySettings(struct{}{}), Providers: func(struct{}, string) (core.Provider, error) { return translation.Adapter{Provider: provider}, nil }})
				if err != nil {
					t.Fatal(err)
				}
				surface := core.ModelSurfaceMessages
				body := `{"messages":[{"role":"user","content":"hi"}],"max_tokens":32,"stream":true}`
				if native == core.UpstreamAPIMessages {
					surface = core.ModelSurfaceChatCompletions
					body = `{"messages":[{"role":"user","content":"hi"}],"max_tokens":32,"stream":true`
					if include != "absent" {
						body += `,"stream_options":{"include_usage":` + include + `}`
					}
					body += `}`
				}
				request := core.Request{Surface: surface, Model: "m", Body: []byte(body), ContentType: core.ContentTypeJSON}
				prepared, err := r.PrepareCall(t.Context(), core.LocalCaller(), "one", request, core.CallPreparationOptions{Mode: core.CallModeStream, AllowAdaptation: true})
				if err != nil {
					t.Fatal(err)
				}
				stream, err := r.StreamPrepared(t.Context(), core.LocalCaller(), "one", request, prepared, core.CallPolicy{AllowAdaptation: true})
				if err != nil {
					t.Fatal(err)
				}
				var delivered strings.Builder
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
				observation, ok := core.SourceUsageOf(stream)
				if !ok {
					t.Fatal("actual native source observation lost through composed stream")
				}
				if observation.API != native || observation.Usage.OutputTokens.Value != 7 || observation.Usage.Quality != translate.CompletionComplete {
					t.Fatalf("source result=%+v", observation)
				}
				input := observation.Usage.InputTokens
				if native == core.UpstreamAPIMessages {
					input = observation.Usage.BaseInputTokens
				}
				if input.Presence != translate.CountPresent || input.Value != 11 {
					t.Fatalf("late native input replaced by estimate/placeholder: %+v", observation)
				}
				fromHandle, ok := core.SourceUsageOf(prepared)
				if !ok || fromHandle.Usage.OutputTokens.Value != 7 {
					t.Fatal("runtime handle did not retain stream source usage")
				}
				if native == core.UpstreamAPIMessages && include != "true" && strings.Contains(delivered.String(), `"prompt_tokens":`) {
					t.Fatalf("accounting observation leaked a target usage chunk: %s", delivered.String())
				}
				_ = stream.Close()
			})
		}
	}
}

func TestPreparedSourceUsageInvokeZeroAndPaidResponseRefusalRemainObservable(t *testing.T) {
	for _, refusal := range []bool{false, true} {
		t.Run(map[bool]string{false: "reported-zero", true: "paid-fidelity-refusal"}[refusal], func(t *testing.T) {
			var calls atomic.Int32
			wire := `{"id":"c1","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`
			if refusal {
				wire = `{"id":"c1","choices":[{"index":0,"message":{"content":"first"}},{"index":1,"message":{"content":"second"}}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}`
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { calls.Add(1); _, _ = io.WriteString(w, wire) }))
			defer server.Close()
			provider, err := providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{BaseURL: server.URL, ForwardAllFields: true})
			if err != nil {
				t.Fatal(err)
			}
			r, err := coreruntime.New(coreruntime.Options[struct{}]{Settings: coreruntime.NewMemorySettings(struct{}{}), Providers: func(struct{}, string) (core.Provider, error) { return translation.Adapter{Provider: provider}, nil }})
			if err != nil {
				t.Fatal(err)
			}
			surface := core.ModelSurfaceChatCompletions
			if refusal {
				surface = core.ModelSurfaceMessages
			}
			request := core.Request{Surface: surface, Model: "m", Body: []byte(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}`), ContentType: core.ContentTypeJSON}
			prepared, err := r.PrepareCall(t.Context(), core.LocalCaller(), "native", request, core.CallPreparationOptions{AllowAdaptation: true})
			if err != nil {
				t.Fatal(err)
			}
			_, err = r.InvokePrepared(t.Context(), core.LocalCaller(), "native", request, prepared, core.CallPolicy{AllowAdaptation: true})
			if (err != nil) != refusal {
				t.Fatalf("refusal=%t err=%v", refusal, err)
			}
			observation, ok := core.SourceUsageOf(prepared)
			if !ok || observation.Instance != "native" || observation.Model != "m" || observation.Usage.Quality != translate.CompletionComplete || calls.Load() != 1 {
				t.Fatalf("paid source evidence missing: %+v calls=%d", observation, calls.Load())
			}
			wanted := int64(0)
			if refusal {
				wanted = 7
			}
			if observation.Usage.OutputTokens.Presence != translate.CountPresent || observation.Usage.OutputTokens.Value != wanted || observation.Usage.OutputTokens.Provenance != translate.UsageReported {
				t.Fatalf("source zero/refusal became placeholder: %+v", observation)
			}
		})
	}
}

func TestPreparedSourceUsageRetainsPartialCancellationAndTruncation(t *testing.T) {
	for _, cancelled := range []bool{false, true} {
		t.Run(map[bool]string{false: "truncated", true: "cancelled"}[cancelled], func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":2}}\n\n")
				w.(http.Flusher).Flush()
				if cancelled {
					<-r.Context().Done()
				}
			}))
			defer server.Close()
			provider, err := providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{BaseURL: server.URL, ForwardAllFields: true})
			if err != nil {
				t.Fatal(err)
			}
			r, err := coreruntime.New(coreruntime.Options[struct{}]{Settings: coreruntime.NewMemorySettings(struct{}{}), Providers: func(struct{}, string) (core.Provider, error) { return provider, nil }})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			request := core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "m", Body: []byte(`{"messages":[{"role":"user","content":"hi"}],"stream":true}`), ContentType: core.ContentTypeJSON}
			prepared, err := r.PrepareCall(ctx, core.LocalCaller(), "native", request, core.CallPreparationOptions{Mode: core.CallModeStream})
			if err != nil {
				t.Fatal(err)
			}
			stream, err := r.StreamPrepared(ctx, core.LocalCaller(), "native", request, prepared, core.CallPolicy{})
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			if _, err := stream.Next(); err != nil {
				t.Fatal(err)
			}
			if cancelled {
				cancel()
			}
			_, err = stream.Next()
			if cancelled && !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation=%v", err)
			}
			observation, ok := core.SourceUsageOf(stream)
			quality := translate.CompletionTruncated
			if cancelled {
				quality = translate.CompletionPartial
			}
			if !ok || observation.Cancelled != cancelled || observation.Usage.Quality != quality || observation.Usage.InputTokens.Value != 11 || observation.Usage.OutputTokens.Value != 2 {
				t.Fatalf("partial source lost: %+v", observation)
			}
		})
	}
}

func TestPreparedNativeResponsesSourcePreservesLargeCountsAndSubsets(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		wire := `{"id":"r1","object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":9007199254740993,"output_tokens":7,"total_tokens":9007199254741000,"input_tokens_details":{"cached_tokens":4,"cache_write_tokens":3},"output_tokens_details":{"reasoning_tokens":2}}}`
		if streaming {
			wire = "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"status\":\"in_progress\",\"output\":[]}}\n\ndata: {\"type\":\"response.completed\",\"response\":" + wire + "}\n\n"
		}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			if streaming {
				w.Header().Set("Content-Type", "text/event-stream")
			}
			_, _ = io.WriteString(w, wire)
		}))
		defer server.Close()
		provider, err := providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{BaseURL: server.URL, RegistryID: "openai", ForwardAllFields: true})
		if err != nil {
			t.Fatal(err)
		}
		r, err := coreruntime.New(coreruntime.Options[struct{}]{Settings: coreruntime.NewMemorySettings(struct{}{}), Providers: func(struct{}, string) (core.Provider, error) { return provider, nil }})
		if err != nil {
			t.Fatal(err)
		}
		options := core.CallPreparationOptions{}
		if streaming {
			options.Mode = core.CallModeStream
		}
		request := core.Request{Surface: core.ModelSurfaceResponses, Model: "m", Body: []byte(`{"input":"hi"}`), ContentType: core.ContentTypeJSON}
		prepared, err := r.PrepareCall(t.Context(), core.LocalCaller(), "native", request, options)
		if err != nil {
			t.Fatal(err)
		}
		if streaming {
			stream, err := r.StreamPrepared(t.Context(), core.LocalCaller(), "native", request, prepared, core.CallPolicy{})
			if err != nil {
				t.Fatal(err)
			}
			for {
				_, err := stream.Next()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Fatal(err)
				}
			}
			_ = stream.Close()
		} else {
			if _, err := r.InvokePrepared(t.Context(), core.LocalCaller(), "native", request, prepared, core.CallPolicy{}); err != nil {
				t.Fatal(err)
			}
		}
		observation, ok := core.SourceUsageOf(prepared)
		if !ok || observation.API != core.UpstreamAPIResponses || observation.Usage.InputTokens.Value != 9007199254740993 || observation.Usage.TotalTokens.Value != 9007199254741000 || observation.Usage.BaseInputTokens.Value != 9007199254740986 || observation.Usage.OutputTokens.Value != 7 || observation.Usage.ReasoningTokens.Value != 2 || observation.Usage.Quality != translate.CompletionComplete {
			t.Fatalf("native source precision/subset drift: %+v", observation)
		}
	}
}
