package providers

import (
	"encoding/json"
	"io"
	"strings"
	"testing"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

func googleSourceUsageTest(t *testing.T) (*preparedNativeUsage, *core.SourceUsageCollector) {
	t.Helper()
	collector := core.NewSourceUsageCollectorFor(core.Target{Provider: "google", Model: "m"})
	record := core.BeginSourceUsage(core.WithSourceUsageCollector(t.Context(), collector), core.UpstreamAPIGoogleGenerateContent, translate.UsageDialectChat)
	return newPreparedNativeUsage(core.UpstreamAPIGoogleGenerateContent, record), collector
}

func TestGoogleSourceUsageDerivesInclusiveOutputBeforeSubsetValidation(t *testing.T) {
	usage, collector := googleSourceUsageTest(t)
	usage.merge(map[string]any{"promptTokenCount": 7, "candidatesTokenCount": 3, "thoughtsTokenCount": 29, "totalTokenCount": 39})
	observation, _ := collector.SourceUsage()
	if observation.Usage.OutputTokens.Value != 32 || observation.Usage.ReasoningTokens.Value != 29 || observation.Usage.TotalTokens.Value != 39 || observation.Usage.OutputTokens.Provenance != translate.UsageDerived || len(observation.Usage.Issues) != 0 {
		t.Fatalf("valid Google disjoint components misclassified: %+v", observation)
	}
}

func TestGoogleSourceUsageOptionalThoughtsAndToolPrompt(t *testing.T) {
	for _, tc := range []struct {
		name               string
		raw                map[string]any
		output             int64
		present, reasoning bool
	}{
		{"native11-5-16-without-thoughts", map[string]any{"promptTokenCount": 11, "candidatesTokenCount": 5, "totalTokenCount": 16}, 5, true, false},
		{"reported-candidate-zero", map[string]any{"promptTokenCount": 11, "candidatesTokenCount": 0, "totalTokenCount": 11}, 0, true, false},
		{"all-absent", map[string]any{}, 0, false, false},
		{"candidate-absent", map[string]any{"promptTokenCount": 11}, 0, false, false},
		{"thinking-summed-once", map[string]any{"promptTokenCount": 11, "candidatesTokenCount": 5, "thoughtsTokenCount": 2, "totalTokenCount": 18}, 7, true, true},
		{"tool-prompt-with-thoughts", map[string]any{"promptTokenCount": 11, "candidatesTokenCount": 5, "thoughtsTokenCount": 2, "toolUsePromptTokenCount": 3, "totalTokenCount": 21}, 7, true, true},
		{"tool-prompt-without-thoughts", map[string]any{"promptTokenCount": 11, "candidatesTokenCount": 5, "toolUsePromptTokenCount": 3, "totalTokenCount": 19}, 5, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			usage, collector := googleSourceUsageTest(t)
			usage.merge(tc.raw)
			o, _ := collector.SourceUsage()
			presence := translate.CountAbsent
			if tc.present {
				presence = translate.CountPresent
			}
			if o.Usage.OutputTokens.Presence != presence || o.Usage.OutputTokens.Value != tc.output || len(o.Usage.Issues) != 0 {
				t.Fatalf("native counters became estimates/invalid: %+v", o)
			}
			if tc.present && o.Usage.OutputTokens.Provenance != translate.UsageDerived {
				t.Fatalf("aggregate relabelled reported: %+v", o)
			}
			if !tc.reasoning && o.Usage.ReasoningTokens.Presence != translate.CountAbsent {
				t.Fatalf("missing reasoning invented: %+v", o)
			}
			if tc.reasoning && o.Usage.ReasoningTokens.Value != 2 {
				t.Fatalf("reasoning not retained separately: %+v", o)
			}
			if tc.present && !tc.reasoning && !strings.Contains(o.Usage.OutputTokens.Path, "lower_bound") {
				t.Fatalf("incomplete aggregate not qualified: %+v", o)
			}
			if _, hasTool := tc.raw["toolUsePromptTokenCount"]; hasTool && string(o.Usage.Raw["toolUsePromptTokenCount"]) != "3" {
				t.Fatalf("native tool category lost: %+v", o)
			}
		})
	}
}

func TestGoogleSourceUsageLateThoughtsReplaceLowerBound(t *testing.T) {
	usage, collector := googleSourceUsageTest(t)
	usage.merge(map[string]any{"promptTokenCount": 11, "candidatesTokenCount": 5, "totalTokenCount": 16})
	usage.merge(map[string]any{"thoughtsTokenCount": 2, "totalTokenCount": 18})
	usage.merge(map[string]any{"candidatesTokenCount": 6, "totalTokenCount": 19})
	o, _ := collector.SourceUsage()
	if o.Usage.OutputTokens.Value != 8 || o.Usage.ReasoningTokens.Value != 2 || len(o.Usage.Issues) != 0 || strings.Contains(o.Usage.OutputTokens.Path, "lower_bound") {
		t.Fatalf("late/cumulative components summed twice or erased: %+v", o)
	}
	usage.merge(map[string]any{"thoughtsTokenCount": 0, "totalTokenCount": 17})
	o, _ = collector.SourceUsage()
	if o.Usage.OutputTokens.Value != 6 || o.Usage.ReasoningTokens.Presence != translate.CountPresent || o.Usage.ReasoningTokens.Value != 0 || len(o.Usage.Issues) != 0 {
		t.Fatalf("reported reasoning zero did not replace prior count: %+v", o)
	}
}

func TestGoogleSourceUsageRejectsToolAndTotalContradictions(t *testing.T) {
	for _, tc := range []struct {
		name  string
		raw   map[string]any
		issue string
	}{
		{"negative-tool", map[string]any{"promptTokenCount": 11, "candidatesTokenCount": 5, "toolUsePromptTokenCount": -1}, "toolUsePromptTokenCount"},
		{"overflow-tool", map[string]any{"toolUsePromptTokenCount": json.Number("9223372036854775808")}, "toolUsePromptTokenCount"},
		{"total-below-known-components", map[string]any{"promptTokenCount": 11, "candidatesTokenCount": 5, "thoughtsTokenCount": 2, "totalTokenCount": 16}, "totalTokenCount"},
		{"total-below-known-tool-components", map[string]any{"promptTokenCount": 11, "candidatesTokenCount": 5, "toolUsePromptTokenCount": 3, "totalTokenCount": 16}, "totalTokenCount"},
		{"total-above-all-reported-components", map[string]any{"promptTokenCount": 11, "candidatesTokenCount": 5, "thoughtsTokenCount": 2, "toolUsePromptTokenCount": 3, "totalTokenCount": 22}, "totalTokenCount"},
		{"aggregate-overflow", map[string]any{"promptTokenCount": json.Number("9223372036854775807"), "candidatesTokenCount": 1, "totalTokenCount": json.Number("9223372036854775807")}, "overflow"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			usage, collector := googleSourceUsageTest(t)
			usage.merge(tc.raw)
			o, _ := collector.SourceUsage()
			if !strings.Contains(strings.Join(o.Usage.Issues, " "), tc.issue) {
				t.Fatalf("genuine invalid evidence accepted: %+v", o)
			}
		})
	}
}

func TestGoogleSourceUsageRetainsRealInvalidOverflowAndCacheIssues(t *testing.T) {
	for _, tc := range []struct {
		name          string
		raw           map[string]any
		invalidOutput bool
		issue         string
	}{
		{"negative thoughts", map[string]any{"promptTokenCount": 7, "candidatesTokenCount": 3, "thoughtsTokenCount": -1}, true, "invalid count"},
		{"output overflow", map[string]any{"candidatesTokenCount": json.Number("9223372036854775807"), "thoughtsTokenCount": 1}, true, "overflow"},
		{"cache exceeds prompt", map[string]any{"promptTokenCount": 7, "cachedContentTokenCount": 8, "candidatesTokenCount": 3, "thoughtsTokenCount": 0}, false, "cached content exceeds"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			usage, collector := googleSourceUsageTest(t)
			usage.merge(tc.raw)
			observation, _ := collector.SourceUsage()
			if tc.invalidOutput && observation.Usage.OutputTokens.Presence != translate.CountInvalid {
				t.Fatalf("invalid became reported0: %+v", observation)
			}
			if !strings.Contains(strings.Join(observation.Usage.Issues, " "), tc.issue) {
				t.Fatalf("genuine source issue lost: %+v", observation)
			}
		})
	}
}

func TestGoogleSourceUsagePreservesAbsentAndPreviousNativeTotal(t *testing.T) {
	usage, collector := googleSourceUsageTest(t)
	usage.merge(map[string]any{"promptTokenCount": 7, "candidatesTokenCount": 3, "thoughtsTokenCount": 29})
	observation, _ := collector.SourceUsage()
	if observation.Usage.OutputTokens.Value != 32 || observation.Usage.TotalTokens.Presence != translate.CountAbsent {
		t.Fatalf("omitted native total fabricated: %+v", observation)
	}
	usage.merge(map[string]any{"totalTokenCount": 39})
	usage.merge(map[string]any{"candidatesTokenCount": 3})
	observation, _ = collector.SourceUsage()
	if observation.Usage.TotalTokens.Presence != translate.CountPresent || observation.Usage.TotalTokens.Value != 39 || observation.Usage.TotalTokens.Provenance != translate.UsageReported || observation.Usage.TotalTokens.Path != "totalTokenCount" {
		t.Fatalf("omitted frame erased/relabelled previous total: %+v", observation)
	}
}

func TestPreparedNativeGoogleAndOllamaUsageComesBeforeChatFacade(t *testing.T) {
	for _, api := range []core.UpstreamAPI{core.UpstreamAPIGoogleGenerateContent, core.UpstreamAPIOllamaChat} {
		for _, streaming := range []bool{false, true} {
			t.Run(string(api)+map[bool]string{false: "/invoke", true: "/stream"}[streaming], func(t *testing.T) {
				var provider core.Provider
				if api == core.UpstreamAPIGoogleGenerateContent {
					wire := `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3,"thoughtsTokenCount":29,"totalTokenCount":39}}`
					if streaming {
						_, base := newGoogleStreamFake(t, 200, "data: "+wire+"\n\n")
						provider = newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
					} else {
						_, base := newGoogleFake(t, googleAnswer(200, wire))
						provider = newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
					}
				} else {
					wire := `{"message":{"content":"ok"},"done":true,"prompt_eval_count":9007199254740993,"eval_count":7,"total_duration":99}`
					if streaming {
						wire = `{"message":{"content":"ok"},"done":false}` + "\n" + `{"message":{"content":""},"done":true,"prompt_eval_count":9007199254740993,"eval_count":7,"total_duration":99}` + "\n"
					}
					provider, _ = ollamaDaemon(t, 200, wire)
				}
				collector := core.NewSourceUsageCollectorFor(core.Target{Provider: "native", Model: "fixture"})
				ctx := core.WithSourceUsageCollector(t.Context(), collector)
				options := core.CallPreparationOptions{AllowAdaptation: true}
				if streaming {
					options.Mode = core.CallModeStream
				}
				call, err := core.PrepareCall(ctx, provider, googleChatRequest("fixture", `{"messages":[{"role":"user","content":"hi"}]}`, googleKey("fixture")), options)
				if err != nil {
					t.Fatal(err)
				}
				if streaming {
					stream, err := call.Stream(ctx, googleKey("fixture"))
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
					if _, err := call.Invoke(ctx, googleKey("fixture")); err != nil {
						t.Fatal(err)
					}
				}
				observation, ok := collector.SourceUsage()
				if !ok || observation.API != api || observation.Usage.Quality != translate.CompletionComplete {
					t.Fatalf("source observation=%+v", observation)
				}
				if _, wrong := observation.Usage.Raw["prompt_tokens"]; wrong {
					t.Fatal("converted Chat usage relabeled native source")
				}
				if api == core.UpstreamAPIGoogleGenerateContent {
					if observation.Usage.InputTokens.Value != 7 || observation.Usage.OutputTokens.Value != 32 || observation.Usage.ReasoningTokens.Value != 29 || observation.Usage.TotalTokens.Value != 39 || len(observation.Usage.Issues) != 0 {
						t.Fatalf("Google source=%+v", observation)
					}
				} else if observation.Usage.InputTokens.Value != 9007199254740993 || observation.Usage.OutputTokens.Value != 7 || observation.Usage.TotalTokens.Value != 9007199254741000 {
					t.Fatalf("Ollama original exact counters lost: %+v", observation)
				}
			})
		}
	}
}

func TestPreparedNativeGoogleAndOllamaUsageRetainsFailureAndTruncation(t *testing.T) {
	for _, api := range []core.UpstreamAPI{core.UpstreamAPIGoogleGenerateContent, core.UpstreamAPIOllamaChat} {
		for _, failed := range []bool{false, true} {
			t.Run(string(api)+map[bool]string{false: "/truncated", true: "/failed"}[failed], func(t *testing.T) {
				var provider core.Provider
				if api == core.UpstreamAPIGoogleGenerateContent {
					wire := "data: " + `{"candidates":[{"content":{"parts":[{"text":"partial"}]}}],"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":7,"thoughtsTokenCount":0}}` + "\n\n"
					if failed {
						wire += "data: {broken}\n\n"
					}
					_, base := newGoogleStreamFake(t, 200, wire)
					provider = newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
				} else {
					wire := `{"message":{"content":"partial"},"done":false,"prompt_eval_count":11,"eval_count":7}` + "\n"
					if failed {
						wire += `{"error":"generation failed"}` + "\n"
					}
					provider, _ = ollamaDaemon(t, 200, wire)
				}
				collector := core.NewSourceUsageCollector()
				ctx := core.WithSourceUsageCollector(t.Context(), collector)
				call, err := core.PrepareCall(ctx, provider, googleChatRequest("fixture", `{"messages":[{"role":"user","content":"hi"}]}`, googleKey("fixture")), core.CallPreparationOptions{Mode: core.CallModeStream, AllowAdaptation: true})
				if err != nil {
					t.Fatal(err)
				}
				stream, err := call.Stream(ctx, googleKey("fixture"))
				if err != nil {
					t.Fatal(err)
				}
				for {
					if _, err := stream.Next(); err != nil {
						break
					}
				}
				_ = stream.Close()
				observation, ok := collector.SourceUsage()
				quality := translate.CompletionTruncated
				if failed {
					quality = translate.CompletionFailed
				}
				if !ok || observation.Usage.Quality != quality || observation.Usage.InputTokens.Value != 11 || observation.Usage.OutputTokens.Value != 7 {
					t.Fatalf("native failure/truncation evidence lost: %+v", observation)
				}
			})
		}
	}
}
