package core_test

import (
	"encoding/json"
	"testing"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

func TestSourceUsageCollectorRetainsNativeLateCountsAndDetachedEvidence(t *testing.T) {
	collector := core.NewSourceUsageCollector()
	ctx := core.WithSourceUsageCollector(t.Context(), collector)
	if _, ok := core.SourceUsageOf(collector); ok {
		t.Fatal("observation fabricated before an attempt")
	}
	record := core.BeginSourceUsage(ctx, core.UpstreamAPIChatCompletions, translate.UsageDialectChat)
	record.Merge(map[string]any{"prompt_tokens": 11})
	record.Merge(map[string]any{"completion_tokens": 7, "total_tokens": 18, "prompt_tokens_details": map[string]any{"cached_tokens": 3, "cache_write_tokens": 2}, "completion_tokens_details": map[string]any{"reasoning_tokens": 2}})
	record.Finish(translate.CompletionComplete, false)
	observation, ok := core.SourceUsageOf(collector)
	if !ok || observation.API != core.UpstreamAPIChatCompletions || observation.Usage.InputTokens.Value != 11 || observation.Usage.OutputTokens.Value != 7 || observation.Usage.TotalTokens.Value != 18 || observation.Usage.ReasoningTokens.Value != 2 {
		t.Fatalf("source counts changed: %+v", observation)
	}
	observation.Usage.Raw["prompt_tokens"] = json.RawMessage(`0`)
	history := collector.SourceUsageHistory()
	history.Attempts[0].Usage.Raw["completion_tokens"] = json.RawMessage(`0`)
	again, _ := collector.SourceUsage()
	if string(again.Usage.Raw["prompt_tokens"]) != "11" || string(again.Usage.Raw["completion_tokens"]) != "7" {
		t.Fatalf("source evidence aliases consumer snapshot: %+v", again)
	}
}

func TestSourceUsageCollectorDistinguishesAbsentZeroInvalidAndPartial(t *testing.T) {
	collector := core.NewSourceUsageCollector()
	ctx := core.WithSourceUsageCollector(t.Context(), collector)
	record := core.BeginSourceUsage(ctx, core.UpstreamAPIResponses, translate.UsageDialectResponses)
	record.Merge(map[string]any{"input_tokens": json.Number("9007199254740993"), "output_tokens": 4})
	record.Merge(map[string]any{"output_tokens": 0})
	record.Finish(translate.CompletionTruncated, true)
	observation, _ := collector.SourceUsage()
	if observation.Usage.InputTokens.Value != 9007199254740993 || observation.Usage.OutputTokens.Presence != translate.CountPresent || observation.Usage.OutputTokens.Value != 0 || observation.Usage.CacheReadTokens.Presence != translate.CountAbsent || !observation.Cancelled || observation.Usage.Quality != translate.CompletionTruncated {
		t.Fatalf("presence/quality changed: %+v", observation)
	}
	failed := core.BeginSourceUsage(ctx, core.UpstreamAPIResponses, translate.UsageDialectResponses)
	failed.Merge(map[string]any{"input_tokens": -1})
	failed.Finish(translate.CompletionFailed, false)
	last, _ := collector.SourceUsage()
	if last.Attempt != 2 || last.Usage.InputTokens.Presence != translate.CountInvalid || last.Usage.OutputTokens.Presence != translate.CountAbsent || len(collector.SourceUsageHistory().Attempts) != 2 {
		t.Fatalf("attempt/presence evidence lost: %+v", last)
	}
}

func TestSourceUsageCollectorKeepsDisjointMessagesRelationships(t *testing.T) {
	collector := core.NewSourceUsageCollector()
	ctx := core.WithSourceUsageCollector(t.Context(), collector)
	record := core.BeginSourceUsage(ctx, core.UpstreamAPIMessages, translate.UsageDialectMessages)
	record.Merge(map[string]any{"input_tokens": 7, "cache_read_input_tokens": 11, "cache_creation_input_tokens": 13})
	record.Merge(map[string]any{"output_tokens": 5})
	record.Finish(translate.CompletionComplete, false)
	observation, _ := collector.SourceUsage()
	if observation.Usage.InputTokens.Value != 31 || observation.Usage.TotalTokens.Value != 36 || observation.Usage.InputTokens.Provenance != translate.UsageDerived || observation.Usage.BaseInputTokens.Provenance != translate.UsageReported {
		t.Fatalf("Anthropic categories double counted: %+v", observation)
	}
}

func TestSourceUsageCollectorDetailsAndHistoryBounds(t *testing.T) {
	collector := core.NewSourceUsageCollectorFor(core.Target{Provider: "p", Model: "m"})
	ctx := core.WithSourceUsageCollector(t.Context(), collector)
	for range 65 {
		record := core.BeginSourceUsage(ctx, core.UpstreamAPIChatCompletions, translate.UsageDialectChat)
		record.Merge(map[string]any{"prompt_tokens": 100, "completion_tokens": 30, "prompt_tokens_details": map[string]any{"audio_tokens": 12}, "completion_tokens_details": map[string]any{"audio_tokens": 7, "accepted_prediction_tokens": 5, "rejected_prediction_tokens": 3}})
		record.Finish(translate.CompletionComplete, false)
	}
	history := collector.SourceUsageHistory()
	last, _ := collector.SourceUsage()
	if !history.Truncated || len(history.Attempts) != 64 || history.Attempts[0].Attempt != 2 || last.Details.InputAudioTokens.Value != 12 || last.Details.OutputAudioTokens.Value != 7 || last.Details.AcceptedPredictionTokens.Value != 5 || last.Details.RejectedPredictionTokens.Value != 3 || last.Usage.OutputTokens.Value != 30 {
		t.Fatalf("detail/history normalization changed: %+v", last)
	}
}
