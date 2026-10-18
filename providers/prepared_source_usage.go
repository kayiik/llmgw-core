package providers

import (
	"math"
	"sync"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

// Provider-native counters are normalized from original usage keys, never a
// Chat envelope rebuilt by the facade. Native raw evidence remains bounded.
type preparedNativeUsage struct {
	mu          sync.Mutex
	api         core.UpstreamAPI
	counts, raw *translate.UsageAccumulator
	components  *translate.UsageAccumulator
	toolPrompt  *translate.UsageAccumulator
	record      *core.SourceUsageRecorder
}

func newPreparedNativeUsage(api core.UpstreamAPI, record *core.SourceUsageRecorder) *preparedNativeUsage {
	return &preparedNativeUsage{api: api, record: record, counts: translate.NewUsageAccumulator(translate.UsageDialectChat), raw: translate.NewUsageAccumulator(translate.UsageDialectChat), components: translate.NewUsageAccumulator(translate.UsageDialectResponses), toolPrompt: translate.NewUsageAccumulator(translate.UsageDialectResponses)}
}

func (u *preparedNativeUsage) merge(value any) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if value == nil {
		return
	}
	raw, ok := value.(map[string]any)
	if !ok {
		u.raw.Merge(value)
		snapshot := translate.ObserveUsageWithDetails(translate.UsageDialectChat, value, translate.CompletionPartial)
		u.record.SetWithDetails(snapshot)
		return
	}
	u.raw.Merge(raw)
	normalized := map[string]any{}
	if u.api == core.UpstreamAPIGoogleGenerateContent {
		for source, target := range map[string]string{"promptTokenCount": "prompt_tokens", "totalTokenCount": "total_tokens"} {
			if value, set := raw[source]; set {
				normalized[target] = value
			}
		}
		parts := map[string]any{}
		if value, set := raw["candidatesTokenCount"]; set {
			parts["input_tokens"] = value
		}
		if value, set := raw["thoughtsTokenCount"]; set {
			parts["output_tokens"] = value
		}
		u.components.Merge(parts)
		components := u.components.Snapshot(translate.CompletionPartial)
		output := googleObservedOutput(components.InputTokens, components.OutputTokens)
		if output.Presence == translate.CountPresent {
			normalized["completion_tokens"] = output.Value
		} else if output.Presence == translate.CountInvalid {
			normalized["completion_tokens"] = nil
		}
		if value, set := raw["toolUsePromptTokenCount"]; set {
			u.toolPrompt.Merge(map[string]any{"input_tokens": value})
		}
		if value, set := raw["cachedContentTokenCount"]; set {
			normalized["prompt_tokens_details"] = map[string]any{"cached_tokens": value}
		}
		if value, set := raw["thoughtsTokenCount"]; set {
			normalized["completion_tokens_details"] = map[string]any{"reasoning_tokens": value}
		}
	} else {
		if value, set := raw["prompt_eval_count"]; set {
			normalized["prompt_tokens"] = value
		}
		if value, set := raw["eval_count"]; set {
			normalized["completion_tokens"] = value
		}
	}
	u.counts.Merge(normalized)
	snapshot := u.counts.SnapshotWithDetails(translate.CompletionPartial)
	native := u.raw.Snapshot(translate.CompletionPartial)
	snapshot.Raw, snapshot.RawTruncated = native.Raw, native.RawTruncated
	if u.api == core.UpstreamAPIGoogleGenerateContent {
		snapshot.InputTokens.Path = "promptTokenCount"
		snapshot.CacheReadTokens.Path = "cachedContentTokenCount"
		snapshot.ReasoningTokens.Path = "thoughtsTokenCount"
		snapshot.TotalTokens.Path = "totalTokenCount"
		parts := u.components.Snapshot(translate.CompletionPartial)
		candidate, thought := parts.InputTokens, parts.OutputTokens
		output := googleObservedOutput(candidate, thought)
		if output.Presence == translate.CountInvalid && candidate.Presence == translate.CountPresent && thought.Presence == translate.CountPresent {
			snapshot.Issues = append(snapshot.Issues, "Google candidates+thoughts output overflow")
		}
		snapshot.OutputTokens = output
		// Google's total can contain a separate tool-use prompt category. If
		// it was not supplied, do not turn a generic candidate-only sum into
		// a fictional native inclusive total.
		if snapshot.TotalTokens.Provenance == translate.UsageDerived {
			snapshot.TotalTokens = translate.CountObservation{Presence: translate.CountAbsent, Provenance: translate.UsageReported, Path: "totalTokenCount"}
		}
		// Google cache reads are a prompt subset; no fictitious write category
		// is introduced to make a generic Chat partition look complete.
		if snapshot.InputTokens.Presence == translate.CountPresent && snapshot.CacheReadTokens.Presence == translate.CountPresent && snapshot.CacheReadTokens.Value <= snapshot.InputTokens.Value {
			snapshot.BaseInputTokens = translate.CountObservation{Value: snapshot.InputTokens.Value - snapshot.CacheReadTokens.Value, Presence: translate.CountPresent, Provenance: translate.UsageDerived, Path: "promptTokenCount-cachedContentTokenCount"}
		}
		if snapshot.InputTokens.Presence == translate.CountPresent && snapshot.CacheReadTokens.Presence == translate.CountPresent && snapshot.CacheReadTokens.Value > snapshot.InputTokens.Value {
			snapshot.Issues = append(snapshot.Issues, "Google cached content exceeds promptTokenCount")
		}
		// Tool-use prompt tokens remain a separate native raw category, not
		// an output token or a reason to invalidate otherwise valid evidence.
		tool := u.toolPrompt.Snapshot(translate.CompletionPartial).InputTokens
		if tool.Presence == translate.CountInvalid {
			snapshot.Issues = append(snapshot.Issues, "invalid count at toolUsePromptTokenCount")
		}
		snapshot.Issues = append(snapshot.Issues, googleTotalIssues(snapshot.TotalTokens, snapshot.InputTokens, candidate, thought, tool)...)
	} else {
		snapshot.InputTokens.Path = "prompt_eval_count"
		snapshot.OutputTokens.Path = "eval_count"
	}
	u.record.SetWithDetails(snapshot)
}

func googleObservedOutput(candidate, thought translate.CountObservation) translate.CountObservation {
	output := translate.CountObservation{Presence: translate.CountAbsent, Provenance: translate.UsageDerived, Path: "candidatesTokenCount+thoughtsTokenCount"}
	if candidate.Presence == translate.CountInvalid || thought.Presence == translate.CountInvalid {
		output.Presence = translate.CountInvalid
	} else if candidate.Presence == translate.CountPresent {
		output.Presence, output.Value = translate.CountPresent, candidate.Value
		if thought.Presence == translate.CountAbsent {
			// A measured candidate count is still a known output lower bound.
			// Do not fabricate reported reasoning0 to complete the aggregate.
			output.Path = "candidatesTokenCount (lower_bound; thoughtsTokenCount absent)"
		} else if candidate.Value > math.MaxInt64-thought.Value {
			output.Presence, output.Value = translate.CountInvalid, 0
		} else {
			output.Value += thought.Value
		}
	}
	return output
}

// Google's total includes prompt, candidate, thought and tool-use prompt
// categories. Missing categories are unknown, not reported zeros: validate a
// lower bound unless every category is present, then require exact equality.
func googleTotalIssues(total translate.CountObservation, components ...translate.CountObservation) []string {
	var sum int64
	complete := true
	for _, component := range components {
		if component.Presence == translate.CountInvalid {
			return nil // The original invalid component already has an issue.
		}
		if component.Presence != translate.CountPresent {
			complete = false
			continue
		}
		if component.Value > math.MaxInt64-sum {
			return []string{"Google totalTokenCount component sum overflow"}
		}
		sum += component.Value
	}
	if total.Presence == translate.CountPresent && (total.Value < sum || complete && total.Value != sum) {
		return []string{"Google totalTokenCount contradicts observed prompt/candidate/thought/tool-use components"}
	}
	return nil
}

func ollamaNativeUsage(event map[string]any) map[string]any {
	usage := map[string]any{}
	for _, key := range []string{"prompt_eval_count", "eval_count", "total_duration", "load_duration", "prompt_eval_duration", "eval_duration"} {
		if value, present := event[key]; present {
			usage[key] = value
		}
	}
	return usage
}
