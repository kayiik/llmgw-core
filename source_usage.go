package core

import (
	"context"
	"maps"
	"slices"
	"sync"

	translate "github.com/kayiik/llm-translate"
)

// SourceUsageObservation is evidence from one actual native attempt, before
// target conversion. Usage quality describes the source, not client delivery.
type SourceUsageObservation struct {
	API             UpstreamAPI
	Instance, Model string
	Attempt         uint64
	Usage           translate.UsageObservation
	Details         translate.UsageDetails
	Cancelled       bool
}

type SourceUsageHistory struct {
	Attempts  []SourceUsageObservation
	Truncated bool
}
type SourceUsageReporter interface {
	SourceUsage() (SourceUsageObservation, bool)
}
type SourceUsageHistoryReporter interface{ SourceUsageHistory() SourceUsageHistory }

// SourceUsageOf reads only an explicit observation interface. It never guesses
// measured usage from a rebuilt target body, its required zeros or its shape.
func SourceUsageOf(value any) (SourceUsageObservation, bool) {
	if reporter, ok := value.(SourceUsageReporter); ok {
		observation, available := reporter.SourceUsage()
		return cloneSourceUsage(observation), available
	}
	return SourceUsageObservation{}, false
}
func SourceUsageHistoryOf(value any) SourceUsageHistory {
	if reporter, ok := value.(SourceUsageHistoryReporter); ok {
		history := reporter.SourceUsageHistory()
		history.Attempts = slices.Clone(history.Attempts)
		for i := range history.Attempts {
			history.Attempts[i] = cloneSourceUsage(history.Attempts[i])
		}
		return history
	}
	return SourceUsageHistory{}
}

func cloneSourceUsage(observation SourceUsageObservation) SourceUsageObservation {
	observation.Usage.Issues = slices.Clone(observation.Usage.Issues)
	observation.Usage.Raw = maps.Clone(observation.Usage.Raw)
	for key, value := range observation.Usage.Raw {
		observation.Usage.Raw[key] = slices.Clone(value)
	}
	return observation
}

// SourceUsageCollector bounds retained usage attempts, never inference bodies.
// Native retry/refresh attempts remain separate; cumulative frames replace
// fields within an attempt instead of being summed across network chunks.
type SourceUsageCollector struct {
	mu        sync.Mutex
	next      uint64
	attempts  []*SourceUsageRecorder
	truncated bool
	target    Target
}

func NewSourceUsageCollector() *SourceUsageCollector { return &SourceUsageCollector{} }
func NewSourceUsageCollectorFor(target Target) *SourceUsageCollector {
	return &SourceUsageCollector{target: target}
}

type sourceUsageKey struct{}

func WithSourceUsageCollector(ctx context.Context, collector *SourceUsageCollector) context.Context {
	return context.WithValue(ctx, sourceUsageKey{}, collector)
}

// SourceUsageHistoryFromContext returns a detached snapshot of the existing
// context collector, or an empty history when no collector is attached.
func SourceUsageHistoryFromContext(ctx context.Context) SourceUsageHistory {
	if ctx == nil {
		return SourceUsageHistory{}
	}
	collector, _ := ctx.Value(sourceUsageKey{}).(*SourceUsageCollector)
	return collector.SourceUsageHistory()
}

func (c *SourceUsageCollector) SourceUsage() (SourceUsageObservation, bool) {
	if c == nil {
		return SourceUsageObservation{}, false
	}
	c.mu.Lock()
	if len(c.attempts) == 0 {
		c.mu.Unlock()
		return SourceUsageObservation{}, false
	}
	last := c.attempts[len(c.attempts)-1]
	c.mu.Unlock()
	return last.snapshot(), true
}
func (c *SourceUsageCollector) SourceUsageHistory() SourceUsageHistory {
	if c == nil {
		return SourceUsageHistory{}
	}
	c.mu.Lock()
	records := slices.Clone(c.attempts)
	truncated := c.truncated
	c.mu.Unlock()
	history := SourceUsageHistory{Truncated: truncated}
	for _, record := range records {
		history.Attempts = append(history.Attempts, record.snapshot())
	}
	return history
}

type SourceUsageRecorder struct {
	mu                  sync.Mutex
	api                 UpstreamAPI
	attempt             uint64
	accumulator         *translate.UsageAccumulator
	explicit            *translate.UsageObservationWithDetails
	quality             translate.CompletionQuality
	cancelled, finished bool
	target              Target
}

func BeginSourceUsage(ctx context.Context, api UpstreamAPI, dialect translate.UsageDialect) *SourceUsageRecorder {
	record := &SourceUsageRecorder{api: api, attempt: 1, accumulator: translate.NewUsageAccumulator(dialect), quality: translate.CompletionPartial}
	if collector, ok := ctx.Value(sourceUsageKey{}).(*SourceUsageCollector); ok && collector != nil {
		collector.mu.Lock()
		collector.next++
		record.attempt = collector.next
		record.target = collector.target
		if len(collector.attempts) >= 64 {
			copy(collector.attempts, collector.attempts[1:])
			collector.attempts[len(collector.attempts)-1] = record
			collector.truncated = true
		} else {
			collector.attempts = append(collector.attempts, record)
		}
		collector.mu.Unlock()
	}
	return record
}

func (r *SourceUsageRecorder) Merge(raw any) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.finished {
		r.accumulator.Merge(raw)
	}
}
func (r *SourceUsageRecorder) Set(usage translate.UsageObservation) {
	r.SetWithDetails(translate.UsageObservationWithDetails{UsageObservation: usage, Details: translate.ObserveUsageWithDetails(usage.Dialect, nil, usage.Quality).Details})
}
func (r *SourceUsageRecorder) SetWithDetails(usage translate.UsageObservationWithDetails) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.finished {
		return
	}
	copy := usage
	copy.UsageObservation = cloneSourceUsage(SourceUsageObservation{Usage: usage.UsageObservation}).Usage
	r.explicit = &copy
}
func (r *SourceUsageRecorder) Finish(quality translate.CompletionQuality, cancelled bool) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.finished {
		return
	}
	r.quality, r.cancelled, r.finished = quality, cancelled, true
}
func (r *SourceUsageRecorder) snapshot() SourceUsageObservation {
	r.mu.Lock()
	defer r.mu.Unlock()
	usage := r.accumulator.SnapshotWithDetails(r.quality)
	if r.explicit != nil {
		usage = *r.explicit
		usage.Quality = r.quality
	}
	return cloneSourceUsage(SourceUsageObservation{API: r.api, Instance: r.target.Provider, Model: r.target.Model, Attempt: r.attempt, Usage: usage.UsageObservation, Details: usage.Details, Cancelled: r.cancelled})
}

func (r *SourceUsageRecorder) SourceUsage() (SourceUsageObservation, bool) {
	if r == nil {
		return SourceUsageObservation{}, false
	}
	return r.snapshot(), true
}
func (r *SourceUsageRecorder) SourceUsageHistory() SourceUsageHistory {
	if r == nil {
		return SourceUsageHistory{}
	}
	return SourceUsageHistory{Attempts: []SourceUsageObservation{r.snapshot()}}
}

func (c *SourceUsageCollector) WrapStream(stream StreamIter) StreamIter {
	if stream == nil {
		return nil
	}
	return &sourceUsageStream{StreamIter: stream, collector: c}
}

type sourceUsageStream struct {
	StreamIter
	collector *SourceUsageCollector
}

func (s *sourceUsageStream) SourceUsage() (SourceUsageObservation, bool) {
	return s.collector.SourceUsage()
}
func (s *sourceUsageStream) SourceUsageHistory() SourceUsageHistory {
	return s.collector.SourceUsageHistory()
}
func (s *sourceUsageStream) Losses() []Loss { return StreamLosses(s.StreamIter) }
