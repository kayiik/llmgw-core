// Package observedstream shares finite prepared response conversion between
// providers and the outer adapter without changing legacy streaming routes.
package observedstream

import (
	"context"
	"errors"
	"io"
	"slices"
	"strings"
	"sync"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

// preparedObservedStream is incremental: one pull/push rendezvous at a time,
// with loss evidence observed before affected output and no inference replay.
type preparedObservedStream struct {
	upstream       core.StreamIter
	frames         chan []byte
	stop, finished chan struct{}
	once           sync.Once
	mu             sync.Mutex
	losses         []core.Loss
	err            error
}

func New(ctx context.Context, upstream core.StreamIter, model string, id translate.ConversionID, requirements translate.FidelityRequirements, includeUsage bool, losses []core.Loss, request map[string]any) core.StreamIter {
	s := &preparedObservedStream{upstream: upstream, frames: make(chan []byte), stop: make(chan struct{}), finished: make(chan struct{}), losses: slices.Clone(losses)}
	go s.run(ctx, model, id, requirements, includeUsage, request)
	return s
}

func (s *preparedObservedStream) fail(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err == nil {
		s.err = err
	}
}
func (s *preparedObservedStream) failed() bool { s.mu.Lock(); defer s.mu.Unlock(); return s.err != nil }
func (s *preparedObservedStream) closing() bool {
	select {
	case <-s.stop:
		return true
	default:
		return false
	}
}
func (s *preparedObservedStream) send(frame []byte) {
	if s.failed() || s.closing() {
		return
	}
	select {
	case s.frames <- frame:
	case <-s.stop:
	}
}

func (s *preparedObservedStream) run(ctx context.Context, model string, id translate.ConversionID, requirements translate.FidelityRequirements, includeUsage bool, request map[string]any) {
	defer close(s.finished)
	defer close(s.frames)
	defer s.upstream.Close()
	stopCancellation := context.AfterFunc(ctx, func() { s.fail(ctx.Err()); s.once.Do(func() { close(s.stop) }); _ = s.upstream.Close() })
	defer stopCancellation()
	observe := func(observation translate.StreamObservation) {
		s.mu.Lock()
		s.losses = translate.NewReport(slices.Concat(s.losses, observation.Assessment.Report.Losses)...).Losses
		s.mu.Unlock()
		if len(observation.Assessment.UnknownPaths) > 0 || observation.Assessment.Feasibility == translate.FeasibilityUnknown {
			paths := slices.Clone(observation.Assessment.UnknownPaths)
			if len(paths) == 0 {
				paths = []string{"$"}
			}
			var findings []core.Loss
			for _, path := range paths {
				findings = append(findings, core.Loss{Path: path, Class: translate.LossUnsupported, Severity: translate.LossMaterial, Detail: "the conversion has not assessed this response field; its upstream semantics remain unknown"})
			}
			s.mu.Lock()
			s.losses = translate.NewReport(slices.Concat(s.losses, findings)...).Losses
			s.mu.Unlock()
			s.fail(&core.CallPlanError{Reason: "stream_response_semantics_unassessed", UnknownPaths: paths, Losses: findings})
		}
		if err := (core.LossPolicy{}).Check(observation.Assessment.Report.Losses); err != nil {
			s.fail(err)
		}
		for _, loss := range observation.Assessment.Report.Losses {
			if loss.Class == translate.LossUnsupported {
				s.fail(&core.LossPolicyError{Losses: []core.Loss{loss}})
			}
		}
		if observation.Assessment.Feasibility == translate.FeasibilityUnsupported && !s.failed() {
			s.fail(&core.CallPlanError{Reason: "stream_response_semantics_unsupported", Losses: slices.Clone(observation.Assessment.Report.Losses)})
		}
	}
	options := translate.StreamObservationOptions{IncludeUsage: includeUsage, Requirements: requirements, Observe: observe}
	if id == translate.ConversionChatToMessages {
		var pending []string
		lines := func() (string, bool) {
			for len(pending) == 0 {
				if s.failed() || s.closing() {
					return "", false
				}
				frame, err := s.upstream.Next()
				if err != nil {
					if !errors.Is(err, io.EOF) {
						s.fail(err)
					}
					return "", false
				}
				pending = strings.Split(strings.ReplaceAll(string(frame), "\r\n", "\n"), "\n")
			}
			line := pending[0]
			pending = pending[1:]
			return line, true
		}
		result := translate.AnthropicSSEToOpenAIChunksObserved(lines, model, func(chunk string) { s.send([]byte("data: " + chunk + "\n\n")) }, options)
		if result.Err != nil {
			s.fail(result.Err)
		}
		if result.Completed && !s.failed() {
			s.send([]byte("data: [DONE]\n\n"))
		}
	} else {
		chunks := func() (string, bool) {
			for !s.failed() && !s.closing() {
				frame, err := s.upstream.Next()
				if err != nil {
					if !errors.Is(err, io.EOF) {
						s.fail(err)
					}
					return "", false
				}
				if payload, ok := ssePayload(frame); ok {
					return payload, true
				}
			}
			return "", false
		}
		var result translate.StreamObservation
		switch id {
		case translate.ConversionMessagesToChat:
			result = translate.OpenAIStreamToAnthropicSSEObserved(chunks, model, func(event string) { s.send([]byte(event)) }, options)
		case translate.ConversionResponsesToChat:
			result = translate.OpenAIStreamToResponsesSSEObserved(chunks, model, request, func(event string) { s.send([]byte(event)) }, options)
		case translate.ConversionChatToResponses:
			events := func() (string, bool) {
				value, ok := chunks()
				if value == "[DONE]" {
					return "", false
				}
				return value, ok
			}
			result = translate.ResponsesSSEToOpenAIChunksObserved(events, model, func(chunk string) { s.send([]byte("data: " + chunk + "\n\n")) }, options)
		default:
			s.fail(&core.CallPlanError{Reason: "stream_conversion_unsupported"})
			return
		}
		if result.Err != nil {
			s.fail(result.Err)
		}
		if id == translate.ConversionChatToResponses && result.Completed && !s.failed() {
			s.send([]byte("data: [DONE]\n\n"))
		}
	}
}

func ssePayload(frame []byte) (string, bool) {
	var data []string
	for _, line := range strings.Split(strings.TrimRight(string(frame), "\r\n"), "\n") {
		line = strings.TrimRight(line, "\r")
		if value, ok := strings.CutPrefix(line, "data:"); ok {
			data = append(data, strings.TrimPrefix(value, " "))
		}
	}
	if len(data) == 0 {
		return "", false
	}
	return strings.Join(data, "\n"), true
}

func (s *preparedObservedStream) Next() ([]byte, error) {
	if frame, ok := <-s.frames; ok {
		return frame, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.err != nil {
		return nil, s.err
	}
	return nil, io.EOF
}
func (s *preparedObservedStream) Close() error {
	s.once.Do(func() { close(s.stop) })
	err := s.upstream.Close()
	for range s.frames {
	}
	<-s.finished
	return err
}
func (s *preparedObservedStream) Losses() []core.Loss {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.losses)
}

// Source evidence belongs to the native attempt beneath every conversion leg,
// independently of whether an intermediate Chat stream emits a usage chunk.
func (s *preparedObservedStream) SourceUsage() (core.SourceUsageObservation, bool) {
	return core.SourceUsageOf(s.upstream)
}

func (s *preparedObservedStream) SourceUsageHistory() core.SourceUsageHistory {
	return core.SourceUsageHistoryOf(s.upstream)
}
