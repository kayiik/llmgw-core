// Package sourceusage observes the actual native protocol before conversion.
// It never reads target codec placeholders or adds wire usage events.
package sourceusage

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

func Decode(raw []byte) (map[string]any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value map[string]any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	if value == nil {
		return nil, errors.New("native observation is not a JSON object")
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return nil, errors.New("native observation contains trailing JSON")
	}
	return value, nil
}

// Response observes the source object before validation/conversion can discard
// it. Native errors may still carry evidence, which remains source evidence.
func Response(record *core.SourceUsageRecorder, raw []byte) {
	if value, err := Decode(raw); err == nil {
		if usage, set := value["usage"]; set {
			record.Merge(usage)
		}
	}
}

func Finish(record *core.SourceUsageRecorder, ctx context.Context, quality translate.CompletionQuality) {
	if ctx.Err() != nil {
		record.Finish(translate.CompletionPartial, true)
	} else {
		record.Finish(quality, false)
	}
}

type stream struct {
	core.StreamIter
	ctx                         context.Context
	api                         core.UpstreamAPI
	record                      *core.SourceUsageRecorder
	mu                          sync.Mutex
	terminal, failed, truncated bool
}

func Stream(ctx context.Context, upstream core.StreamIter, api core.UpstreamAPI, record *core.SourceUsageRecorder) core.StreamIter {
	return &stream{StreamIter: upstream, ctx: ctx, api: api, record: record}
}

func payload(frame []byte) string {
	var lines []string
	for _, line := range strings.Split(string(frame), "\n") {
		line = strings.TrimRight(line, "\r")
		if value, ok := strings.CutPrefix(line, "data:"); ok {
			lines = append(lines, strings.TrimPrefix(value, " "))
		}
	}
	return strings.Join(lines, "\n")
}

func (s *stream) observe(frame []byte) {
	data := payload(frame)
	if strings.TrimSpace(data) == "" {
		return
	}
	if strings.TrimSpace(data) == "[DONE]" {
		s.mu.Lock()
		s.terminal = true
		s.mu.Unlock()
		return
	}
	value, err := Decode([]byte(data))
	if err != nil {
		s.mu.Lock()
		s.failed = true
		s.mu.Unlock()
		return
	}
	if usage, set := value["usage"]; set {
		s.record.Merge(usage)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if value["error"] != nil {
		s.failed = true
		s.terminal = true
	}
	switch s.api {
	case core.UpstreamAPIResponses:
		if response, ok := value["response"].(map[string]any); ok {
			if usage, set := response["usage"]; set {
				s.record.Merge(usage)
			}
		}
		switch value["type"] {
		case "response.completed":
			s.terminal = true
		case "response.incomplete":
			s.terminal = true
			s.truncated = true
		case "response.failed", "error":
			s.terminal = true
			s.failed = true
		}
	case core.UpstreamAPIMessages:
		if message, ok := value["message"].(map[string]any); ok {
			if usage, set := message["usage"]; set {
				s.record.Merge(usage)
			}
		}
		switch value["type"] {
		case "message_stop":
			s.terminal = true
		case "error":
			s.terminal = true
			s.failed = true
		}
		if delta, ok := value["delta"].(map[string]any); ok && delta["stop_reason"] == "max_tokens" {
			s.truncated = true
		}
	case core.UpstreamAPIChatCompletions:
		if choices, ok := value["choices"].([]any); ok {
			for _, raw := range choices {
				choice, _ := raw.(map[string]any)
				if choice["finish_reason"] == "length" {
					s.truncated = true
				}
			}
		}
	}
}

func (s *stream) finish(err error, closing bool) {
	s.mu.Lock()
	terminal, failed, truncated := s.terminal, s.failed, s.truncated
	s.mu.Unlock()
	if failed {
		s.record.Finish(translate.CompletionFailed, false)
		return
	}
	if terminal {
		quality := translate.CompletionComplete
		if truncated {
			quality = translate.CompletionTruncated
		}
		s.record.Finish(quality, false)
		return
	}
	if s.ctx.Err() != nil {
		s.record.Finish(translate.CompletionPartial, true)
		return
	}
	quality := translate.CompletionTruncated
	if err != nil && !errors.Is(err, io.EOF) {
		quality = translate.CompletionFailed
	} else if closing {
		quality = translate.CompletionPartial
	}
	s.record.Finish(quality, false)
}

func (s *stream) Next() ([]byte, error) {
	frame, err := s.StreamIter.Next()
	if len(frame) > 0 {
		s.observe(frame)
	}
	if err != nil {
		s.finish(err, false)
	}
	return frame, err
}
func (s *stream) Close() error                                     { err := s.StreamIter.Close(); s.finish(err, true); return err }
func (s *stream) Losses() []core.Loss                              { return core.StreamLosses(s.StreamIter) }
func (s *stream) SourceUsage() (core.SourceUsageObservation, bool) { return s.record.SourceUsage() }
func (s *stream) SourceUsageHistory() core.SourceUsageHistory      { return s.record.SourceUsageHistory() }
