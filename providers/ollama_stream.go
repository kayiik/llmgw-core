package providers

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"slices"
	"strings"
	"sync"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/internal/sourceusage"
)

// ollamaStream converts Ollama's NDJSON stream to Chat chunks as the
// gateway does, each an SSE frame, and ends it with [DONE] once Ollama
// reports it done or closes it. An event with neither content, tool calls
// nor done is skipped, as is a line that is not a JSON object. The event
// that reports done ends the stream: with content it is the last content
// chunk, and without it a chunk that finishes with stop. A line is bounded
// to maxStreamRecordWireSize.
type ollamaStream struct {
	ctx                                 context.Context
	body                                io.ReadCloser
	reader                              *bufio.Reader
	model                               string
	completionID                        string
	losses                              []core.Loss
	emitted, done, ended                bool
	err                                 error
	sourceRecord                        *core.SourceUsageRecorder
	sourceUsage                         *preparedNativeUsage
	sourceMu                            sync.Mutex
	nativeDone, sourceEOF, nativeFailed bool
}

var _ core.LossReporter = (*ollamaStream)(nil)

func newOllamaStream(ctx context.Context, body io.ReadCloser, model string, losses []core.Loss) *ollamaStream {
	return &ollamaStream{ctx: ctx, body: body, reader: bufio.NewReader(body), model: model, losses: losses, completionID: newChatCompletionID("ollama")}
}

func (s *ollamaStream) Next() ([]byte, error) {
	for s.err == nil && !s.done {
		line, err := readBoundedLine(s.reader, maxStreamRecordWireSize, "NDJSON")
		if err == io.EOF {
			s.sourceMu.Lock()
			s.sourceEOF = true
			s.sourceMu.Unlock()
		}
		var tooLarge *streamRecordTooLargeError
		if errors.As(err, &tooLarge) {
			s.err = streamFailure(s.ctx, "Ollama", err)
			break
		}
		if frame := s.frame(line); frame != nil {
			s.sourceMu.Lock()
			nativeDone := s.nativeDone
			s.sourceMu.Unlock()
			if nativeDone {
				s.finishSource(nil, false)
			}
			return frame, nil
		}
		switch {
		case err == io.EOF && s.emitted:
			s.done = true
		case err == io.EOF:
			// The gateway reports a stream that ends before any content,
			// and before done, as an empty answer another target may serve.
			s.err = &core.ProviderError{
				Message: "Ollama returned an empty response", Class: core.ProviderErrorUpstream,
				Classification: core.ProviderErrorClassification{FailoverEligible: true},
			}
		case err != nil:
			// Unlike a relayed SSE stream, the gateway lets a broken Ollama
			// stream count against the provider and be repeated.
			s.err = transportFailure(s.ctx, "the Ollama stream broke off", err)
		}
	}
	switch {
	case s.err != nil:
		s.finishSource(s.err, false)
		return nil, s.err
	case s.ended:
		s.finishSource(nil, false)
		return nil, io.EOF
	}
	s.ended = true
	s.finishSource(nil, false)
	return []byte("data: [DONE]\n\n"), nil
}

// frame converts one NDJSON line, or returns nil for a line with nothing to
// relay.
func (s *ollamaStream) frame(line []byte) []byte {
	text := strings.TrimSpace(string(line))
	var event map[string]any
	decodeErr := json.Unmarshal([]byte(text), &event)
	if s.sourceRecord != nil {
		event, decodeErr = sourceusage.Decode([]byte(text))
	}
	if text == "" || decodeErr != nil {
		if text != "" && s.sourceRecord != nil {
			s.sourceMu.Lock()
			s.nativeFailed = true
			s.sourceMu.Unlock()
		}
		return nil
	}
	if s.sourceUsage != nil {
		s.sourceUsage.merge(ollamaNativeUsage(event))
		if event["error"] != nil {
			s.sourceMu.Lock()
			s.nativeFailed = true
			s.sourceMu.Unlock()
		}
	}
	message, _ := event["message"].(map[string]any)
	content, _ := message["content"].(string)
	toolCalls := ollamaToolCalls(message["tool_calls"])
	done, _ := event["done"].(bool)
	s.sourceMu.Lock()
	s.nativeDone = s.nativeDone || done
	s.sourceMu.Unlock()
	s.done = s.done || done
	switch {
	case content != "" || toolCalls != nil:
		delta := map[string]any{}
		if content != "" {
			delta["content"] = content
		}
		if toolCalls != nil {
			delta["tool_calls"] = toolCalls
		}
		s.emitted = true
		return ollamaChunk(s.model, s.completionID, delta, nil)
	case done:
		return ollamaChunk(s.model, s.completionID, map[string]any{}, "stop")
	}
	return nil
}

// ollamaChunk is one Chat chunk as the gateway encodes it, framed as SSE.
func ollamaChunk(model, completionID string, delta map[string]any, finish any) []byte {
	chunk, _ := json.Marshal(map[string]any{
		"id": completionID, "object": "chat.completion.chunk", "model": model,
		"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
	})
	return []byte("data: " + string(chunk) + "\n\n")
}

func (s *ollamaStream) Close() error { err := s.body.Close(); s.finishSource(err, true); return err }

func (s *ollamaStream) finishSource(err error, closing bool) {
	if s.sourceRecord == nil {
		return
	}
	quality := translate.CompletionPartial
	cancelled := s.ctx.Err() != nil
	s.sourceMu.Lock()
	failed, done, eof := s.nativeFailed, s.nativeDone, s.sourceEOF
	s.sourceMu.Unlock()
	if failed {
		quality = translate.CompletionFailed
		cancelled = false
	} else if done {
		quality = translate.CompletionComplete
		cancelled = false
	} else if eof {
		quality = translate.CompletionTruncated
	} else if err != nil && !cancelled {
		quality = translate.CompletionFailed
	} else if !closing {
		quality = translate.CompletionPartial
	}
	s.sourceRecord.Finish(quality, cancelled)
}
func (s *ollamaStream) SourceUsage() (core.SourceUsageObservation, bool) {
	if s.sourceRecord == nil {
		return core.SourceUsageObservation{}, false
	}
	return s.sourceRecord.SourceUsage()
}
func (s *ollamaStream) SourceUsageHistory() core.SourceUsageHistory {
	if s.sourceRecord == nil {
		return core.SourceUsageHistory{}
	}
	return s.sourceRecord.SourceUsageHistory()
}

// Losses implements core.LossReporter. They are known before the first
// frame.
func (s *ollamaStream) Losses() []core.Loss { return slices.Clone(s.losses) }
