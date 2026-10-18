package providers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"sync"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

// streamChat performs one Chat Completions request over
// streamGenerateContent, whose answer Google sends as server-sent events
// when asked with alt=sse, and returns it as Chat chunks, as googleStream
// renders them. The request is mapped as Invoke maps it, and the stream
// reports its losses through core.LossReporter.
func (p *Google) streamChat(ctx context.Context, request core.Request) (core.StreamIter, error) {
	call, err := googleChatPayload(request, true)
	if err != nil {
		return nil, err
	}
	if err := (core.LossPolicy{}).Check(call.losses); err != nil {
		return nil, err
	}
	access, err := p.access(request.Credential)
	if err != nil {
		return nil, err
	}
	endpoint, err := p.modelURL(request.Model, access.project, "streamGenerateContent")
	if err != nil {
		return nil, err
	}
	authorization, err := p.authorization(access)
	if err != nil {
		return nil, err
	}
	response, err := p.open(ctx, authorization, endpoint+"?alt=sse", call.payload)
	if err != nil {
		return nil, err
	}
	return &googleStream{
		ctx: ctx, provider: p, model: request.Model, body: response.Body, reader: newSSERecordReader(response.Body),
		includeUsage: call.includeUsage, losses: call.losses,
		completionID: newChatCompletionID("google"),
	}, nil
}

// googleStream renders Gemini's stream as Chat chunks, one SSE record per
// frame: each event's text and function calls as one delta, the first with
// the assistant role, then, once Google ends the stream, a chunk with the
// finish reason googleFinishReason names, the usage of Google's last count
// when the request asked for it with include_usage, and [DONE]. Every chunk
// names the model Google reports serving, or else the request's, and a tool
// call arrives whole in one delta, its arguments a JSON string.
//
// A stream that ends without a finish reason, or that sends an event that
// is not JSON, cannot be used, nor can an error Google sends in place of an
// event, which is classified by its status. A stream that ends with nothing
// to show fails as Invoke fails on an empty answer, and nothing has reached
// the caller then, so another target may serve the request.
type googleStream struct {
	ctx          context.Context
	provider     *Google
	model        string
	body         io.ReadCloser
	reader       *sseRecordReader
	includeUsage bool
	losses       []core.Loss
	// frames are the frames ready to return.
	frames                    [][]byte
	modelKnown                bool
	responseID                string
	completionID              string
	exactNumbers              bool
	responseAssessment        func(map[string]any) core.CallAssessment
	sourceRecord              *core.SourceUsageRecorder
	sourceUsage               *preparedNativeUsage
	sourceMu                  sync.Mutex
	sourceComplete, sourceEOF bool
	sourceTruncated           bool
	sourceFailed              bool
	// calls counts the tool calls sent, which number the next ones.
	calls int
	// output reports a delta was sent, finish the reason Google gave, if
	// any, and usage Google's last count.
	output  bool
	finish  string
	usage   map[string]any
	ended   bool
	failure error
}

var (
	_ core.StreamIter   = (*googleStream)(nil)
	_ core.LossReporter = (*googleStream)(nil)
)

// Next returns the next frame. A failure repeats on every later call.
func (s *googleStream) Next() ([]byte, error) {
	for len(s.frames) == 0 {
		switch {
		case s.failure != nil:
			return nil, s.failure
		case s.ended:
			return nil, io.EOF
		}
		s.failure = s.advance()
		if s.failure != nil || s.ended {
			s.finishSource(s.failure, false)
		}
	}
	frame := s.frames[0]
	s.frames = s.frames[1:]
	return frame, nil
}

func (s *googleStream) Close() error { err := s.body.Close(); s.finishSource(err, true); return err }

func (s *googleStream) finishSource(err error, closing bool) {
	if s.sourceRecord == nil {
		return
	}
	quality := translate.CompletionPartial
	cancelled := s.ctx.Err() != nil
	s.sourceMu.Lock()
	complete, eof, truncated, failed := s.sourceComplete, s.sourceEOF, s.sourceTruncated, s.sourceFailed
	s.sourceMu.Unlock()
	if failed {
		quality = translate.CompletionFailed
		cancelled = false
	} else if complete {
		quality = translate.CompletionComplete
		cancelled = false
		if truncated {
			quality = translate.CompletionTruncated
		}
	} else if eof {
		quality = translate.CompletionTruncated
	} else if err != nil && !cancelled {
		quality = translate.CompletionFailed
	} else if !closing && !cancelled {
		quality = translate.CompletionPartial
	}
	s.sourceRecord.Finish(quality, cancelled)
}
func (s *googleStream) SourceUsage() (core.SourceUsageObservation, bool) {
	if s.sourceRecord == nil {
		return core.SourceUsageObservation{}, false
	}
	return s.sourceRecord.SourceUsage()
}
func (s *googleStream) SourceUsageHistory() core.SourceUsageHistory {
	if s.sourceRecord == nil {
		return core.SourceUsageHistory{}
	}
	return s.sourceRecord.SourceUsageHistory()
}

// Losses implements core.LossReporter.
func (s *googleStream) Losses() []core.Loss { return s.losses }

// advance reads one event and queues the frames it makes.
func (s *googleStream) advance() error {
	label := s.provider.label()
	data, err := s.reader.NextData()
	if err == io.EOF {
		s.sourceMu.Lock()
		s.sourceEOF = true
		s.sourceMu.Unlock()
		return s.end()
	}
	if err != nil {
		return streamFailure(s.ctx, label, err)
	}
	var event map[string]any
	var decodeErr error
	if s.exactNumbers {
		event, decodeErr = preparedJSON(core.Request{Body: []byte(data), ContentType: core.ContentTypeJSON})
	} else {
		decodeErr = json.Unmarshal([]byte(data), &event)
	}
	if decodeErr != nil || event == nil {
		s.sourceMu.Lock()
		s.sourceFailed = true
		s.sourceMu.Unlock()
		return unusableResponse("the "+label+" stream sent an event that is not JSON", nil)
	}
	if s.sourceUsage != nil {
		if usage, set := event["usageMetadata"]; set {
			s.sourceUsage.merge(usage)
		}
		if reason := googleCandidateFinish(event); reason != "" {
			s.sourceMu.Lock()
			s.sourceComplete = true
			s.sourceTruncated = reason == "MAX_TOKENS"
			s.sourceMu.Unlock()
		}
	}
	if envelope, ok := event["error"].(map[string]any); ok {
		s.sourceMu.Lock()
		s.sourceFailed = true
		s.sourceMu.Unlock()
		return s.refusal(envelope, data, event)
	}
	if s.responseAssessment != nil {
		a := s.responseAssessment(event)
		s.losses = append(s.losses, a.Losses...)
		if err := (core.LossPolicy{}).Check(a.Losses); err != nil {
			return err
		}
		if a.Feasibility != core.SupportSupported {
			return &core.CallPlanError{Reason: "google_response_semantics_unassessed", Losses: a.Losses, UnknownPaths: a.UnknownPaths}
		}
	}
	if version, _ := event["modelVersion"].(string); version != "" && !s.modelKnown {
		s.model, s.modelKnown = version, true
	}
	if s.responseID == "" {
		s.responseID = googleResponseID(event)
	}
	if usage, ok := event["usageMetadata"].(map[string]any); ok {
		s.usage = usage
	}
	parts := googleParts(event)
	delta := map[string]any{}
	if text := googleText(parts); text != "" {
		delta["content"] = text
	}
	if calls := googleToolCalls(parts, s.responseID, s.calls); len(calls) > 0 {
		for index, call := range calls {
			call["index"] = s.calls + index
		}
		s.calls += len(calls)
		delta["tool_calls"] = calls
	}
	if len(delta) > 0 {
		if !s.output {
			delta["role"] = "assistant"
			s.output = true
		}
		s.frames = append(s.frames, s.chunk(delta, nil))
	}
	if reason := googleCandidateFinish(event); reason != "" {
		s.finish = reason
	} else if feedback, _ := event["promptFeedback"].(map[string]any); feedback != nil {
		// A blocked prompt gets no candidate, and so no finish reason.
		if reason, _ := feedback["blockReason"].(string); reason != "" {
			s.finish = reason
		}
	}
	return nil
}

// end queues the frames that close a stream Google ended, or reports why it
// cannot be used.
func (s *googleStream) end() error {
	label := s.provider.label()
	switch {
	case s.finish == "":
		return unusableResponse("the "+label+" stream ended before the model finished", nil)
	case !s.output:
		return googleEmptyReplyError(label, map[string]any{
			"candidates": []any{map[string]any{"finishReason": s.finish}}, "usageMetadata": s.usage,
		})
	}
	s.frames = append(s.frames, s.chunk(map[string]any{}, googleFinishReason(s.finish, s.calls > 0)))
	if s.includeUsage {
		input, output, total := googleUsage(map[string]any{"usageMetadata": s.usage})
		s.frames = append(s.frames, googleFrame(map[string]any{
			"id": s.completionID, "object": "chat.completion.chunk", "model": s.model, "choices": []any{},
			"usage": map[string]any{"prompt_tokens": input, "completion_tokens": output, "total_tokens": total},
		}))
	}
	s.frames = append(s.frames, []byte("data: [DONE]\n\n"))
	s.ended = true
	return nil
}

// refusal reports an error Google sent in place of an event, classified by
// the status its envelope names as a refusal with that status would be. An
// envelope without a status cannot be classified, so the stream is unusable.
func (s *googleStream) refusal(envelope map[string]any, data string, event map[string]any) error {
	code, _ := envelope["code"].(float64)
	if number, ok := envelope["code"].(json.Number); ok {
		if value, err := number.Int64(); err == nil {
			code = float64(value)
		}
	}
	status := int(code)
	if status < http.StatusBadRequest || status > 599 || float64(status) != code {
		return unusableResponse("the "+s.provider.label()+" stream reported an error", nil)
	}
	return s.provider.statusFailure(&http.Response{StatusCode: status, Header: http.Header{}}, []byte(data), event)
}

// chunk renders one Chat chunk with a delta and a finish reason.
func (s *googleStream) chunk(delta map[string]any, finish any) []byte {
	return googleFrame(map[string]any{
		"id": s.completionID, "object": "chat.completion.chunk", "model": s.model,
		"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
	})
}

// googleFrame renders a chunk as one SSE record.
func googleFrame(chunk map[string]any) []byte {
	encoded, _ := json.Marshal(chunk)
	return []byte("data: " + string(encoded) + "\n\n")
}
