package providers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

// googleFixture reads a fixture under testdata/google with LF line ends,
// whatever the checkout made of them.
func googleFixture(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "google", name))
	if err != nil {
		t.Fatal(err)
	}
	return strings.ReplaceAll(string(raw), "\r\n", "\n")
}

// googleWire is a stream fixture as Google sends it, every line ended by
// CRLF.
func googleWire(fixture string) string { return strings.ReplaceAll(fixture, "\n", "\r\n") }

// googleStreamCall is one request a fake streaming Google received.
type googleStreamCall struct {
	path, query, accept, apiKey, authorization, requestType, body string
}

// newGoogleStreamFake is a synthetic Google API that answers every request
// with status and body, as an event stream when the status is 200, and
// records each request.
func newGoogleStreamFake(t *testing.T, status int, body string) (func() []googleStreamCall, string) {
	t.Helper()
	var mu sync.Mutex
	var calls []googleStreamCall
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		mu.Lock()
		calls = append(calls, googleStreamCall{
			path: r.URL.Path, query: r.URL.RawQuery, accept: r.Header.Get("Accept"), apiKey: r.Header.Get("x-goog-api-key"),
			authorization: r.Header.Get("Authorization"), requestType: r.Header.Get("X-Vertex-AI-LLM-Request-Type"), body: string(raw),
		})
		mu.Unlock()
		if status == http.StatusOK {
			w.Header().Set("Content-Type", core.ContentTypeEventStream)
		} else {
			w.Header().Set("Content-Type", core.ContentTypeJSON)
		}
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(server.Close)
	return func() []googleStreamCall {
		mu.Lock()
		defer mu.Unlock()
		taken := calls
		calls = nil
		return taken
	}, server.URL
}

// googleFrames drains a stream: its frames, joined, and the error that
// ended it unless that was io.EOF. A failure must repeat on the next call.
func googleFrames(t *testing.T, stream core.StreamIter) (string, error) {
	t.Helper()
	defer stream.Close()
	var frames strings.Builder
	for {
		frame, err := stream.Next()
		if err == io.EOF {
			return frames.String(), nil
		}
		if err != nil {
			if _, again := stream.Next(); again != err {
				t.Fatalf("the stream failed with %v, then with %v", err, again)
			}
			return frames.String(), err
		}
		if !strings.HasPrefix(string(frame), "data: ") || !strings.HasSuffix(string(frame), "\n\n") || strings.Count(string(frame), "\n\n") != 1 {
			t.Fatalf("frame %q is not one SSE record", frame)
		}
		frames.Write(frame)
	}
}

// A Chat stream is sent to streamGenerateContent with alt=sse, mapped as
// Invoke maps the request, and Google's events come back as Chat chunks:
// one per event that carries text, the first with the role, then the
// finish reason, the usage the request asked for, and [DONE]. Google
// separates its events with CRLF; a text part that only carries a thought
// signature is no output.
func TestGoogleStreamsTextAsChatChunks(t *testing.T) {
	t.Parallel()
	take, base := newGoogleStreamFake(t, http.StatusOK, googleWire(googleFixture(t, "stream-text.sse")))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	stream, err := provider.Stream(context.Background(), googleChatRequest("gemini-3.5-flash", `{"model":"gemini-3.5-flash","stream":true,`+
		`"stream_options":{"include_usage":true},"max_tokens":256,"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"Why is the sky blue?"}]}`, googleKey("secret")))
	if err != nil {
		t.Fatal(err)
	}
	frames, err := googleFrames(t, stream)
	if err != nil {
		t.Fatal(err)
	}
	if want := googleFixture(t, "stream-text.chat.golden"); completionGolden(t, frames, "google") != want {
		t.Fatalf("frames:\n%s\nwant:\n%s", frames, want)
	}
	want := googleStreamCall{
		path: "/models/gemini-3.5-flash:streamGenerateContent", query: "alt=sse", accept: core.ContentTypeEventStream, apiKey: "secret",
		body: `{"contents":[{"parts":[{"text":"Why is the sky blue?"}],"role":"user"}],"generationConfig":{"maxOutputTokens":256},` +
			`"systemInstruction":{"parts":[{"text":"be brief"}]}}`,
	}
	if calls := take(); !reflect.DeepEqual(calls, []googleStreamCall{want}) {
		t.Fatalf("upstream = %+v", calls)
	}
	if losses := core.StreamLosses(stream); len(losses) != 0 {
		t.Fatalf("losses = %+v", losses)
	}
}

// A function call arrives whole, as Google sends it: a tool call delta
// with its arguments as a JSON string, an ID unique to the answer, and the
// thought signature at extra_content.google, and the stream finishes with
// tool_calls. The calls, returned on the next turn with their results,
// reach Google as the model made them: the signature on the first call and
// none on the second, then every result in one user turn.
func TestGoogleStreamsToolCallsAndTakesThemBack(t *testing.T) {
	t.Parallel()
	take, base := newGoogleStreamFake(t, http.StatusOK, googleWire(googleFixture(t, "stream-tool-calls.sse")))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	const tools = `"tools":[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city",` +
		`"parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},` +
		`"required":["city"],"additionalProperties":false}}}]`
	const question = `{"role":"user","content":"Weather in Paris and Rome?"}`
	stream, err := provider.Stream(context.Background(), googleChatRequest("gemini-3.5-flash", `{"stream":true,"messages":[`+question+`],`+tools+`}`, googleKey("k")))
	if err != nil {
		t.Fatal(err)
	}
	frames, err := googleFrames(t, stream)
	if err != nil {
		t.Fatal(err)
	}
	if want := googleFixture(t, "stream-tool-calls.chat.golden"); completionGolden(t, frames, "google") != want {
		t.Fatalf("frames:\n%s\nwant:\n%s", frames, want)
	}
	const declarations = `"tools":[{"functionDeclarations":[{"description":"Current weather for a city","name":"get_weather",` +
		`"parametersJsonSchema":{"additionalProperties":false,"properties":{"city":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"],"type":"string"}},` +
		`"required":["city"],"type":"object"}}]}]`
	if calls := take(); len(calls) != 1 || calls[0].body != `{"contents":[{"parts":[{"text":"Weather in Paris and Rome?"}],"role":"user"}],`+declarations+`}` {
		t.Fatalf("upstream = %+v", calls)
	}

	// The client returns the assistant message it assembled, with the
	// results of both calls.
	var calls []map[string]any
	for _, frame := range strings.Split(strings.TrimSpace(frames), "\n\n") {
		var chunk struct {
			Choices []struct {
				Delta struct {
					ToolCalls []map[string]any `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(strings.TrimPrefix(frame, "data: ")), &chunk) == nil && len(chunk.Choices) > 0 {
			calls = append(calls, chunk.Choices[0].Delta.ToolCalls...)
		}
	}
	assistant, _ := json.Marshal(map[string]any{"role": "assistant", "content": "Checking both cities.", "tool_calls": calls})
	followUp := `{"messages":[` + question + `,` + string(assistant) + `,` +
		`{"role":"tool","tool_call_id":"call_resp-tools-fixture_0","content":"{\"temperature\":21,\"sky\":\"clear\"}"},` +
		`{"role":"tool","tool_call_id":"call_resp-tools-fixture_1","content":[{"type":"text","text":"18 degrees, "},{"type":"text","text":"cloudy"}]}],` + tools + `}`
	take, base = newGoogleStreamFake(t, http.StatusOK, googleWire(googleFixture(t, "stream-text.sse")))
	provider = newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	stream, err = provider.Stream(context.Background(), googleChatRequest("gemini-3.5-flash", followUp, googleKey("k")))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := googleFrames(t, stream); err != nil {
		t.Fatal(err)
	}
	want := `{"contents":[{"parts":[{"text":"Weather in Paris and Rome?"}],"role":"user"},` +
		`{"parts":[{"text":"Checking both cities."},` +
		`{"functionCall":{"args":{"city":"Paris","unit":"celsius"},"name":"get_weather"},"thoughtSignature":"c2lnbmF0dXJlLWNhbGwtZml4dHVyZQ=="},` +
		`{"functionCall":{"args":{"city":"Rome","unit":"celsius"},"name":"get_weather"}}],"role":"model"},` +
		`{"parts":[{"functionResponse":{"name":"get_weather","response":{"sky":"clear","temperature":21}}},` +
		`{"functionResponse":{"name":"get_weather","response":{"result":"18 degrees, cloudy"}}}],"role":"user"}],` + declarations + `}`
	if calls := take(); len(calls) != 1 || calls[0].body != want {
		t.Fatalf("upstream = %+v\nwant body %s", calls, want)
	}
	if losses := core.StreamLosses(stream); len(losses) != 0 {
		t.Fatalf("losses = %+v", losses)
	}
}

// A stream ends as Google ends it. A finish reason other than STOP names
// Chat's; a stream cut short, an event that is not JSON and an error event
// without a status cannot be used; an error event is classified by its
// status; and a stream with nothing to show fails before any frame, as
// Invoke fails on an empty answer, so another target may serve it.
func TestGoogleStreamEndsAsGoogleEndsIt(t *testing.T) {
	t.Parallel()
	const event = `data: {"candidates": [{"content": {"parts": [{"text": "Hello"}],"role": "model"},"index": 0}],"modelVersion": "gemini-3.5-flash"}` + "\n\n"
	const hello = `data: {"choices":[{"delta":{"content":"Hello","role":"assistant"},"finish_reason":null,"index":0}],"id":"chatcmpl-google","model":"gemini-3.5-flash","object":"chat.completion.chunk"}` + "\n\n"
	finished := func(reason string) string {
		return `data: {"choices":[{"delta":{},"finish_reason":"` + reason + `","index":0}],"id":"chatcmpl-google","model":"gemini-3.5-flash","object":"chat.completion.chunk"}` + "\n\ndata: [DONE]\n\n"
	}
	unusable := core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}
	for _, testCase := range []struct {
		name, wire, frames, message string
		class                       core.ProviderErrorClass
		classification              core.ProviderErrorClassification
	}{
		{
			name:   "output limit",
			wire:   event + `data: {"candidates": [{"content": {"parts": [{"text": ""}],"role": "model"},"finishReason": "MAX_TOKENS","index": 0}]}` + "\n\n",
			frames: hello + finished("length"),
		},
		{
			name:   "safety",
			wire:   event + `data: {"candidates": [{"finishReason": "SAFETY","index": 0}]}` + "\n\n",
			frames: hello + finished("content_filter"),
		},
		{name: "a proxy's [DONE]", wire: event + `data: {"candidates": [{"finishReason": "STOP"}]}` + "\n\ndata: [DONE]\n\n", frames: hello + finished("stop")},
		{
			name: "cut short", wire: event, frames: hello, message: "the ai_studio stream ended before the model finished",
			class: core.ProviderErrorUpstream, classification: unusable,
		},
		{
			name: "not JSON", wire: event + "data: {\"candidates\": [\n\n", frames: hello, message: "the ai_studio stream sent an event that is not JSON",
			class: core.ProviderErrorUpstream, classification: unusable,
		},
		{
			name: "error event", wire: googleFixture(t, "stream-error.sse"),
			frames: strings.Replace(hello, "Hello", "The first part of the answer", 1), message: "ai_studio returned HTTP 503 (UNAVAILABLE)",
			class: core.ProviderErrorUpstream, classification: core.ProviderErrorClassification{StatusCode: 503, Retryable: true, FailoverEligible: true, CircuitFailure: true},
		},
		{
			name: "error event without a status", wire: event + `data: {"error": {"message": "something broke"}}` + "\n\n", frames: hello,
			message: "the ai_studio stream reported an error", class: core.ProviderErrorUpstream, classification: unusable,
		},
		{
			name: "reasoning spent the budget",
			wire: `data: {"candidates": [{"content": {"role": "model"},"finishReason": "MAX_TOKENS","index": 0}],` +
				`"usageMetadata": {"promptTokenCount": 7,"totalTokenCount": 39,"thoughtsTokenCount": 32}}` + "\n\n",
			message: "ai_studio: the model spent its entire output budget on reasoning (32 thinking tokens) and returned no text — raise max_tokens or omit it",
			class:   core.ProviderErrorUpstream, classification: core.ProviderErrorClassification{FailoverEligible: true},
		},
		{
			name:    "blocked prompt",
			wire:    `data: {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},"usageMetadata": {"promptTokenCount": 7,"totalTokenCount": 7}}` + "\n\n",
			message: "ai_studio: the model returned no text (finish reason PROHIBITED_CONTENT)",
			class:   core.ProviderErrorUpstream, classification: core.ProviderErrorClassification{FailoverEligible: true},
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			_, base := newGoogleStreamFake(t, http.StatusOK, googleWire(testCase.wire))
			provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
			stream, err := provider.Stream(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[{"role":"user","content":"hi"}]}`, googleKey("k")))
			if err != nil {
				t.Fatal(err)
			}
			frames, err := googleFrames(t, stream)
			if frames != "" {
				frames = completionGolden(t, frames, "google")
			}
			if frames != testCase.frames {
				t.Fatalf("frames:\n%s\nwant:\n%s", frames, testCase.frames)
			}
			if testCase.message == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			var failure *core.ProviderError
			if !errors.As(err, &failure) || err.Error() != testCase.message || failure.Class != testCase.class || failure.Classification != testCase.classification {
				t.Fatalf("error = %#v, want %q, class %s and %+v", err, testCase.message, testCase.class, testCase.classification)
			}
		})
	}
}

// A refusal arrives before any event, and is reported as Invoke reports it,
// with the wait Google's RetryInfo asks for. Vertex AI streams from its
// own route, with the instance's request type and the bearer.
func TestGoogleStreamRefusalsAndVertexAI(t *testing.T) {
	t.Parallel()
	take, base := newGoogleStreamFake(t, http.StatusTooManyRequests, `{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED",`+
		`"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"7s"}]}}`)
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Project: "fixture-project", RequestType: "paygo"})
	stream, err := provider.Stream(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[{"role":"user","content":"hi"}]}`, googleBearer("token")))
	classification := core.ClassifyError(err)
	if stream != nil || classification.StatusCode != http.StatusTooManyRequests || !classification.Retryable || classification.RetryAfter.Seconds() != 7 {
		t.Fatalf("stream = %v, err = %v, classification = %+v", stream, err, classification)
	}
	want := googleStreamCall{
		path:  "/v1/projects/fixture-project/locations/global/publishers/google/models/gemini-3.5-flash:streamGenerateContent",
		query: "alt=sse", accept: core.ContentTypeEventStream, authorization: "Bearer token", requestType: "shared",
		body: `{"contents":[{"parts":[{"text":"hi"}],"role":"user"}]}`,
	}
	if calls := take(); !reflect.DeepEqual(calls, []googleStreamCall{want}) {
		t.Fatalf("upstream = %+v", calls)
	}
}

// A stream that breaks off after it opened is a transport failure, which
// another target may get past while nothing has reached the caller.
func TestGoogleStreamThatBreaksOff(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", core.ContentTypeEventStream)
		_, _ = io.WriteString(w, "data: {\"candidates\": [{\"content\": {\"parts\": [{\"text\": \"Hel")
		w.(http.Flusher).Flush()
		panic(http.ErrAbortHandler)
	}))
	t.Cleanup(server.Close)
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: server.URL})
	stream, err := provider.Stream(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[{"role":"user","content":"hi"}]}`, googleKey("k")))
	if err != nil {
		t.Fatal(err)
	}
	frames, err := googleFrames(t, stream)
	var failure *core.ProviderError
	if frames != "" || !errors.As(err, &failure) || failure.Class != core.ProviderErrorTransport ||
		failure.Classification != (core.ProviderErrorClassification{FailoverEligible: true}) {
		t.Fatalf("frames = %q, err = %#v", frames, err)
	}
}

// A stream reports what its request lost, as Invoke does, but serves
// include_usage, which only a stream can.
func TestGoogleStreamReportsItsLosses(t *testing.T) {
	t.Parallel()
	_, base := newGoogleStreamFake(t, http.StatusOK, googleFixture(t, "stream-text.sse"))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	dropped := func(path, detail string) core.Loss {
		return core.Loss{Path: path, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: detail}
	}
	for body, want := range map[string][]core.Loss{
		`{"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`: nil,
		`{"stream_options":{"include_usage":false,"include_obfuscation":true},"top_p":0.5,"messages":[{"role":"user","content":"hi"}]}`: {
			dropped("stream_options", "Google does not send this Chat field"), dropped("top_p", "Google does not send this Chat field"),
		},
	} {
		stream, err := provider.Stream(context.Background(), googleChatRequest("gemini-3.5-flash", body, googleKey("k")))
		if err != nil {
			t.Fatal(err)
		}
		frames, err := googleFrames(t, stream)
		if err != nil {
			t.Fatal(err)
		}
		if losses := core.StreamLosses(stream); !reflect.DeepEqual(losses, want) {
			t.Fatalf("%s: losses = %+v, want %+v", body, losses, want)
		}
		if usage := strings.Contains(frames, `"usage"`); usage != strings.Contains(body, `"include_usage":true`) {
			t.Fatalf("%s: frames = %s", body, frames)
		}
	}
}

// Embeddings do not stream, and a surface Google does not serve is a
// surface error. Nothing is sent.
func TestGoogleStreamRefusesWhatDoesNotStream(t *testing.T) {
	t.Parallel()
	take, base := newGoogleStreamFake(t, http.StatusOK, "")
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	embeddings := core.Request{Surface: core.ModelSurfaceEmbeddings, Model: "gemini-embedding-001", Body: []byte(`{"input":"a"}`), ContentType: core.ContentTypeJSON}
	stream, err := provider.Stream(context.Background(), embeddings)
	var failure *core.ProviderError
	if stream != nil || !errors.As(err, &failure) || failure.Class != core.ProviderErrorUnsupported || core.ClassifyError(err).Disposition() != core.DispositionFailover {
		t.Fatalf("stream = %v, err = %v, want a refusal that permits failover", stream, err)
	}
	messages := core.Request{Surface: core.ModelSurfaceMessages, Model: "gemini-3.5-flash", Body: []byte(`{}`), ContentType: core.ContentTypeJSON}
	var surface *core.SurfaceError
	if _, err := provider.Stream(context.Background(), messages); !errors.As(err, &surface) {
		t.Fatalf("stream err = %v, want a surface error", err)
	}
	if _, err := provider.Invoke(context.Background(), messages); !errors.As(err, &surface) {
		t.Fatalf("invoke err = %v, want a surface error", err)
	}
	if calls := take(); len(calls) != 0 {
		t.Fatalf("upstream = %+v", calls)
	}
}
