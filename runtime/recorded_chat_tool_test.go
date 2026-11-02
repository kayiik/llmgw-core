package runtime_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	translate "github.com/kayiik/llm-translate"

	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/providers"
	coreruntime "github.com/kayiik/llmgw-core/runtime"
)

type recordedChatExchange struct {
	Method              string          `json:"method"`
	Path                string          `json:"path"`
	Request             json.RawMessage `json:"request"`
	Response            json.RawMessage `json:"response"`
	Status              int             `json:"status"`
	ContentType         string          `json:"contentType"`
	ResponseContentType string          `json:"responseContentType,omitempty"`
}

func recordedChatConversation(t *testing.T) []recordedChatExchange {
	t.Helper()
	raw, err := os.ReadFile("testdata/recorded_chat_tool_conversation.json")
	if err != nil {
		t.Fatal(err)
	}
	var exchanges []recordedChatExchange
	if err := json.Unmarshal(raw, &exchanges); err != nil {
		t.Fatal(err)
	}
	if len(exchanges) != 2 {
		t.Fatalf("recorded conversation has %d exchanges, want 2", len(exchanges))
	}
	return exchanges
}

func recordedChatJSON(raw []byte) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return nil, fmt.Errorf("extra data after recorded JSON: %v", err)
	}
	return value, nil
}

// This conversation's replay advances only on an exact semantic request match.
// In particular, a missing or changed tool_call_id cannot consume turn two.
type recordedChatReplay struct {
	mu                        sync.Mutex
	exchanges                 []recordedChatExchange
	calls, accepted, rejected int
}

func (replay *recordedChatReplay) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	replay.mu.Lock()
	defer replay.mu.Unlock()
	replay.calls++
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	actual, decodeErr := recordedChatJSON(body)
	if err == nil && decodeErr == nil && replay.accepted < len(replay.exchanges) {
		exchange := replay.exchanges[replay.accepted]
		expected, expectedErr := recordedChatJSON(exchange.Request)
		if expectedErr == nil && r.Method == exchange.Method && r.URL.RequestURI() == exchange.Path &&
			r.Header.Get("Content-Type") == exchange.ContentType && reflect.DeepEqual(actual, expected) {
			replay.accepted++
			responseContentType := exchange.ResponseContentType
			if responseContentType == "" {
				responseContentType = exchange.ContentType
			}
			w.Header().Set("Content-Type", responseContentType)
			w.WriteHeader(exchange.Status)
			_, _ = w.Write(exchange.Response)
			return
		}
	}
	replay.rejected++
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusUnprocessableEntity)
	_, _ = io.WriteString(w, `{"error":{"message":"request does not match recorded conversation","type":"invalid_request_error"}}`)
}

func (replay *recordedChatReplay) counts() (int, int, int) {
	replay.mu.Lock()
	defer replay.mu.Unlock()
	return replay.calls, replay.accepted, replay.rejected
}

// A per-client guard rejects other destinations before the underlying transport
// can resolve or dial them. No DefaultClient, environment or process mutation.
type recordedChatLoopback struct {
	host    string
	next    http.RoundTripper
	blocked atomic.Int32
}

func (transport *recordedChatLoopback) RoundTrip(request *http.Request) (*http.Response, error) {
	if request.URL.Scheme != "http" || request.URL.Host != transport.host {
		transport.blocked.Add(1)
		return nil, fmt.Errorf("recorded Chat replay forbids non-replay destination")
	}
	return transport.next.RoundTrip(request)
}

func newRecordedChatRuntime(t *testing.T, exchanges []recordedChatExchange) (*coreruntime.Runtime[struct{}], *recordedChatReplay, *http.Client, *recordedChatLoopback) {
	t.Helper()
	replay := &recordedChatReplay{exchanges: exchanges}
	server := httptest.NewServer(replay)
	t.Cleanup(server.Close)
	client := server.Client()
	transport := &recordedChatLoopback{host: server.Listener.Addr().String(), next: client.Transport}
	client.Transport = transport
	client.Timeout = 5 * time.Second
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	provider, err := providers.NewOpenAICompatible(providers.OpenAICompatibleConfig{
		BaseURL: server.URL + "/api/v1", Client: client, CatalogClient: client, ForwardAllFields: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := coreruntime.New(coreruntime.Options[struct{}]{
		Settings:  coreruntime.NewMemorySettings(struct{}{}),
		Providers: func(struct{}, string) (core.Provider, error) { return provider, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	return runtime, replay, client, transport
}

func recordedChatRequest(t *testing.T, body json.RawMessage) core.Request {
	t.Helper()
	var payload struct{ Model string }
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	return core.Request{Surface: core.ModelSurfaceChatCompletions, Model: payload.Model, Body: body, ContentType: core.ContentTypeJSON}
}

func invokeRecordedChat(t *testing.T, runtime *coreruntime.Runtime[struct{}], body json.RawMessage) (core.Response, *coreruntime.PreparedCall, error) {
	t.Helper()
	request := recordedChatRequest(t, body)
	prepared, err := runtime.PrepareCall(t.Context(), core.LocalCaller(), "recorded", request, core.CallPreparationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	description := prepared.Description()
	if description.UpstreamAPI != core.UpstreamAPIChatCompletions || len(description.Conversions) != 0 {
		t.Fatalf("recorded Chat was not native: %+v", description)
	}
	response, err := runtime.InvokePrepared(t.Context(), core.LocalCaller(), "recorded", request, prepared, core.CallPolicy{})
	return response, prepared, err
}

// Continue with the actual returned assistant message, including optional
// reasoning fields, and the recorded tool result. Do not replace its call ID.
func recordedChatContinuation(t *testing.T, template, firstResponse json.RawMessage) json.RawMessage {
	t.Helper()
	var payload map[string]json.RawMessage
	var messages []json.RawMessage
	var response struct {
		Choices []struct{ Message json.RawMessage }
	}
	if err := json.Unmarshal(template, &payload); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(payload["messages"], &messages); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(firstResponse, &response); err != nil {
		t.Fatal(err)
	}
	if len(messages) != 3 || len(response.Choices) != 1 {
		t.Fatal("recorded conversation requires user, assistant tool call and tool result")
	}
	var assistant struct {
		ToolCalls []struct {
			ID       string
			Type     string
			Function struct{ Name, Arguments string }
		} `json:"tool_calls"`
	}
	var result struct {
		Role       string
		Content    string
		ToolCallID string `json:"tool_call_id"`
	}
	if err := json.Unmarshal(response.Choices[0].Message, &assistant); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(messages[2], &result); err != nil {
		t.Fatal(err)
	}
	if len(assistant.ToolCalls) != 1 {
		t.Fatal("recorded response lost its single tool call")
	}
	call := assistant.ToolCalls[0]
	if call.ID == "" || call.Type != "function" || call.Function.Name != "get_weather" || call.Function.Arguments != `{"city": "London"}` ||
		result.Role != "tool" || result.ToolCallID != call.ID || result.Content != `{"temperature_c":16,"condition":"cloudy"}` {
		t.Fatal("recorded tool name, arguments or actual call/result association changed")
	}
	messages[1] = response.Choices[0].Message
	var err error
	payload["messages"], err = json.Marshal(messages)
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func TestRecordedChatToolConversation(t *testing.T) {
	t.Parallel()
	exchanges := recordedChatConversation(t)
	runtime, replay, _, transport := newRecordedChatRuntime(t, exchanges)
	body := exchanges[0].Request
	counts := [][3]int64{{46, 50, 96}, {389, 48, 437}}
	for turn, exchange := range exchanges {
		response, prepared, err := invokeRecordedChat(t, runtime, body)
		if err != nil {
			t.Fatalf("turn %d: %v", turn+1, err)
		}
		if !bytes.Equal(response.Body, exchange.Response) || response.ContentType != exchange.ContentType || len(response.Losses) != 0 {
			t.Fatalf("turn %d changed recorded output, optional fields, usage or finish state", turn+1)
		}
		var terminal struct {
			Choices []struct {
				FinishReason       string `json:"finish_reason"`
				NativeFinishReason string `json:"native_finish_reason"`
				Message            struct{ Content *string }
			}
		}
		if err := json.Unmarshal(response.Body, &terminal); err != nil || len(terminal.Choices) != 1 {
			t.Fatalf("turn %d has no single terminal choice: %v", turn+1, err)
		}
		choice := terminal.Choices[0]
		if turn == 0 {
			if choice.FinishReason != "tool_calls" || choice.NativeFinishReason != "tool_call" || choice.Message.Content != nil {
				t.Fatal("first turn did not terminate with the recorded tool call")
			}
		} else if choice.FinishReason != "stop" || choice.NativeFinishReason != "stop" || choice.Message.Content == nil ||
			*choice.Message.Content != "The current weather in London is **16°C** and **cloudy**." {
			t.Fatal("second turn did not return the recorded final answer and stop")
		}
		observation, ok := core.SourceUsageOf(prepared)
		if !ok || observation.API != core.UpstreamAPIChatCompletions || observation.Instance != "recorded" ||
			observation.Model != "recorded-chat-model" || observation.Attempt != 1 || observation.Cancelled ||
			observation.Usage.Quality != translate.CompletionComplete {
			t.Fatalf("turn %d lost native source completion: %+v", turn+1, observation)
		}
		for index, count := range []translate.CountObservation{observation.Usage.InputTokens, observation.Usage.OutputTokens, observation.Usage.TotalTokens} {
			if count.Presence != translate.CountPresent || count.Provenance != translate.UsageReported || count.Value != counts[turn][index] {
				t.Fatalf("turn %d count %d is not captured source usage: %+v", turn+1, index, count)
			}
		}
		if turn == 0 {
			body = recordedChatContinuation(t, exchanges[1].Request, response.Body)
		}
	}
	if calls, accepted, rejected := replay.counts(); calls != 2 || accepted != 2 || rejected != 0 || transport.blocked.Load() != 0 {
		t.Fatalf("calls=%d accepted=%d rejected=%d blocked=%d, want exactly two loopback native requests", calls, accepted, rejected, transport.blocked.Load())
	}
}

func TestRecordedChatToolConversationRejectsBrokenAssociation(t *testing.T) {
	t.Parallel()
	for _, mutation := range []string{"wrong-call-id", "missing-call-id", "missing-tool-result"} {
		t.Run(mutation, func(t *testing.T) {
			t.Parallel()
			exchanges := recordedChatConversation(t)
			runtime, replay, _, _ := newRecordedChatRuntime(t, exchanges)
			first, _, err := invokeRecordedChat(t, runtime, exchanges[0].Request)
			if err != nil {
				t.Fatal(err)
			}
			valid := recordedChatContinuation(t, exchanges[1].Request, first.Body)
			var broken map[string]any
			if err := json.Unmarshal(valid, &broken); err != nil {
				t.Fatal(err)
			}
			messages := broken["messages"].([]any)
			result := messages[2].(map[string]any)
			switch mutation {
			case "wrong-call-id":
				result["tool_call_id"] = "unmatched-call"
			case "missing-call-id":
				delete(result, "tool_call_id")
			case "missing-tool-result":
				broken["messages"] = messages[:2]
			}
			body, err := json.Marshal(broken)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := invokeRecordedChat(t, runtime, body); err == nil {
				t.Fatal("broken tool/result association received the recorded successful answer")
			}
			if calls, accepted, rejected := replay.counts(); calls != 2 || accepted != 1 || rejected != 1 {
				t.Fatalf("negative control did not reach and fail replay: calls=%d accepted=%d rejected=%d", calls, accepted, rejected)
			}
			response, _, err := invokeRecordedChat(t, runtime, valid)
			if err != nil || !bytes.Equal(response.Body, exchanges[1].Response) {
				t.Fatalf("rejected request consumed the second recorded answer: %v", err)
			}
			if calls, accepted, rejected := replay.counts(); calls != 3 || accepted != 2 || rejected != 1 {
				t.Fatalf("unexpected recovery counts: calls=%d accepted=%d rejected=%d", calls, accepted, rejected)
			}
		})
	}
}

func TestRecordedChatToolConversationBlocksExternalTransport(t *testing.T) {
	t.Parallel()
	_, replay, client, transport := newRecordedChatRuntime(t, recordedChatConversation(t))
	if _, err := client.Get("https://provider.invalid/chat/completions"); err == nil {
		t.Fatal("non-replay destination was admitted")
	}
	if calls, _, _ := replay.counts(); calls != 0 || transport.blocked.Load() != 1 {
		t.Fatalf("destination guard failed: loopback=%d blocked=%d", calls, transport.blocked.Load())
	}
}
