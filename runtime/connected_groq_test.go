package runtime_test

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

func connectedGroqExchanges(t *testing.T) []recordedChatExchange {
	t.Helper()
	raw, err := os.ReadFile("testdata/connected_groq_chat.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Provenance      string
		GatewayRequests []json.RawMessage `json:"gateway_requests"`
		Responses       []json.RawMessage
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.GatewayRequests) != 3 || len(fixture.Responses) != 2 || !strings.Contains(fixture.Provenance, "not upstream HTTP capture") {
		t.Fatal("fixture must retain its Gateway observation provenance and three cases")
	}
	var exchanges []recordedChatExchange
	for index, observed := range fixture.GatewayRequests {
		// SYNTHETIC provider request reconstruction, not recorded upstream bytes:
		// the public Gateway request has a provider-qualified model selector;
		// the native SDK also sends an explicit stream flag for Invoke.
		var request map[string]json.RawMessage
		if err := json.Unmarshal(observed, &request); err != nil {
			t.Fatal(err)
		}
		if string(request["model"]) != `"fixture-provider/recorded-groq-model"` {
			t.Fatal("observed Gateway model lost its provider prefix")
		}
		request["model"] = json.RawMessage(`"recorded-groq-model"`)
		if index < 2 {
			request["stream"] = json.RawMessage(`false`)
		}
		body, err := json.Marshal(request)
		if err != nil {
			t.Fatal(err)
		}
		exchange := recordedChatExchange{Method: http.MethodPost, Path: "/api/v1/chat/completions", Request: body, Status: http.StatusOK, ContentType: core.ContentTypeJSON}
		if index < 2 {
			exchange.Response = fixture.Responses[index]
		} else {
			exchange.Response, err = os.ReadFile("testdata/connected_groq_chat.sse")
			if err != nil {
				t.Fatal(err)
			}
			exchange.ResponseContentType = "text/event-stream"
		}
		exchanges = append(exchanges, exchange)
	}
	return exchanges
}

func connectedGroqContinuation(t *testing.T, template, actual json.RawMessage) json.RawMessage {
	t.Helper()
	var request map[string]json.RawMessage
	var response struct {
		Choices []struct{ Message json.RawMessage }
	}
	if err := json.Unmarshal(template, &request); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(actual, &response); err != nil || len(response.Choices) != 1 {
		t.Fatalf("missing actual assistant message: %v", err)
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(request["messages"], &messages); err != nil || len(messages) != 3 {
		t.Fatalf("missing conversation continuation: %v", err)
	}
	var assistant struct {
		ToolCalls []struct {
			ID       string
			Type     string
			Function struct{ Name, Arguments string }
		} `json:"tool_calls"`
	}
	var result struct {
		Role, Content string
		ToolCallID    string `json:"tool_call_id"`
	}
	if err := json.Unmarshal(response.Choices[0].Message, &assistant); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(messages[2], &result); err != nil {
		t.Fatal(err)
	}
	if len(assistant.ToolCalls) != 1 {
		t.Fatal("expected exactly one returned function call")
	}
	call := assistant.ToolCalls[0]
	if call.ID == "" || call.Type != "function" || call.Function.Name != "qualification_sum" || call.Function.Arguments != `{"a":2,"b":3}` ||
		result.Role != "tool" || result.ToolCallID != call.ID || result.Content != "5" {
		t.Fatal("tool name, arguments or actual call/result association changed")
	}
	// Carry the entire returned message, not a re-created tool_call alone.
	messages[1] = response.Choices[0].Message
	request["messages"], _ = json.Marshal(messages)
	body, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func assertConnectedGroqUsage(t *testing.T, source any, counts [3]int64) {
	t.Helper()
	observation, ok := core.SourceUsageOf(source)
	if !ok || observation.API != core.UpstreamAPIChatCompletions || observation.Instance != "recorded" ||
		observation.Model != "recorded-groq-model" || observation.Attempt != 1 || observation.Cancelled || observation.Usage.Quality != translate.CompletionComplete {
		t.Fatalf("native source completion lost: %+v", observation)
	}
	for index, count := range []translate.CountObservation{observation.Usage.InputTokens, observation.Usage.OutputTokens, observation.Usage.TotalTokens} {
		if count.Presence != translate.CountPresent || count.Provenance != translate.UsageReported || count.Value != counts[index] {
			t.Fatalf("source count %d changed: %+v", index, count)
		}
	}
}

func TestConnectedGroqToolConversation(t *testing.T) {
	t.Parallel()
	exchanges := connectedGroqExchanges(t)[:2]
	runtime, replay, _, transport := newRecordedChatRuntime(t, exchanges)
	body := exchanges[0].Request
	for turn, exchange := range exchanges {
		response, prepared, err := invokeRecordedChat(t, runtime, body)
		if err != nil {
			t.Fatalf("turn %d: %v", turn+1, err)
		}
		if !bytes.Equal(response.Body, exchange.Response) || response.ContentType != core.ContentTypeJSON || len(response.Losses) != 0 {
			t.Fatalf("turn %d lost observed response fields", turn+1)
		}
		var wire struct {
			Choices []struct {
				FinishReason string `json:"finish_reason"`
				Message      struct{ Content string }
			}
		}
		if err := json.Unmarshal(response.Body, &wire); err != nil || len(wire.Choices) != 1 {
			t.Fatalf("missing recorded terminal choice: %v", err)
		}
		if turn == 0 {
			if wire.Choices[0].FinishReason != "tool_calls" {
				t.Fatal("initial response did not terminate in a tool call")
			}
			assertConnectedGroqUsage(t, prepared, [3]int64{140, 54, 194})
			body = connectedGroqContinuation(t, exchanges[1].Request, response.Body)
		} else {
			if wire.Choices[0].FinishReason != "stop" || wire.Choices[0].Message.Content != "The sum of 2 and 3 is **5**." {
				t.Fatal("continuation answer or terminal state changed")
			}
			assertConnectedGroqUsage(t, prepared, [3]int64{202, 17, 219})
		}
	}
	if calls, accepted, rejected := replay.counts(); calls != 2 || accepted != 2 || rejected != 0 || transport.blocked.Load() != 0 {
		t.Fatalf("want exactly two accepted loopback calls; got %d/%d/%d", calls, accepted, rejected)
	}
}

func TestConnectedGroqRejectsWrongToolResult(t *testing.T) {
	t.Parallel()
	exchanges := connectedGroqExchanges(t)[:2]
	runtime, replay, _, _ := newRecordedChatRuntime(t, exchanges)
	first, _, err := invokeRecordedChat(t, runtime, exchanges[0].Request)
	if err != nil {
		t.Fatal(err)
	}
	valid := connectedGroqContinuation(t, exchanges[1].Request, first.Body)
	broken := bytes.Replace(valid, []byte(`"tool_call_id":"fixture-id-1"`), []byte(`"tool_call_id":"unmatched-call"`), 1)
	if bytes.Equal(valid, broken) {
		t.Fatal("negative control did not change association")
	}
	if _, _, err := invokeRecordedChat(t, runtime, broken); err == nil {
		t.Fatal("broken association received a recorded answer")
	}
	if calls, accepted, rejected := replay.counts(); calls != 2 || accepted != 1 || rejected != 1 {
		t.Fatalf("broken request did not reach and fail replay: %d/%d/%d", calls, accepted, rejected)
	}
	response, _, err := invokeRecordedChat(t, runtime, valid)
	if err != nil || !bytes.Equal(response.Body, exchanges[1].Response) {
		t.Fatalf("rejection consumed the valid continuation: %v", err)
	}
}

func TestConnectedGroqNativeStream(t *testing.T) {
	t.Parallel()
	exchange := connectedGroqExchanges(t)[2]
	runtime, replay, _, transport := newRecordedChatRuntime(t, []recordedChatExchange{exchange})
	request := recordedChatRequest(t, exchange.Request)
	prepared, err := runtime.PrepareCall(t.Context(), core.LocalCaller(), "recorded", request, core.CallPreparationOptions{Mode: core.CallModeStream})
	if err != nil {
		t.Fatal(err)
	}
	if description := prepared.Description(); description.UpstreamAPI != core.UpstreamAPIChatCompletions || len(description.Conversions) != 0 {
		t.Fatalf("stream was not native: %+v", description)
	}
	stream, err := runtime.StreamPrepared(t.Context(), core.LocalCaller(), "recorded", request, prepared, core.CallPolicy{})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var delivered bytes.Buffer
	var content, reasoning strings.Builder
	frames, terminals, done := 0, 0, 0
	for {
		frame, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if done != 0 {
			t.Fatal("frame delivered after DONE")
		}
		delivered.Write(frame)
		data := bytes.TrimSpace(bytes.TrimPrefix(frame, []byte("data:")))
		if bytes.Equal(data, []byte("[DONE]")) {
			done++
			continue
		}
		var chunk struct {
			ID      string
			Choices []struct {
				Delta        struct{ Content, Reasoning, Channel string }
				FinishReason *string `json:"finish_reason"`
			}
		}
		if err := json.Unmarshal(data, &chunk); err != nil || chunk.ID != "fixture-id-6" || len(chunk.Choices) != 1 {
			t.Fatalf("stream identity or chunk shape changed: %v", err)
		}
		choice := chunk.Choices[0]
		content.WriteString(choice.Delta.Content)
		reasoning.WriteString(choice.Delta.Reasoning)
		if choice.Delta.Reasoning != "" && choice.Delta.Channel != "analysis" {
			t.Fatal("reasoning channel lost")
		}
		if choice.FinishReason != nil {
			if *choice.FinishReason != "stop" {
				t.Fatal("recorded stream did not stop normally")
			}
			terminals++
		}
		frames++
	}
	// Compare all observed data fields and framing, ignoring trailing blank padding.
	if !bytes.Equal(bytes.TrimSpace(delivered.Bytes()), bytes.TrimSpace(exchange.Response)) || len(core.StreamLosses(stream)) != 0 {
		t.Fatal("native stream changed observed raw fields or reported conversion losses")
	}
	if frames != 26 || terminals != 1 || done != 1 || content.String() != "OK" ||
		reasoning.String() != `The user says: "Reply with exactly OK." So we must respond with exactly "OK". No other characters.` {
		t.Fatalf("stream completion changed: frames=%d terminals=%d done=%d", frames, terminals, done)
	}
	assertConnectedGroqUsage(t, stream, [3]int64{76, 34, 110})
	assertConnectedGroqUsage(t, prepared, [3]int64{76, 34, 110})
	if calls, accepted, rejected := replay.counts(); calls != 1 || accepted != 1 || rejected != 0 || transport.blocked.Load() != 0 {
		t.Fatalf("want one accepted loopback stream: %d/%d/%d", calls, accepted, rejected)
	}
}

func TestConnectedGroqBlocksExternalTransport(t *testing.T) {
	t.Parallel()
	_, replay, client, transport := newRecordedChatRuntime(t, connectedGroqExchanges(t)[:2])
	if _, err := client.Get("https://provider.invalid/chat/completions"); err == nil {
		t.Fatal("external transport admitted")
	}
	if calls, _, _ := replay.counts(); calls != 0 || transport.blocked.Load() != 1 {
		t.Fatalf("external request reached underlying transport: loopback=%d blocked=%d", calls, transport.blocked.Load())
	}
}
