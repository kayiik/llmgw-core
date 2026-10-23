package providers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"testing"

	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/translation"
)

type capturedChatProvider struct {
	core.Provider
	response core.Response
}

func (p *capturedChatProvider) Invoke(ctx context.Context, request core.Request) (core.Response, error) {
	response, err := p.Provider.Invoke(ctx, request)
	p.response = response
	return response, err
}

func TestChatCompletionIdentitySurvivesSurfaceConversion(t *testing.T) {
	t.Parallel()
	_, base := newGoogleFake(t, googleAnswer(200, googleHelloAnswer))
	google := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	ollama, _ := ollamaDaemon(t, 200, `{"message":{"content":"ok"},"done":true}`)
	for _, provider := range []core.Provider{google, ollama} {
		for _, surface := range []core.ModelSurface{core.ModelSurfaceMessages, core.ModelSurfaceResponses} {
			captured := &capturedChatProvider{Provider: provider}
			adapter := translation.Adapter{Provider: captured}
			body := `{"messages":[{"role":"user","content":"hi"}],"max_tokens":32}`
			if surface == core.ModelSurfaceResponses {
				body = `{"input":"hi"}`
			}
			request := googleChatRequest("m", body, googleKey("fixture"))
			request.Surface = surface
			response, err := adapter.Invoke(t.Context(), request)
			if err != nil {
				t.Fatal(err)
			}
			want := completionID(t, captured.response.Body)
			if surface == core.ModelSurfaceResponses {
				want = "resp_" + strings.TrimPrefix(want, "chatcmpl-")
			}
			if got := completionID(t, response.Body); got != want {
				t.Fatalf("%T %s lost identity: %s != %s", provider, surface, got, want)
			}
		}
	}
}

// Substitute only the validated operation ID in characterization goldens.
// Tool IDs and every other payload field remain exact; raw IDs are tested below.
func completionGolden(t *testing.T, body, provider string) string {
	t.Helper()
	pattern := regexp.MustCompile(`"id":"(chatcmpl-` + provider + `-[A-Z2-7]{26})"`)
	matches := pattern.FindAllStringSubmatch(body, -1)
	if len(matches) == 0 {
		t.Fatalf("missing operation completion ID: %s", body)
	}
	if len(matches) != strings.Count(body, `"object":"chat.completion`) {
		t.Fatalf("not every completion has an operation ID: %s", body)
	}
	for _, match := range matches {
		if match[1] != matches[0][1] {
			t.Fatalf("completion ID changed in stream: %s", body)
		}
	}
	return strings.ReplaceAll(body, `"id":"`+matches[0][1]+`"`, `"id":"chatcmpl-`+provider+`"`)
}

func googleRefusedChat(t *testing.T, body string, stream bool) []core.Loss {
	t.Helper()
	fake, base := newGoogleFake(t, googleAnswer(200, googleHelloAnswer))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	request := googleChatRequest("m", body, googleKey("fixture"))
	var err error
	var response core.Response
	if stream {
		var iter core.StreamIter
		iter, err = provider.Stream(t.Context(), request)
		if iter != nil {
			_ = iter.Close()
		}
	} else {
		response, err = provider.Invoke(t.Context(), request)
	}
	var refusal *core.LossPolicyError
	if !errors.As(err, &refusal) {
		t.Fatalf("error=%v, want loss refusal", err)
	}
	if !stream && len(response.Losses) < len(refusal.Losses) {
		t.Fatalf("missing response losses: %v", response.Losses)
	}
	if calls := fake.take(); len(calls) != 0 {
		t.Fatalf("refused request reached upstream: %v", calls)
	}
	return refusal.Losses
}

// Equivalent text forms must reach the upstream, including the tool-result
// path which already understood text parts before ordinary messages did.
func TestGoogleChatPreservesTextContentParts(t *testing.T) {
	t.Parallel()
	for _, deployment := range []GoogleDeployment{GoogleAIStudio, GoogleVertexAI} {
		for _, stream := range []bool{false, true} {
			for _, role := range []string{"system", "developer", "user", "assistant", "tool"} {
				prefix, link := "", ""
				if role == "tool" {
					prefix = `{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},`
					link = `,"tool_call_id":"c1"`
				}
				var sent string
				for index, content := range []string{`"hé \n世界"`, `[{"type":"text","text":"hé "},{"type":"text","text":""},{"type":"text","text":"\n世界"}]`} {
					body, losses := googleSent(t, deployment, stream, `{"messages":[`+prefix+`{"role":"`+role+`","content":`+content+link+`}]}`)
					if len(losses) != 0 {
						t.Fatalf("%s stream=%t role=%s: losses=%v", deployment, stream, role, losses)
					}
					if index == 0 {
						sent = body
					} else if body != sent {
						t.Fatalf("%s stream=%t role=%s: parts=%s, string=%s", deployment, stream, role, body, sent)
					}
				}
			}
		}
	}
}

func TestOllamaChatPreservesTextContentParts(t *testing.T) {
	t.Parallel()
	for _, stream := range []bool{false, true} {
		for _, role := range []string{"system", "developer", "user", "assistant", "tool"} {
			var sent string
			for index, content := range []string{`"hé \n世界"`, `[{"type":"text","text":"hé "},{"type":"text","text":""},{"type":"text","text":"\n世界"}]`} {
				provider, calls := ollamaDaemon(t, 200, `{"message":{"content":"ok"},"done":true}`)
				request := ollamaChatRequest("m", `{"messages":[{"role":"`+role+`","content":`+content+`}]}`)
				if stream {
					iter, err := provider.Stream(t.Context(), request)
					if err != nil {
						t.Fatal(err)
					}
					_ = iter.Close()
				} else if _, err := provider.Invoke(t.Context(), request); err != nil {
					t.Fatal(err)
				}
				got := calls()
				if len(got) != 1 {
					t.Fatalf("calls=%v", got)
				}
				if index == 0 {
					sent = got[0].body
				} else if got[0].body != sent {
					t.Fatalf("stream=%t role=%s: parts=%s, string=%s", stream, role, got[0].body, sent)
				}
			}
		}
	}
}

func TestMaterialChatLossRefusesBeforeUpstream(t *testing.T) {
	t.Parallel()
	cases := []struct{ name, body, path string }{
		{"mixed image", `{"messages":[{"role":"user","content":[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"data:,"}}]}]}`, "messages.0.content.1"},
		{"nonobject part", `{"messages":[{"role":"user","content":["hello"]}]}`, "messages.0.content.0"},
		{"missing text", `{"messages":[{"role":"user","content":[{"type":"text"}]}]}`, "messages.0.content.0"},
		{"invalid text", `{"messages":[{"role":"user","content":[{"type":"text","text":4}]}]}`, "messages.0.content.0"},
		{"unmapped part state", `{"messages":[{"role":"user","content":[{"type":"text","text":"hello","signature":"opaque-fixture"}]}]}`, "messages.0.content.0.signature"},
		{"scalar content", `{"messages":[{"role":"user","content":4}]}`, "messages.0.content"},
		{"structural option", `{"messages":[{"role":"user","content":"hi"}],"n":2}`, "n"},
	}
	for _, tc := range cases {
		for _, wrapped := range []bool{false, true} {
			for _, stream := range []bool{false, true} {
				t.Run(tc.name+"/"+map[bool]string{false: "direct", true: "adapter"}[wrapped]+"/"+map[bool]string{false: "invoke", true: "stream"}[stream], func(t *testing.T) {
					fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
					ollama, calls := ollamaDaemon(t, 200, `{"message":{"content":"ok"},"done":true}`)
					google := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
					for _, p := range []core.Provider{google, ollama} {
						if wrapped {
							p = translation.Adapter{Provider: p}
						}
						request := googleChatRequest("m", tc.body, googleKey("fixture"))
						var err error
						if stream {
							var iter core.StreamIter
							iter, err = p.Stream(t.Context(), request)
							if iter != nil {
								_ = iter.Close()
							}
						} else {
							var response core.Response
							response, err = p.Invoke(t.Context(), request)
							if err != nil && len(response.Losses) == 0 {
								t.Error("refusal lost its request findings")
							}
						}
						var loss *core.LossPolicyError
						if !errors.As(err, &loss) {
							t.Errorf("%T: error=%v, want loss refusal", p, err)
							continue
						}
						found := false
						for _, item := range loss.Losses {
							found = found || item.Path == tc.path
						}
						if !found {
							t.Errorf("missing %s in %v", tc.path, loss.Losses)
						}
					}
					if got := fake.take(); len(got) != 0 {
						t.Errorf("Google sent %d requests", len(got))
					}
					if got := calls(); len(got) != 0 {
						t.Errorf("Ollama sent %d requests", len(got))
					}
				})
			}
		}
	}
}

func completionID(t *testing.T, body []byte) string {
	t.Helper()
	var envelope map[string]any
	if err := json.Unmarshal(body, &envelope); err != nil {
		t.Fatal(err)
	}
	id, _ := envelope["id"].(string)
	if id == "" {
		t.Fatalf("missing completion ID: %s", body)
	}
	return id
}

// Inspect actual chunks here, not normalized characterization goldens.
func streamCompletionID(t *testing.T, iter core.StreamIter) string {
	t.Helper()
	defer iter.Close()
	id := ""
	chunks := 0
	for {
		frame, err := iter.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		data := strings.TrimSpace(strings.TrimPrefix(string(frame), "data: "))
		if data == "[DONE]" {
			continue
		}
		got := completionID(t, []byte(data))
		if id != "" && got != id {
			t.Fatalf("stream ID changed: %s -> %s", id, got)
		}
		id = got
		chunks++
	}
	if chunks < 2 {
		t.Fatalf("need content and terminal chunks, got %d", chunks)
	}
	return id
}

func TestChatCompletionIDsIdentifyOperations(t *testing.T) {
	t.Parallel()
	_, base := newGoogleFake(t, googleAnswer(200, googleHelloAnswer))
	google := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	_, repeatedBase := newGoogleFake(t, googleAnswer(200, `{"responseId":"repeated","candidates":[{"content":{"parts":[{"text":"ok"}]}}]}`))
	repeated := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: repeatedBase})
	ollama, _ := ollamaDaemon(t, 200, `{"message":{"content":"ok"},"done":true}`)
	for _, provider := range []core.Provider{google, repeated, ollama} {
		seen := map[string]bool{}
		for range 12 {
			response, err := provider.Invoke(t.Context(), googleChatRequest("m", `{"messages":[{"role":"user","content":"hi"}]}`, googleKey("fixture")))
			if err != nil {
				t.Fatal(err)
			}
			id := completionID(t, response.Body)
			if seen[id] {
				t.Fatalf("%T reused completion ID %s", provider, id)
			}
			seen[id] = true
		}
	}
}

func TestChatStreamCompletionIDIsStableAndUnique(t *testing.T) {
	t.Parallel()
	// The response ID first appears late and then changes. It must not change
	// an already delivered completion ID, including on the final usage chunk.
	_, base := newGoogleStreamFake(t, 200, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"a\"}]}}]}\n\n"+
		"data: {\"responseId\":\"late\",\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"f\",\"args\":{}}}]}}]}\n\n"+
		"data: {\"responseId\":\"changed\",\"candidates\":[{\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":2}}\n\n")
	google := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	ollama, _ := ollamaDaemon(t, 200, "{\"message\":{\"content\":\"a\"}}\n{\"done\":true}\n")
	for _, provider := range []core.Provider{google, ollama} {
		first, err := provider.Stream(t.Context(), googleChatRequest("m", `{"messages":[{"role":"user","content":"hi"}],"stream_options":{"include_usage":true}}`, googleKey("fixture")))
		if err != nil {
			t.Fatal(err)
		}
		second, err := provider.Stream(t.Context(), googleChatRequest("m", `{"messages":[{"role":"user","content":"hi"}],"stream_options":{"include_usage":true}}`, googleKey("fixture")))
		if err != nil {
			_ = first.Close()
			t.Fatal(err)
		}
		ids := []string{streamCompletionID(t, first), streamCompletionID(t, second)}
		if ids[0] == ids[1] {
			t.Fatalf("%T streams share %s", provider, ids[0])
		}
	}
}

func TestConcurrentChatCompletionIDsAreIsolated(t *testing.T) {
	t.Parallel()
	_, base := newGoogleFake(t, googleAnswer(200, googleHelloAnswer))
	google := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	ollama, _ := ollamaDaemon(t, 200, `{"message":{"content":"ok"},"done":true}`)
	for _, provider := range []core.Provider{google, ollama} {
		const count = 16
		bodies := make(chan []byte, count)
		var workers sync.WaitGroup
		for range count {
			workers.Go(func() {
				response, err := provider.Invoke(t.Context(), googleChatRequest("m", `{"messages":[{"role":"user","content":"hi"}]}`, googleKey("fixture")))
				if err != nil {
					t.Error(err)
					return
				}
				bodies <- response.Body
			})
		}
		workers.Wait()
		close(bodies)
		seen := map[string]bool{}
		for body := range bodies {
			id := completionID(t, body)
			if seen[id] {
				t.Errorf("%T reused concurrent operation ID %s", provider, id)
			}
			seen[id] = true
		}
		if len(seen) != count {
			t.Fatalf("%T: %d distinct completions, want %d", provider, len(seen), count)
		}
	}
}

func TestGoogleSupportedSignedHistoryStillReachesUpstream(t *testing.T) {
	t.Parallel()
	for _, stream := range []bool{false, true} {
		for _, signature := range []string{
			`"extra_content":{"google":{"thought_signature":"fixture-signature"}},"function":{"name":"f","arguments":"{}"}`,
			`"function":{"name":"f","arguments":"{}","thought_signature":"fixture-signature"}`,
			`"thought_signature":"fixture-signature","function":{"name":"f","arguments":"{}"}`,
		} {
			body, losses := googleSent(t, GoogleAIStudio, stream, `{"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function",`+signature+`}]},{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"ok"}]}]}`)
			if len(losses) != 0 || !strings.Contains(body, `"thoughtSignature":"fixture-signature"`) || !strings.Contains(body, `"functionResponse":{"name":"f","response":{"result":"ok"}}`) {
				t.Fatalf("signed history changed: %s losses=%v", body, losses)
			}
		}
	}
}

func TestChatEmptyAndWhitespaceTextIsPreserved(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct{ content, text string }{
		{`""`, ""}, {`[]`, ""}, {`[{"type":"text","text":""}]`, ""},
		{`"   "`, "   "}, {`[{"type":"text","text":"   "}]`, "   "},
	} {
		body := `{"messages":[{"role":"user","content":` + tc.content + `}]}`
		google, losses := googleSent(t, GoogleAIStudio, false, body)
		if len(losses) != 0 || !strings.Contains(google, `"parts":[{"text":"`+tc.text+`"}]`) {
			t.Fatalf("Google text=%s losses=%v", google, losses)
		}
		ollama, calls := ollamaDaemon(t, 200, `{"message":{"content":"ok"},"done":true}`)
		response, err := ollama.Invoke(t.Context(), ollamaChatRequest("m", body))
		if err != nil || len(response.Losses) != 0 {
			t.Fatalf("Ollama error=%v losses=%v", err, response.Losses)
		}
		got := calls()
		if len(got) != 1 || !strings.Contains(got[0].body, `"messages":[{"content":"`+tc.text+`","role":"user"}]`) {
			t.Fatalf("Ollama text=%v", got)
		}
	}
}
