package providers

import (
	"errors"
	"reflect"
	"slices"
	"testing"

	core "github.com/kayiik/llmgw-core"
)

// ollamaRichChat includes unrepresentable image/schema input. Preparation
// retains its findings, but execution must now refuse it before inference.
const ollamaRichChat = `{"model":"ignored","stream":true,"messages":[` +
	`{"role":"developer","content":"Answer <briefly> & kindly."},` +
	`{"role":"system","content":"   "},` +
	`{"content":"What is in this image?"},` +
	`{"role":"user","content":[{"type":"text","text":"Describe it"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo=","detail":"low"}}]},` +
	`{"role":"assistant","content":null,"refusal":null,"tool_calls":[{"id":"call_fixture","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"fixture\"}"}}]},` +
	`{"role":"tool","tool_call_id":"call_fixture","name":"lookup","content":"{\"answer\":42}"},` +
	`{"role":"user","content":"Thanks","name":"fixture-user"}],` +
	`"temperature":0.25,"max_tokens":1e2,"top_p":1,"max_completion_tokens":50,"stop":["END"],"seed":7,` +
	`"tool_choice":"auto","response_format":{"type":"json_object"},` +
	`"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"q":{"type":"string","maxLength":64.0}}}}}]}`

// ollamaRichLosses is what that conversion loses, the image first among it.
var ollamaRichLosses = []string{
	"advisory dropped max_completion_tokens",
	"advisory approximated messages.0.role",
	"material dropped messages.3.content.1",
	"material dropped response_format",
	"advisory dropped seed",
	"advisory dropped stop",
	"advisory dropped tool_choice",
}

// The expected bodies are what the gateway's ollamaBuildPayload, fed by its
// Chat facade, encodes for the same Chat bodies.
func TestOllamaSendsWhatTheGatewaySends(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name, model, body, want string
		stream                  bool
		reject                  bool
		losses                  []string
	}{
		{
			name: "plain", model: "llama3.2", body: `{"model":"llama3.2","messages":[{"role":"user","content":"Say hello"}]}`,
			want: `{"messages":[{"content":"Say hello","role":"user"}],"model":"llama3.2","stream":false}`,
		},
		{name: "every rule", model: "qwen3:8b", body: ollamaRichChat, stream: true, reject: true, losses: ollamaRichLosses},
		{
			// encoding/json matches field names regardless of case, as the
			// gateway's Chat facade decodes them, and a null is absent.
			name: "folded field names", model: "m", body: `{"messages":[{"role":"user","content":"hi"}],"Temperature":0.5,"MAX_TOKENS":8,"top_P":null}`,
			want: `{"messages":[{"content":"hi","role":"user"}],"model":"m","options":{"num_predict":8,"temperature":0.5},"stream":false}`,
		},
		{
			name: "no messages", model: "m", body: `{"messages":null,"stream_options":{"include_usage":true}}`,
			want: `{"messages":[],"model":"m","stream":false}`, losses: []string{"advisory dropped stream_options"},
		},
		{
			name: "message fields", model: "m", body: `{"messages":[{"role":"assistant","content":"x","reasoning_content":"why","name":7}]}`,
			want: `{"messages":[{"content":"x","role":"assistant"}],"model":"m","stream":false}`, losses: []string{"advisory dropped messages.0.reasoning_content"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			provider, calls := ollamaDaemon(t, 200, `{"message":{"role":"assistant","content":"hi"},"done":true}`)
			request := ollamaChatRequest(tc.model, tc.body)
			if tc.reject {
				response, invokeErr := provider.Invoke(t.Context(), request)
				stream, streamErr := provider.Stream(t.Context(), request)
				if stream != nil {
					_ = stream.Close()
				}
				for _, err := range []error{invokeErr, streamErr} {
					var failure *core.LossPolicyError
					if !errors.As(err, &failure) {
						t.Fatalf("error=%v, want loss refusal", err)
					}
				}
				if got := lossPaths(response.Losses); !slices.Equal(got, tc.losses) {
					t.Fatalf("losses=%v, want %v", got, tc.losses)
				}
				if got := calls(); len(got) != 0 {
					t.Fatalf("refused request sent: %v", got)
				}
				return
			}
			var losses []core.Loss
			if tc.stream {
				stream, err := provider.Stream(t.Context(), request)
				if err != nil {
					t.Fatal(err)
				}
				losses = core.StreamLosses(stream)
				_ = stream.Close()
			} else {
				response, err := provider.Invoke(t.Context(), request)
				if err != nil {
					t.Fatal(err)
				}
				losses = response.Losses
			}
			want := []ollamaRecorded{{method: "POST", path: "/api/chat", contentType: core.ContentTypeJSON, body: tc.want}}
			if got := calls(); !reflect.DeepEqual(got, want) {
				t.Fatalf("upstream = %+v\nwant       %+v", got, want)
			}
			if got := lossPaths(losses); !slices.Equal(got, tc.losses) {
				t.Fatalf("losses = %q\nwant     %q", got, tc.losses)
			}
		})
	}
}
