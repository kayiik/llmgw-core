package runtime_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"

	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/providers"
	coreruntime "github.com/kayiik/llmgw-core/runtime"
	"github.com/kayiik/llmgw-core/translation"
)

// Text parts retain their content through the runtime and provider mapper.
const (
	ollamaRuntimeChat = `{"model":"llama3.2","messages":[{"role":"developer","content":"Be brief."},` +
		`{"role":"user","content":[{"type":"text","text":"Say hello"}]}],` +
		`"max_tokens":64,"stop":["END"]}`
	ollamaUpstreamChat = `{"messages":[{"content":"Be brief.","role":"system"},` +
		`{"content":"Say hello","role":"user"}],` +
		`"model":"llama3.2","options":{"num_predict":64},"stream":false}`
	ollamaUpstreamStream = `{"messages":[{"content":"Say hello","role":"user"}],"model":"llama3.2","stream":true}`
)

type ollamaCall struct{ path, body string }

// ollamaBackend is a synthetic Ollama daemon.
type ollamaBackend struct {
	mu    sync.Mutex
	calls []ollamaCall
}

func (b *ollamaBackend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	b.mu.Lock()
	b.calls = append(b.calls, ollamaCall{path: r.URL.Path, body: string(body)})
	b.mu.Unlock()
	switch {
	case r.URL.Path == "/api/tags":
		_, _ = io.WriteString(w, `{"models":[{"name":"llama3.2:latest","details":{"family":"llama"}}]}`)
	case strings.Contains(string(body), `"stream":true`):
		_, _ = io.WriteString(w, `{"message":{"role":"assistant","content":"Hello"},"done":false}`+"\n"+
			`{"message":{"role":"assistant","content":" there"},"done":false}`+"\n"+`{"done":true,"eval_count":2}`+"\n")
	default:
		_, _ = io.WriteString(w, `{"message":{"role":"assistant","content":"Hello from ollama"},"done":true,"prompt_eval_count":3,"eval_count":4}`)
	}
}

func (b *ollamaBackend) take() []ollamaCall {
	b.mu.Lock()
	defer b.mu.Unlock()
	calls := b.calls
	b.calls = nil
	return calls
}

func TestRuntimeServesTheOllamaVertical(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	backend := &ollamaBackend{}
	server := httptest.NewServer(backend)
	defer server.Close()
	runtime, err := coreruntime.New(coreruntime.Options[string]{
		Settings: coreruntime.NewMemorySettings(server.URL),
		Providers: func(base string, _ string) (core.Provider, error) {
			provider, err := providers.NewOllama(providers.OllamaConfig{BaseURL: base, Client: server.Client(), CatalogClient: server.Client()})
			if err != nil {
				return nil, err
			}
			return translation.Adapter{Provider: provider}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	caller := core.Caller{ID: "visitor", Kind: core.CallerHuman}
	chat := func(body string) core.Request {
		return core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "llama3.2", ContentType: core.ContentTypeJSON, Body: []byte(body)}
	}

	t.Run("chat preserves text parts and advisory reports", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, caller, "ollama", chat(ollamaRuntimeChat))
		const want = `{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"Hello from ollama","role":"assistant"}}],` +
			`"id":"chatcmpl-ollama","model":"llama3.2","object":"chat.completion","usage":{"completion_tokens":4,"prompt_tokens":3,"total_tokens":7}}`
		if err != nil || completionGolden(t, string(response.Body), "ollama") != want {
			t.Fatalf("response = %s, err = %v", response.Body, err)
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []ollamaCall{{path: "/api/chat", body: ollamaUpstreamChat}}) {
			t.Fatalf("upstream = %+v", calls)
		}
		var paths []string
		for _, loss := range response.Losses {
			paths = append(paths, string(loss.Severity)+" "+loss.Path)
		}
		if want := []string{"advisory messages.0.role", "advisory stop"}; !slices.Equal(paths, want) {
			t.Fatalf("losses = %v, want %v", paths, want)
		}
	})

	t.Run("material content is refused before invoke or stream", func(t *testing.T) {
		request := chat(`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:,"}}]}]}`)
		response, invokeErr := runtime.Invoke(ctx, caller, "ollama", request)
		stream, streamErr := runtime.Stream(ctx, caller, "ollama", request)
		if stream != nil {
			_ = stream.Close()
		}
		for _, err := range []error{invokeErr, streamErr} {
			var failure *core.LossPolicyError
			if !errors.As(err, &failure) {
				t.Fatalf("error=%v", err)
			}
		}
		if len(response.Losses) != 1 || response.Losses[0].Path != "messages.0.content.0" {
			t.Fatalf("losses=%v", response.Losses)
		}
		if calls := backend.take(); len(calls) != 0 {
			t.Fatalf("refused input sent: %v", calls)
		}
	})

	t.Run("stream", func(t *testing.T) {
		stream, err := runtime.Stream(ctx, caller, "ollama", chat(`{"model":"llama3.2","stream":true,"messages":[{"role":"user","content":"Say hello"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		defer stream.Close()
		var frames strings.Builder
		for {
			frame, err := stream.Next()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			frames.Write(frame)
		}
		chunk := func(delta, finish string) string {
			return `data: {"choices":[{"delta":` + delta + `,"finish_reason":` + finish + `,"index":0}],"id":"chatcmpl-ollama","model":"llama3.2","object":"chat.completion.chunk"}` + "\n\n"
		}
		if want := chunk(`{"content":"Hello"}`, "null") + chunk(`{"content":" there"}`, "null") + chunk(`{}`, `"stop"`) + "data: [DONE]\n\n"; completionGolden(t, frames.String(), "ollama") != want {
			t.Fatalf("frames = %q\nwant %q", frames.String(), want)
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []ollamaCall{{path: "/api/chat", body: ollamaUpstreamStream}}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("messages through the adapter over native chat", func(t *testing.T) {
		response, err := runtime.Invoke(ctx, caller, "ollama", core.Request{
			Surface: core.ModelSurfaceMessages, Model: "llama3.2", ContentType: core.ContentTypeJSON,
			Body: []byte(`{"model":"llama3.2","max_tokens":64,"system":"Be brief.","messages":[{"role":"user","content":"Say hello"}]}`),
		})
		if err != nil || !strings.Contains(string(response.Body), `"text":"Hello from ollama"`) || !strings.Contains(string(response.Body), `"type":"message"`) {
			t.Fatalf("response = %s, err = %v", response.Body, err)
		}
		const want = `{"messages":[{"content":"Be brief.","role":"system"},{"content":"Say hello","role":"user"}],"model":"llama3.2","options":{"num_predict":64},"stream":false}`
		if calls := backend.take(); !reflect.DeepEqual(calls, []ollamaCall{{path: "/api/chat", body: want}}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})

	t.Run("catalog", func(t *testing.T) {
		record, err := runtime.ListModels(ctx, caller, "ollama")
		want := []core.ModelInfo{{ID: "llama3.2:latest", Object: "model", OwnedBy: "llama", Vendor: "llama"}}
		if err != nil || record.Evidence.Status != core.CatalogDiscovered || !reflect.DeepEqual(record.Evidence.Models, want) {
			t.Fatalf("catalog = %+v, err = %v", record, err)
		}
		if calls := backend.take(); !reflect.DeepEqual(calls, []ollamaCall{{path: "/api/tags"}}) {
			t.Fatalf("upstream = %+v", calls)
		}
	})
}
