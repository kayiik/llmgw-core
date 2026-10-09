package extension

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	core "github.com/kayiik/llmgw-core"
)

// slowDaemon answers every request after delay.
func slowDaemon(t *testing.T, delay time.Duration) string {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(delay):
		case <-r.Context().Done():
			return
		}
		if r.URL.Path == PathPrefix+"p/stream" {
			w.Header().Set("Content-Type", core.ContentTypeEventStream)
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		_, _ = io.WriteString(w, `{"version":"1","providers":[]}`)
	}))
	t.Cleanup(server.Close)
	return server.URL
}

// The default client gives up on an answer that has not begun after its
// wait, except for an operation with a timeout, which waits as long as its
// daemon. A product's own client waits as its transport does.
func TestOnlyAnOperationWithATimeoutOutwaitsTheDefaultClient(t *testing.T) {
	base := slowDaemon(t, 200*time.Millisecond)
	client, err := NewClient(Config{BaseURL: base})
	if err != nil {
		t.Fatal(err)
	}
	client.answerWait = 20 * time.Millisecond
	request := core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "m"}
	stream := func(ctx context.Context) error {
		iter, err := client.Stream(ctx, "p", request)
		if err == nil {
			_ = iter.Close()
		}
		return err
	}

	if _, err := client.Info(context.Background()); !errors.Is(err, errNoAnswer) {
		t.Errorf("info: err = %v, want no answer", err)
	}
	_, err = client.Invoke(context.Background(), "p", request)
	if !errors.Is(err, errNoAnswer) || core.ClassifyError(err).Disposition() != core.DispositionRetryable {
		t.Errorf("invoke without a timeout: err = %v, want a retryable failure to answer", err)
	}
	if err := stream(context.Background()); !errors.Is(err, errNoAnswer) {
		t.Errorf("stream without a timeout: err = %v, want no answer", err)
	}

	withTimeout := WithTimeout(context.Background(), time.Millisecond)
	if _, err := client.Invoke(withTimeout, "p", request); err != nil {
		t.Errorf("invoke with a timeout: %v", err)
	}
	if err := stream(withTimeout); err != nil {
		t.Errorf("stream with a timeout: %v", err)
	}

	own, err := NewClient(Config{BaseURL: base, HTTPClient: &http.Client{}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := own.Info(context.Background()); err != nil {
		t.Errorf("a product's own client gave up: %v", err)
	}
}
