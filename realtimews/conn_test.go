package realtimews

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestUpgradePreservesCustomResponseHeaders(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-LLMGW-Usage-Scope", "session")
		w.Header().Set("X-LLMGW-Realtime-Profile", "openai-ga")
		for key, value := range map[string]string{
			"Connection": "close", "Upgrade": "wrong",
			"Sec-WebSocket-Accept": "wrong", "Sec-WebSocket-Protocol": "unrequested",
			"Sec-WebSocket-Extensions": "permessage-deflate", "Content-Length": "999",
			"Transfer-Encoding": "chunked",
		} {
			w.Header().Set(key, value)
		}
		c, err := Upgrade(w, r)
		if err != nil {
			return
		}
		defer c.Close()
		_, _ = c.Read()
	}))
	defer server.Close()
	address := strings.TrimPrefix(server.URL, "http://")
	socket, err := net.DialTimeout("tcp", address, 3*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer socket.Close()
	if err := socket.SetDeadline(time.Now().Add(3 * time.Second)); err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(http.MethodGet, server.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Connection", "Upgrade")
	request.Header.Set("Upgrade", "websocket")
	request.Header.Set("Sec-WebSocket-Version", "13")
	request.Header.Set("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
	if err := request.Write(socket); err != nil {
		t.Fatal(err)
	}
	response, err := http.ReadResponse(bufio.NewReader(socket), request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusSwitchingProtocols {
		t.Fatalf("upgrade status: %d", response.StatusCode)
	}
	for key, want := range map[string]string{
		"X-LLMGW-Usage-Scope": "session", "X-LLMGW-Realtime-Profile": "openai-ga",
		"Connection": "Upgrade", "Upgrade": "websocket",
		"Sec-WebSocket-Accept": "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=",
	} {
		if values := response.Header.Values(key); len(values) != 1 || values[0] != want {
			t.Errorf("handshake header %s = %q, want one %q", key, values, want)
		}
	}
	for _, key := range []string{"Sec-WebSocket-Protocol", "Sec-WebSocket-Extensions", "Content-Length", "Transfer-Encoding"} {
		if values := response.Header.Values(key); len(values) != 0 {
			t.Errorf("conflicting handshake header %s retained: %q", key, values)
		}
	}
}

func echoServer(t *testing.T) string {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := Upgrade(w, r)
		if err != nil {
			return
		}
		defer c.Close()
		for {
			data, err := c.Read()
			if err != nil {
				return
			}
			if err := c.Write(data); err != nil {
				return
			}
		}
	}))
	t.Cleanup(server.Close)
	return "ws" + strings.TrimPrefix(server.URL, "http")
}

func dialTest(t *testing.T, address string, headers http.Header) *Conn {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	t.Cleanup(cancel)
	c, err := Dial(ctx, address, headers)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func TestAudioJSONAndConcurrentWriters(t *testing.T) {
	c := dialTest(t, echoServer(t), nil)
	pcm := []byte{0, 0, 1, 0, 255, 127, 0, 128}
	for _, event := range []string{
		`{"type":"session.update","session":{"type":"realtime","audio":{"input":{"format":{"type":"audio/pcm","rate":16000}},"output":{"format":{"type":"audio/pcm","rate":24000},"voice":"Puck"}}}}`,
		`{"type":"input_audio_buffer.append","audio":"` + base64.StdEncoding.EncodeToString(pcm) + `"}`,
		`{"type":"response.output_audio.delta","delta":"` + base64.StdEncoding.EncodeToString(pcm) + `"}`,
	} {
		if err := c.Write([]byte(event)); err != nil {
			t.Fatal(err)
		}
		got, err := c.Read()
		if err != nil || string(got) != event {
			t.Fatalf("audio JSON changed: %q %v", got, err)
		}
		var fields map[string]any
		if err := json.Unmarshal(got, &fields); err != nil {
			t.Fatal(err)
		}
		for _, key := range []string{"audio", "delta"} {
			if encoded, ok := fields[key].(string); ok {
				decoded, err := base64.StdEncoding.DecodeString(encoded)
				if err != nil || !bytes.Equal(decoded, pcm) {
					t.Fatalf("PCM bytes changed: %x %v", decoded, err)
				}
			}
		}
	}
	const count = 24
	errs := make(chan error, count)
	var writers sync.WaitGroup
	for i := range count {
		writers.Add(1)
		go func() { defer writers.Done(); errs <- c.Write([]byte(fmt.Sprintf(`{"sequence":%d}`, i))) }()
	}
	seen := make(map[int]bool)
	for range count {
		data, err := c.Read()
		if err != nil {
			t.Fatal(err)
		}
		var event struct {
			Sequence int `json:"sequence"`
		}
		if err := json.Unmarshal(data, &event); err != nil {
			t.Fatal(err)
		}
		if seen[event.Sequence] {
			t.Fatal("duplicated writer message")
		}
		seen[event.Sequence] = true
	}
	writers.Wait()
	for range count {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
}

func TestCloseAndContextCancelUnblockRead(t *testing.T) {
	address := echoServer(t)
	for _, mode := range []string{"close", "cancel", "deadline"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
			if mode == "deadline" {
				cancel()
				ctx, cancel = context.WithTimeout(t.Context(), 100*time.Millisecond)
			}
			defer cancel()
			c, err := Dial(ctx, address, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			done := make(chan error, 1)
			go func() { _, err := c.Read(); done <- err }()
			if mode == "close" {
				_ = c.Close()
				_ = c.Close()
			} else if mode == "cancel" {
				cancel()
			}
			select {
			case err := <-done:
				if err == nil {
					t.Fatal("read completed without a close error")
				}
			case <-time.After(2 * time.Second):
				t.Fatal("pending Read was not cancelled")
			}
			if err := c.Write([]byte(`{}`)); err == nil {
				t.Fatal("closed connection accepted a write")
			}
		})
	}
}

func TestMessageBoundariesAndJSONOnly(t *testing.T) {
	c := dialTest(t, echoServer(t), nil)
	limit := []byte(`"` + strings.Repeat("a", maxMessageBytes-2) + `"`)
	if err := c.Write(limit); err != nil {
		t.Fatal(err)
	}
	got, err := c.Read()
	if err != nil || !bytes.Equal(got, limit) {
		t.Fatalf("exact 1 MiB round trip failed: %v", err)
	}
	for _, invalid := range [][]byte{append(limit[:len(limit):len(limit)], ' '), []byte(`{broken}`), {'"', 0xff, '"'}} {
		if err := c.Write(invalid); err == nil {
			t.Fatal("invalid outbound message accepted")
		}
	}
	if err := c.Write([]byte(`{"still":"usable"}`)); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Read(); err != nil {
		t.Fatal(err)
	}
	for _, frame := range []struct {
		name string
		kind int
		data []byte
	}{
		{"binary", websocket.BinaryMessage, []byte(`{}`)},
		{"invalid-json", websocket.TextMessage, []byte(`not json`)},
		{"invalid-utf8", websocket.TextMessage, []byte{'"', 0xff, '"'}},
		{"oversized", websocket.TextMessage, append(limit[:len(limit):len(limit)], ' ')},
	} {
		t.Run(frame.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				ws, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
				if err != nil {
					return
				}
				defer ws.Close()
				_ = ws.WriteMessage(frame.kind, frame.data)
			}))
			defer server.Close()
			conn := dialTest(t, "ws"+strings.TrimPrefix(server.URL, "http"), nil)
			if data, err := conn.Read(); err == nil || data != nil {
				t.Fatalf("invalid inbound frame accepted: %d %v", len(data), err)
			}
		})
	}
}

func TestUpgradeOriginAndHandshake(t *testing.T) {
	address := echoServer(t)
	for _, origin := range []string{"", "http" + strings.TrimPrefix(address, "ws")} {
		headers := http.Header{}
		if origin != "" {
			headers.Set("Origin", origin)
		}
		c := dialTest(t, address, headers)
		_ = c.Close()
	}
	for _, origin := range []string{"https" + strings.TrimPrefix(address, "ws"), "https://different.invalid", "null", ""} {
		if _, err := Dial(t.Context(), address, http.Header{"Origin": {origin}}); err == nil {
			t.Fatalf("cross/invalid Origin accepted: %q", origin)
		}
	}
	response, err := http.Get("http" + strings.TrimPrefix(address, "ws"))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusSwitchingProtocols {
		t.Fatal("non-WebSocket request upgraded")
	}
}

func TestDialSafeFailureNoRedirectAndNormalTLSChecks(t *testing.T) {
	var forwarded atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { forwarded.Add(1); w.WriteHeader(500) }))
	defer target.Close()
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL+"?key=private", http.StatusTemporaryRedirect)
	}))
	defer redirect.Close()
	_, err := Dial(t.Context(), "ws"+strings.TrimPrefix(redirect.URL, "http")+"?key=private", http.Header{"Authorization": {"Bearer private"}})
	if err == nil || err.Error() != errDial.Error() || forwarded.Load() != 0 {
		t.Fatalf("unsafe redirect/dial failure: %v forwarded=%d", err, forwarded.Load())
	}
	for _, address := range []string{"://private", "ws://user:private@127.0.0.1/", "https://private.invalid/"} {
		if _, err := Dial(t.Context(), address, nil); err == nil || err.Error() != errDial.Error() {
			t.Fatalf("dial failure not static: %v", err)
		}
	}
	tlsServer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { t.Error("untrusted TLS should not reach handler") }))
	defer tlsServer.Close()
	if _, err := Dial(t.Context(), "wss"+strings.TrimPrefix(tlsServer.URL, "https"), nil); err == nil || err.Error() != errDial.Error() {
		t.Fatalf("untrusted TLS not rejected safely: %v", err)
	}
}

func TestConnectionLifetimeIsCapped(t *testing.T) {
	server := echoServer(t)
	c, err := Dial(t.Context(), server, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	deadline, ok := c.ctx.Deadline()
	if !ok || time.Until(deadline) > 2*time.Hour || time.Until(deadline) < time.Hour {
		t.Fatal("connection lifetime not capped at two hours")
	}
}
