// Package realtimews provides bounded text-JSON WebSocket transport. It does
// not translate realtime events, audio formats, or conversation semantics.
package realtimews

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/gorilla/websocket"
)

const (
	maxMessageBytes = 1 << 20
	writeTimeout    = 10 * time.Second
	maxLifetime     = 2 * time.Hour
)

var (
	errDial        = errors.New("realtimews: connection failed")
	errUpgrade     = errors.New("realtimews: upgrade failed")
	errClosed      = errors.New("realtimews: connection closed")
	errRead        = errors.New("realtimews: read failed")
	errWrite       = errors.New("realtimews: write failed")
	errTextOnly    = errors.New("realtimews: text JSON messages required")
	errInvalidJSON = errors.New("realtimews: invalid JSON message")
	errTooLarge    = errors.New("realtimews: message exceeds 1 MiB")
)

// Conn supports one reader, concurrent serialized writers, and concurrent
// Close. Close or cancellation of its originating context unblocks Read.
// Connections expire no later than two hours after establishment.
type Conn struct {
	socket    *websocket.Conn
	ctx       context.Context
	cancel    context.CancelFunc
	writeMu   sync.Mutex
	closeOnce sync.Once
}

// Dial establishes a WebSocket without following redirects. Headers are sent
// only to rawURL; errors never expose the URL, credentials, or response body.
// The context governs both connection establishment and connection lifetime.
func Dial(ctx context.Context, rawURL string, headers http.Header) (*Conn, error) {
	dialer := websocket.Dialer{Proxy: http.ProxyFromEnvironment, HandshakeTimeout: writeTimeout}
	socket, response, err := dialer.DialContext(ctx, rawURL, headers.Clone())
	if err != nil {
		if response != nil && response.Body != nil {
			_ = response.Body.Close()
		}
		return nil, errDial
	}
	return newConn(ctx, socket), nil
}

// Upgrade accepts same-origin browser requests and server-to-server requests
// without Origin. Its request context governs the connection lifetime.
func Upgrade(w http.ResponseWriter, r *http.Request) (*Conn, error) {
	upgrader := websocket.Upgrader{
		HandshakeTimeout: writeTimeout,
		CheckOrigin:      sameOrigin,
		Error: func(w http.ResponseWriter, _ *http.Request, status int, _ error) {
			http.Error(w, "WebSocket upgrade failed", status)
		},
	}
	responseHeaders := w.Header().Clone()
	// Gorilla writes the 101 directly: carry application headers explicitly,
	// leaving negotiation and HTTP connection/framing fields to the upgrader.
	for key := range responseHeaders {
		lower := strings.ToLower(key)
		switch lower {
		case "connection", "upgrade", "content-length", "transfer-encoding", "trailer", "keep-alive", "proxy-connection", "te":
			delete(responseHeaders, key)
		default:
			if strings.HasPrefix(lower, "sec-websocket-") {
				delete(responseHeaders, key)
			}
		}
	}
	socket, err := upgrader.Upgrade(w, r, responseHeaders)
	if err != nil {
		return nil, errUpgrade
	}
	return newConn(r.Context(), socket), nil
}

func sameOrigin(r *http.Request) bool {
	origins := r.Header.Values("Origin")
	if len(origins) == 0 {
		return true
	}
	if len(origins) != 1 {
		return false
	}
	origin, err := url.Parse(origins[0])
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	return err == nil && origin.Scheme == scheme && strings.EqualFold(origin.Host, r.Host) &&
		origin.User == nil && origin.Path == "" && origin.RawQuery == "" && origin.Fragment == ""
}

func newConn(parent context.Context, socket *websocket.Conn) *Conn {
	ctx, cancel := context.WithTimeout(parent, maxLifetime)
	c := &Conn{socket: socket, ctx: ctx, cancel: cancel}
	socket.SetReadLimit(maxMessageBytes)
	context.AfterFunc(ctx, func() { _ = c.Close() })
	return c
}

// Read returns one complete text JSON message, limited to 1 MiB. Invalid or
// binary messages close the connection. A normal peer close returns io.EOF.
func (c *Conn) Read() ([]byte, error) {
	kind, data, err := c.socket.ReadMessage()
	if err != nil {
		failure := errRead
		if errors.Is(err, websocket.ErrReadLimit) {
			failure = errTooLarge
		} else if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
			failure = io.EOF
		} else if c.ctx.Err() != nil {
			failure = errClosed
		}
		_ = c.Close()
		return nil, failure
	}
	if kind != websocket.TextMessage {
		_ = c.Close()
		return nil, errTextOnly
	}
	if !utf8.Valid(data) || !json.Valid(data) {
		_ = c.Close()
		return nil, errInvalidJSON
	}
	return data, nil
}

// Write serializes complete text JSON messages with a ten-second write
// deadline. Invalid or oversized caller input is rejected without sending it.
func (c *Conn) Write(data []byte) error {
	if len(data) > maxMessageBytes {
		return errTooLarge
	}
	if !utf8.Valid(data) || !json.Valid(data) {
		return errInvalidJSON
	}
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	if c.ctx.Err() != nil {
		return errClosed
	}
	if err := c.socket.SetWriteDeadline(time.Now().Add(writeTimeout)); err != nil {
		_ = c.Close()
		return errWrite
	}
	if err := c.socket.WriteMessage(websocket.TextMessage, data); err != nil {
		_ = c.Close()
		return errWrite
	}
	return nil
}

// Close is idempotent and immediately closes the underlying socket, including
// pending reads/writes. It does not wait for a peer close handshake.
func (c *Conn) Close() error {
	c.closeOnce.Do(func() { c.cancel(); _ = c.socket.Close() })
	return nil
}
