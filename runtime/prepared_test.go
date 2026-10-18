package runtime_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kayiik/llm-provider-auth/tokenstore"

	core "github.com/kayiik/llmgw-core"
	coreruntime "github.com/kayiik/llmgw-core/runtime"
)

type validationBarrierStore struct {
	core.CredentialStore
	resolves         atomic.Int32
	blockAt          atomic.Int32
	blocked          atomic.Bool
	entered, release chan struct{}
}

func (s *validationBarrierStore) Resolve(ctx context.Context, caller core.Caller, instance string) (string, error) {
	if s.resolves.Add(1) == s.blockAt.Load() {
		s.blocked.Store(true)
	}
	return s.CredentialStore.Resolve(ctx, caller, instance)
}
func (s *validationBarrierStore) Load(ctx context.Context, key string) (tokenstore.Record, error) {
	if s.blocked.Swap(false) {
		close(s.entered)
		select {
		case <-s.release:
		case <-ctx.Done():
			return tokenstore.Record{}, ctx.Err()
		}
	}
	return s.CredentialStore.Load(ctx, key)
}

func TestPreparedFinalValidationFencesGenerationAfterCredentialRead(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	settings := coreruntime.NewMemorySettings(struct{}{})
	counts := &counters{}
	p := &describedProvider{fakeProvider: &fakeProvider{counters: counts}}
	base := core.NewMemoryCredentialStore()
	if _, err := base.Save(ctx, "key", core.APIKeyRecord("fixture")); err != nil {
		t.Fatal(err)
	}
	base.BindShared("one", "key")
	store := &validationBarrierStore{CredentialStore: base, entered: make(chan struct{}), release: make(chan struct{})}
	r, err := coreruntime.New(coreruntime.Options[struct{}]{Settings: settings, Credentials: store, Providers: func(struct{}, string) (core.Provider, error) { return p, nil }})
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := r.PrepareCall(ctx, core.LocalCaller(), "one", chat, core.CallPreparationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// Entry validation (including final key read), Token resolution, post-
	// Token validation, then the execution-context fence's final Load.
	store.blockAt.Store(store.resolves.Load() + 6)
	done := make(chan error, 1)
	go func() {
		_, err := r.InvokePrepared(ctx, core.LocalCaller(), "one", chat, prepared, core.CallPolicy{})
		done <- err
	}()
	select {
	case <-store.entered:
	case <-ctx.Done():
		t.Fatal("final validation did not reach barrier")
	}
	settings.Update(struct{}{})
	close(store.release)
	if err := <-done; !errors.Is(err, core.ErrPreparedCallStale) {
		t.Fatalf("final fence error=%v", err)
	}
	if counts.invokes.Load() != 0 {
		t.Fatal("old generation reached inference after credential wait")
	}
}

type describedProvider struct {
	*fakeProvider
	preparations atomic.Int32
}

func (p *describedProvider) PrepareCall(_ context.Context, request core.Request, options core.CallPreparationOptions) (core.PreparedCall, error) {
	p.preparations.Add(1)
	frozen := request
	frozen.Body = append([]byte(nil), request.Body...)
	d := core.CallDescription{Ingress: request.Surface, Model: request.Model, Mode: options.Mode, Operation: core.ModelOperationChat,
		UpstreamAPI: core.UpstreamAPIChatCompletions, WirePreserved: core.SupportSupported,
		UpstreamTransport: core.CallTransportJSON, OutputTransport: core.CallTransportJSON, Assessment: core.CallAssessment{Feasibility: core.SupportSupported}}
	return core.NewPreparedCall(d, func(ctx context.Context, credential *core.Credential) (core.Response, error) {
		frozen.Credential = credential
		return p.Invoke(ctx, frozen)
	},
		func(ctx context.Context, credential *core.Credential) (core.StreamIter, error) {
			frozen.Credential = credential
			return p.Stream(ctx, frozen)
		}), nil
}

func TestPreparedRuntimeRejectsBindingChangesBeforeIO(t *testing.T) {
	for _, change := range []string{"body", "model", "raw-model-bytes", "surface", "headers", "caller", "kind", "project", "instance", "generation", "credential", "raw-credential-bytes", "runtime"} {
		t.Run(change, func(t *testing.T) {
			ctx := t.Context()
			settings := coreruntime.NewMemorySettings(struct{}{})
			counts := &counters{}
			provider := &describedProvider{fakeProvider: &fakeProvider{counters: counts}}
			store := core.NewMemoryCredentialStore()
			material := "before"
			if change == "raw-credential-bytes" {
				material = "before\xff"
			}
			_, err := store.Save(ctx, "key", core.APIKeyRecord(material))
			if err != nil {
				t.Fatal(err)
			}
			store.BindShared("instance", "key")
			options := coreruntime.Options[struct{}]{Settings: settings, Credentials: store, Providers: func(struct{}, string) (core.Provider, error) { return provider, nil }}
			r, err := coreruntime.New(options)
			if err != nil {
				t.Fatal(err)
			}
			caller := core.Caller{ID: "caller", Kind: core.CallerService, ProjectID: "project"}
			request := core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "m", Body: []byte(`{"messages":[]}`), ContentType: core.ContentTypeJSON, Header: http.Header{"X-Test": {"original"}}}
			if change == "raw-model-bytes" {
				request.Model = "m\xff"
			}
			prepared, err := r.PrepareCall(ctx, caller, "instance", request, core.CallPreparationOptions{})
			if err != nil {
				t.Fatal(err)
			}
			instance := "instance"
			switch change {
			case "body":
				request.Body[1] = 'x'
			case "model":
				request.Model = "other"
			case "raw-model-bytes":
				request.Model = "m\xfe"
			case "surface":
				request.Surface = core.ModelSurfaceResponses
			case "headers":
				request.Header.Set("X-Test", "changed")
			case "caller":
				caller.ID = "other"
			case "kind":
				caller.Kind = core.CallerHuman
			case "project":
				caller.ProjectID = "other"
			case "instance":
				instance = "other"
			case "generation":
				settings.Update(struct{}{})
			case "credential":
				_, err = store.Save(ctx, "key", core.APIKeyRecord("after"))
				if err != nil {
					t.Fatal(err)
				}
			case "raw-credential-bytes":
				_, err = store.Save(ctx, "key", core.APIKeyRecord("before\xfe"))
				if err != nil {
					t.Fatal(err)
				}
			case "runtime":
				r, err = coreruntime.New(options)
				if err != nil {
					t.Fatal(err)
				}
			}
			_, err = r.InvokePrepared(ctx, caller, instance, request, prepared, core.CallPolicy{})
			if !errors.Is(err, core.ErrPreparedCallStale) {
				t.Fatalf("%s: err=%v", change, err)
			}
			if counts.invokes.Load() != 0 {
				t.Fatal("stale binding reached inference")
			}
		})
	}
}

func TestPreparedRuntimeExecutesOnceWithoutRepreparing(t *testing.T) {
	counts := &counters{}
	p := &describedProvider{fakeProvider: &fakeProvider{counters: counts}}
	r, err := coreruntime.New(coreruntime.Options[struct{}]{Settings: coreruntime.NewMemorySettings(struct{}{}), Providers: func(struct{}, string) (core.Provider, error) { return p, nil }})
	if err != nil {
		t.Fatal(err)
	}
	caller := core.LocalCaller()
	prepared, err := r.PrepareCall(t.Context(), caller, "one", chat, core.CallPreparationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	copied := *prepared
	if _, err := r.InvokePrepared(t.Context(), caller, "one", chat, prepared, core.CallPolicy{}); err != nil {
		t.Fatal(err)
	}
	if _, err := r.InvokePrepared(t.Context(), caller, "one", chat, &copied, core.CallPolicy{}); !errors.Is(err, core.ErrPreparedCallConsumed) {
		t.Fatalf("reused: %v", err)
	}
	if counts.invokes.Load() != 1 || p.preparations.Load() != 1 {
		t.Fatalf("invokes=%d preparations=%d", counts.invokes.Load(), p.preparations.Load())
	}
}

func TestPreparedRuntimeRefreshWaitRevalidatesBinding(t *testing.T) {
	for _, stage := range []string{"expired-token", "rejected-token"} {
		for _, change := range []string{"none", "generation", "connection"} {
			t.Run(stage+"/"+change, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
				defer cancel()
				now := time.Now()
				settings := coreruntime.NewMemorySettings(struct{}{})
				store := core.NewMemoryCredentialStore()
				expiry := now.Add(time.Hour)
				if stage == "expired-token" {
					expiry = now.Add(-time.Hour)
				}
				if _, err := store.Save(ctx, "key", tokenstore.Record{AccessToken: "old", RefreshToken: "refresh", TokenType: "Bearer", AccountID: "account", Expiry: expiry}); err != nil {
					t.Fatal(err)
				}
				if _, err := store.Save(ctx, "other", core.APIKeyRecord("different")); err != nil {
					t.Fatal(err)
				}
				store.BindShared("one", "key")
				counts := &counters{}
				p := &describedProvider{fakeProvider: &fakeProvider{counters: counts, reject: func(c *core.Credential) bool { return c.Token == "old" }}}
				entered, release := make(chan struct{}), make(chan struct{})
				var refreshes atomic.Int32
				r, err := coreruntime.New(coreruntime.Options[struct{}]{Settings: settings, Credentials: store, Providers: func(struct{}, string) (core.Provider, error) { return p, nil },
					Refresh: func(struct{}, string) tokenstore.RefreshFunc {
						return func(ctx context.Context, current tokenstore.Record) (tokenstore.Record, error) {
							refreshes.Add(1)
							close(entered)
							select {
							case <-release:
							case <-ctx.Done():
								return tokenstore.Record{}, ctx.Err()
							}
							current.AccessToken = "fresh"
							current.Expiry = now.Add(time.Hour)
							return current, nil
						}
					}})
				if err != nil {
					t.Fatal(err)
				}
				prepared, err := r.PrepareCall(ctx, core.LocalCaller(), "one", chat, core.CallPreparationOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if refreshes.Load() != 0 || counts.invokes.Load() != 0 {
					t.Fatal("preparation performed refresh/inference")
				}
				done := make(chan error, 1)
				go func() {
					_, err := r.InvokePrepared(ctx, core.LocalCaller(), "one", chat, prepared, core.CallPolicy{})
					done <- err
				}()
				select {
				case <-entered:
				case <-ctx.Done():
					t.Fatal("refresh did not reach barrier")
				}
				switch change {
				case "generation":
					settings.Update(struct{}{})
				case "connection":
					store.BindShared("one", "other")
				}
				close(release)
				err = <-done
				if change == "none" {
					if err != nil {
						t.Fatal(err)
					}
				} else if !errors.Is(err, core.ErrPreparedCallStale) {
					t.Fatalf("changed binding: %v", err)
				}
				want := int32(0)
				if stage == "rejected-token" {
					want++
				}
				if change == "none" {
					want++
				}
				if counts.invokes.Load() != want || refreshes.Load() != 1 || p.preparations.Load() != 1 {
					t.Fatalf("invokes=%d want=%d refreshes=%d prepares=%d", counts.invokes.Load(), want, refreshes.Load(), p.preparations.Load())
				}
			})
		}
	}
}

type postFrameAuthError struct{ sent bool }

func (s *postFrameAuthError) Next() ([]byte, error) {
	if !s.sent {
		s.sent = true
		return []byte("data: {\"delta\":\"partial\"}\n\n"), nil
	}
	return nil, &core.ProviderError{Message: "late auth error", Class: core.ProviderErrorAuth, Classification: core.ProviderErrorClassification{StatusCode: 401}}
}
func (*postFrameAuthError) Close() error { return nil }

type preparedStreamProvider struct {
	*describedProvider
	opens atomic.Int32
}

func (p *preparedStreamProvider) PrepareCall(ctx context.Context, request core.Request, options core.CallPreparationOptions) (core.PreparedCall, error) {
	base, err := p.describedProvider.PrepareCall(ctx, request, options)
	if err != nil {
		return nil, err
	}
	d := base.Description()
	d.UpstreamTransport = core.CallTransportSSE
	d.OutputTransport = core.CallTransportSSE
	return core.NewPreparedCall(d, nil, func(_ context.Context, c *core.Credential) (core.StreamIter, error) {
		p.opens.Add(1)
		if c.Token == "old" {
			return nil, &core.ProviderError{Message: "open auth error", Class: core.ProviderErrorAuth, Classification: core.ProviderErrorClassification{StatusCode: 401}}
		}
		return &postFrameAuthError{}, nil
	}), nil
}

func TestPreparedRuntimeRefreshesOnlyStreamOpenNeverAfterFrame(t *testing.T) {
	now := time.Now()
	store := core.NewMemoryCredentialStore()
	if _, err := store.Save(t.Context(), "key", tokenstore.Record{AccessToken: "old", RefreshToken: "refresh", TokenType: "Bearer", AccountID: "account", Expiry: now.Add(time.Hour)}); err != nil {
		t.Fatal(err)
	}
	store.BindShared("one", "key")
	p := &preparedStreamProvider{describedProvider: &describedProvider{fakeProvider: &fakeProvider{counters: &counters{}}}}
	var refreshes atomic.Int32
	r, err := coreruntime.New(coreruntime.Options[struct{}]{Settings: coreruntime.NewMemorySettings(struct{}{}), Credentials: store, Providers: func(struct{}, string) (core.Provider, error) { return p, nil }, Refresh: func(struct{}, string) tokenstore.RefreshFunc {
		return func(_ context.Context, current tokenstore.Record) (tokenstore.Record, error) {
			refreshes.Add(1)
			current.AccessToken = "fresh"
			return current, nil
		}
	}})
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := r.PrepareCall(t.Context(), core.LocalCaller(), "one", chat, core.CallPreparationOptions{Mode: core.CallModeStream})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := r.StreamPrepared(t.Context(), core.LocalCaller(), "one", chat, prepared, core.CallPolicy{})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if _, err := stream.Next(); err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Next(); err == nil || err == io.EOF {
		t.Fatalf("lost post-frame failure: %v", err)
	}
	if _, err := r.StreamPrepared(t.Context(), core.LocalCaller(), "one", chat, prepared, core.CallPolicy{}); !errors.Is(err, core.ErrPreparedCallConsumed) {
		t.Fatalf("post-frame replay: %v", err)
	}
	if p.opens.Load() != 2 || refreshes.Load() != 1 || p.preparations.Load() != 1 {
		t.Fatalf("opens=%d refresh=%d prepare=%d", p.opens.Load(), refreshes.Load(), p.preparations.Load())
	}
}

func TestPreparedConcurrentExpiredTokensShareRefresh(t *testing.T) {
	now := time.Now()
	store := core.NewMemoryCredentialStore()
	if _, err := store.Save(t.Context(), "key", tokenstore.Record{AccessToken: "old", RefreshToken: "refresh", TokenType: "Bearer", AccountID: "account", Expiry: now.Add(-time.Hour)}); err != nil {
		t.Fatal(err)
	}
	store.BindShared("one", "key")
	counts := &counters{}
	p := &describedProvider{fakeProvider: &fakeProvider{counters: counts}}
	var refreshes atomic.Int32
	r, err := coreruntime.New(coreruntime.Options[struct{}]{Settings: coreruntime.NewMemorySettings(struct{}{}), Credentials: store, Providers: func(struct{}, string) (core.Provider, error) { return p, nil }, Refresh: func(struct{}, string) tokenstore.RefreshFunc {
		return func(_ context.Context, current tokenstore.Record) (tokenstore.Record, error) {
			refreshes.Add(1)
			current.AccessToken = "fresh"
			current.Expiry = now.Add(time.Hour)
			return current, nil
		}
	}})
	if err != nil {
		t.Fatal(err)
	}
	const count = 8
	prepared := make([]*coreruntime.PreparedCall, count)
	for i := range prepared {
		prepared[i], err = r.PrepareCall(t.Context(), core.LocalCaller(), "one", chat, core.CallPreparationOptions{})
		if err != nil {
			t.Fatal(err)
		}
	}
	if refreshes.Load() != 0 {
		t.Fatal("preparation refreshed expired token")
	}
	var workers sync.WaitGroup
	for _, call := range prepared {
		workers.Go(func() {
			if _, err := r.InvokePrepared(t.Context(), core.LocalCaller(), "one", chat, call, core.CallPolicy{}); err != nil {
				t.Error(err)
			}
		})
	}
	workers.Wait()
	if refreshes.Load() != 1 || counts.invokes.Load() != count {
		t.Fatalf("refresh=%d inference=%d", refreshes.Load(), counts.invokes.Load())
	}
	for _, credential := range counts.credentials() {
		if credential.Token != "fresh" {
			t.Fatal("expired token reached inference")
		}
	}
}

func TestPreparedCancellationAndModeMismatchDoNotExecute(t *testing.T) {
	counts := &counters{}
	p := &describedProvider{fakeProvider: &fakeProvider{counters: counts}}
	r, err := coreruntime.New(coreruntime.Options[struct{}]{Settings: coreruntime.NewMemorySettings(struct{}{}), Providers: func(struct{}, string) (core.Provider, error) { return p, nil }})
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := r.PrepareCall(t.Context(), core.LocalCaller(), "one", chat, core.CallPreparationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.StreamPrepared(t.Context(), core.LocalCaller(), "one", chat, prepared, core.CallPolicy{}); !errors.Is(err, core.ErrPreparedCallMode) {
		t.Fatalf("wrong mode: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := r.InvokePrepared(ctx, core.LocalCaller(), "one", chat, prepared, core.CallPolicy{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation=%v", err)
	}
	if counts.invokes.Load() != 0 {
		t.Fatal("canceled/mismatched operation executed")
	}
}

func TestPreparedHandleFormattingDoesNotExposeBoundPrivateState(t *testing.T) {
	store := core.NewMemoryCredentialStore()
	if _, err := store.Save(t.Context(), "key", core.APIKeyRecord("fixture-secret-never-log")); err != nil {
		t.Fatal(err)
	}
	store.BindShared("one", "key")
	p := &describedProvider{fakeProvider: &fakeProvider{counters: &counters{}}}
	r, err := coreruntime.New(coreruntime.Options[string]{Settings: coreruntime.NewMemorySettings("private-config-never-log"), Credentials: store, Providers: func(string, string) (core.Provider, error) { return p, nil }})
	if err != nil {
		t.Fatal(err)
	}
	prepared, err := r.PrepareCall(t.Context(), core.LocalCaller(), "one", chat, core.CallPreparationOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for _, format := range []string{"%v", "%+v", "%#v", "%s", "%q", "%x", "%d"} {
		text := fmt.Sprintf(format, prepared)
		if strings.Contains(text, "never-log") {
			t.Fatalf("handle leaked state for %s", format)
		}
	}
}
