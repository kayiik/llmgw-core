package runtime_test

import (
	"context"
	"errors"
	"maps"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kayiik/llm-provider-auth/tokenstore"
	core "github.com/kayiik/llmgw-core"
	coreruntime "github.com/kayiik/llmgw-core/runtime"
)

// This is a conditional reproduction, not proof of the saved Antigravity
// failure's cause: its telemetry contains no before/after binding dimensions.
// A provider refresh may discover a project without changing caller, account,
// request, connection or settings. Core nevertheless binds project metadata.
func TestQualificationPreparedRefreshProjectMetadata(t *testing.T) {
	for _, stage := range []string{"expired-token", "rejected-token"} {
		for _, projectChange := range []string{"unchanged", "added", "replaced"} {
			t.Run(stage+"/"+projectChange, func(t *testing.T) {
				ctx := t.Context()
				now := time.Now()
				expiry := now.Add(time.Hour)
				if stage == "expired-token" {
					expiry = now.Add(-time.Hour)
				}
				metadata := map[string]string{"project_id": "project-before"}
				if projectChange == "added" {
					metadata = nil
				}
				store := core.NewMemoryCredentialStore()
				if _, err := store.Save(ctx, "private-connection", tokenstore.Record{
					AccessToken: "synthetic-old", RefreshToken: "synthetic-refresh",
					TokenType: "Bearer", AccountID: "synthetic-account", Expiry: expiry,
					Metadata: metadata,
				}); err != nil {
					t.Fatal(err)
				}
				caller := core.Caller{ID: "synthetic-human", Kind: core.CallerHuman}
				store.Bind(caller, "antigravity-fixture", "private-connection")
				settings := coreruntime.NewMemorySettings(struct{}{})
				_, generation := settings.Snapshot()
				counts := &counters{}
				provider := &describedProvider{fakeProvider: &fakeProvider{
					counters: counts,
					reject: func(c *core.Credential) bool {
						return c != nil && c.Token == "synthetic-old"
					},
				}}
				var refreshes atomic.Int32
				r, err := coreruntime.New(coreruntime.Options[struct{}]{
					Settings: settings, Credentials: store,
					Providers: func(struct{}, string) (core.Provider, error) { return provider, nil },
					Refresh: func(struct{}, string) tokenstore.RefreshFunc {
						return func(_ context.Context, current tokenstore.Record) (tokenstore.Record, error) {
							refreshes.Add(1)
							current.AccessToken = "synthetic-fresh"
							current.Expiry = now.Add(time.Hour)
							current.Metadata = maps.Clone(current.Metadata)
							if projectChange != "unchanged" {
								if current.Metadata == nil {
									current.Metadata = map[string]string{}
								}
								current.Metadata["project_id"] = "project-after"
							}
							return current, nil
						}
					},
				})
				if err != nil {
					t.Fatal(err)
				}
				request := core.Request{Surface: core.ModelSurfaceChatCompletions,
					Model: "gemini-2.5-flash-lite", ContentType: core.ContentTypeJSON,
					Body: []byte(`{"messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}`)}
				prepare := func() *coreruntime.PreparedCall {
					t.Helper()
					call, err := r.PrepareCall(ctx, caller, "antigravity-fixture", request, core.CallPreparationOptions{})
					if err != nil {
						t.Fatal(err)
					}
					return call
				}
				prepared := prepare()
				if refreshes.Load() != 0 || counts.invokes.Load() != 0 {
					t.Fatal("preparation performed refresh or inference")
				}
				_, err = r.InvokePrepared(ctx, caller, "antigravity-fixture", request, prepared, core.CallPolicy{})
				wantInvokes := int32(1)
				if projectChange == "unchanged" {
					if err != nil {
						t.Fatalf("same-account token-only refresh: %v", err)
					}
					if stage == "rejected-token" {
						wantInvokes = 2
					}
				} else {
					if !errors.Is(err, core.ErrPreparedCallStale) {
						t.Fatalf("project metadata drift error=%v, want prepared stale", err)
					}
					if stage == "expired-token" {
						wantInvokes = 0
					}
				}
				if counts.invokes.Load() != wantInvokes || refreshes.Load() != 1 {
					t.Fatalf("inference=%d want=%d refresh=%d want=1", counts.invokes.Load(), wantInvokes, refreshes.Load())
				}
				key, resolveErr := store.Resolve(ctx, caller, "antigravity-fixture")
				stored, loadErr := store.Load(ctx, "private-connection")
				_, finalGeneration := settings.Snapshot()
				if resolveErr != nil || loadErr != nil || key != "private-connection" ||
					stored.AccountID != "synthetic-account" || stored.TokenType != "Bearer" ||
					stored.AccessToken != "synthetic-fresh" || finalGeneration != generation {
					t.Fatal("fixture unexpectedly changed account, binding or settings, or failed to persist refresh")
				}
				if projectChange != "unchanged" {
					if stored.Metadata["project_id"] != "project-after" {
						t.Fatal("refresh did not persist project discovery")
					}
					// A new admission using the now-stable metadata succeeds; the
					// stale handle is never reused and no refresh loop is hidden.
					if _, err := r.InvokePrepared(ctx, caller, "antigravity-fixture", request, prepare(), core.CallPolicy{}); err != nil {
						t.Fatalf("fresh admission after project discovery: %v", err)
					}
					if refreshes.Load() != 1 || counts.invokes.Load() != wantInvokes+1 {
						t.Fatal("fresh admission repeated refresh or did not perform exactly one inference")
					}
				}
			})
		}
	}
}
