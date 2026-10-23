package extension

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	core "github.com/kayiik/llmgw-core"
)

func TestAllowanceRoundtripAndCredentialSeparation(t *testing.T) {
	zero, fraction := 0.0, 0.75
	now := time.Date(2026, 10, 22, 19, 0, 0, 0, time.UTC)
	reset := now.Add(time.Hour)
	want := AllowanceResponse{Status: "observed", Observations: []AllowanceObservation{
		{ProjectID: "project", ModelID: "m-zero", ObservedAt: now, RemainingFraction: &zero, ResetAt: &reset},
		{ModelID: "m-unknown", ObservedAt: now},
		{ModelID: "m-fraction", ObservedAt: now, RemainingFraction: &fraction},
	}}
	credentials := []*core.Credential{
		{Token: "access-one", TokenType: "Bearer", AccountID: "account-one", Metadata: map[string]string{"project": "one"}},
		{APIKey: "api-two", AccountID: "account-two"},
		nil,
	}
	index := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != PathPrefix+"antigravity/allowance" || r.Header.Get("Accept") != core.ContentTypeJSON {
			t.Errorf("wrong allowance request: %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer daemon-secret" {
			t.Error("daemon authorization missing or replaced by provider credential")
		}
		expected := make(http.Header)
		SetCredentialHeaders(expected, credentials[index])
		for _, name := range []string{HeaderCredentialToken, HeaderCredentialAccount, HeaderCredentialType, HeaderCredentialMetadata} {
			if r.Header.Get(name) != expected.Get(name) {
				t.Errorf("credential header %s leaked across calls", name)
			}
		}
		index++
		w.Header().Set("Content-Type", core.ContentTypeJSON)
		_ = json.NewEncoder(w).Encode(want)
	}))
	defer server.Close()
	client, err := NewClient(Config{BaseURL: server.URL, Secret: "daemon-secret"})
	if err != nil {
		t.Fatal(err)
	}
	for _, credential := range credentials {
		got, err := client.Allowance(t.Context(), "antigravity", credential)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("roundtrip: got %+v, err %v, want %+v", got, err, want)
		}
		raw, err := json.Marshal(got)
		if err != nil || !strings.Contains(string(raw), `"remaining_fraction":0`) || !strings.Contains(string(raw), `"model_id":"m-unknown","observed_at":"2026-10-22T19:00:00Z"}`) {
			t.Fatalf("zero/missing wire distinction lost: %s, %v", raw, err)
		}
	}
}

type allowanceTransport func(*http.Request) (*http.Response, error)

func (f allowanceTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestAllowanceUnsupportedAndSanitizedFailures(t *testing.T) {
	for _, status := range []int{404, 401, 500} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			client, err := NewClient(Config{BaseURL: "http://daemon.test", HTTPClient: &http.Client{Transport: allowanceTransport(func(r *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: status, Header: http.Header{"Retry-After": []string{"2"}}, Body: io.NopCloser(strings.NewReader(`{"error":{"message":"provider-secret daemon-secret raw-diagnostic"}}`))}, nil
			})}})
			if err != nil {
				t.Fatal(err)
			}
			got, err := client.Allowance(t.Context(), "antigravity", &core.Credential{Token: "provider-secret"})
			if err == nil || !reflect.DeepEqual(got, AllowanceResponse{}) || core.ClassifyError(err).StatusCode != status {
				t.Fatalf("status lost or fabricated observation: %+v / %v", got, err)
			}
			for _, secret := range []string{"provider-secret", "daemon-secret", "raw-diagnostic"} {
				if strings.Contains(err.Error(), secret) {
					t.Fatalf("error exposed daemon diagnostic: %v", err)
				}
			}
		})
	}
}

func TestAllowanceBoundedJSONAndFractions(t *testing.T) {
	for _, body := range []string{
		`{"status":"observed","observations":[{"remaining_fraction":-0.1}]}`,
		`{"status":"observed","observations":[{"remaining_fraction":1.1}]}`,
		`{"status":"observed","observations":[{"remaining_fraction":1e999}]}`,
		`{"status":"invalid"}`, `null`, `not-json provider-secret`,
		strings.Repeat(" ", maxJSONBody) + `{"status":"unknown"}`,
	} {
		client, err := NewClient(Config{BaseURL: "http://daemon.test", HTTPClient: &http.Client{Transport: allowanceTransport(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(body))}, nil
		})}})
		if err != nil {
			t.Fatal(err)
		}
		got, err := client.Allowance(t.Context(), "antigravity", nil)
		if err == nil || strings.Contains(err.Error(), "provider-secret") || !reflect.DeepEqual(got, AllowanceResponse{}) {
			t.Fatalf("invalid observation accepted/exposed: %+v / %v", got, err)
		}
	}
	client, err := NewClient(Config{BaseURL: "http://daemon.test", HTTPClient: &http.Client{Transport: allowanceTransport(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"status":"unknown"}`))}, nil
	})}})
	if err != nil {
		t.Fatal(err)
	}
	if got, err := client.Allowance(t.Context(), "antigravity", nil); err != nil || got.Status != "unknown" || len(got.Observations) != 0 {
		t.Fatalf("unknown fabricated data: %+v / %v", got, err)
	}
}

func TestAllowanceControlDeadlineAndCancellation(t *testing.T) {
	for _, cancelCaller := range []bool{false, true} {
		ctx, cancel := context.WithCancel(t.Context())
		client, err := NewClient(Config{BaseURL: "http://daemon.test", HTTPClient: &http.Client{Transport: allowanceTransport(func(r *http.Request) (*http.Response, error) {
			deadline, ok := r.Context().Deadline()
			if !ok || time.Until(deadline) > DefaultControlTimeout {
				t.Error("allowance lacks bounded control deadline")
			}
			if cancelCaller {
				cancel()
				<-r.Context().Done()
				return nil, r.Context().Err()
			}
			return nil, errors.New("transport diagnostic provider-secret")
		})}})
		if err != nil {
			t.Fatal(err)
		}
		_, err = client.Allowance(ctx, "antigravity", nil)
		cancel()
		if err == nil || strings.Contains(err.Error(), "provider-secret") || (cancelCaller && !errors.Is(err, context.Canceled)) {
			t.Fatalf("cancellation/static transport error changed: %v", err)
		}
	}
}
