package providers

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	core "github.com/kayiik/llmgw-core"
)

// Ported from the gateway's TestUpstreamErrorsNameTheRealCause. The message
// names the cause and never quotes Google; the cause's diagnostic is the
// gateway's message.
func TestGoogleUpstreamErrorsNameTheRealCause(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		name, body, message, diagnostic string
		status                          int
		class                           core.ProviderErrorClass
		transient, billing              bool
	}{
		// Billing is its own class: it fails over and counts against the
		// provider, but waiting will not mend it.
		{
			name: "billing exhausted", status: 429, class: core.ProviderErrorBilling, billing: true,
			body:       `{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"Your prepayment credits are depleted."}}`,
			message:    "ai_studio returned HTTP 429 (RESOURCE_EXHAUSTED): provider billing exhausted",
			diagnostic: "ai_studio: provider billing exhausted — Your prepayment credits are depleted.",
		},
		{
			name: "rate limited", status: 429, class: core.ProviderErrorRateLimited, transient: true,
			body:       `{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"Quota exceeded for requests per minute."}}`,
			message:    "ai_studio returned HTTP 429 (RESOURCE_EXHAUSTED)",
			diagnostic: "ai_studio: Quota exceeded for requests per minute.",
		},
		{
			name: "model not available", status: 404, class: core.ProviderErrorUpstream,
			body:       `{"error":{"code":404,"status":"NOT_FOUND","message":"Publisher model ... was not found or your project does not have access to it."}}`,
			message:    "ai_studio returned HTTP 404 (NOT_FOUND): model not available to this project or location",
			diagnostic: "ai_studio: model not available to this project or location — Publisher model ... was not found or your project does not have access to it.",
		},
		{
			name: "credential rejected", status: 403, class: core.ProviderErrorForbidden,
			body:       `{"error":{"code":403,"status":"PERMISSION_DENIED","message":"denied"}}`,
			message:    "ai_studio returned HTTP 403 (PERMISSION_DENIED): credential rejected",
			diagnostic: "ai_studio: credential rejected — denied",
		},
		{
			name: "not an envelope", status: 500, class: core.ProviderErrorUpstream, transient: true,
			body: "  upstream broke \n", message: "ai_studio returned HTTP 500", diagnostic: "ai_studio: upstream broke",
		},
		{
			name: "no body", status: 501, class: core.ProviderErrorUpstream, message: "ai_studio returned HTTP 501", diagnostic: "ai_studio: ",
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			_, base := newGoogleFake(t, googleAnswer(testCase.status, testCase.body))
			provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
			_, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[{"role":"user","content":"hi"}]}`, googleKey("k")))
			var failure *core.ProviderError
			var cause *InvocationError
			if !errors.As(err, &failure) || failure.Message != testCase.message || failure.Class != testCase.class ||
				!errors.As(err, &cause) || cause.Msg != testCase.diagnostic || cause.Status != testCase.status {
				t.Fatalf("error = %#v (cause %#v)", err, cause)
			}
			// The gateway's status set: only transient statuses retry, fail
			// over and count against the provider.
			want := core.ProviderErrorClassification{StatusCode: testCase.status, Retryable: testCase.transient,
				FailoverEligible: testCase.transient || testCase.billing, CircuitFailure: testCase.transient || testCase.billing}
			if classification := core.ClassifyError(err); classification != want {
				t.Fatalf("classification = %+v, want %+v", classification, want)
			}
		})
	}
}

func TestGoogleRefusedAnswersKeepRetryAfterAndRedactTheDiagnostic(t *testing.T) {
	t.Parallel()
	secret := "sk-" + strings.Repeat("q", 24)
	fake, base := newGoogleFake(t, func(r *http.Request) (int, string) {
		return http.StatusTooManyRequests, `{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"key ` + secret + ` of owner@example.test ran out of credit"}}`
	})
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	_, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, googleKey("k")))
	var cause *InvocationError
	if !errors.As(err, &cause) || !strings.Contains(err.Error(), "provider billing exhausted") {
		t.Fatalf("error = %v", err)
	}
	for _, text := range []string{err.Error(), cause.Msg} {
		if strings.Contains(text, secret) || strings.Contains(text, "owner@example.test") {
			t.Fatalf("error exposed upstream data: %q", text)
		}
	}
	if !strings.Contains(cause.Msg, redactedDiagnostic) || !strings.HasPrefix(cause.Msg, "ai_studio: provider billing exhausted — key ") {
		t.Fatalf("diagnostic = %q", cause.Msg)
	}
	if len(fake.take()) != 1 {
		t.Fatal("the request was not sent once")
	}

	later := newGoogleTest(t, GoogleConfig{
		Deployment: GoogleAIStudio, BaseURL: "https://generativelanguage.example.test/v1beta",
		Client: &http.Client{Transport: retryAfterTransport{seconds: "7"}},
	})
	_, err = later.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, googleKey("k")))
	if classification := core.ClassifyError(err); classification.RetryAfter != 7*time.Second || !classification.Retryable {
		t.Fatalf("classification = %+v, want Retry-After kept", classification)
	}
}

// retryAfterTransport answers 503 with a Retry-After.
type retryAfterTransport struct{ seconds string }

// The Gemini API asks for a wait in a RetryInfo detail of its error envelope
// rather than a Retry-After header. A header still wins, and a delay that is
// not a positive duration asks for nothing.
func TestGoogleRefusalsKeepTheRetryInfoDelay(t *testing.T) {
	t.Parallel()
	envelope := func(delay string) string {
		return `{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"Quota exceeded.","details":[` +
			`{"@type":"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaMetric":"m"}]},` +
			`{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"` + delay + `"}]}}`
	}
	for _, check := range []struct {
		name, body, header string
		want               time.Duration
	}{
		{name: "whole seconds", body: envelope("37s"), want: 37 * time.Second},
		{name: "fractional seconds", body: envelope("1.5s"), want: 1500 * time.Millisecond},
		{name: "the header wins", body: envelope("37s"), header: "4", want: 4 * time.Second},
		{name: "unreadable delay", body: envelope("soon"), want: 0},
		{name: "negative delay", body: envelope("-3s"), want: 0},
		{name: "no detail", body: `{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"Quota exceeded.","details":[]}}`, want: 0},
	} {
		t.Run(check.name, func(t *testing.T) {
			t.Parallel()
			_, base := newGoogleFake(t, googleAnswer(http.StatusTooManyRequests, check.body))
			client := http.DefaultClient
			if check.header != "" {
				client = &http.Client{Transport: headerTransport{next: http.DefaultTransport, name: "Retry-After", value: check.header}}
			}
			provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base, Client: client})
			_, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, googleKey("k")))
			var cause *InvocationError
			if !errors.As(err, &cause) || cause.RetryAfter != check.want {
				t.Fatalf("error = %v (cause %+v), want a %v wait", err, cause, check.want)
			}
			if classification := core.ClassifyError(err); classification.RetryAfter != check.want || !classification.Retryable {
				t.Fatalf("classification = %+v", classification)
			}
		})
	}
}

// headerTransport adds a header to every answer next returns.
type headerTransport struct {
	next        http.RoundTripper
	name, value string
}

func (h headerTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	response, err := h.next.RoundTrip(request)
	if err == nil {
		response.Header.Set(h.name, h.value)
	}
	return response, err
}

// googleUnreachable fails every request as a network would.
type googleUnreachable struct{}

var errGoogleUnreachable = errors.New("connection reset by peer")

func (googleUnreachable) RoundTrip(*http.Request) (*http.Response, error) {
	return nil, errGoogleUnreachable
}

func (r retryAfterTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusServiceUnavailable, Header: http.Header{"Retry-After": {r.seconds}},
		Body: http.NoBody, Request: request,
	}, nil
}

// An answer that is not a nonempty JSON object is unusable, as in the
// gateway: another target may serve the request, and it counts against the
// provider, but repeating it would get the same answer.
func TestGoogleRefusesUnusableAnswers(t *testing.T) {
	t.Parallel()
	for _, answer := range []string{`not json`, `{}`, `null`, `[1]`} {
		_, base := newGoogleFake(t, googleAnswer(http.StatusOK, answer))
		provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
		_, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, googleKey("k")))
		var failure *core.ProviderError
		if !errors.As(err, &failure) || failure.Class != core.ProviderErrorUpstream || err.Error() != "ai_studio: invalid JSON in upstream response" ||
			core.ClassifyError(err) != (core.ProviderErrorClassification{FailoverEligible: true, CircuitFailure: true}) {
			t.Errorf("answer %s: %v %+v", answer, err, core.ClassifyError(err))
		}
	}
}

// A transport failure may repeat, unless the caller gave up.
func TestGoogleTransportFailures(t *testing.T) {
	t.Parallel()
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
	unreachable := newGoogleTest(t, GoogleConfig{
		Deployment: GoogleVertexAI, Project: "p", Client: &http.Client{Transport: googleUnreachable{}},
	})
	_, err := unreachable.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, googleKey("k")))
	var failure *core.ProviderError
	if !errors.As(err, &failure) || failure.Class != core.ProviderErrorTransport || err.Error() != "vertex_ai could not be reached" ||
		!errors.Is(err, errGoogleUnreachable) ||
		core.ClassifyError(err) != (core.ProviderErrorClassification{Retryable: true, FailoverEligible: true, CircuitFailure: true}) {
		t.Fatalf("error = %v %+v", err, core.ClassifyError(err))
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleVertexAI, BaseURL: base + "/v1", Project: "p"})
	_, err = provider.Invoke(ctx, googleChatRequest("gemini-3.5-flash", `{"messages":[]}`, googleKey("k")))
	if err == nil || core.ClassifyError(err) != (core.ProviderErrorClassification{}) {
		t.Fatalf("a caller who gave up: %v %+v, want nothing permitted", err, core.ClassifyError(err))
	}
	if calls := fake.take(); len(calls) != 0 {
		t.Fatalf("upstream = %+v", calls)
	}
}
