package providers

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"testing"

	core "github.com/kayiik/llmgw-core"
)

// refusalFixture is one refusal, in the shape its provider documents, and
// the class it must be given.
type refusalFixture struct {
	Name   string          `json:"name"`
	Entry  string          `json:"entry"`
	Status int             `json:"status"`
	Body   json.RawMessage `json:"body"`
	Class  string          `json:"class"`
}

// Each provider names the reasons a product routes on from its refusals:
// billing, a context overflow, and tools or media the model does not take,
// while any other refusal keeps the class its status gives. Billing fails
// over and counts against the instance; a refused shape fails over and
// counts against nothing; neither is worth repeating.
func TestProvidersNameTheReasonsOfTheirRefusals(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile("testdata/refusals.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures []refusalFixture
	if err := json.Unmarshal(raw, &fixtures); err != nil {
		t.Fatal(err)
	}
	for _, fixture := range fixtures {
		t.Run(fixture.Name, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", core.ContentTypeJSON)
				w.WriteHeader(fixture.Status)
				_, _ = w.Write(fixture.Body)
			}))
			t.Cleanup(server.Close)
			entry, ok := DefaultRegistry().ByID(fixture.Entry)
			if !ok {
				t.Fatalf("the registry holds no %s", fixture.Entry)
			}
			path := ""
			if entry.RuntimeType == "openai_compatible" {
				base, _ := url.Parse(entry.DefaultBaseURL)
				path = base.Path
			}
			provider, err := Build(entry, Instance{BaseURL: server.URL + path, Client: server.Client()})
			if err != nil {
				t.Fatal(err)
			}
			request := core.Request{Surface: core.ModelSurfaceChatCompletions, Model: "fixture-model", ContentType: core.ContentTypeJSON,
				Body: []byte(`{"messages":[{"role":"user","content":"hi"}]}`), Credential: &core.Credential{APIKey: "fixture-key", TokenType: core.TokenTypeAPIKey}}
			if entry.RuntimeType == "anthropic" {
				request.Surface, request.Body = core.ModelSurfaceMessages, []byte(`{"max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`)
			}
			_, err = provider.Invoke(context.Background(), request)
			var failure *core.ProviderError
			if !errors.As(err, &failure) || string(failure.Class) != fixture.Class {
				t.Fatalf("error = %#v, want class %s", err, fixture.Class)
			}
			classification := core.ClassifyError(err)
			if classification.StatusCode != fixture.Status {
				t.Fatalf("classification = %+v, want status %d", classification, fixture.Status)
			}
			switch core.ProviderErrorClass(fixture.Class) {
			case core.ProviderErrorBilling:
				if classification.Retryable || !classification.FailoverEligible || !classification.CircuitFailure {
					t.Fatalf("billing classification = %+v", classification)
				}
			case core.ProviderErrorContextOverflow, core.ProviderErrorToolsUnsupported, core.ProviderErrorMediaUnsupported:
				if classification.Retryable || !classification.FailoverEligible || classification.CircuitFailure {
					t.Fatalf("refused shape classification = %+v", classification)
				}
			default:
				if want := statusClassification(fixture.Status, 0); classification != want {
					t.Fatalf("classification = %+v, want the status's %+v", classification, want)
				}
			}
		})
	}
}

// refusalClass reads only a client error's body, and a 402 whatever it
// says.
func TestRefusalClassReadsClientErrorsOnly(t *testing.T) {
	t.Parallel()
	for _, testCase := range []struct {
		status int
		body   string
		class  core.ProviderErrorClass
	}{
		{http.StatusPaymentRequired, "", core.ProviderErrorBilling},
		{http.StatusPaymentRequired, `{"error":"payment required"}`, core.ProviderErrorBilling},
		{http.StatusRequestEntityTooLarge, `{"error":{"message":"Prompt contains 40000 tokens, too large for model with 32768 maximum context length"}}`, core.ProviderErrorContextOverflow},
		{http.StatusForbidden, `{"error":{"message":"This API method requires billing to be enabled."}}`, core.ProviderErrorBilling},
		{http.StatusBadGateway, `context length`, ""},
		{http.StatusOK, `context length`, ""},
		{http.StatusBadRequest, `{"error":"model does not support images"}`, core.ProviderErrorMediaUnsupported},
	} {
		if class := refusalClass(testCase.status, []byte(testCase.body)); class != testCase.class {
			t.Errorf("%d %s: class %q, want %q", testCase.status, testCase.body, class, testCase.class)
		}
	}
}
