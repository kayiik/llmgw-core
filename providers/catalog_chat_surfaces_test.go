package providers

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"strings"
	"testing"

	core "github.com/kayiik/llmgw-core"
)

func TestAzureChatCatalogDoesNotClaimMessagesTranslation(t *testing.T) {
	t.Parallel()
	backend, server := newAzureBackend(t, nil)
	provider := newTestAzure(t, server)
	credential := azureKey("fixture-key")
	models, err := provider.ListModels(context.Background(), credential)
	if err != nil || len(models) != 1 {
		t.Fatalf("catalog = %+v, %v", models, err)
	}
	backend.take()
	request := azureChatRequest(`{"messages":[{"role":"user","content":"hi"}]}`, credential)
	assertChatCatalogWithoutMessages(t, provider, models[0], request, func() int { return len(backend.take()) })
	response, err := provider.Invoke(context.Background(), request)
	if err != nil || string(response.Body) != azureFixtureCompletion {
		t.Fatalf("native Chat = %s, %v", response.Body, err)
	}
	if calls := backend.take(); len(calls) != 1 || calls[0].path != "/openai/v1/chat/completions" {
		t.Fatalf("native Chat calls = %+v", calls)
	}
}

func TestGoogleChatCatalogDoesNotClaimMessagesTranslation(t *testing.T) {
	t.Parallel()
	for _, deployment := range []GoogleDeployment{GoogleAIStudio, GoogleVertexAI} {
		t.Run(string(deployment), func(t *testing.T) {
			t.Parallel()
			fake, base := newGoogleFake(t, func(r *http.Request) (int, string) {
				if r.Method == http.MethodPost {
					return http.StatusOK, googleHelloAnswer
				}
				if deployment == GoogleVertexAI {
					return http.StatusOK, `{"publisherModels":[{"name":"publishers/google/models/gemini-chat","supportedActions":{"openGenerationAiStudio":{}}}]}`
				}
				return http.StatusOK, `{"models":[{"name":"models/gemini-chat","supportedGenerationMethods":["generateContent"]}]}`
			})
			provider := newGoogleTest(t, GoogleConfig{Deployment: deployment, BaseURL: base, Project: "fixture-project"})
			credential := googleKey("fixture-key")
			if deployment == GoogleVertexAI {
				credential = googleBearer("fixture-token")
			}
			models, err := provider.ListModels(context.Background(), credential)
			if err != nil || len(models) != 1 {
				t.Fatalf("catalog = %+v, %v", models, err)
			}
			fake.take()
			request := googleChatRequest(models[0].ID, `{"messages":[{"role":"user","content":"hi"}]}`, credential)
			assertChatCatalogWithoutMessages(t, provider, models[0], request, func() int { return len(fake.take()) })
			response, err := provider.Invoke(context.Background(), request)
			if err != nil || response.ContentType != core.ContentTypeJSON || len(response.Body) == 0 {
				t.Fatalf("native Chat = %s, %v", response.Body, err)
			}
			if calls := fake.take(); len(calls) != 1 || calls[0].method != http.MethodPost ||
				!strings.HasSuffix(calls[0].path, "/models/gemini-chat:generateContent") {
				t.Fatalf("native Chat calls = %+v", calls)
			}
		})
	}
}

// A catalog must not promote a translation adapter's surface to a direct
// provider claim. Omitting it leaves capability evidence unknown, not denied.
func assertChatCatalogWithoutMessages(t *testing.T, provider core.Provider, model core.ModelInfo, request core.Request, takeCalls func() int) {
	t.Helper()
	if !slices.Equal(model.SupportedAPIs, []string{"/v1/chat/completions"}) {
		t.Errorf("catalog endpoints = %v, want only direct Chat", model.SupportedAPIs)
	}
	if capabilities := model.Capabilities; capabilities == nil ||
		capabilities.Surfaces.ChatCompletions != core.SupportSupported ||
		capabilities.Surfaces.Messages != core.SupportUnknown ||
		capabilities.Tools != core.SupportUnknown || capabilities.Reasoning != core.SupportUnknown {
		t.Errorf("catalog capabilities = %+v, want Chat supported and undeclared capabilities unknown", capabilities)
	}
	if !core.ServesNatively(provider, model.ID, core.ModelSurfaceChatCompletions) ||
		core.ServesNatively(provider, model.ID, core.ModelSurfaceMessages) {
		t.Fatal("provider's direct surfaces differ from Chat-only catalog evidence")
	}
	request.Surface = core.ModelSurfaceMessages
	for _, stream := range []bool{false, true} {
		var err error
		if stream {
			var iter core.StreamIter
			iter, err = provider.Stream(context.Background(), request)
			if iter != nil {
				_ = iter.Close()
			}
		} else {
			_, err = provider.Invoke(context.Background(), request)
		}
		var surfaceError *core.SurfaceError
		if !errors.As(err, &surfaceError) || surfaceError.Surface != core.ModelSurfaceMessages || surfaceError.Model != model.ID {
			t.Errorf("Messages stream=%v: %v, want direct surface refusal", stream, err)
		}
	}
	if calls := takeCalls(); calls != 0 {
		t.Errorf("Messages contacted upstream %d times without a translation adapter", calls)
	}
}
