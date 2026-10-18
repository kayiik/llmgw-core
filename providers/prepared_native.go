package providers

import (
	"context"
	"encoding/json"
	translate "github.com/kayiik/llm-translate"
	"net/http"

	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/internal/sourceusage"
)

func (p *Anthropic) PrepareCall(ctx context.Context, request core.Request, options core.CallPreparationOptions) (core.PreparedCall, error) {
	call, err := p.prepare(ctx, request, options.Mode == core.CallModeStream)
	if err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(call.payload)
	if err != nil {
		return nil, err
	}
	d := preparedDescription(request, options, core.UpstreamAPIMessages, core.SupportSupported, call.payload)
	d.APIVersion, d.Deployment = anthropicAPIVersion, "anthropic_messages"
	protocol := http.Header{}
	protocol.Set("anthropic-version", anthropicAPIVersion)
	anthropicProtocolHeaders(protocol, call.header)
	d.APIVersion = protocol.Get("anthropic-version")
	return core.NewPreparedCall(d, func(ctx context.Context, c *core.Credential) (core.Response, error) {
		selected := call
		key, err := anthropicCredential(c)
		if err != nil {
			return core.Response{}, err
		}
		selected.apiKey = key
		record := core.BeginSourceUsage(ctx, core.UpstreamAPIMessages, translate.UsageDialectMessages)
		defer sourceusage.Finish(record, ctx, translate.CompletionFailed)
		observe := func(raw []byte) { sourceusage.Response(record, raw) }
		response, err := p.postEncodedObserved(ctx, selected, "/v1/messages", anthropicLabel, encoded, observe)
		if err != nil {
			return core.Response{}, err
		}
		defer response.Body.Close()
		body, err := readAnthropicMessageObserved(ctx, response, observe)
		if err == nil {
			record.Finish(translate.CompletionComplete, false)
		}
		return core.Response{Body: body, ContentType: core.ContentTypeJSON}, err
	}, func(ctx context.Context, c *core.Credential) (core.StreamIter, error) {
		selected := call
		key, err := anthropicCredential(c)
		if err != nil {
			return nil, err
		}
		selected.apiKey = key
		record := core.BeginSourceUsage(ctx, core.UpstreamAPIMessages, translate.UsageDialectMessages)
		response, err := p.postEncodedObserved(ctx, selected, "/v1/messages", anthropicLabel, encoded, func(raw []byte) { sourceusage.Response(record, raw) })
		if err != nil {
			sourceusage.Finish(record, ctx, translate.CompletionFailed)
			return nil, err
		}
		return sourceusage.Stream(ctx, &anthropicStream{ctx: ctx, body: response.Body, reader: newSSERecordReader(response.Body)}, core.UpstreamAPIMessages, record), nil
	}), nil
}

func (p *AzureOpenAI) PrepareCall(_ context.Context, request core.Request, options core.CallPreparationOptions) (core.PreparedCall, error) {
	call, err := p.prepare(request, options.Mode == core.CallModeStream)
	if err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(call.body)
	if err != nil {
		return nil, err
	}
	d := preparedDescription(request, options, core.UpstreamAPIChatCompletions, core.SupportUnsupported, call.body)
	d.APIVersion, d.Deployment = "v1", "azure_openai"
	d.Assessment = preparedAssessment(call.losses)
	return core.NewPreparedCall(d, func(ctx context.Context, c *core.Credential) (core.Response, error) {
		selected := call
		key, err := azureCredential(c)
		if err != nil {
			return core.Response{}, err
		}
		selected.key = key
		record := core.BeginSourceUsage(ctx, core.UpstreamAPIChatCompletions, translate.UsageDialectChat)
		defer sourceusage.Finish(record, ctx, translate.CompletionFailed)
		response, err := p.postEncodedObserved(ctx, selected, encoded, func(raw []byte) { sourceusage.Response(record, raw) })
		if err != nil {
			return core.Response{Losses: d.Assessment.Losses}, err
		}
		defer response.Body.Close()
		raw, err := readInvocationResponseBody(ctx, response, azureLabel)
		sourceusage.Response(record, raw)
		if err == nil {
			err = azureCheckCompletion(raw)
		}
		if err == nil {
			record.Finish(translate.CompletionComplete, false)
		}
		return core.Response{Body: raw, ContentType: core.ContentTypeJSON, Losses: d.Assessment.Losses}, err
	}, func(ctx context.Context, c *core.Credential) (core.StreamIter, error) {
		selected := call
		key, err := azureCredential(c)
		if err != nil {
			return nil, err
		}
		selected.key = key
		record := core.BeginSourceUsage(ctx, core.UpstreamAPIChatCompletions, translate.UsageDialectChat)
		response, err := p.postEncodedObserved(ctx, selected, encoded, func(raw []byte) { sourceusage.Response(record, raw) })
		if err != nil {
			sourceusage.Finish(record, ctx, translate.CompletionFailed)
			return nil, err
		}
		return &openAIStream{StreamIter: sourceusage.Stream(ctx, newSSEFrameStream(ctx, azureLabel, response.Body), core.UpstreamAPIChatCompletions, record), losses: d.Assessment.Losses}, nil
	}), nil
}

// Check native Responses refusals without inventing a post-admission fallback.
func (p *OpenAICompatible) preparedRefusal(api core.UpstreamAPI, model string, response *http.Response, raw []byte) error {
	if api == core.UpstreamAPIResponses && (response.StatusCode == 404 || response.StatusCode == 405) {
		return &core.ProviderError{Message: "the selected Responses endpoint is unavailable", Class: core.ProviderErrorUnsupported,
			Classification: core.ProviderErrorClassification{StatusCode: response.StatusCode, FailoverEligible: true}, Cause: &core.SurfaceError{Surface: core.ModelSurfaceResponses, Model: model}}
	}
	if response.StatusCode >= 400 {
		return p.refused(response, raw)
	}
	return nil
}
