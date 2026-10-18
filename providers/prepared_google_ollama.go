package providers

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/internal/callcontext"
	"github.com/kayiik/llmgw-core/internal/sourceusage"
)

func (p *Google) PrepareCall(_ context.Context, request core.Request, options core.CallPreparationOptions) (core.PreparedCall, error) {
	if request.Surface != core.ModelSurfaceChatCompletions {
		return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	payload, err := preparedJSON(request)
	if err != nil {
		return nil, err
	}
	stream := options.Mode == core.CallModeStream
	call, err := googleChatPayloadBody(payload, stream)
	if err != nil {
		return nil, err
	}
	access, err := p.access(request.Credential)
	if err != nil {
		return nil, err
	}
	action := "generateContent"
	if stream {
		action = "streamGenerateContent"
	}
	endpoint, err := p.modelURL(request.Model, access.project, action)
	if err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(call.payload)
	if err != nil {
		return nil, err
	}
	d := preparedDescription(request, options, core.UpstreamAPIGoogleGenerateContent, core.SupportUnsupported, payload)
	d.Deployment = string(p.deployment)
	d.APIVersion = "configured" // Endpoint already binds version/project/location.
	d.Assessment = assessGooglePrepared(payload, call.losses)
	preparedEdge(&d, "chat-to-google-generate-content", core.UpstreamAPIChatCompletions)
	authorization := func(ctx context.Context, c *core.Credential) (googleAuthorization, error) {
		current, err := p.access(c)
		if err != nil {
			return googleAuthorization{}, err
		}
		if current.project != access.project {
			return googleAuthorization{}, core.ErrPreparedCallStale
		}
		auth, err := p.authorization(current)
		if err != nil {
			return googleAuthorization{}, err
		}
		// Google may mint a token here. Fence after that wait too.
		if err := callcontext.Check(ctx); err != nil {
			return googleAuthorization{}, err
		}
		return auth, nil
	}
	return core.NewPreparedCall(d, func(ctx context.Context, c *core.Credential) (core.Response, error) {
		auth, err := authorization(ctx, c)
		if err != nil {
			return core.Response{Losses: d.Assessment.Losses}, err
		}
		record := core.BeginSourceUsage(ctx, core.UpstreamAPIGoogleGenerateContent, translate.UsageDialectChat)
		defer sourceusage.Finish(record, ctx, translate.CompletionFailed)
		usage := newPreparedNativeUsage(core.UpstreamAPIGoogleGenerateContent, record)
		decoded, err := p.postDecodedObserved(ctx, auth, endpoint, json.RawMessage(encoded), true, func(value map[string]any) {
			if native, set := value["usageMetadata"]; set {
				usage.merge(native)
			}
		})
		if err != nil {
			return core.Response{Losses: d.Assessment.Losses}, err
		}
		quality := translate.CompletionComplete
		if googleCandidateFinish(decoded) == "MAX_TOKENS" {
			quality = translate.CompletionTruncated
		}
		record.Finish(quality, false)
		assessment := assessGooglePreparedResponse(decoded)
		losses := slices.Concat(d.Assessment.Losses, assessment.Losses)
		if err := (core.LossPolicy{}).Check(assessment.Losses); err != nil {
			return core.Response{Losses: losses}, err
		}
		if assessment.Feasibility != core.SupportSupported {
			return core.Response{Losses: losses}, &core.CallPlanError{Reason: "google_response_semantics_unassessed", Losses: losses, UnknownPaths: assessment.UnknownPaths}
		}
		if err := googleEmptyReplyError(p.label(), decoded); err != nil {
			return core.Response{Losses: d.Assessment.Losses}, err
		}
		body, err := json.Marshal(googleToOpenAIChat(request.Model, decoded))
		return core.Response{Body: body, ContentType: core.ContentTypeJSON, Losses: losses}, err
	}, func(ctx context.Context, c *core.Credential) (core.StreamIter, error) {
		auth, err := authorization(ctx, c)
		if err != nil {
			return nil, err
		}
		record := core.BeginSourceUsage(ctx, core.UpstreamAPIGoogleGenerateContent, translate.UsageDialectChat)
		usage := newPreparedNativeUsage(core.UpstreamAPIGoogleGenerateContent, record)
		response, err := p.openObserved(ctx, auth, endpoint+"?alt=sse", json.RawMessage(encoded), func(value map[string]any) {
			if native, set := value["usageMetadata"]; set {
				usage.merge(native)
			}
		})
		if err != nil {
			sourceusage.Finish(record, ctx, translate.CompletionFailed)
			return nil, err
		}
		return &googleStream{ctx: ctx, provider: p, model: request.Model, body: response.Body, reader: newSSERecordReader(response.Body), includeUsage: call.includeUsage, losses: slices.Clone(d.Assessment.Losses), completionID: newChatCompletionID("google"), exactNumbers: true, responseAssessment: assessGooglePreparedResponse, sourceRecord: record, sourceUsage: usage}, nil
	}), nil
}

func (p *Ollama) PrepareCall(_ context.Context, request core.Request, options core.CallPreparationOptions) (core.PreparedCall, error) {
	if request.Surface != core.ModelSurfaceChatCompletions {
		return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	payload, err := preparedJSON(request)
	if err != nil {
		return nil, err
	}
	messages, err := googleChatMessages(payload["messages"])
	if err != nil {
		return nil, err
	}
	stream := options.Mode == core.CallModeStream
	call, err := ollamaPrepareBody(request.Model, stream, ollamaChatBody{Messages: messages, Temperature: payload["temperature"], MaxTokens: payload["max_tokens"], TopP: payload["top_p"], Tools: payload["tools"]}, payload)
	if err != nil {
		return nil, err
	}
	d := preparedDescription(request, options, core.UpstreamAPIOllamaChat, core.SupportUnsupported, payload)
	d.Deployment = "ollama_native"
	if stream {
		d.UpstreamTransport = core.CallTransportNDJSON
	}
	d.Assessment = assessOllamaPrepared(payload, call.losses)
	preparedEdge(&d, "chat-to-ollama-chat", core.UpstreamAPIChatCompletions)
	return core.NewPreparedCall(d, func(ctx context.Context, _ *core.Credential) (core.Response, error) {
		record := core.BeginSourceUsage(ctx, core.UpstreamAPIOllamaChat, translate.UsageDialectChat)
		defer sourceusage.Finish(record, ctx, translate.CompletionFailed)
		usage := newPreparedNativeUsage(core.UpstreamAPIOllamaChat, record)
		response, err := p.post(ctx, call.body)
		if err != nil {
			return core.Response{Losses: d.Assessment.Losses}, err
		}
		defer response.Body.Close()
		raw, err := readInvocationResponseBody(ctx, response, "Ollama")
		sourceDone := false
		if value, decodeErr := sourceusage.Decode(raw); decodeErr == nil {
			usage.merge(ollamaNativeUsage(value))
			sourceDone, _ = value["done"].(bool)
		}
		if err != nil {
			return core.Response{Losses: d.Assessment.Losses}, err
		}
		if response.StatusCode >= http.StatusBadRequest {
			return core.Response{Losses: d.Assessment.Losses}, httpStatusFailure("Ollama", response, raw, p.now())
		}
		body, err := ollamaCompletion(request.Model, raw)
		if err == nil {
			quality := translate.CompletionTruncated
			if sourceDone {
				quality = translate.CompletionComplete
			}
			record.Finish(quality, false)
		}
		return core.Response{Body: body, ContentType: core.ContentTypeJSON, Losses: d.Assessment.Losses}, err
	}, func(ctx context.Context, _ *core.Credential) (core.StreamIter, error) {
		record := core.BeginSourceUsage(ctx, core.UpstreamAPIOllamaChat, translate.UsageDialectChat)
		response, err := p.post(ctx, call.body)
		if err != nil {
			sourceusage.Finish(record, ctx, translate.CompletionFailed)
			return nil, err
		}
		if response.StatusCode >= http.StatusBadRequest {
			defer response.Body.Close()
			raw, _ := readInvocationResponseBody(ctx, response, "Ollama")
			if value, err := sourceusage.Decode(raw); err == nil {
				newPreparedNativeUsage(core.UpstreamAPIOllamaChat, record).merge(ollamaNativeUsage(value))
			}
			sourceusage.Finish(record, ctx, translate.CompletionFailed)
			return nil, httpStatusFailure("Ollama", response, raw, p.now())
		}
		stream := newOllamaStream(ctx, response.Body, request.Model, slices.Clone(d.Assessment.Losses))
		stream.sourceRecord = record
		stream.sourceUsage = newPreparedNativeUsage(core.UpstreamAPIOllamaChat, record)
		return stream, nil
	}), nil
}
