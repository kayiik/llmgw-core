package providers

import (
	"context"
	"encoding/json"
	"maps"
	"net/http"
	"slices"
	"strings"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/internal/observedstream"
	"github.com/kayiik/llmgw-core/internal/sourceusage"
)

func preparedTranslateAssessment(a translate.Assessment) core.CallAssessment {
	result := core.CallAssessment{Feasibility: core.SupportUnknown, UnknownPaths: slices.Clone(a.UnknownPaths), Losses: slices.Clone(a.Report.Losses)}
	switch a.Feasibility {
	case translate.FeasibilitySupported:
		result.Feasibility = core.SupportSupported
	case translate.FeasibilityUnsupported:
		result.Feasibility = core.SupportUnsupported
	}
	for _, feature := range a.Evaluated {
		result.Evaluated = append(result.Evaluated, string(feature))
	}
	return result
}

func preparedFidelity(r core.CallRequirements) translate.FidelityRequirements {
	return translate.FidelityRequirements{Content: r.Content, Tools: r.Tools, Schema: r.Schema, Billing: r.Billing, Replay: r.Replay}
}

func (p *OpenAICompatible) PrepareCall(_ context.Context, request core.Request, options core.CallPreparationOptions) (core.PreparedCall, error) {
	if request.Surface != core.ModelSurfaceChatCompletions && request.Surface != core.ModelSurfaceResponses {
		return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	if p.ovhNVR(request.Model) {
		return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	payload, err := preparedJSON(request)
	if err != nil {
		return nil, err
	}
	// Capture the catalogue callback once. No execution consults Models again.
	selected := *p
	row, known := p.row(request.Model)
	selected.models = func(string) (core.ModelInfo, bool) { return row, known }
	if request.Surface == core.ModelSurfaceResponses && !selected.responsesNative(request.Model) {
		return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	stream := options.Mode == core.CallModeStream
	buffered := stream && known && core.PreparedCapabilityDenialsUsable(row.Capabilities, p.now()) && row.Capabilities.Streaming == core.SupportUnsupported
	api := core.UpstreamAPIChatCompletions
	body := map[string]any(nil)
	var mapped openAIChat
	d := preparedDescription(request, options, api, core.SupportSupported, payload)
	d.Deployment = "openai_compatible"
	if p.label == "Bedrock" {
		d.Deployment = "bedrock_compatible"
	}
	d.APIVersion = "configured"
	path := p.chatPath()
	vision := false
	if request.Surface == core.ModelSurfaceResponses {
		api = core.UpstreamAPIResponses
		path = "/responses"
		body = openAIResponsesBody(openAICompatibleCall{model: request.Model, payload: payload}, stream)
		vision = openAIResponsesImages(body["input"])
	} else {
		if !options.AllowAdaptation {
			selected.adapt = false
			payload["force_api_support"] = false
		}
		mapped, err = selected.chat(openAICompatibleCall{model: request.Model, payload: payload}, stream)
		if err != nil {
			return nil, err
		}
		body = mapped.body
		mappingLosses := slices.Clone(mapped.losses)
		if stream && mapped.responses && preparedUsageOptionsSupported(payload["stream_options"]) {
			mappingLosses = slices.DeleteFunc(mappingLosses, func(loss core.Loss) bool { return loss.Path == "stream_options" })
		}
		d.Assessment = preparedAssessment(mappingLosses)
		vision = openAIChatImages(mapped.messages)
		if mapped.responses {
			api = core.UpstreamAPIResponses
			path = "/responses"
			vision = openAIResponsesImages(body["input"])
			source := maps.Clone(payload)
			delete(source, "force_api_support")
			delete(source, "stream_options")
			for key := range source {
				if strings.HasPrefix(key, "_") {
					delete(source, key)
				}
			}
			a := translate.AssessConversion(translate.ConversionChatToResponses, translate.PhaseRequest, source, preparedFidelity(d.Requirements))
			d.Assessment = preparedTranslateAssessment(a)
			d.Assessment.Losses = translate.NewReport(slices.Concat(d.Assessment.Losses, preparedAssessment(mappingLosses).Losses)...).Losses
			d.WirePreserved = core.SupportUnsupported
			if stream {
				if buffered {
					d.UpstreamTransport = core.CallTransportJSON
					d.OutputTransport = core.CallTransportBufferedSSE
				} else {
					body["stream"] = true
					d.UpstreamTransport = core.CallTransportSSE
					d.OutputTransport = core.CallTransportSSE
				}
			}
		} else if len(mapped.losses) > 0 {
			for _, loss := range mapped.losses {
				if loss.Class == translate.LossRenamed {
					preparedEdge(&d, "chat-reasoning-output-limit", core.UpstreamAPIChatCompletions)
					d.WirePreserved = core.SupportUnsupported
					break
				}
			}
		}
	}
	d.UpstreamAPI = api
	if mapped.responses {
		preparedEdge(&d, string(translate.ConversionChatToResponses), core.UpstreamAPIChatCompletions)
	}
	if known && core.PreparedCapabilityDenialsUsable(row.Capabilities, p.now()) {
		surface := request.Surface
		if mapped.responses {
			surface = core.ModelSurfaceResponses
		}
		support := row.Capabilities.SurfaceCompatibility(surface)
		if support == core.SupportUnsupported || row.Capabilities.OperationCompatibility(core.ModelOperationChat) == core.SupportUnsupported {
			d.Assessment.Feasibility = core.SupportUnsupported
		}
	}
	encoded, err := openAIEncode(body)
	if err != nil {
		return nil, err
	}
	requestLosses := slices.Clone(d.Assessment.Losses)
	// All callbacks below reuse the frozen bytes/path. In particular neither
	// completeOverResponses nor openAIStripRejected is reachable here.
	send := func(ctx context.Context, c *core.Credential, accept string) (*http.Response, error) {
		access, err := p.access(c)
		if err != nil {
			return nil, err
		}
		return p.send(ctx, path, p.header(access, accept, vision), encoded)
	}
	read := func(ctx context.Context, c *core.Credential) ([]byte, error) {
		response, err := send(ctx, c, core.ContentTypeJSON)
		if err != nil {
			return nil, err
		}
		raw, err := p.read(ctx, response)
		if err != nil {
			return raw, err
		}
		if err := p.preparedRefusal(api, request.Model, response, raw); err != nil {
			return raw, err
		}
		return raw, nil
	}
	return core.NewPreparedCall(d, func(ctx context.Context, c *core.Credential) (core.Response, error) {
		dialect := translate.UsageDialectChat
		if api == core.UpstreamAPIResponses {
			dialect = translate.UsageDialectResponses
		}
		record := core.BeginSourceUsage(ctx, api, dialect)
		defer sourceusage.Finish(record, ctx, translate.CompletionFailed)
		raw, err := read(ctx, c)
		sourceusage.Response(record, raw)
		if err != nil {
			return core.Response{Losses: requestLosses}, err
		}
		if api == core.UpstreamAPIResponses {
			answer, err := p.responsesAnswer(raw)
			if err != nil {
				return core.Response{Losses: requestLosses}, err
			}
			quality := translate.CompletionComplete
			if answer["status"] == "incomplete" {
				quality = translate.CompletionTruncated
			}
			record.Finish(quality, false)
			if mapped.responses {
				assessment := translate.AssessConversion(translate.ConversionChatToResponses, translate.PhaseResponse, answer, preparedFidelity(d.Requirements))
				converted := translate.ResponsesToChatWithReport(request.Model, answer)
				losses := translate.NewReport(slices.Concat(requestLosses, assessment.Report.Losses, converted.Report.Losses)...).Losses
				if err := (core.LossPolicy{}).Check(assessment.Report.Losses); err != nil {
					return core.Response{Losses: losses}, err
				}
				if assessment.Feasibility != translate.FeasibilitySupported {
					return core.Response{Losses: losses}, &core.CallPlanError{Reason: "response_semantics_unassessed", Losses: losses, UnknownPaths: slices.Clone(assessment.UnknownPaths)}
				}
				openAIReasoningAsContent(converted.Value)
				encoded, err := json.Marshal(converted.Value)
				return core.Response{Body: encoded, ContentType: core.ContentTypeJSON, Losses: losses}, err
			}
		} else if err := p.checkChat(raw); err != nil {
			return core.Response{Losses: requestLosses}, err
		}
		record.Finish(translate.CompletionComplete, false)
		return core.Response{Body: raw, ContentType: core.ContentTypeJSON, Losses: requestLosses}, nil
	}, func(ctx context.Context, c *core.Credential) (core.StreamIter, error) {
		dialect := translate.UsageDialectChat
		if api == core.UpstreamAPIResponses {
			dialect = translate.UsageDialectResponses
		}
		record := core.BeginSourceUsage(ctx, api, dialect)
		if mapped.responses && !buffered {
			response, err := send(ctx, c, core.ContentTypeEventStream)
			if err != nil {
				sourceusage.Finish(record, ctx, translate.CompletionFailed)
				return nil, err
			}
			if response.StatusCode >= 400 {
				raw, _ := p.read(ctx, response)
				sourceusage.Response(record, raw)
				sourceusage.Finish(record, ctx, translate.CompletionFailed)
				return nil, p.preparedRefusal(api, request.Model, response, raw)
			}
			upstream := sourceusage.Stream(ctx, newSSEFrameStream(ctx, p.label, response.Body), api, record)
			return observedstream.New(ctx, upstream, request.Model, translate.ConversionChatToResponses, preparedFidelity(d.Requirements), includesPreparedUsage(payload), requestLosses, payload), nil
		}
		if mapped.responses {
			defer sourceusage.Finish(record, ctx, translate.CompletionFailed)
			raw, err := read(ctx, c)
			sourceusage.Response(record, raw)
			if err != nil {
				return nil, err
			}
			answer, err := p.responsesAnswer(raw)
			if err != nil {
				return nil, err
			}
			quality := translate.CompletionComplete
			if answer["status"] == "incomplete" {
				quality = translate.CompletionTruncated
			}
			record.Finish(quality, false)
			assessment := translate.AssessConversion(translate.ConversionChatToResponses, translate.PhaseResponse, answer, preparedFidelity(d.Requirements))
			if err := (core.LossPolicy{}).Check(assessment.Report.Losses); err != nil {
				return nil, err
			}
			if assessment.Feasibility != translate.FeasibilitySupported {
				return nil, &core.CallPlanError{Reason: "response_semantics_unassessed", Losses: assessment.Report.Losses, UnknownPaths: slices.Clone(assessment.UnknownPaths)}
			}
			converted := translate.ResponsesToChatChunksWithReport(request.Model, answer)
			if err := (core.LossPolicy{}).Check(converted.Report.Losses); err != nil {
				return nil, err
			}
			frames := make([][]byte, 0, len(converted.Value)+1)
			for _, chunk := range converted.Value {
				frames = append(frames, []byte("data: "+chunk+"\n\n"))
			}
			frames = append(frames, []byte("data: [DONE]\n\n"))
			return &openAIStream{StreamIter: &openAIFrames{frames: frames}, losses: translate.NewReport(slices.Concat(requestLosses, assessment.Report.Losses, converted.Report.Losses)...).Losses}, nil
		}
		response, err := send(ctx, c, core.ContentTypeEventStream)
		if err != nil {
			sourceusage.Finish(record, ctx, translate.CompletionFailed)
			return nil, err
		}
		if response.StatusCode >= 400 {
			raw, _ := p.read(ctx, response)
			sourceusage.Response(record, raw)
			sourceusage.Finish(record, ctx, translate.CompletionFailed)
			return nil, p.preparedRefusal(api, request.Model, response, raw)
		}
		return &openAIStream{StreamIter: sourceusage.Stream(ctx, newSSEFrameStream(ctx, p.label, response.Body), api, record), losses: requestLosses}, nil
	}), nil
}

func includesPreparedUsage(payload map[string]any) bool {
	options, _ := payload["stream_options"].(map[string]any)
	included, _ := options["include_usage"].(bool)
	return included
}

func preparedUsageOptionsSupported(value any) bool {
	if value == nil {
		return true
	}
	options, ok := value.(map[string]any)
	if !ok {
		return false
	}
	for key, value := range options {
		if key != "include_usage" {
			return false
		}
		if _, ok := value.(bool); !ok && value != nil {
			return false
		}
	}
	return true
}
