package translation

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"maps"
	"mime"
	"slices"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/internal/observedstream"
)

func preparedPayload(request core.Request) (map[string]any, error) {
	media, _, err := mime.ParseMediaType(request.ContentType)
	if err != nil || media != core.ContentTypeJSON {
		return nil, &core.ProviderError{Message: "a prepared conversion requires JSON", Class: core.ProviderErrorInvalidRequest}
	}
	decoder := json.NewDecoder(bytes.NewReader(request.Body))
	decoder.UseNumber()
	var payload map[string]any
	if err := decoder.Decode(&payload); err != nil || payload == nil {
		return nil, &core.ProviderError{Message: "a prepared conversion requires a JSON object", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, &core.ProviderError{Message: "a prepared conversion requires one JSON object", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	return payload, nil
}

func conversionID(r route) translate.ConversionID {
	switch {
	case r.from == core.ModelSurfaceMessages && r.target == core.ModelSurfaceChatCompletions:
		return translate.ConversionMessagesToChat
	case r.from == core.ModelSurfaceChatCompletions && r.target == core.ModelSurfaceMessages:
		return translate.ConversionChatToMessages
	case r.from == core.ModelSurfaceChatCompletions && r.target == core.ModelSurfaceResponses:
		return translate.ConversionChatToResponses
	case r.from == core.ModelSurfaceResponses && r.target == core.ModelSurfaceChatCompletions:
		return translate.ConversionResponsesToChat
	}
	return ""
}

func callAssessment(a translate.Assessment) core.CallAssessment {
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

func mergeAssessments(outer, inner core.CallAssessment) core.CallAssessment {
	result := core.CallAssessment{Feasibility: core.SupportSupported,
		UnknownPaths: slices.Concat(outer.UnknownPaths, inner.UnknownPaths), Losses: translate.NewReport(slices.Concat(outer.Losses, inner.Losses)...).Losses}
	if outer.Feasibility == core.SupportUnsupported || inner.Feasibility == core.SupportUnsupported {
		result.Feasibility = core.SupportUnsupported
	} else if outer.Feasibility != core.SupportSupported || inner.Feasibility != core.SupportSupported {
		result.Feasibility = core.SupportUnknown
	}
	for _, feature := range outer.Evaluated {
		if slices.Contains(inner.Evaluated, feature) {
			result.Evaluated = append(result.Evaluated, feature)
		}
	}
	return result
}

// These explicit Chat defaults request the ordinary single text generation
// already provided by Messages. Canonicalize only this conversion's input;
// retain the original payload for source semantics and other adapter legs.
func preparedMessagesNoops(payload map[string]any) map[string]any {
	normalized := maps.Clone(payload)
	if n, ok := normalized["n"].(json.Number); ok {
		if value, err := n.Int64(); err == nil && value == 1 {
			delete(normalized, "n")
		}
	}
	if normalized["logprobs"] == false {
		delete(normalized, "logprobs")
	}
	if format, ok := normalized["response_format"].(map[string]any); ok && len(format) == 1 && format["type"] == "text" {
		delete(normalized, "response_format")
	}
	if normalized["parallel_tool_calls"] == true {
		delete(normalized, "parallel_tool_calls")
	}
	return normalized
}

// PrepareCall composes existing finite adapter edges with the inner provider's
// actual preparation. The baseline Responses-to-Chat-to-Messages path is named
// explicitly; preparation does not search an arbitrary conversion graph.
func (a Adapter) PrepareCall(ctx context.Context, request core.Request, options core.CallPreparationOptions) (core.PreparedCall, error) {
	if core.ServesNatively(a.Provider, request.Model, request.Surface) {
		return core.PrepareCall(ctx, a.Provider, request, options)
	}
	stream := options.Mode == core.CallModeStream
	r, ok := a.route(request.Model, request.Surface, stream)
	innerProvider := a.Provider
	baselineMessages := false
	// These newly observed finite edges are prepared-only. Legacy Adapter
	// Stream/ServesStream keep their old availability and lifecycle contract.
	if !ok && stream {
		if candidate, available := a.route(request.Model, request.Surface, false); available && (conversionID(candidate) == translate.ConversionChatToResponses || conversionID(candidate) == translate.ConversionResponsesToChat) {
			r, ok = candidate, true
		}
	}
	if !ok && request.Surface == core.ModelSurfaceResponses && core.ServesNatively(a.Provider, request.Model, core.ModelSurfaceMessages) {
		// Restore only the baseline Responses facade over native Messages.
		// Each existing leg performs its own request/response assessment, and
		// the resulting description retains both ordered conversion IDs.
		r, ok = routeFor(core.ModelSurfaceResponses, []core.ModelSurface{core.ModelSurfaceChatCompletions}, false)
		innerProvider = Adapter{Provider: a.Provider, Policy: a.Policy}
		baselineMessages = true
	}
	if !ok {
		return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	payload, err := preparedPayload(request)
	if err != nil {
		return nil, err
	}
	id := conversionID(r)
	conversionPayload := payload
	if id == translate.ConversionChatToMessages {
		conversionPayload = preparedMessagesNoops(payload)
	}
	explicit := options.Requirements
	fidelity := translate.FidelityRequirements{Content: explicit.Content, Tools: explicit.Tools, Schema: explicit.Schema, Billing: explicit.Billing, Replay: explicit.Replay}
	assessment := translate.AssessConversion(id, translate.PhaseRequest, conversionPayload, fidelity)
	translated, requestLosses, err := r.request(request.Model, conversionPayload, stream)
	if err != nil {
		return nil, &core.ProviderError{Message: "the prepared request cannot be translated", Class: core.ProviderErrorInvalidRequest, Cause: err}
	}
	if baselineMessages {
		// ResponsesRequestToChat retains this private limit hint for its
		// caller to bind to the selected upstream protocol. This named path
		// uses Chat max_tokens, which the existing Messages mapper preserves.
		if limit, present := translated["_max_output_tokens"]; present {
			translated["max_tokens"] = limit
			delete(translated, "_max_output_tokens")
		}
	}
	encoded, err := json.Marshal(translated)
	if err != nil {
		return nil, err
	}
	upstream := request
	upstream.Surface, upstream.Body, upstream.ContentType = r.target, encoded, core.ContentTypeJSON
	options.Requirements = core.CallRequirements{Content: assessment.Requirements.Content, Tools: assessment.Requirements.Tools, Schema: assessment.Requirements.Schema, Billing: assessment.Requirements.Billing, Replay: assessment.Requirements.Replay, Incremental: explicit.Incremental}
	innerOptions := options
	text, _ := payload["text"].(map[string]any)
	format, _ := text["format"].(map[string]any)
	plainTextEnvelope := baselineMessages && format["type"] == "text" && !explicit.Schema && !translate.InferFidelityRequirements(translated).Schema
	if plainTextEnvelope {
		// The outer leg assesses/echoes text.format=text. It imposes no
		// structured-generation control on the native Messages request.
		innerOptions.Requirements.Schema = false
	}
	inner, err := core.PrepareCall(ctx, innerProvider, upstream, innerOptions)
	if err != nil {
		return nil, err
	}
	d := inner.Description()
	outer := callAssessment(assessment)
	outer.Losses = translate.NewReport(slices.Concat(outer.Losses, requestLosses)...).Losses
	d.Ingress = request.Surface
	d.WirePreserved = core.SupportUnsupported
	d.Requirements = options.Requirements
	d.Assessment = mergeAssessments(outer, d.Assessment)
	if plainTextEnvelope && slices.Contains(outer.Evaluated, string(translate.FidelitySchema)) && !slices.Contains(d.Assessment.Evaluated, string(translate.FidelitySchema)) {
		d.Assessment.Evaluated = append(d.Assessment.Evaluated, string(translate.FidelitySchema))
	}
	d.Conversions = slices.Concat([]core.CallConversion{{ID: string(id), FromAPI: core.UpstreamAPI(r.from), ToAPI: core.UpstreamAPI(r.target), Assessment: outer}}, d.Conversions)
	fidelity = assessment.Requirements
	return core.NewPreparedCall(d, func(ctx context.Context, c *core.Credential) (core.Response, error) {
		if err := a.Policy.Check(d.Assessment.Losses); err != nil {
			return core.Response{Losses: d.Assessment.Losses}, err
		}
		response, err := inner.Invoke(ctx, c)
		losses := translate.NewReport(slices.Concat(d.Assessment.Losses, response.Losses)...).Losses
		if err != nil {
			return core.Response{Losses: losses}, err
		}
		source, err := preparedPayload(core.Request{Body: response.Body, ContentType: response.ContentType})
		if err != nil {
			return core.Response{Losses: losses}, err
		}
		responseAssessment := translate.AssessConversion(id, translate.PhaseResponse, source, fidelity)
		converted, responseLosses := r.response(request.Model, source)
		if id == translate.ConversionResponsesToChat {
			// Echo the original assessed Responses contract, not the narrower
			// mapped Chat request or generic response-envelope defaults.
			result := translate.ChatResponseToResponsesWithRequestAndReport(request.Model, source, payload)
			converted, responseLosses = result.Value, result.Report.Losses
		}
		losses = translate.NewReport(slices.Concat(losses, responseAssessment.Report.Losses, responseLosses)...).Losses
		if err := (core.LossPolicy{}).Check(responseAssessment.Report.Losses); err != nil {
			return core.Response{Losses: losses}, err
		}
		if err := a.Policy.Check(responseAssessment.Report.Losses); err != nil {
			return core.Response{Losses: losses}, err
		}
		if responseAssessment.Feasibility != translate.FeasibilitySupported {
			return core.Response{Losses: losses}, &core.CallPlanError{Reason: "response_semantics_unassessed", Losses: losses, UnknownPaths: slices.Clone(responseAssessment.UnknownPaths)}
		}
		body, err := json.Marshal(converted)
		return core.Response{Body: body, ContentType: core.ContentTypeJSON, Losses: losses}, err
	}, func(ctx context.Context, c *core.Credential) (core.StreamIter, error) {
		if err := a.Policy.Check(d.Assessment.Losses); err != nil {
			return nil, err
		}
		upstream, err := inner.Stream(ctx, c)
		if err != nil {
			return nil, err
		}
		losses := translate.NewReport(slices.Concat(d.Assessment.Losses, core.StreamLosses(upstream))...).Losses
		return observedstream.New(ctx, upstream, request.Model, id, fidelity, includesUsage(payload), losses, payload), nil
	}), nil
}
