package extension

import (
	"context"
	"slices"
	"strings"
	"time"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/internal/sourceusage"
	"github.com/kayiik/llmgw-core/translation"
)

var _ core.CallPreparer = (*Provider)(nil)

// PrepareCall selects only declared native surfaces from a local model lookup.
// The existing finite adapter supplies any explicitly permitted conversion;
// neither preparation nor execution discovers/reselects a daemon endpoint.
func (p *Provider) PrepareCall(ctx context.Context, request core.Request, options core.CallPreparationOptions) (core.PreparedCall, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := request.Validate(); err != nil {
		return nil, err
	}
	if options.Mode == "" {
		options.Mode = core.CallModeInvoke
	}
	if options.Mode != core.CallModeInvoke && options.Mode != core.CallModeStream {
		return nil, core.ErrPreparedCallMode
	}
	surfaces, capabilities, known := p.preparedSurfaces(request.Model)
	if !known || len(surfaces) == 0 {
		return core.NewPreparedCall(core.CallDescription{Ingress: request.Surface, Model: request.Model, Mode: options.Mode, UpstreamAPI: core.UpstreamAPIUnknown, Deployment: "extension/v1", Assessment: core.CallAssessment{Feasibility: core.SupportUnknown, UnknownPaths: []string{"model.native_surfaces"}}}, nil, nil), nil
	}
	selected := declaredProvider{owner: p, model: request.Model, surfaces: slices.Clone(surfaces), capabilities: capabilities}
	if slices.Contains(surfaces, request.Surface) {
		return selected.PrepareCall(ctx, request, options)
	}
	if !options.AllowAdaptation {
		return nil, &core.CallPlanError{Reason: "adaptation_required: extension model requires an explicitly allowed local protocol mapping"}
	}
	return (translation.Adapter{Provider: selected}).PrepareCall(ctx, request, options)
}

func (p *Provider) preparedSurfaces(model string) ([]core.ModelSurface, *core.ModelCapabilities, bool) {
	defaults := knownSurfaces(p.info.Surfaces)
	if p.modelLookup == nil && !p.requireModelDeclaration {
		return defaults, nil, len(defaults) > 0
	}
	if p.modelLookup == nil {
		return nil, nil, false
	}
	row, found := p.modelLookup(model)
	if !found || (row.ID != "" && row.ID != model) {
		return nil, nil, false
	}
	var capabilities *core.ModelCapabilities
	if row.Capabilities != nil && row.Capabilities.SchemaVersion == core.ModelCapabilitiesSchemaVersion {
		copy := *row.Capabilities
		capabilities = &copy
	} else if row.Capabilities == nil {
		capabilities = core.InferCapabilities(row, time.Time{}, time.Time{})
	}
	var surfaces []core.ModelSurface
	switch {
	case len(row.SupportedAPIs) > 0:
		// Endpoint declarations override provider defaults, including a
		// response-only row behind a legacy Chat-shaped facade. WS is not SSE.
		for _, path := range row.SupportedAPIs {
			for _, surface := range []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses, core.ModelSurfaceMessages, core.ModelSurfaceEmbeddings, core.ModelSurfaceAudioTranscriptions, core.ModelSurfaceAudioSpeech, core.ModelSurfaceImages, core.ModelSurfaceVideos} {
				canonical := core.SurfacePath(surface)
				normalized := strings.ToLower(strings.TrimSpace(path))
				if normalized == canonical || normalized == strings.TrimPrefix(canonical, "/v1") {
					surfaces = appendSurface(surfaces, surface)
				}
			}
		}
	case capabilities != nil && capabilities.Surfaces != (core.ModelSurfaceCapabilities{}):
		for _, surface := range []core.ModelSurface{core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses, core.ModelSurfaceMessages} {
			if capabilities.SurfaceCompatibility(surface) == core.SupportSupported {
				surfaces = append(surfaces, surface)
			}
		}
	default:
		// An operation/feature alone does not identify an actual wire API.
		// Lookup mode is used where provider defaults may be facade dispatch
		// availability, so an undeclared row never upgrades those to native.
		return nil, capabilities, false
	}
	if capabilities != nil && core.PreparedCapabilityDenialsUsable(capabilities, time.Now()) {
		surfaces = slices.DeleteFunc(surfaces, func(surface core.ModelSurface) bool {
			return capabilities.SurfaceCompatibility(surface) == core.SupportUnsupported || capabilities.OperationCompatibility(operationForSurface(surface)) == core.SupportUnsupported
		})
	}
	return surfaces, capabilities, len(surfaces) > 0
}

func knownSurfaces(values []core.ModelSurface) []core.ModelSurface {
	var result []core.ModelSurface
	for _, surface := range values {
		if core.KnownModelSurface(surface) {
			result = appendSurface(result, surface)
		}
	}
	return result
}
func appendSurface(values []core.ModelSurface, surface core.ModelSurface) []core.ModelSurface {
	if !slices.Contains(values, surface) {
		return append(values, surface)
	}
	return values
}

func operationForSurface(surface core.ModelSurface) core.ModelOperation {
	switch surface {
	case core.ModelSurfaceChatCompletions, core.ModelSurfaceResponses, core.ModelSurfaceMessages:
		return core.ModelOperationChat
	case core.ModelSurfaceEmbeddings:
		return core.ModelOperationEmbeddings
	case core.ModelSurfaceAudioTranscriptions:
		return core.ModelOperationAudioIn
	case core.ModelSurfaceAudioSpeech:
		return core.ModelOperationAudioOut
	case core.ModelSurfaceImages:
		return core.ModelOperationImage
	case core.ModelSurfaceVideos:
		return core.ModelOperationVideo
	}
	return ""
}

// declaredProvider is one immutable model/surface snapshot used by Adapter.
// Calling Client directly avoids legacy provider-wide dispatch rejecting an
// explicit model override. No catalog callback is retained in its executors.
type declaredProvider struct {
	owner        *Provider
	model        string
	surfaces     []core.ModelSurface
	capabilities *core.ModelCapabilities
}

func (p declaredProvider) NativeSurfaces(string) []core.ModelSurface { return slices.Clone(p.surfaces) }
func (p declaredProvider) ListModels(ctx context.Context, c *core.Credential) ([]core.ModelInfo, error) {
	return p.owner.ListModels(ctx, c)
}
func (p declaredProvider) Invoke(ctx context.Context, r core.Request) (core.Response, error) {
	return p.owner.client.Invoke(ctx, p.owner.info.ID, r)
}
func (p declaredProvider) Stream(ctx context.Context, r core.Request) (core.StreamIter, error) {
	return p.owner.client.Stream(ctx, p.owner.info.ID, r)
}

func (p declaredProvider) PrepareCall(_ context.Context, request core.Request, options core.CallPreparationOptions) (core.PreparedCall, error) {
	if request.Model != p.model || !slices.Contains(p.surfaces, request.Surface) {
		return nil, &core.SurfaceError{Surface: request.Surface, Model: request.Model}
	}
	request.Body = slices.Clone(request.Body)
	request.Header = request.Header.Clone()
	request.Credential = nil
	transport := core.CallTransportJSON
	if options.Mode == core.CallModeStream {
		transport = core.CallTransportSSE
	}
	requirements := options.Requirements
	if payload, err := sourceusage.Decode(request.Body); err == nil {
		inferred := translate.InferFidelityRequirements(payload)
		requirements.Content = requirements.Content || inferred.Content
		requirements.Tools = requirements.Tools || inferred.Tools
		requirements.Schema = requirements.Schema || inferred.Schema
		requirements.Billing = requirements.Billing || inferred.Billing
		requirements.Replay = requirements.Replay || inferred.Replay
	}
	d := core.CallDescription{Ingress: request.Surface, Model: request.Model, Mode: options.Mode, Operation: operationForSurface(request.Surface), UpstreamAPI: core.UpstreamAPI(request.Surface), Deployment: "extension/v1", WirePreserved: core.SupportUnknown, UpstreamTransport: transport, OutputTransport: transport, Requirements: requirements, Assessment: core.CallAssessment{Feasibility: core.SupportSupported, Evaluated: []string{"content", "tools", "schema", "billing", "replay"}}}
	if c := p.capabilities; c != nil && core.PreparedCapabilityDenialsUsable(c, time.Now()) {
		if options.Mode == core.CallModeStream && c.Streaming == core.SupportUnsupported {
			d.Assessment.Feasibility = core.SupportUnsupported
		}
	}
	return core.NewPreparedCall(d, func(ctx context.Context, c *core.Credential) (core.Response, error) {
		selected := request
		selected.Credential = c
		var record *core.SourceUsageRecorder
		if dialect := nativeSourceDialect(request.Surface); dialect != "" {
			record = core.BeginSourceUsage(ctx, d.UpstreamAPI, dialect)
		}
		response, err := p.owner.client.Invoke(ctx, p.owner.info.ID, selected)
		// The v1 Chat facade may itself convert another API, so its Chat
		// usage is not classified as original native evidence. Explicit
		// Responses/Messages operations have known native source grammars.
		if record != nil {
			quality := translate.CompletionComplete
			if err != nil {
				quality = translate.CompletionFailed
			}
			if value, decodeErr := sourceusage.Decode(response.Body); decodeErr == nil {
				if usage, exists := value["usage"]; exists {
					record.Merge(usage)
				}
				if value["error"] != nil || value["status"] == "failed" || value["status"] == "cancelled" {
					quality = translate.CompletionFailed
				} else if quality == translate.CompletionComplete && (value["status"] == "incomplete" || value["stop_reason"] == "max_tokens") {
					quality = translate.CompletionTruncated
				}
			} else if err == nil {
				quality = translate.CompletionFailed
			}
			sourceusage.Finish(record, ctx, quality)
		}
		return response, err
	}, func(ctx context.Context, c *core.Credential) (core.StreamIter, error) {
		selected := request
		selected.Credential = c
		var record *core.SourceUsageRecorder
		if dialect := nativeSourceDialect(request.Surface); dialect != "" {
			record = core.BeginSourceUsage(ctx, d.UpstreamAPI, dialect)
		}
		stream, err := p.owner.client.Stream(ctx, p.owner.info.ID, selected)
		if err != nil {
			if record != nil {
				sourceusage.Finish(record, ctx, translate.CompletionFailed)
			}
			return nil, err
		}
		if record != nil {
			stream = sourceusage.Stream(ctx, stream, d.UpstreamAPI, record)
		}
		return stream, nil
	}), nil
}

func nativeSourceDialect(surface core.ModelSurface) translate.UsageDialect {
	switch surface {
	case core.ModelSurfaceResponses:
		return translate.UsageDialectResponses
	case core.ModelSurfaceMessages:
		return translate.UsageDialectMessages
	}
	return ""
}
