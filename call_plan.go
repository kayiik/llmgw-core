package core

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	translate "github.com/kayiik/llm-translate"
	"github.com/kayiik/llmgw-core/internal/callcontext"
)

// CallMode is the lifecycle selected before an upstream operation is opened.
type CallMode string

const (
	CallModeInvoke CallMode = "invoke"
	CallModeStream CallMode = "stream"
)

// UpstreamAPI identifies the actual upstream grammar, not adapter availability.
type UpstreamAPI string

const (
	UpstreamAPIUnknown               UpstreamAPI = "unknown"
	UpstreamAPIChatCompletions       UpstreamAPI = "chat_completions"
	UpstreamAPIResponses             UpstreamAPI = "responses"
	UpstreamAPIMessages              UpstreamAPI = "messages"
	UpstreamAPIGoogleGenerateContent UpstreamAPI = "google_generate_content"
	UpstreamAPIOllamaChat            UpstreamAPI = "ollama_chat"
)

type CallTransport string

const (
	CallTransportJSON        CallTransport = "json"
	CallTransportSSE         CallTransport = "sse"
	CallTransportNDJSON      CallTransport = "ndjson"
	CallTransportBufferedSSE CallTransport = "buffered_sse"
)

// CallRequirements are fidelity requirements, not inferred model capabilities.
type CallRequirements struct{ Content, Tools, Schema, Billing, Replay, Incremental bool }

type CallPreparationOptions struct {
	Mode            CallMode
	AllowAdaptation bool
	Requirements    CallRequirements
}

type CallConversion struct {
	ID             string
	FromAPI, ToAPI UpstreamAPI
	Assessment     CallAssessment
}

type CallAssessment struct {
	Feasibility  Support
	Evaluated    []string
	UnknownPaths []string
	Losses       []Loss
}

// CallDescription is a secret-free observation of a selected execution.
// Supported feasibility concerns the evaluated mapping, not model entitlement.
type CallDescription struct {
	Ingress                            ModelSurface
	Model                              string
	Mode                               CallMode
	Operation                          ModelOperation
	UpstreamAPI                        UpstreamAPI
	APIVersion, Deployment             string
	WirePreserved                      Support
	UpstreamTransport, OutputTransport CallTransport
	Conversions                        []CallConversion
	Assessment                         CallAssessment
	Requirements                       CallRequirements
}

// CallPreparer is optional. Preparation must not refresh credentials or call
// upstreams. It retains the selected payload instead of planning again at I/O.
type CallPreparer interface {
	PrepareCall(context.Context, Request, CallPreparationOptions) (PreparedCall, error)
}

// PreparedCall executes a selected mapping with a credential rebound by the
// runtime. Its implementations must never reselect APIs or weaken controls.
// Runtime handles, rather than this reusable low-level object, enforce one use.
type PreparedCall interface {
	Description() CallDescription
	Invoke(context.Context, *Credential) (Response, error)
	Stream(context.Context, *Credential) (StreamIter, error)
}

var (
	ErrPreparedCallStale    = errors.New("core: prepared call binding is stale; prepare and admit again")
	ErrPreparedCallConsumed = errors.New("core: prepared call was already executed")
	ErrPreparedCallMode     = errors.New("core: prepared call mode does not match execution")
)

type preparedCall struct {
	description CallDescription
	invoke      func(context.Context, *Credential) (Response, error)
	stream      func(context.Context, *Credential) (StreamIter, error)
}

// NewPreparedCall snapshots description. Executors own their private prepared
// payload; neither the description nor a caller's source maps may mutate it.
func NewPreparedCall(description CallDescription, invoke func(context.Context, *Credential) (Response, error), stream func(context.Context, *Credential) (StreamIter, error)) PreparedCall {
	return &preparedCall{description: cloneCallDescription(description), invoke: invoke, stream: stream}
}

func cloneCallDescription(d CallDescription) CallDescription {
	d.Conversions = slices.Clone(d.Conversions)
	for i := range d.Conversions {
		d.Conversions[i].Assessment.Evaluated = slices.Clone(d.Conversions[i].Assessment.Evaluated)
		d.Conversions[i].Assessment.UnknownPaths = slices.Clone(d.Conversions[i].Assessment.UnknownPaths)
		d.Conversions[i].Assessment.Losses = slices.Clone(d.Conversions[i].Assessment.Losses)
	}
	d.Assessment.Evaluated = slices.Clone(d.Assessment.Evaluated)
	d.Assessment.UnknownPaths = slices.Clone(d.Assessment.UnknownPaths)
	d.Assessment.Losses = slices.Clone(d.Assessment.Losses)
	return d
}

func (p *preparedCall) Description() CallDescription { return cloneCallDescription(p.description) }

func (p *preparedCall) Invoke(ctx context.Context, credential *Credential) (Response, error) {
	if err := callcontext.Check(ctx); err != nil {
		return Response{}, err
	}
	if p.description.Mode != CallModeInvoke || p.invoke == nil {
		return Response{}, ErrPreparedCallMode
	}
	if _, err := PlanPreparedCall(p.description, CallPolicy{AllowAdaptation: true}); err != nil {
		return Response{Losses: slices.Clone(p.description.Assessment.Losses)}, err
	}
	return p.invoke(ctx, credential)
}

func (p *preparedCall) Stream(ctx context.Context, credential *Credential) (StreamIter, error) {
	if err := callcontext.Check(ctx); err != nil {
		return nil, err
	}
	if p.description.Mode != CallModeStream || p.stream == nil {
		return nil, ErrPreparedCallMode
	}
	if _, err := PlanPreparedCall(p.description, CallPolicy{AllowAdaptation: true}); err != nil {
		return nil, err
	}
	return p.stream(ctx, credential)
}

// PrepareCall uses only a provider's explicit preparation contract. An old
// custom/extension provider remains unknown, never inferred native by brand,
// advertised ingress, empty losses or a decorator's Unwrap method.
func PrepareCall(ctx context.Context, provider Provider, request Request, options CallPreparationOptions) (PreparedCall, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := request.Validate(); err != nil {
		return nil, err
	}
	if options.Mode == "" {
		options.Mode = CallModeInvoke
	}
	if options.Mode != CallModeInvoke && options.Mode != CallModeStream {
		return nil, ErrPreparedCallMode
	}
	request.Body = slices.Clone(request.Body)
	request.Header = request.Header.Clone()
	if request.Credential != nil {
		credential := *request.Credential
		credential.Headers = maps.Clone(credential.Headers)
		credential.Metadata = maps.Clone(credential.Metadata)
		request.Credential = &credential
	}
	if preparer, ok := provider.(CallPreparer); ok {
		return preparer.PrepareCall(ctx, request, options)
	}
	return NewPreparedCall(CallDescription{Ingress: request.Surface, Model: request.Model, Mode: options.Mode,
		UpstreamAPI: UpstreamAPIUnknown, Assessment: CallAssessment{Feasibility: SupportUnknown}}, nil, nil), nil
}

type CallPolicy struct{ AllowAdaptation, RequireWirePreservation, RequireIncremental, RequireReplay bool }
type CallPlan struct {
	Description CallDescription
	Disposition TransportDisposition
}

type CallPlanError struct {
	Reason       string
	Losses       []Loss
	UnknownPaths []string
}

// PreparedCapabilityDenialsUsable prevents expired dynamic cache entries from
// becoming permanent denials. Static/operator declarations are configuration
// policy, not expiring discovery. True is not confirmation of positive model
// capability, freshness or entitlement; legacy PlanTransport is unchanged.
func PreparedCapabilityDenialsUsable(capabilities *ModelCapabilities, at time.Time) bool {
	if capabilities == nil || capabilities.SchemaVersion != ModelCapabilitiesSchemaVersion {
		return false
	}
	if capabilities.Provenance.Source == ModelCapabilitySourceOperatorOverride || capabilities.Provenance.Source == ModelCapabilitySourceRegistryStatic {
		return true
	}
	return capabilities.Freshness.Status(at) != FreshnessExpired
}

func (e *CallPlanError) Error() string {
	message := "core: prepared call rejected: " + e.Reason
	var paths []string
	add := func(path string) {
		if path == "" || len(paths) >= 8 || slices.Contains(paths, path) {
			return
		}
		path = strings.Map(func(r rune) rune {
			if r < ' ' || r == 0x7f {
				return ' '
			}
			return r
		}, path)
		if len(path) > 160 {
			path = strings.ToValidUTF8(path[:160], "") + "..."
		}
		paths = append(paths, path)
	}
	for _, loss := range e.Losses {
		if loss.Class == translate.LossUnsupported || (LossPolicy{}).Decide(loss) == LossReject {
			add(loss.Path)
		}
	}
	for _, path := range e.UnknownPaths {
		add(path)
	}
	if len(paths) > 0 {
		message += " at " + strings.Join(paths, ", ")
	}
	return message
}
func (e *CallPlanError) ProviderErrorClassification() ProviderErrorClassification {
	return ProviderErrorClassification{FailoverEligible: true}
}

// PlanPreparedCall admits evidence, not a provider/model catalogue guess. It
// does not replace product authorization, freshness or exact-target checks.
func PlanPreparedCall(d CallDescription, policy CallPolicy) (CallPlan, error) {
	reject := func(reason string) (CallPlan, error) {
		return CallPlan{Description: cloneCallDescription(d), Disposition: TransportReject}, &CallPlanError{Reason: reason, Losses: slices.Clone(d.Assessment.Losses), UnknownPaths: slices.Clone(d.Assessment.UnknownPaths)}
	}
	if policy.RequireWirePreservation && policy.AllowAdaptation {
		return reject("transparent_adaptation_policy_conflict")
	}
	if d.Mode != CallModeInvoke && d.Mode != CallModeStream {
		return reject("invalid_mode")
	}
	if d.UpstreamAPI == "" || d.UpstreamAPI == UpstreamAPIUnknown {
		return reject("upstream_api_unknown: a reviewed actual API declaration is required")
	}
	adapted := len(d.Conversions) > 0 || string(d.Ingress) != string(d.UpstreamAPI)
	if adapted && !policy.AllowAdaptation {
		return reject(fmt.Sprintf("adaptation_required: %s ingress uses %s; explicitly allow compatibility or configure route adaptation", d.Ingress, d.UpstreamAPI))
	}
	if adapted {
		if len(d.Conversions) == 0 {
			return reject("conversion_description_missing")
		}
		previous := UpstreamAPI(d.Ingress)
		for _, edge := range d.Conversions {
			if edge.ID == "" || edge.FromAPI != previous || edge.ToAPI == "" || edge.ToAPI == UpstreamAPIUnknown {
				return reject("conversion_chain_invalid")
			}
			previous = edge.ToAPI
		}
		if previous != d.UpstreamAPI {
			return reject("conversion_chain_target_mismatch")
		}
	}
	for _, loss := range d.Assessment.Losses {
		if loss.Class == translate.LossUnsupported || (LossPolicy{}).Decide(loss) == LossReject {
			return reject("request_loss")
		}
	}
	if d.Assessment.Feasibility == SupportUnsupported {
		return reject("request_unsupported")
	}
	if adapted && (d.Assessment.Feasibility != SupportSupported || len(d.Assessment.UnknownPaths) > 0) {
		return reject("required_conversion_semantics_unassessed")
	}
	if adapted {
		for _, feature := range []struct {
			name     string
			required bool
		}{
			{"content", d.Requirements.Content}, {"tools", d.Requirements.Tools}, {"schema", d.Requirements.Schema}, {"billing", d.Requirements.Billing}, {"replay", d.Requirements.Replay || policy.RequireReplay},
		} {
			if feature.required && !slices.Contains(d.Assessment.Evaluated, feature.name) {
				return reject(feature.name + "_semantics_unassessed")
			}
		}
	}
	if policy.RequireWirePreservation && (adapted || d.WirePreserved != SupportSupported) {
		return reject("wire_preservation_required")
	}
	if policy.RequireIncremental || d.Requirements.Incremental {
		if d.Mode != CallModeStream || (d.UpstreamTransport != CallTransportSSE && d.UpstreamTransport != CallTransportNDJSON) || d.OutputTransport != CallTransportSSE {
			return reject("incremental_stream_required")
		}
	}
	if adapted && (policy.RequireReplay || d.Requirements.Replay) && !slices.Contains(d.Assessment.Evaluated, "replay") {
		return reject("replay_semantics_unassessed")
	}
	disposition := TransportNative
	if adapted {
		disposition = TransportAdapted
	}
	return CallPlan{Description: cloneCallDescription(d), Disposition: disposition}, nil
}
