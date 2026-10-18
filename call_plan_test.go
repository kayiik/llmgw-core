package core_test

import (
	"context"
	"errors"
	"testing"
	"time"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

func TestPreparedPlanSeparatesActualAPIIngressAndWire(t *testing.T) {
	description := core.CallDescription{Ingress: core.ModelSurfaceChatCompletions, Model: "m", Mode: core.CallModeInvoke,
		Operation: core.ModelOperationChat, UpstreamAPI: core.UpstreamAPIChatCompletions,
		WirePreserved: core.SupportUnsupported, UpstreamTransport: core.CallTransportJSON, OutputTransport: core.CallTransportJSON,
		Assessment: core.CallAssessment{Feasibility: core.SupportSupported}}
	if plan, err := core.PlanPreparedCall(description, core.CallPolicy{}); err != nil || plan.Disposition != core.TransportNative {
		t.Fatalf("rebuilt native Chat rejected: %+v %v", plan, err)
	}
	if _, err := core.PlanPreparedCall(description, core.CallPolicy{RequireWirePreservation: true}); err == nil {
		t.Fatal("rebuilt Chat labeled transparent")
	}
	description.UpstreamAPI = core.UpstreamAPIGoogleGenerateContent
	description.Conversions = []core.CallConversion{{ID: "chat-to-google", FromAPI: core.UpstreamAPIChatCompletions, ToAPI: description.UpstreamAPI}}
	if _, err := core.PlanPreparedCall(description, core.CallPolicy{}); err == nil {
		t.Fatal("Google facade admitted as native")
	}
	if plan, err := core.PlanPreparedCall(description, core.CallPolicy{AllowAdaptation: true}); err != nil || plan.Disposition != core.TransportAdapted {
		t.Fatalf("feasible mapping rejected: %+v %v", plan, err)
	}
	description.Assessment.Losses = []core.Loss{{Path: "messages.0.content.1", Severity: translate.LossMaterial, Class: translate.LossDropped}}
	if _, err := core.PlanPreparedCall(description, core.CallPolicy{AllowAdaptation: true}); err == nil {
		t.Fatal("permission waived material loss")
	}
	description.Assessment.Losses = nil
	description.Assessment.Feasibility = core.SupportUnknown
	if _, err := core.PlanPreparedCall(description, core.CallPolicy{AllowAdaptation: true}); err == nil {
		t.Fatal("empty report promoted unassessed mapping")
	}
	description.UpstreamAPI = core.UpstreamAPIUnknown
	if _, err := core.PlanPreparedCall(description, core.CallPolicy{AllowAdaptation: true}); err == nil {
		t.Fatal("unknown API promoted by compatibility")
	}
}

func TestPreparedDescriptionDetachedAndModeChecked(t *testing.T) {
	description := core.CallDescription{Mode: core.CallModeInvoke, Conversions: []core.CallConversion{{ID: "original"}},
		Assessment: core.CallAssessment{Evaluated: []string{"content"}, UnknownPaths: []string{"path"}, Losses: []core.Loss{{Path: "original"}}}}
	invocations := 0
	call := core.NewPreparedCall(description, func(context.Context, *core.Credential) (core.Response, error) {
		invocations++
		return core.Response{}, nil
	}, nil)
	description.Conversions[0].ID = "mutated"
	copy := call.Description()
	copy.Assessment.Losses[0].Path = "erased"
	copy.Assessment.Evaluated[0] = "erased"
	copy.Assessment.UnknownPaths[0] = "erased"
	copy.Conversions[0].ID = "erased"
	got := call.Description()
	if got.Conversions[0].ID != "original" || got.Assessment.Losses[0].Path != "original" || got.Assessment.Evaluated[0] != "content" || got.Assessment.UnknownPaths[0] != "path" {
		t.Fatalf("mutable description: %+v", got)
	}
	if _, err := call.Stream(context.Background(), nil); !errors.Is(err, core.ErrPreparedCallMode) {
		t.Fatalf("wrong mode: %v", err)
	}
	if invocations != 0 {
		t.Fatal("wrong-mode call executed")
	}
}

func TestPreparedPlanRejectsSyntheticIncrementalAndUnassessedReplay(t *testing.T) {
	d := core.CallDescription{Ingress: core.ModelSurfaceChatCompletions, Model: "m", Mode: core.CallModeStream, Operation: core.ModelOperationChat,
		UpstreamAPI: core.UpstreamAPIResponses, UpstreamTransport: core.CallTransportJSON, OutputTransport: core.CallTransportBufferedSSE,
		Conversions: []core.CallConversion{{ID: "chat-to-responses", FromAPI: core.UpstreamAPIChatCompletions, ToAPI: core.UpstreamAPIResponses}}, Assessment: core.CallAssessment{Feasibility: core.SupportSupported}}
	if _, err := core.PlanPreparedCall(d, core.CallPolicy{AllowAdaptation: true, RequireIncremental: true}); err == nil {
		t.Fatal("buffered response admitted as incremental")
	}
	if _, err := core.PlanPreparedCall(d, core.CallPolicy{AllowAdaptation: true, RequireReplay: true}); err == nil {
		t.Fatal("unassessed replay accepted")
	}
}

func TestPreparedPlanRejectsEveryUnassessedRequiredFeature(t *testing.T) {
	for name, requirements := range map[string]core.CallRequirements{
		"content": {Content: true}, "tools": {Tools: true}, "schema": {Schema: true}, "billing": {Billing: true}, "replay": {Replay: true},
	} {
		t.Run(name, func(t *testing.T) {
			d := core.CallDescription{Ingress: core.ModelSurfaceChatCompletions, Model: "m", Mode: core.CallModeInvoke,
				UpstreamAPI: core.UpstreamAPIResponses, Conversions: []core.CallConversion{{ID: "chat-to-responses", FromAPI: core.UpstreamAPIChatCompletions, ToAPI: core.UpstreamAPIResponses}},
				Requirements: requirements, Assessment: core.CallAssessment{Feasibility: core.SupportSupported}}
			if _, err := core.PlanPreparedCall(d, core.CallPolicy{AllowAdaptation: true}); err == nil {
				t.Fatalf("unassessed %s admitted", name)
			}
			d.Assessment.Evaluated = []string{name}
			if _, err := core.PlanPreparedCall(d, core.CallPolicy{AllowAdaptation: true}); err != nil {
				t.Fatalf("assessed %s rejected: %v", name, err)
			}
		})
	}
}

func TestPreparedPlanRequiresNamedContinuousConversionChain(t *testing.T) {
	base := core.CallDescription{Ingress: core.ModelSurfaceMessages, Model: "m", Mode: core.CallModeInvoke, UpstreamAPI: core.UpstreamAPIGoogleGenerateContent,
		Assessment: core.CallAssessment{Feasibility: core.SupportSupported}}
	for _, edges := range [][]core.CallConversion{
		nil,
		{{FromAPI: core.UpstreamAPIMessages, ToAPI: core.UpstreamAPIGoogleGenerateContent}},
		{{ID: "wrong-source", FromAPI: core.UpstreamAPIChatCompletions, ToAPI: core.UpstreamAPIGoogleGenerateContent}},
		{{ID: "incomplete", FromAPI: core.UpstreamAPIMessages, ToAPI: core.UpstreamAPIChatCompletions}},
	} {
		d := base
		d.Conversions = edges
		if _, err := core.PlanPreparedCall(d, core.CallPolicy{AllowAdaptation: true}); err == nil {
			t.Fatalf("invalid chain admitted: %+v", edges)
		}
	}
}

func TestPreparedPlanRejectsContradictoryTransparentPermission(t *testing.T) {
	d := core.CallDescription{Ingress: core.ModelSurfaceChatCompletions, Model: "m", Mode: core.CallModeInvoke, UpstreamAPI: core.UpstreamAPIChatCompletions, WirePreserved: core.SupportSupported, Assessment: core.CallAssessment{Feasibility: core.SupportSupported}}
	if _, err := core.PlanPreparedCall(d, core.CallPolicy{AllowAdaptation: true, RequireWirePreservation: true}); err == nil {
		t.Fatal("transparent permission combined with adaptation")
	}
}

func TestPreparedCapabilityDenialsRespectDynamicExpiry(t *testing.T) {
	now := time.Unix(200, 0)
	past, future := now.Add(-time.Second), now.Add(time.Second)
	capabilities := &core.ModelCapabilities{SchemaVersion: core.ModelCapabilitiesSchemaVersion, Tools: core.SupportUnsupported, Freshness: core.ModelCapabilityFreshness{ExpiresAt: &past}, Provenance: core.ModelCapabilityProvenance{Source: core.ModelCapabilitySourceUpstreamReported}}
	if core.PreparedCapabilityDenialsUsable(capabilities, now) {
		t.Fatal("expired dynamic denial retained")
	}
	capabilities.Freshness.ExpiresAt = &future
	if !core.PreparedCapabilityDenialsUsable(capabilities, now) {
		t.Fatal("fresh dynamic denial ignored")
	}
	capabilities.Freshness.ExpiresAt = &past
	for _, source := range []core.ModelCapabilitySource{core.ModelCapabilitySourceRegistryStatic, core.ModelCapabilitySourceOperatorOverride} {
		capabilities.Provenance.Source = source
		if !core.PreparedCapabilityDenialsUsable(capabilities, now) {
			t.Fatal("configuration declaration incorrectly expired")
		}
	}
}
