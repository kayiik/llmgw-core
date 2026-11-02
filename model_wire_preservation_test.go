package core_test

import (
	"encoding/json"
	"maps"
	"reflect"
	"testing"

	core "github.com/kayiik/llmgw-core"
)

func TestModelInfoWirePreservationRoundTrip(t *testing.T) {
	const payload = `{"id":"scoped-model","object":"model","created":0,"owned_by":"fixture","supported_apis":["/v1/chat/completions"],"wire_preservation":{"chat_completions":"unsupported","responses":"supported","messages":"unknown"}}`
	var row core.ModelInfo
	if err := json.Unmarshal([]byte(payload), &row); err != nil {
		t.Fatal(err)
	}
	want := map[core.ModelSurface]core.SupportState{
		core.ModelSurfaceChatCompletions: core.SupportUnsupported,
		core.ModelSurfaceResponses:       core.SupportSupported,
		core.ModelSurfaceMessages:        core.SupportUnknown,
	}
	if !maps.Equal(row.WirePreservation, want) {
		t.Fatalf("wire preservation = %v, want %v", row.WirePreservation, want)
	}
	if !reflect.DeepEqual(row.SupportedAPIs, []string{"/v1/chat/completions"}) {
		t.Fatalf("endpoint support changed: %v", row.SupportedAPIs)
	}
	encoded, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	var decoded core.ModelInfo
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, row) {
		t.Fatalf("round trip = %#v, want %#v", decoded, row)
	}
	for _, input := range []string{
		`{"id":"legacy-model"}`,
		`{"id":"legacy-model","wire_preservation":null}`,
		`{"id":"legacy-model","wire_preservation":{}}`,
	} {
		var legacy core.ModelInfo
		if err := json.Unmarshal([]byte(input), &legacy); err != nil {
			t.Fatal(err)
		}
		if _, declared := legacy.WirePreservation[core.ModelSurfaceChatCompletions]; declared {
			t.Fatal("missing preservation became a declaration")
		}
		encoded, err := json.Marshal(legacy)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(encoded, &fields); err != nil {
			t.Fatal(err)
		}
		if _, present := fields["wire_preservation"]; present {
			t.Fatalf("empty preservation not omitted: %s", encoded)
		}
	}
}

func TestCatalogWirePreservationSnapshotCopies(t *testing.T) {
	for _, conditional := range []bool{false, true} {
		name := "Save"
		if conditional {
			name = "SaveIf"
		}
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			store := core.NewMemoryCatalogStore()
			key := core.CatalogKey{Instance: "fixture", CredentialKey: "owner-a"}
			original := map[core.ModelSurface]core.SupportState{
				core.ModelSurfaceResponses: core.SupportSupported,
				core.ModelSurfaceMessages:  core.SupportUnsupported,
			}
			evidence := core.CatalogEvidence{Models: []core.ModelInfo{
				{ID: "scoped-model", WirePreservation: maps.Clone(original)},
				{ID: "legacy-model"},
			}}
			var err error
			if conditional {
				_, err = store.SaveIf(ctx, key, evidence, "")
			} else {
				_, err = store.Save(ctx, key, evidence)
			}
			if err != nil {
				t.Fatal(err)
			}
			evidence.Models[0].WirePreservation[core.ModelSurfaceResponses] = core.SupportUnsupported
			first, err := store.Load(ctx, key)
			if err != nil {
				t.Fatal(err)
			}
			if !maps.Equal(first.Evidence.Models[0].WirePreservation, original) {
				t.Fatal("save retained the caller's mutable preservation map")
			}
			delete(first.Evidence.Models[0].WirePreservation, core.ModelSurfaceMessages)
			first.Evidence.Models[0].WirePreservation[core.ModelSurfaceChatCompletions] = core.SupportSupported
			second, err := store.Load(ctx, key)
			if err != nil {
				t.Fatal(err)
			}
			if !maps.Equal(second.Evidence.Models[0].WirePreservation, original) {
				t.Fatal("load exposed the stored mutable preservation map")
			}
			if second.Evidence.Models[1].WirePreservation != nil {
				t.Fatal("copy changed absent preservation into a declaration")
			}
		})
	}
}
