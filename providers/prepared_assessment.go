package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"mime"
	"slices"
	"strings"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

func preparedJSON(request core.Request) (map[string]any, error) {
	media, _, err := mime.ParseMediaType(request.ContentType)
	if err != nil || media != core.ContentTypeJSON {
		return nil, openAIInvalid("a prepared request must be JSON", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(request.Body))
	decoder.UseNumber()
	var payload map[string]any
	if err := decoder.Decode(&payload); err != nil || payload == nil {
		return nil, openAIInvalid("a prepared request must be a JSON object", err)
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, openAIInvalid("a prepared request must contain one JSON object", err)
	}
	return payload, nil
}

func preparedRequirements(payload map[string]any, explicit core.CallRequirements) core.CallRequirements {
	inferred := translate.InferFidelityRequirements(payload)
	explicit.Content = explicit.Content || inferred.Content
	explicit.Tools = explicit.Tools || inferred.Tools
	explicit.Schema = explicit.Schema || inferred.Schema
	explicit.Billing = explicit.Billing || inferred.Billing
	explicit.Replay = explicit.Replay || inferred.Replay
	return explicit
}

// Request controls dropped by a facade are not waived by a compatibility
// grant. Legacy entry points retain their original advisory interpretation.
func preparedAssessment(losses []core.Loss) core.CallAssessment {
	a := core.CallAssessment{Feasibility: core.SupportSupported, Evaluated: []string{"content", "tools", "schema", "billing", "replay"}}
	for _, loss := range losses {
		if loss.Class == translate.LossDropped && !strings.Contains(loss.Detail, "already behaves") {
			loss.Severity = translate.LossMaterial
		}
		a.Losses = append(a.Losses, loss)
		if loss.Class == translate.LossUnsupported || (core.LossPolicy{}).Decide(loss) == core.LossReject {
			a.Feasibility = core.SupportUnsupported
		}
	}
	return a
}

func preparedDescription(request core.Request, options core.CallPreparationOptions, api core.UpstreamAPI, wire core.Support, payload map[string]any) core.CallDescription {
	mode := options.Mode
	if mode == "" {
		mode = core.CallModeInvoke
	}
	transport := core.CallTransportJSON
	if mode == core.CallModeStream {
		transport = core.CallTransportSSE
	}
	return core.CallDescription{Ingress: request.Surface, Model: request.Model, Mode: mode, Operation: core.ModelOperationChat, UpstreamAPI: api,
		WirePreserved: wire, UpstreamTransport: transport, OutputTransport: transport,
		Requirements: preparedRequirements(payload, options.Requirements), Assessment: preparedAssessment(nil)}
}

func preparedEdge(description *core.CallDescription, id string, from core.UpstreamAPI) {
	description.Conversions = append(description.Conversions, core.CallConversion{ID: id, FromAPI: from, ToAPI: description.UpstreamAPI, Assessment: description.Assessment})
}

// Internal provider mappers have their own finite coverage; they are not
// translate edges. In particular, copying Chat history to native Ollama does
// not establish the native argument-object/tool_name replay contract.
func assessOllamaPrepared(payload map[string]any, losses []core.Loss) core.CallAssessment {
	a := preparedAssessment(losses)
	auditPreparedToolDeclarations(payload, &a, false)
	messages, _ := payload["messages"].([]any)
	for i, raw := range messages {
		message, _ := raw.(map[string]any)
		for _, field := range []string{"tool_calls", "tool_call_id"} {
			if message[field] != nil {
				a.Losses = append(a.Losses, core.Loss{Path: fmt.Sprintf("messages.%d.%s", i, field), Class: translate.LossUnsupported, Severity: translate.LossMaterial, Detail: "native Ollama history linkage requires a reviewed argument-object/tool_name mapping"})
				a.Feasibility = core.SupportUnsupported
			}
		}
	}
	return a
}

func unknownPreparedFields(value any, path, known string, a *core.CallAssessment) {
	object, _ := value.(map[string]any)
	for _, key := range slices.Sorted(maps.Keys(object)) {
		if object[key] == nil || slices.Contains(strings.Fields(known), key) {
			continue
		}
		a.UnknownPaths = append(a.UnknownPaths, path+"."+key)
		if a.Feasibility != core.SupportUnsupported {
			a.Feasibility = core.SupportUnknown
		}
		a.Evaluated = slices.DeleteFunc(a.Evaluated, func(feature string) bool { return feature == "tools" || feature == "schema" })
	}
}

// Audit finite control envelopes, not user argument/schema-property data.
// Those data objects are carried whole; treating their arbitrary property
// names as envelope controls would incorrectly reject valid schemas/tools.
func auditPreparedToolDeclarations(payload map[string]any, a *core.CallAssessment, google bool) {
	tools, _ := payload["tools"].([]any)
	for index, raw := range tools {
		path := fmt.Sprintf("tools.%d", index)
		unknownPreparedFields(raw, path, "type function", a)
		tool, _ := raw.(map[string]any)
		known := "name description parameters"
		if google {
			known += " strict"
		}
		unknownPreparedFields(tool["function"], path+".function", known, a)
	}
}

func assessGooglePrepared(payload map[string]any, losses []core.Loss) core.CallAssessment {
	// Name/description label a schema; dropping a valid label does not weaken
	// its constraints. Strict guarantees and all actual schema/control losses
	// remain material in the prepared path.
	format, _ := payload["response_format"].(map[string]any)
	schema, _ := format["json_schema"].(map[string]any)
	var semantic, metadata []core.Loss
	for _, loss := range losses {
		if loss.Class == translate.LossDropped && loss.Severity == translate.LossAdvisory && (loss.Path == "response_format.json_schema.name" || loss.Path == "response_format.json_schema.description") {
			field := strings.TrimPrefix(loss.Path, "response_format.json_schema.")
			if _, valid := schema[field].(string); valid {
				metadata = append(metadata, loss)
				continue
			}
		}
		semantic = append(semantic, loss)
	}
	a := preparedAssessment(semantic)
	a.Losses = translate.NewReport(slices.Concat(a.Losses, metadata)...).Losses
	for _, field := range []string{"name", "description", "strict"} {
		value := schema[field]
		if value == nil {
			continue
		}
		valid := false
		if field == "strict" {
			_, valid = value.(bool)
		} else {
			_, valid = value.(string)
		}
		if !valid {
			a.Feasibility = core.SupportUnsupported
			a.Losses = append(a.Losses, core.Loss{Path: "response_format.json_schema." + field, Class: translate.LossUnsupported, Severity: translate.LossMaterial, Detail: "the response format metadata/control has an invalid type"})
		}
	}
	auditPreparedToolDeclarations(payload, &a, true)
	messages, _ := payload["messages"].([]any)
	for index, raw := range messages {
		message, _ := raw.(map[string]any)
		calls, _ := message["tool_calls"].([]any)
		for callIndex, rawCall := range calls {
			path := fmt.Sprintf("messages.%d.tool_calls.%d", index, callIndex)
			call, _ := rawCall.(map[string]any)
			unknownPreparedFields(call, path, "id type function extra_content thought_signature", &a)
			unknownPreparedFields(call["function"], path+".function", "name arguments thought_signature", &a)
			unknownPreparedFields(call["extra_content"], path+".extra_content", "google", &a)
			extra, _ := call["extra_content"].(map[string]any)
			unknownPreparedFields(extra["google"], path+".extra_content.google", "thought_signature", &a)
		}
	}
	return a
}

// Future opaque response state cannot be repaired from displayed text. The
// prepared facade refuses it when observed, without changing native/legacy
// entry points or claiming a pre-inference refusal after work already ran.
func assessGooglePreparedResponse(response map[string]any) core.CallAssessment {
	a := core.CallAssessment{Feasibility: core.SupportSupported, Evaluated: []string{"content", "tools", "replay"}}
	lost := func(path, detail string) {
		a.Losses = append(a.Losses, core.Loss{Path: path, Class: translate.LossDropped, Severity: translate.LossMaterial, Detail: detail})
		a.Feasibility = core.SupportUnsupported
	}
	candidates, _ := response["candidates"].([]any)
	if len(candidates) > 1 {
		lost("candidates", "the Chat facade only represents one Google candidate")
	}
	for index, part := range googleParts(response) {
		path := fmt.Sprintf("candidates.0.content.parts.%d", index)
		unknownPreparedFields(part, path, "text thought thoughtSignature functionCall", &a)
		function, hasCall := part["functionCall"].(map[string]any)
		if signature, _ := part["thoughtSignature"].(string); signature != "" && !hasCall {
			lost(path+".thoughtSignature", "opaque text/thinking/media state is not represented by the Chat facade")
		}
		if hasCall {
			unknownPreparedFields(function, path+".functionCall", "id name args", &a)
			if name, _ := function["name"].(string); name == "" {
				lost(path+".functionCall.name", "the function call has no representable name")
			}
			if args, ok := function["args"]; ok && args != nil {
				if _, object := args.(map[string]any); !object {
					lost(path+".functionCall.args", "the function arguments are not a JSON object")
				}
			}
		}
	}
	return a
}
