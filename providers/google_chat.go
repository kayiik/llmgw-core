package providers

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"slices"
	"strconv"
	"strings"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

// chat performs one Chat Completions request over generateContent and
// returns Google's answer reshaped as the gateway reshapes it.
func (p *Google) chat(ctx context.Context, request core.Request) (core.Response, error) {
	call, err := googleChatPayload(request, false)
	if err != nil {
		return core.Response{}, err
	}
	access, err := p.access(request.Credential)
	if err != nil {
		return core.Response{}, err
	}
	endpoint, err := p.modelURL(request.Model, access.project, "generateContent")
	if err != nil {
		return core.Response{}, err
	}
	authorization, err := p.authorization(access)
	if err != nil {
		return core.Response{}, err
	}
	decoded, err := p.post(ctx, authorization, endpoint, call.payload)
	if err != nil {
		return core.Response{}, err
	}
	if err := googleEmptyReplyError(p.label(), decoded); err != nil {
		return core.Response{}, err
	}
	body, err := json.Marshal(googleToOpenAIChat(request.Model, decoded))
	if err != nil {
		return core.Response{}, unusableResponse("the "+p.label()+" answer could not be encoded", err)
	}
	return core.Response{Body: body, ContentType: core.ContentTypeJSON, Losses: call.losses}, nil
}

// GenerateImages asks an image-capable Gemini model for inline images over
// generateContent, as the gateway does, and returns at most Count of them
// with Google's usage, whose modality-tagged token counts carry the image
// cost. A model that answers without image data, such as a text-only one,
// fails and permits failover.
func (p *Google) GenerateImages(ctx context.Context, request core.GenerateImagesRequest, credential *core.Credential) (core.GenerateImagesResult, error) {
	if strings.TrimSpace(request.Prompt) == "" {
		return core.GenerateImagesResult{}, &core.ProviderError{Message: p.label() + ": a prompt is required", Class: core.ProviderErrorInvalidRequest}
	}
	access, err := p.access(credential)
	if err != nil {
		return core.GenerateImagesResult{}, err
	}
	endpoint, err := p.modelURL(request.Model, access.project, "generateContent")
	if err != nil {
		return core.GenerateImagesResult{}, err
	}
	payload := googleContentRequest([]map[string]any{{"role": "user", "content": request.Prompt}}, nil, nil, []string{"TEXT", "IMAGE"})
	authorization, err := p.authorization(access)
	if err != nil {
		return core.GenerateImagesResult{}, err
	}
	decoded, err := p.post(ctx, authorization, endpoint, payload)
	if err != nil {
		return core.GenerateImagesResult{}, err
	}
	images := make([]core.GeneratedImage, 0, 1)
	for _, part := range googleParts(decoded) {
		inline, ok := part["inlineData"].(map[string]any)
		if !ok {
			continue
		}
		encoded, _ := inline["data"].(string)
		mimeType, _ := inline["mimeType"].(string)
		raw, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil || len(raw) == 0 {
			continue
		}
		images = append(images, core.GeneratedImage{Data: raw, MimeType: mimeType})
		if request.Count > 0 && len(images) >= request.Count {
			break
		}
	}
	if len(images) == 0 {
		message := p.label() + ": the model returned no image data — it may be a text-only model"
		return core.GenerateImagesResult{}, &core.ProviderError{
			Message: message, Class: core.ProviderErrorUnsupported, Classification: core.ProviderErrorClassification{FailoverEligible: true},
			Cause: &InvocationError{Msg: message, FailoverEligible: true},
		}
	}
	usage, _ := decoded["usageMetadata"].(map[string]any)
	return core.GenerateImagesResult{Images: images, Usage: usage}, nil
}

// googleChatCall is a Chat Completions request mapped for Google.
type googleChatCall struct {
	payload map[string]any
	losses  []core.Loss
	// includeUsage asks a stream for a usage chunk before it ends, as
	// stream_options' include_usage does.
	includeUsage bool
}

// googleChatPayload maps a Chat Completions body to the generateContent
// body Google sends for it. The text of the messages, max_tokens and
// temperature are mapped as the gateway maps them. Function tools become
// function declarations and tool_choice the function calling config, an
// assistant's tool calls become functionCall parts and tool results
// functionResponse parts; see googleChatContents. Every other field is
// dropped and reported at its path: a field that changes the structure of
// the answer unless it asks for what Google does anyway as a material loss,
// and the rest as advisory ones. A stream honours stream_options'
// include_usage, so a stream does not report it.
func googleChatPayload(request core.Request, stream bool) (googleChatCall, error) {
	body, err := googleJSONBody(request)
	if err != nil {
		return googleChatCall{}, err
	}
	messages, err := googleChatMessages(body["messages"])
	if err != nil {
		return googleChatCall{}, err
	}
	contents, system, contentLosses, err := googleChatContents(messages)
	if err != nil {
		return googleChatCall{}, err
	}
	payload := googleGenerateRequest(contents, system, body["max_tokens"], body["temperature"], nil)
	declarations, toolLosses := googleFunctionDeclarations(body["tools"])
	if len(declarations) > 0 {
		payload["tools"] = []map[string]any{{"functionDeclarations": declarations}}
	}
	config, choiceLosses := googleFunctionCallingConfig(body["tool_choice"], declarations)
	if config != nil {
		payload["toolConfig"] = map[string]any{"functionCallingConfig": config}
	}
	losses := slices.Concat(googleChatFieldLosses(body, stream), googleChatMessageLosses(messages), contentLosses, toolLosses, choiceLosses)
	options, _ := body["stream_options"].(map[string]any)
	usage, _ := options["include_usage"].(bool)
	return googleChatCall{payload: payload, losses: translate.NewReport(losses...).Losses, includeUsage: stream && usage}, nil
}

// googleChatMessages reads the messages as the gateway's Chat facade
// decodes them: an array of objects, where null is an empty message.
func googleChatMessages(value any) ([]map[string]any, error) {
	if value == nil {
		return nil, nil
	}
	list, ok := value.([]any)
	if !ok {
		return nil, &core.ProviderError{Message: "the Chat messages must be an array", Class: core.ProviderErrorInvalidRequest}
	}
	messages := make([]map[string]any, 0, len(list))
	for _, raw := range list {
		message, ok := raw.(map[string]any)
		if !ok && raw != nil {
			return nil, &core.ProviderError{Message: "each Chat message must be an object", Class: core.ProviderErrorInvalidRequest}
		}
		messages = append(messages, message)
	}
	return messages, nil
}

// googleChatFieldLosses reports the Chat fields Google drops, in order, so
// the losses do not depend on map order. Tools and tool_choice are mapped,
// and googleFunctionDeclarations and googleFunctionCallingConfig report
// what of them is not.
func googleChatFieldLosses(body map[string]any, stream bool) []core.Loss {
	var losses []core.Loss
	for _, field := range slices.Sorted(maps.Keys(body)) {
		value := body[field]
		switch field {
		case "model", "stream", "messages", "max_tokens", "temperature", "tools", "tool_choice":
			// The request's model and the operation decide the first two,
			// and Google maps the rest.
			continue
		case "stream_options":
			if stream && googleStreamOptionsServed(value) {
				continue
			}
		}
		if value == nil {
			continue
		}
		loss := core.Loss{Path: field, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: "Google does not send this Chat field"}
		if structuralChatField(field) {
			if chatFieldDefault(field, value) {
				loss.Detail = "Google already behaves as this value asks"
			} else {
				loss.Severity = translate.LossMaterial
			}
		}
		losses = append(losses, loss)
	}
	return losses
}

// googleStreamOptionsServed reports stream options a Google stream serves
// in full: none, or include_usage alone.
func googleStreamOptionsServed(value any) bool {
	options, ok := value.(map[string]any)
	if !ok {
		return value == nil
	}
	for name := range options {
		if name != "include_usage" {
			return false
		}
	}
	return true
}

// googleChatMessageLosses reports what Google drops from each message. An
// assistant's tool calls and a tool result's tool_call_id are mapped, and
// googleChatContents reports what of them is not, as it reports a tool
// result's content.
func googleChatMessageLosses(messages []map[string]any) []core.Loss {
	var losses []core.Loss
	for index, message := range messages {
		role, _ := message["role"].(string)
		for _, field := range slices.Sorted(maps.Keys(message)) {
			value := message[field]
			switch {
			case field == "role" || value == nil:
				continue
			case role == "assistant" && field == "tool_calls", role == "tool" && (field == "tool_call_id" || field == "content"):
				continue
			}
			path := "messages." + strconv.Itoa(index) + "." + field
			if field == "content" {
				if _, text := value.(string); !text {
					losses = append(losses, core.Loss{Path: path, Class: translate.LossDropped, Severity: translate.LossMaterial,
						Detail: "Google receives only text content, so this content is sent as empty text"})
				}
				continue
			}
			severity := translate.LossAdvisory
			switch field {
			case "tool_calls", "function_call", "tool_call_id":
				severity = translate.LossMaterial
			}
			losses = append(losses, core.Loss{Path: path, Class: translate.LossDropped, Severity: severity, Detail: "Google does not send this message field"})
		}
	}
	return losses
}

// googleSkipSignature is the thought signature Google documents for a
// function call the model did not make, or whose signature is lost: it
// skips the check that would refuse the call without one, at some cost to
// the answer's quality.
const googleSkipSignature = "skip_thought_signature_validator"

// googleChatContents maps messages to Gemini's contents, as the gateway
// does for text: system and developer text becomes the system instruction,
// since Gemini has no system role, an assistant is the model, and every
// other message keeps its role, or is the user's when it has none, with its
// string content as one text part.
//
// An assistant's tool calls become functionCall parts after its text, each
// with the arguments it holds as a JSON object and the thought signature it
// carries at extra_content.google.thought_signature, function or call level.
// When no call of a message carries one, its first call is sent with
// Google's skip signature, since Gemini 3 refuses a function call without
// a signature. A tool result becomes a functionResponse part naming the
// function of the call its tool_call_id answers: its text as the JSON object
// it holds, or else as {"result": text}. The results that follow one another
// share one user turn, as Gemini wants every response to a turn's calls
// together. A tool result that answers no earlier call cannot be named, so
// it is refused as an invalid request.
func googleChatContents(messages []map[string]any) ([]map[string]any, []map[string]any, []core.Loss, error) {
	contents := make([]map[string]any, 0, len(messages))
	var system []map[string]any
	var losses []core.Loss
	// names holds the function of each tool call made so far, by call ID;
	// "" for a call that could not be sent.
	names := map[string]string{}
	// results is the turn collecting consecutive tool results, if any.
	var results map[string]any
	for index, message := range messages {
		role, _ := message["role"].(string)
		text, _ := message["content"].(string)
		path := "messages." + strconv.Itoa(index)
		switch role {
		case "system", "developer":
			system = append(system, map[string]any{"text": text})
			continue
		case "tool":
			part, partLosses, err := googleFunctionResponse(path, message, names)
			if err != nil {
				return nil, nil, nil, err
			}
			losses = append(losses, partLosses...)
			if part == nil {
				continue
			}
			if results == nil {
				results = map[string]any{"role": "user", "parts": []map[string]any{}}
				contents = append(contents, results)
			}
			results["parts"] = append(results["parts"].([]map[string]any), part)
			continue
		}
		results = nil
		parts := []map[string]any{{"text": text}}
		switch role {
		case "assistant":
			role = "model"
			calls, callLosses := googleFunctionCalls(path, message["tool_calls"], names)
			losses = append(losses, callLosses...)
			if len(calls) > 0 {
				if text == "" {
					parts = parts[:0]
				}
				parts = append(parts, calls...)
			}
		case "":
			role = "user"
		}
		contents = append(contents, map[string]any{"role": role, "parts": parts})
	}
	return contents, system, losses, nil
}

// googleFunctionCalls maps an assistant's tool calls to functionCall parts
// and records the function of each by its ID in names. A call that is not a
// function call with a name cannot be sent, and arguments that are not a
// JSON object are sent as an empty one; both are material losses.
func googleFunctionCalls(path string, value any, names map[string]string) ([]map[string]any, []core.Loss) {
	calls, ok := value.([]any)
	if !ok {
		if value == nil {
			return nil, nil
		}
		return nil, []core.Loss{{Path: path + ".tool_calls", Class: translate.LossDropped, Severity: translate.LossMaterial,
			Detail: "the tool calls are not a list, so Google cannot send them"}}
	}
	var parts []map[string]any
	var losses []core.Loss
	signed := false
	for index, raw := range calls {
		callPath := path + ".tool_calls." + strconv.Itoa(index)
		call, _ := raw.(map[string]any)
		function, _ := call["function"].(map[string]any)
		name, _ := function["name"].(string)
		id, _ := call["id"].(string)
		if kind, set := call["type"]; (set && kind != "function") || strings.TrimSpace(name) == "" {
			if id != "" {
				names[id] = ""
			}
			losses = append(losses, core.Loss{Path: callPath, Class: translate.LossDropped, Severity: translate.LossMaterial,
				Detail: "Google sends only function tool calls with a name"})
			continue
		}
		if id != "" {
			names[id] = name
		}
		args, ok := googleCallArguments(function["arguments"])
		if !ok {
			losses = append(losses, core.Loss{Path: callPath + ".function.arguments", Class: translate.LossDropped, Severity: translate.LossMaterial,
				Detail: "Google takes a tool call's arguments as a JSON object, so these are sent as an empty one"})
		}
		part := map[string]any{"functionCall": map[string]any{"name": name, "args": args}}
		if signature := googleCallSignature(call, function); signature != "" {
			part["thoughtSignature"] = signature
			signed = true
		}
		parts = append(parts, part)
	}
	if !signed && len(parts) > 0 {
		parts[0]["thoughtSignature"] = googleSkipSignature
	}
	return parts, losses
}

// googleCallArguments decodes a tool call's arguments to the JSON object
// Gemini takes, numbers kept as written: a string holding a JSON object, an
// object, or nothing, which is an empty object. It reports false for
// anything else, which is sent as an empty object.
func googleCallArguments(value any) (map[string]any, bool) {
	switch arguments := value.(type) {
	case nil:
		return map[string]any{}, true
	case map[string]any:
		return arguments, true
	case string:
		if strings.TrimSpace(arguments) == "" {
			return map[string]any{}, true
		}
		if object, ok := googleJSONObject(arguments); ok {
			return object, true
		}
	}
	return map[string]any{}, false
}

// googleJSONObject decodes text that holds exactly one JSON object, with
// its numbers kept as written.
func googleJSONObject(text string) (map[string]any, bool) {
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.UseNumber()
	var object map[string]any
	if decoder.Decode(&object) != nil || object == nil {
		return nil, false
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, false
	}
	return object, true
}

// googleCallSignature returns the thought signature a tool call carries, at
// the first of the places products keep it: extra_content.google, the
// function, or the call itself.
func googleCallSignature(call, function map[string]any) string {
	extra, _ := call["extra_content"].(map[string]any)
	google, _ := extra["google"].(map[string]any)
	for _, holder := range []map[string]any{google, function, call} {
		if signature, _ := holder["thought_signature"].(string); signature != "" {
			return signature
		}
	}
	return ""
}

// googleFunctionResponse maps a tool result to the functionResponse part
// that answers the call its tool_call_id names. The result of a call that
// could not be sent is dropped as a material loss, and nil is returned.
func googleFunctionResponse(path string, message map[string]any, names map[string]string) (map[string]any, []core.Loss, error) {
	id, _ := message["tool_call_id"].(string)
	name, known := names[id]
	switch {
	case !known:
		return nil, nil, &core.ProviderError{
			Message: path + ": the tool result answers no tool call of an earlier assistant message", Class: core.ProviderErrorInvalidRequest,
		}
	case name == "":
		return nil, []core.Loss{{Path: path, Class: translate.LossDropped, Severity: translate.LossMaterial,
			Detail: "the tool call this result answers was not sent, so Google does not send it either"}}, nil
	}
	text, losses := googleToolResultText(path+".content", message["content"])
	response, ok := googleJSONObject(text)
	if !ok {
		response = map[string]any{"result": text}
	}
	return map[string]any{"functionResponse": map[string]any{"name": name, "response": response}}, losses, nil
}

// googleToolResultText is the text of a tool result's content: a string,
// or the text of its text parts. Any other part, or content of another
// kind, is a material loss.
func googleToolResultText(path string, content any) (string, []core.Loss) {
	lost := func(path string) core.Loss {
		return core.Loss{Path: path, Class: translate.LossDropped, Severity: translate.LossMaterial, Detail: "Google receives only the text of a tool result"}
	}
	switch value := content.(type) {
	case nil:
		return "", nil
	case string:
		return value, nil
	case []any:
		var text strings.Builder
		var losses []core.Loss
		for index, raw := range value {
			part, _ := raw.(map[string]any)
			if part["type"] != "text" {
				losses = append(losses, lost(path+"."+strconv.Itoa(index)))
				continue
			}
			value, _ := part["text"].(string)
			text.WriteString(value)
		}
		return text.String(), losses
	}
	return "", []core.Loss{lost(path)}
}

// googleFunctionDeclarations maps Chat function tools to Gemini function
// declarations: the name, the description and the parameters, sent as the
// JSON Schema the tool holds. A tool that is not a function with a name
// cannot be sent, a material loss, and strict is dropped as an advisory one,
// since Google does not enforce the schema.
func googleFunctionDeclarations(value any) ([]map[string]any, []core.Loss) {
	tools, ok := value.([]any)
	if !ok {
		if value == nil {
			return nil, nil
		}
		return nil, []core.Loss{{Path: "tools", Class: translate.LossDropped, Severity: translate.LossMaterial,
			Detail: "the tools are not a list, so Google cannot send them"}}
	}
	var declarations []map[string]any
	var losses []core.Loss
	for index, raw := range tools {
		path := "tools." + strconv.Itoa(index)
		tool, _ := raw.(map[string]any)
		function, _ := tool["function"].(map[string]any)
		name, _ := function["name"].(string)
		if kind, set := tool["type"]; (set && kind != "function") || strings.TrimSpace(name) == "" {
			losses = append(losses, core.Loss{Path: path, Class: translate.LossDropped, Severity: translate.LossMaterial,
				Detail: "Google sends only function tools with a name, so the model cannot call this one"})
			continue
		}
		declaration := map[string]any{"name": name}
		if description, _ := function["description"].(string); description != "" {
			declaration["description"] = description
		}
		switch parameters := function["parameters"].(type) {
		case nil:
		case map[string]any:
			declaration["parametersJsonSchema"] = parameters
		default:
			losses = append(losses, core.Loss{Path: path + ".function.parameters", Class: translate.LossDropped, Severity: translate.LossMaterial,
				Detail: "the parameters are not a JSON Schema object, so Google declares the function without them"})
		}
		if strict, _ := function["strict"].(bool); strict {
			losses = append(losses, core.Loss{Path: path + ".function.strict", Class: translate.LossDropped, Severity: translate.LossAdvisory,
				Detail: "Google does not enforce the schema of a tool's arguments"})
		}
		declarations = append(declarations, declaration)
	}
	return declarations, losses
}

// googleFunctionCallingConfig maps tool_choice to Gemini's function calling
// config: auto is Gemini's default, none is NONE, required is ANY, and a
// named function is ANY limited to it. A choice that calls a function needs
// a declared one, and any other choice cannot be sent; both are material
// losses.
func googleFunctionCallingConfig(value any, declarations []map[string]any) (map[string]any, []core.Loss) {
	lost := func(detail string) []core.Loss {
		return []core.Loss{{Path: "tool_choice", Class: translate.LossDropped, Severity: translate.LossMaterial, Detail: detail}}
	}
	var config map[string]any
	switch choice := value.(type) {
	case nil:
		return nil, nil
	case string:
		switch choice {
		case "auto":
			return nil, nil
		case "none":
			if len(declarations) == 0 {
				return nil, nil
			}
			return map[string]any{"mode": "NONE"}, nil
		case "required":
			config = map[string]any{"mode": "ANY"}
		}
	case map[string]any:
		function, _ := choice["function"].(map[string]any)
		if name, _ := function["name"].(string); choice["type"] == "function" && name != "" {
			if !slices.ContainsFunc(declarations, func(declaration map[string]any) bool { return declaration["name"] == name }) {
				return nil, lost("the tool choice names a function Google does not send")
			}
			config = map[string]any{"mode": "ANY", "allowedFunctionNames": []string{name}}
		}
	}
	switch {
	case config == nil:
		return nil, lost("Google cannot send this tool choice")
	case len(declarations) == 0:
		return nil, lost("the tool choice asks for a tool call, but Google sends no tool")
	}
	return config, nil
}

// googleContentRequest maps text messages to a generateContent body, as
// googleChatContents maps them, with maxTokens, temperature and the
// response modalities.
func googleContentRequest(messages []map[string]any, maxTokens, temperature any, modalities []string) map[string]any {
	contents, system, _, _ := googleChatContents(messages)
	return googleGenerateRequest(contents, system, maxTokens, temperature, modalities)
}

// googleGenerateRequest assembles a generateContent body. maxTokens and
// temperature pass through as the request holds them.
func googleGenerateRequest(contents, system []map[string]any, maxTokens, temperature any, modalities []string) map[string]any {
	request := map[string]any{"contents": contents}
	if len(system) > 0 {
		request["systemInstruction"] = map[string]any{"parts": system}
	}
	generation := map[string]any{}
	if maxTokens != nil {
		generation["maxOutputTokens"] = maxTokens
	}
	if temperature != nil {
		generation["temperature"] = temperature
	}
	if len(modalities) > 0 {
		generation["responseModalities"] = modalities
	}
	if len(generation) > 0 {
		request["generationConfig"] = generation
	}
	return request
}

// googleToOpenAIChat reshapes a generateContent answer into the Chat
// Completions envelope the gateway returns, with the tool calls of its
// functionCall parts and the finish reason googleFinishReason names.
func googleToOpenAIChat(model string, decoded map[string]any) map[string]any {
	parts := googleParts(decoded)
	message := map[string]any{"role": "assistant", "content": googleText(parts)}
	calls := googleToolCalls(parts, googleResponseID(decoded), 0)
	if len(calls) > 0 {
		message["tool_calls"] = calls
		if message["content"] == "" {
			message["content"] = nil
		}
	}
	inputTokens, outputTokens, totalTokens := googleUsage(decoded)
	served := model
	if value, ok := decoded["modelVersion"].(string); ok && value != "" {
		served = value
	}
	return map[string]any{
		"id":     "chatcmpl-google",
		"object": "chat.completion",
		"model":  served,
		"choices": []any{map[string]any{
			"index":         0,
			"finish_reason": googleFinishReason(googleCandidateFinish(decoded), len(calls) > 0),
			"message":       message,
		}},
		"usage": map[string]any{
			"prompt_tokens": inputTokens, "completion_tokens": outputTokens, "total_tokens": totalTokens,
		},
	}
}

// googleText joins the text of parts, but for thought summaries, which are
// the model's reasoning rather than its answer.
func googleText(parts []map[string]any) string {
	var text strings.Builder
	for _, part := range parts {
		if thought, _ := part["thought"].(bool); thought {
			continue
		}
		if value, ok := part["text"].(string); ok {
			text.WriteString(value)
		}
	}
	return text.String()
}

// googleToolCalls maps the functionCall parts of parts to Chat tool calls,
// numbered from first, each with its arguments as a JSON string and the
// thought signature its part carries at extra_content.google, where a
// client returns it on the next turn. A call keeps the ID Gemini gave it;
// otherwise its ID is unique to the response.
func googleToolCalls(parts []map[string]any, responseID string, first int) []map[string]any {
	var calls []map[string]any
	for _, part := range parts {
		function, ok := part["functionCall"].(map[string]any)
		if !ok {
			continue
		}
		name, _ := function["name"].(string)
		arguments := "{}"
		if args, ok := function["args"].(map[string]any); ok {
			if encoded, err := json.Marshal(args); err == nil {
				arguments = string(encoded)
			}
		}
		id, _ := function["id"].(string)
		if id == "" {
			id = googleCallID(responseID, first+len(calls))
		}
		call := map[string]any{"id": id, "type": "function", "function": map[string]any{"name": name, "arguments": arguments}}
		if signature, _ := part["thoughtSignature"].(string); signature != "" {
			call["extra_content"] = map[string]any{"google": map[string]any{"thought_signature": signature}}
		}
		calls = append(calls, call)
	}
	return calls
}

// googleResponseID is the ID Gemini gave its answer, or "".
func googleResponseID(decoded map[string]any) string {
	id, _ := decoded["responseId"].(string)
	return id
}

// googleCallID names a tool call Gemini gave no ID: by the answer's ID and
// the call's number, so that calls are named apart across the turns of a
// conversation, or at random when the answer has no ID.
func googleCallID(responseID string, index int) string {
	clean := strings.Map(func(r rune) rune {
		if r == '-' || r == '_' || (r >= '0' && r <= '9') || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			return r
		}
		return -1
	}, responseID)
	if clean == "" {
		random := make([]byte, 12)
		_, _ = rand.Read(random)
		clean = hex.EncodeToString(random)
	}
	return "call_" + clean + "_" + strconv.Itoa(index)
}

// googleCandidateFinish is the finish reason of the first candidate, or "".
func googleCandidateFinish(decoded map[string]any) string {
	candidates, _ := decoded["candidates"].([]any)
	if len(candidates) == 0 {
		return ""
	}
	first, _ := candidates[0].(map[string]any)
	reason, _ := first["finishReason"].(string)
	return reason
}

// googleFinishReason names Gemini's finish reason as Chat does: tool_calls
// for an answer that calls one, length when the output limit ended it,
// content_filter when a safety or policy check did, and otherwise stop.
func googleFinishReason(reason string, calls bool) string {
	switch {
	case calls:
		return "tool_calls"
	case reason == "MAX_TOKENS":
		return "length"
	}
	switch reason {
	case "SAFETY", "RECITATION", "LANGUAGE", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "MODEL_ARMOR",
		"IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION":
		return "content_filter"
	}
	return "stop"
}

// googleParts returns the parts of the first candidate.
func googleParts(decoded map[string]any) []map[string]any {
	candidates, _ := decoded["candidates"].([]any)
	if len(candidates) == 0 {
		return nil
	}
	first, _ := candidates[0].(map[string]any)
	content, _ := first["content"].(map[string]any)
	rawParts, _ := content["parts"].([]any)
	parts := make([]map[string]any, 0, len(rawParts))
	for _, raw := range rawParts {
		if part, ok := raw.(map[string]any); ok {
			parts = append(parts, part)
		}
	}
	return parts
}

// googleUsage reads usageMetadata. Thinking tokens are output the caller
// pays for, so they count as completion tokens.
func googleUsage(decoded map[string]any) (int, int, int) {
	usage, _ := decoded["usageMetadata"].(map[string]any)
	number := func(key string) int {
		if value, ok := usage[key].(float64); ok {
			return int(value)
		}
		return 0
	}
	input := number("promptTokenCount")
	output := number("candidatesTokenCount") + number("thoughtsTokenCount")
	total := number("totalTokenCount")
	if total == 0 {
		total = input + output
	}
	return input, output, total
}

// googleEmptyReplyError explains an answer that holds no text, image or
// tool call. Gemini's thinking models spend maxOutputTokens on reasoning
// before they answer, so a small max_tokens yields a 200 with an empty
// string, which a caller cannot tell from success. The diagnostic names the
// cause, as the gateway's does: the reasoning, or the finish reason, or the
// reason Google blocked the prompt. Another target may still answer, so it
// permits failover; the upstream is healthy, so it counts against nothing.
func googleEmptyReplyError(label string, decoded map[string]any) error {
	for _, part := range googleParts(decoded) {
		if thought, _ := part["thought"].(bool); thought {
			continue
		}
		if text, ok := part["text"].(string); ok && strings.TrimSpace(text) != "" {
			return nil
		}
		if _, ok := part["inlineData"]; ok {
			return nil
		}
		if _, ok := part["functionCall"]; ok {
			return nil
		}
	}
	usage, _ := decoded["usageMetadata"].(map[string]any)
	thoughts, _ := usage["thoughtsTokenCount"].(float64)
	finish := googleCandidateFinish(decoded)
	if feedback, _ := decoded["promptFeedback"].(map[string]any); finish == "" {
		finish, _ = feedback["blockReason"].(string)
	}
	diagnostic := label + ": the model returned no text"
	message := diagnostic
	switch {
	case thoughts > 0:
		diagnostic = fmt.Sprintf("%s: the model spent its entire output budget on reasoning (%d thinking tokens) and returned no text — raise max_tokens or omit it",
			label, int(thoughts))
		message = diagnostic
	case finish != "" && finish != "STOP":
		diagnostic = fmt.Sprintf("%s: the model returned no text (finish reason %s)", label, finish)
		// A finish reason is one of Google's enum values; anything else is
		// upstream text, which stays out of the message.
		if codexErrorIdentifier.MatchString(finish) {
			message = diagnostic
		}
	}
	return &core.ProviderError{
		Message: message, Class: core.ProviderErrorUpstream,
		Classification: core.ProviderErrorClassification{FailoverEligible: true},
		Cause:          &InvocationError{Msg: sanitizeDiagnosticTextLimit(diagnostic, diagnosticErrorLimit), FailoverEligible: true},
	}
}
