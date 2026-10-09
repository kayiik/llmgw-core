package providers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"regexp"
	"testing"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

const googleWeatherTool = `{"type":"function","function":{"name":"get_weather","description":"Current weather for a city",` +
	`"parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}`

const googleWeatherDeclaration = `"tools":[{"functionDeclarations":[{"description":"Current weather for a city","name":"get_weather",` +
	`"parametersJsonSchema":{"additionalProperties":false,"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"}}]}]`

// googleChatBody sends body through Invoke and returns the generateContent
// body Google received and the losses reported.
func googleChatBody(t *testing.T, body string) (string, []core.Loss) {
	t.Helper()
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	response, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", body, googleKey("k")))
	if err != nil {
		t.Fatal(err)
	}
	calls := fake.take()
	if len(calls) != 1 {
		t.Fatalf("upstream = %+v", calls)
	}
	return calls[0].body, response.Losses
}

func googleLoss(path string, severity translate.LossSeverity, detail string) core.Loss {
	return core.Loss{Path: path, Class: translate.LossDropped, Severity: severity, Detail: detail}
}

// Function tools are declared with the JSON Schema they hold, which Gemini
// takes whole, additionalProperties included. strict is dropped, since
// Google does not enforce the schema, and a tool that is not a function
// cannot be sent.
func TestGoogleChatDeclaresFunctionTools(t *testing.T) {
	t.Parallel()
	body, losses := googleChatBody(t, `{"messages":[{"role":"user","content":"hi"}],"tools":[`+
		`{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","strict":true,`+
		`"parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}},`+
		`{"type":"function","function":{"name":"ping"}},{"type":"custom","custom":{"name":"grammar"}},`+
		`{"type":"function","function":{"name":"bad","parameters":"not a schema"}}]}`)
	want := `{"contents":[{"parts":[{"text":"hi"}],"role":"user"}],"tools":[{"functionDeclarations":[` +
		`{"description":"Current weather for a city","name":"get_weather","parametersJsonSchema":{"additionalProperties":false,"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"}},` +
		`{"name":"ping"},{"name":"bad"}]}]}`
	if body != want {
		t.Fatalf("body = %s\nwant   %s", body, want)
	}
	if want := []core.Loss{
		googleLoss("tools.0.function.strict", translate.LossAdvisory, "Google does not enforce the schema of a tool's arguments"),
		googleLoss("tools.2", translate.LossMaterial, "Google sends only function tools with a name, so the model cannot call this one"),
		googleLoss("tools.3.function.parameters", translate.LossMaterial, "the parameters are not a JSON Schema object, so Google declares the function without them"),
	}; !reflect.DeepEqual(losses, want) {
		t.Fatalf("losses = %+v", losses)
	}
}

// tool_choice maps to Gemini's function calling config. A choice that asks
// for a call Google cannot make, or that Gemini cannot express, is a
// material loss.
func TestGoogleChatMapsToolChoice(t *testing.T) {
	t.Parallel()
	const prefix = `{"contents":[{"parts":[{"text":"hi"}],"role":"user"}]`
	for _, testCase := range []struct {
		name, tools, choice, config string
		loss                        string
	}{
		{name: "auto", tools: googleWeatherTool, choice: `"auto"`},
		{name: "none", tools: googleWeatherTool, choice: `"none"`, config: `"toolConfig":{"functionCallingConfig":{"mode":"NONE"}}`},
		{name: "required", tools: googleWeatherTool, choice: `"required"`, config: `"toolConfig":{"functionCallingConfig":{"mode":"ANY"}}`},
		{
			name: "named", tools: googleWeatherTool, choice: `{"type":"function","function":{"name":"get_weather"}}`,
			config: `"toolConfig":{"functionCallingConfig":{"allowedFunctionNames":["get_weather"],"mode":"ANY"}}`,
		},
		{
			name: "named but not declared", tools: googleWeatherTool, choice: `{"type":"function","function":{"name":"get_time"}}`,
			loss: "the tool choice names a function Google does not send",
		},
		{
			name: "allowed tools", tools: googleWeatherTool, choice: `{"type":"allowed_tools","allowed_tools":{"mode":"auto","tools":[]}}`,
			loss: "Google cannot send this tool choice",
		},
		{name: "required without tools", choice: `"required"`, loss: "the tool choice asks for a tool call, but Google sends no tool"},
		{name: "none without tools", choice: `"none"`},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			request := `{"messages":[{"role":"user","content":"hi"}],"tool_choice":` + testCase.choice
			want := prefix
			if testCase.tools != "" {
				request += `,"tools":[` + testCase.tools + `]`
			}
			if testCase.config != "" {
				want += "," + testCase.config
			}
			if testCase.tools != "" {
				want += "," + googleWeatherDeclaration
			}
			body, losses := googleChatBody(t, request+`}`)
			if body != want+"}" {
				t.Fatalf("body = %s\nwant   %s}", body, want)
			}
			var wantLosses []core.Loss
			if testCase.loss != "" {
				wantLosses = []core.Loss{googleLoss("tool_choice", translate.LossMaterial, testCase.loss)}
			}
			if !reflect.DeepEqual(losses, wantLosses) {
				t.Fatalf("losses = %+v", losses)
			}
		})
	}
}

// An assistant's tool calls reach Gemini as functionCall parts with the
// thought signature each carries, wherever the client kept it, and with
// Google's skip signature on the first call of a turn whose calls lost
// theirs. Tool results reach it as functionResponse parts naming their
// call's function, each run of them in one user turn: a JSON object as it
// is, numbers as written, and anything else as {"result": text}.
func TestGoogleChatSendsToolHistoryWithItsSignatures(t *testing.T) {
	t.Parallel()
	call := func(id, arguments, signature string) string {
		return `{"id":"` + id + `","type":"function","function":{"name":"get_weather","arguments":` + arguments + signature + `}}`
	}
	body, losses := googleChatBody(t, `{"tools":[`+googleWeatherTool+`],"messages":[`+
		`{"role":"user","content":"Weather in Paris and Rome?"},`+
		`{"role":"assistant","content":null,"tool_calls":[`+
		`{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"},"extra_content":{"google":{"thought_signature":"sig-google"}}},`+
		call("c2", `"{\"city\":\"Rome\"}"`, "")+`]},`+
		`{"role":"tool","tool_call_id":"c1","content":"{\"temperature\":12345678901234567890}"},`+
		`{"role":"system","content":"between results"},`+
		`{"role":"tool","tool_call_id":"c2","content":"plain"},`+
		`{"role":"user","content":"And Oslo?"},`+
		`{"role":"assistant","content":"Let me see.","tool_calls":[`+call("c3", `"{\"city\":\"Oslo\"}"`, `,"thought_signature":"sig-function"`)+`]},`+
		`{"role":"tool","tool_call_id":"c3","content":"[1,2]"},`+
		`{"role":"assistant","tool_calls":[{"id":"c4","type":"function","function":{"name":"get_weather","arguments":"{}"},"thought_signature":"sig-call"}]},`+
		`{"role":"tool","tool_call_id":"c4","content":null},`+
		`{"role":"assistant","content":"","tool_calls":[`+call("c5", `""`, "")+`,`+call("c6", `"not json"`, "")+`]},`+
		`{"role":"tool","tool_call_id":"c5","content":[{"type":"text","text":"sunny"},{"type":"image_url","image_url":{"url":"data:,"}}]},`+
		`{"role":"tool","tool_call_id":"c6","content":"windy"}]}`)
	functionCall := func(args, signature string) string {
		part := `{"functionCall":{"args":` + args + `,"name":"get_weather"}`
		if signature != "" {
			part += `,"thoughtSignature":"` + signature + `"`
		}
		return part + "}"
	}
	response := func(value string) string {
		return `{"functionResponse":{"name":"get_weather","response":` + value + `}}`
	}
	want := `{"contents":[` +
		`{"parts":[{"text":"Weather in Paris and Rome?"}],"role":"user"},` +
		`{"parts":[` + functionCall(`{"city":"Paris"}`, "sig-google") + `,` + functionCall(`{"city":"Rome"}`, "") + `],"role":"model"},` +
		`{"parts":[` + response(`{"temperature":12345678901234567890}`) + `,` + response(`{"result":"plain"}`) + `],"role":"user"},` +
		`{"parts":[{"text":"And Oslo?"}],"role":"user"},` +
		`{"parts":[{"text":"Let me see."},` + functionCall(`{"city":"Oslo"}`, "sig-function") + `],"role":"model"},` +
		`{"parts":[` + response(`{"result":"[1,2]"}`) + `],"role":"user"},` +
		`{"parts":[` + functionCall(`{}`, "sig-call") + `],"role":"model"},` +
		`{"parts":[` + response(`{"result":""}`) + `],"role":"user"},` +
		`{"parts":[` + functionCall(`{}`, googleSkipSignature) + `,` + functionCall(`{}`, "") + `],"role":"model"},` +
		`{"parts":[` + response(`{"result":"sunny"}`) + `,` + response(`{"result":"windy"}`) + `],"role":"user"}],` +
		`"systemInstruction":{"parts":[{"text":"between results"}]},` + googleWeatherDeclaration + `}`
	if body != want {
		t.Fatalf("body = %s\nwant   %s", body, want)
	}
	if want := []core.Loss{
		googleLoss("messages.10.tool_calls.1.function.arguments", translate.LossMaterial,
			"Google takes a tool call's arguments as a JSON object, so these are sent as an empty one"),
		googleLoss("messages.11.content.1", translate.LossMaterial, "Google receives only the text of a tool result"),
	}; !reflect.DeepEqual(losses, want) {
		t.Fatalf("losses = %+v", losses)
	}
}

// Gemini needs the function a tool result answers, so a result whose call
// is nowhere in the history is refused before anything is sent.
func TestGoogleChatRefusesAToolResultWithoutItsCall(t *testing.T) {
	t.Parallel()
	fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
	provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
	for _, body := range []string{
		`{"messages":[{"role":"user","content":"hi"},{"role":"tool","tool_call_id":"c1","content":"r"}]}`,
		`{"messages":[{"role":"tool","content":"r"},{"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"f"}}]}]}`,
	} {
		_, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", body, googleKey("k")))
		var failure *core.ProviderError
		if !errors.As(err, &failure) || failure.Class != core.ProviderErrorInvalidRequest || core.ClassifyError(err).Disposition() != core.DispositionTerminal {
			t.Fatalf("%s: error = %v, want an invalid request", body, err)
		}
	}
	if calls := fake.take(); len(calls) != 0 {
		t.Fatalf("upstream = %+v", calls)
	}
}

// Gemini's function calls come back as Chat tool calls: the ID Gemini gave
// a call, or one unique to the answer, the arguments as a JSON string, and
// the thought signature at extra_content.google, where a client returns it.
// The finish reason is tool_calls, and an answer of calls alone has no
// content. Thought summaries are not the answer.
func TestGoogleChatReturnsToolCalls(t *testing.T) {
	t.Parallel()
	answer := func(t *testing.T, decoded string) map[string]any {
		t.Helper()
		_, base := newGoogleFake(t, googleAnswer(http.StatusOK, decoded))
		provider := newGoogleTest(t, GoogleConfig{Deployment: GoogleAIStudio, BaseURL: base})
		response, err := provider.Invoke(context.Background(), googleChatRequest("gemini-3.5-flash", `{"messages":[{"role":"user","content":"hi"}]}`, googleKey("k")))
		if err != nil {
			t.Fatal(err)
		}
		var completion map[string]any
		if err := json.Unmarshal(response.Body, &completion); err != nil {
			t.Fatal(err)
		}
		return completion["choices"].([]any)[0].(map[string]any)
	}
	choice := answer(t, `{"candidates":[{"content":{"role":"model","parts":[{"text":"Weighing it up.","thought":true},{"text":"Checking."},`+
		`{"functionCall":{"id":"fc-gemini","name":"get_weather","args":{"city":"Paris"}},"thoughtSignature":"sig-1"},`+
		`{"functionCall":{"name":"get_time","args":{}}}]},"finishReason":"STOP"}],"responseId":"resp-fixture"}`)
	want := map[string]any{
		"index": float64(0), "finish_reason": "tool_calls",
		"message": map[string]any{"role": "assistant", "content": "Checking.", "tool_calls": []any{
			map[string]any{
				"id": "fc-gemini", "type": "function", "function": map[string]any{"name": "get_weather", "arguments": `{"city":"Paris"}`},
				"extra_content": map[string]any{"google": map[string]any{"thought_signature": "sig-1"}},
			},
			map[string]any{"id": "call_resp-fixture_1", "type": "function", "function": map[string]any{"name": "get_time", "arguments": `{}`}},
		}},
	}
	if !reflect.DeepEqual(choice, want) {
		t.Fatalf("choice = %+v", choice)
	}
	choice = answer(t, `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_time"}}]},"finishReason":"STOP"}]}`)
	message := choice["message"].(map[string]any)
	calls := message["tool_calls"].([]any)
	id := calls[0].(map[string]any)["id"].(string)
	if message["content"] != nil || len(calls) != 1 || !regexp.MustCompile(`^call_[0-9a-f]{24}_0$`).MatchString(id) {
		t.Fatalf("message = %+v", message)
	}
}

// The finish reason follows Gemini's: the output limit is length, and a
// safety or policy stop is content_filter.
func TestGoogleChatNamesTheFinishReason(t *testing.T) {
	t.Parallel()
	for reason, want := range map[string]string{
		"": "stop", "STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter", "RECITATION": "content_filter",
		"PROHIBITED_CONTENT": "content_filter", "OTHER": "stop", "MALFORMED_FUNCTION_CALL": "stop",
	} {
		decoded := map[string]any{"candidates": []any{map[string]any{
			"finishReason": reason, "content": map[string]any{"parts": []any{map[string]any{"text": "partial"}}},
		}}}
		choice := googleToOpenAIChat("m", decoded)["choices"].([]any)[0].(map[string]any)
		if choice["finish_reason"] != want {
			t.Errorf("%q: finish_reason = %v, want %s", reason, choice["finish_reason"], want)
		}
	}
}
