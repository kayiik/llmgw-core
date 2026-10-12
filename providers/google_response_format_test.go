package providers

import (
	"context"
	"net/http"
	"reflect"
	"testing"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

// googleVerdictSchema is a JSON Schema as a client writes it, $ref
// included, and googleVerdictSent the same schema as Google receives it:
// whole, its keys sorted as encoding/json sorts them.
const (
	googleVerdictSchema = `{"type":"object","properties":{"verdict":{"$ref":"#/$defs/verdict"},"confidence":{"type":"number","minimum":0,"maximum":1}},` +
		`"required":["verdict"],"additionalProperties":false,"$defs":{"verdict":{"type":"string","enum":["yes","no"]}}}`
	googleVerdictSent = `{"$defs":{"verdict":{"enum":["yes","no"],"type":"string"}},"additionalProperties":false,` +
		`"properties":{"confidence":{"maximum":1,"minimum":0,"type":"number"},"verdict":{"$ref":"#/$defs/verdict"}},"required":["verdict"],"type":"object"}`
)

// googleSent sends a Chat body to a deployment through Invoke, or through
// Stream when stream is set, and returns the generateContent body Google
// received on the deployment's route and the losses reported.
func googleSent(t *testing.T, deployment GoogleDeployment, stream bool, body string) (string, []core.Loss) {
	t.Helper()
	config, route := GoogleConfig{Deployment: deployment}, "/models/gemini-3.5-flash:"
	if deployment == GoogleVertexAI {
		config.Project, route = "fixture-project", "/projects/fixture-project/locations/global/publishers/google/models/gemini-3.5-flash:"
	}
	request := googleChatRequest("gemini-3.5-flash", body, googleKey("k"))
	if !stream {
		fake, base := newGoogleFake(t, googleAnswer(http.StatusOK, googleHelloAnswer))
		config.BaseURL = base
		response, err := newGoogleTest(t, config).Invoke(context.Background(), request)
		if err != nil {
			t.Fatal(err)
		}
		calls := fake.take()
		if len(calls) != 1 || calls[0].path != route+"generateContent" {
			t.Fatalf("upstream = %+v", calls)
		}
		return calls[0].body, response.Losses
	}
	take, base := newGoogleStreamFake(t, http.StatusOK, googleWire(googleFixture(t, "stream-text.sse")))
	config.BaseURL = base
	chunks, err := newGoogleTest(t, config).Stream(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := googleFrames(t, chunks); err != nil {
		t.Fatal(err)
	}
	calls := take()
	if len(calls) != 1 || calls[0].path != route+"streamGenerateContent" || calls[0].query != "alt=sse" {
		t.Fatalf("upstream = %+v", calls)
	}
	return calls[0].body, core.StreamLosses(chunks)
}

// response_format asks Gemini for structured output, the same on both
// deployments and in a stream: json_object asks for JSON, and json_schema
// for JSON that follows its schema, sent whole as responseJsonSchema beside
// the request's other generation settings. A text format, or none, asks for
// nothing, so nothing is sent or reported.
func TestGoogleChatAsksForStructuredOutput(t *testing.T) {
	t.Parallel()
	const contents = `{"contents":[{"parts":[{"text":"Is the sky blue?"}],"role":"user"}]`
	for _, testCase := range []struct{ name, fields, config string }{
		{name: "json_object", fields: `"response_format":{"type":"json_object"}`, config: `"responseMimeType":"application/json"`},
		{
			name:   "json_schema",
			fields: `"max_tokens":64,"temperature":0.2,"response_format":{"type":"json_schema","json_schema":{"schema":` + googleVerdictSchema + `}}`,
			config: `"maxOutputTokens":64,"responseJsonSchema":` + googleVerdictSent + `,"responseMimeType":"application/json","temperature":0.2`,
		},
		{name: "text", fields: `"response_format":{"type":"text"}`},
		{name: "none", fields: `"max_tokens":64,"response_format":null`, config: `"maxOutputTokens":64`},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			want := contents + "}"
			if testCase.config != "" {
				want = contents + `,"generationConfig":{` + testCase.config + `}}`
			}
			for _, deployment := range []GoogleDeployment{GoogleAIStudio, GoogleVertexAI} {
				for _, stream := range []bool{false, true} {
					body, losses := googleSent(t, deployment, stream, `{"messages":[{"role":"user","content":"Is the sky blue?"}],`+testCase.fields+`}`)
					if body != want || len(losses) != 0 {
						t.Fatalf("%s, stream %t: body = %s, losses = %+v\nwant   %s", deployment, stream, body, losses, want)
					}
				}
			}
		})
	}
}

// Gemini takes a schema but has no place for its name or description, and
// no strict mode, so the schema is sent and the rest reported as advisory
// losses: the answer still has the structure the schema asks for. An empty
// name or description, and strict false or null, lose nothing.
func TestGoogleChatReportsWhatAJSONSchemaFormatLoses(t *testing.T) {
	t.Parallel()
	const want = `{"contents":[{"parts":[{"text":"hi"}],"role":"user"}],` +
		`"generationConfig":{"responseJsonSchema":{"type":"object"},"responseMimeType":"application/json"}}`
	for members, losses := range map[string][]core.Loss{
		`"name":"verdict","description":"A yes or no verdict","strict":true`: {
			googleLoss("response_format.json_schema.description", translate.LossAdvisory, "Google does not send this response format field"),
			googleLoss("response_format.json_schema.name", translate.LossAdvisory, "Google does not send this response format field"),
			googleLoss("response_format.json_schema.strict", translate.LossAdvisory, "Google does not guarantee that the answer matches the schema exactly"),
		},
		`"name":"","description":"","strict":false`:    nil,
		`"name":null,"description":null,"strict":null`: nil,
	} {
		for _, stream := range []bool{false, true} {
			body, reported := googleSent(t, GoogleAIStudio, stream, `{"messages":[{"role":"user","content":"hi"}],`+
				`"response_format":{"type":"json_schema","json_schema":{`+members+`,"schema":{"type":"object"}}}}`)
			if body != want || !reflect.DeepEqual(reported, losses) {
				t.Fatalf("%s, stream %t: body = %s, losses = %+v", members, stream, body, reported)
			}
		}
	}
}

// A response format Gemini cannot express is not sent, and its loss is
// material, so a product refuses the target rather than answer without the
// structure the request asks for: a format that is not an object, of a type
// Chat does not define, a json_schema format without a JSON Schema object,
// and a format with a member Chat does not define.
func TestGoogleChatDropsResponseFormatsItCannotSend(t *testing.T) {
	t.Parallel()
	const unsendable = "Google cannot send this response format"
	const schemaless = "the response format holds no JSON Schema object, so Google cannot send it"
	for _, testCase := range []struct{ name, format, detail string }{
		{name: "not an object", format: `"json_object"`, detail: "the response format is not an object, so Google cannot send it"},
		{name: "an unknown type", format: `{"type":"json_mode"}`, detail: unsendable},
		{name: "no type", format: `{"json_schema":{"schema":{"type":"object"}}}`, detail: unsendable},
		{name: "text with another member", format: `{"type":"text","json_schema":{"schema":{"type":"object"}}}`, detail: unsendable},
		{name: "json_object with another member", format: `{"type":"json_object","schema":{"type":"object"}}`, detail: unsendable},
		{name: "json_schema with another member", format: `{"type":"json_schema","json_schema":{"schema":{"type":"object"}},"strict":true}`, detail: unsendable},
		{name: "a json_schema member Chat does not define", format: `{"type":"json_schema","json_schema":{"schema":{"type":"object"},"examples":[{}]}}`, detail: unsendable},
		{name: "no json_schema", format: `{"type":"json_schema"}`, detail: schemaless},
		{name: "a json_schema that is not an object", format: `{"type":"json_schema","json_schema":"verdict"}`, detail: schemaless},
		{name: "no schema", format: `{"type":"json_schema","json_schema":{"name":"verdict","strict":true}}`, detail: schemaless},
		{name: "a schema that is not an object", format: `{"type":"json_schema","json_schema":{"name":"verdict","schema":"{\"type\":\"object\"}"}}`, detail: schemaless},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()
			for _, stream := range []bool{false, true} {
				body, losses := googleSent(t, GoogleAIStudio, stream, `{"messages":[{"role":"user","content":"hi"}],"max_tokens":64,"response_format":`+testCase.format+`}`)
				if want := `{"contents":[{"parts":[{"text":"hi"}],"role":"user"}],"generationConfig":{"maxOutputTokens":64}}`; body != want {
					t.Fatalf("stream %t: body = %s\nwant   %s", stream, body, want)
				}
				if want := []core.Loss{googleLoss("response_format", translate.LossMaterial, testCase.detail)}; !reflect.DeepEqual(losses, want) {
					t.Fatalf("stream %t: losses = %+v", stream, losses)
				}
			}
		})
	}
}
