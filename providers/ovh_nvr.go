package providers

import (
	"bytes"
	"context"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strings"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

// OVHcloud AI Endpoints serves NVIDIA Riva voices, its nvr-tts models,
// beside its OpenAI-compatible API rather than through it: each model at its
// own host, as text_to_audio, which takes Riva's synthesis request and
// answers with WAV audio.
const (
	ovhRegistryID = "ovh_ai_endpoints"
	ovhNVRPath    = "/api/v1/tts/text_to_audio"
	ovhNVRLabel   = "OVHcloud AI Endpoints"
	// ovhNVRSampleRate is the rate a voice is asked for. OVHcloud's WAV
	// header names 16 kHz whatever rate is asked, so audio at any other
	// rate would play at the wrong speed.
	ovhNVRSampleRate = 16000
	// ovhNVRLinearPCM is Riva's LINEAR_PCM encoding: 16-bit signed
	// little-endian samples.
	ovhNVRLinearPCM = 1
)

// ovhNVRModel matches an NVR voice model, nvr-tts-<language>-<region>,
// whose name is a valid DNS label, since it names the model's host.
var ovhNVRModel = regexp.MustCompile(`^nvr-tts-([a-z]{2})-([a-z]{2})$`)

// ovhNVR reports a speech model an OVHcloud AI Endpoints instance serves as
// an NVR voice rather than over its OpenAI-compatible API.
func (p *OpenAICompatible) ovhNVR(model string) bool {
	return p.registryID == ovhRegistryID && ovhNVRModel.MatchString(model)
}

// ovhNVRVoices are each language's voice prefix and base voice, as
// OVHcloud documents them; a model of another language needs its voice
// named.
func ovhNVRVoices(language string) (prefix, base string) {
	switch language {
	case "en-US":
		return "English-US.", "English-US.Female-1"
	case "es-ES":
		return "Spanish-ES-", "Spanish-ES-Female-1"
	case "de-DE":
		return "German-DE-", "German-DE-Male-1"
	case "it-IT":
		return "Italian-IT-", "Italian-IT-Female-1"
	}
	return "", ""
}

// invokeNVR serves an OpenAI speech request on an NVR voice model. The
// request's input is the text and its model names the language. A voice of
// that language, such as English-US.Male-1, is sent as given; any other,
// such as an OpenAI voice name, or none, is the language's base voice, the
// first reported as an advisory loss. The answer is WAV audio at 16 kHz,
// so a response_format other than wav is refused, and speed and
// instructions, which Riva's request has no place for, are dropped as
// advisory losses. Authorization and configured headers are the
// instance's, and a keyless instance sends none.
func (p *OpenAICompatible) invokeNVR(ctx context.Context, request core.Request) (core.Response, error) {
	payload, err := openAIPayload(request)
	if err != nil {
		return core.Response{}, err
	}
	text, _ := payload["input"].(string)
	if strings.TrimSpace(text) == "" {
		return core.Response{}, openAIInvalid("an OVHcloud NVR speech request needs input", nil)
	}
	if format, _ := payload["response_format"].(string); format != "" && !strings.EqualFold(format, "wav") {
		return core.Response{}, openAIInvalid("OVHcloud NVR voices answer only response_format wav", nil)
	}
	parts := ovhNVRModel.FindStringSubmatch(request.Model)
	language := parts[1] + "-" + strings.ToUpper(parts[2])
	voice, losses := ovhNVRVoice(language, payload)
	if voice == "" {
		return core.Response{}, openAIInvalid("an OVHcloud NVR model of "+language+" needs its voice named", nil)
	}
	endpoint, err := p.ovhNVREndpoint(request.Model)
	if err != nil {
		return core.Response{}, err
	}
	body, err := openAIEncode(map[string]any{
		"encoding": ovhNVRLinearPCM, "language_code": language, "sample_rate_hz": ovhNVRSampleRate, "text": text, "voice_name": voice,
	})
	if err != nil {
		return core.Response{}, openAIInvalid("the OVHcloud NVR request could not be encoded", err)
	}
	access, err := p.access(request.Credential)
	if err != nil {
		return core.Response{}, err
	}
	if access.anonymous {
		access.authorization = ""
	}
	upstream, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return core.Response{}, core.NewConfigurationError("the OVHcloud NVR request could not be created", err)
	}
	upstream.Header = p.header(access, "application/octet-stream", false)
	response, err := p.client.Do(upstream)
	if err != nil {
		return core.Response{}, transportFailure(ctx, ovhNVRLabel+" could not be reached", err)
	}
	defer response.Body.Close()
	raw, err := readInvocationResponseBody(ctx, response, ovhNVRLabel)
	if err != nil {
		return core.Response{}, err
	}
	if response.StatusCode >= http.StatusBadRequest {
		return core.Response{}, httpStatusFailure(ovhNVRLabel, response, raw, p.now())
	}
	if len(raw) < 12 || string(raw[:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return core.Response{}, unusableResponse("the "+ovhNVRLabel+" speech answer is not WAV audio", nil)
	}
	return core.Response{Body: raw, ContentType: "audio/wav", Losses: translate.NewReport(losses...).Losses}, nil
}

// ovhNVRVoice returns the voice to send for a request in language, and the
// losses of the request's other fields.
func ovhNVRVoice(language string, payload map[string]any) (string, []core.Loss) {
	var losses []core.Loss
	advisory := func(path, detail string) {
		losses = append(losses, core.Loss{Path: path, Class: translate.LossDropped, Severity: translate.LossAdvisory, Detail: detail})
	}
	for _, field := range slices.Sorted(maps.Keys(payload)) {
		switch value := payload[field]; field {
		case "model", "input", "voice", "response_format":
		case "speed":
			if speed, ok := value.(float64); !ok || speed != 1 {
				advisory(field, "OVHcloud NVR voices speak at their own speed")
			}
		default:
			if value != nil {
				advisory(field, "OVHcloud NVR voices do not take this field")
			}
		}
	}
	prefix, base := ovhNVRVoices(language)
	voice, _ := payload["voice"].(string)
	voice = strings.TrimSpace(voice)
	switch {
	case voice != "" && (prefix == "" || strings.HasPrefix(voice, prefix)):
		return voice, losses
	case voice != "":
		advisory("voice", "the voice is not one of the model's language, so its base voice speaks")
	}
	return base, losses
}

// ovhNVREndpoint is the text_to_audio endpoint of model: the host of the
// instance's OpenAI-compatible API, oai.<domain>, with the model in place
// of oai.
func (p *OpenAICompatible) ovhNVREndpoint(model string) (string, error) {
	base, err := url.Parse(p.baseURL)
	host, ok := "", false
	if err == nil {
		host, ok = strings.CutPrefix(base.Host, "oai.")
	}
	if !ok || host == "" {
		return "", core.NewConfigurationError(fmt.Sprintf(
			"OVHcloud serves %s at its own host beside oai.endpoints.kepler.ai.cloud.ovh.net; the instance's base URL names no such host", model), nil)
	}
	return base.Scheme + "://" + model + "." + host + ovhNVRPath, nil
}
