package providers

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	core "github.com/kayiik/llmgw-core"
)

// Instance is how one provider instance of a registry entry is reached, as
// a product's settings describe it: everything Build makes a provider from
// but the entry itself. It holds no credential, since each request carries
// its own, as a Runtime resolves it. Every field is optional, and a field
// the entry's provider does not read is ignored.
type Instance struct {
	// BaseURL is where the instance's API lives. Empty uses the entry's
	// default_base_url, and an entry that requires a base URL but has no
	// default is refused without one. Bedrock and Vertex AI, which have
	// none, reach the endpoint of their region or location.
	BaseURL string
	// Headers are sent with every request the provider makes; see each
	// provider config's Headers for the headers they never replace.
	Headers map[string]string
	// Client performs inference and CatalogClient lists models. Nil uses
	// each provider's default.
	Client, CatalogClient *http.Client
	// Region is a Bedrock instance's AWS region. Empty uses the entry's
	// default_region.
	Region string
	// Project, Location and RequestType configure a Vertex AI instance, as
	// GoogleConfig describes them. An empty Location uses the entry's
	// default_location.
	Project, Location, RequestType string
	// Models, ForwardAllFields and ForceAPISupport configure an
	// OpenAI-compatible or Bedrock instance, as OpenAICompatibleConfig
	// describes them.
	Models           func(model string) (core.ModelInfo, bool)
	ForwardAllFields bool
	ForceAPISupport  bool
	// Preamble is an Anthropic instance's preamble hook; see
	// AnthropicConfig.
	Preamble func(ctx context.Context) string
	// TranscriptionModels and SpeechModels are an audio instance's models,
	// as ElevenLabsConfig and MiMoConfig describe them: nil selects the
	// reviewed defaults and an empty list none.
	TranscriptionModels, SpeechModels []string
	// Now is the clock. Nil uses time.Now.
	Now func() time.Time
}

// UnsupportedEntryError reports a registry entry Build makes no provider
// for: one whose runtime type core implements no provider for, such as a
// product's own, or one a product or an extension daemon serves, with the
// reason. Build returns it as the cause of a configuration error.
type UnsupportedEntryError struct {
	Entry, RuntimeType, Reason string
}

func (e *UnsupportedEntryError) Error() string {
	if e.RuntimeType == "" {
		return fmt.Sprintf("registry entry %q: %s", e.Entry, e.Reason)
	}
	return fmt.Sprintf("registry entry %q, runtime type %q: %s", e.Entry, e.RuntimeType, e.Reason)
}

// SupportedRuntimeTypes lists the runtime types Build makes a provider
// for, for a product's ValidationOptions. An entry of one of them may still
// be one Build refuses; see Build.
func SupportedRuntimeTypes() []string {
	return []string{"ai_studio", "anthropic", "azure_openai", "bedrock", "elevenlabs", "mimo", "ollama", "openai_compatible", "vertex_ai"}
}

// Build makes the provider an instance of entry serves, by the entry's
// runtime type: openai_compatible is OpenAICompatible with the entry's id
// as its RegistryID, bedrock is NewBedrock, anthropic is Anthropic,
// ai_studio and vertex_ai are Google for that deployment, azure_openai is
// AzureOpenAI, ollama is Ollama, and elevenlabs and mimo are ElevenLabs and
// MiMo. A product that builds its providers here serves a runtime type core
// adds by upgrading. The provider speaks only its native surfaces; a
// translation.Adapter in front serves the others.
//
// Build makes none for an entry that is not available, an entry another
// party serves (OpenAI Codex and OpenCode Zen, which need more than the
// OpenAI-compatible wire, and the GitHub Copilot, Google Antigravity and
// Edge TTS runtime types, which an extension daemon serves), or a runtime
// type it does not know. Each such failure, and every refusal of the
// instance's settings, is a *core.ProviderError of class configuration,
// which permits failover; the first's cause is an *UnsupportedEntryError.
func Build(entry RegistryEntry, instance Instance) (core.Provider, error) {
	id := strings.ToLower(strings.TrimSpace(entry.ID))
	runtimeType := strings.ToLower(strings.TrimSpace(entry.RuntimeType))
	if reason := unsupportedEntry(id, runtimeType, entry.Availability); reason != "" {
		failure := &UnsupportedEntryError{Entry: id, RuntimeType: runtimeType, Reason: reason}
		return nil, core.NewConfigurationError(failure.Error(), failure)
	}
	base := strings.TrimSpace(instance.BaseURL)
	if base == "" {
		base = entry.DefaultBaseURL
	}
	if base == "" && entry.RequiresBaseURL {
		return nil, core.NewConfigurationError(fmt.Sprintf("registry entry %q needs a base URL", id), nil)
	}
	provider, err := buildRuntimeType(entry, runtimeType, base, instance)
	if err != nil {
		var failure *core.ProviderError
		if !errors.As(err, &failure) {
			err = core.NewConfigurationError(err.Error(), err)
		}
		return nil, err
	}
	return provider, nil
}

// unsupportedEntry returns why Build makes no provider for an entry, or ""
// when it makes one.
func unsupportedEntry(id, runtimeType, availability string) string {
	switch {
	case availability != "" && availability != ProviderAvailable:
		return "the entry is " + strings.ReplaceAll(availability, "_", " ") + ", not available"
	case id == "openai_codex":
		return "OpenAI Codex signs in with a ChatGPT account and speaks Responses to its own backend, which an extension daemon serves"
	case id == "opencode_zen":
		return "OpenCode Zen serves its models over several wires, which the product's own provider chooses between"
	}
	switch runtimeType {
	case "github_copilot", "google_antigravity", "edge_tts":
		return "core implements no provider for this runtime type; an extension daemon serves it"
	}
	for _, supported := range SupportedRuntimeTypes() {
		if runtimeType == supported {
			return ""
		}
	}
	return "core implements no provider for this runtime type"
}

// buildRuntimeType calls the constructor of a supported runtime type.
func buildRuntimeType(entry RegistryEntry, runtimeType, base string, instance Instance) (core.Provider, error) {
	openAI := OpenAICompatibleConfig{
		BaseURL: base, RegistryID: entry.ID, Models: instance.Models, Headers: instance.Headers,
		ForwardAllFields: instance.ForwardAllFields, ForceAPISupport: instance.ForceAPISupport,
		Client: instance.Client, CatalogClient: instance.CatalogClient, Now: instance.Now,
	}
	switch runtimeType {
	case "openai_compatible":
		return NewOpenAICompatible(openAI)
	case "bedrock":
		region := strings.TrimSpace(instance.Region)
		if region == "" {
			region = entry.DefaultRegion
		}
		openAI.RegistryID = ""
		return NewBedrock(region, base, openAI)
	case "anthropic":
		return NewAnthropic(AnthropicConfig{
			BaseURL: base, Client: instance.Client, CatalogClient: instance.CatalogClient, Headers: instance.Headers,
			Preamble: instance.Preamble, Now: instance.Now,
		})
	case "ai_studio", "vertex_ai":
		config := GoogleConfig{
			Deployment: GoogleDeployment(runtimeType), BaseURL: base, Client: instance.Client, Headers: instance.Headers, Now: instance.Now,
		}
		if runtimeType == "vertex_ai" {
			config.Project, config.Location, config.RequestType = instance.Project, strings.TrimSpace(instance.Location), instance.RequestType
			if config.Location == "" {
				config.Location = entry.DefaultLocation
			}
		}
		return NewGoogle(config)
	case "azure_openai":
		return NewAzureOpenAI(AzureOpenAIConfig{
			BaseURL: base, Client: instance.Client, CatalogClient: instance.CatalogClient, Headers: instance.Headers, Now: instance.Now,
		})
	case "ollama":
		return NewOllama(OllamaConfig{
			BaseURL: base, Client: instance.Client, CatalogClient: instance.CatalogClient, Headers: instance.Headers, Now: instance.Now,
		})
	case "elevenlabs":
		return NewElevenLabs(ElevenLabsConfig{
			BaseURL: base, Client: instance.Client, Headers: instance.Headers,
			TranscriptionModels: instance.TranscriptionModels, SpeechModels: instance.SpeechModels,
		})
	case "mimo":
		return NewMiMo(MiMoConfig{BaseURL: base, Client: instance.Client, Headers: instance.Headers, SpeechModels: instance.SpeechModels})
	}
	// unsupportedEntry admits only the runtime types above.
	return nil, core.NewConfigurationError(fmt.Sprintf("runtime type %q has no constructor", runtimeType), nil)
}
