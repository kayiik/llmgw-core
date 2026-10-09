package runtime

import (
	"fmt"

	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/providers"
	"github.com/kayiik/llmgw-core/translation"
)

// RegistryInstance is one provider instance as a product's settings
// describe it: the registry entry it serves, how it is reached, and whether
// a translation.Adapter serves the chat surfaces its provider does not speak.
type RegistryInstance struct {
	// Entry is the id, or an alias, of the instance's registry entry.
	Entry string
	// Instance is how the instance is reached; see providers.Instance.
	Instance providers.Instance
	// Translate puts a translation.Adapter with Policy in front of the
	// provider, so it also serves Chat Completions, Messages and Responses
	// over whichever of them it speaks.
	Translate bool
	Policy    core.LossPolicy
}

// RegistryProviders returns a ProviderFactory that builds each instance from
// its registry entry: describe tells the instance's entry and settings,
// registry resolves the entry, nil meaning providers.DefaultRegistry, and
// providers.Build makes its provider. A product that builds its providers
// this way serves a runtime type core adds by upgrading, with no change to
// describe.
//
// An entry the registry does not hold fails as providers.Build fails on an
// entry it cannot make a provider for: a configuration error, which permits
// failover, whose cause is a *providers.UnsupportedEntryError. An error
// describe returns is returned as it is.
func RegistryProviders[S any](registry *providers.Registry, describe func(settings S, instance string) (RegistryInstance, error)) ProviderFactory[S] {
	if registry == nil {
		registry = providers.DefaultRegistry()
	}
	return func(settings S, instance string) (core.Provider, error) {
		described, err := describe(settings, instance)
		if err != nil {
			return nil, err
		}
		entry, ok := registry.Lookup(described.Entry)
		if !ok {
			failure := &providers.UnsupportedEntryError{Entry: described.Entry, Reason: "the registry holds no such entry"}
			return nil, core.NewConfigurationError(fmt.Sprintf("provider instance %s: %v", instance, failure), failure)
		}
		provider, err := providers.Build(entry, described.Instance)
		if err != nil {
			return nil, err
		}
		if described.Translate {
			return translation.Adapter{Provider: provider, Policy: described.Policy}, nil
		}
		return provider, nil
	}
}
