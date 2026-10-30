package core

// StreamUsageProvider declares whether a native streaming surface can provide
// terminal usage when asked. Translators use it only for generated controls.
type StreamUsageProvider interface {
	SupportsStreamUsage(model string, surface ModelSurface) bool
}

// SupportsStreamUsage reports the nearest native provider's declaration.
func SupportsStreamUsage(provider Provider, model string, surface ModelSurface) bool {
	for provider != nil {
		if !ServesNatively(provider, model, surface) {
			return false
		}
		if usage, ok := provider.(StreamUsageProvider); ok {
			return usage.SupportsStreamUsage(model, surface)
		}
		provider = unwrapProvider(provider)
	}
	return false
}
