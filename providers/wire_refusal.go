package providers

import (
	"net/http"
	"strings"
	"time"

	core "github.com/kayiik/llmgw-core"
)

// refusalScanLimit bounds how much of a refusal's body is read for its
// reason.
const refusalScanLimit = 64 << 10

// refusalReasons are the words a refusal's body uses for each reason a
// product routes on, as the providers core serves word them: in an error
// code or type, such as OpenAI's insufficient_quota and
// context_length_exceeded, or in a message, such as Anthropic's "credit
// balance is too low" and "prompt is too long", Google's "input token
// count", or Ollama's "does not support tools". They are read in order.
func refusalReasons() []struct {
	class core.ProviderErrorClass
	words []string
} {
	return []struct {
		class core.ProviderErrorClass
		words []string
	}{
		{core.ProviderErrorBilling, []string{
			"insufficient_quota", "insufficient quota", "insufficient balance", "insufficient_balance", "credit balance is too low",
			"credits are depleted", "out of credits", "billing to be enabled", "billing account",
		}},
		{core.ProviderErrorContextOverflow, []string{
			"context_length_exceeded", "context length", "context window", "prompt is too long", "input token count",
		}},
		{core.ProviderErrorToolsUnsupported, []string{
			"support tools", "tools are not supported", "tool use is not supported", "tool calling is not supported",
			"support function calling", "function calling is not supported",
		}},
		{core.ProviderErrorMediaUnsupported, []string{
			"support image", "images are not supported", "image input is not supported", "image_url is only supported",
			"support vision", "vision is not supported", "support audio input",
		}},
	}
}

// refusalClass names the reason a refusal gives when it is one a product
// routes on, and "" otherwise. A 402 is billing whatever it says; any other
// client error is read for its reason, while a server error is not, since
// it refuses no request.
func refusalClass(status int, raw []byte) core.ProviderErrorClass {
	switch {
	case status == http.StatusPaymentRequired:
		return core.ProviderErrorBilling
	case status < http.StatusBadRequest || status >= http.StatusInternalServerError:
		return ""
	}
	text := strings.ToLower(string(raw[:min(len(raw), refusalScanLimit)]))
	for _, reason := range refusalReasons() {
		for _, word := range reason.words {
			if strings.Contains(text, word) {
				return reason.class
			}
		}
	}
	return ""
}

// refusalClassification is how a refusal of class routes, with its status
// and wait. Billing permits failover and counts against the instance, so a
// health policy can hold the key back for as long as billing takes to
// mend, apart from rate limits. A refused shape permits failover to a
// model that takes it, but repeating it would only be refused again, and
// it says nothing against the instance.
func refusalClassification(class core.ProviderErrorClass, status int, retryAfter time.Duration) core.ProviderErrorClassification {
	classification := core.ProviderErrorClassification{StatusCode: status, FailoverEligible: true, RetryAfter: retryAfter}
	if class == core.ProviderErrorBilling {
		classification.CircuitFailure = true
	}
	return classification
}
