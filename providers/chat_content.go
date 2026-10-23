package providers

import (
	"crypto/rand"
	"maps"
	"slices"
	"strconv"
	"strings"

	translate "github.com/kayiik/llm-translate"
	core "github.com/kayiik/llmgw-core"
)

// chatText preserves equivalent string and text-part content without adding
// separators. Unsupported parts are findings, never printed or empty prompts
// silently sent to inference. Execution checks these findings before I/O.
func chatText(path string, content any, detail string) (string, []core.Loss) {
	lost := func(path string) core.Loss {
		return core.Loss{Path: path, Class: translate.LossDropped, Severity: translate.LossMaterial, Detail: detail}
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
			partPath := path + "." + strconv.Itoa(index)
			part, _ := raw.(map[string]any)
			value, valid := part["text"].(string)
			if part["type"] != "text" || !valid {
				losses = append(losses, lost(partPath))
				continue
			}
			text.WriteString(value)
			for _, field := range slices.Sorted(maps.Keys(part)) {
				if field != "type" && field != "text" && part[field] != nil {
					losses = append(losses, lost(partPath+"."+field))
				}
			}
		}
		return text.String(), losses
	default:
		return "", []core.Loss{lost(path)}
	}
}

// A completion identity belongs to an operation, not its provider or an
// upstream frame (whose response ID may arrive after output was delivered).
func newChatCompletionID(provider string) string {
	return "chatcmpl-" + provider + "-" + rand.Text()
}
