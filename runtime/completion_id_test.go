package runtime_test

import (
	"regexp"
	"strings"
	"testing"
)

// Normalize only a validated, stable completion ID for the existing wire
// goldens; the provider regression suite checks uniqueness across operations.
func completionGolden(t *testing.T, body, provider string) string {
	t.Helper()
	matches := regexp.MustCompile(`"id":"(chatcmpl-`+provider+`-[A-Z2-7]{26})"`).FindAllStringSubmatch(body, -1)
	if len(matches) == 0 {
		t.Fatalf("missing completion identity: %s", body)
	}
	if len(matches) != strings.Count(body, `"object":"chat.completion`) {
		t.Fatalf("not every completion has an operation ID: %s", body)
	}
	for _, match := range matches {
		if match[1] != matches[0][1] {
			t.Fatalf("completion identity changed: %s", body)
		}
	}
	return strings.ReplaceAll(body, `"id":"`+matches[0][1]+`"`, `"id":"chatcmpl-`+provider+`"`)
}
