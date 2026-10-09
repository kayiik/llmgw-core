package providers

import (
	"fmt"
	"net/http"
	"slices"
	"strings"

	core "github.com/kayiik/llmgw-core"
)

// extraHeaders are the headers a provider's configuration adds to every
// request it sends, such as an attribution header, or one a proxy in front
// of the upstream routes or authenticates by.
//
// They go on first, so every header the provider sets itself replaces a
// configured one of the same name. When the provider authenticates a
// request with the operation's credential, it also removes each configured
// header its upstream could read a credential from, Authorization and the
// provider's own key header, so a configured header never stands in for the
// credential or competes with it: Vertex AI, for one, refuses a request that
// presents two. A request sent without a credential carries them as
// configured.
type extraHeaders http.Header

// newExtraHeaders checks and copies a provider configuration's headers. A
// name that is not an HTTP token, or a value with a control character,
// would make Go's transport refuse every request, which reads as an
// upstream out of reach and counts against its circuit, so it is refused
// here as a configuration error, which never quotes a value. So are two
// names that differ only in case and carry different values, since only
// one of them could be sent.
func newExtraHeaders(label string, headers map[string]string) (extraHeaders, error) {
	if len(headers) == 0 {
		return nil, nil
	}
	extra := make(extraHeaders, len(headers))
	for name, value := range headers {
		if !httpToken(name) {
			return nil, core.NewConfigurationError(fmt.Sprintf("%s: the configured header name %q is not an HTTP token", label, name), nil)
		}
		key := http.CanonicalHeaderKey(name)
		if !httpFieldValue(value) {
			return nil, core.NewConfigurationError(fmt.Sprintf("%s: the configured %s header holds a control character", label, key), nil)
		}
		if previous, ok := extra[key]; ok && previous[0] != value {
			return nil, core.NewConfigurationError(fmt.Sprintf("%s: the configured headers set %s twice, to different values", label, key), nil)
		}
		extra[key] = []string{value}
	}
	return extra, nil
}

// header returns a new request header holding the configured headers, for
// the provider to set its own on.
func (e extraHeaders) header() http.Header {
	header := make(http.Header, len(e)+4)
	for name, values := range e {
		header[name] = slices.Clone(values)
	}
	return header
}

// setCredentialHeader sets the header a request authenticates with, after
// removing every configured header the upstream could read a credential
// from: Authorization, name itself, and others.
func setCredentialHeader(header http.Header, name, value string, others ...string) {
	header.Del("Authorization")
	for _, other := range others {
		header.Del(other)
	}
	header.Set(name, value)
}

// httpToken reports a header name Go's transport sends: a nonempty RFC 9110
// token.
func httpToken(name string) bool {
	return name != "" && !strings.ContainsFunc(name, func(r rune) bool {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			return false
		}
		return !strings.ContainsRune("!#$%&'*+-.^_`|~", r)
	})
}

// httpFieldValue reports a header value Go's transport sends: one without
// a control character other than a horizontal tab.
func httpFieldValue(value string) bool {
	return !strings.ContainsFunc(value, func(r rune) bool {
		return (r < ' ' && r != '\t') || r == 0x7f
	})
}
