package runtime

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
	"maps"
	"slices"
	"strings"
	"sync/atomic"

	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/internal/callcontext"
)

// PreparedCall is one logical runtime operation, not an authorization grant.
// The embedding product must keep it request-local and enforce its own key/
// route grants; two service keys can share Caller but not gateway permissions.
// Copies share the use guard. Descriptions contain no credential or body.
type PreparedCall struct {
	owner         any
	caller        core.Caller
	instance      string
	generation    uint64
	source, scope [32]byte
	binding       binding
	call          core.PreparedCall
	snapshot      core.PreparedCall
	used          *atomic.Bool
	usage         *core.SourceUsageCollector
}

// A runtime binding points at credential stores and private provider state.
// Logging the handle must never recursively format those objects; callers
// explicitly request the detached, secret-free Description instead.
func (PreparedCall) String() string     { return "runtime.PreparedCall{private binding}" }
func (p PreparedCall) GoString() string { return p.String() }
func (p PreparedCall) Format(state fmt.State, verb rune) {
	switch verb {
	case 'v', 's', 'q', 'x', 'X':
		fmt.Fprintf(state, fmt.FormatString(state, verb), p.String())
	default:
		fmt.Fprintf(state, "%%!%c(runtime.PreparedCall=%s)", verb, p.String())
	}
}

func (p *PreparedCall) Description() core.CallDescription {
	if p == nil {
		return core.CallDescription{}
	}
	return p.snapshot.Description()
}

func (p *PreparedCall) SourceUsage() (core.SourceUsageObservation, bool) {
	if p == nil {
		return core.SourceUsageObservation{}, false
	}
	return p.usage.SourceUsage()
}
func (p *PreparedCall) SourceUsageHistory() core.SourceUsageHistory {
	if p == nil {
		return core.SourceUsageHistory{}
	}
	return p.usage.SourceUsageHistory()
}

func preparedSource(request core.Request) core.Request {
	request.Body = slices.Clone(request.Body)
	request.Header = request.Header.Clone()
	return request
}

func requestDigest(request core.Request) [32]byte {
	// Credential is always resolved from Runtime's store, as in legacy Invoke.
	// Length-delimited raw bytes avoid JSON's replacement of invalid UTF-8,
	// which must not make distinct target/header/credential strings collide.
	h := sha256.New()
	for _, value := range []string{string(request.Surface), request.Model, request.ContentType} {
		digestBytes(h, []byte(value))
	}
	digestBytes(h, request.Body)
	digestCount(h, len(request.Header))
	for _, key := range slices.Sorted(maps.Keys(request.Header)) {
		digestBytes(h, []byte(key))
		digestCount(h, len(request.Header[key]))
		for _, value := range request.Header[key] {
			digestBytes(h, []byte(value))
		}
	}
	return [32]byte(h.Sum(nil))
}

func digestBytes(h hash.Hash, data []byte) { digestCount(h, len(data)); _, _ = h.Write(data) }
func digestCount(h hash.Hash, count int) {
	var size [8]byte
	binary.BigEndian.PutUint64(size[:], uint64(count))
	_, _ = h.Write(size[:])
}
func digestMap(h hash.Hash, values map[string]string) {
	digestCount(h, len(values))
	for _, key := range slices.Sorted(maps.Keys(values)) {
		digestBytes(h, []byte(key))
		digestBytes(h, []byte(values[key]))
	}
}

func credentialScope(key string, credential *core.Credential) [32]byte {
	if credential == nil {
		return sha256.Sum256([]byte("no-credential:" + key))
	}
	copy := *credential
	copy.CredentialRevision = 0
	// A known same account may renew its access token, but static rotations
	// and tokens with no established account need a fresh admission.
	if copy.AccountID != "" && strings.EqualFold(copy.TokenType, "Bearer") {
		copy.Token = ""
		copy.APIKey = ""
	}
	h := sha256.New()
	for _, value := range []string{key, copy.ConnectionID, copy.AccountID, copy.TokenType, copy.APIKey, copy.Token} {
		digestBytes(h, []byte(value))
	}
	digestMap(h, copy.Headers)
	digestMap(h, copy.Metadata)
	return [32]byte(h.Sum(nil))
}

// PrepareCall binds the original request, settings and locally stored identity.
// It neither mints nor refreshes credentials and performs no upstream calls.
func (r *Runtime[S]) PrepareCall(ctx context.Context, caller core.Caller, instance string, request core.Request, options core.CallPreparationOptions) (*PreparedCall, error) {
	request = preparedSource(request)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := caller.Validate(); err != nil {
		return nil, invalidRequest(err)
	}
	if err := request.Validate(); err != nil {
		return nil, invalidRequest(err)
	}
	b, err := r.bind(instance)
	if err != nil {
		return nil, err
	}
	key, credential, err := r.storedCredential(ctx, caller, instance)
	if err != nil {
		return nil, err
	}
	source := requestDigest(request)
	request.Credential = credential
	call, err := core.PrepareCall(ctx, b.provider, request, options)
	if err != nil {
		return nil, err
	}
	if call == nil {
		return nil, core.NewConfigurationError("provider returned no prepared call", nil)
	}
	description := call.Description()
	mode := options.Mode
	if mode == "" {
		mode = core.CallModeInvoke
	}
	if description.Ingress != request.Surface || description.Model != request.Model || description.Mode != mode {
		return nil, core.ErrPreparedCallStale
	}
	p := &PreparedCall{owner: r, caller: caller, instance: instance, generation: b.generation, source: source, scope: credentialScope(key, credential), binding: b, call: call,
		snapshot: core.NewPreparedCall(description, nil, nil), used: &atomic.Bool{}, usage: core.NewSourceUsageCollectorFor(core.Target{Provider: instance, Model: request.Model})}
	if err := r.checkPrepared(ctx, caller, instance, request, p, mode); err != nil {
		return nil, err
	}
	return p, nil
}

func (r *Runtime[S]) checkPrepared(ctx context.Context, caller core.Caller, instance string, request core.Request, p *PreparedCall, mode core.CallMode) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if p == nil || p.owner != r || p.caller != caller || p.instance != instance || p.source != requestDigest(request) {
		return core.ErrPreparedCallStale
	}
	if p.Description().Mode != mode {
		return core.ErrPreparedCallMode
	}
	_, generation := r.options.Settings.Snapshot()
	if generation != p.generation {
		return core.ErrPreparedCallStale
	}
	key, credential, err := r.storedCredential(ctx, caller, instance)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return core.ErrPreparedCallStale
	}
	if credentialScope(key, credential) != p.scope {
		return core.ErrPreparedCallStale
	}
	resolvedKey, err := r.credentialKey(ctx, caller, instance)
	if err != nil || resolvedKey != key {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return core.ErrPreparedCallStale
	}
	// Resolve/Load can wait behind a connection update. The final generation
	// fence must follow those awaited reads, not only precede them.
	_, finalGeneration := r.options.Settings.Snapshot()
	if finalGeneration != p.generation {
		return core.ErrPreparedCallStale
	}
	return nil
}

func (r *Runtime[S]) beginPrepared(ctx context.Context, caller core.Caller, instance string, request core.Request, p *PreparedCall, mode core.CallMode, policy core.CallPolicy) (credential, error) {
	if err := r.checkPrepared(ctx, caller, instance, request, p, mode); err != nil {
		return credential{}, err
	}
	if _, err := core.PlanPreparedCall(p.Description(), policy); err != nil {
		return credential{}, err
	}
	if !p.used.CompareAndSwap(false, true) {
		return credential{}, core.ErrPreparedCallConsumed
	}
	c, err := r.credential(ctx, caller, instance, p.binding)
	if err != nil {
		return credential{}, err
	}
	// Token may have waited for a refresh while settings or connection scope
	// changed. Fence after the wait as well as immediately before every I/O.
	if err := r.checkPreparedCredential(ctx, caller, instance, request, p, mode, c); err != nil {
		return credential{}, err
	}
	return c, nil
}

func (r *Runtime[S]) checkPreparedCredential(ctx context.Context, caller core.Caller, instance string, request core.Request, p *PreparedCall, mode core.CallMode, c credential) error {
	if err := r.checkPrepared(ctx, caller, instance, request, p, mode); err != nil {
		return err
	}
	if credentialScope(c.key, c.value) != p.scope {
		return core.ErrPreparedCallStale
	}
	return nil
}

// InvokePrepared performs only the admitted mapping. A same-identity OAuth
// rejection permits one refresh, not a second endpoint or conversion choice.
func (r *Runtime[S]) InvokePrepared(ctx context.Context, caller core.Caller, instance string, request core.Request, p *PreparedCall, policy core.CallPolicy) (core.Response, error) {
	request = preparedSource(request)
	if p != nil {
		ctx = core.WithSourceUsageCollector(ctx, p.usage)
	}
	c, err := r.beginPrepared(ctx, caller, instance, request, p, core.CallModeInvoke, policy)
	if err != nil {
		return core.Response{}, err
	}
	executionContext := func(c credential) context.Context {
		return callcontext.WithCheck(ctx, func(checkContext context.Context) error {
			return r.checkPreparedCredential(checkContext, caller, instance, request, p, core.CallModeInvoke, c)
		})
	}
	response, err := p.call.Invoke(executionContext(c), c.value)
	if rejectedCredential(err, c) {
		if next, ok := refreshed(ctx, p.binding, c); ok {
			c = next
			if check := r.checkPreparedCredential(ctx, caller, instance, request, p, core.CallModeInvoke, c); check != nil {
				return core.Response{Losses: response.Losses}, check
			}
			response, err = p.call.Invoke(executionContext(c), c.value)
		}
	}
	r.observe(ctx, caller, instance, c, core.EvidenceOperationInvoke, request.Surface, request.Model, err)
	return response, err
}

// StreamPrepared refreshes only a rejected open. Once an iterator is returned,
// later errors/401s are the caller's observation and never reopen inference.
func (r *Runtime[S]) StreamPrepared(ctx context.Context, caller core.Caller, instance string, request core.Request, p *PreparedCall, policy core.CallPolicy) (core.StreamIter, error) {
	request = preparedSource(request)
	if p != nil {
		ctx = core.WithSourceUsageCollector(ctx, p.usage)
	}
	c, err := r.beginPrepared(ctx, caller, instance, request, p, core.CallModeStream, policy)
	if err != nil {
		return nil, err
	}
	executionContext := func(c credential) context.Context {
		return callcontext.WithCheck(ctx, func(checkContext context.Context) error {
			return r.checkPreparedCredential(checkContext, caller, instance, request, p, core.CallModeStream, c)
		})
	}
	stream, err := p.call.Stream(executionContext(c), c.value)
	if rejectedCredential(err, c) {
		if stream != nil {
			_ = stream.Close()
			stream = nil
		}
		if next, ok := refreshed(ctx, p.binding, c); ok {
			c = next
			if check := r.checkPreparedCredential(ctx, caller, instance, request, p, core.CallModeStream, c); check != nil {
				return nil, check
			}
			stream, err = p.call.Stream(executionContext(c), c.value)
		}
	}
	r.observe(ctx, caller, instance, c, core.EvidenceOperationStream, request.Surface, request.Model, err)
	return p.usage.WrapStream(stream), err
}
