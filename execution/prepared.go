package execution

import (
	"context"

	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/internal/callcontext"
)

// PrepareCall preserves policy wrapping without re-running provider selection.
// Each retry consumes the same low-level preparation and rechecks the runtime
// fence after any retry delay; streams only retry opening, never Next.
func (r *resilient) PrepareCall(ctx context.Context, request core.Request, options core.CallPreparationOptions) (core.PreparedCall, error) {
	call, err := core.PrepareCall(ctx, r.provider, request, options)
	if err != nil {
		return nil, err
	}
	return core.NewPreparedCall(call.Description(), func(ctx context.Context, c *core.Credential) (core.Response, error) {
		return guard(ctx, r, r.repeatable(request), func(ctx context.Context) (core.Response, error) {
			if err := callcontext.Check(ctx); err != nil {
				return core.Response{}, err
			}
			return call.Invoke(ctx, c)
		}, nil)
	}, func(ctx context.Context, c *core.Credential) (core.StreamIter, error) {
		return guard(ctx, r, r.repeatable(request), func(ctx context.Context) (core.StreamIter, error) {
			if err := callcontext.Check(ctx); err != nil {
				return nil, err
			}
			return call.Stream(ctx, c)
		}, closeStream)
	}), nil
}
