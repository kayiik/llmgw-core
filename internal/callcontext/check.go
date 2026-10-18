// Package callcontext carries an internal runtime validity fence through
// prepared decorators, so a retry after a wait validates the same binding.
package callcontext

import "context"

type key struct{}

func WithCheck(ctx context.Context, check func(context.Context) error) context.Context {
	return context.WithValue(ctx, key{}, check)
}

func Check(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if check, ok := ctx.Value(key{}).(func(context.Context) error); ok {
		if err := check(ctx); err != nil {
			return err
		}
	}
	return ctx.Err()
}
