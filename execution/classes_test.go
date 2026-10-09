package execution_test

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	core "github.com/kayiik/llmgw-core"
	"github.com/kayiik/llmgw-core/execution"
)

// A rejected credential ends an execution by default. With
// CredentialFailover it moves on to a candidate whose credential may serve,
// its trace entry permits failover, and its instance is held against
// nothing: it answered.
func TestExecuteFailsOverPastARejectedCredentialWhenAsked(t *testing.T) {
	t.Parallel()
	for _, status := range []int{401, 403} {
		health := tracker(newClock(), execution.HealthPolicy{FailureThreshold: 1, OpenDuration: time.Minute})
		script := &calls{outcomes: map[string][]error{"a": {credentialRejected(status)}}}
		_, err := execution.Execute(context.Background(), execution.Executor[string]{Health: health, Key: identity}, []string{"a", "b"}, script.run)
		if core.ClassifyError(err).StatusCode != status || !slices.Equal(script.made(), []string{"a"}) {
			t.Fatalf("%d by default: err=%v calls=%v, want the execution ended", status, err, script.made())
		}

		script = &calls{outcomes: map[string][]error{"a": {credentialRejected(status)}}}
		executor := execution.Executor[string]{Health: health, Key: identity, CredentialFailover: true}
		result, err := execution.Execute(context.Background(), executor, []string{"a", "b"}, script.run)
		if err != nil || result.Candidate != "b" || !slices.Equal(script.made(), []string{"a", "b"}) {
			t.Fatalf("%d with credential failover: result=%+v err=%v calls=%v", status, result, err, script.made())
		}
		if first := result.Attempts[0]; first.Disposition != core.DispositionFailover ||
			(first.Class != core.ProviderErrorAuth && first.Class != core.ProviderErrorForbidden) {
			t.Fatalf("%d: rejected entry=%+v", status, first)
		}
		assertAvailable(t, health, "a")
	}
	// Other terminal failures still end the execution.
	script := &calls{outcomes: map[string][]error{"a": {rejected(400)}}}
	executor := execution.Executor[string]{CredentialFailover: true}
	if _, err := execution.Execute(context.Background(), executor, []string{"a", "b"}, script.run); err == nil || !slices.Equal(script.made(), []string{"a"}) {
		t.Fatalf("a bad request: err=%v calls=%v, want the execution ended", err, script.made())
	}
}

// admission refuses the keys it holds, until the time it holds them, and
// records each key it is asked about.
type admission struct {
	mu    sync.Mutex
	held  map[string]time.Time
	asked []string
}

func (a *admission) Admit(key string) (bool, time.Time) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.asked = append(a.asked, key)
	if until, held := a.held[key]; held {
		return false, until
	}
	return true, time.Time{}
}

// An Admission moves a candidate it refuses on before any try, as Health
// does with one it holds, and is asked only about the candidates Health
// admits. When it refuses every candidate, the execution is unavailable
// until the earliest one is admitted.
func TestExecuteAsksTheAdmissionBeforeATry(t *testing.T) {
	t.Parallel()
	clock := newClock()
	health := tracker(clock, execution.HealthPolicy{FailureThreshold: 1, OpenDuration: time.Minute})
	health.Record("unhealthy", overloaded())
	limiter := &admission{held: map[string]time.Time{"limited": clock.Now().Add(20 * time.Second)}}
	script := &calls{}
	executor := execution.Executor[string]{Health: health, Admission: limiter, Key: identity, Now: clock.Now}
	result, err := execution.Execute(context.Background(), executor, []string{"unhealthy", "limited", "open"}, script.run)
	if err != nil || result.Candidate != "open" || !slices.Equal(script.made(), []string{"open"}) {
		t.Fatalf("result=%+v err=%v calls=%v", result, err, script.made())
	}
	if !slices.Equal(limiter.asked, []string{"limited", "open"}) {
		t.Fatalf("the admission was asked about %v", limiter.asked)
	}
	if skipped := result.Attempts[1]; !skipped.Unadmitted || skipped.Unavailable || skipped.Tries != 0 ||
		!skipped.Until.Equal(clock.Now().Add(20*time.Second)) {
		t.Fatalf("refused entry=%+v", skipped)
	}

	limiter.held["open"] = clock.Now().Add(5 * time.Second)
	_, err = execution.Execute(context.Background(), executor, []string{"limited", "open"}, script.run)
	var unavailable *execution.UnavailableError
	if !errors.As(err, &unavailable) || unavailable.RetryAfter != 5*time.Second || core.ClassifyError(err).Disposition() != core.DispositionFailover {
		t.Fatalf("err=%v, want every candidate unavailable for 5s", err)
	}
	if _, err := execution.Execute(context.Background(), execution.Executor[string]{Admission: limiter}, []string{"a"}, script.run); err == nil {
		t.Fatal("an Admission without a Key was accepted")
	}
}

// The default Observe names a billing failure's class, so a Backoff holds
// a key that failed for billing back longer than one that was overloaded.
func TestHealthBackoffTellsBillingApart(t *testing.T) {
	t.Parallel()
	clock := newClock()
	health := execution.NewHealthTracker(execution.HealthOptions{Now: clock.Now, Policy: func(string) execution.HealthPolicy {
		return execution.HealthPolicy{FailureThreshold: 1, Backoff: func(failure execution.Failure) time.Duration {
			if failure.Class == string(core.ProviderErrorBilling) {
				return time.Hour
			}
			return time.Minute
		}}
	}})
	health.Record("billed", billing())
	health.Record("overloaded", overloaded())
	assertUnavailableUntil(t, health, "billed", clock.Now().Add(time.Hour))
	assertUnavailableUntil(t, health, "overloaded", clock.Now().Add(time.Minute))
	if state := health.State("billed"); state.ClassFailures["billing"] != 1 {
		t.Fatalf("billed state=%+v", state)
	}
}
