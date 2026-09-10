package chat

import "context"

// ContinuationStore loads prior-turn history for previous_response_id.
// Persistence of new turns is owned by Lifecycle, not this interface.
type ContinuationStore interface {
	// Load returns history items that should precede the current Turn.
	// The application authorizes access and validates target ownership.
	Load(context.Context, string) ([]Item, error)
}

// Lifecycle is the optional application-owned response identity and
// persistence seam. Accept returns per-request state on Acceptance;
// Finish (when set) finalizes that request. Agent remains the sole
// execution owner — Acceptance does not run the agent.
//
// Ordering:
//  1. Validate request, resolve store policy, load continuation.
//  2. Accept — reserve identity, reject conflicts, or return Replay.
//     Capture request-local resources (idempotency key, cancel funcs)
//     on Acceptance.Finish rather than reconnecting via global maps.
//  3. Agent.Run (skipped on Replay); Durable detaches client cancel only.
//  4. Finish exactly once for accepted executions, with the execution
//     context. That context may already be cancelled. Finish owns any
//     detached, bounded cleanup work (for example
//     context.WithTimeout(context.WithoutCancel(ctx), timeout)).
//     Not called for Accept errors, completed Replays, or when Finish is nil.
//     Treat TurnResult.State as read-only: its nested data is shared with the
//     response encoded after Finish returns. Call State.Clone() before
//     retaining or modifying it.
//  5. Advertise success only after Finish succeeds.
//
// Activity: set Acceptance.Activity and emit Activity(name, json). Keep
// application-specific fields outside standard envelopes.
type Lifecycle interface {
	Accept(context.Context, *TurnRequest) (Acceptance, error)
}

// TurnRequest is the input to Lifecycle.Accept.
// Retention is on Request (Store wire value and Retain effective).
type TurnRequest struct {
	Request        *Request // Populated Turn, Input, Retain; treat as read-only.
	Stream         bool     // Whether the client requested streaming.
	IdempotencyKey string   // Idempotency-Key header value, if any.
}

// Acceptance is the per-request result of Lifecycle.Accept.
//
// Finish closes the request-local reservation. It should hold any resources
// that must be released or persisted for this acceptance (idempotency key,
// durable cancel, reserved record pointers). Context values from Accept remain
// available through the execution context passed to Finish; start cleanup
// deadlines when cleanup begins, not at Accept time.
type Acceptance struct {
	ID       string          // Response ID; empty uses a library default.
	Created  int64           // Unix created time; zero uses a library default.
	Replay   *ResponseState  // When set, skip Agent.Run and encode this result.
	Durable  bool            // Client disconnect does not cancel execution.
	Context  context.Context // Optional bounded run context when Durable.
	Activity bool            // When true, Responses may emit EventActivity.
	// Finish is called exactly once after Agent.Run for a new acceptance.
	// Nil when Replay is set or when no terminal persistence is needed.
	Finish func(context.Context, *TurnResult) error
}

// TurnResult is the input to Acceptance.Finish.
type TurnResult struct {
	ID      string   // Response ID that was accepted (after library defaults).
	Created int64    // Creation timestamp used in envelopes.
	Request *Request // Same request; Turn is the submitted input.
	// State is the client-visible terminal payload. Nested fields (Output,
	// Usage, Error, Metadata) are shared with the response encoded after
	// Finish returns. Treat State as read-only; call State.Clone() before
	// retaining or modifying it.
	State  ResponseState
	Err    error // Operational execution error, if any.
	Stream bool  // Whether the client requested streaming.
}
