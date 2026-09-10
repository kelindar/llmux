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
// persistence seam.
//
// Ordering:
//  1. Validate request, resolve store policy, load continuation.
//  2. Accept — reserve identity, reject conflicts, or return Replay.
//  3. Agent.Run (skipped on Replay); Durable detaches client cancel only.
//  4. Finalize exactly once for accepted executions (uses Acceptance.Finalize
//     when set). Not called for Accept errors or completed Replays.
//  5. Advertise success only after Finalize succeeds.
//
// Activity: set Acceptance.Activity and emit Activity(name, json). Keep
// application-specific fields outside standard envelopes.
type Lifecycle interface {
	Accept(context.Context, *TurnRequest) (Acceptance, error)
	Finalize(context.Context, *TurnResult) error
}

// TurnRequest is the input to Lifecycle.Accept.
// Retention is on Request (Store wire value and Retain effective).
type TurnRequest struct {
	Request        *Request // Populated Turn, Input, Retain; treat as read-only.
	Stream         bool     // Whether the client requested streaming.
	IdempotencyKey string   // Idempotency-Key header value, if any.
}

// Acceptance is the per-request result of Lifecycle.Accept.
type Acceptance struct {
	ID       string          // Response ID; empty uses a library default.
	Created  int64           // Unix created time; zero uses a library default.
	Replay   *ResponseState  // When set, skip Agent.Run and encode this result.
	Durable  bool            // Client disconnect does not cancel execution.
	Context  context.Context // Optional bounded run context when Durable.
	Finalize context.Context // Optional bounded cleanup context for Finalize.
	Activity bool            // When true, Responses may emit EventActivity.
}

// TurnResult is the input to Lifecycle.Finalize.
type TurnResult struct {
	ID      string        // Response ID that was accepted.
	Created int64         // Creation timestamp used in envelopes.
	Request *Request      // Same request; Turn is the submitted input.
	State   ResponseState // Client-visible terminal state.
	Err     error         // Operational execution error, if any.
	Stream  bool          // Whether the client requested streaming.
}
