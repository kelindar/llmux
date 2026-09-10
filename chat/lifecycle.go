package chat

import (
	"context"
	"time"
)

// ContinuationStore loads prior-turn history for previous_response_id.
// Persistence of new turns is owned by Lifecycle, not this interface.
type ContinuationStore interface {
	// Load returns history items that should precede the current Turn.
	// The application authorizes access and validates target ownership.
	Load(context.Context, string) ([]Item, error)
}

// Lifecycle accepts one request and returns per-request acceptance state.
// Finish (when set) finalizes that request. Agent remains the sole execution
// owner — Lifecycle does not run the agent.
//
// Pass a method value to llmux.WithLifecycle, for example store.Accept.
//
// Ordering:
//  1. Validate request, resolve store policy, load continuation.
//  2. Accept — reserve identity, reject conflicts, or return Replay.
//     Capture request-local resources on Acceptance.Finish.
//  3. Agent.Run (skipped on Replay). RunTimeout > 0 detaches client cancel
//     and bounds execution; llmux owns that context and cancels it on exit.
//  4. Finish exactly once for accepted executions, with the execution
//     context. That context may already be cancelled. Finish owns any
//     detached, bounded cleanup work (for example
//     context.WithTimeout(context.WithoutCancel(ctx), timeout)).
//     Not called for Accept errors, completed Replays, or when Finish is nil.
//     Treat the Response as read-only: nested data is shared with the
//     response encoded after Finish returns. Call Response.Clone() before
//     retaining or modifying it.
//  5. Advertise success only after Finish succeeds.
//
// Activity: set Acceptance.Activity and emit Activity(name, json). Keep
// application-specific fields outside standard envelopes.
type Lifecycle func(context.Context, *TurnRequest) (Acceptance, error)

// TurnRequest is the input to Lifecycle.
//
// Request is the canonical execution request (read-only). Turn is the items
// submitted in this HTTP request only and is distinguishable from
// Request.Input (effective history+turn). Retention, continuation, and
// idempotency are acceptance concerns, not Agent.Run inputs.
type TurnRequest struct {
	Request        *Request          // Execution request; treat as read-only.
	Turn           []Item            // Items submitted in this request only.
	Previous       *string           // Prior response ID for continuation.
	Metadata       map[string]string // Application metadata for the response.
	Store          *bool             // Wire store flag; nil when omitted.
	Retain         bool              // Effective retention after StoreDefault.
	IdempotencyKey string            // Idempotency-Key header value, if any.
	Stream         bool              // Whether the client requested streaming.
}

// Acceptance is the per-request result of Lifecycle.
//
// For new work, Response carries identity (ID/Created; empty uses library
// defaults). For replay, Replay holds the complete stored Response and Run is
// skipped. Finish closes the request-local reservation and should capture any
// resources that must be released or persisted (idempotency key, turn items).
type Acceptance struct {
	Response Response  // Identity for new work; ignored when Replay is set.
	Replay   *Response // When set, skip Agent.Run and encode this result.

	// RunTimeout controls execution cancellation:
	//   0  — follow the HTTP request context (cancel on client disconnect)
	//   >0 — detach client cancel, preserve values, bound by this duration;
	//        llmux owns cancel and releases it on every exit. Delivery
	//        failures after disconnect do not cancel execution.
	//   <0 — invalid; the handler rejects and calls Finish when set so
	//        reserved resources cannot leak.
	RunTimeout time.Duration
	Activity   bool // When true, Responses may emit EventActivity.

	// Finish is called exactly once after Agent.Run for a new acceptance.
	// Nil when Replay is set or when no terminal persistence is needed.
	// The Response is read-only and shared with subsequent encoding; Clone
	// before retaining or modifying. err is the operational execution error.
	Finish func(context.Context, *Response, error) error
}
