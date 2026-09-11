// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
package chat

import (
	"context"
	"time"
)

// TurnRequest is the input to Store.Accept (configured via llmux.WithStore).
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

// Acceptance is the per-request result of Store.Accept.
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
