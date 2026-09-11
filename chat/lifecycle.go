// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package chat

import (
	"context"
	"time"
)

// TurnRequest is the input to Store.Accept (configured via llmux.WithStore).
//
// Request is the canonical execution request (read-only at Accept). Turn is the
// items submitted in this HTTP request only. When Previous is set, Request.Input
// equals Turn at Accept time: history is not merged yet. Retention,
// continuation, and idempotency are acceptance concerns, not Agent.Run inputs.
//
// CatalogAgent is the Agent returned by Catalog.Load for Request.Target. Store
// implementations may type-assert it to reuse load-time binding. Nil when no
// catalog is configured or Load was not performed for this turn.
type TurnRequest struct {
	Request        *Request          // Execution request; treat as read-only.
	Turn           []Item            // Items submitted in this request only.
	Previous       *string           // Prior response ID for continuation.
	Metadata       map[string]string // Application metadata for the response.
	Store          *bool             // Wire store flag; nil when omitted.
	Retain         bool              // Effective retention after StoreDefault.
	IdempotencyKey string            // Idempotency-Key header value, if any.
	Stream         bool              // Whether the client requested streaming.
	CatalogAgent   Agent             // Agent from Catalog.Load for this target.
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

	// Agent, when non-nil, is the request-specific executable for this
	// acceptance. It replaces the Catalog agent for Agent.Run only. Catalog.Load
	// still authorizes discovery-independent invocation and supplies Info.
	// When set, llmux does not load or merge continuation history into
	// Request.Input: the Agent owns effective input. Nil keeps the Catalog
	// agent and the ordinary post-Accept history merge for Previous.
	Agent Agent

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
