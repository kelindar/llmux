// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package llmux

import (
	"context"

	"github.com/kelindar/llmux/chat"
)

// Catalog is the application integration for agent discovery and loading.
// Both methods receive the authenticated request context.
//
// List returns caller-visible targets and their Info without constructing
// agents. Map keys are the targets accepted by Load. llmux reads the returned
// map and never mutates it or its values. List failures are operational errors
// and are never exposed to clients.
//
// Load obtains one executable agent and independently checks execution
// authorization. It returns current Info for capability validation; execution
// never calls List merely to obtain capabilities. Listing visibility never
// replaces Load authorization.
//
// A nil Catalog is allowed: List projects an empty catalog, and Load fails with
// a clear operational error. llmux never implements Load by scanning List and
// never caches catalogs across authenticated callers.
type Catalog interface {
	List(context.Context) (map[string]chat.Info, error)
	Load(context.Context, string) (chat.Agent, chat.Info, error)
}

// Store is the optional application integration for continuation and response
// lifecycle. Configure it with WithStore.
//
// Load authorizes access and returns the ordered history that should precede
// the current turn for previous_response_id. A Store that does not support
// continuation may return an appropriate error.
//
// Accept reserves new work, rejects conflicts, or returns an existing response
// for replay. Acceptance.Finish remains the per-request completion callback: it
// captures reserved application state and persists the terminal result.
// Implementations may share underlying persistence; llmux does not require
// coordination maps between Load and Accept.
//
// Ordering for a retained request:
//  1. Validate request, resolve store policy, load continuation via Load.
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
type Store interface {
	Load(context.Context, string) ([]chat.Item, error)
	Accept(context.Context, *chat.TurnRequest) (chat.Acceptance, error)
}
