// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package llmux

import (
	"context"
	"maps"
	"net/http"
	"sync"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/agui"
)

// WithAGUI enables POST /ag-ui, a stored-turn AG-UI HTTP/SSE profile.
// Disabled by default. Requires WithStore; every new acceptance must supply
// Finish to persist its result before rich UI delivery or success.
func WithAGUI() Option { return func(h *Handler) { h.aguiEnabled = true } }

func (h *Handler) serveAGUI(w http.ResponseWriter, r *http.Request) {
	if h.store == nil {
		writeProtocolError(w, protocolAGUI, chat.Unsupported("store", "AG-UI requires a Store"))
		return
	}
	body, err := h.readBody(w, r, h.limits.MaxRequestBytes)
	if err != nil {
		writeProtocolError(w, protocolAGUI, err)
		return
	}
	parsed, err := agui.ParseRequest(body)
	if err != nil {
		writeProtocolError(w, protocolAGUI, err)
		return
	}
	if key := r.Header.Get("Idempotency-Key"); key != "" && key != parsed.RunID {
		writeProtocolError(w, protocolAGUI, chat.Invalid("runId", "runId must agree with Idempotency-Key"))
		return
	}
	h.serveParsed(w, r, parsed)
}

func activeRun(response chat.Response) *chat.Error {
	metadata := maps.Clone(response.Metadata)
	if metadata == nil {
		metadata = make(map[string]string)
	}
	metadata["response_id"] = response.ID
	return &chat.Error{Status: http.StatusConflict, Type: "invalid_request_error", Code: "run_in_progress", Message: "the original run is still in progress", Metadata: metadata}
}

// uiAgent validates extension payloads before execution retains them. Invalid
// generated UI must not enter the persisted partial response.
type uiAgent struct {
	agent    chat.Agent
	validate func(chat.Event) error
}

func (a uiAgent) Run(ctx context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
	var invalid error
	var mu sync.Mutex
	outcome, err := a.agent.Run(ctx, req, func(event chat.Event) error {
		if event.Type == chat.EventItem && event.Item.Type == chat.ItemExtension {
			if cause := a.validate(event); cause != nil {
				mu.Lock()
				invalid = cause
				mu.Unlock()
				return cause
			}
		}
		return emit(event)
	})
	mu.Lock()
	defer mu.Unlock()
	if err == nil {
		err = invalid
	}
	return outcome, err
}
