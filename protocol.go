// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package llmux

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/anthropic"
	completions "github.com/kelindar/llmux/internal/completions"
	"github.com/kelindar/llmux/internal/execution"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/kelindar/llmux/internal/responses"
	"github.com/rs/xid"
)

func newID() string { return xid.New().String() }

type protocol = internalprotocol.Kind

const (
	protocolChat      = internalprotocol.Chat
	protocolResponses = internalprotocol.Responses
	protocolAnthropic = internalprotocol.Anthropic
)

type parsedRequest = internalprotocol.ParsedRequest
type responseMeta = internalprotocol.Meta
type protocolAdapter = internalprotocol.Adapter
type streamEncoder = internalprotocol.StreamEncoder

func adapterFor(p protocol) protocolAdapter {
	switch p {
	case protocolChat:
		return completions.NewAdapter()
	case protocolResponses:
		return responses.NewAdapter()
	case protocolAnthropic:
		return anthropic.NewAdapter()
	default:
		panic("llmux: unknown protocol")
	}
}

type sseWriter struct {
	w       http.ResponseWriter
	limits  chat.Limits
	inner   *internalprotocol.SSEWriter
	started bool
}

func (s *sseWriter) writer() *internalprotocol.SSEWriter {
	if s.inner == nil {
		s.inner = internalprotocol.NewSSEWriter(s.w, s.limits)
	}
	return s.inner
}

func (s *sseWriter) start() error {
	if err := s.writer().Start(); err != nil {
		return err
	}
	s.started = true
	return nil
}

func (s *sseWriter) write(event string, value any) error {
	if err := s.writer().Write(event, value); err != nil {
		return err
	}
	s.started = true
	return nil
}

func (s *sseWriter) done() error {
	if err := s.writer().Done(); err != nil {
		return err
	}
	s.started = true
	return nil
}

// Started reports whether any SSE bytes have been written for this stream.
func (s *sseWriter) Started() bool { return s.started || s.writer().Started() }

func writeProtocolError(w http.ResponseWriter, p protocol, err error) {
	internalprotocol.WriteError(w, p, err)
}

func asAPIError(err error) *chat.Error { return internalprotocol.AsError(err) }

func anthropicErrorType(err *chat.Error) string {
	return internalprotocol.AnthropicErrorType(err)
}

func (h *Handler) serveParsed(w http.ResponseWriter, r *http.Request, parsed parsedRequest) {
	agent, capabilities, err := h.resolve(r.Context(), parsed.Request.Target)
	if err != nil {
		h.logError(r.Context(), err)
		writeProtocolError(w, parsed.Kind, err)
		return
	}
	if err := h.validateParsed(&parsed, capabilities); err != nil {
		writeProtocolError(w, parsed.Kind, err)
		return
	}
	if err := h.prepareParsed(r.Context(), &parsed); err != nil {
		h.logError(r.Context(), err)
		writeProtocolError(w, parsed.Kind, err)
		return
	}
	if err := h.validateParsed(&parsed, capabilities); err != nil {
		writeProtocolError(w, parsed.Kind, err)
		return
	}

	acceptance, accepted, err := h.acceptTurn(r, &parsed)
	if err != nil {
		writeProtocolError(w, parsed.Kind, err)
		return
	}

	meta := responseMeta{
		Response: internalprotocol.InitialResponse(acceptance.Response, &parsed),
		Activity: acceptance.Activity,
	}
	acceptance.Response = meta.Response

	adapter := adapterFor(parsed.Kind)
	validateEvent := func(event chat.Event) error {
		switch {
		case parsed.Kind == protocolResponses && requiresImageGeneration(event) && !parsed.Request.Controls.ImageGeneration:
			return chat.Unsupported("output", "image output requires the Responses image_generation tool")
		case internalprotocol.RequiresReasoningSummary(event) && (parsed.Request.Controls.Reasoning == nil || !parsed.Request.Controls.Reasoning.Summary):
			return chat.Unsupported("reasoning.summary", "reasoning summary output was not requested")
		case event.Type == chat.EventActivity && !acceptance.Activity:
			return chat.Unsupported("output", "activity events were not enabled for this request")
		}
		if err := adapter.ValidateEvent(event); err != nil {
			return err
		}
		if event.Type == chat.EventActivity {
			return nil
		}
		if err := internalprotocol.ValidateOutputEvent(event, capabilities); err != nil {
			return err
		}
		return internalprotocol.ValidateRequestedOutput(event, parsed.Request.Output.Modalities)
	}

	if acceptance.Replay != nil {
		h.writeReplay(w, parsed, adapter, meta, acceptance.Replay.Clone())
		return
	}
	if !accepted {
		// No lifecycle configured: fall through with library defaults.
	}

	if parsed.Stream {
		if acceptance.RunTimeout < 0 {
			invalid := chat.Invalid("run_timeout", "run timeout must be zero or positive")
			if acceptance.Finish != nil {
				failed := meta.Response
				failed.Status = chat.StatusFailed
				failed.Error = publicAPIError(invalid)
				failed.CompletedAt = unixNow()
				_ = acceptance.Finish(r.Context(), &failed, invalid)
			}
			writeProtocolError(w, parsed.Kind, invalid)
			return
		}
		runCtx := r.Context()
		if acceptance.RunTimeout > 0 {
			var cancel context.CancelFunc
			runCtx, cancel = context.WithTimeout(context.WithoutCancel(r.Context()), acceptance.RunTimeout)
			defer cancel()
		}
		h.serveStream(w, r, parsed, agent, adapter, meta, acceptance, runCtx, validateEvent)
		return
	}
	h.serveOrdinary(w, r, parsed, agent, adapter, meta, acceptance, validateEvent)
}

func (h *Handler) acceptTurn(r *http.Request, parsed *parsedRequest) (chat.Acceptance, bool, error) {
	return h.acceptContext(r.Context(), r.Header.Get("Idempotency-Key"), parsed)
}

// acceptContext applies Store.Accept for one canonical request. Idempotency keys
// come from the calling protocol: the Idempotency-Key header for chat
// endpoints and none for MCP tool calls.
func (h *Handler) acceptContext(ctx context.Context, idempotencyKey string, parsed *parsedRequest) (chat.Acceptance, bool, error) {
	if h.store == nil {
		return chat.Acceptance{}, false, nil
	}
	accepted, err := h.store.Accept(ctx, &chat.TurnRequest{
		Request:        &parsed.Request,
		Turn:           parsed.Turn,
		Previous:       parsed.Previous,
		Metadata:       parsed.Metadata,
		Store:          parsed.Store,
		Retain:         parsed.Retain,
		Stream:         parsed.Stream,
		IdempotencyKey: idempotencyKey,
	})
	if err != nil {
		return chat.Acceptance{}, true, err
	}
	return accepted, true, nil
}

func (h *Handler) writeReplay(w http.ResponseWriter, parsed parsedRequest, adapter protocolAdapter, meta responseMeta, resp chat.Response) {
	meta.Response = resp
	result := execution.Result{
		Items:   resp.Output,
		Outcome: outcomeFromResponse(resp),
	}
	if parsed.Stream {
		stream := adapter.Stream(w, parsed, &meta, h.limits)
		for _, item := range result.Items {
			if err := stream.Event(chat.OutputItem(item)); err != nil {
				h.logError(context.Background(), err)
				switch {
				case stream.Started():
					_ = stream.Fail(err)
				default:
					writeProtocolError(w, parsed.Kind, err)
				}
				return
			}
		}
		if err := stream.Complete(result.Outcome, result.Items); err != nil {
			h.logError(context.Background(), err)
			switch {
			case stream.Started():
				_ = stream.Fail(err)
			default:
				writeProtocolError(w, parsed.Kind, err)
			}
		}
		return
	}
	body, err := adapter.Response(parsed.Request, result, meta)
	if err != nil {
		writeProtocolError(w, parsed.Kind, err)
		return
	}
	writeJSON(w, http.StatusOK, body)
}

func (h *Handler) serveStream(
	w http.ResponseWriter,
	r *http.Request,
	parsed parsedRequest,
	agent chat.Agent,
	adapter protocolAdapter,
	meta responseMeta,
	acceptance chat.Acceptance,
	runCtx context.Context,
	validateEvent func(chat.Event) error,
) {
	stream := adapter.Stream(w, parsed, &meta, h.limits)
	delivering := true
	result, runErr := execution.Run(runCtx, &parsed.Request, agent, h.limits, func(event chat.Event) error {
		if err := validateEvent(event); err != nil {
			return err
		}
		if !delivering {
			return nil
		}
		if err := stream.Event(event); err != nil {
			if acceptance.RunTimeout > 0 && errors.Is(err, chat.ErrDelivery) {
				delivering = false
				return nil
			}
			return err
		}
		return nil
	})

	resp := buildResponse(meta.Response, result, runErr)
	meta.Response = resp
	finalErr := h.finishTurn(runCtx, acceptance, &resp, runErr)
	if runErr != nil {
		h.logError(r.Context(), runErr)
		switch {
		case delivering && stream.Started():
			_ = stream.Fail(runErr)
		case delivering:
			writeProtocolError(w, parsed.Kind, runErr)
		}
		return
	}
	if finalErr != nil {
		h.logError(r.Context(), finalErr)
		switch {
		case delivering && stream.Started():
			_ = stream.Fail(finalErr)
		case delivering:
			writeProtocolError(w, parsed.Kind, finalErr)
		}
		return
	}
	if !delivering {
		return
	}
	if err := stream.Complete(outcomeFromResponse(resp), resp.Output); err != nil {
		h.logError(r.Context(), err)
		switch {
		case stream.Started():
			_ = stream.Fail(err)
		default:
			writeProtocolError(w, parsed.Kind, err)
		}
	}
}

func (h *Handler) serveOrdinary(
	w http.ResponseWriter,
	r *http.Request,
	parsed parsedRequest,
	agent chat.Agent,
	adapter protocolAdapter,
	meta responseMeta,
	acceptance chat.Acceptance,
	validateEvent func(chat.Event) error,
) {
	resp, err := h.executeOrdinary(r.Context(), parsed, agent, &meta, acceptance, validateEvent)
	if err != nil {
		h.logError(r.Context(), err)
		writeProtocolError(w, parsed.Kind, err)
		return
	}
	body, err := adapter.Response(parsed.Request, execution.Result{Items: resp.Output, Outcome: outcomeFromResponse(resp)}, meta)
	if err != nil {
		h.logError(r.Context(), err)
		writeProtocolError(w, parsed.Kind, err)
		return
	}
	writeJSON(w, http.StatusOK, body)
}

// executeOrdinary runs a fully validated parsed request to completion without
// streaming delivery. It owns bounded execution (Acceptance.RunTimeout),
// lifecycle Finish exactly once for accepted executions, and error
// sanitization. The returned response is finalized and safe to encode; err is
// the operational error to surface through the calling protocol. The response
// is persisted (Finish) before a successful result is returned.
func (h *Handler) executeOrdinary(
	ctx context.Context,
	parsed parsedRequest,
	agent chat.Agent,
	meta *responseMeta,
	acceptance chat.Acceptance,
	validateEvent func(chat.Event) error,
) (chat.Response, error) {
	if acceptance.RunTimeout < 0 {
		invalid := chat.Invalid("run_timeout", "run timeout must be zero or positive")
		if acceptance.Finish != nil {
			failed := meta.Response
			failed.Status = chat.StatusFailed
			failed.Error = publicAPIError(invalid)
			failed.CompletedAt = unixNow()
			_ = acceptance.Finish(ctx, &failed, invalid)
		}
		return chat.Response{}, invalid
	}

	runCtx := ctx
	if acceptance.RunTimeout > 0 {
		var cancel context.CancelFunc
		runCtx, cancel = context.WithTimeout(context.WithoutCancel(ctx), acceptance.RunTimeout)
		defer cancel()
	}

	result, runErr := execution.Run(runCtx, &parsed.Request, agent, h.limits, validateEvent)
	resp := buildResponse(meta.Response, result, runErr)
	meta.Response = resp
	finalErr := h.finishTurn(runCtx, acceptance, &resp, runErr)
	switch {
	case runErr != nil:
		return resp, runErr
	case finalErr != nil:
		return resp, finalErr
	}
	return resp, nil
}

func (h *Handler) finishTurn(
	runCtx context.Context,
	acceptance chat.Acceptance,
	resp *chat.Response,
	runErr error,
) error {
	if acceptance.Finish == nil {
		return nil
	}
	return acceptance.Finish(runCtx, resp, runErr)
}

func buildResponse(base chat.Response, result execution.Result, runErr error) chat.Response {
	resp := base
	// Take ownership of execution output; Result is not retained after this.
	resp.Output = result.Items
	resp.Usage = result.Outcome.Usage
	resp.CompletedAt = unixNow()
	switch {
	case runErr != nil:
		resp.Status = chat.StatusFailed
		if result.Outcome.Status == chat.StatusCancelled {
			resp.Status = chat.StatusCancelled
		}
		resp.Error = publicAPIError(runErr)
	case result.Outcome.Status != "":
		resp.Status = result.Outcome.Status
	default:
		resp.Status = chat.StatusCompleted
	}
	if resp.Status == chat.StatusIncomplete {
		resp.Incomplete = incompleteReason(result.Outcome.StopReason)
	}
	if resp.Status == chat.StatusInProgress {
		resp.CompletedAt = 0
	}
	return resp
}

func publicAPIError(err error) *chat.Error {
	api := asAPIError(err)
	if api == nil {
		return nil
	}
	copy := *api
	copy.Err = nil
	return &copy
}

func incompleteReason(reason chat.StopReason) string {
	switch reason {
	case chat.StopLength:
		return "max_output_tokens"
	case "":
		return "incomplete"
	default:
		return string(reason)
	}
}

func outcomeFromResponse(resp chat.Response) chat.Outcome {
	outcome := chat.Outcome{Status: resp.Status, Usage: resp.Usage}
	switch resp.Status {
	case chat.StatusFailed:
		outcome.StopReason = chat.StopError
	case chat.StatusCancelled:
		outcome.StopReason = chat.StopCancelled
	case chat.StatusIncomplete:
		if resp.Incomplete == "max_output_tokens" {
			outcome.StopReason = chat.StopLength
		}
	}
	return outcome
}

func unixNow() int64 { return time.Now().Unix() }

func cloneItems(items []chat.Item) []chat.Item {
	if len(items) == 0 {
		return nil
	}
	out := make([]chat.Item, len(items))
	for n, item := range items {
		out[n] = item.Clone()
	}
	return out
}

func (h *Handler) logError(ctx context.Context, err error) {
	if h.errorLog != nil && err != nil {
		h.errorLog(ctx, err)
	}
}
