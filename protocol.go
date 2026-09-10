package llmux

import (
	"context"
	"net/http"
	"time"

	"github.com/kelindar/llmux/internal/anthropic"
	"github.com/kelindar/llmux/internal/chat"
	"github.com/kelindar/llmux/internal/execution"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/kelindar/llmux/internal/responses"
)

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
		return chat.NewAdapter()
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
	limits  Limits
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

func asAPIError(err error) *APIError { return internalprotocol.AsAPIError(err) }

func anthropicErrorType(err *APIError) string { return internalprotocol.AnthropicErrorType(err) }

func (h *Handler) serveParsed(w http.ResponseWriter, r *http.Request, parsed parsedRequest) {
	agent, capabilities, err := h.resolve(r.Context(), parsed.Request.Target)
	if err != nil {
		h.logError(r.Context(), err)
		writeProtocolError(w, parsed.Kind, err)
		return
	}
	if err := h.validateRequest(&parsed.Request, capabilities); err != nil {
		writeProtocolError(w, parsed.Kind, err)
		return
	}
	if err := h.prepareRequest(r.Context(), &parsed.Request); err != nil {
		h.logError(r.Context(), err)
		writeProtocolError(w, parsed.Kind, err)
		return
	}
	if err := h.validateRequest(&parsed.Request, capabilities); err != nil {
		writeProtocolError(w, parsed.Kind, err)
		return
	}

	acceptance, accepted, err := h.acceptTurn(r, &parsed)
	if err != nil {
		writeProtocolError(w, parsed.Kind, err)
		return
	}

	meta := responseMeta{
		ID:       acceptance.ID,
		Created:  acceptance.Created,
		Model:    parsed.Request.Target,
		Activity: acceptance.Activity,
	}
	if meta.ID == "" {
		meta.ID = newID(responsePrefix(parsed.Kind))
	}
	if meta.Created == 0 {
		meta.Created = unixNow()
	}
	acceptance.ID = meta.ID
	acceptance.Created = meta.Created

	adapter := adapterFor(parsed.Kind)
	validateEvent := func(event Event) error {
		switch {
		case parsed.Kind == protocolResponses && requiresImageGeneration(event) && !parsed.Request.Controls.ImageGeneration:
			return Unsupported("output", "image output requires the Responses image_generation tool")
		case requiresReasoningSummary(event) && (parsed.Request.Controls.Reasoning == nil || !parsed.Request.Controls.Reasoning.Summary):
			return Unsupported("reasoning.summary", "reasoning summary output was not requested")
		case event.Type == EventActivity && !acceptance.Activity:
			return Unsupported("output", "activity events were not enabled for this request")
		}
		if err := adapter.ValidateEvent(event); err != nil {
			return err
		}
		if event.Type == EventActivity {
			return nil
		}
		if err := validateOutputEvent(event, capabilities); err != nil {
			return err
		}
		return validateRequestedOutput(event, parsed.Request.Output.Modalities)
	}

	if acceptance.Replay != nil {
		h.writeReplay(w, parsed, adapter, meta, *acceptance.Replay)
		return
	}
	if !accepted {
		// No lifecycle configured: fall through with library defaults.
	}

	runCtx := r.Context()
	if acceptance.Durable {
		if acceptance.Context != nil {
			runCtx = acceptance.Context
		} else {
			runCtx = context.WithoutCancel(r.Context())
		}
	}

	if parsed.Stream {
		h.serveStream(w, r, parsed, agent, adapter, meta, acceptance, runCtx, validateEvent)
		return
	}
	h.serveOrdinary(w, r, parsed, agent, adapter, meta, acceptance, runCtx, validateEvent)
}

func (h *Handler) acceptTurn(r *http.Request, parsed *parsedRequest) (Acceptance, bool, error) {
	if h.lifecycle == nil {
		return Acceptance{}, false, nil
	}
	accepted, err := h.lifecycle.Accept(r.Context(), &TurnRequest{
		Request:        &parsed.Request,
		Stream:         parsed.Stream,
		IdempotencyKey: r.Header.Get("Idempotency-Key"),
	})
	if err != nil {
		return Acceptance{}, true, err
	}
	return accepted, true, nil
}

func (h *Handler) writeReplay(w http.ResponseWriter, parsed parsedRequest, adapter protocolAdapter, meta responseMeta, replay Replay) {
	result := execution.Result{Items: cloneItems(replay.Output), Outcome: replay.Outcome}
	if parsed.Stream {
		stream := adapter.Stream(w, parsed.Request, meta, h.limits)
		for _, item := range result.Items {
			if err := stream.Event(OutputItem(item)); err != nil {
				h.logError(context.Background(), err)
				if stream.Started() {
					_ = stream.Fail(err)
				} else {
					writeProtocolError(w, parsed.Kind, err)
				}
				return
			}
		}
		if err := stream.Complete(result.Outcome, result.Items); err != nil {
			h.logError(context.Background(), err)
			if stream.Started() {
				_ = stream.Fail(err)
			} else {
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
	agent Agent,
	adapter protocolAdapter,
	meta responseMeta,
	acceptance Acceptance,
	runCtx context.Context,
	validateEvent func(Event) error,
) {
	stream := adapter.Stream(w, parsed.Request, meta, h.limits)
	delivering := true
	result, runErr := execution.Run(runCtx, &parsed.Request, agent, h.limits, func(event Event) error {
		if err := validateEvent(event); err != nil {
			return err
		}
		if !delivering {
			return nil
		}
		if err := stream.Event(event); err != nil {
			// Durable mode stops delivery on transport failure only; invalid
			// events and encoding errors still fail the agent.
			if acceptance.Durable && IsDelivery(err) {
				delivering = false
				return nil
			}
			return err
		}
		return nil
	})

	finalErr := h.finalizeTurn(runCtx, parsed, meta, acceptance, result, runErr)
	if runErr != nil {
		h.logError(r.Context(), runErr)
		if delivering && stream.Started() {
			_ = stream.Fail(runErr)
		} else if delivering {
			writeProtocolError(w, parsed.Kind, runErr)
		}
		return
	}
	if finalErr != nil {
		h.logError(r.Context(), finalErr)
		if delivering && stream.Started() {
			_ = stream.Fail(finalErr)
		} else if delivering {
			writeProtocolError(w, parsed.Kind, finalErr)
		}
		return
	}
	if !delivering {
		return
	}
	if err := stream.Complete(result.Outcome, result.Items); err != nil {
		h.logError(r.Context(), err)
		if stream.Started() {
			_ = stream.Fail(err)
		} else {
			writeProtocolError(w, parsed.Kind, err)
		}
	}
}

func (h *Handler) serveOrdinary(
	w http.ResponseWriter,
	r *http.Request,
	parsed parsedRequest,
	agent Agent,
	adapter protocolAdapter,
	meta responseMeta,
	acceptance Acceptance,
	runCtx context.Context,
	validateEvent func(Event) error,
) {
	result, runErr := execution.Run(runCtx, &parsed.Request, agent, h.limits, func(event Event) error {
		return validateEvent(event)
	})
	finalErr := h.finalizeTurn(runCtx, parsed, meta, acceptance, result, runErr)
	if runErr != nil {
		h.logError(r.Context(), runErr)
		writeProtocolError(w, parsed.Kind, runErr)
		return
	}
	if finalErr != nil {
		h.logError(r.Context(), finalErr)
		writeProtocolError(w, parsed.Kind, finalErr)
		return
	}
	body, err := adapter.Response(parsed.Request, result, meta)
	if err != nil {
		h.logError(r.Context(), err)
		writeProtocolError(w, parsed.Kind, err)
		return
	}
	writeJSON(w, http.StatusOK, body)
}

func (h *Handler) finalizeTurn(
	ctx context.Context,
	parsed parsedRequest,
	meta responseMeta,
	acceptance Acceptance,
	result execution.Result,
	runErr error,
) error {
	if h.lifecycle == nil {
		return nil
	}
	store := parsed.Request.Controls.Store != nil && *parsed.Request.Controls.Store
	return h.lifecycle.Finalize(ctx, &TurnResult{
		ID:      meta.ID,
		Created: meta.Created,
		Request: &parsed.Request,
		Output:  cloneItems(result.Items),
		Outcome: result.Outcome,
		Err:     runErr,
		Store:   store,
		Stream:  parsed.Stream,
	})
}

func responsePrefix(p protocol) string {
	switch p {
	case protocolResponses:
		return "resp_"
	case protocolAnthropic:
		return "msg_"
	default:
		return "chatcmpl_"
	}
}

func unixNow() int64 { return time.Now().Unix() }

func cloneItems(items []Item) []Item {
	if len(items) == 0 {
		return nil
	}
	out := make([]Item, len(items))
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
