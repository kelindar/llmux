// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package protocol

import (
	"cmp"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"time"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/execution"
	"github.com/rs/xid"
)

// Kind identifies one of the wire protocols served by the handler.
type Kind uint8

const (
	// Chat is the OpenAI Chat Completions wire protocol.
	Chat Kind = iota + 1
	// Responses is the OpenAI Responses wire protocol.
	Responses
	// Anthropic is the Anthropic Messages wire protocol.
	Anthropic
)

// ParsedRequest is the protocol-neutral request produced by a wire decoder.
// Request holds agent execution fields. Turn, Previous, Store, Metadata, and
// Retain are acceptance/persistence concerns filled by parsers and the handler.
type ParsedRequest struct {
	Kind         Kind              // wire protocol that parsed the request
	Request      chat.Request      // canonical execution request for Agent.Run
	Turn         []chat.Item       // items submitted in this HTTP request only
	Previous     *string           // prior response ID for continuation
	Store        *bool             // wire store flag; nil when omitted
	Metadata     map[string]string // application metadata for the response
	Retain       bool              // effective retention after StoreDefault
	Stream       bool              // whether the client requested streaming
	IncludeUsage bool              // Chat Completions stream_options.include_usage only
}

// Meta contains the response value shared by all protocol encoders.
type Meta struct {
	Response chat.Response // identity, status, output, and retrieval fields
	Activity bool          // when true, Responses may encode EventActivity frames
}

// StreamEncoder turns canonical events into one protocol's SSE lifecycle.
type StreamEncoder interface {
	// Event encodes one canonical agent event into protocol SSE frames.
	Event(chat.Event) error
	// Complete emits terminal stream events and closes the SSE lifecycle.
	Complete(chat.Outcome, []chat.Item) error
	// Fail writes a protocol-specific error response for the stream.
	Fail(error) error
	// Started reports whether SSE output has begun.
	Started() bool
}

// Adapter is the protocol-specific boundary used by the HTTP handler.
type Adapter interface {
	// ValidateEvent checks whether event can be encoded for this protocol.
	ValidateEvent(chat.Event) error
	// Response builds a non-streaming response body for result.
	Response(chat.Request, execution.Result, Meta) (any, error)
	// Stream returns an SSE encoder for the given response writer.
	// meta is retained for the stream lifetime so terminal Response
	// updates are visible to Complete.
	Stream(http.ResponseWriter, ParsedRequest, *Meta, chat.Limits) StreamEncoder
}

// SSEWriter owns the shared HTTP/SSE mechanics. Protocol codecs only provide
// event names and values; this type handles headers, event-size limits,
// flushing, and the OpenAI terminal marker.
type SSEWriter struct {
	w       http.ResponseWriter
	limits  chat.Limits
	started bool
}

// NewSSEWriter constructs an SSEWriter for the given response writer and limits.
func NewSSEWriter(w http.ResponseWriter, limits chat.Limits) *SSEWriter {
	return &SSEWriter{w: w, limits: limits}
}

// Start writes SSE headers and marks the stream as started.
func (s *SSEWriter) Start() error {
	if s.started {
		return nil
	}
	if _, ok := s.w.(http.Flusher); !ok {
		return errors.New("streaming requires http.Flusher")
	}
	s.w.Header().Set("Content-Type", "text/event-stream")
	s.w.Header().Set("Cache-Control", "no-cache")
	s.w.Header().Set("Connection", "keep-alive")
	s.w.Header().Set("X-Content-Type-Options", "nosniff")
	s.w.WriteHeader(http.StatusOK)
	s.started = true
	return nil
}

// Write marshals value as one SSE data frame, optionally prefixed with event.
func (s *SSEWriter) Write(event string, value any) error {
	data, err := json.Marshal(value)
	switch {
	case err != nil:
		return err
	case int64(len(data)) > s.limits.MaxEventBytes:
		return &chat.Error{Status: http.StatusRequestEntityTooLarge, Type: "invalid_request_error", Code: "event_too_large", Message: "stream event exceeds the configured limit"}
	}
	if err := s.Start(); err != nil {
		return err
	}
	switch {
	case event != "":
		if _, err := fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", event, data); err != nil {
			return fmt.Errorf("%w: %v", chat.ErrDelivery, err)
		}
	default:
		if _, err := fmt.Fprintf(s.w, "data: %s\n\n", data); err != nil {
			return fmt.Errorf("%w: %v", chat.ErrDelivery, err)
		}
	}
	s.w.(http.Flusher).Flush()
	return nil
}

// Done writes the OpenAI-compatible [DONE] marker. Anthropic and audio SSE
// codecs intentionally do not use it because their protocols have typed
// terminal events.
func (s *SSEWriter) Done() error {
	if err := s.Start(); err != nil {
		return err
	}
	if _, err := fmt.Fprint(s.w, "data: [DONE]\n\n"); err != nil {
		return fmt.Errorf("%w: %v", chat.ErrDelivery, err)
	}
	s.w.(http.Flusher).Flush()
	return nil
}

// Started reports whether SSE headers have been written.
func (s *SSEWriter) Started() bool { return s.started }

// WriteError writes the canonical error envelope for the selected protocol.
func WriteError(w http.ResponseWriter, kind Kind, err error) {
	apiErr := AsError(err)
	switch kind {
	case Anthropic:
		WriteJSON(w, apiErr.Status, map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    AnthropicErrorType(apiErr),
				"message": apiErr.Message,
			},
		})
	default:
		body := map[string]any{
			"error": map[string]any{
				"type":    apiErr.Type,
				"code":    apiErr.Code,
				"message": apiErr.Message,
			},
		}
		if apiErr.Param != "" {
			body["error"].(map[string]any)["param"] = apiErr.Param
		}
		WriteJSON(w, apiErr.Status, body)
	}
}

// AsError normalizes err into a chat.APIError with safe defaults.
func AsError(err error) *chat.Error {
	if err == nil {
		return &chat.Error{Status: http.StatusInternalServerError, Type: "server_error", Code: "server_error", Message: "internal server error"}
	}
	apiErr, ok := errors.AsType[*chat.Error](err)
	switch {
	case ok:
		copy := *apiErr
		if copy.Status < http.StatusBadRequest || copy.Status > 599 {
			copy.Status = http.StatusInternalServerError
		}
		copy.Type = cmp.Or(copy.Type, "server_error")
		copy.Code = cmp.Or(copy.Code, "server_error")
		copy.Message = cmp.Or(copy.Message, "internal server error")
		return &copy
	default:
		return &chat.Error{Status: http.StatusInternalServerError, Type: "server_error", Code: "server_error", Message: "internal server error", Err: err}
	}
}

// AnthropicErrorType maps a canonical API error to an Anthropic error type string.
func AnthropicErrorType(err *chat.Error) string {
	switch err.Type {
	case "invalid_request_error", "authentication_error", "permission_error", "not_found_error", "rate_limit_error", "api_error", "overloaded_error":
		return err.Type
	default:
		return "api_error"
	}
}

// WriteJSON marshals value as JSON with the given HTTP status.
func WriteJSON(w http.ResponseWriter, status int, value any) {
	data, err := json.Marshal(value)
	if err != nil {
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(append(data, '\n'))
}

// InitialResponse assembles the initial response envelope shared by every
// creation path, applying library defaults for identity and echoing the
// retrieval fields from the parsed request.
func InitialResponse(seed chat.Response, parsed *ParsedRequest) chat.Response {
	resp := seed
	if resp.ID == "" {
		resp.ID = xid.New().String()
	}
	if resp.Created == 0 {
		resp.Created = time.Now().Unix()
	}
	switch {
	case len(resp.Metadata) == 0:
		resp.Metadata = cloneMetadata(parsed.Metadata)
	default:
		resp.Metadata = cloneMetadata(resp.Metadata)
	}
	resp.Store = parsed.Retain
	resp.Target = parsed.Request.Target
	resp.Instructions = parsed.Request.Instructions
	if parsed.Previous != nil {
		p := *parsed.Previous
		resp.Previous = &p
	}
	return resp
}

func cloneMetadata(src map[string]string) map[string]string {
	if len(src) == 0 {
		return nil
	}
	return maps.Clone(src)
}

func RequiresReasoningSummary(event chat.Event) bool {
	return event.Type == chat.EventItem && event.Item.Type == chat.ItemReasoning
}

// ValidateOutputEvent checks an agent output event against what the selected
// agent declares it can provide.
func ValidateOutputEvent(event chat.Event, info chat.Info) error {
	info = info.Normalize()
	checkPart := func(part chat.Part) error {
		switch {
		case part.Type == chat.PartText && !info.OutputModalities.Has(chat.ModalityText):
			return chat.Unsupported("output", "selected agent does not provide text output")
		case part.Type == chat.PartImage && !info.OutputModalities.Has(chat.ModalityImage):
			return chat.Unsupported("output", "selected agent does not provide image output")
		case part.Type == chat.PartAudio && !info.OutputModalities.Has(chat.ModalityAudio):
			return chat.Unsupported("output", "selected agent does not provide audio output")
		case part.Type == chat.PartFile && !info.OutputModalities.Has(chat.ModalityFile):
			return chat.Unsupported("output", "selected agent does not provide file output")
		}
		return nil
	}
	checkItem := func(item chat.Item) error {
		switch item.Type {
		case chat.ItemMessage, chat.ItemMedia:
			for _, part := range item.Content {
				if err := checkPart(part); err != nil {
					return err
				}
			}
		case chat.ItemFunctionCall:
			if !info.ClientTools {
				return chat.Unsupported("tools", "selected agent does not hand tool calls to the client")
			}
		case chat.ItemFunctionCallOutput:
			for _, part := range item.Output {
				if err := checkPart(part); err != nil {
					return err
				}
			}
		case chat.ItemReasoning:
			if !info.ReasoningSummary {
				return chat.Unsupported("output", "selected agent does not provide reasoning summaries")
			}
		}
		return nil
	}
	switch event.Type {
	case chat.EventItem:
		return checkItem(event.Item)
	case chat.EventTextDelta:
		if !info.OutputModalities.Has(chat.ModalityText) {
			return chat.Unsupported("output", "selected agent does not provide text output")
		}
	case chat.EventToolCallStart, chat.EventToolCallDelta, chat.EventToolCallDone:
		if !info.ClientTools {
			return chat.Unsupported("tools", "selected agent does not hand tool calls to the client")
		}
	}
	return nil
}

// ValidateRequestedOutput checks an agent output event against the modalities
// the caller actually requested.
func ValidateRequestedOutput(event chat.Event, modalities chat.Modality) error {
	checkPart := func(part chat.Part) error {
		var modality chat.Modality
		switch part.Type {
		case chat.PartText:
			modality = chat.ModalityText
		case chat.PartImage:
			modality = chat.ModalityImage
		case chat.PartAudio:
			modality = chat.ModalityAudio
		case chat.PartFile:
			modality = chat.ModalityFile
		default:
			return nil
		}
		if !modalities.Has(modality) {
			return chat.Unsupported("modalities", "agent emitted an output modality that was not requested")
		}
		return nil
	}
	checkItem := func(item chat.Item) error {
		switch item.Type {
		case chat.ItemMessage, chat.ItemMedia:
			for _, part := range item.Content {
				if err := checkPart(part); err != nil {
					return err
				}
			}
		case chat.ItemFunctionCallOutput:
			for _, part := range item.Output {
				if err := checkPart(part); err != nil {
					return err
				}
			}
		}
		return nil
	}
	switch event.Type {
	case chat.EventItem:
		return checkItem(event.Item)
	case chat.EventTextDelta:
		return checkPart(chat.TextPart(event.Delta))
	}
	return nil
}
