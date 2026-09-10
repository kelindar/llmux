package protocol

import (
	json "encoding/json/v2"
	"errors"
	"fmt"
	"net/http"

	"github.com/kelindar/llmux/contract"
	"github.com/kelindar/llmux/internal/execution"
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
type ParsedRequest struct {
	Kind    Kind             // wire protocol that parsed the request
	Request contract.Request // canonical request payload
	Stream  bool             // whether the client requested streaming
}

// Meta contains response identity shared by all protocol encoders.
type Meta struct {
	ID       string // response identifier assigned by the server or Lifecycle
	Created  int64  // Unix timestamp when the response was created
	Model    string // resolved model name echoed to the client
	Activity bool   // when true, Responses may encode EventActivity frames
}

// StreamEncoder turns canonical events into one protocol's SSE lifecycle.
type StreamEncoder interface {
	// Event encodes one canonical agent event into protocol SSE frames.
	Event(contract.Event) error
	// Complete emits terminal stream events and closes the SSE lifecycle.
	Complete(contract.Outcome, []contract.Item) error
	// Fail writes a protocol-specific error response for the stream.
	Fail(error) error
	// Started reports whether SSE output has begun.
	Started() bool
}

// Adapter is the protocol-specific boundary used by the HTTP handler.
type Adapter interface {
	// ValidateEvent checks whether event can be encoded for this protocol.
	ValidateEvent(contract.Event) error
	// Response builds a non-streaming response body for result.
	Response(contract.Request, execution.Result, Meta) (any, error)
	// Stream returns an SSE encoder for the given response writer.
	Stream(http.ResponseWriter, contract.Request, Meta, contract.Limits) StreamEncoder
}

// SSEWriter owns the shared HTTP/SSE mechanics. Protocol codecs only provide
// event names and values; this type handles headers, event-size limits,
// flushing, and the OpenAI terminal marker.
type SSEWriter struct {
	w       http.ResponseWriter
	limits  contract.Limits
	started bool
}

// NewSSEWriter constructs an SSEWriter for the given response writer and limits.
func NewSSEWriter(w http.ResponseWriter, limits contract.Limits) *SSEWriter {
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
	if err != nil {
		return err
	}
	if int64(len(data)) > s.limits.MaxEventBytes {
		return contract.NewAPIError(http.StatusRequestEntityTooLarge, "invalid_request_error", "event_too_large", "", "stream event exceeds the configured limit")
	}
	if err := s.Start(); err != nil {
		return err
	}
	if event != "" {
		if _, err := fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", event, data); err != nil {
			return fmt.Errorf("%w: %v", contract.ErrDelivery, err)
		}
	} else if _, err := fmt.Fprintf(s.w, "data: %s\n\n", data); err != nil {
		return fmt.Errorf("%w: %v", contract.ErrDelivery, err)
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
		return fmt.Errorf("%w: %v", contract.ErrDelivery, err)
	}
	s.w.(http.Flusher).Flush()
	return nil
}

// Started reports whether SSE headers have been written.
func (s *SSEWriter) Started() bool { return s.started }

// WriteError writes the canonical error envelope for the selected protocol.
func WriteError(w http.ResponseWriter, kind Kind, err error) {
	apiErr := AsAPIError(err)
	if kind == Anthropic {
		WriteJSON(w, apiErr.Status, map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    AnthropicErrorType(apiErr),
				"message": apiErr.Message,
			},
		})
		return
	}
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

// AsAPIError normalizes err into a contract.APIError with safe defaults.
func AsAPIError(err error) *contract.APIError {
	if err == nil {
		return contract.NewAPIError(http.StatusInternalServerError, "server_error", "server_error", "", "internal server error")
	}
	if apiErr, ok := errors.AsType[*contract.APIError](err); ok {
		copy := *apiErr
		if copy.Status < http.StatusBadRequest || copy.Status > 599 {
			copy.Status = http.StatusInternalServerError
		}
		if copy.Type == "" {
			copy.Type = "server_error"
		}
		if copy.Code == "" {
			copy.Code = "server_error"
		}
		if copy.Message == "" {
			copy.Message = "internal server error"
		}
		return &copy
	}
	return &contract.APIError{Status: http.StatusInternalServerError, Type: "server_error", Code: "server_error", Message: "internal server error", Err: err}
}

// AnthropicErrorType maps a canonical API error to an Anthropic error type string.
func AnthropicErrorType(err *contract.APIError) string {
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
