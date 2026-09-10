package chat

import (
	"errors"
	"fmt"
	"maps"
)

// StopReason explains why generation ended.
type StopReason string

const (
	StopStop      StopReason = "stop"
	StopToolCall  StopReason = "tool_call"
	StopLength    StopReason = "length"
	StopError     StopReason = "error"
	StopCancelled StopReason = "cancelled"
)

// Usage reports token accounting for a completed run.
// Total is supplied independently; callers are not required to set it to
// Input+Output or any other derived sum.
type Usage struct {
	Input     int
	Output    int
	Total     int
	Cached    int
	Reasoning int
}

// Outcome is the final agent result returned from Agent.Run.
type Outcome struct {
	Status     Status
	StopReason StopReason
	Usage      *Usage
}

// Validate checks an agent outcome.
func (o Outcome) Validate() error {
	switch {
	case o.Status != "" && !ValidStatus(o.Status):
		return fmt.Errorf("invalid outcome status %q", o.Status)
	case o.Status == StatusInProgress:
		return errors.New("outcome status cannot be in_progress")
	case o.Usage != nil && (o.Usage.Input < 0 || o.Usage.Output < 0 || o.Usage.Total < 0 || o.Usage.Cached < 0 || o.Usage.Reasoning < 0):
		return errors.New("outcome usage cannot contain negative values")
	}
	if o.StopReason != "" {
		switch o.StopReason {
		case StopStop, StopToolCall, StopLength, StopError, StopCancelled:
		default:
			return fmt.Errorf("invalid outcome stop reason %q", o.StopReason)
		}
	}
	return nil
}

// State is the client-visible response payload shared by creation,
// finalization, replay, and retrieval. Identity (ID) and creation time remain
// separate and application-controlled.
//
// When passed to Acceptance.Finish, nested data is shared with the response
// subsequently encoded. Finish callbacks must treat it as read-only and call
// Clone before retaining or modifying it.
//
// Error is sanitized public error only. Operational Go errors stay on
// TurnResult.Err and are never copied into Error.Message automatically.
type State struct {
	Status      Status            // completed, failed, incomplete, cancelled, or in_progress
	Output      []Item            // Output items for this response turn
	Usage       *Usage            // Token usage when known
	Error       *APIError         // Sanitized public error when status is failed
	Incomplete  string            // incomplete_details.reason when status is incomplete
	CompletedAt int64             // Unix completion time; zero while in_progress
	Metadata    map[string]string // Response metadata captured at acceptance
	Store       bool              // Effective content-retention policy
}

// Clone returns a deep copy safe for independent retention.
func (s State) Clone() State {
	out := s
	if s.Output != nil {
		out.Output = make([]Item, len(s.Output))
		for i, item := range s.Output {
			out.Output[i] = item.Clone()
		}
	}
	if s.Usage != nil {
		u := *s.Usage
		out.Usage = &u
	}
	if s.Error != nil {
		e := *s.Error
		e.Err = nil
		out.Error = &e
	}
	out.Metadata = CloneMetadata(s.Metadata)
	return out
}

// CloneMetadata returns a shallow copy of metadata, or nil when empty.
func CloneMetadata(src map[string]string) map[string]string {
	if len(src) == 0 {
		return nil
	}
	return maps.Clone(src)
}

// APIError is a sanitized protocol error that an application may return from a resolver or agent.
type APIError struct {
	Status  int
	Type    string
	Code    string
	Param   string
	Message string
	Err     error
}

// Error returns Message, the wrapped error text, or a default API error string.
func (e *APIError) Error() string {
	switch {
	case e == nil:
		return ""
	case e.Message != "":
		return e.Message
	case e.Err != nil:
		return e.Err.Error()
	default:
		return "llmux API error"
	}
}

// Unwrap returns the underlying error.
func (e *APIError) Unwrap() error { return e.Err }

// NewAPIError constructs an APIError with the given response fields.
func NewAPIError(status int, typ, code, param, message string) *APIError {
	return &APIError{Status: status, Type: typ, Code: code, Param: param, Message: message}
}

// Invalid constructs a 400 invalid_request_error for param.
func Invalid(param, message string) *APIError {
	return NewAPIError(400, "invalid_request_error", "invalid_request", param, message)
}

// Unsupported constructs a 400 unsupported invalid_request_error for param.
func Unsupported(param, message string) *APIError {
	return NewAPIError(400, "invalid_request_error", "unsupported", param, message)
}
