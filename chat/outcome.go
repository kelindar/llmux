package chat

import (
	"errors"
	"fmt"
	"maps"
)

// StopReason explains why generation ended.
type StopReason string

const (
	StopNormal    StopReason = "stop"
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
	case o.Status != "" && !validStatus(o.Status):
		return fmt.Errorf("invalid outcome status %q", o.Status)
	case o.Status == StatusInProgress:
		return errors.New("outcome status cannot be in_progress")
	case o.Usage != nil && (o.Usage.Input < 0 || o.Usage.Output < 0 || o.Usage.Total < 0 || o.Usage.Cached < 0 || o.Usage.Reasoning < 0):
		return errors.New("outcome usage cannot contain negative values")
	}
	if o.StopReason != "" {
		switch o.StopReason {
		case StopNormal, StopToolCall, StopLength, StopError, StopCancelled:
		default:
			return fmt.Errorf("invalid outcome stop reason %q", o.StopReason)
		}
	}
	return nil
}

// Response is the complete client-visible response used for creation,
// finalization, replay, and retrieval. Identity, timestamps, output, and
// retrieval fields share one value so GET does not need the original
// execution request or accumulated history.
//
// When passed to Acceptance.Finish, nested data is shared with the response
// subsequently encoded. Finish callbacks must treat it as read-only and call
// Clone before retaining or modifying it.
//
// Error is sanitized public error only. Operational Go errors stay on the
// Finish error argument and are never copied into Error.Message automatically.
type Response struct {
	ID           string            // Response identifier
	Created      int64             // Unix creation time
	CompletedAt  int64             // Unix completion time; zero while in_progress
	Status       Status            // completed, failed, incomplete, cancelled, or in_progress
	Output       []Item            // Output items for this response turn
	Usage        *Usage            // Token usage when known
	Error        *Error            // Sanitized public error when status is failed
	Incomplete   string            // incomplete_details.reason when status is incomplete
	Metadata     map[string]string // Response metadata captured at acceptance
	Store        bool              // Effective content-retention policy
	Target       string            // Model/target echoed for retrieval
	Instructions string            // Instructions echoed for retrieval
	Previous     *string           // Parent response ID for retrieval
}

// Clone returns a deep copy safe for independent retention.
func (r Response) Clone() Response {
	out := r
	if r.Output != nil {
		out.Output = make([]Item, len(r.Output))
		for i, item := range r.Output {
			out.Output[i] = item.Clone()
		}
	}
	if r.Usage != nil {
		u := *r.Usage
		out.Usage = &u
	}
	if r.Error != nil {
		e := *r.Error
		e.Err = nil
		out.Error = &e
	}
	out.Metadata = cloneMetadata(r.Metadata)
	if r.Previous != nil {
		p := *r.Previous
		out.Previous = &p
	}
	return out
}

// cloneMetadata returns a shallow copy of metadata, or nil when empty.
func cloneMetadata(src map[string]string) map[string]string {
	if len(src) == 0 {
		return nil
	}
	return maps.Clone(src)
}

// Error is a sanitized protocol error that an application may return from a resolver or agent.
type Error struct {
	Status  int
	Type    string
	Code    string
	Param   string
	Message string
	Err     error
}

// Error returns Message, the wrapped error text, or a default API error string.
func (e *Error) Error() string {
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
func (e *Error) Unwrap() error { return e.Err }

// Invalid constructs a 400 invalid_request_error for param.
func Invalid(param, message string) *Error {
	return &Error{Status: 400, Type: "invalid_request_error", Code: "invalid_request", Param: param, Message: message}
}

// Unsupported constructs a 400 unsupported invalid_request_error for param.
func Unsupported(param, message string) *Error {
	return &Error{Status: 400, Type: "invalid_request_error", Code: "unsupported", Param: param, Message: message}
}
