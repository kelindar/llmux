package llmux

import (
	"errors"

	"github.com/kelindar/llmux/internal/responses"
)

// ResponsesBody builds the same Responses JSON envelope used for creation,
// idempotent replay, and application-owned GET retrieval.
//
// id and created are application-controlled identity fields. state carries
// status, output, usage, public error, incomplete reason, completion time,
// metadata, and effective store policy.
func ResponsesBody(req Request, state ResponseState, id string, created int64) (any, error) {
	return responses.Render(req, state, id, created)
}

// IsDelivery reports whether err is or wraps ErrDelivery.
func IsDelivery(err error) bool { return errors.Is(err, ErrDelivery) }
