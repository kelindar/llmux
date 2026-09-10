package llmux

import (
	"errors"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/responses"
)

// ResponsesBody builds the same Responses JSON envelope used for creation,
// idempotent replay, and application-owned GET retrieval.
func ResponsesBody(req chat.Request, state chat.ResponseState, id string, created int64) (any, error) {
	return responses.Render(req, state, id, created)
}

// IsDelivery reports whether err is or wraps ErrDelivery.
func IsDelivery(err error) bool { return errors.Is(err, chat.ErrDelivery) }
