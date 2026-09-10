package llmux

import (
	"errors"

	"github.com/kelindar/llmux/internal/responses"
)

// ResponsesBody builds the same Responses JSON envelope used for creation,
// idempotent replay, and application-owned GET retrieval.
func ResponsesBody(req Request, outcome Outcome, items []Item, id string, created int64) (any, error) {
	return responses.Render(req, outcome, items, id, created)
}

// IsDelivery reports whether err is or wraps ErrDelivery.
func IsDelivery(err error) bool { return errors.Is(err, ErrDelivery) }
