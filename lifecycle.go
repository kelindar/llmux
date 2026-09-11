// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package llmux

import (
	"errors"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/responses"
)

// ResponsesBody builds the same Responses JSON envelope used for creation,
// idempotent replay, and application-owned GET retrieval from one Response.
func ResponsesBody(resp chat.Response) (any, error) {
	return responses.Render(resp)
}

// IsDelivery reports whether err is or wraps ErrDelivery.
func IsDelivery(err error) bool { return errors.Is(err, chat.ErrDelivery) }
