// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

// Package parse exposes wire-protocol decoders for applications that need
// canonical chat requests without hosting the full llmux HTTP server.
package parse

import (
	"bytes"
	"encoding/json/jsontext"

	"github.com/kelindar/llmux/chat"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/kelindar/llmux/internal/responses"
	"github.com/kelindar/llmux/internal/wire"
)

// Request is the protocol-neutral request produced by a wire decoder.
type Request = internalprotocol.ParsedRequest

// Responses decodes an OpenAI Responses API request body into a Request.
// Callers that accept non-x application fields should remove those keys
// before calling Responses; x- and namespaced keys are already allowed.
func Responses(raw jsontext.Value) (Request, error) {
	object, err := wire.DecodeObject(bytes.TrimSpace(raw))
	if err != nil {
		return Request{}, chat.Invalid("body", "request body must be a JSON object")
	}
	return responses.ParseRequest(object)
}
