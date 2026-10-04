// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package agui

import (
	"net/http"
	"strings"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/execution"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/kelindar/llmux/internal/wire"
)

// Adapter encodes canonical outputs as AG-UI 1.0 events.
type Adapter struct{}

// NewAdapter returns the AG-UI protocol adapter.
func NewAdapter() internalprotocol.Adapter { return Adapter{} }

// ValidateEvent checks whether event can be encoded for AG-UI.
func (Adapter) ValidateEvent(event chat.Event) error {
	switch event.Type {
	case chat.EventTextDelta, chat.EventTextDone:
		return nil
	case chat.EventActivity:
		if strings.TrimSpace(event.Name) == "" {
			return chat.Invalid("activity", "activity name is required")
		}
		if err := validateActivity(event.Data); err != nil {
			return err
		}
		return nil
	case chat.EventItem:
		switch event.Item.Type {
		case chat.ItemMessage:
			if err := event.Item.Validate(0, true); err != nil {
				return chat.Invalid("output", err.Error())
			}
			for _, part := range event.Item.Content {
				if part.Type != chat.PartText {
					return chat.Unsupported("output", "AG-UI message output supports text parts only")
				}
			}
			return nil
		case chat.ItemExtension:
			_, err := parseUIItem(event.Item)
			return err
		default:
			return chat.Unsupported("output", "AG-UI does not expose client tools or this output item")
		}
	default:
		return chat.Unsupported("output", "AG-UI does not expose client tools or this event")
	}
}

func validateActivity(data []byte) error {
	if err := wire.ValidateObject(data); err != nil {
		return chat.Invalid("activity", "activity content must be a JSON object")
	}
	return nil
}

// Response is unavailable because AG-UI uses a streaming run lifecycle.
func (Adapter) Response(chat.Request, execution.Result, internalprotocol.Meta) (any, error) {
	return nil, chat.Unsupported("stream", "AG-UI requires streaming")
}

// Stream returns an AG-UI event encoder.
func (Adapter) Stream(w http.ResponseWriter, parsed internalprotocol.ParsedRequest, meta *internalprotocol.Meta, limits chat.Limits) internalprotocol.StreamEncoder {
	return &stream{
		w:        internalprotocol.NewSSEWriter(w, limits),
		response: w,
		parsed:   parsed,
		meta:     meta,
		open:     make(map[string]bool),
		emitted:  make(map[string]bool),
	}
}
