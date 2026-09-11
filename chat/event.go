// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
package chat

import "encoding/json/jsontext"

// EventType tags an event emitted during Agent.Run.
type EventType string

const (
	EventTextDelta     EventType = "text_delta"
	EventTextDone      EventType = "text_done"
	EventToolCallStart EventType = "tool_call_start"
	EventToolCallDelta EventType = "tool_call_delta"
	EventToolCallDone  EventType = "tool_call_done"
	EventItem          EventType = "item"
	EventActivity      EventType = "activity"
)

// Event is the only output path from an Agent.
//
// TextDone and ToolCallDone are closure signals: they carry only the item or
// call ID. Execution accumulates streamed text and tool arguments and supplies
// finalized content to protocol encoders through canonical items and its own
// state. Complete output continues to use EventItem with a fully populated Item.
type Event struct {
	Type   EventType
	ItemID string
	CallID string
	Name   string
	Delta  string
	Item   Item
	Data   jsontext.Value
}

// Text returns an EventItem carrying a complete assistant text message.
func Text(text string) Event {
	return Event{Type: EventItem, Item: MessageItem(RoleAssistant, TextPart(text))}
}

// TextDelta returns an EventTextDelta with a text fragment.
func TextDelta(text string) Event { return Event{Type: EventTextDelta, Delta: text} }

// TextDone marks the end of a text stream for an item. Final text comes from
// execution's accumulated state, not from this event.
func TextDone(itemID string) Event {
	return Event{Type: EventTextDone, ItemID: itemID}
}

// Tool returns an EventItem carrying a completed function call.
func Tool(callID, name, arguments string) Event {
	return Event{Type: EventItem, Item: FunctionCallItem(callID, name, arguments)}
}

// ToolStart opens a streaming tool call.
func ToolStart(callID, name string) Event {
	return Event{Type: EventToolCallStart, CallID: callID, Name: name}
}

// ToolDelta carries incremental tool call arguments.
func ToolDelta(callID, delta string) Event {
	return Event{Type: EventToolCallDelta, CallID: callID, Delta: delta}
}

// ToolDone closes a streaming tool call. Final arguments come from execution's
// accumulated state, not from this event.
func ToolDone(callID string) Event { return Event{Type: EventToolCallDone, CallID: callID} }

// MediaItem returns an EventItem carrying a media output item.
func MediaItem(part Part) Event {
	return Event{
		Type: EventItem,
		Item: Item{
			Type:    ItemMedia,
			Status:  StatusCompleted,
			Role:    RoleAssistant,
			Content: []Part{part.Clone()},
		},
	}
}

// Reasoning returns an EventItem carrying a reasoning summary item.
func Reasoning(text string) Event {
	return Event{
		Type: EventItem,
		Item: Item{Type: ItemReasoning, Status: StatusCompleted, Summary: []Part{SummaryPart(text)}},
	}
}

// OutputItem returns an EventItem carrying an output item.
func OutputItem(item Item) Event { return Event{Type: EventItem, Item: item} }

// Activity builds an EventActivity with a structured JSON payload.
func Activity(name string, data jsontext.Value) Event {
	return Event{Type: EventActivity, Name: name, Data: append(jsontext.Value(nil), data...)}
}
