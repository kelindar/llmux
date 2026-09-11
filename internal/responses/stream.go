// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package responses

import (
	"cmp"
	"encoding/base64"
	"encoding/json/jsontext"
	"fmt"
	"strings"

	"github.com/kelindar/llmux/chat"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
)

type responsesStream struct {
	writer    *sseWriter
	request   chat.Request
	meta      *responseMeta
	sequence  int
	indexes   map[string]int
	text      map[string]string
	toolArgs  map[string]string
	toolNames map[string]string
	toolCalls map[string]string
	next      int
	started   bool
}

type responseLifecycleEvent struct {
	Type           string       `json:"type"`
	SequenceNumber int          `json:"sequence_number"`
	Response       wireResponse `json:"response"`
}

type outputItemAddedEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`
	OutputIndex    int    `json:"output_index"`
	Item           any    `json:"item"`
}

type outputItemDoneEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`
	OutputIndex    int    `json:"output_index"`
	Item           any    `json:"item"`
}

type contentPartAddedEvent struct {
	Type           string         `json:"type"`
	SequenceNumber int            `json:"sequence_number"`
	ItemID         string         `json:"item_id"`
	OutputIndex    int            `json:"output_index"`
	ContentIndex   int            `json:"content_index"`
	Part           wireOutputText `json:"part"`
}

type contentPartDoneEvent struct {
	Type           string         `json:"type"`
	SequenceNumber int            `json:"sequence_number"`
	ItemID         string         `json:"item_id"`
	OutputIndex    int            `json:"output_index"`
	ContentIndex   int            `json:"content_index"`
	Part           wireOutputText `json:"part"`
}

type outputTextDeltaEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`
	ItemID         string `json:"item_id"`
	OutputIndex    int    `json:"output_index"`
	ContentIndex   int    `json:"content_index"`
	Delta          string `json:"delta"`
}

type outputTextDoneEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`
	ItemID         string `json:"item_id"`
	OutputIndex    int    `json:"output_index"`
	ContentIndex   int    `json:"content_index"`
	Text           string `json:"text"`
}

type functionCallArgumentsDeltaEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`
	ItemID         string `json:"item_id"`
	OutputIndex    int    `json:"output_index"`
	Delta          string `json:"delta"`
}

type functionCallArgumentsDoneEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`
	ItemID         string `json:"item_id"`
	OutputIndex    int    `json:"output_index"`
	Arguments      string `json:"arguments"`
}

type reasoningSummaryPartAddedEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`
	ItemID         string `json:"item_id"`
	OutputIndex    int    `json:"output_index"`
	SummaryIndex   int    `json:"summary_index"`
}

type reasoningSummaryTextDeltaEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`
	ItemID         string `json:"item_id"`
	OutputIndex    int    `json:"output_index"`
	SummaryIndex   int    `json:"summary_index"`
	Delta          string `json:"delta"`
}

type activityEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`
	Activity       struct {
		Name string         `json:"name"`
		Data jsontext.Value `json:"data"`
	} `json:"activity"`
}

// Started reports whether the Responses SSE stream has begun.
func (s *responsesStream) Started() bool { return s.writer.started }

func (s *responsesStream) nextSeq() int {
	n := s.sequence
	s.sequence++
	return n
}

func (s *responsesStream) emit(eventType string, v any) error {
	return s.writer.write(eventType, v)
}

func emptyOutputTextPart() wireOutputText {
	return wireOutputText{Type: "output_text", Text: "", Annotations: []any{}}
}

func outputTextPart(text string) wireOutputText {
	return wireOutputText{Type: "output_text", Text: text, Annotations: []any{}}
}

func (s *responsesStream) start() error {
	if s.started {
		return nil
	}
	state := s.meta.Response
	state.Status = chat.StatusInProgress
	state.CompletedAt = 0
	base := responseObject(s.request, state, []any{})
	if err := s.emit("response.created", responseLifecycleEvent{
		Type: "response.created", SequenceNumber: s.nextSeq(), Response: base,
	}); err != nil {
		return err
	}
	if err := s.emit("response.in_progress", responseLifecycleEvent{
		Type: "response.in_progress", SequenceNumber: s.nextSeq(), Response: base,
	}); err != nil {
		return err
	}
	s.started = true
	return nil
}

func (s *responsesStream) addMessage(id string, index int) error {
	if _, ok := s.indexes[id]; ok {
		return nil
	}
	s.indexes[id] = index
	if index >= s.next {
		s.next = index + 1
	}
	return s.emit("response.output_item.added", outputItemAddedEvent{
		Type: "response.output_item.added", SequenceNumber: s.nextSeq(), OutputIndex: index,
		Item: wireMessageItem{
			ID: id, Type: "message", Status: chat.StatusInProgress,
			Role: chat.RoleAssistant, Content: []wireOutputText{},
		},
	})
}

// Event encodes one canonical event as a Responses SSE frame.
func (s *responsesStream) Event(event chat.Event) error {
	if err := s.start(); err != nil {
		return err
	}
	switch event.Type {
	case chat.EventItem:
		switch event.Item.Type {
		case chat.ItemMessage:
			index := s.next
			if err := s.addMessage(event.Item.ID, index); err != nil {
				return err
			}
			if len(event.Item.Content) == 0 {
				return s.finishMessage(event.Item.ID, index, nil)
			}
			parts := make([]wireOutputText, 0, len(event.Item.Content))
			for contentIndex, part := range event.Item.Content {
				if part.Type != chat.PartText {
					return chat.Unsupported("output", "Responses message streaming only supports text parts")
				}
				parts = append(parts, outputTextPart(part.Text))
				if err := s.emit("response.content_part.added", contentPartAddedEvent{
					Type: "response.content_part.added", SequenceNumber: s.nextSeq(),
					ItemID: event.Item.ID, OutputIndex: index, ContentIndex: contentIndex,
					Part: emptyOutputTextPart(),
				}); err != nil {
					return err
				}
				if part.Text != "" {
					if err := s.emit("response.output_text.delta", outputTextDeltaEvent{
						Type: "response.output_text.delta", SequenceNumber: s.nextSeq(),
						ItemID: event.Item.ID, OutputIndex: index, ContentIndex: contentIndex,
						Delta: part.Text,
					}); err != nil {
						return err
					}
				}
				if err := s.finishContentPart(event.Item.ID, index, contentIndex, part.Text); err != nil {
					return err
				}
			}
			return s.finishMessage(event.Item.ID, index, parts)
		case chat.ItemFunctionCall:
			if err := s.startTool(event.ItemID, event.Item.CallID, event.Item.Name, s.next); err != nil {
				return err
			}
			index := s.indexes[event.ItemID]
			if event.Item.Arguments != "" {
				if s.toolArgs == nil {
					s.toolArgs = make(map[string]string)
				}
				s.toolArgs[event.ItemID] = event.Item.Arguments
				if err := s.emit("response.function_call_arguments.delta", functionCallArgumentsDeltaEvent{
					Type: "response.function_call_arguments.delta", SequenceNumber: s.nextSeq(),
					ItemID: event.ItemID, OutputIndex: index, Delta: event.Item.Arguments,
				}); err != nil {
					return err
				}
			}
			return s.finishTool(event.ItemID, event.Item.CallID, event.Item.Name, event.Item.Arguments, index)
		case chat.ItemReasoning:
			item := event.Item
			if len(item.Data) > 0 {
				return chat.Unsupported("output", "Responses reasoning output cannot carry opaque data")
			}
			if item.ID == "" {
				item.ID = event.ItemID
			}
			if item.ID == "" {
				item.ID = newID()
			}
			index := s.next
			s.indexes[item.ID] = index
			s.next++
			reasoning := wireReasoningItem{
				ID: item.ID, Type: "reasoning", Status: chat.StatusInProgress,
				Summary: []wireSummaryPart{},
			}
			if len(item.EncryptedContent) > 0 {
				reasoning.EncryptedContent = jsontext.Value(item.EncryptedContent)
			}
			if err := s.emit("response.output_item.added", outputItemAddedEvent{
				Type: "response.output_item.added", SequenceNumber: s.nextSeq(),
				OutputIndex: index, Item: reasoning,
			}); err != nil {
				return err
			}
			if len(item.Summary) > 0 {
				if err := s.emit("response.reasoning_summary_part.added", reasoningSummaryPartAddedEvent{
					Type: "response.reasoning_summary_part.added", SequenceNumber: s.nextSeq(),
					ItemID: item.ID, OutputIndex: index, SummaryIndex: 0,
				}); err != nil {
					return err
				}
				for _, part := range item.Summary {
					if err := s.emit("response.reasoning_summary_text.delta", reasoningSummaryTextDeltaEvent{
						Type: "response.reasoning_summary_text.delta", SequenceNumber: s.nextSeq(),
						ItemID: item.ID, OutputIndex: index, SummaryIndex: 0, Delta: part.Text,
					}); err != nil {
						return err
					}
				}
			}
			reasoning.Status = chat.StatusCompleted
			reasoning.Summary = reasoningSummary(item.Summary)
			return s.emit("response.output_item.done", outputItemDoneEvent{
				Type: "response.output_item.done", SequenceNumber: s.nextSeq(),
				OutputIndex: index, Item: reasoning,
			})
		case chat.ItemMedia:
			item := event.Item
			if item.ID == "" {
				item.ID = event.ItemID
			}
			return s.eventItem(item)
		default:
			return s.eventItem(event.Item)
		}
	case chat.EventTextDelta:
		index, ok := s.indexes[event.ItemID]
		if !ok {
			index = s.next
			if err := s.addMessage(event.ItemID, index); err != nil {
				return err
			}
			if err := s.emit("response.content_part.added", contentPartAddedEvent{
				Type: "response.content_part.added", SequenceNumber: s.nextSeq(),
				ItemID: event.ItemID, OutputIndex: index, ContentIndex: 0,
				Part: emptyOutputTextPart(),
			}); err != nil {
				return err
			}
		}
		s.text[event.ItemID] += event.Delta
		return s.emit("response.output_text.delta", outputTextDeltaEvent{
			Type: "response.output_text.delta", SequenceNumber: s.nextSeq(),
			ItemID: event.ItemID, OutputIndex: index, ContentIndex: 0, Delta: event.Delta,
		})
	case chat.EventTextDone:
		index, ok := s.indexes[event.ItemID]
		if !ok {
			return fmt.Errorf("unknown text item %q", event.ItemID)
		}
		textValue := s.text[event.ItemID]
		if err := s.finishContentPart(event.ItemID, index, 0, textValue); err != nil {
			return err
		}
		return s.finishMessage(event.ItemID, index, []wireOutputText{outputTextPart(textValue)})
	case chat.EventToolCallStart:
		return s.startTool(event.ItemID, event.CallID, event.Name, s.next)
	case chat.EventToolCallDelta:
		index, ok := s.indexes[event.ItemID]
		if !ok {
			return fmt.Errorf("unknown tool item %q", event.ItemID)
		}
		if s.toolArgs == nil {
			s.toolArgs = make(map[string]string)
		}
		s.toolArgs[event.ItemID] += event.Delta
		return s.emit("response.function_call_arguments.delta", functionCallArgumentsDeltaEvent{
			Type: "response.function_call_arguments.delta", SequenceNumber: s.nextSeq(),
			ItemID: event.ItemID, OutputIndex: index, Delta: event.Delta,
		})
	case chat.EventToolCallDone:
		index, ok := s.indexes[event.ItemID]
		if !ok {
			return fmt.Errorf("unknown tool item %q", event.ItemID)
		}
		return s.finishTool(event.ItemID, event.CallID, "", s.toolArgs[event.ItemID], index)
	case chat.EventActivity:
		if !s.meta.Activity {
			return chat.Unsupported("output", "activity events were not enabled for this request")
		}
		name := strings.TrimSpace(event.Name)
		if name == "" || strings.ContainsAny(name, "./ \t\r\n") {
			return chat.Invalid("activity", "activity name must be a non-empty token without separators")
		}
		evt := activityEvent{
			Type: "response.activity." + name, SequenceNumber: s.nextSeq(),
		}
		evt.Activity.Name = name
		evt.Activity.Data = jsontext.Value(event.Data)
		return s.emit("response.activity."+name, evt)
	default:
		return fmt.Errorf("unsupported Responses event %q", event.Type)
	}
}

func (s *responsesStream) eventItem(item chat.Item) error {
	switch item.Type {
	case chat.ItemMessage:
		return s.Event(chat.Event{Type: chat.EventItem, Item: item})
	case chat.ItemFunctionCall:
		return s.Event(chat.Event{Type: chat.EventItem, Item: item})
	case chat.ItemReasoning:
		return s.Event(chat.Event{Type: chat.EventItem, Item: item})
	case chat.ItemMedia:
		if len(item.Content) != 1 || item.Content[0].Type != chat.PartImage || item.Content[0].Media == nil || len(item.Content[0].Media.Data) == 0 {
			return chat.Unsupported("output", "Responses image output requires inline image data")
		}
		index := s.next
		s.next++
		image := wireImageItem{ID: item.ID, Type: "image_generation_call", Status: chat.StatusInProgress}
		if err := s.emit("response.output_item.added", outputItemAddedEvent{
			Type: "response.output_item.added", SequenceNumber: s.nextSeq(),
			OutputIndex: index, Item: image,
		}); err != nil {
			return err
		}
		image.Status = chat.StatusCompleted
		image.Result = base64.StdEncoding.EncodeToString(item.Content[0].Media.Data)
		return s.emit("response.output_item.done", outputItemDoneEvent{
			Type: "response.output_item.done", SequenceNumber: s.nextSeq(),
			OutputIndex: index, Item: image,
		})
	default:
		return chat.Unsupported("output", "unsupported Responses output item")
	}
}

func (s *responsesStream) startTool(itemID, callID, name string, index int) error {
	if _, ok := s.indexes[itemID]; ok {
		return nil
	}
	s.indexes[itemID] = index
	s.toolNames[itemID] = name
	s.toolCalls[itemID] = callID
	s.next = index + 1
	return s.emit("response.output_item.added", outputItemAddedEvent{
		Type: "response.output_item.added", SequenceNumber: s.nextSeq(), OutputIndex: index,
		Item: wireFunctionCallItem{
			ID: itemID, Type: "function_call", Status: chat.StatusInProgress,
			CallID: callID, Name: name, Arguments: "",
		},
	})
}

func (s *responsesStream) finishTool(itemID, callID, name, arguments string, index int) error {
	if callID == "" {
		callID = s.toolCalls[itemID]
	}
	if name == "" {
		name = s.toolNames[itemID]
	}
	if err := s.emit("response.function_call_arguments.done", functionCallArgumentsDoneEvent{
		Type: "response.function_call_arguments.done", SequenceNumber: s.nextSeq(),
		ItemID: itemID, OutputIndex: index, Arguments: arguments,
	}); err != nil {
		return err
	}
	if err := s.emit("response.output_item.done", outputItemDoneEvent{
		Type: "response.output_item.done", SequenceNumber: s.nextSeq(), OutputIndex: index,
		Item: wireFunctionCallItem{
			ID: itemID, Type: "function_call", Status: chat.StatusCompleted,
			CallID: callID, Name: name, Arguments: arguments,
		},
	}); err != nil {
		return err
	}
	delete(s.indexes, itemID)
	delete(s.toolNames, itemID)
	delete(s.toolCalls, itemID)
	return nil
}

func (s *responsesStream) finishMessage(itemID string, index int, parts []wireOutputText) error {
	return s.emit("response.output_item.done", outputItemDoneEvent{
		Type: "response.output_item.done", SequenceNumber: s.nextSeq(), OutputIndex: index,
		Item: wireMessageItem{
			ID: itemID, Type: "message", Status: chat.StatusCompleted,
			Role: chat.RoleAssistant, Content: parts,
		},
	})
}

func (s *responsesStream) finishContentPart(itemID string, index, contentIndex int, textValue string) error {
	part := outputTextPart(textValue)
	if err := s.emit("response.output_text.done", outputTextDoneEvent{
		Type: "response.output_text.done", SequenceNumber: s.nextSeq(),
		ItemID: itemID, OutputIndex: index, ContentIndex: contentIndex, Text: textValue,
	}); err != nil {
		return err
	}
	if err := s.emit("response.content_part.done", contentPartDoneEvent{
		Type: "response.content_part.done", SequenceNumber: s.nextSeq(),
		ItemID: itemID, OutputIndex: index, ContentIndex: contentIndex, Part: part,
	}); err != nil {
		return err
	}
	return nil
}

func reasoningSummary(parts []chat.Part) []wireSummaryPart {
	values := make([]wireSummaryPart, 0, len(parts))
	for _, part := range parts {
		values = append(values, wireSummaryPart{Type: "summary_text", Text: part.Text})
	}
	return values
}

// Complete emits the terminal Responses SSE event matching the response status
// and closes the stream. Failed and incomplete stored/replayed states use
// response.failed / response.incomplete rather than response.completed.
func (s *responsesStream) Complete(outcome chat.Outcome, items []chat.Item) error {
	output := make([]any, 0, len(items))
	for _, item := range items {
		value, err := responseItem(item)
		if err != nil {
			return err
		}
		output = append(output, value)
	}
	if err := s.start(); err != nil {
		return err
	}
	state := s.meta.Response
	state.Status = cmp.Or(state.Status, outcome.Status)
	if state.Usage == nil {
		state.Usage = outcome.Usage
	}
	if len(state.Output) == 0 {
		state.Output = items
	}
	eventType := "response.completed"
	switch state.Status {
	case chat.StatusFailed, chat.StatusCancelled:
		eventType = "response.failed"
	case chat.StatusIncomplete:
		eventType = "response.incomplete"
	}
	if err := s.emit(eventType, responseLifecycleEvent{
		Type: eventType, SequenceNumber: s.nextSeq(),
		Response: responseObject(s.request, state, output),
	}); err != nil {
		return err
	}
	return s.writer.done()
}

// Fail writes a Responses error event or HTTP error envelope.
func (s *responsesStream) Fail(err error) error {
	apiErr := internalprotocol.AsError(err)
	if !s.writer.started {
		return writeProtocolErrorAndReturn(s.writer.w, protocolResponses, apiErr)
	}
	if startErr := s.start(); startErr != nil {
		return startErr
	}
	state := s.meta.Response
	state.Status = chat.StatusFailed
	copy := *apiErr
	copy.Err = nil
	state.Error = &copy
	if state.CompletedAt == 0 {
		state.CompletedAt = s.meta.Response.Created
	}
	s.meta.Response = state
	response := responseObject(s.request, state, []any{})
	if err := s.emit("response.failed", responseLifecycleEvent{
		Type: "response.failed", SequenceNumber: s.nextSeq(), Response: response,
	}); err != nil {
		return err
	}
	return s.writer.done()
}
