package responses

import (
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

// Started reports whether the Responses SSE stream has begun.
func (s *responsesStream) Started() bool { return s.writer.started }

func (s *responsesStream) emit(eventType string, value map[string]any) error {
	value["type"] = eventType
	value["sequence_number"] = s.sequence
	s.sequence++
	return s.writer.write(eventType, value)
}

func (s *responsesStream) start() error {
	if s.started {
		return nil
	}
	state := s.meta.Response
	state.Status = chat.StatusInProgress
	state.CompletedAt = 0
	base := responseObject(s.request, state, []any{})
	if err := s.emit("response.created", map[string]any{"response": base}); err != nil {
		return err
	}
	if err := s.emit("response.in_progress", map[string]any{"response": base}); err != nil {
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
	return s.emit("response.output_item.added", map[string]any{"output_index": index, "item": map[string]any{"id": id, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{}}})
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
			parts := make([]any, 0, len(event.Item.Content))
			for contentIndex, part := range event.Item.Content {
				if part.Type != chat.PartText {
					return chat.Unsupported("output", "Responses message streaming only supports text parts")
				}
				parts = append(parts, map[string]any{"type": "output_text", "text": part.Text, "annotations": []any{}})
				if err := s.emit("response.content_part.added", map[string]any{"item_id": event.Item.ID, "output_index": index, "content_index": contentIndex, "part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}}}); err != nil {
					return err
				}
				if part.Text != "" {
					if err := s.emit("response.output_text.delta", map[string]any{"item_id": event.Item.ID, "output_index": index, "content_index": contentIndex, "delta": part.Text}); err != nil {
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
				if err := s.emit("response.function_call_arguments.delta", map[string]any{"item_id": event.ItemID, "output_index": index, "delta": event.Item.Arguments}); err != nil {
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
			reasoning := map[string]any{"id": item.ID, "type": "reasoning", "status": "in_progress", "summary": []any{}}
			if len(item.EncryptedContent) > 0 {
				reasoning["encrypted_content"] = jsontext.Value(item.EncryptedContent)
			}
			if err := s.emit("response.output_item.added", map[string]any{"output_index": index, "item": reasoning}); err != nil {
				return err
			}
			if len(item.Summary) > 0 {
				if err := s.emit("response.reasoning_summary_part.added", map[string]any{"item_id": item.ID, "output_index": index, "summary_index": 0}); err != nil {
					return err
				}
				for _, part := range item.Summary {
					if err := s.emit("response.reasoning_summary_text.delta", map[string]any{"item_id": item.ID, "output_index": index, "summary_index": 0, "delta": part.Text}); err != nil {
						return err
					}
				}
			}
			reasoning["status"] = "completed"
			reasoning["summary"] = reasoningSummary(item.Summary)
			return s.emit("response.output_item.done", map[string]any{"output_index": index, "item": reasoning})
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
			if err := s.emit("response.content_part.added", map[string]any{"item_id": event.ItemID, "output_index": index, "content_index": 0, "part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}}}); err != nil {
				return err
			}
		}
		s.text[event.ItemID] += event.Delta
		return s.emit("response.output_text.delta", map[string]any{"item_id": event.ItemID, "output_index": index, "content_index": 0, "delta": event.Delta})
	case chat.EventTextDone:
		index, ok := s.indexes[event.ItemID]
		if !ok {
			return fmt.Errorf("unknown text item %q", event.ItemID)
		}
		textValue := s.text[event.ItemID]
		if err := s.finishContentPart(event.ItemID, index, 0, textValue); err != nil {
			return err
		}
		return s.finishMessage(event.ItemID, index, []any{map[string]any{"type": "output_text", "text": textValue, "annotations": []any{}}})
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
		return s.emit("response.function_call_arguments.delta", map[string]any{"item_id": event.ItemID, "output_index": index, "delta": event.Delta})
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
		return s.emit("response.activity."+name, map[string]any{
			"activity": map[string]any{"name": name, "data": jsontext.Value(event.Data)},
		})
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
		image := map[string]any{"id": item.ID, "type": "image_generation_call", "status": "in_progress"}
		if err := s.emit("response.output_item.added", map[string]any{"output_index": index, "item": image}); err != nil {
			return err
		}
		image["status"] = "completed"
		image["result"] = base64.StdEncoding.EncodeToString(item.Content[0].Media.Data)
		return s.emit("response.output_item.done", map[string]any{"output_index": index, "item": image})
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
	return s.emit("response.output_item.added", map[string]any{"output_index": index, "item": map[string]any{"id": itemID, "type": "function_call", "status": "in_progress", "call_id": callID, "name": name, "arguments": ""}})
}

func (s *responsesStream) finishTool(itemID, callID, name, arguments string, index int) error {
	if callID == "" {
		callID = s.toolCalls[itemID]
	}
	if name == "" {
		name = s.toolNames[itemID]
	}
	if err := s.emit("response.function_call_arguments.done", map[string]any{"item_id": itemID, "output_index": index, "arguments": arguments}); err != nil {
		return err
	}
	if err := s.emit("response.output_item.done", map[string]any{"output_index": index, "item": map[string]any{"id": itemID, "type": "function_call", "status": "completed", "call_id": callID, "name": name, "arguments": arguments}}); err != nil {
		return err
	}
	delete(s.indexes, itemID)
	delete(s.toolNames, itemID)
	delete(s.toolCalls, itemID)
	return nil
}

func (s *responsesStream) finishMessage(itemID string, index int, parts []any) error {
	return s.emit("response.output_item.done", map[string]any{"output_index": index, "item": map[string]any{"id": itemID, "type": "message", "status": "completed", "role": "assistant", "content": parts}})
}

func (s *responsesStream) finishContentPart(itemID string, index, contentIndex int, textValue string) error {
	if err := s.emit("response.output_text.done", map[string]any{"item_id": itemID, "output_index": index, "content_index": contentIndex, "text": textValue}); err != nil {
		return err
	}
	if err := s.emit("response.content_part.done", map[string]any{"item_id": itemID, "output_index": index, "content_index": contentIndex, "part": map[string]any{"type": "output_text", "text": textValue, "annotations": []any{}}}); err != nil {
		return err
	}
	return nil
}

func reasoningSummary(parts []chat.Part) []any {
	values := make([]any, 0, len(parts))
	for _, part := range parts {
		values = append(values, map[string]any{"type": "summary_text", "text": part.Text})
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
	if state.Status == "" {
		state.Status = outcome.Status
	}
	if state.Usage == nil {
		state.Usage = outcome.Usage
	}
	if len(state.Output) == 0 {
		state.Output = items
	}
	event := "response.completed"
	switch state.Status {
	case chat.StatusFailed, chat.StatusCancelled:
		event = "response.failed"
	case chat.StatusIncomplete:
		event = "response.incomplete"
	}
	if err := s.emit(event, map[string]any{"response": responseObject(s.request, state, output)}); err != nil {
		return err
	}
	return s.writer.done()
}

// Fail writes a Responses error event or HTTP error envelope.
func (s *responsesStream) Fail(err error) error {
	apiErr := internalprotocol.AsAPIError(err)
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
	if err := s.emit("response.failed", map[string]any{"response": response}); err != nil {
		return err
	}
	return s.writer.done()
}
