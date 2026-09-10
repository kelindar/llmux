package execution

import (
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"

	"github.com/kelindar/llmux/chat"
	"github.com/rs/xid"
)

func newID() string { return xid.New().String() }

type eventState struct {
	items     []chat.Item
	indexes   map[string]int
	textOpen  string
	tools     map[string]int
	toolArgs  map[string]*strings.Builder
	toolOrder []string
	bytes     int64
	maxBytes  int64
	maxMedia  int64
	maxEvent  int64
}

func newEventState(limits chat.Limits) *eventState {
	return &eventState{
		indexes:  make(map[string]int),
		tools:    make(map[string]int),
		toolArgs: make(map[string]*strings.Builder),
		maxBytes: limits.MaxOutputBytes,
		maxMedia: limits.MaxMediaBytes,
		maxEvent: limits.MaxEventBytes,
	}
}

func (s *eventState) addBytes(n int64) error {
	if n < 0 || n > s.maxEvent || s.bytes > s.maxBytes-n {
		return &chat.APIError{Status: http.StatusRequestEntityTooLarge, Type: "invalid_request_error", Code: "output_too_large", Message: "agent output exceeds the configured limit"}
	}
	s.bytes += n
	return nil
}

func (s *eventState) addItem(item chat.Item) error {
	if item.ID == "" {
		item.ID = newID()
	}
	if _, exists := s.indexes[item.ID]; exists {
		return fmt.Errorf("duplicate output item id %q", item.ID)
	}
	if item.Status == "" {
		item.Status = chat.StatusCompleted
	}
	if err := item.Validate(s.maxMedia, true); err != nil {
		return err
	}
	s.indexes[item.ID] = len(s.items)
	s.items = append(s.items, item.Clone())
	return nil
}

func (s *eventState) closeText(id string) (chat.Event, error) {
	if id == "" || s.textOpen != id {
		return chat.Event{}, errors.New("text item is not open")
	}
	index := s.indexes[id]
	item := &s.items[index]
	item.Status = chat.StatusCompleted
	s.textOpen = ""
	return chat.Event{Type: chat.EventTextDone, ItemID: id}, nil
}

func (s *eventState) closeOpenText(events *[]chat.Event) error {
	if s.textOpen == "" {
		return nil
	}
	event, err := s.closeText(s.textOpen)
	if err != nil {
		return err
	}
	*events = append(*events, event)
	return nil
}

func (s *eventState) apply(event chat.Event) ([]chat.Event, error) {
	if event.Type == "" {
		return nil, errors.New("event type is required")
	}
	if event.Type != chat.EventTextDelta && event.Type != chat.EventTextDone && event.Type != chat.EventToolCallDelta {
		if data, err := json.Marshal(event); err == nil && int64(len(data)) > s.maxEvent {
			return nil, &chat.APIError{Status: http.StatusRequestEntityTooLarge, Type: "invalid_request_error", Code: "event_too_large", Message: "agent event exceeds the configured limit"}
		}
	}
	var normalized []chat.Event
	switch event.Type {
	case chat.EventTextDelta:
		id := event.ItemID
		if id == "" {
			id = s.textOpen
		}
		if id == "" {
			id = newID()
		}
		if s.textOpen != "" && s.textOpen != id {
			if err := s.closeOpenText(&normalized); err != nil {
				return nil, err
			}
		}
		index, exists := s.indexes[id]
		if !exists {
			item := chat.Item{Type: chat.ItemMessage, ID: id, Status: chat.StatusInProgress, Role: chat.RoleAssistant, Content: []chat.Part{{Type: chat.PartText}}}
			if err := s.addItem(item); err != nil {
				return nil, err
			}
			index = s.indexes[id]
		} else if s.items[index].Type != chat.ItemMessage || s.items[index].Status != chat.StatusInProgress {
			return nil, fmt.Errorf("text item %q is not open", id)
		}
		if err := s.addBytes(int64(len(event.Delta))); err != nil {
			return nil, err
		}
		if len(s.items[index].Content) == 0 {
			s.items[index].Content = []chat.Part{{Type: chat.PartText}}
		}
		s.items[index].Content[0].Text += event.Delta
		s.textOpen = id
		event.ItemID = id
		normalized = append(normalized, event)

	case chat.EventTextDone:
		id := event.ItemID
		if id == "" {
			id = s.textOpen
		}
		closed, err := s.closeText(id)
		if err != nil {
			return nil, err
		}
		normalized = append(normalized, closed)

	case chat.EventToolCallStart:
		if err := s.closeOpenText(&normalized); err != nil {
			return nil, err
		}
		if event.CallID == "" || event.Name == "" {
			return nil, errors.New("tool call start requires call_id and name")
		}
		if _, exists := s.tools[event.CallID]; exists {
			return nil, fmt.Errorf("duplicate tool call id %q", event.CallID)
		}
		item := chat.Item{Type: chat.ItemFunctionCall, ID: newID(), Status: chat.StatusInProgress, CallID: event.CallID, Name: event.Name, Arguments: ""}
		if err := s.addItemUnchecked(item); err != nil {
			return nil, err
		}
		index := len(s.items) - 1
		s.tools[event.CallID] = index
		s.toolArgs[event.CallID] = &strings.Builder{}
		s.toolOrder = append(s.toolOrder, event.CallID)
		event.ItemID = item.ID
		normalized = append(normalized, event)

	case chat.EventToolCallDelta:
		index, exists := s.tools[event.CallID]
		if !exists {
			return nil, fmt.Errorf("tool call %q is not open", event.CallID)
		}
		if err := s.addBytes(int64(len(event.Delta))); err != nil {
			return nil, err
		}
		s.toolArgs[event.CallID].WriteString(event.Delta)
		s.items[index].Arguments = s.toolArgs[event.CallID].String()
		event.ItemID = s.items[index].ID
		normalized = append(normalized, event)

	case chat.EventToolCallDone:
		index, exists := s.tools[event.CallID]
		if !exists {
			return nil, fmt.Errorf("tool call %q is not open", event.CallID)
		}
		arguments := s.toolArgs[event.CallID].String()
		if arguments == "" || !jsontext.Value(arguments).IsValid() {
			return nil, fmt.Errorf("tool call %q ended with invalid JSON arguments", event.CallID)
		}
		s.items[index].Arguments = arguments
		s.items[index].Status = chat.StatusCompleted
		event.ItemID = s.items[index].ID
		delete(s.tools, event.CallID)
		delete(s.toolArgs, event.CallID)
		normalized = append(normalized, event)

	case chat.EventItem:
		if err := s.closeOpenText(&normalized); err != nil {
			return nil, err
		}
		if len(s.tools) > 0 {
			return nil, errors.New("tool call must be completed before another output item")
		}
		item := event.Item
		switch item.Type {
		case chat.ItemMessage:
			if err := s.addBytes(itemBytes(item)); err != nil {
				return nil, err
			}
			if item.Status == "" {
				item.Status = chat.StatusCompleted
			}
			if err := s.addItem(item); err != nil {
				return nil, err
			}
			event.Item = s.items[len(s.items)-1].Clone()
			event.ItemID = event.Item.ID
			normalized = append(normalized, event)

		case chat.ItemFunctionCall:
			if item.CallID == "" || item.Name == "" || item.Arguments == "" || !jsontext.Value(item.Arguments).IsValid() {
				return nil, errors.New("tool call requires call_id, name, and valid JSON arguments")
			}
			if _, exists := s.tools[item.CallID]; exists {
				return nil, fmt.Errorf("duplicate tool call id %q", item.CallID)
			}
			if err := s.addBytes(itemBytes(item)); err != nil {
				return nil, err
			}
			if err := s.addItem(item); err != nil {
				return nil, err
			}
			event.Item = s.items[len(s.items)-1].Clone()
			event.ItemID = event.Item.ID
			event.CallID = item.CallID
			event.Name = item.Name
			normalized = append(normalized, event)

		case chat.ItemMedia:
			if len(item.Content) != 1 {
				return nil, errors.New("media item requires exactly one content part")
			}
			part := item.Content[0]
			if part.Type != chat.PartImage && part.Type != chat.PartAudio {
				return nil, errors.New("media item must contain image or audio")
			}
			if err := part.Validate(s.maxMedia); err != nil {
				return nil, err
			}
			if part.Media != nil {
				if err := s.addBytes(int64(len(part.Media.Data))); err != nil {
					return nil, err
				}
			}
			if item.Status == "" {
				item.Status = chat.StatusCompleted
			}
			if item.Role == "" {
				item.Role = chat.RoleAssistant
			}
			if err := s.addItem(item); err != nil {
				return nil, err
			}
			event.Item = s.items[len(s.items)-1].Clone()
			event.ItemID = event.Item.ID
			normalized = append(normalized, event)

		case chat.ItemReasoning:
			if item.Status == "" {
				item.Status = chat.StatusCompleted
			}
			if err := s.addBytes(itemBytes(item)); err != nil {
				return nil, err
			}
			if err := s.addItem(item); err != nil {
				return nil, err
			}
			event.Item = s.items[len(s.items)-1].Clone()
			event.ItemID = event.Item.ID
			normalized = append(normalized, event)

		default:
			if err := s.addBytes(itemBytes(item)); err != nil {
				return nil, err
			}
			if err := s.addItem(item); err != nil {
				return nil, err
			}
			event.Item = s.items[len(s.items)-1].Clone()
			event.ItemID = event.Item.ID
			normalized = append(normalized, event)
		}

	case chat.EventActivity:
		if strings.TrimSpace(event.Name) == "" {
			return nil, errors.New("activity event requires a name")
		}
		if len(event.Data) == 0 || !event.Data.IsValid() {
			return nil, errors.New("activity event requires valid JSON data")
		}
		if err := s.addBytes(int64(len(event.Name) + len(event.Data))); err != nil {
			return nil, err
		}
		normalized = append(normalized, event)

	default:
		return nil, fmt.Errorf("unknown event type %q", event.Type)
	}
	return normalized, nil
}

func (s *eventState) addItemUnchecked(item chat.Item) error {
	if item.ID == "" {
		return errors.New("output item id is required")
	}
	if _, exists := s.indexes[item.ID]; exists {
		return fmt.Errorf("duplicate output item id %q", item.ID)
	}
	s.indexes[item.ID] = len(s.items)
	s.items = append(s.items, item.Clone())
	return nil
}

func (s *eventState) finish() ([]chat.Event, error) {
	var events []chat.Event
	if err := s.closeOpenText(&events); err != nil {
		return nil, err
	}
	for _, callID := range s.toolOrder {
		index, exists := s.tools[callID]
		if !exists {
			continue
		}
		arguments := s.toolArgs[callID].String()
		if arguments == "" || !jsontext.Value(arguments).IsValid() {
			return nil, fmt.Errorf("tool call %q ended with invalid JSON arguments", callID)
		}
		s.items[index].Arguments = arguments
		s.items[index].Status = chat.StatusCompleted
		events = append(events, chat.Event{Type: chat.EventToolCallDone, ItemID: s.items[index].ID, CallID: callID})
	}
	s.tools = make(map[string]int)
	s.toolArgs = make(map[string]*strings.Builder)
	s.toolOrder = nil
	return events, nil
}

func itemBytes(item chat.Item) int64 {
	var total int64
	for _, part := range item.Content {
		total += int64(len(part.Text) + len(part.Data))
		if part.Media != nil {
			total += int64(len(part.Media.Data))
		}
	}
	for _, part := range item.Output {
		total += int64(len(part.Text) + len(part.Data))
		if part.Media != nil {
			total += int64(len(part.Media.Data))
		}
	}
	for _, part := range item.Summary {
		total += int64(len(part.Text) + len(part.Data))
	}
	total += int64(len(item.Arguments) + len(item.EncryptedContent) + len(item.Data))
	return total
}

// Result is the normalized agent output collected during execution.
type Result struct {
	Items   []chat.Item
	Outcome chat.Outcome
}

func (s *eventState) result(outcome chat.Outcome) Result {
	items := make([]chat.Item, len(s.items))
	for n, item := range s.items {
		items[n] = item.Clone()
	}
	return Result{Items: items, Outcome: outcome}
}

// Run executes req with agent, normalizing emitted events and enforcing limits.
func Run(ctx context.Context, req *chat.Request, agent chat.Agent, limits chat.Limits, onEvent func(chat.Event) error) (Result, error) {
	if onEvent == nil {
		onEvent = func(chat.Event) error { return nil }
	}
	state := newEventState(limits)
	var active atomic.Bool
	var closed atomic.Bool
	var emitErr error
	emit := func(event chat.Event) error {
		if closed.Load() {
			return chat.ErrEmitClosed
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if !active.CompareAndSwap(false, true) {
			return chat.ErrConcurrentEmit
		}
		defer active.Store(false)
		normalized, err := state.apply(event)
		if err != nil {
			emitErr = err
			return err
		}
		for _, itemEvent := range normalized {
			if err := onEvent(itemEvent); err != nil {
				emitErr = err
				return err
			}
		}
		return nil
	}
	outcome, runErr := agent.Run(ctx, req, emit)
	closed.Store(true)
	switch {
	case runErr != nil:
		return state.result(outcome), runErr
	case emitErr != nil:
		return state.result(outcome), emitErr
	}
	if err := outcome.Validate(); err != nil {
		return state.result(outcome), err
	}
	finalEvents, err := state.finish()
	if err != nil {
		return state.result(outcome), err
	}
	for _, event := range finalEvents {
		if err := onEvent(event); err != nil {
			return state.result(outcome), err
		}
	}
	if outcome.Status == "" {
		outcome.Status = chat.StatusCompleted
	}
	if outcome.StopReason == "" {
		switch outcome.Status {
		case chat.StatusCancelled:
			outcome.StopReason = chat.StopCancelled
		case chat.StatusFailed:
			outcome.StopReason = chat.StopError
		case chat.StatusIncomplete:
			outcome.StopReason = chat.StopLength
		default:
			outcome.StopReason = chat.StopStop
			for _, item := range state.items {
				if item.Type == chat.ItemFunctionCall {
					outcome.StopReason = chat.StopToolCall
					break
				}
			}
		}
	}
	return state.result(outcome), nil
}
