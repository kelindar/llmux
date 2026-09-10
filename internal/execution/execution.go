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

	"github.com/kelindar/llmux/contract"
	"github.com/kelindar/llmux/internal/identity"
)

func newID(prefix string) string { return identity.New(prefix) }

type eventState struct {
	items     []contract.Item
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

func newEventState(limits contract.Limits) *eventState {
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
		return &contract.APIError{Status: http.StatusRequestEntityTooLarge, Type: "invalid_request_error", Code: "output_too_large", Message: "agent output exceeds the configured limit"}
	}
	s.bytes += n
	return nil
}

func (s *eventState) addItem(item contract.Item) error {
	if item.ID == "" {
		prefix := "item_"
		switch item.Type {
		case contract.ItemMessage:
			prefix = "msg_"
		case contract.ItemFunctionCall:
			prefix = "fc_"
		case contract.ItemReasoning:
			prefix = "rs_"
		case contract.ItemMedia:
			prefix = "media_"
		}
		item.ID = newID(prefix)
	}
	if _, exists := s.indexes[item.ID]; exists {
		return fmt.Errorf("duplicate output item id %q", item.ID)
	}
	if item.Status == "" {
		item.Status = contract.StatusCompleted
	}
	if err := item.Validate(s.maxMedia, true); err != nil {
		return err
	}
	s.indexes[item.ID] = len(s.items)
	s.items = append(s.items, item.Clone())
	return nil
}

func (s *eventState) closeText(id string) (contract.Event, error) {
	if id == "" || s.textOpen != id {
		return contract.Event{}, errors.New("text item is not open")
	}
	index := s.indexes[id]
	item := &s.items[index]
	item.Status = contract.StatusCompleted
	s.textOpen = ""
	text := ""
	if len(item.Content) > 0 {
		text = item.Content[0].Text
	}
	return contract.Event{Type: contract.EventTextDone, ItemID: id, Text: text}, nil
}

func (s *eventState) closeOpenText(events *[]contract.Event) error {
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

func (s *eventState) apply(event contract.Event) ([]contract.Event, error) {
	if event.Type == "" {
		return nil, errors.New("event type is required")
	}
	if event.Type != contract.EventTextDelta && event.Type != contract.EventTextDone && event.Type != contract.EventToolCallDelta {
		if data, err := json.Marshal(event); err == nil && int64(len(data)) > s.maxEvent {
			return nil, &contract.APIError{Status: http.StatusRequestEntityTooLarge, Type: "invalid_request_error", Code: "event_too_large", Message: "agent event exceeds the configured limit"}
		}
	}
	var normalized []contract.Event
	switch event.Type {
	case contract.EventMessage:
		if err := s.closeOpenText(&normalized); err != nil {
			return nil, err
		}
		if event.Item.Type == "" {
			event.Item.Type = contract.ItemMessage
		}
		if event.Item.Type != contract.ItemMessage {
			return nil, errors.New("message event requires a message item")
		}
		if err := s.addBytes(itemBytes(event.Item)); err != nil {
			return nil, err
		}
		if event.Item.Status == "" {
			event.Item.Status = contract.StatusCompleted
		}
		if err := s.addItem(event.Item); err != nil {
			return nil, err
		}
		event.Item = s.items[len(s.items)-1].Clone()
		normalized = append(normalized, event)

	case contract.EventTextDelta:
		id := event.ItemID
		if id == "" {
			id = s.textOpen
		}
		if id == "" {
			id = newID("msg_")
		}
		if s.textOpen != "" && s.textOpen != id {
			if err := s.closeOpenText(&normalized); err != nil {
				return nil, err
			}
		}
		index, exists := s.indexes[id]
		if !exists {
			item := contract.Item{Type: contract.ItemMessage, ID: id, Status: contract.StatusInProgress, Role: contract.RoleAssistant, Content: []contract.Part{{Type: contract.PartText}}}
			if err := s.addItem(item); err != nil {
				return nil, err
			}
			index = s.indexes[id]
		} else if s.items[index].Type != contract.ItemMessage || s.items[index].Status != contract.StatusInProgress {
			return nil, fmt.Errorf("text item %q is not open", id)
		}
		if err := s.addBytes(int64(len(event.Delta))); err != nil {
			return nil, err
		}
		if len(s.items[index].Content) == 0 {
			s.items[index].Content = []contract.Part{{Type: contract.PartText}}
		}
		s.items[index].Content[0].Text += event.Delta
		s.textOpen = id
		event.ItemID = id
		normalized = append(normalized, event)

	case contract.EventTextDone:
		id := event.ItemID
		if id == "" {
			id = s.textOpen
		}
		closed, err := s.closeText(id)
		if err != nil {
			return nil, err
		}
		if event.Text != "" && event.Text != closed.Text {
			return nil, errors.New("text_done text does not match accumulated text")
		}
		event.Text = closed.Text
		event.ItemID = id
		normalized = append(normalized, event)

	case contract.EventToolCallStart:
		if err := s.closeOpenText(&normalized); err != nil {
			return nil, err
		}
		if event.CallID == "" || event.Name == "" {
			return nil, errors.New("tool call start requires call_id and name")
		}
		if _, exists := s.tools[event.CallID]; exists {
			return nil, fmt.Errorf("duplicate tool call id %q", event.CallID)
		}
		item := contract.Item{Type: contract.ItemFunctionCall, ID: newID("fc_"), Status: contract.StatusInProgress, CallID: event.CallID, Name: event.Name, Arguments: ""}
		if err := s.addItemUnchecked(item); err != nil {
			return nil, err
		}
		index := len(s.items) - 1
		s.tools[event.CallID] = index
		s.toolArgs[event.CallID] = &strings.Builder{}
		s.toolOrder = append(s.toolOrder, event.CallID)
		event.ItemID = item.ID
		normalized = append(normalized, event)

	case contract.EventToolCallDelta:
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

	case contract.EventToolCallDone:
		index, exists := s.tools[event.CallID]
		if !exists {
			return nil, fmt.Errorf("tool call %q is not open", event.CallID)
		}
		arguments := s.toolArgs[event.CallID].String()
		if event.Arguments != "" && event.Arguments != arguments {
			return nil, fmt.Errorf("tool call %q ended with arguments that do not match its deltas", event.CallID)
		}
		if arguments == "" || !jsontext.Value(arguments).IsValid() {
			return nil, fmt.Errorf("tool call %q ended with invalid JSON arguments", event.CallID)
		}
		s.items[index].Arguments = arguments
		s.items[index].Status = contract.StatusCompleted
		event.ItemID = s.items[index].ID
		event.Arguments = arguments
		delete(s.tools, event.CallID)
		delete(s.toolArgs, event.CallID)
		normalized = append(normalized, event)

	case contract.EventToolCall:
		if err := s.closeOpenText(&normalized); err != nil {
			return nil, err
		}
		if event.CallID == "" || event.Name == "" || event.Arguments == "" || !jsontext.Value(event.Arguments).IsValid() {
			return nil, errors.New("tool call requires call_id, name, and valid JSON arguments")
		}
		if _, exists := s.tools[event.CallID]; exists {
			return nil, fmt.Errorf("duplicate tool call id %q", event.CallID)
		}
		item := contract.FunctionCallItem(event.CallID, event.Name, event.Arguments)
		if err := s.addBytes(itemBytes(item)); err != nil {
			return nil, err
		}
		if err := s.addItem(item); err != nil {
			return nil, err
		}
		event.ItemID = s.items[len(s.items)-1].ID
		normalized = append(normalized, event)

	case contract.EventMedia:
		if err := s.closeOpenText(&normalized); err != nil {
			return nil, err
		}
		if event.Part.Type != contract.PartImage && event.Part.Type != contract.PartAudio {
			return nil, errors.New("media event requires an image or audio part")
		}
		if err := event.Part.Validate(s.maxMedia); err != nil {
			return nil, err
		}
		if err := s.addBytes(int64(len(event.Part.Media.Data))); err != nil {
			return nil, err
		}
		item := contract.Item{Type: contract.ItemMedia, ID: newID("media_"), Status: contract.StatusCompleted, Role: contract.RoleAssistant, Content: []contract.Part{event.Part.Clone()}}
		if err := s.addItem(item); err != nil {
			return nil, err
		}
		event.ItemID = item.ID
		event.Item = s.items[len(s.items)-1].Clone()
		normalized = append(normalized, event)

	case contract.EventReasoning:
		if err := s.closeOpenText(&normalized); err != nil {
			return nil, err
		}
		item := event.Item
		if item.Type == "" {
			item.Type = contract.ItemReasoning
		}
		if item.Type != contract.ItemReasoning {
			return nil, errors.New("reasoning event requires a reasoning item")
		}
		if len(item.Summary) == 0 && event.Text != "" {
			item.Summary = []contract.Part{contract.SummaryPart(event.Text)}
		}
		if item.Status == "" {
			item.Status = contract.StatusCompleted
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

	case contract.EventItem:
		if err := s.closeOpenText(&normalized); err != nil {
			return nil, err
		}
		if len(s.tools) > 0 {
			return nil, errors.New("tool call must be completed before another output item")
		}
		if err := s.addBytes(itemBytes(event.Item)); err != nil {
			return nil, err
		}
		if err := s.addItem(event.Item); err != nil {
			return nil, err
		}
		event.Item = s.items[len(s.items)-1].Clone()
		event.ItemID = event.Item.ID
		normalized = append(normalized, event)

	case contract.EventActivity:
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

func (s *eventState) addItemUnchecked(item contract.Item) error {
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

func (s *eventState) finish() ([]contract.Event, error) {
	var events []contract.Event
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
		s.items[index].Status = contract.StatusCompleted
		events = append(events, contract.Event{Type: contract.EventToolCallDone, ItemID: s.items[index].ID, CallID: callID, Arguments: arguments})
	}
	s.tools = make(map[string]int)
	s.toolArgs = make(map[string]*strings.Builder)
	s.toolOrder = nil
	return events, nil
}

func itemBytes(item contract.Item) int64 {
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
	Items   []contract.Item  // output items emitted by the agent
	Outcome contract.Outcome // final run status, stop reason, and usage
}

func (s *eventState) result(outcome contract.Outcome) Result {
	items := make([]contract.Item, len(s.items))
	for n, item := range s.items {
		items[n] = item.Clone()
	}
	return Result{Items: items, Outcome: outcome}
}

// Run executes req with agent, normalizing emitted events and enforcing limits.
func Run(ctx context.Context, req *contract.Request, agent contract.Agent, limits contract.Limits, onEvent func(contract.Event) error) (Result, error) {
	if onEvent == nil {
		onEvent = func(contract.Event) error { return nil }
	}
	state := newEventState(limits)
	var active atomic.Bool
	var closed atomic.Bool
	var emitErr error
	emit := func(event contract.Event) error {
		if closed.Load() {
			return contract.ErrEmitClosed
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if !active.CompareAndSwap(false, true) {
			return contract.ErrConcurrentEmit
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
	if runErr != nil {
		return state.result(outcome), runErr
	}
	if emitErr != nil {
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
		outcome.Status = contract.StatusCompleted
	}
	if outcome.StopReason == "" {
		switch outcome.Status {
		case contract.StatusCancelled:
			outcome.StopReason = contract.StopCancelled
		case contract.StatusFailed:
			outcome.StopReason = contract.StopError
		case contract.StatusIncomplete:
			outcome.StopReason = contract.StopLength
		default:
			outcome.StopReason = contract.StopStop
			for _, item := range state.items {
				if item.Type == contract.ItemFunctionCall {
					outcome.StopReason = contract.StopToolCall
					break
				}
			}
		}
	}
	return state.result(outcome), nil
}
