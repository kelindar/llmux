package anthropic

import (
	"fmt"
	"strings"

	"github.com/kelindar/llmux/chat"
)

type anthropicStream struct {
	writer     *sseWriter
	request    chat.Request
	meta       *responseMeta
	started    bool
	block      int
	textOpen   bool
	textID     string
	text       strings.Builder
	toolBlocks map[string]int
	toolNames  map[string]string
	toolCalls  map[string]string
	hadTool    bool
}

// Started reports whether the Anthropic SSE stream has begun.
func (s *anthropicStream) Started() bool { return s.writer.started }

func (s *anthropicStream) emit(eventType string, value map[string]any) error {
	value["type"] = eventType
	return s.writer.write(eventType, value)
}

func (s *anthropicStream) start() error {
	if s.started {
		return nil
	}
	if err := s.emit("message_start", map[string]any{"message": map[string]any{"id": s.meta.Response.ID, "type": "message", "role": "assistant", "model": s.meta.Response.Target, "content": []any{}, "stop_reason": nil, "stop_sequence": nil, "usage": map[string]any{}}}); err != nil {
		return err
	}
	s.started = true
	return nil
}

func (s *anthropicStream) openText() error {
	if s.textOpen {
		return nil
	}
	s.textOpen = true
	s.textID = newID()
	return s.emit("content_block_start", map[string]any{"index": s.block, "content_block": map[string]any{"type": "text", "text": ""}})
}

func (s *anthropicStream) closeText() error {
	if !s.textOpen {
		return nil
	}
	if err := s.emit("content_block_stop", map[string]any{"index": s.block}); err != nil {
		return err
	}
	s.block++
	s.textOpen = false
	s.textID = ""
	s.text.Reset()
	return nil
}

// Event encodes one canonical event as Anthropic SSE frames.
func (s *anthropicStream) Event(event chat.Event) error {
	if err := s.start(); err != nil {
		return err
	}
	switch event.Type {
	case chat.EventItem:
		switch event.Item.Type {
		case chat.ItemMessage:
			for _, part := range event.Item.Content {
				if part.Type != chat.PartText {
					return chat.Unsupported("output", "Anthropic stream supports text only")
				}
				if err := s.openText(); err != nil {
					return err
				}
				s.text.WriteString(part.Text)
				if err := s.emit("content_block_delta", map[string]any{"index": s.block, "delta": map[string]any{"type": "text_delta", "text": part.Text}}); err != nil {
					return err
				}
			}
			return s.closeText()
		case chat.ItemFunctionCall:
			if err := s.Event(chat.ToolStart(event.Item.CallID, event.Item.Name)); err != nil {
				return err
			}
			if event.Item.Arguments != "" {
				if err := s.emit("content_block_delta", map[string]any{"index": s.toolBlocks[event.Item.CallID], "delta": map[string]any{"type": "input_json_delta", "partial_json": event.Item.Arguments}}); err != nil {
					return err
				}
			}
			return s.Event(chat.ToolDone(event.Item.CallID))
		default:
			return chat.Unsupported("output", "unsupported Anthropic output item")
		}
	case chat.EventTextDelta:
		if err := s.openText(); err != nil {
			return err
		}
		s.text.WriteString(event.Delta)
		return s.emit("content_block_delta", map[string]any{"index": s.block, "delta": map[string]any{"type": "text_delta", "text": event.Delta}})
	case chat.EventTextDone:
		return s.closeText()
	case chat.EventToolCallStart:
		if err := s.closeText(); err != nil {
			return err
		}
		s.hadTool = true
		s.toolBlocks[event.CallID] = s.block
		s.toolNames[event.CallID] = event.Name
		s.toolCalls[event.CallID] = event.CallID
		if err := s.emit("content_block_start", map[string]any{"index": s.block, "content_block": map[string]any{"type": "tool_use", "id": event.CallID, "name": event.Name, "input": map[string]any{}}}); err != nil {
			return err
		}
		s.block++
		return nil
	case chat.EventToolCallDelta:
		index, ok := s.toolBlocks[event.CallID]
		if !ok {
			return fmt.Errorf("unknown tool call %q", event.CallID)
		}
		return s.emit("content_block_delta", map[string]any{"index": index, "delta": map[string]any{"type": "input_json_delta", "partial_json": event.Delta}})
	case chat.EventToolCallDone:
		index, ok := s.toolBlocks[event.CallID]
		if !ok {
			return fmt.Errorf("unknown tool call %q", event.CallID)
		}
		delete(s.toolBlocks, event.CallID)
		return s.emit("content_block_stop", map[string]any{"index": index})
	default:
		return fmt.Errorf("unsupported Anthropic event %q", event.Type)
	}
}

// Complete emits terminal Anthropic message_delta and message_stop events.
func (s *anthropicStream) Complete(outcome chat.Outcome, items []chat.Item) error {
	if err := s.start(); err != nil {
		return err
	}
	if err := s.closeText(); err != nil {
		return err
	}
	for callID, index := range s.toolBlocks {
		if err := s.emit("content_block_stop", map[string]any{"index": index}); err != nil {
			return err
		}
		delete(s.toolBlocks, callID)
	}
	stop := anthropicStopReason(outcome, s.hadTool || hasFunctionCall(items))
	usage := map[string]any{}
	if outcome.Usage != nil {
		usage["input_tokens"] = outcome.Usage.Input
		usage["output_tokens"] = outcome.Usage.Output
	}
	if err := s.emit("message_delta", map[string]any{"delta": map[string]any{"stop_reason": stop, "stop_sequence": nil}, "usage": usage}); err != nil {
		return err
	}
	if err := s.emit("message_stop", map[string]any{}); err != nil {
		return err
	}
	return nil
}

func hasFunctionCall(items []chat.Item) bool {
	for _, item := range items {
		if item.Type == chat.ItemFunctionCall {
			return true
		}
	}
	return false
}

// Fail writes an Anthropic error event or HTTP error envelope.
func (s *anthropicStream) Fail(err error) error {
	apiErr := asAPIError(err)
	if !s.writer.started {
		return writeProtocolErrorAndReturn(s.writer.w, protocolAnthropic, apiErr)
	}
	if startErr := s.start(); startErr != nil {
		return startErr
	}
	if err := s.emit("error", map[string]any{"error": map[string]any{"type": anthropicErrorType(apiErr), "message": apiErr.Message}}); err != nil {
		return err
	}
	return s.emit("message_stop", map[string]any{})
}
