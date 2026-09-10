package completions

import (
	"encoding/base64"
	"errors"
	"fmt"

	"github.com/kelindar/llmux/chat"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
)

type chatStream struct {
	writer       *sseWriter
	request      chat.Request
	meta         *responseMeta
	includeUsage bool
	started      bool
	roleSent     bool
	toolIndexes  map[string]int
	nextTool     int
	hadTool      bool
}

// Started reports whether the Chat Completions SSE stream has begun.
func (s *chatStream) Started() bool { return s.writer.started }

// Event encodes one canonical event as a Chat Completions chunk.
func (s *chatStream) Event(event chat.Event) error {
	if s.toolIndexes == nil {
		s.toolIndexes = make(map[string]int)
	}
	if event.Type == chat.EventTextDone || event.Type == chat.EventToolCallDone {
		return nil
	}
	chunk := map[string]any{"id": s.meta.ID, "object": "chat.completion.chunk", "created": s.meta.Created, "model": s.meta.Model, "choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": nil}}}
	if s.includeUsage {
		chunk["usage"] = nil
	}
	delta := chunk["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	switch event.Type {
	case chat.EventTextDelta:
		if !s.roleSent {
			delta["role"] = "assistant"
			s.roleSent = true
		}
		delta["content"] = event.Delta
		return s.writer.write("", chunk)
	case chat.EventToolCallStart:
		s.hadTool = true
		if !s.roleSent {
			delta["role"] = "assistant"
			s.roleSent = true
		}
		index := s.nextTool
		if existing, ok := s.toolIndexes[event.CallID]; ok {
			index = existing
		} else {
			s.toolIndexes[event.CallID] = index
			s.nextTool++
		}
		delta["tool_calls"] = []any{map[string]any{"index": index, "id": event.CallID, "type": "function", "function": map[string]any{"name": event.Name, "arguments": ""}}}
		return s.writer.write("", chunk)
	case chat.EventToolCallDelta:
		s.hadTool = true
		index, ok := s.toolIndexes[event.CallID]
		if !ok {
			return fmt.Errorf("unknown tool call %q", event.CallID)
		}
		delta["tool_calls"] = []any{map[string]any{"index": index, "function": map[string]any{"arguments": event.Delta}}}
		return s.writer.write("", chunk)
	case chat.EventItem:
		switch event.Item.Type {
		case chat.ItemMessage:
			for _, part := range event.Item.Content {
				switch part.Type {
				case chat.PartText:
					if !s.roleSent {
						delta["role"] = "assistant"
						s.roleSent = true
					}
					delta["content"] = part.Text
					if err := s.writer.write("", chunk); err != nil {
						return err
					}
				case chat.PartAudio:
					if part.Media == nil || len(part.Media.Data) == 0 {
						return errors.New("streamed audio output requires inline data")
					}
					if !s.roleSent {
						delta["role"] = "assistant"
						s.roleSent = true
					}
					delta["audio"] = map[string]any{"data": base64.StdEncoding.EncodeToString(part.Media.Data)}
					if err := s.writer.write("", chunk); err != nil {
						return err
					}
				default:
					return chat.Unsupported("output", "Chat Completions stream supports text and audio only")
				}
			}
		case chat.ItemFunctionCall:
			s.hadTool = true
			if !s.roleSent {
				delta["role"] = "assistant"
				s.roleSent = true
			}
			index := s.nextTool
			callID := event.Item.CallID
			if existing, ok := s.toolIndexes[callID]; ok {
				index = existing
			} else {
				s.toolIndexes[callID] = index
				s.nextTool++
			}
			delta["tool_calls"] = []any{map[string]any{"index": index, "id": callID, "type": "function", "function": map[string]any{"name": event.Item.Name, "arguments": event.Item.Arguments}}}
			return s.writer.write("", chunk)
		case chat.ItemMedia:
			if len(event.Item.Content) != 1 || event.Item.Content[0].Type != chat.PartAudio {
				return chat.Unsupported("output", "Chat Completions stream only supports audio media")
			}
			part := event.Item.Content[0]
			if part.Media == nil || len(part.Media.Data) == 0 {
				return errors.New("streamed audio output requires inline data")
			}
			if !s.roleSent {
				delta["role"] = "assistant"
				s.roleSent = true
			}
			delta["audio"] = map[string]any{"data": base64.StdEncoding.EncodeToString(part.Media.Data)}
			return s.writer.write("", chunk)
		case chat.ItemReasoning:
			return chat.Unsupported("output", "Chat Completions has no public reasoning-summary output mapping")
		default:
			return chat.Unsupported("output", "unsupported Chat Completions output item")
		}
	default:
		return fmt.Errorf("unsupported Chat Completions event %q", event.Type)
	}
	return nil
}

// Complete emits the final chunk and optional usage chunk for the stream.
func (s *chatStream) Complete(outcome chat.Outcome, items []chat.Item) error {
	finish := chatFinishReason(outcome, s.hadTool)
	chunk := map[string]any{"id": s.meta.ID, "object": "chat.completion.chunk", "created": s.meta.Created, "model": s.meta.Model, "choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": finish}}}
	if s.includeUsage {
		chunk["usage"] = nil
	}
	if err := s.writer.write("", chunk); err != nil {
		return err
	}
	if outcome.Usage != nil && s.includeUsage {
		usageChunk := map[string]any{"id": s.meta.ID, "object": "chat.completion.chunk", "created": s.meta.Created, "model": s.meta.Model, "choices": []any{}, "usage": chatUsage(outcome.Usage)}
		if err := s.writer.write("", usageChunk); err != nil {
			return err
		}
	}
	return s.writer.done()
}

// Fail writes a Chat Completions error chunk or HTTP error envelope.
func (s *chatStream) Fail(err error) error {
	apiErr := internalprotocol.AsAPIError(err)
	if !s.writer.started {
		return writeProtocolErrorAndReturn(s.writer.w, protocolChat, apiErr)
	}
	payload := map[string]any{"error": map[string]any{"message": apiErr.Message, "type": apiErr.Type, "code": apiErr.Code}}
	if apiErr.Param != "" {
		payload["error"].(map[string]any)["param"] = apiErr.Param
	}
	if err := s.writer.write("", payload); err != nil {
		return err
	}
	return s.writer.done()
}
