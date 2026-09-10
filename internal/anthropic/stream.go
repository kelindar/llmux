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

type streamEvent struct {
	Type string `json:"type"`
}

type messageStartEvent struct {
	Type    string              `json:"type"`
	Message messageStartPayload `json:"message"`
}

type messageStartPayload struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Model        string         `json:"model"`
	Content      []any          `json:"content"`
	StopReason   any            `json:"stop_reason"`
	StopSequence any            `json:"stop_sequence"`
	Usage        map[string]int `json:"usage"`
}

type contentBlockStartEvent struct {
	Type         string `json:"type"`
	Index        int    `json:"index"`
	ContentBlock any    `json:"content_block"`
}

type textContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type toolContentBlock struct {
	Type  string         `json:"type"`
	ID    string         `json:"id"`
	Name  string         `json:"name"`
	Input map[string]any `json:"input"`
}

type contentBlockDeltaEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta any    `json:"delta"`
}

type textDelta struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type inputJSONDelta struct {
	Type        string `json:"type"`
	PartialJSON string `json:"partial_json"`
}

type contentBlockStopEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

type messageDeltaEvent struct {
	Type  string           `json:"type"`
	Delta messageDeltaBody `json:"delta"`
	Usage map[string]int   `json:"usage"`
}

type messageDeltaBody struct {
	StopReason   string `json:"stop_reason"`
	StopSequence any    `json:"stop_sequence"`
}

type errorEvent struct {
	Type  string       `json:"type"`
	Error errorPayload `json:"error"`
}

type errorPayload struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// Started reports whether the Anthropic SSE stream has begun.
func (s *anthropicStream) Started() bool { return s.writer.started }

func (s *anthropicStream) emit(eventType string, v any) error {
	return s.writer.write(eventType, v)
}

func (s *anthropicStream) start() error {
	if s.started {
		return nil
	}
	if err := s.emit("message_start", messageStartEvent{
		Type: "message_start",
		Message: messageStartPayload{
			ID:      s.meta.Response.ID,
			Type:    "message",
			Role:    "assistant",
			Model:   s.meta.Response.Target,
			Content: []any{},
			Usage:   map[string]int{},
		},
	}); err != nil {
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
	return s.emit("content_block_start", contentBlockStartEvent{
		Type:         "content_block_start",
		Index:        s.block,
		ContentBlock: textContentBlock{Type: "text", Text: ""},
	})
}

func (s *anthropicStream) closeText() error {
	if !s.textOpen {
		return nil
	}
	if err := s.emit("content_block_stop", contentBlockStopEvent{Type: "content_block_stop", Index: s.block}); err != nil {
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
				if err := s.emit("content_block_delta", contentBlockDeltaEvent{
					Type:  "content_block_delta",
					Index: s.block,
					Delta: textDelta{Type: "text_delta", Text: part.Text},
				}); err != nil {
					return err
				}
			}
			return s.closeText()
		case chat.ItemFunctionCall:
			if err := s.Event(chat.ToolStart(event.Item.CallID, event.Item.Name)); err != nil {
				return err
			}
			if event.Item.Arguments != "" {
				if err := s.emit("content_block_delta", contentBlockDeltaEvent{
					Type:  "content_block_delta",
					Index: s.toolBlocks[event.Item.CallID],
					Delta: inputJSONDelta{Type: "input_json_delta", PartialJSON: event.Item.Arguments},
				}); err != nil {
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
		return s.emit("content_block_delta", contentBlockDeltaEvent{
			Type:  "content_block_delta",
			Index: s.block,
			Delta: textDelta{Type: "text_delta", Text: event.Delta},
		})
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
		if err := s.emit("content_block_start", contentBlockStartEvent{
			Type:  "content_block_start",
			Index: s.block,
			ContentBlock: toolContentBlock{
				Type:  "tool_use",
				ID:    event.CallID,
				Name:  event.Name,
				Input: map[string]any{},
			},
		}); err != nil {
			return err
		}
		s.block++
		return nil
	case chat.EventToolCallDelta:
		index, ok := s.toolBlocks[event.CallID]
		if !ok {
			return fmt.Errorf("unknown tool call %q", event.CallID)
		}
		return s.emit("content_block_delta", contentBlockDeltaEvent{
			Type:  "content_block_delta",
			Index: index,
			Delta: inputJSONDelta{Type: "input_json_delta", PartialJSON: event.Delta},
		})
	case chat.EventToolCallDone:
		index, ok := s.toolBlocks[event.CallID]
		if !ok {
			return fmt.Errorf("unknown tool call %q", event.CallID)
		}
		delete(s.toolBlocks, event.CallID)
		return s.emit("content_block_stop", contentBlockStopEvent{Type: "content_block_stop", Index: index})
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
		if err := s.emit("content_block_stop", contentBlockStopEvent{Type: "content_block_stop", Index: index}); err != nil {
			return err
		}
		delete(s.toolBlocks, callID)
	}
	stop := anthropicStopReason(outcome, s.hadTool || hasFunctionCall(items))
	usage := map[string]int{}
	if outcome.Usage != nil {
		usage["input_tokens"] = outcome.Usage.Input
		usage["output_tokens"] = outcome.Usage.Output
	}
	if err := s.emit("message_delta", messageDeltaEvent{
		Type:  "message_delta",
		Delta: messageDeltaBody{StopReason: stop},
		Usage: usage,
	}); err != nil {
		return err
	}
	return s.emit("message_stop", streamEvent{Type: "message_stop"})
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
	if err := s.emit("error", errorEvent{
		Type:  "error",
		Error: errorPayload{Type: anthropicErrorType(apiErr), Message: apiErr.Message},
	}); err != nil {
		return err
	}
	return s.emit("message_stop", streamEvent{Type: "message_stop"})
}
