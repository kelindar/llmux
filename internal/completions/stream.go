// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

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

type chunkChoice struct {
	Index        int        `json:"index"`
	Delta        chunkDelta `json:"delta"`
	FinishReason any        `json:"finish_reason"` // string or null
}

type chunkDelta struct {
	Role      string          `json:"role,omitzero"`
	Content   *string         `json:"content,omitzero"`
	ToolCalls []chunkToolCall `json:"tool_calls,omitzero"`
	Audio     *chunkAudio     `json:"audio,omitzero"`
}

type chunkToolCall struct {
	Index    int         `json:"index"`
	ID       string      `json:"id,omitzero"`
	Type     string      `json:"type,omitzero"`
	Function chunkToolFn `json:"function"`
}

type chunkToolFn struct {
	Name      string `json:"name,omitzero"`
	Arguments string `json:"arguments"`
}

type chunkAudio struct {
	Data string `json:"data"`
}

// chunk is the default stream frame without a usage field.
type chunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []chunkChoice `json:"choices"`
}

// chunkWithUsage always emits usage (null until the final usage frame).
type chunkWithUsage struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []chunkChoice    `json:"choices"`
	Usage   *completionUsage `json:"usage"`
}

// Started reports whether the Chat Completions SSE stream has begun.
func (s *chatStream) Started() bool { return s.writer.started }

func (s *chatStream) writeChunk(delta chunkDelta, finish any) error {
	choice := chunkChoice{Index: 0, Delta: delta, FinishReason: finish}
	switch {
	case s.includeUsage:
		return s.writer.write("", chunkWithUsage{
			ID:      s.meta.Response.ID,
			Object:  "chat.completion.chunk",
			Created: s.meta.Response.Created,
			Model:   s.meta.Response.Target,
			Choices: []chunkChoice{choice},
		})
	default:
		return s.writer.write("", chunk{
			ID:      s.meta.Response.ID,
			Object:  "chat.completion.chunk",
			Created: s.meta.Response.Created,
			Model:   s.meta.Response.Target,
			Choices: []chunkChoice{choice},
		})
	}
}

// Event encodes one canonical event as a Chat Completions chunk.
func (s *chatStream) Event(event chat.Event) error {
	if s.toolIndexes == nil {
		s.toolIndexes = make(map[string]int)
	}
	if event.Type == chat.EventTextDone || event.Type == chat.EventToolCallDone {
		return nil
	}
	var delta chunkDelta
	switch event.Type {
	case chat.EventTextDelta:
		if !s.roleSent {
			delta.Role = "assistant"
			s.roleSent = true
		}
		content := event.Delta
		delta.Content = &content
		return s.writeChunk(delta, nil)
	case chat.EventToolCallStart:
		s.hadTool = true
		if !s.roleSent {
			delta.Role = "assistant"
			s.roleSent = true
		}
		index := s.nextTool
		switch existing, ok := s.toolIndexes[event.CallID]; {
		case ok:
			index = existing
		default:
			s.toolIndexes[event.CallID] = index
			s.nextTool++
		}
		delta.ToolCalls = []chunkToolCall{{
			Index:    index,
			ID:       event.CallID,
			Type:     "function",
			Function: chunkToolFn{Name: event.Name, Arguments: ""},
		}}
		return s.writeChunk(delta, nil)
	case chat.EventToolCallDelta:
		s.hadTool = true
		index, ok := s.toolIndexes[event.CallID]
		if !ok {
			return fmt.Errorf("unknown tool call %q", event.CallID)
		}
		delta.ToolCalls = []chunkToolCall{{
			Index:    index,
			Function: chunkToolFn{Arguments: event.Delta},
		}}
		return s.writeChunk(delta, nil)
	case chat.EventItem:
		switch event.Item.Type {
		case chat.ItemMessage:
			for _, part := range event.Item.Content {
				switch part.Type {
				case chat.PartText:
					delta = chunkDelta{}
					if !s.roleSent {
						delta.Role = "assistant"
						s.roleSent = true
					}
					content := part.Text
					delta.Content = &content
					if err := s.writeChunk(delta, nil); err != nil {
						return err
					}
				case chat.PartAudio:
					if part.Media == nil || len(part.Media.Data) == 0 {
						return errors.New("streamed audio output requires inline data")
					}
					delta = chunkDelta{}
					if !s.roleSent {
						delta.Role = "assistant"
						s.roleSent = true
					}
					delta.Audio = &chunkAudio{Data: base64.StdEncoding.EncodeToString(part.Media.Data)}
					if err := s.writeChunk(delta, nil); err != nil {
						return err
					}
				default:
					return chat.Unsupported("output", "Chat Completions stream supports text and audio only")
				}
			}
		case chat.ItemFunctionCall:
			s.hadTool = true
			if !s.roleSent {
				delta.Role = "assistant"
				s.roleSent = true
			}
			index := s.nextTool
			callID := event.Item.CallID
			switch existing, ok := s.toolIndexes[callID]; {
			case ok:
				index = existing
			default:
				s.toolIndexes[callID] = index
				s.nextTool++
			}
			delta.ToolCalls = []chunkToolCall{{
				Index:    index,
				ID:       callID,
				Type:     "function",
				Function: chunkToolFn{Name: event.Item.Name, Arguments: event.Item.Arguments},
			}}
			return s.writeChunk(delta, nil)
		case chat.ItemMedia:
			if len(event.Item.Content) != 1 || event.Item.Content[0].Type != chat.PartAudio {
				return chat.Unsupported("output", "Chat Completions stream only supports audio media")
			}
			part := event.Item.Content[0]
			if part.Media == nil || len(part.Media.Data) == 0 {
				return errors.New("streamed audio output requires inline data")
			}
			if !s.roleSent {
				delta.Role = "assistant"
				s.roleSent = true
			}
			delta.Audio = &chunkAudio{Data: base64.StdEncoding.EncodeToString(part.Media.Data)}
			return s.writeChunk(delta, nil)
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
	if err := s.writeChunk(chunkDelta{}, finish); err != nil {
		return err
	}
	if outcome.Usage != nil && s.includeUsage {
		u := chatUsage(outcome.Usage)
		if err := s.writer.write("", chunkWithUsage{
			ID:      s.meta.Response.ID,
			Object:  "chat.completion.chunk",
			Created: s.meta.Response.Created,
			Model:   s.meta.Response.Target,
			Choices: []chunkChoice{},
			Usage:   &u,
		}); err != nil {
			return err
		}
	}
	return s.writer.done()
}

// Fail writes a Chat Completions error chunk or HTTP error envelope.
func (s *chatStream) Fail(err error) error {
	apiErr := internalprotocol.AsError(err)
	if !s.writer.started {
		return writeProtocolErrorAndReturn(s.writer.w, protocolChat, apiErr)
	}
	type errBody struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
		Param   string `json:"param,omitzero"`
	}
	if err := s.writer.write("", struct {
		Error errBody `json:"error"`
	}{Error: errBody{Message: apiErr.Message, Type: apiErr.Type, Code: apiErr.Code, Param: apiErr.Param}}); err != nil {
		return err
	}
	return s.writer.done()
}
