// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package completions

import (
	"encoding/base64"
	"errors"
	"strings"

	chat "github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/execution"
)

type completionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []completionChoice `json:"choices"`
	Usage   *completionUsage   `json:"usage,omitzero"`
}

type completionChoice struct {
	Index        int               `json:"index"`
	Message      completionMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
}

type completionMessage struct {
	Role      string           `json:"role"`
	Content   any              `json:"content"` // string, or null when tools/audio only
	ToolCalls []completionTool `json:"tool_calls,omitzero"`
	Audio     *completionAudio `json:"audio,omitzero"`
}

type completionTool struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function completionToolFn `json:"function"`
}

type completionToolFn struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type completionAudio struct {
	ID         string `json:"id"`
	Data       string `json:"data"`
	ExpiresAt  int64  `json:"expires_at"`
	Transcript string `json:"transcript"`
}

type completionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (Adapter) Response(req chat.Request, result execution.Result, meta responseMeta) (any, error) {
	message, finishReason, err := chatMessage(result.Items, result.Outcome)
	if err != nil {
		return nil, err
	}
	out := completionResponse{
		ID:      meta.Response.ID,
		Object:  "chat.completion",
		Created: meta.Response.Created,
		Model:   meta.Response.Target,
		Choices: []completionChoice{{Index: 0, Message: message, FinishReason: finishReason}},
	}
	if result.Outcome.Usage != nil {
		u := chatUsage(result.Outcome.Usage)
		out.Usage = &u
	}
	return out, nil
}

func chatMessage(items []chat.Item, outcome chat.Outcome) (completionMessage, string, error) {
	message := completionMessage{Role: "assistant"}
	var text strings.Builder
	var calls []completionTool
	var audio *completionAudio
	for _, item := range items {
		switch item.Type {
		case chat.ItemMessage:
			for _, part := range item.Content {
				switch part.Type {
				case chat.PartText:
					text.WriteString(part.Text)
				case chat.PartAudio:
					value, err := chatAudio(part)
					if err != nil {
						return completionMessage{}, "", err
					}
					if audio != nil {
						return completionMessage{}, "", chat.Unsupported("output", "Chat Completions supports one audio output")
					}
					audio = &value
				default:
					return completionMessage{}, "", chat.Unsupported("output", "unsupported Chat Completions output part")
				}
			}
		case chat.ItemFunctionCall:
			calls = append(calls, completionTool{
				ID:       item.CallID,
				Type:     "function",
				Function: completionToolFn{Name: item.Name, Arguments: item.Arguments},
			})
		case chat.ItemReasoning:
			return completionMessage{}, "", chat.Unsupported("output", "Chat Completions has no public reasoning-summary output mapping")
		case chat.ItemMedia:
			if len(item.Content) != 1 || item.Content[0].Type != chat.PartAudio {
				return completionMessage{}, "", chat.Unsupported("output", "generated image output is not part of Chat Completions")
			}
			value, err := chatAudio(item.Content[0])
			if err != nil {
				return completionMessage{}, "", err
			}
			if audio != nil {
				return completionMessage{}, "", chat.Unsupported("output", "Chat Completions supports one audio output")
			}
			audio = &value
		default:
			return completionMessage{}, "", chat.Unsupported("output", "unsupported output item")
		}
	}
	textValue := text.String()
	switch {
	case textValue != "":
		message.Content = textValue
	case len(calls) > 0 || audio != nil:
		message.Content = nil
	default:
		message.Content = ""
	}
	if len(calls) > 0 {
		message.ToolCalls = calls
	}
	if audio != nil {
		message.Audio = audio
	}
	return message, chatFinishReason(outcome, len(calls) > 0), nil
}

func chatFinishReason(outcome chat.Outcome, tools bool) string {
	switch {
	case tools || outcome.StopReason == chat.StopToolCall:
		return "tool_calls"
	case outcome.Status == chat.StatusIncomplete || outcome.StopReason == chat.StopLength:
		return "length"
	default:
		return "stop"
	}
}

func chatUsage(usage *chat.Usage) completionUsage {
	return completionUsage{PromptTokens: usage.Input, CompletionTokens: usage.Output, TotalTokens: usage.Total}
}

func chatAudio(part chat.Part) (completionAudio, error) {
	if part.Media == nil || len(part.Media.Data) == 0 {
		return completionAudio{}, errors.New("Chat Completions audio output requires inline data")
	}
	id := part.Media.Ref
	if id == "" {
		id = newID()
	}
	return completionAudio{
		ID:         id,
		Data:       base64.StdEncoding.EncodeToString(part.Media.Data),
		ExpiresAt:  0,
		Transcript: part.Text,
	}, nil
}
