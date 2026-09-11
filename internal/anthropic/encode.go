// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package anthropic

import (
	json "encoding/json/v2"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/execution"
)

type messageResponse struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Role         string          `json:"role"`
	Model        string          `json:"model"`
	Content      []any           `json:"content"`
	StopReason   string          `json:"stop_reason"`
	StopSequence any             `json:"stop_sequence"`
	Usage        *anthropicUsage `json:"usage"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type textBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type toolUseBlock struct {
	Type  string `json:"type"`
	ID    string `json:"id"`
	Name  string `json:"name"`
	Input any    `json:"input"`
}

// Response builds a non-streaming Anthropic Messages response body.
func (Adapter) Response(req chat.Request, result execution.Result, meta responseMeta) (any, error) {
	content, tools, err := anthropicOutput(result.Items)
	if err != nil {
		return nil, err
	}
	out := messageResponse{
		ID:           meta.Response.ID,
		Type:         "message",
		Role:         "assistant",
		Model:        meta.Response.Target,
		Content:      content,
		StopReason:   anthropicStopReason(result.Outcome, tools),
		StopSequence: nil,
		Usage:        nil,
	}
	if result.Outcome.Usage != nil {
		out.Usage = &anthropicUsage{InputTokens: result.Outcome.Usage.Input, OutputTokens: result.Outcome.Usage.Output}
	}
	return out, nil
}

func anthropicOutput(items []chat.Item) ([]any, bool, error) {
	content := make([]any, 0, len(items))
	hasTools := false
	for _, item := range items {
		switch item.Type {
		case chat.ItemMessage:
			for _, part := range item.Content {
				if part.Type != chat.PartText {
					return nil, false, chat.Unsupported("output", "Anthropic output supports text only")
				}
				content = append(content, textBlock{Type: "text", Text: part.Text})
			}
		case chat.ItemFunctionCall:
			var input any
			if err := json.Unmarshal([]byte(item.Arguments), &input); err != nil {
				return nil, false, err
			}
			content = append(content, toolUseBlock{Type: "tool_use", ID: item.CallID, Name: item.Name, Input: input})
			hasTools = true
		default:
			return nil, false, chat.Unsupported("output", "unsupported Anthropic output item")
		}
	}
	return content, hasTools, nil
}

func anthropicStopReason(outcome chat.Outcome, tools bool) string {
	switch {
	case tools || outcome.StopReason == chat.StopToolCall:
		return "tool_use"
	case outcome.Status == chat.StatusIncomplete || outcome.StopReason == chat.StopLength:
		return "max_tokens"
	default:
		return "end_turn"
	}
}
