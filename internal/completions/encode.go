package completions

import (
	"encoding/base64"
	"errors"
	"strings"

	chat "github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/execution"
)

func (Adapter) Response(req chat.Request, result execution.Result, meta responseMeta) (any, error) {
	message, finishReason, err := chatMessage(result.Items, result.Outcome)
	if err != nil {
		return nil, err
	}
	response := map[string]any{
		"id":      meta.Response.ID,
		"object":  "chat.completion",
		"created": meta.Response.Created,
		"model":   meta.Response.Target,
		"choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": finishReason}},
	}
	if result.Outcome.Usage != nil {
		response["usage"] = chatUsage(result.Outcome.Usage)
	}
	return response, nil
}
func chatMessage(items []chat.Item, outcome chat.Outcome) (map[string]any, string, error) {
	message := map[string]any{"role": "assistant"}
	var text strings.Builder
	var calls []map[string]any
	var audio *map[string]any
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
						return nil, "", err
					}
					if audio != nil {
						return nil, "", chat.Unsupported("output", "Chat Completions supports one audio output")
					}
					audio = &value
				default:
					return nil, "", chat.Unsupported("output", "unsupported Chat Completions output part")
				}
			}
		case chat.ItemFunctionCall:
			calls = append(calls, map[string]any{"id": item.CallID, "type": "function", "function": map[string]any{"name": item.Name, "arguments": item.Arguments}})
		case chat.ItemReasoning:
			return nil, "", chat.Unsupported("output", "Chat Completions has no public reasoning-summary output mapping")
		case chat.ItemMedia:
			if len(item.Content) != 1 || item.Content[0].Type != chat.PartAudio {
				return nil, "", chat.Unsupported("output", "generated image output is not part of Chat Completions")
			}
			value, err := chatAudio(item.Content[0])
			if err != nil {
				return nil, "", err
			}
			if audio != nil {
				return nil, "", chat.Unsupported("output", "Chat Completions supports one audio output")
			}
			audio = &value
		default:
			return nil, "", chat.Unsupported("output", "unsupported output item")
		}
	}
	textValue := text.String()
	switch {
	case textValue != "":
		message["content"] = textValue
	case len(calls) > 0 || audio != nil:
		message["content"] = nil
	default:
		message["content"] = ""
	}
	if len(calls) > 0 {
		message["tool_calls"] = calls
	}
	if audio != nil {
		message["audio"] = *audio
	}
	finishReason := chatFinishReason(outcome, len(calls) > 0)
	return message, finishReason, nil
}

func chatFinishReason(outcome chat.Outcome, tools bool) string {
	if tools || outcome.StopReason == chat.StopToolCall {
		return "tool_calls"
	}
	if outcome.Status == chat.StatusIncomplete || outcome.StopReason == chat.StopLength {
		return "length"
	}
	return "stop"
}

func chatUsage(usage *chat.Usage) map[string]any {
	return map[string]any{"prompt_tokens": usage.Input, "completion_tokens": usage.Output, "total_tokens": usage.Total}
}

func chatAudio(part chat.Part) (map[string]any, error) {
	if part.Media == nil || len(part.Media.Data) == 0 {
		return nil, errors.New("Chat Completions audio output requires inline data")
	}
	id := part.Media.Ref
	if id == "" {
		id = newID()
	}
	return map[string]any{
		"id":         id,
		"data":       base64.StdEncoding.EncodeToString(part.Media.Data),
		"expires_at": 0,
		"transcript": part.Text,
	}, nil
}
