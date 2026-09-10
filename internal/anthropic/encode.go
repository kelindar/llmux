package anthropic

import (
	json "encoding/json/v2"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/execution"
)

// Response builds a non-streaming Anthropic Messages response body.
func (Adapter) Response(req chat.Request, result execution.Result, meta responseMeta) (any, error) {
	content, tools, err := anthropicOutput(result.Items)
	if err != nil {
		return nil, err
	}
	value := map[string]any{
		"id":            meta.Response.ID,
		"type":          "message",
		"role":          "assistant",
		"model":         meta.Response.Target,
		"content":       content,
		"stop_reason":   anthropicStopReason(result.Outcome, tools),
		"stop_sequence": nil,
		"usage":         nil,
	}
	if result.Outcome.Usage != nil {
		value["usage"] = map[string]any{"input_tokens": result.Outcome.Usage.Input, "output_tokens": result.Outcome.Usage.Output}
	}
	return value, nil
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
				content = append(content, map[string]any{"type": "text", "text": part.Text})
			}
		case chat.ItemFunctionCall:
			var input any
			if err := json.Unmarshal([]byte(item.Arguments), &input); err != nil {
				return nil, false, err
			}
			content = append(content, map[string]any{"type": "tool_use", "id": item.CallID, "name": item.Name, "input": input})
			hasTools = true
		default:
			return nil, false, chat.Unsupported("output", "unsupported Anthropic output item")
		}
	}
	return content, hasTools, nil
}

func anthropicStopReason(outcome chat.Outcome, tools bool) string {
	if tools || outcome.StopReason == chat.StopToolCall {
		return "tool_use"
	}
	if outcome.Status == chat.StatusIncomplete || outcome.StopReason == chat.StopLength {
		return "max_tokens"
	}
	return "end_turn"
}
