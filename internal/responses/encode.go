package responses

import (
	"encoding/base64"
	"encoding/json/jsontext"
	"errors"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/execution"
)

// Response builds a non-streaming Responses API response body.
func (Adapter) Response(req chat.Request, result execution.Result, meta responseMeta) (any, error) {
	output := make([]any, 0, len(result.Items))
	for _, item := range result.Items {
		value, err := responseItem(item)
		if err != nil {
			return nil, err
		}
		output = append(output, value)
	}
	state := meta.State
	if len(state.Output) == 0 {
		state.Output = result.Items
	}
	if state.Status == "" {
		state.Status = result.Outcome.Status
	}
	if state.Usage == nil {
		state.Usage = result.Outcome.Usage
	}
	return responseObject(req, meta, state, output), nil
}

// Render builds the same Responses envelope used for creation, replay, and
// application-owned GET retrieval.
func Render(req chat.Request, state chat.ResponseState, id string, created int64) (any, error) {
	output := make([]any, 0, len(state.Output))
	for _, item := range state.Output {
		value, err := responseItem(item)
		if err != nil {
			return nil, err
		}
		output = append(output, value)
	}
	return responseObject(req, responseMeta{
		ID:      id,
		Created: created,
		Model:   req.Target,
		State:   state,
	}, state, output), nil
}

func responseStatus(state chat.ResponseState, outcome chat.Outcome) string {
	switch {
	case state.Status != "":
		return string(state.Status)
	case outcome.Status != "":
		return string(outcome.Status)
	default:
		return string(chat.StatusCompleted)
	}
}

func responseObject(req chat.Request, meta responseMeta, state chat.ResponseState, output []any) map[string]any {
	status := responseStatus(state, chat.Outcome{})
	value := map[string]any{
		"id":                   meta.ID,
		"object":               "response",
		"created_at":           meta.Created,
		"status":               status,
		"error":                nil,
		"incomplete_details":   nil,
		"instructions":         nil,
		"model":                meta.Model,
		"output":               output,
		"parallel_tool_calls":  true,
		"previous_response_id": nil,
		"reasoning":            map[string]any{"effort": nil, "summary": nil},
		"store":                state.Store,
		"temperature":          nil,
		"text":                 map[string]any{"format": map[string]any{"type": "text"}},
		"tool_choice":          "auto",
		"tools":                []any{},
		"top_p":                nil,
		"metadata":             map[string]string{},
	}
	if state.CompletedAt > 0 {
		value["completed_at"] = state.CompletedAt
	}
	if state.Error != nil {
		value["error"] = map[string]any{"type": state.Error.Code, "code": state.Error.Code, "message": state.Error.Message}
	}
	if state.Incomplete != "" {
		value["incomplete_details"] = map[string]any{"reason": state.Incomplete}
	}
	if len(state.Metadata) > 0 {
		value["metadata"] = state.Metadata
	}
	if req.Instructions != "" {
		value["instructions"] = req.Instructions
	}
	if req.Controls.MaxOutputTokens != nil {
		value["max_output_tokens"] = *req.Controls.MaxOutputTokens
	} else {
		value["max_output_tokens"] = nil
	}
	if req.Controls.Temperature != nil {
		value["temperature"] = *req.Controls.Temperature
	}
	if req.Controls.TopP != nil {
		value["top_p"] = *req.Controls.TopP
	}
	if req.Controls.ParallelToolCall != nil {
		value["parallel_tool_calls"] = *req.Controls.ParallelToolCall
	}
	if req.Previous != nil {
		value["previous_response_id"] = *req.Previous
	}
	switch req.Output.Format.Kind {
	case chat.FormatJSONSchema:
		value["text"] = map[string]any{"format": map[string]any{"type": "json_schema", "name": req.Output.Format.Name, "description": req.Output.Format.Description, "schema": jsontext.Value(req.Output.Format.Schema), "strict": req.Output.Format.Strict}}
	case chat.FormatJSONObject:
		value["text"] = map[string]any{"format": map[string]any{"type": "json_object"}}
	}
	if len(req.Controls.Tools) > 0 {
		tools := make([]any, 0, len(req.Controls.Tools))
		for _, tool := range req.Controls.Tools {
			parameters := jsontext.Value(tool.Parameters)
			if len(parameters) == 0 {
				parameters = jsontext.Value(`{"type":"object"}`)
			}
			tools = append(tools, map[string]any{"type": "function", "name": tool.Name, "description": tool.Description, "parameters": parameters, "strict": boolPointerValue(tool.Strict)})
		}
		value["tools"] = tools
	}
	if req.Controls.ImageGeneration {
		tools := value["tools"].([]any)
		value["tools"] = append(tools, map[string]any{"type": "image_generation"})
	}
	if req.Controls.ToolChoice != nil {
		if req.Controls.ToolChoice.Mode == "function" {
			value["tool_choice"] = map[string]any{"type": "function", "name": req.Controls.ToolChoice.Name}
		} else {
			value["tool_choice"] = req.Controls.ToolChoice.Mode
		}
	}
	if state.Usage != nil {
		value["usage"] = responseUsage(state.Usage)
	}
	return value
}

func boolPointerValue(value *bool) bool { return value != nil && *value }

func responseUsage(usage *chat.Usage) map[string]any {
	return map[string]any{
		"input_tokens":          usage.Input,
		"output_tokens":         usage.Output,
		"total_tokens":          usage.Total,
		"input_tokens_details":  map[string]any{"cached_tokens": usage.Cached},
		"output_tokens_details": map[string]any{"reasoning_tokens": usage.Reasoning},
	}
}

func responseItem(item chat.Item) (map[string]any, error) {
	status := item.Status
	if status == "" {
		status = chat.StatusCompleted
	}
	switch item.Type {
	case chat.ItemMessage:
		for _, part := range item.Content {
			if part.Type != chat.PartText {
				return nil, chat.Unsupported("output", "Responses message output supports text parts only")
			}
		}
		return map[string]any{"id": item.ID, "type": "message", "status": status, "role": item.Role, "content": outputTextParts(item.Content)}, nil
	case chat.ItemFunctionCall:
		return map[string]any{"id": item.ID, "type": "function_call", "status": status, "call_id": item.CallID, "name": item.Name, "arguments": item.Arguments}, nil
	case chat.ItemFunctionCallOutput:
		output, err := responseFunctionOutput(item.Output)
		if err != nil {
			return nil, err
		}
		return map[string]any{"id": item.ID, "type": "function_call_output", "status": status, "call_id": item.CallID, "output": output}, nil
	case chat.ItemReasoning:
		if len(item.Data) > 0 {
			return nil, chat.Unsupported("output", "Responses reasoning output cannot carry opaque data")
		}
		summary := make([]any, 0, len(item.Summary))
		for _, part := range item.Summary {
			summary = append(summary, map[string]any{"type": "summary_text", "text": part.Text})
		}
		value := map[string]any{"id": item.ID, "type": "reasoning", "status": status, "summary": summary}
		if len(item.EncryptedContent) > 0 {
			value["encrypted_content"] = jsontext.Value(item.EncryptedContent)
		}
		return value, nil
	case chat.ItemMedia:
		if len(item.Content) != 1 || item.Content[0].Type != chat.PartImage || item.Content[0].Media == nil {
			return nil, chat.Unsupported("output", "Responses image output requires one image media part")
		}
		if len(item.Content[0].Media.Data) == 0 {
			return nil, chat.Unsupported("output", "Responses image output requires inline image data")
		}
		return map[string]any{"id": item.ID, "type": "image_generation_call", "status": status, "result": base64.StdEncoding.EncodeToString(item.Content[0].Media.Data)}, nil
	default:
		return nil, chat.Unsupported("output", "unsupported Responses output item")
	}
}

func responseFunctionOutput(parts []chat.Part) (any, error) {
	if len(parts) == 1 && parts[0].Type == chat.PartText {
		return parts[0].Text, nil
	}
	values := make([]any, 0, len(parts))
	for _, part := range parts {
		switch part.Type {
		case chat.PartText:
			values = append(values, map[string]any{"type": "input_text", "text": part.Text})
		case chat.PartImage, chat.PartFile:
			if part.Media == nil {
				return nil, errors.New("function output media is missing")
			}
			if len(part.Media.Data) > 0 {
				data, err := mediaDataURL(*part.Media)
				if err != nil {
					return nil, err
				}
				if part.Type == chat.PartImage {
					values = append(values, map[string]any{"type": "input_image", "image_url": data, "detail": "auto"})
				} else {
					values = append(values, map[string]any{"type": "input_file", "file_data": base64.StdEncoding.EncodeToString(part.Media.Data), "filename": part.Media.Filename})
				}
			} else {
				return nil, chat.Unsupported("output", "function output media must be inline")
			}
		default:
			return nil, chat.Unsupported("output", "unsupported function output part")
		}
	}
	return values, nil
}
