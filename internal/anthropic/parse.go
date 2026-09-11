// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package anthropic

import (
	"cmp"
	"encoding/base64"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"strings"

	"github.com/kelindar/llmux/chat"
)

// ParseRequest decodes an Anthropic Messages request into a canonical request.
func ParseRequest(object map[string]jsontext.Value) (parsedRequest, error) {
	allowed := map[string]bool{
		"model": true, "max_tokens": true, "messages": true, "stream": true,
		"system": true, "temperature": true, "top_p": true, "stop_sequences": true,
		"tools": true, "tool_choice": true, "metadata": true, "thinking": true,
		"service_tier": true, "container": true, "output_config": true,
		"mcp_servers": true,
	}
	if err := rejectUnknown(object, allowed); err != nil {
		return parsedRequest{}, err
	}
	target, err := requireString(object, "model")
	if err != nil {
		return parsedRequest{}, err
	}
	maxTokens, err := decodeInt(object, "max_tokens")
	if err != nil {
		return parsedRequest{}, err
	}
	if maxTokens == nil || *maxTokens < 1 {
		return parsedRequest{}, chat.Invalid("max_tokens", "max_tokens must be positive")
	}
	rawMessages, ok := object["messages"]
	if !ok {
		return parsedRequest{}, chat.Invalid("messages", "messages is required")
	}
	messageValues, err := rawArray(rawMessages, "messages")
	if err != nil {
		return parsedRequest{}, err
	}
	if len(messageValues) == 0 {
		return parsedRequest{}, chat.Invalid("messages", "messages must not be empty")
	}
	input := make([]chat.Item, 0, len(messageValues))
	for _, value := range messageValues {
		items, err := parseAnthropicMessage(value)
		if err != nil {
			return parsedRequest{}, err
		}
		input = append(input, items...)
	}
	controls := chat.Controls{MaxOutputTokens: maxTokens, Extensions: namespacedExtensions(object, allowed)}
	switch value, err := decodeFloat(object, "temperature"); {
	case err != nil:
		return parsedRequest{}, err
	case value != nil:
		if *value < 0 || *value > 1 {
			return parsedRequest{}, chat.Invalid("temperature", "temperature must be between 0 and 1")
		}
		controls.Temperature = value
	}
	switch value, err := decodeFloat(object, "top_p"); {
	case err != nil:
		return parsedRequest{}, err
	case value != nil:
		if *value < 0 || *value > 1 {
			return parsedRequest{}, chat.Invalid("top_p", "top_p must be between 0 and 1")
		}
		controls.TopP = value
	}
	if _, ok := object["stop_sequences"]; ok {
		controls.Stop, err = decodeStringSlice(object, "stop_sequences")
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if raw, ok := object["tools"]; ok {
		controls.Tools, err = parseAnthropicTools(raw)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if raw, ok := object["tool_choice"]; ok {
		controls.ToolChoice, err = parseAnthropicToolChoice(raw)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	var metadata map[string]string
	if raw, ok := object["metadata"]; ok {
		if err := json.Unmarshal(raw, &metadata); err != nil {
			return parsedRequest{}, fmtError("metadata", "metadata must be an object of strings", err)
		}
	}
	if raw, ok := object["thinking"]; ok {
		thinking, err := rawObject(raw, "thinking")
		if err != nil {
			return parsedRequest{}, err
		}
		if err := rejectUnknownStrict(thinking, map[string]bool{"type": true}); err != nil {
			return parsedRequest{}, err
		}
		typeName, err := requireString(thinking, "type")
		if err != nil {
			return parsedRequest{}, err
		}
		if typeName != "disabled" {
			return parsedRequest{}, chat.Unsupported("thinking", "internal thinking is not exposed by the canonical agent contract")
		}
	}
	for _, key := range []string{"service_tier", "container", "output_config", "mcp_servers"} {
		if _, ok := object[key]; ok {
			return parsedRequest{}, chat.Unsupported(key, key+" is not supported")
		}
	}
	instructions := ""
	if raw, ok := object["system"]; ok {
		instructions, err = parseAnthropicSystem(raw)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	stream := false
	switch value, err := decodeBool(object, "stream"); {
	case err != nil:
		return parsedRequest{}, err
	case value != nil:
		stream = *value
	}
	return parsedRequest{
		Kind:     protocolAnthropic,
		Request:  chat.Request{Target: target, Instructions: instructions, Input: input, Controls: controls, Output: chat.OutputSpec{Modalities: chat.ModalityText}},
		Metadata: metadata,
		Stream:   stream,
	}, nil
}

func parseAnthropicMessage(raw jsontext.Value) ([]chat.Item, error) {
	object, err := rawObject(raw, "messages")
	if err != nil {
		return nil, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"role": true, "content": true}); err != nil {
		return nil, err
	}
	roleValue, err := requireString(object, "role")
	if err != nil {
		return nil, err
	}
	switch roleValue {
	case "user", "assistant":
	default:
		return nil, chat.Invalid("messages.role", "Anthropic messages only support user and assistant roles")
	}
	rawContent, ok := object["content"]
	if !ok {
		return nil, chat.Invalid("messages.content", "message content is required")
	}
	values, err := anthropicContent(rawContent)
	if err != nil {
		return nil, err
	}
	items := make([]chat.Item, 0, len(values)+1)
	var text []chat.Part
	for _, value := range values {
		if value.item.Type == chat.ItemMessage {
			text = append(text, value.item.Content...)
			continue
		}
		if len(text) > 0 {
			items = append(items, chat.MessageItem(chat.Role(roleValue), text...))
			text = nil
		}
		items = append(items, value.item)
	}
	if len(text) > 0 || len(items) == 0 {
		items = append(items, chat.MessageItem(chat.Role(roleValue), text...))
	}
	return items, nil
}

type anthropicContentValue struct{ item chat.Item }

func anthropicContent(raw jsontext.Value) ([]anthropicContentValue, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []anthropicContentValue{{item: chat.MessageItem(chat.RoleUser, chat.TextPart(text))}}, nil
	}
	values, err := rawArray(raw, "messages.content")
	if err != nil {
		return nil, err
	}
	out := make([]anthropicContentValue, 0, len(values))
	for _, value := range values {
		object, err := rawObject(value, "messages.content")
		if err != nil {
			return nil, err
		}
		typeName, err := requireString(object, "type")
		if err != nil {
			return nil, err
		}
		switch typeName {
		case "text":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "text": true}); err != nil {
				return nil, err
			}
			text, err := requireString(object, "text")
			if err != nil {
				return nil, err
			}
			out = append(out, anthropicContentValue{item: chat.MessageItem(chat.RoleUser, chat.TextPart(text))})
		case "image":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "source": true}); err != nil {
				return nil, err
			}
			media, err := parseAnthropicImage(object)
			if err != nil {
				return nil, err
			}
			out = append(out, anthropicContentValue{item: chat.MessageItem(chat.RoleUser, chat.ImagePart(media))})
		case "document":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "source": true}); err != nil {
				return nil, err
			}
			media, err := parseAnthropicDocument(object)
			if err != nil {
				return nil, err
			}
			out = append(out, anthropicContentValue{item: chat.MessageItem(chat.RoleUser, chat.FilePart(media))})
		case "tool_use":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "id": true, "name": true, "input": true}); err != nil {
				return nil, err
			}
			callID, err := requireString(object, "id")
			if err != nil {
				return nil, err
			}
			name, err := requireString(object, "name")
			if err != nil {
				return nil, err
			}
			input, ok := object["input"]
			if !ok || !input.IsValid() {
				return nil, chat.Invalid("messages.content.input", "tool input must be valid JSON")
			}
			call := chat.FunctionCallItem(callID, name, string(input))
			id, err := optionalString(object, "id")
			if err != nil {
				return nil, err
			}
			if id != "" {
				call.ID = id
			}
			out = append(out, anthropicContentValue{item: call})
		case "tool_result":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "tool_use_id": true, "content": true}); err != nil {
				return nil, err
			}
			callID, err := requireString(object, "tool_use_id")
			if err != nil {
				return nil, err
			}
			parts := []chat.Part{}
			if rawContent, ok := object["content"]; ok {
				content, err := anthropicContentParts(rawContent)
				if err != nil {
					return nil, err
				}
				parts = content
			}
			if len(parts) == 0 {
				parts = []chat.Part{chat.TextPart("")}
			}
			out = append(out, anthropicContentValue{item: chat.FunctionCallOutputItem(callID, parts...)})
		case "thinking":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "thinking": true, "signature": true}); err != nil {
				return nil, err
			}
			item := chat.Item{Type: chat.ItemReasoning, Data: append(jsontext.Value(nil), value...), Status: chat.StatusCompleted}
			out = append(out, anthropicContentValue{item: item})
		case "redacted_thinking":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "data": true}); err != nil {
				return nil, err
			}
			item := chat.Item{Type: chat.ItemReasoning, Data: append(jsontext.Value(nil), value...), Status: chat.StatusCompleted}
			out = append(out, anthropicContentValue{item: item})
		default:
			return nil, chat.Unsupported("messages.content.type", "unsupported Anthropic content type "+typeName)
		}
	}
	return out, nil
}

func anthropicContentParts(raw jsontext.Value) ([]chat.Part, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []chat.Part{chat.TextPart(text)}, nil
	}
	values, err := rawArray(raw, "tool_result.content")
	if err != nil {
		return nil, err
	}
	parts := make([]chat.Part, 0, len(values))
	for _, value := range values {
		object, err := rawObject(value, "tool_result.content")
		if err != nil {
			return nil, err
		}
		typeName, err := requireString(object, "type")
		if err != nil {
			return nil, err
		}
		switch typeName {
		case "text":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "text": true}); err != nil {
				return nil, err
			}
			text, err := requireString(object, "text")
			if err != nil {
				return nil, err
			}
			parts = append(parts, chat.TextPart(text))
		case "image":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "source": true}); err != nil {
				return nil, err
			}
			media, err := parseAnthropicImage(object)
			if err != nil {
				return nil, err
			}
			parts = append(parts, chat.ImagePart(media))
		default:
			return nil, chat.Unsupported("tool_result.content", "unsupported tool result type "+typeName)
		}
	}
	return parts, nil
}

func parseAnthropicImage(object map[string]jsontext.Value) (chat.Media, error) {
	source, err := rawObject(object["source"], "messages.content.source")
	if err != nil {
		return chat.Media{}, err
	}
	typeName, err := requireString(source, "type")
	if err != nil {
		return chat.Media{}, err
	}
	switch typeName {
	case "base64":
		if err := rejectUnknownStrict(source, map[string]bool{"type": true, "media_type": true, "data": true}); err != nil {
			return chat.Media{}, err
		}
		mime, err := requireString(source, "media_type")
		if err != nil {
			return chat.Media{}, err
		}
		encoded, err := requireString(source, "data")
		if err != nil {
			return chat.Media{}, err
		}
		data, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return chat.Media{}, fmtError("messages.content.source.data", "must be valid base64", err)
		}
		if !strings.HasPrefix(mime, "image/") {
			return chat.Media{}, chat.Invalid("messages.content.source.media_type", "Anthropic image media_type must be an image MIME type")
		}
		return chat.InlineMedia(mime, data), nil
	case "url":
		if err := rejectUnknownStrict(source, map[string]bool{"type": true, "url": true}); err != nil {
			return chat.Media{}, err
		}
		value, err := requireString(source, "url")
		if err != nil {
			return chat.Media{}, err
		}
		return chat.RemoteMedia("image/*", value), nil
	default:
		return chat.Media{}, chat.Unsupported("messages.content.source.type", "unsupported Anthropic image source "+typeName)
	}
}

func parseAnthropicDocument(object map[string]jsontext.Value) (chat.Media, error) {
	source, err := rawObject(object["source"], "messages.content.source")
	if err != nil {
		return chat.Media{}, err
	}
	typeName, err := requireString(source, "type")
	if err != nil {
		return chat.Media{}, err
	}
	switch typeName {
	case "base64":
		if err := rejectUnknownStrict(source, map[string]bool{"type": true, "media_type": true, "data": true}); err != nil {
			return chat.Media{}, err
		}
		mime, err := optionalString(source, "media_type")
		if err != nil {
			return chat.Media{}, err
		}
		mime = cmp.Or(mime, "application/octet-stream")
		encoded, err := requireString(source, "data")
		if err != nil {
			return chat.Media{}, err
		}
		data, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return chat.Media{}, fmtError("messages.content.source.data", "must be valid base64", err)
		}
		return chat.InlineMedia(mime, data), nil
	case "url":
		if err := rejectUnknownStrict(source, map[string]bool{"type": true, "url": true}); err != nil {
			return chat.Media{}, err
		}
		value, err := requireString(source, "url")
		if err != nil {
			return chat.Media{}, err
		}
		return chat.RemoteMedia("application/octet-stream", value), nil
	default:
		return chat.Media{}, chat.Unsupported("messages.content.source.type", "unsupported Anthropic document source "+typeName)
	}
}

func parseAnthropicSystem(raw jsontext.Value) (string, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}
	values, err := rawArray(raw, "system")
	if err != nil {
		return "", err
	}
	var builder strings.Builder
	for _, value := range values {
		object, err := rawObject(value, "system")
		if err != nil {
			return "", err
		}
		typeName, err := requireString(object, "type")
		if err != nil {
			return "", err
		}
		if typeName != "text" {
			return "", chat.Unsupported("system", "only text system blocks are supported")
		}
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "text": true}); err != nil {
			return "", err
		}
		value, err := requireString(object, "text")
		if err != nil {
			return "", err
		}
		builder.WriteString(value)
	}
	return builder.String(), nil
}

func parseAnthropicTools(raw jsontext.Value) ([]chat.FunctionTool, error) {
	values, err := rawArray(raw, "tools")
	if err != nil {
		return nil, err
	}
	tools := make([]chat.FunctionTool, 0, len(values))
	for _, value := range values {
		object, err := rawObject(value, "tools")
		if err != nil {
			return nil, err
		}
		if err := rejectUnknownStrict(object, map[string]bool{"name": true, "description": true, "input_schema": true}); err != nil {
			return nil, err
		}
		name, err := requireString(object, "name")
		if err != nil {
			return nil, err
		}
		inputSchema, ok := object["input_schema"]
		if !ok || !inputSchema.IsValid() {
			return nil, chat.Invalid("tools.input_schema", "input_schema must be valid JSON")
		}
		description, err := optionalString(object, "description")
		if err != nil {
			return nil, err
		}
		tools = append(tools, chat.FunctionTool{Name: name, Description: description, Parameters: append(jsontext.Value(nil), inputSchema...)})
	}
	return tools, nil
}

func parseAnthropicToolChoice(raw jsontext.Value) (*chat.ToolChoice, error) {
	object, err := rawObject(raw, "tool_choice")
	if err != nil {
		return nil, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"type": true, "name": true}); err != nil {
		return nil, err
	}
	typeName, err := requireString(object, "type")
	if err != nil {
		return nil, err
	}
	switch typeName {
	case "auto", "none", "any":
		if _, ok := object["name"]; ok {
			return nil, chat.Invalid("tool_choice.name", "tool_choice name is only valid for type tool")
		}
		return &chat.ToolChoice{Mode: typeName}, nil
	case "tool":
		name, err := requireString(object, "name")
		if err != nil {
			return nil, err
		}
		return &chat.ToolChoice{Mode: "function", Name: name}, nil
	default:
		return nil, chat.Unsupported("tool_choice.type", "unsupported Anthropic tool choice "+typeName)
	}
}
