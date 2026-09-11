// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package responses

import (
	"encoding/json/jsontext"
	json "encoding/json/v2"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/wire"
)

func optionalString(object map[string]jsontext.Value, key string) (string, error) {
	value, _, err := decodeString(object, key)
	return value, err
}

func parseToolChoice(raw jsontext.Value) (*chat.ToolChoice, error) {
	var value string
	if err := json.Unmarshal(raw, &value); err == nil {
		switch value {
		case "auto", "none", "required":
			return &chat.ToolChoice{Mode: value}, nil
		default:
			return nil, chat.Invalid("tool_choice", "unsupported tool_choice "+value)
		}
	}
	object, err := rawObject(raw, "tool_choice")
	if err != nil {
		return nil, err
	}
	if functionRaw, ok := object["function"]; ok {
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "function": true}); err != nil {
			return nil, err
		}
		switch typeName, ok, err := decodeString(object, "type"); {
		case err != nil:
			return nil, err
		case ok && typeName != "function":
			return nil, chat.Unsupported("tool_choice.type", "only function tool choices are supported")
		}
		function, err := rawObject(functionRaw, "tool_choice.function")
		if err != nil {
			return nil, err
		}
		if err := rejectUnknownStrict(function, map[string]bool{"name": true}); err != nil {
			return nil, err
		}
		name, err := requireString(function, "name")
		if err != nil {
			return nil, err
		}
		return &chat.ToolChoice{Mode: "function", Name: name}, nil
	}
	if err := rejectUnknownStrict(object, map[string]bool{"type": true, "name": true}); err != nil {
		return nil, err
	}
	mode, err := requireString(object, "type")
	if err != nil {
		return nil, err
	}
	if mode != "function" {
		return nil, chat.Invalid("tool_choice", "unsupported tool_choice type "+mode)
	}
	name, err := requireString(object, "name")
	if err != nil {
		return nil, err
	}
	return &chat.ToolChoice{Mode: "function", Name: name}, nil
}

// ParseRequest decodes a Responses API request into a canonical request.
func ParseRequest(object map[string]jsontext.Value) (parsedRequest, error) {
	allowed := map[string]bool{
		"model": true, "input": true, "instructions": true, "stream": true,
		"stream_options": true, "tools": true, "tool_choice": true,
		"parallel_tool_calls": true, "temperature": true, "top_p": true,
		"max_output_tokens": true, "text": true, "reasoning": true, "store": true,
		"previous_response_id": true, "metadata": true, "background": true,
		"conversation": true, "include": true, "service_tier": true,
		"truncation": true, "user": true, "prompt": true, "max_tool_calls": true,
	}
	if err := rejectUnknown(object, allowed); err != nil {
		return parsedRequest{}, err
	}
	target, err := requireString(object, "model")
	if err != nil {
		return parsedRequest{}, err
	}
	rawInput, ok := object["input"]
	if !ok {
		return parsedRequest{}, chat.Invalid("input", "input is required")
	}
	input, err := parseResponsesInput(rawInput)
	if err != nil {
		return parsedRequest{}, err
	}
	controls := chat.Controls{Extensions: namespacedExtensions(object, allowed)}
	switch value, err := decodeInt(object, "max_output_tokens"); {
	case err != nil:
		return parsedRequest{}, err
	case value != nil:
		if *value < 1 {
			return parsedRequest{}, chat.Invalid("max_output_tokens", "max_output_tokens must be positive")
		}
		controls.MaxOutputTokens = value
	}
	switch value, err := decodeFloat(object, "temperature"); {
	case err != nil:
		return parsedRequest{}, err
	case value != nil:
		if *value < 0 || *value > 2 {
			return parsedRequest{}, chat.Invalid("temperature", "temperature must be between 0 and 2")
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
	if raw, ok := object["tools"]; ok {
		controls.Tools, controls.ImageGeneration, err = parseResponsesTools(raw)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if raw, ok := object["tool_choice"]; ok {
		controls.ToolChoice, err = parseToolChoice(raw)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if _, ok := object["parallel_tool_calls"]; ok {
		controls.ParallelToolCall, err = decodeBool(object, "parallel_tool_calls")
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if raw, ok := object["stream_options"]; ok {
		streamOptions, err := rawObject(raw, "stream_options")
		if err != nil {
			return parsedRequest{}, err
		}
		if err := rejectUnknownStrict(streamOptions, map[string]bool{"include_obfuscation": true}); err != nil {
			return parsedRequest{}, err
		}
	}
	output := chat.OutputSpec{Modalities: chat.ModalityText}
	if controls.ImageGeneration {
		output.Modalities |= chat.ModalityImage
	}
	if raw, ok := object["text"]; ok {
		format, err := parseResponsesText(raw)
		if err != nil {
			return parsedRequest{}, err
		}
		output.Format = format
	}
	if raw, ok := object["reasoning"]; ok {
		reasoning, err := parseResponsesReasoning(raw)
		if err != nil {
			return parsedRequest{}, err
		}
		controls.Reasoning = reasoning
	}
	var store *bool
	if _, ok := object["store"]; ok {
		reqStore, err := decodeBool(object, "store")
		if err != nil {
			return parsedRequest{}, err
		}
		store = reqStore
	}
	var previous *string
	if raw, ok := object["previous_response_id"]; ok {
		value, err := requireString(map[string]jsontext.Value{"previous_response_id": raw}, "previous_response_id")
		if err != nil {
			return parsedRequest{}, err
		}
		previous = &value
	}
	var metadata map[string]string
	if raw, ok := object["metadata"]; ok {
		if err := json.Unmarshal(raw, &metadata); err != nil {
			return parsedRequest{}, fmtError("metadata", "must be an object of strings", err)
		}
	}
	if _, ok := object["background"]; ok {
		return parsedRequest{}, chat.Unsupported("background", "background execution is not supported")
	}
	for _, key := range []string{"conversation", "include", "service_tier", "truncation", "user", "prompt", "max_tool_calls"} {
		if _, ok := object[key]; ok {
			return parsedRequest{}, chat.Unsupported(key, key+" is not supported")
		}
	}
	stream := false
	switch value, err := decodeBool(object, "stream"); {
	case err != nil:
		return parsedRequest{}, err
	case value != nil:
		stream = *value
	}
	instructions := ""
	if _, ok := object["instructions"]; ok {
		instructions, _, err = decodeString(object, "instructions")
		if err != nil {
			return parsedRequest{}, err
		}
	}
	return parsedRequest{
		Kind:     protocolResponses,
		Request:  chat.Request{Target: target, Instructions: instructions, Input: input, Controls: controls, Output: output},
		Previous: previous,
		Store:    store,
		Metadata: metadata,
		Stream:   stream,
	}, nil
}

func parseResponsesInput(raw jsontext.Value) ([]chat.Item, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []chat.Item{chat.MessageItem(chat.RoleUser, chat.TextPart(text))}, nil
	}
	values, err := rawArray(raw, "input")
	if err != nil {
		return nil, err
	}
	if len(values) == 0 {
		return nil, chat.Invalid("input", "input must not be empty")
	}
	items := make([]chat.Item, 0, len(values))
	for _, value := range values {
		item, err := parseResponsesItem(value)
		if err != nil {
			return nil, err
		}
		items = append(items, item...)
	}
	return items, nil
}

func parseResponsesItem(raw jsontext.Value) ([]chat.Item, error) {
	object, err := rawObject(raw, "input")
	if err != nil {
		return nil, err
	}
	typeName, ok, err := decodeString(object, "type")
	if err != nil {
		return nil, err
	}
	if !ok {
		typeName = "message"
	}
	switch typeName {
	case "message":
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "role": true, "content": true, "id": true}); err != nil {
			return nil, err
		}
		roleValue, err := requireString(object, "role")
		if err != nil {
			return nil, err
		}
		role := chat.Role(roleValue)
		if !validRole(role) {
			return nil, chat.Invalid("input.role", "unsupported message role "+roleValue)
		}
		rawContent, ok := object["content"]
		if !ok {
			return nil, chat.Invalid("input.content", "message content is required")
		}
		content, err := parseResponsesContent(rawContent, "input.content")
		if err != nil {
			return nil, err
		}
		item := chat.MessageItem(role, content...)
		item.ID, err = optionalString(object, "id")
		if err != nil {
			return nil, err
		}
		return []chat.Item{item}, nil
	case "function_call":
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "id": true, "call_id": true, "name": true, "arguments": true}); err != nil {
			return nil, err
		}
		callID, err := requireString(object, "call_id")
		if err != nil {
			return nil, err
		}
		name, err := requireString(object, "name")
		if err != nil {
			return nil, err
		}
		arguments, err := requireString(object, "arguments")
		if err != nil {
			return nil, err
		}
		if !jsontext.Value(arguments).IsValid() {
			return nil, chat.Invalid("input.arguments", "function call arguments must be valid JSON")
		}
		item := chat.FunctionCallItem(callID, name, arguments)
		item.ID, err = optionalString(object, "id")
		if err != nil {
			return nil, err
		}
		return []chat.Item{item}, nil
	case "function_call_output":
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "id": true, "call_id": true, "output": true}); err != nil {
			return nil, err
		}
		callID, err := requireString(object, "call_id")
		if err != nil {
			return nil, err
		}
		rawOutput, ok := object["output"]
		if !ok {
			return nil, chat.Invalid("input.output", "function call output is required")
		}
		parts, err := parseResponsesOutput(rawOutput, "input.output")
		if err != nil {
			return nil, err
		}
		item := chat.FunctionCallOutputItem(callID, parts...)
		item.ID, err = optionalString(object, "id")
		if err != nil {
			return nil, err
		}
		return []chat.Item{item}, nil
	case "reasoning":
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "id": true, "summary": true, "encrypted_content": true}); err != nil {
			return nil, err
		}
		itemID, err := optionalString(object, "id")
		if err != nil {
			return nil, err
		}
		item := chat.Item{Type: chat.ItemReasoning, ID: itemID, Status: chat.StatusCompleted}
		if rawSummary, ok := object["summary"]; ok {
			values, err := rawArray(rawSummary, "input.summary")
			if err != nil {
				return nil, err
			}
			for _, value := range values {
				summaryObject, err := rawObject(value, "input.summary")
				if err != nil {
					return nil, err
				}
				if err := rejectUnknownStrict(summaryObject, map[string]bool{"type": true, "text": true}); err != nil {
					return nil, err
				}
				typeName, ok, err := decodeString(summaryObject, "type")
				if err != nil {
					return nil, err
				}
				if ok && typeName != "summary_text" {
					return nil, chat.Unsupported("input.summary.type", "unsupported reasoning summary type "+typeName)
				}
				text, err := requireString(summaryObject, "text")
				if err != nil {
					return nil, err
				}
				item.Summary = append(item.Summary, chat.SummaryPart(text))
			}
		}
		if raw, ok := object["encrypted_content"]; ok {
			item.EncryptedContent = append(jsontext.Value(nil), raw...)
		}
		return []chat.Item{item}, nil
	default:
		return nil, chat.Unsupported("input.type", "input item type "+typeName+" is not supported")
	}
}

func parseResponsesContent(raw jsontext.Value, param string) ([]chat.Part, error) {
	return parseStringOrContent(raw, param, func(value jsontext.Value) ([]chat.Part, error) {
		values, err := rawArray(value, param)
		if err != nil {
			return nil, err
		}
		parts := make([]chat.Part, 0, len(values))
		for _, value := range values {
			object, err := rawObject(value, param)
			if err != nil {
				return nil, err
			}
			typeName, err := requireString(object, "type")
			if err != nil {
				return nil, err
			}
			switch typeName {
			case "input_text", "output_text":
				if err := rejectUnknownStrict(object, map[string]bool{"type": true, "text": true}); err != nil {
					return nil, err
				}
				text, err := requireString(object, "text")
				if err != nil {
					return nil, err
				}
				parts = append(parts, chat.TextPart(text))
			case "input_image":
				if err := rejectUnknownStrict(object, map[string]bool{"type": true, "image_url": true, "file_id": true, "detail": true}); err != nil {
					return nil, err
				}
				imageURL, ok, err := decodeString(object, "image_url")
				if err != nil {
					return nil, err
				}
				var media chat.Media
				fileID, fileIDOK, fileIDErr := decodeString(object, "file_id")
				if fileIDErr != nil {
					return nil, fileIDErr
				}
				if ok && fileIDOK {
					return nil, chat.Invalid(param, "input_image accepts only one of image_url or file_id")
				}
				detail, err := optionalString(object, "detail")
				if err != nil {
					return nil, err
				}
				if err := wire.ValidateImageDetail(detail, param+".detail"); err != nil {
					return nil, err
				}
				switch {
				case ok:
					media, err = parseMediaURL(imageURL, detail)
				case fileIDOK:
					media = chat.AssetMedia("image/*", fileID)
				default:
					return nil, chat.Invalid(param, "input_image requires image_url or file_id")
				}
				if err != nil {
					return nil, err
				}
				parts = append(parts, chat.Part{Type: chat.PartImage, Media: &media, Detail: detail})
			case "input_file":
				if err := rejectUnknownStrict(object, map[string]bool{"type": true, "file_data": true, "file_url": true, "file_id": true, "filename": true}); err != nil {
					return nil, err
				}
				media, filename, err := parseFileMedia(object, param)
				if err != nil {
					return nil, err
				}
				media.Filename = filename
				parts = append(parts, chat.FilePart(media))
			case "input_audio", "input_video":
				return nil, chat.Unsupported(param, typeName+" is not supported")
			default:
				return nil, chat.Unsupported(param, "unsupported Responses content type "+typeName)
			}
		}
		return parts, nil
	})
}

func parseResponsesOutput(raw jsontext.Value, param string) ([]chat.Part, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []chat.Part{chat.TextPart(text)}, nil
	}
	return parseResponsesContent(raw, param)
}

func parseResponsesTools(raw jsontext.Value) ([]chat.FunctionTool, bool, error) {
	values, err := rawArray(raw, "tools")
	if err != nil {
		return nil, false, err
	}
	tools := make([]chat.FunctionTool, 0, len(values))
	imageGeneration := false
	for _, value := range values {
		object, err := rawObject(value, "tools")
		if err != nil {
			return nil, false, err
		}
		typeName, ok, err := decodeString(object, "type")
		if err != nil {
			return nil, false, err
		}
		if !ok {
			typeName = "function"
		}
		switch typeName {
		case "image_generation":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true}); err != nil {
				return nil, false, err
			}
			switch {
			case len(object) != 1:
				return nil, false, chat.Unsupported("tools", "image_generation does not accept options in this compatibility profile")
			case imageGeneration:
				return nil, false, chat.Invalid("tools", "only one image_generation tool is supported")
			}
			imageGeneration = true
			continue
		case "function":
		default:
			return nil, false, chat.Unsupported("tools", "only function and image_generation tools are supported")
		}
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "name": true, "description": true, "parameters": true, "strict": true}); err != nil {
			return nil, false, err
		}
		name, err := requireString(object, "name")
		if err != nil {
			return nil, false, err
		}
		description, err := optionalString(object, "description")
		if err != nil {
			return nil, false, err
		}
		tool := chat.FunctionTool{Name: name, Description: description}
		if parameters, ok := object["parameters"]; ok {
			tool.Parameters = append(jsontext.Value(nil), parameters...)
		}
		switch strict, err := decodeBool(object, "strict"); {
		case err != nil:
			return nil, false, err
		default:
			tool.Strict = strict
		}
		tools = append(tools, tool)
	}
	return tools, imageGeneration, nil
}

func parseResponsesText(raw jsontext.Value) (chat.OutputFormat, error) {
	object, err := rawObject(raw, "text")
	if err != nil {
		return chat.OutputFormat{}, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"format": true}); err != nil {
		return chat.OutputFormat{}, err
	}
	if rawFormat, ok := object["format"]; ok {
		format, err := rawObject(rawFormat, "text.format")
		if err != nil {
			return chat.OutputFormat{}, err
		}
		if err := rejectUnknownStrict(format, map[string]bool{"type": true, "name": true, "description": true, "schema": true, "strict": true}); err != nil {
			return chat.OutputFormat{}, err
		}
		typeName, err := requireString(format, "type")
		if err != nil {
			return chat.OutputFormat{}, err
		}
		switch typeName {
		case "text":
			return chat.OutputFormat{Kind: chat.FormatText}, nil
		case "json_object":
			return chat.OutputFormat{Kind: chat.FormatJSONObject, Name: "response", Schema: jsontext.Value(`{"type":"object"}`)}, nil
		case "json_schema":
			name, err := requireString(format, "name")
			if err != nil {
				return chat.OutputFormat{}, err
			}
			schema := append(jsontext.Value(nil), format["schema"]...)
			if len(schema) == 0 || !schema.IsValid() {
				return chat.OutputFormat{}, chat.Invalid("text.format.schema", "schema must be valid JSON")
			}
			description, err := optionalString(format, "description")
			if err != nil {
				return chat.OutputFormat{}, err
			}
			strict, err := decodeBool(format, "strict")
			if err != nil {
				return chat.OutputFormat{}, err
			}
			return chat.OutputFormat{Kind: chat.FormatJSONSchema, Name: name, Description: description, Schema: schema, Strict: strict != nil && *strict}, nil
		default:
			return chat.OutputFormat{}, chat.Unsupported("text.format", "unsupported text format "+typeName)
		}
	}
	return chat.OutputFormat{Kind: chat.FormatText}, nil
}

func parseResponsesReasoning(raw jsontext.Value) (*chat.ReasoningControl, error) {
	object, err := rawObject(raw, "reasoning")
	if err != nil {
		return nil, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"effort": true, "summary": true}); err != nil {
		return nil, err
	}
	reasoning := &chat.ReasoningControl{}
	switch value, ok, err := decodeString(object, "effort"); {
	case err != nil:
		return nil, err
	case ok:
		reasoning.Effort = value
	}
	if raw, ok := object["summary"]; ok && string(raw) != "null" {
		value, _, err := decodeString(object, "summary")
		if err != nil {
			return nil, err
		}
		switch value {
		case "none":
			reasoning.Summary = false
		case "auto", "concise", "detailed":
			reasoning.Summary = true
		default:
			return nil, chat.Unsupported("reasoning.summary", "supported values are none, auto, concise, and detailed")
		}
	}
	return reasoning, nil
}
