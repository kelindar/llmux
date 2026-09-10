package completions

import (
	"encoding/json/jsontext"
	json "encoding/json/v2"

	"github.com/kelindar/llmux/chat"
)

// ParseRequest decodes a Chat Completions request into a canonical request.
func ParseRequest(object map[string]jsontext.Value) (parsedRequest, error) {
	allowed := map[string]bool{
		"model": true, "messages": true, "stream": true, "stream_options": true,
		"max_tokens": true, "max_completion_tokens": true, "temperature": true,
		"top_p": true, "stop": true, "tools": true, "tool_choice": true,
		"parallel_tool_calls": true, "response_format": true, "modalities": true,
		"audio": true, "n": true, "logprobs": true, "top_logprobs": true,
		"user": true, "store": true, "metadata": true, "reasoning_effort": true,
	}
	if err := rejectUnknown(object, allowed); err != nil {
		return parsedRequest{}, err
	}
	target, err := requireString(object, "model")
	if err != nil {
		return parsedRequest{}, err
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
		items, err := parseChatMessage(value)
		if err != nil {
			return parsedRequest{}, err
		}
		input = append(input, items...)
	}
	controls := chat.Controls{Extensions: namespacedExtensions(object, allowed)}
	if maxTokens, err := decodeInt(object, "max_tokens"); err != nil {
		return parsedRequest{}, err
	} else if maxTokens != nil {
		controls.MaxOutputTokens = maxTokens
	}
	if maxTokens, err := decodeInt(object, "max_completion_tokens"); err != nil {
		return parsedRequest{}, err
	} else if maxTokens != nil {
		if controls.MaxOutputTokens != nil {
			return parsedRequest{}, chat.Invalid("max_completion_tokens", "max_tokens and max_completion_tokens cannot both be set")
		}
		controls.MaxOutputTokens = maxTokens
	}
	if controls.MaxOutputTokens != nil && *controls.MaxOutputTokens < 1 {
		return parsedRequest{}, chat.Invalid("max_tokens", "max_tokens must be positive")
	}
	if value, err := decodeFloat(object, "temperature"); err != nil {
		return parsedRequest{}, err
	} else if value != nil {
		if *value < 0 || *value > 2 {
			return parsedRequest{}, chat.Invalid("temperature", "temperature must be between 0 and 2")
		}
		controls.Temperature = value
	}
	if value, err := decodeFloat(object, "top_p"); err != nil {
		return parsedRequest{}, err
	} else if value != nil {
		if *value < 0 || *value > 1 {
			return parsedRequest{}, chat.Invalid("top_p", "top_p must be between 0 and 1")
		}
		controls.TopP = value
	}
	stop, err := parseStop(object["stop"])
	if err != nil {
		return parsedRequest{}, err
	}
	controls.Stop = stop
	if raw, ok := object["tools"]; ok {
		controls.Tools, err = parseChatTools(raw)
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
		if err := rejectUnknownStrict(streamOptions, map[string]bool{"include_usage": true}); err != nil {
			return parsedRequest{}, err
		}
		includeUsage, err := decodeBool(streamOptions, "include_usage")
		if err != nil {
			return parsedRequest{}, err
		}
		controls.IncludeUsage = includeUsage != nil && *includeUsage
	}
	output := chat.OutputSpec{Modalities: chat.ModalityText}
	modalitiesProvided := false
	if raw, ok := object["modalities"]; ok {
		modalitiesProvided = true
		output.Modalities, err = parseModalities(raw, "modalities")
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if raw, ok := object["audio"]; ok {
		controls.Audio, err = parseAudioControls(raw)
		if err != nil {
			return parsedRequest{}, err
		}
		output.Modalities |= chat.ModalityAudio
	}
	switch {
	case controls.Audio != nil && modalitiesProvided && !output.Modalities.Has(chat.ModalityAudio):
		return parsedRequest{}, chat.Invalid("modalities", "audio output requires audio in modalities")
	case output.Modalities.Has(chat.ModalityAudio) && controls.Audio == nil:
		return parsedRequest{}, chat.Invalid("audio", "audio controls are required for audio output")
	}
	if raw, ok := object["response_format"]; ok {
		format, err := parseResponseFormat(raw)
		if err != nil {
			return parsedRequest{}, err
		}
		output.Format = format
	}
	if raw, ok := object["n"]; ok {
		var n int
		if err := json.Unmarshal(raw, &n); err != nil {
			return parsedRequest{}, fmtError("n", "must be an integer", err)
		}
		if n != 1 {
			return parsedRequest{}, chat.Unsupported("n", "only n=1 is supported")
		}
	}
	if raw, ok := object["logprobs"]; ok {
		var enabled bool
		if err := json.Unmarshal(raw, &enabled); err != nil {
			return parsedRequest{}, fmtError("logprobs", "must be a boolean", err)
		}
		if enabled {
			return parsedRequest{}, chat.Unsupported("logprobs", "log probabilities are not supported")
		}
	}
	if _, ok := object["top_logprobs"]; ok {
		return parsedRequest{}, chat.Unsupported("top_logprobs", "log probabilities are not supported")
	}
	var store *bool
	if _, ok := object["store"]; ok {
		store, err = decodeBool(object, "store")
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if raw, ok := object["reasoning_effort"]; ok {
		effort, err := requireString(map[string]jsontext.Value{"reasoning_effort": raw}, "reasoning_effort")
		if err != nil {
			return parsedRequest{}, err
		}
		controls.Reasoning = &chat.ReasoningControl{Effort: effort}
	}
	var metadata map[string]string
	if raw, ok := object["metadata"]; ok {
		if err := json.Unmarshal(raw, &metadata); err != nil {
			return parsedRequest{}, fmtError("metadata", "must be an object of strings", err)
		}
	}
	if raw, ok := object["user"]; ok {
		if _, err := requireString(map[string]jsontext.Value{"user": raw}, "user"); err != nil {
			return parsedRequest{}, err
		}
		return parsedRequest{}, chat.Unsupported("user", "the user field is not supported")
	}
	stream := false
	if value, err := decodeBool(object, "stream"); err != nil {
		return parsedRequest{}, err
	} else if value != nil {
		stream = *value
	}
	return parsedRequest{Kind: protocolChat, Request: chat.Request{Target: target, Input: input, Controls: controls, Output: output, Store: store, Metadata: metadata}, Stream: stream}, nil
}

func parseChatMessage(raw jsontext.Value) ([]chat.Item, error) {
	object, err := rawObject(raw, "messages")
	if err != nil {
		return nil, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"role": true, "content": true, "tool_calls": true, "tool_call_id": true}); err != nil {
		return nil, err
	}
	roleValue, err := requireString(object, "role")
	if err != nil {
		return nil, err
	}
	role := chat.Role(roleValue)
	if !validRole(role) {
		return nil, chat.Invalid("messages.role", "unsupported message role "+roleValue)
	}
	var content []chat.Part
	if rawContent, ok := object["content"]; ok {
		content, err = parseChatContent(rawContent)
		if err != nil {
			return nil, err
		}
	}
	if role == chat.RoleTool {
		callID, err := requireString(object, "tool_call_id")
		if err != nil {
			return nil, err
		}
		return []chat.Item{chat.FunctionCallOutputItem(callID, content...)}, nil
	}
	items := make([]chat.Item, 0, 2)
	if len(content) > 0 || role != chat.RoleAssistant {
		items = append(items, chat.MessageItem(role, content...))
	}
	if rawCalls, ok := object["tool_calls"]; ok {
		calls, err := rawArray(rawCalls, "messages.tool_calls")
		if err != nil {
			return nil, err
		}
		for _, rawCall := range calls {
			call, err := parseChatToolCall(rawCall)
			if err != nil {
				return nil, err
			}
			items = append(items, call)
		}
	}
	if len(items) == 0 {
		items = append(items, chat.MessageItem(role))
	}
	return items, nil
}

func parseChatToolCall(raw jsontext.Value) (chat.Item, error) {
	object, err := rawObject(raw, "messages.tool_calls")
	if err != nil {
		return chat.Item{}, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"id": true, "type": true, "function": true}); err != nil {
		return chat.Item{}, err
	}
	if typeName, ok, err := decodeString(object, "type"); err != nil {
		return chat.Item{}, err
	} else if ok && typeName != "function" {
		return chat.Item{}, chat.Unsupported("messages.tool_calls.type", "only function tool calls are supported")
	}
	callID, err := requireString(object, "id")
	if err != nil {
		return chat.Item{}, err
	}
	function, err := rawObject(object["function"], "messages.tool_calls.function")
	if err != nil {
		return chat.Item{}, err
	}
	if err := rejectUnknownStrict(function, map[string]bool{"name": true, "arguments": true}); err != nil {
		return chat.Item{}, err
	}
	name, err := requireString(function, "name")
	if err != nil {
		return chat.Item{}, err
	}
	arguments, err := requireString(function, "arguments")
	if err != nil {
		return chat.Item{}, err
	}
	if !jsontext.Value(arguments).IsValid() {
		return chat.Item{}, chat.Invalid("messages.tool_calls.function.arguments", "arguments must be valid JSON")
	}
	return chat.FunctionCallItem(callID, name, arguments), nil
}

func parseChatTools(raw jsontext.Value) ([]chat.FunctionTool, error) {
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
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "function": true}); err != nil {
			return nil, err
		}
		typeName := "function"
		if value, ok, err := decodeString(object, "type"); err != nil {
			return nil, err
		} else if ok {
			typeName = value
		}
		if typeName != "function" {
			return nil, chat.Unsupported("tools", "only function tools are supported")
		}
		function, err := rawObject(object["function"], "tools.function")
		if err != nil {
			return nil, err
		}
		if err := rejectUnknownStrict(function, map[string]bool{"name": true, "description": true, "parameters": true, "strict": true}); err != nil {
			return nil, err
		}
		name, err := requireString(function, "name")
		if err != nil {
			return nil, err
		}
		tool := chat.FunctionTool{Name: name}
		if value, ok, err := decodeString(function, "description"); err != nil {
			return nil, err
		} else if ok {
			tool.Description = value
		}
		if parameters, ok := function["parameters"]; ok {
			tool.Parameters = append(jsontext.Value(nil), parameters...)
		}
		if strict, err := decodeBool(function, "strict"); err != nil {
			return nil, err
		} else {
			tool.Strict = strict
		}
		tools = append(tools, tool)
	}
	return tools, nil
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
		if typeName, ok, err := decodeString(object, "type"); err != nil {
			return nil, err
		} else if ok && typeName != "function" {
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

func parseStop(raw jsontext.Value) ([]string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		return []string{one}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err != nil {
		return nil, fmtError("stop", "must be a string or array of strings", err)
	}
	return many, nil
}

func parseModalities(raw jsontext.Value, param string) (chat.Modality, error) {
	values, err := rawArray(raw, param)
	if err != nil {
		return 0, err
	}
	var modalities chat.Modality
	for _, value := range values {
		var name string
		if err := json.Unmarshal(value, &name); err != nil {
			return 0, fmtError(param, "must contain strings", err)
		}
		switch name {
		case "text":
			modalities |= chat.ModalityText
		case "audio":
			modalities |= chat.ModalityAudio
		default:
			return 0, chat.Unsupported(param, "unsupported output modality "+name)
		}
	}
	if modalities == 0 {
		return 0, chat.Invalid(param, "at least one output modality is required")
	}
	return modalities, nil
}

func parseAudioControls(raw jsontext.Value) (*chat.AudioControls, error) {
	object, err := rawObject(raw, "audio")
	if err != nil {
		return nil, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"voice": true, "format": true}); err != nil {
		return nil, err
	}
	voice, err := requireString(object, "voice")
	if err != nil {
		return nil, err
	}
	format, err := requireString(object, "format")
	if err != nil {
		return nil, err
	}
	return &chat.AudioControls{Voice: voice, Format: format}, nil
}

func parseResponseFormat(raw jsontext.Value) (chat.OutputFormat, error) {
	object, err := rawObject(raw, "response_format")
	if err != nil {
		return chat.OutputFormat{}, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"type": true, "json_schema": true}); err != nil {
		return chat.OutputFormat{}, err
	}
	typeName, err := requireString(object, "type")
	if err != nil {
		return chat.OutputFormat{}, err
	}
	switch typeName {
	case "text":
		return chat.OutputFormat{Kind: chat.FormatText}, nil
	case "json_object":
		return chat.OutputFormat{Kind: chat.FormatJSONObject, Name: "response", Schema: jsontext.Value(`{"type":"object"}`)}, nil
	case "json_schema":
		schemaObject, err := rawObject(object["json_schema"], "response_format.json_schema")
		if err != nil {
			return chat.OutputFormat{}, err
		}
		if err := rejectUnknownStrict(schemaObject, map[string]bool{"name": true, "description": true, "schema": true, "strict": true}); err != nil {
			return chat.OutputFormat{}, err
		}
		name, err := requireString(schemaObject, "name")
		if err != nil {
			return chat.OutputFormat{}, err
		}
		schema := append(jsontext.Value(nil), schemaObject["schema"]...)
		if len(schema) == 0 || !schema.IsValid() {
			return chat.OutputFormat{}, chat.Invalid("response_format.json_schema.schema", "schema must be valid JSON")
		}
		description := ""
		if value, ok, err := decodeString(schemaObject, "description"); err != nil {
			return chat.OutputFormat{}, err
		} else if ok {
			description = value
		}
		strict, err := decodeBool(schemaObject, "strict")
		if err != nil {
			return chat.OutputFormat{}, err
		}
		return chat.OutputFormat{Kind: chat.FormatJSONSchema, Name: name, Description: description, Schema: schema, Strict: strict != nil && *strict}, nil
	default:
		return chat.OutputFormat{}, chat.Unsupported("response_format", "unsupported response format "+typeName)
	}
}
