// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package completions

import (
	"bytes"
	"encoding/base64"
	"encoding/json/jsontext"
	"strings"

	"github.com/buger/jsonparser"
	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/wire"
)

type field = wire.Value

type requestDecoder struct {
	model               field
	messages            field
	stream              field
	streamOptions       field
	maxTokens           field
	maxCompletionTokens field
	temperature         field
	topP                field
	stop                field
	tools               field
	toolChoice          field
	parallelToolCalls   field
	responseFormat      field
	modalities          field
	audio               field
	n                   field
	logprobs            field
	topLogprobs         field
	user                field
	store               field
	metadata            field
	reasoningEffort     field
	extensions          map[string]jsontext.Value
}

// ParseRequest decodes a Chat Completions request directly from its body.
// Returned strings, slices, maps, and raw JSON own their data independently
// of body after this function returns.
func ParseRequest(data []byte) (parsedRequest, error) {
	if err := wire.ValidateObject(data); err != nil {
		return parsedRequest{}, fmtError("body", "request body must be valid JSON", err)
	}

	var decoder requestDecoder
	if err := jsonparser.ObjectEach(data, decoder.field); err != nil {
		if _, ok := err.(*chat.Error); ok {
			return parsedRequest{}, err
		}
		return parsedRequest{}, fmtError("body", "request body must be valid JSON", err)
	}
	return decoder.parse()
}

func (d *requestDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("model")):
		d.model = value
	case bytes.Equal(key, []byte("messages")):
		d.messages = value
	case bytes.Equal(key, []byte("stream")):
		d.stream = value
	case bytes.Equal(key, []byte("stream_options")):
		d.streamOptions = value
	case bytes.Equal(key, []byte("max_tokens")):
		d.maxTokens = value
	case bytes.Equal(key, []byte("max_completion_tokens")):
		d.maxCompletionTokens = value
	case bytes.Equal(key, []byte("temperature")):
		d.temperature = value
	case bytes.Equal(key, []byte("top_p")):
		d.topP = value
	case bytes.Equal(key, []byte("stop")):
		d.stop = value
	case bytes.Equal(key, []byte("tools")):
		d.tools = value
	case bytes.Equal(key, []byte("tool_choice")):
		d.toolChoice = value
	case bytes.Equal(key, []byte("parallel_tool_calls")):
		d.parallelToolCalls = value
	case bytes.Equal(key, []byte("response_format")):
		d.responseFormat = value
	case bytes.Equal(key, []byte("modalities")):
		d.modalities = value
	case bytes.Equal(key, []byte("audio")):
		d.audio = value
	case bytes.Equal(key, []byte("n")):
		d.n = value
	case bytes.Equal(key, []byte("logprobs")):
		d.logprobs = value
	case bytes.Equal(key, []byte("top_logprobs")):
		d.topLogprobs = value
	case bytes.Equal(key, []byte("user")):
		d.user = value
	case bytes.Equal(key, []byte("store")):
		d.store = value
	case bytes.Equal(key, []byte("metadata")):
		d.metadata = value
	case bytes.Equal(key, []byte("reasoning_effort")):
		d.reasoningEffort = value
	case bytes.HasPrefix(key, []byte("x-")) || bytes.IndexByte(key, ':') >= 0:
		if d.extensions == nil {
			d.extensions = make(map[string]jsontext.Value)
		}
		d.extensions[string(key)] = wire.Copy(raw)
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

func (d requestDecoder) parse() (parsedRequest, error) {
	target, err := requiredString(d.model, "model")
	if err != nil {
		return parsedRequest{}, err
	}
	if !d.messages.Present() {
		return parsedRequest{}, chat.Invalid("messages", "messages is required")
	}
	input, err := parseMessages(d.messages)
	if err != nil {
		return parsedRequest{}, err
	}
	controls := chat.Controls{Extensions: d.extensions}

	maxTokens, err := decodeInt(d.maxTokens, "max_tokens")
	if err != nil {
		return parsedRequest{}, err
	}
	controls.MaxOutputTokens = maxTokens
	maxCompletionTokens, err := decodeInt(d.maxCompletionTokens, "max_completion_tokens")
	if err != nil {
		return parsedRequest{}, err
	}
	if maxCompletionTokens != nil {
		if controls.MaxOutputTokens != nil {
			return parsedRequest{}, chat.Invalid("max_completion_tokens", "max_tokens and max_completion_tokens cannot both be set")
		}
		controls.MaxOutputTokens = maxCompletionTokens
	}
	if controls.MaxOutputTokens != nil && *controls.MaxOutputTokens < 1 {
		return parsedRequest{}, chat.Invalid("max_tokens", "max_tokens must be positive")
	}

	controls.Temperature, err = decodeFloat(d.temperature, "temperature")
	if err != nil {
		return parsedRequest{}, err
	}
	if controls.Temperature != nil && (*controls.Temperature < 0 || *controls.Temperature > 2) {
		return parsedRequest{}, chat.Invalid("temperature", "temperature must be between 0 and 2")
	}
	controls.TopP, err = decodeFloat(d.topP, "top_p")
	if err != nil {
		return parsedRequest{}, err
	}
	if controls.TopP != nil && (*controls.TopP < 0 || *controls.TopP > 1) {
		return parsedRequest{}, chat.Invalid("top_p", "top_p must be between 0 and 1")
	}

	controls.Stop, err = parseStop(d.stop)
	if err != nil {
		return parsedRequest{}, err
	}
	if d.tools.Present() {
		controls.Tools, err = parseChatTools(d.tools)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if d.toolChoice.Present() {
		controls.ToolChoice, err = parseToolChoice(d.toolChoice)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	controls.ParallelToolCall, err = decodeBool(d.parallelToolCalls, "parallel_tool_calls")
	if err != nil {
		return parsedRequest{}, err
	}

	includeUsage, err := parseStreamOptions(d.streamOptions)
	if err != nil {
		return parsedRequest{}, err
	}
	output := chat.OutputSpec{Modalities: chat.ModalityText}
	modalitiesProvided := d.modalities.Present()
	if modalitiesProvided {
		output.Modalities, err = parseModalities(d.modalities, "modalities")
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if d.audio.Present() {
		controls.Audio, err = parseAudioControls(d.audio)
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
	if d.responseFormat.Present() {
		output.Format, err = parseResponseFormat(d.responseFormat)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if d.n.Present() {
		n, err := decodeInt(d.n, "n")
		if err != nil {
			return parsedRequest{}, fmtError("n", "must be an integer", err)
		}
		if n == nil || *n != 1 {
			return parsedRequest{}, chat.Unsupported("n", "only n=1 is supported")
		}
	}
	if d.logprobs.Present() {
		enabled, err := decodeBool(d.logprobs, "logprobs")
		if err != nil {
			return parsedRequest{}, fmtError("logprobs", "must be a boolean", err)
		}
		if enabled != nil && *enabled {
			return parsedRequest{}, chat.Unsupported("logprobs", "log probabilities are not supported")
		}
	}
	if d.topLogprobs.Present() {
		return parsedRequest{}, chat.Unsupported("top_logprobs", "log probabilities are not supported")
	}
	store, err := decodeBool(d.store, "store")
	if err != nil {
		return parsedRequest{}, err
	}
	if d.reasoningEffort.Present() {
		effort, err := requiredString(d.reasoningEffort, "reasoning_effort")
		if err != nil {
			return parsedRequest{}, err
		}
		controls.Reasoning = &chat.ReasoningControl{Effort: effort}
	}
	metadata, err := parseMetadata(d.metadata)
	if err != nil {
		return parsedRequest{}, err
	}
	if d.user.Present() {
		if _, err := requiredString(d.user, "user"); err != nil {
			return parsedRequest{}, err
		}
		return parsedRequest{}, chat.Unsupported("user", "the user field is not supported")
	}
	stream, err := decodeBool(d.stream, "stream")
	if err != nil {
		return parsedRequest{}, err
	}

	return parsedRequest{
		Kind: protocolChat,
		Request: chat.Request{
			Target: target, Input: input, Controls: controls, Output: output,
		},
		Store:        store,
		Metadata:     metadata,
		Stream:       stream != nil && *stream,
		IncludeUsage: includeUsage,
	}, nil
}

func parseMessages(value field) ([]chat.Item, error) {
	if value.Type != jsonparser.Array {
		return nil, chat.Invalid("messages", "must be a JSON array")
	}
	items := make([]chat.Item, 0, 4)
	count := 0
	var parseErr error
	_, err := jsonparser.ArrayEach(value.Raw, func(raw []byte, typ jsonparser.ValueType, _ int, callbackErr error) {
		if parseErr != nil {
			return
		}
		if callbackErr != nil {
			parseErr = callbackErr
			return
		}
		count++
		parseErr = appendChatMessage(&items, raw, typ)
	})
	if parseErr != nil {
		if _, ok := parseErr.(*chat.Error); ok {
			return nil, parseErr
		}
		return nil, chat.Invalid("messages", "must be a JSON array")
	}
	if err != nil {
		return nil, chat.Invalid("messages", "must be a JSON array")
	}
	if count == 0 {
		return nil, chat.Invalid("messages", "messages must not be empty")
	}
	return items, nil
}

type messageDecoder struct {
	role       field
	content    field
	toolCalls  field
	toolCallID field
}

func (d *messageDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("role")):
		d.role = value
	case bytes.Equal(key, []byte("content")):
		d.content = value
	case bytes.Equal(key, []byte("tool_calls")):
		d.toolCalls = value
	case bytes.Equal(key, []byte("tool_call_id")):
		d.toolCallID = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

func appendChatMessage(items *[]chat.Item, raw []byte, typ jsonparser.ValueType) error {
	if typ != jsonparser.Object {
		return chat.Invalid("messages", "must be a JSON object")
	}
	var decoder messageDecoder
	if err := objectEach(raw, "messages", decoder.field); err != nil {
		return err
	}
	roleValue, err := requiredString(decoder.role, "role")
	if err != nil {
		return err
	}
	role := chat.Role(roleValue)
	if !validRole(role) {
		return chat.Invalid("messages.role", "unsupported message role "+roleValue)
	}
	var content []chat.Part
	if decoder.content.Present() {
		content, err = parseChatContent(decoder.content)
		if err != nil {
			return err
		}
	}
	if role == chat.RoleTool {
		callID, err := requiredString(decoder.toolCallID, "tool_call_id")
		if err != nil {
			return err
		}
		*items = append(*items, chat.Item{Type: chat.ItemFunctionCallOutput, CallID: callID, Output: content})
		return nil
	}
	start := len(*items)
	if len(content) > 0 || role != chat.RoleAssistant {
		*items = append(*items, chat.Item{Type: chat.ItemMessage, Role: role, Content: content})
	}
	if decoder.toolCalls.Present() {
		if err := appendChatToolCalls(items, decoder.toolCalls); err != nil {
			return err
		}
	}
	if len(*items) == start {
		*items = append(*items, chat.Item{Type: chat.ItemMessage, Role: role})
	}
	return nil
}

func appendChatToolCalls(items *[]chat.Item, value field) error {
	if value.Type != jsonparser.Array {
		return chat.Invalid("messages.tool_calls", "must be a JSON array")
	}
	var parseErr error
	_, err := jsonparser.ArrayEach(value.Raw, func(raw []byte, typ jsonparser.ValueType, _ int, callbackErr error) {
		if parseErr != nil {
			return
		}
		if callbackErr != nil {
			parseErr = callbackErr
			return
		}
		call, err := parseChatToolCall(raw, typ)
		if err != nil {
			parseErr = err
			return
		}
		*items = append(*items, call)
	})
	if parseErr != nil {
		return parseErr
	}
	if err != nil {
		return chat.Invalid("messages.tool_calls", "must be a JSON array")
	}
	return nil
}

type toolCallDecoder struct {
	id       field
	typeName field
	function field
}

func (d *toolCallDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("id")):
		d.id = value
	case bytes.Equal(key, []byte("type")):
		d.typeName = value
	case bytes.Equal(key, []byte("function")):
		d.function = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

type functionCallDecoder struct {
	name      field
	arguments field
}

func (d *functionCallDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("name")):
		d.name = value
	case bytes.Equal(key, []byte("arguments")):
		d.arguments = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

func parseChatToolCall(raw []byte, typ jsonparser.ValueType) (chat.Item, error) {
	if typ != jsonparser.Object {
		return chat.Item{}, chat.Invalid("messages.tool_calls", "must be a JSON object")
	}
	var decoder toolCallDecoder
	if err := objectEach(raw, "messages.tool_calls", decoder.field); err != nil {
		return chat.Item{}, err
	}
	if decoder.typeName.Present() {
		typeName, _, err := decodeString(decoder.typeName, "messages.tool_calls.type")
		if err != nil {
			return chat.Item{}, err
		}
		if typeName != "function" {
			return chat.Item{}, chat.Unsupported("messages.tool_calls.type", "only function tool calls are supported")
		}
	}
	callID, err := requiredString(decoder.id, "id")
	if err != nil {
		return chat.Item{}, err
	}
	if !decoder.function.Present() {
		return chat.Item{}, chat.Invalid("messages.tool_calls.function", "must be a JSON object")
	}
	var function functionCallDecoder
	if err := objectEach(decoder.function.Raw, "messages.tool_calls.function", function.field); err != nil {
		return chat.Item{}, err
	}
	name, err := requiredString(function.name, "name")
	if err != nil {
		return chat.Item{}, err
	}
	arguments, err := requiredString(function.arguments, "arguments")
	if err != nil {
		return chat.Item{}, err
	}
	if !jsontext.Value(arguments).IsValid() {
		return chat.Item{}, chat.Invalid("messages.tool_calls.function.arguments", "arguments must be valid JSON")
	}
	return chat.FunctionCallItem(callID, name, arguments), nil
}

type toolDecoder struct {
	typeName    field
	function    field
	name        field
	description field
	parameters  field
	strict      field
}

func (d *toolDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("type")):
		d.typeName = value
	case bytes.Equal(key, []byte("function")):
		d.function = value
	case bytes.Equal(key, []byte("name")):
		d.name = value
	case bytes.Equal(key, []byte("description")):
		d.description = value
	case bytes.Equal(key, []byte("parameters")):
		d.parameters = value
	case bytes.Equal(key, []byte("strict")):
		d.strict = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

type functionToolDecoder struct {
	name        field
	description field
	parameters  field
	strict      field
}

func (d *functionToolDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("name")):
		d.name = value
	case bytes.Equal(key, []byte("description")):
		d.description = value
	case bytes.Equal(key, []byte("parameters")):
		d.parameters = value
	case bytes.Equal(key, []byte("strict")):
		d.strict = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

func parseChatTools(value field) ([]chat.FunctionTool, error) {
	if value.Type != jsonparser.Array {
		return nil, chat.Invalid("tools", "must be a JSON array")
	}
	tools := make([]chat.FunctionTool, 0, 4)
	var parseErr error
	_, err := jsonparser.ArrayEach(value.Raw, func(raw []byte, typ jsonparser.ValueType, _ int, callbackErr error) {
		if parseErr != nil {
			return
		}
		if callbackErr != nil {
			parseErr = callbackErr
			return
		}
		if typ != jsonparser.Object {
			parseErr = chat.Invalid("tools", "must be a JSON object")
			return
		}
		var decoder toolDecoder
		if err := objectEach(raw, "tools", decoder.field); err != nil {
			parseErr = err
			return
		}
		typeName := "function"
		if decoder.typeName.Present() {
			decoded, _, err := decodeString(decoder.typeName, "type")
			if err != nil {
				parseErr = err
				return
			}
			typeName = decoded
		}
		if typeName != "function" {
			parseErr = chat.Unsupported("tools", "only function tools are supported")
			return
		}
		if !decoder.function.Present() {
			parseErr = chat.Invalid("tools.function", "must be a JSON object")
			return
		}
		var function functionToolDecoder
		if err := objectEach(decoder.function.Raw, "tools.function", function.field); err != nil {
			parseErr = err
			return
		}
		name, err := requiredString(function.name, "name")
		if err != nil {
			parseErr = err
			return
		}
		description, _, err := decodeString(function.description, "description")
		if err != nil {
			parseErr = err
			return
		}
		strict, err := decodeBool(function.strict, "strict")
		if err != nil {
			parseErr = err
			return
		}
		tool := chat.FunctionTool{Name: name, Description: description, Strict: strict}
		if function.parameters.Present() {
			tool.Parameters = wire.Copy(function.parameters.Raw)
		}
		tools = append(tools, tool)
	})
	if parseErr != nil {
		return nil, parseErr
	}
	if err != nil {
		return nil, chat.Invalid("tools", "must be a JSON array")
	}
	return tools, nil
}

type choiceDecoder struct {
	typeName field
	function field
	name     field
}

func (d *choiceDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("type")):
		d.typeName = value
	case bytes.Equal(key, []byte("function")):
		d.function = value
	case bytes.Equal(key, []byte("name")):
		d.name = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

type choiceFunctionDecoder struct{ name field }

func (d *choiceFunctionDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	if bytes.Equal(key, []byte("name")) {
		d.name = field{Raw: raw, Type: typ}
		return nil
	}
	return chat.Unsupported(string(key), "request field is not supported by llmux")
}

func parseToolChoice(value field) (*chat.ToolChoice, error) {
	if value.Type == jsonparser.String || value.Type == jsonparser.Null {
		mode, _, err := decodeString(value, "tool_choice")
		if err != nil {
			return nil, err
		}
		switch mode {
		case "auto", "none", "required":
			return &chat.ToolChoice{Mode: mode}, nil
		default:
			return nil, chat.Invalid("tool_choice", "unsupported tool_choice "+mode)
		}
	}
	if value.Type != jsonparser.Object {
		return nil, chat.Invalid("tool_choice", "must be a JSON object")
	}
	var decoder choiceDecoder
	if err := objectEach(value.Raw, "tool_choice", decoder.field); err != nil {
		return nil, err
	}
	if decoder.function.Present() {
		if decoder.name.Present() {
			return nil, chat.Unsupported("name", "request field is not supported by llmux")
		}
		if decoder.typeName.Present() {
			typeName, _, err := decodeString(decoder.typeName, "tool_choice.type")
			if err != nil {
				return nil, err
			}
			if typeName != "function" {
				return nil, chat.Unsupported("tool_choice.type", "only function tool choices are supported")
			}
		}
		if decoder.function.Type != jsonparser.Object {
			return nil, chat.Invalid("tool_choice.function", "must be a JSON object")
		}
		var function choiceFunctionDecoder
		if err := objectEach(decoder.function.Raw, "tool_choice.function", function.field); err != nil {
			return nil, err
		}
		name, err := requiredString(function.name, "name")
		if err != nil {
			return nil, err
		}
		return &chat.ToolChoice{Mode: "function", Name: name}, nil
	}
	typeName, err := requiredString(decoder.typeName, "type")
	if err != nil {
		return nil, err
	}
	if typeName != "function" {
		return nil, chat.Invalid("tool_choice", "unsupported tool_choice type "+typeName)
	}
	name, err := requiredString(decoder.name, "name")
	if err != nil {
		return nil, err
	}
	return &chat.ToolChoice{Mode: "function", Name: name}, nil
}

func parseStop(value field) ([]string, error) {
	if !value.Present() || value.Type == jsonparser.Null {
		return nil, nil
	}
	if value.Type == jsonparser.String {
		stop, _, err := decodeString(value, "stop")
		if err != nil {
			return nil, fmtError("stop", "must be a string or array of strings", err)
		}
		return []string{stop}, nil
	}
	if value.Type != jsonparser.Array {
		return nil, fmtError("stop", "must be a string or array of strings", wire.ErrType)
	}
	stop := make([]string, 0, 4)
	var parseErr error
	_, err := jsonparser.ArrayEach(value.Raw, func(raw []byte, typ jsonparser.ValueType, _ int, callbackErr error) {
		if parseErr != nil {
			return
		}
		if callbackErr != nil {
			parseErr = callbackErr
			return
		}
		decoded, err := wire.String(raw, typ)
		if err != nil {
			parseErr = err
			return
		}
		stop = append(stop, decoded)
	})
	if parseErr != nil || err != nil {
		if parseErr == nil {
			parseErr = err
		}
		return nil, fmtError("stop", "must be a string or array of strings", parseErr)
	}
	return stop, nil
}

func parseModalities(value field, param string) (chat.Modality, error) {
	if value.Type != jsonparser.Array {
		return 0, chat.Invalid(param, "must be a JSON array")
	}
	var modalities chat.Modality
	var parseErr error
	_, err := jsonparser.ArrayEach(value.Raw, func(raw []byte, typ jsonparser.ValueType, _ int, callbackErr error) {
		if parseErr != nil {
			return
		}
		if callbackErr != nil {
			parseErr = callbackErr
			return
		}
		name, err := wire.String(raw, typ)
		if err != nil {
			parseErr = fmtError(param, "must contain strings", err)
			return
		}
		switch name {
		case "text":
			modalities |= chat.ModalityText
		case "audio":
			modalities |= chat.ModalityAudio
		default:
			parseErr = chat.Unsupported(param, "unsupported output modality "+name)
		}
	})
	if parseErr != nil {
		return 0, parseErr
	}
	if err != nil {
		return 0, chat.Invalid(param, "must be a JSON array")
	}
	if modalities == 0 {
		return 0, chat.Invalid(param, "at least one output modality is required")
	}
	return modalities, nil
}

type audioDecoder struct {
	voice  field
	format field
}

func (d *audioDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("voice")):
		d.voice = value
	case bytes.Equal(key, []byte("format")):
		d.format = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

func parseAudioControls(value field) (*chat.AudioControls, error) {
	if value.Type != jsonparser.Object {
		return nil, chat.Invalid("audio", "must be a JSON object")
	}
	var decoder audioDecoder
	if err := objectEach(value.Raw, "audio", decoder.field); err != nil {
		return nil, err
	}
	voice, err := requiredString(decoder.voice, "voice")
	if err != nil {
		return nil, err
	}
	format, err := requiredString(decoder.format, "format")
	if err != nil {
		return nil, err
	}
	return &chat.AudioControls{Voice: voice, Format: format}, nil
}

type streamOptionsDecoder struct{ includeUsage field }

func (d *streamOptionsDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	if bytes.Equal(key, []byte("include_usage")) {
		d.includeUsage = field{Raw: raw, Type: typ}
		return nil
	}
	return chat.Unsupported(string(key), "request field is not supported by llmux")
}

func parseStreamOptions(value field) (bool, error) {
	if !value.Present() {
		return false, nil
	}
	if value.Type != jsonparser.Object {
		return false, chat.Invalid("stream_options", "must be a JSON object")
	}
	var decoder streamOptionsDecoder
	if err := objectEach(value.Raw, "stream_options", decoder.field); err != nil {
		return false, err
	}
	includeUsage, err := decodeBool(decoder.includeUsage, "include_usage")
	if err != nil {
		return false, err
	}
	return includeUsage != nil && *includeUsage, nil
}

type responseFormatDecoder struct {
	typeName field
	schema   field
}

func (d *responseFormatDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	switch {
	case bytes.Equal(key, []byte("type")):
		d.typeName = field{Raw: raw, Type: typ}
	case bytes.Equal(key, []byte("json_schema")):
		d.schema = field{Raw: raw, Type: typ}
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

type schemaDecoder struct {
	name        field
	description field
	schema      field
	strict      field
}

func (d *schemaDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("name")):
		d.name = value
	case bytes.Equal(key, []byte("description")):
		d.description = value
	case bytes.Equal(key, []byte("schema")):
		d.schema = value
	case bytes.Equal(key, []byte("strict")):
		d.strict = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

func parseResponseFormat(value field) (chat.OutputFormat, error) {
	if value.Type != jsonparser.Object {
		return chat.OutputFormat{}, chat.Invalid("response_format", "must be a JSON object")
	}
	var decoder responseFormatDecoder
	if err := objectEach(value.Raw, "response_format", decoder.field); err != nil {
		return chat.OutputFormat{}, err
	}
	typeName, err := requiredString(decoder.typeName, "type")
	if err != nil {
		return chat.OutputFormat{}, err
	}
	switch typeName {
	case "text":
		return chat.OutputFormat{Kind: chat.FormatText}, nil
	case "json_object":
		return chat.OutputFormat{Kind: chat.FormatJSONObject, Name: "response", Schema: jsontext.Value(`{"type":"object"}`)}, nil
	case "json_schema":
		if !decoder.schema.Present() || decoder.schema.Type != jsonparser.Object {
			return chat.OutputFormat{}, chat.Invalid("response_format.json_schema", "must be a JSON object")
		}
		var schema schemaDecoder
		if err := objectEach(decoder.schema.Raw, "response_format.json_schema", schema.field); err != nil {
			return chat.OutputFormat{}, err
		}
		name, err := requiredString(schema.name, "name")
		if err != nil {
			return chat.OutputFormat{}, err
		}
		if !schema.schema.Present() {
			return chat.OutputFormat{}, chat.Invalid("response_format.json_schema.schema", "schema must be valid JSON")
		}
		copiedSchema := wire.Copy(schema.schema.Raw)
		if !copiedSchema.IsValid() {
			return chat.OutputFormat{}, chat.Invalid("response_format.json_schema.schema", "schema must be valid JSON")
		}
		description, _, err := decodeString(schema.description, "description")
		if err != nil {
			return chat.OutputFormat{}, err
		}
		strict, err := decodeBool(schema.strict, "strict")
		if err != nil {
			return chat.OutputFormat{}, err
		}
		return chat.OutputFormat{Kind: chat.FormatJSONSchema, Name: name, Description: description, Schema: copiedSchema, Strict: strict != nil && *strict}, nil
	default:
		return chat.OutputFormat{}, chat.Unsupported("response_format", "unsupported response format "+typeName)
	}
}

type contentDecoder struct {
	typeName   field
	text       field
	imageURL   field
	inputAudio field
	file       field
}

func (d *contentDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("type")):
		d.typeName = value
	case bytes.Equal(key, []byte("text")):
		d.text = value
	case bytes.Equal(key, []byte("image_url")):
		d.imageURL = value
	case bytes.Equal(key, []byte("input_audio")):
		d.inputAudio = value
	case bytes.Equal(key, []byte("file")):
		d.file = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

type imageURLDecoder struct {
	url    field
	detail field
}

func (d *imageURLDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("url")):
		d.url = value
	case bytes.Equal(key, []byte("detail")):
		d.detail = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

type inputAudioDecoder struct {
	data   field
	format field
}

func (d *inputAudioDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("data")):
		d.data = value
	case bytes.Equal(key, []byte("format")):
		d.format = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

func parseChatContent(value field) ([]chat.Part, error) {
	if !value.Present() || value.Type == jsonparser.Null {
		return nil, nil
	}
	if value.Type == jsonparser.String {
		text, err := wire.String(value.Raw, value.Type)
		if err != nil {
			return nil, chat.Invalid("messages.content", "must be a string or array")
		}
		return []chat.Part{chat.TextPart(text)}, nil
	}
	if value.Type != jsonparser.Array {
		return nil, chat.Invalid("messages.content", "must be a string or array")
	}
	parts := make([]chat.Part, 0, 4)
	var parseErr error
	_, err := jsonparser.ArrayEach(value.Raw, func(raw []byte, typ jsonparser.ValueType, _ int, callbackErr error) {
		if parseErr != nil {
			return
		}
		if callbackErr != nil {
			parseErr = callbackErr
			return
		}
		part, err := parseContentPart(raw, typ)
		if err != nil {
			parseErr = err
			return
		}
		parts = append(parts, part)
	})
	if parseErr != nil {
		return nil, parseErr
	}
	if err != nil {
		return nil, chat.Invalid("messages.content", "must be a JSON array")
	}
	return parts, nil
}

func parseContentPart(raw []byte, typ jsonparser.ValueType) (chat.Part, error) {
	if typ != jsonparser.Object {
		return chat.Part{}, chat.Invalid("messages.content", "must be a JSON object")
	}
	var decoder contentDecoder
	if err := objectEach(raw, "messages.content", decoder.field); err != nil {
		return chat.Part{}, err
	}
	typeName, err := requiredString(decoder.typeName, "type")
	if err != nil {
		return chat.Part{}, err
	}
	switch typeName {
	case "text":
		if decoder.imageURL.Present() || decoder.inputAudio.Present() || decoder.file.Present() {
			return chat.Part{}, chat.Unsupported("messages.content", "request field is not supported by llmux")
		}
		text, err := requiredString(decoder.text, "text")
		if err != nil {
			return chat.Part{}, err
		}
		return chat.TextPart(text), nil
	case "image_url":
		if decoder.text.Present() || decoder.inputAudio.Present() || decoder.file.Present() {
			return chat.Part{}, chat.Unsupported("messages.content", "request field is not supported by llmux")
		}
		if decoder.imageURL.Type != jsonparser.Object {
			return chat.Part{}, chat.Invalid("messages.content.image_url", "must be a JSON object")
		}
		var image imageURLDecoder
		if err := objectEach(decoder.imageURL.Raw, "messages.content.image_url", image.field); err != nil {
			return chat.Part{}, err
		}
		imageURL, err := requiredString(image.url, "url")
		if err != nil {
			return chat.Part{}, err
		}
		detail, _, err := decodeString(image.detail, "detail")
		if err != nil {
			return chat.Part{}, err
		}
		if err := wire.ValidateImageDetail(detail, "messages.content.image_url.detail"); err != nil {
			return chat.Part{}, err
		}
		media, err := parseMediaURL(imageURL, detail)
		if err != nil {
			return chat.Part{}, chat.Invalid("messages.content.image_url.url", err.Error())
		}
		return chat.Part{Type: chat.PartImage, Media: &media, Detail: detail}, nil
	case "input_audio":
		if decoder.text.Present() || decoder.imageURL.Present() || decoder.file.Present() {
			return chat.Part{}, chat.Unsupported("messages.content", "request field is not supported by llmux")
		}
		if decoder.inputAudio.Type != jsonparser.Object {
			return chat.Part{}, chat.Invalid("messages.content.input_audio", "must be a JSON object")
		}
		var audio inputAudioDecoder
		if err := objectEach(decoder.inputAudio.Raw, "messages.content.input_audio", audio.field); err != nil {
			return chat.Part{}, err
		}
		encoded, err := requiredString(audio.data, "data")
		if err != nil {
			return chat.Part{}, err
		}
		data, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return chat.Part{}, chat.Invalid("messages.content.input_audio.data", "must be valid base64")
		}
		format, err := requiredString(audio.format, "format")
		if err != nil {
			return chat.Part{}, err
		}
		if format != "wav" && format != "mp3" {
			return chat.Part{}, chat.Unsupported("messages.content.input_audio.format", "only wav and mp3 audio input are supported")
		}
		media := chat.InlineMedia(audioMIME(format), data)
		media.Format = format
		return chat.AudioPart(media), nil
	case "file":
		if decoder.text.Present() || decoder.imageURL.Present() || decoder.inputAudio.Present() {
			return chat.Part{}, chat.Unsupported("messages.content", "request field is not supported by llmux")
		}
		media, filename, err := parseFileMedia(decoder.file.Raw, "messages.content.file")
		if err != nil {
			return chat.Part{}, err
		}
		media.Filename = filename
		return chat.FilePart(media), nil
	default:
		return chat.Part{}, chat.Invalid("messages.content", "unsupported Chat Completions content type "+typeName)
	}
}

type fileDecoder struct {
	filename field
	data     field
	url      field
	id       field
}

func (d *fileDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("type")):
	case bytes.Equal(key, []byte("filename")):
		d.filename = value
	case bytes.Equal(key, []byte("file_data")):
		d.data = value
	case bytes.Equal(key, []byte("file_url")):
		d.url = value
	case bytes.Equal(key, []byte("file_id")):
		d.id = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

func parseFileMedia(raw []byte, param string) (chat.Media, string, error) {
	if len(raw) == 0 {
		return chat.Media{}, "", chat.Invalid(param, "must be a JSON object")
	}
	var decoder fileDecoder
	if err := objectEach(raw, param, decoder.field); err != nil {
		return chat.Media{}, "", err
	}
	filename, _, err := decodeString(decoder.filename, "filename")
	if err != nil {
		return chat.Media{}, "", err
	}
	sources := 0
	if decoder.data.Present() {
		sources++
	}
	if decoder.url.Present() {
		sources++
	}
	if decoder.id.Present() {
		sources++
	}
	if sources != 1 {
		return chat.Media{}, "", chat.Invalid(param, "file requires exactly one of file_data, file_url, or file_id")
	}
	switch {
	case decoder.data.Present():
		encoded, err := requiredString(decoder.data, "file_data")
		if err != nil {
			return chat.Media{}, "", err
		}
		media, err := wire.ParseFileData(encoded, param+".file_data")
		return media, filename, err
	case decoder.url.Present():
		value, err := requiredString(decoder.url, "file_url")
		if err != nil {
			return chat.Media{}, "", err
		}
		media, err := parseMediaURL(value, "")
		return media, filename, err
	default:
		value, err := requiredString(decoder.id, "file_id")
		if err != nil {
			return chat.Media{}, "", err
		}
		return chat.AssetMedia("application/octet-stream", value), filename, nil
	}
}

func parseMetadata(value field) (map[string]string, error) {
	if !value.Present() || value.Type == jsonparser.Null {
		return nil, nil
	}
	if value.Type != jsonparser.Object {
		return nil, fmtError("metadata", "must be an object of strings", wire.ErrType)
	}
	metadata := make(map[string]string)
	if err := jsonparser.ObjectEach(value.Raw, func(key, raw []byte, typ jsonparser.ValueType, _ int) error {
		decoded, err := wire.String(raw, typ)
		if err != nil {
			return fmtError("metadata", "must be an object of strings", err)
		}
		metadata[string(key)] = decoded
		return nil
	}); err != nil {
		if _, ok := err.(*chat.Error); ok {
			return nil, err
		}
		return nil, fmtError("metadata", "must be an object of strings", err)
	}
	return metadata, nil
}

func objectEach(raw []byte, param string, callback func([]byte, []byte, jsonparser.ValueType, int) error) error {
	if err := jsonparser.ObjectEach(raw, callback); err != nil {
		if _, ok := err.(*chat.Error); ok {
			return err
		}
		return chat.Invalid(param, "must be a JSON object")
	}
	return nil
}

func requiredString(value field, key string) (string, error) {
	decoded, ok, err := decodeString(value, key)
	if err != nil {
		return "", err
	}
	if !ok || strings.TrimSpace(decoded) == "" {
		return "", chat.Invalid(key, key+" is required")
	}
	return decoded, nil
}

func decodeString(value field, key string) (string, bool, error) {
	if !value.Present() {
		return "", false, nil
	}
	decoded, err := wire.String(value.Raw, value.Type)
	if err != nil {
		return "", true, chat.Invalid(key, "must be a string")
	}
	return decoded, true, nil
}

func decodeBool(value field, key string) (*bool, error) {
	if !value.Present() {
		return nil, nil
	}
	decoded, err := wire.Bool(value.Raw, value.Type)
	if err != nil {
		return nil, chat.Invalid(key, "must be a boolean")
	}
	return &decoded, nil
}

func decodeInt(value field, key string) (*int, error) {
	if !value.Present() {
		return nil, nil
	}
	decoded, err := wire.Int(value.Raw, value.Type)
	if err != nil {
		return nil, chat.Invalid(key, "must be an integer")
	}
	return &decoded, nil
}

func decodeFloat(value field, key string) (*float64, error) {
	if !value.Present() {
		return nil, nil
	}
	decoded, err := wire.Float(value.Raw, value.Type)
	if err != nil {
		return nil, chat.Invalid(key, "must be a number")
	}
	return &decoded, nil
}
