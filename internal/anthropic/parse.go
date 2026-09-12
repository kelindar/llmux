// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package anthropic

import (
	"bytes"
	"cmp"
	"encoding/base64"
	"encoding/json/jsontext"
	"strings"

	"github.com/buger/jsonparser"
	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/wire"
)

type field = wire.Value

type requestDecoder struct {
	model         field
	maxTokens     field
	messages      field
	stream        field
	system        field
	temperature   field
	topP          field
	stopSequences field
	tools         field
	toolChoice    field
	metadata      field
	thinking      field
	serviceTier   field
	container     field
	outputConfig  field
	mcpServers    field
	extensions    map[string]jsontext.Value
}

// ParseRequest decodes an Anthropic Messages request directly from its body.
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
	case bytes.Equal(key, []byte("max_tokens")):
		d.maxTokens = value
	case bytes.Equal(key, []byte("messages")):
		d.messages = value
	case bytes.Equal(key, []byte("stream")):
		d.stream = value
	case bytes.Equal(key, []byte("system")):
		d.system = value
	case bytes.Equal(key, []byte("temperature")):
		d.temperature = value
	case bytes.Equal(key, []byte("top_p")):
		d.topP = value
	case bytes.Equal(key, []byte("stop_sequences")):
		d.stopSequences = value
	case bytes.Equal(key, []byte("tools")):
		d.tools = value
	case bytes.Equal(key, []byte("tool_choice")):
		d.toolChoice = value
	case bytes.Equal(key, []byte("metadata")):
		d.metadata = value
	case bytes.Equal(key, []byte("thinking")):
		d.thinking = value
	case bytes.Equal(key, []byte("service_tier")):
		d.serviceTier = value
	case bytes.Equal(key, []byte("container")):
		d.container = value
	case bytes.Equal(key, []byte("output_config")):
		d.outputConfig = value
	case bytes.Equal(key, []byte("mcp_servers")):
		d.mcpServers = value
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
	maxTokens, err := decodeInt(d.maxTokens, "max_tokens")
	if err != nil {
		return parsedRequest{}, err
	}
	if maxTokens == nil || *maxTokens < 1 {
		return parsedRequest{}, chat.Invalid("max_tokens", "max_tokens must be positive")
	}
	if !d.messages.Present() {
		return parsedRequest{}, chat.Invalid("messages", "messages is required")
	}
	input, err := parseMessages(d.messages)
	if err != nil {
		return parsedRequest{}, err
	}

	controls := chat.Controls{MaxOutputTokens: maxTokens, Extensions: d.extensions}
	controls.Temperature, err = decodeFloat(d.temperature, "temperature")
	if err != nil {
		return parsedRequest{}, err
	}
	if controls.Temperature != nil && (*controls.Temperature < 0 || *controls.Temperature > 1) {
		return parsedRequest{}, chat.Invalid("temperature", "temperature must be between 0 and 1")
	}
	controls.TopP, err = decodeFloat(d.topP, "top_p")
	if err != nil {
		return parsedRequest{}, err
	}
	if controls.TopP != nil && (*controls.TopP < 0 || *controls.TopP > 1) {
		return parsedRequest{}, chat.Invalid("top_p", "top_p must be between 0 and 1")
	}
	if d.stopSequences.Present() {
		controls.Stop, err = decodeStrings(d.stopSequences, "stop_sequences")
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if d.tools.Present() {
		controls.Tools, err = parseAnthropicTools(d.tools)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if d.toolChoice.Present() {
		controls.ToolChoice, err = parseAnthropicToolChoice(d.toolChoice)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	metadata, err := parseMetadata(d.metadata)
	if err != nil {
		return parsedRequest{}, err
	}
	if d.thinking.Present() {
		if d.thinking.Type != jsonparser.Object {
			return parsedRequest{}, chat.Invalid("thinking", "must be a JSON object")
		}
		var thinking thinkingDecoder
		if err := objectEach(d.thinking.Raw, "thinking", thinking.field); err != nil {
			return parsedRequest{}, err
		}
		if thinking.unknown != "" {
			return parsedRequest{}, chat.Unsupported(thinking.unknown, "request field is not supported by llmux")
		}
		typeName, err := requiredString(thinking.typeName, "type")
		if err != nil {
			return parsedRequest{}, err
		}
		if typeName != "disabled" {
			return parsedRequest{}, chat.Unsupported("thinking", "internal thinking is not exposed by the canonical agent contract")
		}
	}
	for _, unsupported := range []struct {
		value field
		name  string
	}{
		{d.serviceTier, "service_tier"},
		{d.container, "container"},
		{d.outputConfig, "output_config"},
		{d.mcpServers, "mcp_servers"},
	} {
		if unsupported.value.Present() {
			return parsedRequest{}, chat.Unsupported(unsupported.name, unsupported.name+" is not supported")
		}
	}

	instructions, err := parseAnthropicSystem(d.system)
	if err != nil {
		return parsedRequest{}, err
	}
	stream, err := decodeBool(d.stream, "stream")
	if err != nil {
		return parsedRequest{}, err
	}
	return parsedRequest{
		Kind: protocolAnthropic,
		Request: chat.Request{
			Target: target, Instructions: instructions, Input: input, Controls: controls,
			Output: chat.OutputSpec{Modalities: chat.ModalityText},
		},
		Metadata: metadata,
		Stream:   stream != nil && *stream,
	}, nil
}

func parseMessages(value field) ([]chat.Item, error) {
	if value.Type != jsonparser.Array {
		return nil, chat.Invalid("messages", "must be a JSON array")
	}
	items := make([]chat.Item, 0, 4)
	var parseErr error
	_, err := jsonparser.ArrayEach(value.Raw, func(raw []byte, typ jsonparser.ValueType, _ int, callbackErr error) {
		if parseErr != nil {
			return
		}
		if callbackErr != nil {
			parseErr = callbackErr
			return
		}
		parsed, err := parseAnthropicMessage(raw, typ)
		if err != nil {
			parseErr = err
			return
		}
		items = append(items, parsed...)
	})
	if parseErr != nil {
		return nil, parseErr
	}
	if err != nil {
		return nil, chat.Invalid("messages", "must be a JSON array")
	}
	if len(items) == 0 {
		return nil, chat.Invalid("messages", "messages must not be empty")
	}
	return items, nil
}

type messageDecoder struct {
	role    field
	content field
	unknown string
}

func (d *messageDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("role")):
		d.role = value
	case bytes.Equal(key, []byte("content")):
		d.content = value
	default:
		if d.unknown == "" {
			d.unknown = string(key)
		}
	}
	return nil
}

func parseAnthropicMessage(raw []byte, typ jsonparser.ValueType) ([]chat.Item, error) {
	if typ != jsonparser.Object {
		return nil, chat.Invalid("messages", "must be a JSON object")
	}
	var decoder messageDecoder
	if err := objectEach(raw, "messages", decoder.field); err != nil {
		return nil, err
	}
	if decoder.unknown != "" {
		return nil, chat.Unsupported(decoder.unknown, "request field is not supported by llmux")
	}
	roleValue, err := requiredString(decoder.role, "role")
	if err != nil {
		return nil, err
	}
	switch roleValue {
	case "user", "assistant":
	default:
		return nil, chat.Invalid("messages.role", "Anthropic messages only support user and assistant roles")
	}
	if !decoder.content.Present() {
		return nil, chat.Invalid("messages.content", "message content is required")
	}
	values, err := anthropicContent(decoder.content)
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

type contentDecoder struct {
	typeName  field
	text      field
	source    field
	id        field
	name      field
	input     field
	toolID    field
	content   field
	thinking  field
	signature field
	data      field
	unknown   string
}

func (d *contentDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("type")):
		d.typeName = value
	case bytes.Equal(key, []byte("text")):
		d.text = value
	case bytes.Equal(key, []byte("source")):
		d.source = value
	case bytes.Equal(key, []byte("id")):
		d.id = value
	case bytes.Equal(key, []byte("name")):
		d.name = value
	case bytes.Equal(key, []byte("input")):
		d.input = value
	case bytes.Equal(key, []byte("tool_use_id")):
		d.toolID = value
	case bytes.Equal(key, []byte("content")):
		d.content = value
	case bytes.Equal(key, []byte("thinking")):
		d.thinking = value
	case bytes.Equal(key, []byte("signature")):
		d.signature = value
	case bytes.Equal(key, []byte("data")):
		d.data = value
	default:
		if d.unknown == "" {
			d.unknown = string(key)
		}
	}
	return nil
}

func (d contentDecoder) rejectUnknown() error {
	if d.unknown != "" {
		return chat.Unsupported(d.unknown, "request field is not supported by llmux")
	}
	return nil
}

func anthropicContent(value field) ([]anthropicContentValue, error) {
	if value.Type == jsonparser.String || value.Type == jsonparser.Null {
		text, err := wire.String(value.Raw, value.Type)
		if err != nil {
			return nil, chat.Invalid("messages.content", "must be a string or array")
		}
		return []anthropicContentValue{{item: chat.MessageItem(chat.RoleUser, chat.TextPart(text))}}, nil
	}
	if value.Type != jsonparser.Array {
		return nil, chat.Invalid("messages.content", "must be a JSON array")
	}
	out := make([]anthropicContentValue, 0, 4)
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
			parseErr = chat.Invalid("messages.content", "must be a JSON object")
			return
		}
		var decoder contentDecoder
		if err := objectEach(raw, "messages.content", decoder.field); err != nil {
			parseErr = err
			return
		}
		typeName, err := requiredString(decoder.typeName, "type")
		if err != nil {
			parseErr = err
			return
		}
		switch typeName {
		case "text":
			if err := decoder.rejectUnknown(); err != nil {
				parseErr = err
				return
			}
			text, err := requiredString(decoder.text, "text")
			if err != nil {
				parseErr = err
				return
			}
			out = append(out, anthropicContentValue{item: chat.MessageItem(chat.RoleUser, chat.TextPart(text))})
		case "image":
			if err := decoder.rejectUnknown(); err != nil {
				parseErr = err
				return
			}
			media, err := parseAnthropicImage(decoder.source)
			if err != nil {
				parseErr = err
				return
			}
			out = append(out, anthropicContentValue{item: chat.MessageItem(chat.RoleUser, chat.ImagePart(media))})
		case "document":
			if err := decoder.rejectUnknown(); err != nil {
				parseErr = err
				return
			}
			media, err := parseAnthropicDocument(decoder.source)
			if err != nil {
				parseErr = err
				return
			}
			out = append(out, anthropicContentValue{item: chat.MessageItem(chat.RoleUser, chat.FilePart(media))})
		case "tool_use":
			if err := decoder.rejectUnknown(); err != nil {
				parseErr = err
				return
			}
			callID, err := requiredString(decoder.id, "id")
			if err != nil {
				parseErr = err
				return
			}
			name, err := requiredString(decoder.name, "name")
			if err != nil {
				parseErr = err
				return
			}
			if !decoder.input.Present() || !jsontext.Value(decoder.input.Raw).IsValid() {
				parseErr = chat.Invalid("messages.content.input", "tool input must be valid JSON")
				return
			}
			call := chat.FunctionCallItem(callID, name, string(decoder.input.Raw))
			call.ID = callID
			out = append(out, anthropicContentValue{item: call})
		case "tool_result":
			if err := decoder.rejectUnknown(); err != nil {
				parseErr = err
				return
			}
			callID, err := requiredString(decoder.toolID, "tool_use_id")
			if err != nil {
				parseErr = err
				return
			}
			parts := []chat.Part(nil)
			if decoder.content.Present() {
				parts, err = anthropicContentParts(decoder.content)
				if err != nil {
					parseErr = err
					return
				}
			}
			if len(parts) == 0 {
				parts = []chat.Part{chat.TextPart("")}
			}
			out = append(out, anthropicContentValue{item: chat.FunctionCallOutputItem(callID, parts...)})
		case "thinking":
			if err := decoder.rejectUnknown(); err != nil {
				parseErr = err
				return
			}
			out = append(out, anthropicContentValue{item: chat.Item{Type: chat.ItemReasoning, Data: wire.Copy(raw), Status: chat.StatusCompleted}})
		case "redacted_thinking":
			if err := decoder.rejectUnknown(); err != nil {
				parseErr = err
				return
			}
			out = append(out, anthropicContentValue{item: chat.Item{Type: chat.ItemReasoning, Data: wire.Copy(raw), Status: chat.StatusCompleted}})
		default:
			parseErr = chat.Unsupported("messages.content.type", "unsupported Anthropic content type "+typeName)
		}
	})
	if parseErr != nil {
		return nil, parseErr
	}
	if err != nil {
		return nil, chat.Invalid("messages.content", "must be a JSON array")
	}
	return out, nil
}

func anthropicContentParts(value field) ([]chat.Part, error) {
	if value.Type == jsonparser.String || value.Type == jsonparser.Null {
		text, err := wire.String(value.Raw, value.Type)
		if err != nil {
			return nil, chat.Invalid("tool_result.content", "must be a string or array")
		}
		return []chat.Part{chat.TextPart(text)}, nil
	}
	if value.Type != jsonparser.Array {
		return nil, chat.Invalid("tool_result.content", "must be a JSON array")
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
		if typ != jsonparser.Object {
			parseErr = chat.Invalid("tool_result.content", "must be a JSON object")
			return
		}
		var decoder contentDecoder
		if err := objectEach(raw, "tool_result.content", decoder.field); err != nil {
			parseErr = err
			return
		}
		typeName, err := requiredString(decoder.typeName, "type")
		if err != nil {
			parseErr = err
			return
		}
		switch typeName {
		case "text":
			if err := decoder.rejectUnknown(); err != nil {
				parseErr = err
				return
			}
			text, err := requiredString(decoder.text, "text")
			if err != nil {
				parseErr = err
				return
			}
			parts = append(parts, chat.TextPart(text))
		case "image":
			if err := decoder.rejectUnknown(); err != nil {
				parseErr = err
				return
			}
			media, err := parseAnthropicImage(decoder.source)
			if err != nil {
				parseErr = err
				return
			}
			parts = append(parts, chat.ImagePart(media))
		default:
			parseErr = chat.Unsupported("tool_result.content", "unsupported tool result type "+typeName)
		}
	})
	if parseErr != nil {
		return nil, parseErr
	}
	if err != nil {
		return nil, chat.Invalid("tool_result.content", "must be a JSON array")
	}
	return parts, nil
}

type sourceDecoder struct {
	typeName field
	media    field
	data     field
	url      field
	unknown  string
}

func (d *sourceDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("type")):
		d.typeName = value
	case bytes.Equal(key, []byte("media_type")):
		d.media = value
	case bytes.Equal(key, []byte("data")):
		d.data = value
	case bytes.Equal(key, []byte("url")):
		d.url = value
	default:
		if d.unknown == "" {
			d.unknown = string(key)
		}
	}
	return nil
}

func parseAnthropicImage(value field) (chat.Media, error) {
	if value.Type != jsonparser.Object {
		return chat.Media{}, chat.Invalid("messages.content.source", "must be a JSON object")
	}
	var source sourceDecoder
	if err := objectEach(value.Raw, "messages.content.source", source.field); err != nil {
		return chat.Media{}, err
	}
	typeName, err := requiredString(source.typeName, "type")
	if err != nil {
		return chat.Media{}, err
	}
	switch typeName {
	case "base64":
		if source.unknown != "" {
			return chat.Media{}, chat.Unsupported(source.unknown, "request field is not supported by llmux")
		}
		mime, err := requiredString(source.media, "media_type")
		if err != nil {
			return chat.Media{}, err
		}
		encoded, err := requiredString(source.data, "data")
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
		if source.unknown != "" {
			return chat.Media{}, chat.Unsupported(source.unknown, "request field is not supported by llmux")
		}
		value, err := requiredString(source.url, "url")
		if err != nil {
			return chat.Media{}, err
		}
		return chat.RemoteMedia("image/*", value), nil
	default:
		return chat.Media{}, chat.Unsupported("messages.content.source.type", "unsupported Anthropic image source "+typeName)
	}
}

func parseAnthropicDocument(value field) (chat.Media, error) {
	if value.Type != jsonparser.Object {
		return chat.Media{}, chat.Invalid("messages.content.source", "must be a JSON object")
	}
	var source sourceDecoder
	if err := objectEach(value.Raw, "messages.content.source", source.field); err != nil {
		return chat.Media{}, err
	}
	typeName, err := requiredString(source.typeName, "type")
	if err != nil {
		return chat.Media{}, err
	}
	switch typeName {
	case "base64":
		if source.unknown != "" {
			return chat.Media{}, chat.Unsupported(source.unknown, "request field is not supported by llmux")
		}
		mime, _, err := decodeString(source.media, "media_type")
		if err != nil {
			return chat.Media{}, err
		}
		mime = cmp.Or(mime, "application/octet-stream")
		encoded, err := requiredString(source.data, "data")
		if err != nil {
			return chat.Media{}, err
		}
		data, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return chat.Media{}, fmtError("messages.content.source.data", "must be valid base64", err)
		}
		return chat.InlineMedia(mime, data), nil
	case "url":
		if source.unknown != "" {
			return chat.Media{}, chat.Unsupported(source.unknown, "request field is not supported by llmux")
		}
		value, err := requiredString(source.url, "url")
		if err != nil {
			return chat.Media{}, err
		}
		return chat.RemoteMedia("application/octet-stream", value), nil
	default:
		return chat.Media{}, chat.Unsupported("messages.content.source.type", "unsupported Anthropic document source "+typeName)
	}
}

type thinkingDecoder struct {
	typeName field
	unknown  string
}

func (d *thinkingDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	if bytes.Equal(key, []byte("type")) {
		d.typeName = field{Raw: raw, Type: typ}
		return nil
	}
	if d.unknown == "" {
		d.unknown = string(key)
	}
	return nil
}

type systemDecoder struct {
	typeName field
	text     field
	unknown  string
}

func (d *systemDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("type")):
		d.typeName = value
	case bytes.Equal(key, []byte("text")):
		d.text = value
	default:
		if d.unknown == "" {
			d.unknown = string(key)
		}
	}
	return nil
}

func parseAnthropicSystem(value field) (string, error) {
	if !value.Present() {
		return "", nil
	}
	if value.Type == jsonparser.String || value.Type == jsonparser.Null {
		return wire.String(value.Raw, value.Type)
	}
	if value.Type != jsonparser.Array {
		return "", chat.Invalid("system", "must be a JSON array")
	}
	var builder strings.Builder
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
			parseErr = chat.Invalid("system", "must be a JSON object")
			return
		}
		var decoder systemDecoder
		if err := objectEach(raw, "system", decoder.field); err != nil {
			parseErr = err
			return
		}
		typeName, err := requiredString(decoder.typeName, "type")
		if err != nil {
			parseErr = err
			return
		}
		if typeName != "text" {
			parseErr = chat.Unsupported("system", "only text system blocks are supported")
			return
		}
		if decoder.unknown != "" {
			parseErr = chat.Unsupported(decoder.unknown, "request field is not supported by llmux")
			return
		}
		text, err := requiredString(decoder.text, "text")
		if err != nil {
			parseErr = err
			return
		}
		builder.WriteString(text)
	})
	if parseErr != nil {
		return "", parseErr
	}
	if err != nil {
		return "", chat.Invalid("system", "must be a JSON array")
	}
	return builder.String(), nil
}

type toolDecoder struct {
	name        field
	description field
	inputSchema field
	unknown     string
}

func (d *toolDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("name")):
		d.name = value
	case bytes.Equal(key, []byte("description")):
		d.description = value
	case bytes.Equal(key, []byte("input_schema")):
		d.inputSchema = value
	default:
		if d.unknown == "" {
			d.unknown = string(key)
		}
	}
	return nil
}

func parseAnthropicTools(value field) ([]chat.FunctionTool, error) {
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
		if decoder.unknown != "" {
			parseErr = chat.Unsupported(decoder.unknown, "request field is not supported by llmux")
			return
		}
		name, err := requiredString(decoder.name, "name")
		if err != nil {
			parseErr = err
			return
		}
		inputSchema := decoder.inputSchema
		if !inputSchema.Present() || !jsontext.Value(inputSchema.Raw).IsValid() {
			parseErr = chat.Invalid("tools.input_schema", "input_schema must be valid JSON")
			return
		}
		description, _, err := decodeString(decoder.description, "description")
		if err != nil {
			parseErr = err
			return
		}
		tools = append(tools, chat.FunctionTool{
			Name: name, Description: description, Parameters: wire.Copy(inputSchema.Raw),
		})
	})
	if parseErr != nil {
		return nil, parseErr
	}
	if err != nil {
		return nil, chat.Invalid("tools", "must be a JSON array")
	}
	return tools, nil
}

type toolChoiceDecoder struct {
	typeName field
	name     field
	unknown  string
}

func (d *toolChoiceDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("type")):
		d.typeName = value
	case bytes.Equal(key, []byte("name")):
		d.name = value
	default:
		if d.unknown == "" {
			d.unknown = string(key)
		}
	}
	return nil
}

func parseAnthropicToolChoice(value field) (*chat.ToolChoice, error) {
	if value.Type != jsonparser.Object {
		return nil, chat.Invalid("tool_choice", "must be a JSON object")
	}
	var decoder toolChoiceDecoder
	if err := objectEach(value.Raw, "tool_choice", decoder.field); err != nil {
		return nil, err
	}
	if decoder.unknown != "" {
		return nil, chat.Unsupported(decoder.unknown, "request field is not supported by llmux")
	}
	typeName, err := requiredString(decoder.typeName, "type")
	if err != nil {
		return nil, err
	}
	switch typeName {
	case "auto", "none", "any":
		if decoder.name.Present() {
			return nil, chat.Invalid("tool_choice.name", "tool_choice name is only valid for type tool")
		}
		return &chat.ToolChoice{Mode: typeName}, nil
	case "tool":
		name, err := requiredString(decoder.name, "name")
		if err != nil {
			return nil, err
		}
		return &chat.ToolChoice{Mode: "function", Name: name}, nil
	default:
		return nil, chat.Unsupported("tool_choice.type", "unsupported Anthropic tool choice "+typeName)
	}
}

func parseMetadata(value field) (map[string]string, error) {
	if !value.Present() || value.Type == jsonparser.Null {
		return nil, nil
	}
	if value.Type != jsonparser.Object {
		return nil, fmtError("metadata", "metadata must be an object of strings", wire.ErrType)
	}
	metadata := make(map[string]string)
	err := jsonparser.ObjectEach(value.Raw, func(key, raw []byte, typ jsonparser.ValueType, _ int) error {
		decoded, err := wire.String(raw, typ)
		if err != nil {
			return fmtError("metadata", "metadata must be an object of strings", err)
		}
		metadata[string(key)] = decoded
		return nil
	})
	if err != nil {
		if _, ok := err.(*chat.Error); ok {
			return nil, err
		}
		return nil, fmtError("metadata", "metadata must be an object of strings", err)
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

func decodeStrings(value field, key string) ([]string, error) {
	values, err := wire.Strings(value.Raw, value.Type)
	if err != nil {
		return nil, chat.Invalid(key, "must be an array of strings")
	}
	return values, nil
}
