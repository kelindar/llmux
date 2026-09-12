// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package responses

import (
	"bytes"
	"encoding/json/jsontext"
	"strings"

	"github.com/buger/jsonparser"
	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/wire"
)

type field = wire.Value

type requestDecoder struct {
	model            field
	input            field
	instructions     field
	stream           field
	streamOptions    field
	tools            field
	toolChoice       field
	parallelToolCall field
	temperature      field
	topP             field
	maxOutputTokens  field
	text             field
	reasoning        field
	store            field
	previous         field
	metadata         field
	background       field
	conversation     field
	include          field
	serviceTier      field
	truncation       field
	user             field
	prompt           field
	maxToolCalls     field
	extensions       map[string]jsontext.Value
}

// ParseRequest decodes a Responses API request directly from its body.
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
	case bytes.Equal(key, []byte("input")):
		d.input = value
	case bytes.Equal(key, []byte("instructions")):
		d.instructions = value
	case bytes.Equal(key, []byte("stream")):
		d.stream = value
	case bytes.Equal(key, []byte("stream_options")):
		d.streamOptions = value
	case bytes.Equal(key, []byte("tools")):
		d.tools = value
	case bytes.Equal(key, []byte("tool_choice")):
		d.toolChoice = value
	case bytes.Equal(key, []byte("parallel_tool_calls")):
		d.parallelToolCall = value
	case bytes.Equal(key, []byte("temperature")):
		d.temperature = value
	case bytes.Equal(key, []byte("top_p")):
		d.topP = value
	case bytes.Equal(key, []byte("max_output_tokens")):
		d.maxOutputTokens = value
	case bytes.Equal(key, []byte("text")):
		d.text = value
	case bytes.Equal(key, []byte("reasoning")):
		d.reasoning = value
	case bytes.Equal(key, []byte("store")):
		d.store = value
	case bytes.Equal(key, []byte("previous_response_id")):
		d.previous = value
	case bytes.Equal(key, []byte("metadata")):
		d.metadata = value
	case bytes.Equal(key, []byte("background")):
		d.background = value
	case bytes.Equal(key, []byte("conversation")):
		d.conversation = value
	case bytes.Equal(key, []byte("include")):
		d.include = value
	case bytes.Equal(key, []byte("service_tier")):
		d.serviceTier = value
	case bytes.Equal(key, []byte("truncation")):
		d.truncation = value
	case bytes.Equal(key, []byte("user")):
		d.user = value
	case bytes.Equal(key, []byte("prompt")):
		d.prompt = value
	case bytes.Equal(key, []byte("max_tool_calls")):
		d.maxToolCalls = value
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
	if !d.input.Present() {
		return parsedRequest{}, chat.Invalid("input", "input is required")
	}
	input, err := parseResponsesInput(d.input)
	if err != nil {
		return parsedRequest{}, err
	}
	controls := chat.Controls{Extensions: d.extensions}
	controls.MaxOutputTokens, err = decodeInt(d.maxOutputTokens, "max_output_tokens")
	if err != nil {
		return parsedRequest{}, err
	}
	if controls.MaxOutputTokens != nil && *controls.MaxOutputTokens < 1 {
		return parsedRequest{}, chat.Invalid("max_output_tokens", "max_output_tokens must be positive")
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
	if d.tools.Present() {
		controls.Tools, controls.ImageGeneration, err = parseResponsesTools(d.tools)
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
	controls.ParallelToolCall, err = decodeBool(d.parallelToolCall, "parallel_tool_calls")
	if err != nil {
		return parsedRequest{}, err
	}
	if d.streamOptions.Present() {
		if d.streamOptions.Type != jsonparser.Object {
			return parsedRequest{}, chat.Invalid("stream_options", "must be a JSON object")
		}
		if err := objectEach(d.streamOptions.Raw, "stream_options", func(key, _ []byte, _ jsonparser.ValueType, _ int) error {
			if bytes.Equal(key, []byte("include_obfuscation")) {
				return nil
			}
			return chat.Unsupported(string(key), "request field is not supported by llmux")
		}); err != nil {
			return parsedRequest{}, err
		}
	}
	output := chat.OutputSpec{Modalities: chat.ModalityText}
	if controls.ImageGeneration {
		output.Modalities |= chat.ModalityImage
	}
	if d.text.Present() {
		output.Format, err = parseResponsesText(d.text)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if d.reasoning.Present() {
		controls.Reasoning, err = parseResponsesReasoning(d.reasoning)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	store, err := decodeBool(d.store, "store")
	if err != nil {
		return parsedRequest{}, err
	}
	var previous *string
	if d.previous.Present() {
		value, err := requiredString(d.previous, "previous_response_id")
		if err != nil {
			return parsedRequest{}, err
		}
		previous = &value
	}
	metadata, err := parseMetadata(d.metadata)
	if err != nil {
		return parsedRequest{}, err
	}
	if d.background.Present() {
		return parsedRequest{}, chat.Unsupported("background", "background execution is not supported")
	}
	for _, unsupported := range []struct {
		value field
		name  string
	}{
		{d.conversation, "conversation"},
		{d.include, "include"},
		{d.serviceTier, "service_tier"},
		{d.truncation, "truncation"},
		{d.user, "user"},
		{d.prompt, "prompt"},
		{d.maxToolCalls, "max_tool_calls"},
	} {
		if unsupported.value.Present() {
			return parsedRequest{}, chat.Unsupported(unsupported.name, unsupported.name+" is not supported")
		}
	}
	stream, err := decodeBool(d.stream, "stream")
	if err != nil {
		return parsedRequest{}, err
	}
	instructions, _, err := decodeString(d.instructions, "instructions")
	if err != nil {
		return parsedRequest{}, err
	}
	return parsedRequest{
		Kind: protocolResponses,
		Request: chat.Request{
			Target: target, Instructions: instructions, Input: input, Controls: controls, Output: output,
		},
		Previous: previous,
		Store:    store,
		Metadata: metadata,
		Stream:   stream != nil && *stream,
	}, nil
}

type itemDecoder struct {
	typeName         field
	role             field
	content          field
	id               field
	status           field
	callID           field
	name             field
	arguments        field
	output           field
	summary          field
	encryptedContent field
}

func (d *itemDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("type")):
		d.typeName = value
	case bytes.Equal(key, []byte("role")):
		d.role = value
	case bytes.Equal(key, []byte("content")):
		d.content = value
	case bytes.Equal(key, []byte("id")):
		d.id = value
	case bytes.Equal(key, []byte("status")):
		d.status = value
	case bytes.Equal(key, []byte("call_id")):
		d.callID = value
	case bytes.Equal(key, []byte("name")):
		d.name = value
	case bytes.Equal(key, []byte("arguments")):
		d.arguments = value
	case bytes.Equal(key, []byte("output")):
		d.output = value
	case bytes.Equal(key, []byte("summary")):
		d.summary = value
	case bytes.Equal(key, []byte("encrypted_content")):
		d.encryptedContent = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

func parseResponsesInput(value field) ([]chat.Item, error) {
	if value.Type == jsonparser.String || value.Type == jsonparser.Null {
		text, err := wire.String(value.Raw, value.Type)
		if err != nil {
			return nil, chat.Invalid("input", "must be a string or array")
		}
		return []chat.Item{{Type: chat.ItemMessage, Role: chat.RoleUser, Content: []chat.Part{chat.TextPart(text)}}}, nil
	}
	if value.Type != jsonparser.Array {
		return nil, chat.Invalid("input", "must be a string or array")
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
		parsed, err := parseResponsesItem(raw, typ)
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
		return nil, chat.Invalid("input", "must be a JSON array")
	}
	if len(items) == 0 {
		return nil, chat.Invalid("input", "input must not be empty")
	}
	return items, nil
}

func parseResponsesItem(raw []byte, typ jsonparser.ValueType) ([]chat.Item, error) {
	if typ != jsonparser.Object {
		return nil, chat.Invalid("input", "must be a JSON object")
	}
	var decoder itemDecoder
	if err := objectEach(raw, "input", decoder.field); err != nil {
		return nil, err
	}
	typeName := "message"
	if decoder.typeName.Present() {
		decoded, _, err := decodeString(decoder.typeName, "type")
		if err != nil {
			return nil, err
		}
		typeName = decoded
	}
	switch typeName {
	case "message":
		if err := itemExtra(decoder, "message"); err != nil {
			return nil, err
		}
		roleValue, err := requiredString(decoder.role, "role")
		if err != nil {
			return nil, err
		}
		role := chat.Role(roleValue)
		if !validRole(role) {
			return nil, chat.Invalid("input.role", "unsupported message role "+roleValue)
		}
		if !decoder.content.Present() {
			return nil, chat.Invalid("input.content", "message content is required")
		}
		content, err := parseResponsesContent(decoder.content, "input.content")
		if err != nil {
			return nil, err
		}
		item := chat.Item{Type: chat.ItemMessage, Role: role, Content: content}
		item.ID, _, err = decodeString(decoder.id, "id")
		if err != nil {
			return nil, err
		}
		item.Status, err = parseItemStatus(decoder.status)
		if err != nil {
			return nil, err
		}
		return []chat.Item{item}, nil
	case "function_call":
		if err := itemExtra(decoder, "function_call"); err != nil {
			return nil, err
		}
		callID, err := requiredString(decoder.callID, "call_id")
		if err != nil {
			return nil, err
		}
		name, err := requiredString(decoder.name, "name")
		if err != nil {
			return nil, err
		}
		arguments, err := requiredString(decoder.arguments, "arguments")
		if err != nil {
			return nil, err
		}
		if !jsontext.Value(arguments).IsValid() {
			return nil, chat.Invalid("input.arguments", "function call arguments must be valid JSON")
		}
		item := chat.FunctionCallItem(callID, name, arguments)
		item.ID, _, err = decodeString(decoder.id, "id")
		if err != nil {
			return nil, err
		}
		if status, err := parseItemStatus(decoder.status); err != nil {
			return nil, err
		} else if status != "" {
			item.Status = status
		}
		return []chat.Item{item}, nil
	case "function_call_output":
		if err := itemExtra(decoder, "function_call_output"); err != nil {
			return nil, err
		}
		callID, err := requiredString(decoder.callID, "call_id")
		if err != nil {
			return nil, err
		}
		if !decoder.output.Present() {
			return nil, chat.Invalid("input.output", "function call output is required")
		}
		parts, err := parseResponsesOutput(decoder.output, "input.output")
		if err != nil {
			return nil, err
		}
		item := chat.Item{Type: chat.ItemFunctionCallOutput, CallID: callID, Output: parts}
		item.ID, _, err = decodeString(decoder.id, "id")
		if err != nil {
			return nil, err
		}
		item.Status, err = parseItemStatus(decoder.status)
		if err != nil {
			return nil, err
		}
		return []chat.Item{item}, nil
	case "reasoning":
		if err := itemExtra(decoder, "reasoning"); err != nil {
			return nil, err
		}
		itemID, _, err := decodeString(decoder.id, "id")
		if err != nil {
			return nil, err
		}
		item := chat.Item{Type: chat.ItemReasoning, ID: itemID, Status: chat.StatusCompleted}
		if status, err := parseItemStatus(decoder.status); err != nil {
			return nil, err
		} else if status != "" {
			item.Status = status
		}
		if decoder.summary.Present() {
			if decoder.summary.Type != jsonparser.Array {
				return nil, chat.Invalid("input.summary", "must be a JSON array")
			}
			var parseErr error
			_, err := jsonparser.ArrayEach(decoder.summary.Raw, func(raw []byte, typ jsonparser.ValueType, _ int, callbackErr error) {
				if parseErr != nil {
					return
				}
				if callbackErr != nil {
					parseErr = callbackErr
					return
				}
				if typ != jsonparser.Object {
					parseErr = chat.Invalid("input.summary", "must be a JSON object")
					return
				}
				var summary summaryDecoder
				if err := objectEach(raw, "input.summary", summary.field); err != nil {
					parseErr = err
					return
				}
				if summary.typeName.Present() {
					typeName, _, err := decodeString(summary.typeName, "input.summary.type")
					if err != nil {
						parseErr = err
						return
					}
					if typeName != "summary_text" {
						parseErr = chat.Unsupported("input.summary.type", "unsupported reasoning summary type "+typeName)
						return
					}
				}
				text, err := requiredString(summary.text, "text")
				if err != nil {
					parseErr = err
					return
				}
				item.Summary = append(item.Summary, chat.SummaryPart(text))
			})
			if parseErr != nil {
				return nil, parseErr
			}
			if err != nil {
				return nil, chat.Invalid("input.summary", "must be a JSON array")
			}
		}
		if decoder.encryptedContent.Present() {
			item.EncryptedContent = wire.Copy(decoder.encryptedContent.Raw)
		}
		return []chat.Item{item}, nil
	default:
		return nil, chat.Unsupported("input.type", "input item type "+typeName+" is not supported")
	}
}

func itemExtra(d itemDecoder, typeName string) error {
	if typeName == "message" {
		if d.callID.Present() || d.name.Present() || d.arguments.Present() || d.output.Present() || d.summary.Present() || d.encryptedContent.Present() {
			return chat.Unsupported("input", "request field is not supported by llmux")
		}
		return nil
	}
	if typeName == "function_call" {
		if d.role.Present() || d.content.Present() || d.output.Present() || d.summary.Present() || d.encryptedContent.Present() {
			return chat.Unsupported("input", "request field is not supported by llmux")
		}
		return nil
	}
	if typeName == "function_call_output" {
		if d.role.Present() || d.content.Present() || d.name.Present() || d.arguments.Present() || d.summary.Present() || d.encryptedContent.Present() {
			return chat.Unsupported("input", "request field is not supported by llmux")
		}
		return nil
	}
	if d.role.Present() || d.content.Present() || d.callID.Present() || d.name.Present() || d.arguments.Present() || d.output.Present() {
		return chat.Unsupported("input", "request field is not supported by llmux")
	}
	return nil
}

type summaryDecoder struct {
	typeName field
	text     field
}

func (d *summaryDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	switch {
	case bytes.Equal(key, []byte("type")):
		d.typeName = field{Raw: raw, Type: typ}
	case bytes.Equal(key, []byte("text")):
		d.text = field{Raw: raw, Type: typ}
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

func parseItemStatus(value field) (chat.Status, error) {
	status, ok, err := decodeString(value, "input.status")
	if err != nil {
		return "", err
	}
	if !ok || status == "" {
		return "", nil
	}
	switch chat.Status(status) {
	case chat.StatusInProgress, chat.StatusCompleted, chat.StatusIncomplete, chat.StatusFailed, chat.StatusCancelled:
		return chat.Status(status), nil
	default:
		return "", chat.Invalid("input.status", "unsupported item status "+status)
	}
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

type contentDecoder struct {
	typeName field
	text     field
	imageURL field
	fileID   field
	detail   field
	fileData field
	fileURL  field
	filename field
	unknown  string
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
	case bytes.Equal(key, []byte("file_id")):
		d.fileID = value
	case bytes.Equal(key, []byte("detail")):
		d.detail = value
	case bytes.Equal(key, []byte("file_data")):
		d.fileData = value
	case bytes.Equal(key, []byte("file_url")):
		d.fileURL = value
	case bytes.Equal(key, []byte("filename")):
		d.filename = value
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

func parseResponsesContent(value field, param string) ([]chat.Part, error) {
	if value.Type == jsonparser.String || value.Type == jsonparser.Null {
		text, err := wire.String(value.Raw, value.Type)
		if err != nil {
			return nil, chat.Invalid(param, "must be a string or array")
		}
		return []chat.Part{chat.TextPart(text)}, nil
	}
	if value.Type != jsonparser.Array {
		return nil, chat.Invalid(param, "must be a string or array")
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
			parseErr = chat.Invalid(param, "must be a JSON object")
			return
		}
		var decoder contentDecoder
		if err := objectEach(raw, param, decoder.field); err != nil {
			parseErr = err
			return
		}
		typeName, err := requiredString(decoder.typeName, "type")
		if err != nil {
			parseErr = err
			return
		}
		var part chat.Part
		switch typeName {
		case "input_text", "output_text":
			if err := decoder.rejectUnknown(); err != nil {
				parseErr = err
				return
			}
			if decoder.imageURL.Present() || decoder.fileID.Present() || decoder.detail.Present() || decoder.fileData.Present() || decoder.fileURL.Present() || decoder.filename.Present() {
				parseErr = chat.Unsupported(param, "request field is not supported by llmux")
				return
			}
			text, err := requiredString(decoder.text, "text")
			if err != nil {
				parseErr = err
				return
			}
			part = chat.TextPart(text)
		case "input_image":
			if err := decoder.rejectUnknown(); err != nil {
				parseErr = err
				return
			}
			if decoder.text.Present() || decoder.fileData.Present() || decoder.fileURL.Present() || decoder.filename.Present() {
				parseErr = chat.Unsupported(param, "request field is not supported by llmux")
				return
			}
			imageURL, ok, err := decodeString(decoder.imageURL, "image_url")
			if err != nil {
				parseErr = err
				return
			}
			fileID, fileIDOK, err := decodeString(decoder.fileID, "file_id")
			if err != nil {
				parseErr = err
				return
			}
			if ok && fileIDOK {
				parseErr = chat.Invalid(param, "input_image accepts only one of image_url or file_id")
				return
			}
			detail, _, err := decodeString(decoder.detail, "detail")
			if err != nil {
				parseErr = err
				return
			}
			if err := wire.ValidateImageDetail(detail, param+".detail"); err != nil {
				parseErr = err
				return
			}
			var media chat.Media
			switch {
			case ok:
				media, err = parseMediaURL(imageURL, detail)
			case fileIDOK:
				media = chat.AssetMedia("image/*", fileID)
			default:
				parseErr = chat.Invalid(param, "input_image requires image_url or file_id")
				return
			}
			if err != nil {
				parseErr = err
				return
			}
			part = chat.Part{Type: chat.PartImage, Media: &media, Detail: detail}
		case "input_file":
			if err := decoder.rejectUnknown(); err != nil {
				parseErr = err
				return
			}
			if decoder.text.Present() || decoder.imageURL.Present() || decoder.detail.Present() {
				parseErr = chat.Unsupported(param, "request field is not supported by llmux")
				return
			}
			media, filename, err := wire.ParseFileMediaValue(wire.Value{Raw: raw, Type: typ}, param)
			if err != nil {
				parseErr = err
				return
			}
			media.Filename = filename
			part = chat.FilePart(media)
		case "input_audio", "input_video":
			parseErr = chat.Unsupported(param, typeName+" is not supported")
			return
		default:
			parseErr = chat.Unsupported(param, "unsupported Responses content type "+typeName)
			return
		}
		parts = append(parts, part)
	})
	if parseErr != nil {
		return nil, parseErr
	}
	if err != nil {
		return nil, chat.Invalid(param, "must be a JSON array")
	}
	return parts, nil
}

func parseResponsesOutput(value field, param string) ([]chat.Part, error) {
	return parseResponsesContent(value, param)
}

type toolDecoder struct {
	typeName    field
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

func parseResponsesTools(value field) ([]chat.FunctionTool, bool, error) {
	if value.Type != jsonparser.Array {
		return nil, false, chat.Invalid("tools", "must be a JSON array")
	}
	tools := make([]chat.FunctionTool, 0, 4)
	imageGeneration := false
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
		if typeName == "image_generation" {
			if decoder.name.Present() || decoder.description.Present() || decoder.parameters.Present() || decoder.strict.Present() {
				parseErr = chat.Unsupported("tools", "image_generation does not accept options in this compatibility profile")
				return
			}
			if imageGeneration {
				parseErr = chat.Invalid("tools", "only one image_generation tool is supported")
				return
			}
			imageGeneration = true
			return
		}
		if typeName != "function" {
			parseErr = chat.Unsupported("tools", "only function and image_generation tools are supported")
			return
		}
		name, err := requiredString(decoder.name, "name")
		if err != nil {
			parseErr = err
			return
		}
		description, _, err := decodeString(decoder.description, "description")
		if err != nil {
			parseErr = err
			return
		}
		strict, err := decodeBool(decoder.strict, "strict")
		if err != nil {
			parseErr = err
			return
		}
		tool := chat.FunctionTool{Name: name, Description: description, Strict: strict}
		if decoder.parameters.Present() {
			tool.Parameters = wire.Copy(decoder.parameters.Raw)
		}
		tools = append(tools, tool)
	})
	if parseErr != nil {
		return nil, false, parseErr
	}
	if err != nil {
		return nil, false, chat.Invalid("tools", "must be a JSON array")
	}
	return tools, imageGeneration, nil
}

type textDecoder struct{ format field }

func (d *textDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	if bytes.Equal(key, []byte("format")) {
		d.format = field{Raw: raw, Type: typ}
		return nil
	}
	return chat.Unsupported(string(key), "request field is not supported by llmux")
}

type formatDecoder struct {
	typeName    field
	name        field
	description field
	schema      field
	strict      field
}

func (d *formatDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("type")):
		d.typeName = value
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

func parseResponsesText(value field) (chat.OutputFormat, error) {
	if value.Type != jsonparser.Object {
		return chat.OutputFormat{}, chat.Invalid("text", "must be a JSON object")
	}
	var decoder textDecoder
	if err := objectEach(value.Raw, "text", decoder.field); err != nil {
		return chat.OutputFormat{}, err
	}
	if !decoder.format.Present() {
		return chat.OutputFormat{Kind: chat.FormatText}, nil
	}
	if decoder.format.Type != jsonparser.Object {
		return chat.OutputFormat{}, chat.Invalid("text.format", "must be a JSON object")
	}
	var format formatDecoder
	if err := objectEach(decoder.format.Raw, "text.format", format.field); err != nil {
		return chat.OutputFormat{}, err
	}
	typeName, err := requiredString(format.typeName, "type")
	if err != nil {
		return chat.OutputFormat{}, err
	}
	switch typeName {
	case "text":
		return chat.OutputFormat{Kind: chat.FormatText}, nil
	case "json_object":
		return chat.OutputFormat{Kind: chat.FormatJSONObject, Name: "response", Schema: jsontext.Value(`{"type":"object"}`)}, nil
	case "json_schema":
		name, err := requiredString(format.name, "name")
		if err != nil {
			return chat.OutputFormat{}, err
		}
		if !format.schema.Present() {
			return chat.OutputFormat{}, chat.Invalid("text.format.schema", "schema must be valid JSON")
		}
		schema := wire.Copy(format.schema.Raw)
		if !schema.IsValid() {
			return chat.OutputFormat{}, chat.Invalid("text.format.schema", "schema must be valid JSON")
		}
		description, _, err := decodeString(format.description, "description")
		if err != nil {
			return chat.OutputFormat{}, err
		}
		strict, err := decodeBool(format.strict, "strict")
		if err != nil {
			return chat.OutputFormat{}, err
		}
		return chat.OutputFormat{Kind: chat.FormatJSONSchema, Name: name, Description: description, Schema: schema, Strict: strict != nil && *strict}, nil
	default:
		return chat.OutputFormat{}, chat.Unsupported("text.format", "unsupported text format "+typeName)
	}
}

type reasoningDecoder struct {
	effort  field
	summary field
}

func (d *reasoningDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	switch {
	case bytes.Equal(key, []byte("effort")):
		d.effort = field{Raw: raw, Type: typ}
	case bytes.Equal(key, []byte("summary")):
		d.summary = field{Raw: raw, Type: typ}
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

func parseResponsesReasoning(value field) (*chat.ReasoningControl, error) {
	if value.Type != jsonparser.Object {
		return nil, chat.Invalid("reasoning", "must be a JSON object")
	}
	var decoder reasoningDecoder
	if err := objectEach(value.Raw, "reasoning", decoder.field); err != nil {
		return nil, err
	}
	reasoning := &chat.ReasoningControl{}
	effort, ok, err := decodeString(decoder.effort, "effort")
	if err != nil {
		return nil, err
	}
	if ok {
		reasoning.Effort = effort
	}
	if decoder.summary.Present() && decoder.summary.Type != jsonparser.Null {
		summary, _, err := decodeString(decoder.summary, "summary")
		if err != nil {
			return nil, err
		}
		switch summary {
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
