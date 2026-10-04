// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package agui

import (
	"bytes"
	"encoding/base64"
	"encoding/json/jsontext"

	"github.com/buger/jsonparser"
	"github.com/kelindar/llmux/chat"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/kelindar/llmux/internal/wire"
)

const maxActionBytes = 64 << 10

type field = wire.Value

type requestDecoder struct {
	threadID       field
	runID          field
	messages       field
	forwardedProps field
	protocol       field
	parentRunID    field
	state          field
	tools          field
	context        field
	resume         field
}

// ParseRequest decodes one AG-UI run request. Returned values own their data
// independently of body.
func ParseRequest(body []byte) (internalprotocol.ParsedRequest, error) {
	if err := wire.ValidateObject(body); err != nil {
		return internalprotocol.ParsedRequest{}, chat.Invalid("body", "request body must be a JSON object")
	}

	var decoder requestDecoder
	if err := jsonparser.ObjectEach(body, decoder.field); err != nil {
		return internalprotocol.ParsedRequest{}, err
	}
	return decoder.parse()
}

func (d *requestDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch string(key) {
	case "threadId":
		d.threadID = value
	case "runId":
		d.runID = value
	case "messages":
		d.messages = value
	case "forwardedProps":
		d.forwardedProps = value
	case "protocolVersion":
		d.protocol = value
	case "parentRunId":
		d.parentRunID = value
	case "state":
		d.state = value
	case "tools":
		d.tools = value
	case "context":
		d.context = value
	case "resume":
		d.resume = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

func (d requestDecoder) parse() (internalprotocol.ParsedRequest, error) {
	thread, err := requiredString(d.threadID, "threadId")
	if err != nil {
		return internalprotocol.ParsedRequest{}, err
	}
	runID, err := requiredString(d.runID, "runId")
	if err != nil {
		return internalprotocol.ParsedRequest{}, err
	}
	if d.protocol.Present() {
		version, err := requiredString(d.protocol, "protocolVersion")
		if err != nil {
			return internalprotocol.ParsedRequest{}, err
		}
		if version != "1.0" {
			return internalprotocol.ParsedRequest{}, chat.Unsupported("protocolVersion", "only AG-UI protocol version 1.0 is supported")
		}
	}
	if d.parentRunID.Present() {
		parent, err := requiredString(d.parentRunID, "parentRunId")
		if err != nil {
			return internalprotocol.ParsedRequest{}, err
		}
		if parent != "" {
			return internalprotocol.ParsedRequest{}, chat.Unsupported("parentRunId", "nested runs are not supported")
		}
	}
	for _, value := range []struct {
		field field
		name  string
	}{{d.tools, "tools"}, {d.context, "context"}} {
		if value.field.Present() && !emptyArray(value.field) {
			return internalprotocol.ParsedRequest{}, chat.Unsupported(value.name, value.name+" are not supported")
		}
	}
	if d.state.Present() && !emptyState(d.state) {
		return internalprotocol.ParsedRequest{}, chat.Unsupported("state", "initial state is not supported")
	}
	if d.resume.Present() && !emptyArray(d.resume) {
		return internalprotocol.ParsedRequest{}, chat.Unsupported("resume", "resume is not supported")
	}
	if !d.messages.Present() {
		return internalprotocol.ParsedRequest{}, chat.Invalid("messages", "messages is required")
	}

	target, previous, ui, action, err := parseForwarded(d.forwardedProps)
	if err != nil {
		return internalprotocol.ParsedRequest{}, err
	}
	items, err := parseMessages(d.messages)
	if err != nil {
		return internalprotocol.ParsedRequest{}, err
	}
	if action != nil {
		if len(items) != 0 {
			return internalprotocol.ParsedRequest{}, chat.Invalid("messages", "an action request must have an empty messages array")
		}
		if previous == nil || *previous != action.SourceResponseID {
			return internalprotocol.ParsedRequest{}, chat.Invalid("forwardedProps.llmux.action.sourceResponseId", "action sourceResponseId must match previousResponseId")
		}
		data := make(jsontext.Value, 0, len(action.raw)+len(`{"kind":"ui_action",}`)-1)
		data = append(data, `{"kind":"ui_action",`...)
		raw := bytes.TrimSpace(action.raw)
		data = append(data, raw[1:len(raw)-1]...)
		data = append(data, '}')
		items = []chat.Item{{Type: chat.ItemExtension, Data: data}}
	} else if len(items) != 1 {
		return internalprotocol.ParsedRequest{}, chat.Invalid("messages", "exactly one new user message is required")
	}

	controls := chat.Controls{}
	if ui != nil {
		controls.Extensions = map[string]jsontext.Value{"x-ui": wire.Copy(ui.raw)}
	}
	store := true
	return internalprotocol.ParsedRequest{
		Thread: thread,
		RunID:  runID,
		Kind:   internalprotocol.AGUI,
		Request: chat.Request{
			Target: target, Input: items, Controls: controls,
			Output: chat.OutputSpec{Modalities: chat.ModalityText},
		},
		Turn:     items,
		Previous: previous,
		Store:    &store,
		Retain:   true,
		Stream:   true,
	}, nil
}

type forwardedDecoder struct {
	llmux field
}

func (d *forwardedDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	if string(key) != "llmux" {
		return chat.Unsupported("forwardedProps."+string(key), "forwarded property is not supported")
	}
	d.llmux = field{Raw: raw, Type: typ}
	return nil
}

type llmuxDecoder struct {
	target   field
	previous field
	ui       field
	action   field
}

func (d *llmuxDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch string(key) {
	case "target":
		d.target = value
	case "previousResponseId":
		d.previous = value
	case "ui":
		d.ui = value
	case "action":
		d.action = value
	default:
		return chat.Unsupported("forwardedProps.llmux."+string(key), "forwarded property is not supported")
	}
	return nil
}

type uiSupport struct {
	Format    string `json:"format"`
	Version   string `json:"version"`
	CatalogID string `json:"catalogId"`
	raw       []byte
}

func parseForwarded(value field) (string, *string, *uiSupport, *uiAction, error) {
	if !value.Present() || value.Type != jsonparser.Object {
		return "", nil, nil, nil, chat.Invalid("forwardedProps", "forwardedProps.llmux is required")
	}
	var forwarded forwardedDecoder
	if err := jsonparser.ObjectEach(value.Raw, forwarded.field); err != nil {
		return "", nil, nil, nil, err
	}
	if !forwarded.llmux.Present() || forwarded.llmux.Type != jsonparser.Object {
		return "", nil, nil, nil, chat.Invalid("forwardedProps.llmux", "must be a JSON object")
	}
	var decoder llmuxDecoder
	if err := jsonparser.ObjectEach(forwarded.llmux.Raw, decoder.field); err != nil {
		return "", nil, nil, nil, err
	}
	target, err := requiredString(decoder.target, "forwardedProps.llmux.target")
	if err != nil {
		return "", nil, nil, nil, err
	}
	var previous *string
	if decoder.previous.Present() {
		value, err := requiredString(decoder.previous, "forwardedProps.llmux.previousResponseId")
		if err != nil {
			return "", nil, nil, nil, err
		}
		previous = &value
	}
	var ui *uiSupport
	if decoder.ui.Present() {
		ui, err = parseUISupport(decoder.ui)
		if err != nil {
			return "", nil, nil, nil, err
		}
	}
	var action *uiAction
	if decoder.action.Present() {
		if len(decoder.action.Raw) > maxActionBytes {
			return "", nil, nil, nil, &chat.Error{Status: 413, Type: "invalid_request_error", Code: "action_too_large", Param: "forwardedProps.llmux.action", Message: "action exceeds 64 KiB"}
		}
		action, err = parseAction(decoder.action)
		if err != nil {
			return "", nil, nil, nil, err
		}
	}
	return target, previous, ui, action, nil
}

func parseUISupport(value field) (*uiSupport, error) {
	var decoded uiSupport
	if value.Type != jsonparser.Object {
		return nil, chat.Invalid("forwardedProps.llmux.ui", "must be a JSON object")
	}
	err := jsonparser.ObjectEach(value.Raw, func(key, raw []byte, typ jsonparser.ValueType, _ int) error {
		var dest *string
		switch string(key) {
		case "format":
			dest = &decoded.Format
		case "version":
			dest = &decoded.Version
		case "catalogId":
			dest = &decoded.CatalogID
		default:
			return chat.Unsupported("forwardedProps.llmux.ui."+string(key), "UI property is not supported")
		}
		parsed, err := wire.String(raw, typ)
		if err != nil || parsed == "" {
			return chat.Invalid("forwardedProps.llmux.ui."+string(key), "must be a non-empty string")
		}
		*dest = parsed
		return nil
	})
	if err != nil {
		return nil, err
	}
	if decoded.Format != "a2ui" || decoded.Version != "0.9.1" || decoded.CatalogID == "" {
		return nil, chat.Unsupported("forwardedProps.llmux.ui", "only A2UI version 0.9.1 is supported")
	}
	decoded.raw = value.Raw
	return &decoded, nil
}

type uiAction struct {
	Name              string
	SurfaceID         string
	SourceComponentID string
	SourceResponseID  string
	SourceItemID      string
	context           bool
	raw               []byte
}

func parseAction(value field) (*uiAction, error) {
	if value.Type != jsonparser.Object {
		return nil, chat.Invalid("forwardedProps.llmux.action", "must be a JSON object")
	}
	var action uiAction
	err := jsonparser.ObjectEach(value.Raw, func(key, raw []byte, typ jsonparser.ValueType, _ int) error {
		var dest *string
		switch string(key) {
		case "name":
			dest = &action.Name
		case "surfaceId":
			dest = &action.SurfaceID
		case "sourceComponentId":
			dest = &action.SourceComponentID
		case "sourceResponseId":
			dest = &action.SourceResponseID
		case "sourceItemId":
			dest = &action.SourceItemID
		case "context":
			if typ != jsonparser.Object {
				return chat.Invalid("forwardedProps.llmux.action.context", "must be a JSON object")
			}
			action.context = true
			return nil
		default:
			return chat.Unsupported("forwardedProps.llmux.action."+string(key), "action property is not supported")
		}
		parsed, err := wire.String(raw, typ)
		if err != nil || parsed == "" {
			return chat.Invalid("forwardedProps.llmux.action."+string(key), "must be a non-empty string")
		}
		*dest = parsed
		return nil
	})
	if err != nil {
		return nil, err
	}
	if action.Name == "" || action.SurfaceID == "" || action.SourceComponentID == "" || action.SourceResponseID == "" || action.SourceItemID == "" || !action.context {
		return nil, chat.Invalid("forwardedProps.llmux.action", "all action fields are required")
	}
	action.raw = value.Raw
	return &action, nil
}

func parseMessages(value field) ([]chat.Item, error) {
	if value.Type != jsonparser.Array {
		return nil, chat.Invalid("messages", "must be a JSON array")
	}
	var items []chat.Item
	var parseErr error
	_, err := jsonparser.ArrayEach(value.Raw, func(raw []byte, typ jsonparser.ValueType, _ int, callbackErr error) {
		if parseErr != nil {
			return
		}
		if callbackErr != nil {
			parseErr = callbackErr
			return
		}
		if len(items) != 0 {
			parseErr = chat.Invalid("messages", "only one new user message is supported")
			return
		}
		item, err := parseMessage(raw, typ)
		if err != nil {
			parseErr = err
			return
		}
		items = append(items, item)
	})
	if parseErr != nil {
		return nil, parseErr
	}
	if err != nil {
		return nil, chat.Invalid("messages", "must be a JSON array")
	}
	return items, nil
}

func parseMessage(raw []byte, typ jsonparser.ValueType) (chat.Item, error) {
	if typ != jsonparser.Object {
		return chat.Item{}, chat.Invalid("messages", "message must be a JSON object")
	}
	var id, role, content field
	err := jsonparser.ObjectEach(raw, func(key, raw []byte, typ jsonparser.ValueType, _ int) error {
		value := field{Raw: raw, Type: typ}
		switch string(key) {
		case "id":
			id = value
		case "role":
			role = value
		case "content":
			content = value
		default:
			return chat.Unsupported("messages."+string(key), "message property is not supported")
		}
		return nil
	})
	if err != nil {
		return chat.Item{}, err
	}
	messageID, err := requiredString(id, "messages.id")
	if err != nil {
		return chat.Item{}, err
	}
	messageRole, err := requiredString(role, "messages.role")
	if err != nil {
		return chat.Item{}, err
	}
	if messageRole != string(chat.RoleUser) {
		return chat.Item{}, chat.Unsupported("messages.role", "only new user messages are supported")
	}
	parts, err := parseContent(content)
	if err != nil {
		return chat.Item{}, err
	}
	return chat.Item{Type: chat.ItemMessage, Role: chat.RoleUser, Content: parts, ID: messageID}, nil
}

func parseContent(value field) ([]chat.Part, error) {
	if value.Type == jsonparser.String {
		text, err := wire.String(value.Raw, value.Type)
		if err != nil {
			return nil, chat.Invalid("messages.content", "must be a string or typed content array")
		}
		return []chat.Part{chat.TextPart(text)}, nil
	}
	if value.Type != jsonparser.Array {
		return nil, chat.Invalid("messages.content", "must be a string or typed content array")
	}
	parts := make([]chat.Part, 0, 2)
	var parseErr error
	_, err := jsonparser.ArrayEach(value.Raw, func(raw []byte, typ jsonparser.ValueType, _ int, callbackErr error) {
		if parseErr != nil {
			return
		}
		if callbackErr != nil {
			parseErr = callbackErr
			return
		}
		part, err := parsePart(raw, typ)
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
	if len(parts) == 0 {
		return nil, chat.Invalid("messages.content", "must contain at least one content part")
	}
	return parts, nil
}

func parsePart(raw []byte, typ jsonparser.ValueType) (chat.Part, error) {
	if typ != jsonparser.Object {
		return chat.Part{}, chat.Invalid("messages.content", "content part must be an object")
	}
	var kind, text, source field
	err := jsonparser.ObjectEach(raw, func(key, raw []byte, typ jsonparser.ValueType, _ int) error {
		value := field{Raw: raw, Type: typ}
		switch string(key) {
		case "type":
			kind = value
		case "text":
			text = value
		case "source":
			source = value
		case "id", "metadata":
			return nil
		default:
			return chat.Unsupported("messages.content."+string(key), "content part property is not supported")
		}
		return nil
	})
	if err != nil {
		return chat.Part{}, err
	}
	typeName, err := requiredString(kind, "messages.content.type")
	if err != nil {
		return chat.Part{}, err
	}
	switch typeName {
	case "text":
		if !text.Present() || text.Type != jsonparser.String {
			return chat.Part{}, chat.Invalid("messages.content.text", "must be a string")
		}
		value, err := wire.String(text.Raw, text.Type)
		if err != nil {
			return chat.Part{}, chat.Invalid("messages.content.text", "must be a string")
		}
		if source.Present() {
			return chat.Part{}, chat.Unsupported("messages.content.source", "text parts do not accept a source")
		}
		return chat.TextPart(value), nil
	case "image":
		if text.Present() {
			return chat.Part{}, chat.Unsupported("messages.content.text", "image parts do not accept text")
		}
		return parseImage(source)
	default:
		return chat.Part{}, chat.Unsupported("messages.content.type", "only text and image content parts are supported")
	}
}

type sourceDecoder struct {
	typeName field
	value    field
	mimeType field
	provider field
}

func (d *sourceDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := field{Raw: raw, Type: typ}
	switch string(key) {
	case "type":
		d.typeName = value
	case "value":
		d.value = value
	case "mimeType":
		d.mimeType = value
	case "provider":
		d.provider = value
	default:
		return chat.Unsupported("messages.content.source."+string(key), "image source property is not supported")
	}
	return nil
}

func parseImage(value field) (chat.Part, error) {
	if !value.Present() || value.Type != jsonparser.Object {
		return chat.Part{}, chat.Invalid("messages.content.source", "image source must be a JSON object")
	}
	var decoder sourceDecoder
	if err := jsonparser.ObjectEach(value.Raw, decoder.field); err != nil {
		return chat.Part{}, err
	}
	typeName, err := requiredString(decoder.typeName, "messages.content.source.type")
	if err != nil {
		return chat.Part{}, err
	}
	mediaValue, err := requiredString(decoder.value, "messages.content.source.value")
	if err != nil {
		return chat.Part{}, err
	}
	mimeType := ""
	if decoder.mimeType.Present() {
		mimeType, err = requiredString(decoder.mimeType, "messages.content.source.mimeType")
		if err != nil {
			return chat.Part{}, err
		}
	}
	provider := ""
	if decoder.provider.Present() {
		provider, err = wire.String(decoder.provider.Raw, decoder.provider.Type)
		if err != nil {
			return chat.Part{}, chat.Invalid("messages.content.source.provider", "must be a string")
		}
	}
	var media chat.Media
	switch typeName {
	case "data":
		if provider != "" {
			return chat.Part{}, chat.Unsupported("messages.content.source.provider", "provider-specific image handles are not supported")
		}
		if mimeType == "" {
			return chat.Part{}, chat.Invalid("messages.content.source.mimeType", "data sources require a MIME type")
		}
		data, err := base64.StdEncoding.DecodeString(mediaValue)
		if err != nil {
			return chat.Part{}, chat.Invalid("messages.content.source.value", "data source value must be base64")
		}
		media = chat.Media{MIMEType: mimeType, Data: data}
	case "url":
		if provider != "" {
			return chat.Part{}, chat.Unsupported("messages.content.source.provider", "provider is only supported for file sources")
		}
		media = chat.RemoteMedia(mimeType, mediaValue)
	case "file":
		if provider != "" {
			return chat.Part{}, chat.Unsupported("messages.content.source.provider", "provider-specific image handles are not supported")
		}
		media = chat.AssetMedia(mimeType, mediaValue)
	default:
		return chat.Part{}, chat.Unsupported("messages.content.source.type", "only data, url, and file image sources are supported")
	}
	part := chat.ImagePart(media)
	if err := part.Validate(0); err != nil {
		return chat.Part{}, chat.Invalid("messages.content.source", err.Error())
	}
	return part, nil
}

func requiredString(value field, param string) (string, error) {
	if !value.Present() || value.Type != jsonparser.String {
		return "", chat.Invalid(param, "must be a non-empty string")
	}
	decoded, err := wire.String(value.Raw, value.Type)
	if err != nil || decoded == "" {
		return "", chat.Invalid(param, "must be a non-empty string")
	}
	return decoded, nil
}

func emptyArray(value field) bool {
	raw := bytes.TrimSpace(value.Raw)
	return value.Type == jsonparser.Array && len(raw) >= 2 && len(bytes.TrimSpace(raw[1:len(raw)-1])) == 0
}

func emptyState(value field) bool {
	if value.Type != jsonparser.Object && value.Type != jsonparser.Array {
		return false
	}
	raw := bytes.TrimSpace(value.Raw)
	return len(raw) >= 2 && len(bytes.TrimSpace(raw[1:len(raw)-1])) == 0
}
