// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package agui

import (
	"encoding/json/jsontext"

	"github.com/buger/jsonparser"
	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/wire"
)

const maxUIBytes = 256 << 10

type uiOutput struct {
	ID        string
	Format    string
	Version   string
	CatalogID string
	Payload   jsontext.Value
}

func parseUIItem(item chat.Item) (uiOutput, error) {
	if item.Type != chat.ItemExtension || len(item.Data) > maxUIBytes {
		return uiOutput{}, chat.Unsupported("output", "unsupported AG-UI output item")
	}
	if err := wire.ValidateObject(item.Data); err != nil {
		return uiOutput{}, chat.Invalid("output", "UI item data must be a JSON object")
	}
	var value uiOutput
	var kind string
	err := jsonparser.ObjectEach(item.Data, func(key, raw []byte, typ jsonparser.ValueType, _ int) error {
		switch string(key) {
		case "kind":
			kind, _ = wire.String(raw, typ)
		case "format":
			parsed, err := requiredString(field{Raw: raw, Type: typ}, "output.format")
			if err != nil {
				return err
			}
			value.Format = parsed
		case "version":
			parsed, err := requiredString(field{Raw: raw, Type: typ}, "output.version")
			if err != nil {
				return err
			}
			value.Version = parsed
		case "catalogId":
			parsed, err := requiredString(field{Raw: raw, Type: typ}, "output.catalogId")
			if err != nil {
				return err
			}
			value.CatalogID = parsed
		case "payload":
			if typ != jsonparser.Array {
				return chat.Invalid("output.payload", "must be a JSON array")
			}
			value.Payload = jsontext.Value(raw)
		default:
			return chat.Unsupported("output."+string(key), "UI item property is not supported")
		}
		return nil
	})
	if err != nil {
		return uiOutput{}, err
	}
	if kind != "ui" || value.Format != "a2ui" || value.Version != "0.9.1" || value.CatalogID == "" || len(value.Payload) == 0 {
		return uiOutput{}, chat.Unsupported("output", "only A2UI version 0.9.1 UI items are supported")
	}
	value.ID = item.ID
	if err := validatePayload(value.Payload, value.CatalogID); err != nil {
		return uiOutput{}, err
	}
	return value, nil
}

func validatePayload(payload jsontext.Value, catalogID string) error {
	var parseErr error
	count := 0
	surfaceID := ""
	created, components, dataModel := false, false, false
	_, err := jsonparser.ArrayEach(payload, func(raw []byte, typ jsonparser.ValueType, _ int, callbackErr error) {
		if parseErr != nil {
			return
		}
		if callbackErr != nil {
			parseErr = callbackErr
			return
		}
		message, err := validateEnvelope(raw, typ)
		if err != nil {
			parseErr = err
			return
		}
		if count == 0 {
			if message.kind != "createSurface" || message.catalogID != catalogID {
				parseErr = chat.Invalid("output.payload", "A2UI payload must begin with a matching createSurface")
				return
			}
			created, surfaceID = true, message.surfaceID
		} else {
			if message.surfaceID != surfaceID {
				parseErr = chat.Invalid("output.payload.surfaceId", "A2UI messages must use one surfaceId")
				return
			}
			switch message.kind {
			case "updateComponents":
				components = true
			case "updateDataModel":
				dataModel = true
			default:
				parseErr = chat.Unsupported("output.payload", "only one surface with components and data is supported")
				return
			}
		}
		count++
	})
	if parseErr != nil {
		return parseErr
	}
	if err != nil {
		return chat.Invalid("output.payload", "must be a JSON array")
	}
	if !created || !components || !dataModel {
		return chat.Invalid("output.payload", "A2UI payload requires a surface, components, and data model")
	}
	return nil
}

type envelope struct {
	kind      string
	surfaceID string
	catalogID string
}

func validateEnvelope(raw []byte, typ jsonparser.ValueType) (envelope, error) {
	if typ != jsonparser.Object {
		return envelope{}, chat.Invalid("output.payload", "A2UI envelope must be a JSON object")
	}
	var version field
	var message field
	messageType := ""
	count := 0
	err := jsonparser.ObjectEach(raw, func(key, raw []byte, typ jsonparser.ValueType, _ int) error {
		count++
		value := field{Raw: raw, Type: typ}
		if string(key) == "version" {
			version = value
			return nil
		}
		switch string(key) {
		case "createSurface", "updateComponents", "updateDataModel", "deleteSurface":
			if messageType != "" {
				return chat.Invalid("output.payload", "A2UI envelope must have exactly one message")
			}
			messageType, message = string(key), value
		default:
			return chat.Unsupported("output.payload."+string(key), "A2UI envelope property is not supported")
		}
		return nil
	})
	if err != nil {
		return envelope{}, err
	}
	protocolVersion, err := requiredString(version, "output.payload.version")
	if err != nil {
		return envelope{}, err
	}
	if protocolVersion != "v0.9.1" || messageType == "" || message.Type != jsonparser.Object || count != 2 {
		return envelope{}, chat.Invalid("output.payload", "A2UI envelope requires version v0.9.1 and one message")
	}
	var surface, catalog, components, sendDataModel, modelPath, modelValue field
	err = jsonparser.ObjectEach(message.Raw, func(key, raw []byte, typ jsonparser.ValueType, _ int) error {
		switch string(key) {
		case "surfaceId":
			surface = field{Raw: raw, Type: typ}
		case "catalogId":
			catalog = field{Raw: raw, Type: typ}
		case "components":
			components = field{Raw: raw, Type: typ}
		case "sendDataModel":
			sendDataModel = field{Raw: raw, Type: typ}
		case "path":
			modelPath = field{Raw: raw, Type: typ}
		case "value":
			modelValue = field{Raw: raw, Type: typ}
		}
		return nil
	})
	if err != nil {
		return envelope{}, chat.Invalid("output.payload", "A2UI message must be a JSON object")
	}
	surfaceID, err := requiredString(surface, "output.payload.surfaceId")
	if err != nil {
		return envelope{}, err
	}
	var catalogID string
	switch messageType {
	case "createSurface":
		catalogID, err = requiredString(catalog, "output.payload.catalogId")
		if err != nil {
			return envelope{}, err
		}
		if sendDataModel.Present() {
			if sendDataModel.Type != jsonparser.Boolean {
				return envelope{}, chat.Invalid("output.payload.sendDataModel", "must be a boolean")
			}
			enabled, _ := wire.Bool(sendDataModel.Raw, sendDataModel.Type)
			if enabled {
				return envelope{}, chat.Unsupported("output.payload.sendDataModel", "client data synchronization is not supported")
			}
		}
	case "updateComponents":
		if components.Type != jsonparser.Array || emptyArray(components) {
			return envelope{}, chat.Invalid("output.payload.components", "must be a non-empty JSON array")
		}
	case "updateDataModel":
		if modelPath.Present() && modelPath.Type != jsonparser.String || !modelValue.Present() {
			return envelope{}, chat.Invalid("output.payload.updateDataModel", "requires a value and an optional string path")
		}
	case "deleteSurface":
	default:
		return envelope{}, chat.Unsupported("output.payload", "unsupported A2UI message")
	}
	return envelope{kind: messageType, surfaceID: surfaceID, catalogID: catalogID}, nil
}

// MatchesSupport checks that a UI output item matches the request's declared
// A2UI format, version, and catalog.
func MatchesSupport(item chat.Item, support jsontext.Value) error {
	output, err := parseUIItem(item)
	if err != nil {
		return err
	}
	return matchesOutput(output, support)
}

func matchesOutput(output uiOutput, support jsontext.Value) error {
	declared, err := parseUISupport(field{Raw: support, Type: jsonparser.Object})
	if err != nil {
		return err
	}
	if output.Format != declared.Format || output.Version != declared.Version || output.CatalogID != declared.CatalogID {
		return chat.Unsupported("output", "UI item does not match the requested A2UI catalog")
	}
	return nil
}
