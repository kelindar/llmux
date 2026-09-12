// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package wire

import (
	"bytes"
	"encoding/json/jsontext"
	"errors"
	"strings"

	"github.com/buger/jsonparser"
	"github.com/kelindar/llmux/chat"
)

// Value is a borrowed JSON value from the request body. Raw is valid until
// the current parse returns; parsers copy it when storing it in the result.
type Value struct {
	Raw  []byte
	Type jsonparser.ValueType
}

// Present reports whether the field existed in the containing object.
func (v Value) Present() bool { return v.Raw != nil }

var (
	ErrInvalidJSON = errors.New("request body must be valid JSON")
	ErrNotObject   = errors.New("request body must be a JSON object")
	ErrType        = errors.New("value has the wrong JSON type")
	ErrValue       = errors.New("value is not valid JSON")
)

// ValidateObject validates one complete JSON object, including unique names,
// UTF-8, and nested values, before jsonparser traverses its borrowed slices.
func ValidateObject(data []byte) error {
	if !jsontext.Value(data).IsValid() {
		return ErrInvalidJSON
	}
	data = bytes.TrimSpace(data)
	if len(data) == 0 || data[0] != '{' {
		return ErrNotObject
	}
	return nil
}

// String decodes a borrowed JSON string. JSON null has encoding/json's zero
// value semantics for optional scalar fields.
func String(raw []byte, typ jsonparser.ValueType) (string, error) {
	switch typ {
	case jsonparser.String:
		value, err := jsonparser.ParseString(raw)
		if err != nil {
			return "", ErrValue
		}
		return value, nil
	case jsonparser.Null:
		return "", nil
	default:
		return "", ErrType
	}
}

// Bool decodes a borrowed JSON boolean.
func Bool(raw []byte, typ jsonparser.ValueType) (bool, error) {
	switch typ {
	case jsonparser.Boolean:
		value, err := jsonparser.ParseBoolean(raw)
		if err != nil {
			return false, ErrValue
		}
		return value, nil
	case jsonparser.Null:
		return false, nil
	default:
		return false, ErrType
	}
}

// Int decodes a JSON integer using the platform int range.
func Int(raw []byte, typ jsonparser.ValueType) (int, error) {
	switch typ {
	case jsonparser.Number:
		value, err := jsonparser.ParseInt(raw)
		if err != nil || int64(int(value)) != value {
			return 0, ErrValue
		}
		return int(value), nil
	case jsonparser.Null:
		return 0, nil
	default:
		return 0, ErrType
	}
}

// Float decodes a JSON number.
func Float(raw []byte, typ jsonparser.ValueType) (float64, error) {
	switch typ {
	case jsonparser.Number:
		value, err := jsonparser.ParseFloat(raw)
		if err != nil {
			return 0, ErrValue
		}
		return value, nil
	case jsonparser.Null:
		return 0, nil
	default:
		return 0, ErrType
	}
}

// Copy owns a raw JSON value after the request body is released.
func Copy(raw []byte) jsontext.Value { return append(jsontext.Value(nil), raw...) }

// Strings decodes a JSON array of strings.
func Strings(raw []byte, typ jsonparser.ValueType) ([]string, error) {
	switch typ {
	case jsonparser.Null:
		return nil, nil
	case jsonparser.Array:
	default:
		return nil, ErrType
	}

	values := make([]string, 0, 4)
	var parseErr error
	_, err := jsonparser.ArrayEach(raw, func(value []byte, typ jsonparser.ValueType, _ int, _ error) {
		if parseErr != nil {
			return
		}
		decoded, err := String(value, typ)
		if err != nil {
			parseErr = err
			return
		}
		values = append(values, decoded)
	})
	if err != nil {
		return nil, err
	}
	if parseErr != nil {
		return nil, parseErr
	}
	return values, nil
}

// ParseFileMediaValue parses a borrowed file reference object without first
// materializing it as a map.
func ParseFileMediaValue(value Value, param string) (chat.Media, string, error) {
	if value.Type != jsonparser.Object {
		return chat.Media{}, "", chat.Invalid(param, "must be a JSON object")
	}

	var filename Value
	var data Value
	var url Value
	var id Value
	err := jsonparser.ObjectEach(value.Raw, func(key, raw []byte, typ jsonparser.ValueType, _ int) error {
		value := Value{Raw: raw, Type: typ}
		switch {
		case bytes.Equal(key, []byte("type")):
		case bytes.Equal(key, []byte("filename")):
			filename = value
		case bytes.Equal(key, []byte("file_data")):
			data = value
		case bytes.Equal(key, []byte("file_url")):
			url = value
		case bytes.Equal(key, []byte("file_id")):
			id = value
		default:
			return chat.Unsupported(string(key), "request field is not supported by llmux")
		}
		return nil
	})
	if err != nil {
		if _, ok := err.(*chat.Error); ok {
			return chat.Media{}, "", err
		}
		return chat.Media{}, "", chat.Invalid(param, "must be a JSON object")
	}

	filenameValue, err := String(filename.Raw, filename.Type)
	if filename.Present() && err != nil {
		return chat.Media{}, "", chat.Invalid("filename", "must be a string")
	}
	sources := 0
	if data.Present() {
		sources++
	}
	if url.Present() {
		sources++
	}
	if id.Present() {
		sources++
	}
	if sources != 1 {
		return chat.Media{}, "", chat.Invalid(param, "file requires exactly one of file_data, file_url, or file_id")
	}

	switch {
	case data.Present():
		encoded, err := requiredStringValue(data, "file_data")
		if err != nil {
			return chat.Media{}, "", err
		}
		media, err := ParseFileData(encoded, param+".file_data")
		return media, filenameValue, err
	case url.Present():
		value, err := requiredStringValue(url, "file_url")
		if err != nil {
			return chat.Media{}, "", err
		}
		media, err := ParseMediaURL(value, "")
		return media, filenameValue, err
	default:
		value, err := requiredStringValue(id, "file_id")
		if err != nil {
			return chat.Media{}, "", err
		}
		return chat.AssetMedia("application/octet-stream", value), filenameValue, nil
	}
}

func requiredStringValue(value Value, key string) (string, error) {
	if !value.Present() {
		return "", chat.Invalid(key, key+" is required")
	}
	decoded, err := String(value.Raw, value.Type)
	if err != nil {
		return "", chat.Invalid(key, "must be a string")
	}
	if strings.TrimSpace(decoded) == "" {
		return "", chat.Invalid(key, key+" is required")
	}
	return decoded, nil
}
