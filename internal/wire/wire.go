// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package wire

import (
	"cmp"
	"encoding/base64"
	"errors"
	"net/http"
	"net/url"
	"strings"

	"github.com/kelindar/llmux/chat"
)

// Error builds an invalid_request_error chat.APIError for param with cause attached.
func Error(param, message string, cause error) *chat.Error {
	return &chat.Error{
		Status:  http.StatusBadRequest,
		Type:    "invalid_request_error",
		Code:    "invalid_request",
		Param:   param,
		Message: message,
		Err:     cause,
	}
}

// ParseMediaURL parses an HTTP(S) URL or data URL into chat.Media.
func ParseMediaURL(value, detail string) (chat.Media, error) {
	switch {
	case value == "":
		return chat.Media{}, errors.New("media URL is empty")
	case strings.HasPrefix(value, "data:"):
		return ParseDataURL(value)
	}
	parsed, err := url.ParseRequestURI(value)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return chat.Media{}, errors.New("media URL must be an absolute HTTP or HTTPS URL")
	}
	return chat.Media{URL: value}, nil
}

// ParseDataURL decodes a base64 data URL into inline chat.Media.
func ParseDataURL(value string) (chat.Media, error) {
	meta, encoded, ok := strings.Cut(value, ",")
	if !ok || !strings.HasPrefix(meta, "data:") {
		return chat.Media{}, errors.New("invalid data URL")
	}
	meta = strings.TrimPrefix(meta, "data:")
	mime := ""
	semi := strings.IndexByte(meta, ';')
	switch {
	case semi >= 0:
		mime = meta[:semi]
		meta = meta[semi+1:]
	default:
		mime = meta
		meta = ""
	}
	if !strings.EqualFold(meta, "base64") {
		return chat.Media{}, errors.New("data URL must use base64 encoding")
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return chat.Media{}, errors.New("data URL contains invalid base64")
	}
	return chat.InlineMedia(mime, data), nil
}

// ParseFileData parses base64 or data URL file data into inline chat.Media.
func ParseFileData(value, param string) (chat.Media, error) {
	switch {
	case strings.HasPrefix(value, "data:"):
		media, err := ParseDataURL(value)
		if err != nil {
			return chat.Media{}, chat.Invalid(param, "must be a valid base64 data URL")
		}
		media.MIMEType = cmp.Or(media.MIMEType, "application/octet-stream")
		return media, nil
	default:
		data, err := base64.StdEncoding.DecodeString(value)
		if err != nil {
			return chat.Media{}, chat.Invalid(param, "must be valid base64")
		}
		return chat.InlineMedia("application/octet-stream", data), nil
	}
}

// ValidateImageDetail checks that detail is an allowed image detail value.
func ValidateImageDetail(detail, param string) error {
	if detail == "" || detail == "auto" || detail == "low" || detail == "high" {
		return nil
	}
	return chat.Unsupported(param, "supported image detail values are auto, low, and high")
}

// AudioMIME maps a short audio format name to a MIME type.
func AudioMIME(format string) string {
	switch strings.ToLower(format) {
	case "wav":
		return "audio/wav"
	case "mp3":
		return "audio/mpeg"
	case "ogg", "opus":
		return "audio/ogg"
	case "flac":
		return "audio/flac"
	case "m4a", "mp4":
		return "audio/mp4"
	default:
		return "audio/" + strings.ToLower(format)
	}
}

// CollectText concatenates text and reasoning-summary parts.
func CollectText(parts []chat.Part) string {
	var builder strings.Builder
	for _, part := range parts {
		if part.Type == chat.PartText || part.Type == chat.PartReasoningSummary {
			builder.WriteString(part.Text)
		}
	}
	return builder.String()
}

// OutputTextParts converts text parts into Responses output_text objects.
func OutputTextParts(parts []chat.Part) []map[string]any {
	out := make([]map[string]any, 0, len(parts))
	for _, part := range parts {
		if part.Type == chat.PartText {
			out = append(out, map[string]any{"type": "output_text", "text": part.Text, "annotations": []any{}})
		}
	}
	return out
}

// InputTextParts converts text parts into Responses input_text objects.
func InputTextParts(parts []chat.Part) []map[string]any {
	out := make([]map[string]any, 0, len(parts))
	for _, part := range parts {
		if part.Type == chat.PartText {
			out = append(out, map[string]any{"type": "input_text", "text": part.Text})
		}
	}
	return out
}

// MediaDataURL encodes inline media as a base64 data URL.
func MediaDataURL(media chat.Media) (string, error) {
	if len(media.Data) == 0 {
		return "", errors.New("media has no inline data")
	}
	if media.MIMEType == "" {
		return "", errors.New("inline media MIME type is required")
	}
	return "data:" + media.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(media.Data), nil
}
