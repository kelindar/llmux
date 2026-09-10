package wire

import (
	"encoding/base64"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"net/http"
	"net/url"
	"strings"

	"github.com/kelindar/llmux/contract"
)

// DecodeObject decodes exactly one JSON object. encoding/json/v2 rejects
// duplicate names and trailing top-level values by default.
func DecodeObject(data []byte) (map[string]jsontext.Value, error) {
	var object map[string]jsontext.Value
	if err := json.Unmarshal(data, &object); err != nil {
		return nil, err
	}
	if object == nil {
		return nil, errors.New("request body must be a JSON object")
	}
	return object, nil
}

// DecodeString reads an optional string field from object.
func DecodeString(object map[string]jsontext.Value, key string) (string, bool, error) {
	raw, ok := object[key]
	if !ok {
		return "", false, nil
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", true, contract.Invalid(key, "must be a string")
	}
	return value, true, nil
}

// DecodeBool reads an optional boolean field from object.
func DecodeBool(object map[string]jsontext.Value, key string) (*bool, error) {
	raw, ok := object[key]
	if !ok {
		return nil, nil
	}
	var value bool
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, contract.Invalid(key, "must be a boolean")
	}
	return &value, nil
}

// DecodeInt reads an optional integer field from object.
func DecodeInt(object map[string]jsontext.Value, key string) (*int, error) {
	raw, ok := object[key]
	if !ok {
		return nil, nil
	}
	var value int
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, contract.Invalid(key, "must be an integer")
	}
	return &value, nil
}

// DecodeFloat reads an optional number field from object.
func DecodeFloat(object map[string]jsontext.Value, key string) (*float64, error) {
	raw, ok := object[key]
	if !ok {
		return nil, nil
	}
	var value float64
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, contract.Invalid(key, "must be a number")
	}
	return &value, nil
}

// DecodeStringSlice reads an optional string array field from object.
func DecodeStringSlice(object map[string]jsontext.Value, key string) ([]string, error) {
	raw, ok := object[key]
	if !ok {
		return nil, nil
	}
	var values []string
	if err := json.Unmarshal(raw, &values); err != nil {
		return nil, contract.Invalid(key, "must be an array of strings")
	}
	return values, nil
}

// Error builds an invalid_request_error contract.APIError for param with cause attached.
func Error(param, message string, cause error) *contract.APIError {
	return &contract.APIError{
		Status:  http.StatusBadRequest,
		Type:    "invalid_request_error",
		Code:    "invalid_request",
		Param:   param,
		Message: message,
		Err:     cause,
	}
}

// RejectUnknown rejects object keys that are not allowed or namespaced extensions.
func RejectUnknown(object map[string]jsontext.Value, allowed map[string]bool) error {
	for key := range object {
		if allowed[key] || strings.HasPrefix(key, "x-") || strings.Contains(key, ":") {
			continue
		}
		return contract.Unsupported(key, "request field is not supported by llmux")
	}
	return nil
}

// RejectUnknownStrict rejects any object key not present in allowed.
func RejectUnknownStrict(object map[string]jsontext.Value, allowed map[string]bool) error {
	for key := range object {
		if !allowed[key] {
			return contract.Unsupported(key, "request field is not supported by llmux")
		}
	}
	return nil
}

// NamespacedExtensions collects x- prefixed and namespaced extension fields.
func NamespacedExtensions(object map[string]jsontext.Value, allowed map[string]bool) map[string]jsontext.Value {
	var extensions map[string]jsontext.Value
	for key, raw := range object {
		if allowed[key] || (!strings.HasPrefix(key, "x-") && !strings.Contains(key, ":")) {
			continue
		}
		if extensions == nil {
			extensions = make(map[string]jsontext.Value)
		}
		extensions[key] = append(jsontext.Value(nil), raw...)
	}
	return extensions
}

// RawObject decodes raw as a JSON object, reporting param on failure.
func RawObject(raw jsontext.Value, param string) (map[string]jsontext.Value, error) {
	value, err := DecodeObject(raw)
	if err != nil {
		return nil, contract.Invalid(param, "must be a JSON object")
	}
	return value, nil
}

// RawArray decodes raw as a JSON array, reporting param on failure.
func RawArray(raw jsontext.Value, param string) ([]jsontext.Value, error) {
	var values []jsontext.Value
	if err := json.Unmarshal(raw, &values); err != nil {
		return nil, contract.Invalid(param, "must be a JSON array")
	}
	return values, nil
}

// RequireString reads a required non-empty string field from object.
func RequireString(object map[string]jsontext.Value, key string) (string, error) {
	value, ok, err := DecodeString(object, key)
	if err != nil {
		return "", err
	}
	if !ok || strings.TrimSpace(value) == "" {
		return "", contract.Invalid(key, key+" is required")
	}
	return value, nil
}

// ParseMediaURL parses an HTTP(S) URL or data URL into contract.Media.
func ParseMediaURL(value, detail string) (contract.Media, error) {
	switch {
	case value == "":
		return contract.Media{}, errors.New("media URL is empty")
	case strings.HasPrefix(value, "data:"):
		return ParseDataURL(value)
	}
	parsed, err := url.ParseRequestURI(value)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return contract.Media{}, errors.New("media URL must be an absolute HTTP or HTTPS URL")
	}
	return contract.Media{URL: value}, nil
}

// ParseDataURL decodes a base64 data URL into inline contract.Media.
func ParseDataURL(value string) (contract.Media, error) {
	meta, encoded, ok := strings.Cut(value, ",")
	switch {
	case !ok, !strings.HasPrefix(meta, "data:"):
		return contract.Media{}, errors.New("invalid data URL")
	}
	meta = strings.TrimPrefix(meta, "data:")
	mime := ""
	if semi := strings.IndexByte(meta, ';'); semi >= 0 {
		mime = meta[:semi]
		meta = meta[semi+1:]
	} else {
		mime = meta
		meta = ""
	}
	if !strings.EqualFold(meta, "base64") {
		return contract.Media{}, errors.New("data URL must use base64 encoding")
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return contract.Media{}, errors.New("data URL contains invalid base64")
	}
	return contract.InlineMedia(mime, data), nil
}

// ParseStringOrContent accepts either a plain string or structured content via parser.
func ParseStringOrContent(raw jsontext.Value, param string, parser func(jsontext.Value) ([]contract.Part, error)) ([]contract.Part, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []contract.Part{contract.TextPart(text)}, nil
	}
	return parser(raw)
}

// ParseChatContent parses OpenAI Chat Completions message content into parts.
func ParseChatContent(raw jsontext.Value) ([]contract.Part, error) {
	if string(raw) == "null" || len(raw) == 0 {
		return nil, nil
	}
	return ParseStringOrContent(raw, "messages.content", func(value jsontext.Value) ([]contract.Part, error) {
		parts, err := RawArray(value, "messages.content")
		if err != nil {
			return nil, err
		}
		out := make([]contract.Part, 0, len(parts))
		for _, rawPart := range parts {
			object, err := RawObject(rawPart, "messages.content")
			if err != nil {
				return nil, err
			}
			typeName, err := RequireString(object, "type")
			if err != nil {
				return nil, err
			}
			switch typeName {
			case "text":
				if err := RejectUnknownStrict(object, map[string]bool{"type": true, "text": true}); err != nil {
					return nil, err
				}
				text, err := RequireString(object, "text")
				if err != nil {
					return nil, err
				}
				out = append(out, contract.TextPart(text))
			case "image_url":
				if err := RejectUnknownStrict(object, map[string]bool{"type": true, "image_url": true}); err != nil {
					return nil, err
				}
				imageObject, err := RawObject(object["image_url"], "messages.content.image_url")
				if err != nil {
					return nil, err
				}
				if err := RejectUnknownStrict(imageObject, map[string]bool{"url": true, "detail": true}); err != nil {
					return nil, err
				}
				imageURL, err := RequireString(imageObject, "url")
				if err != nil {
					return nil, err
				}
				detail := ""
				if value, ok, err := DecodeString(imageObject, "detail"); err != nil {
					return nil, err
				} else if ok {
					detail = value
				}
				if err := ValidateImageDetail(detail, "messages.content.image_url.detail"); err != nil {
					return nil, err
				}
				media, err := ParseMediaURL(imageURL, detail)
				if err != nil {
					return nil, contract.Invalid("messages.content.image_url.url", err.Error())
				}
				out = append(out, contract.Part{Type: contract.PartImage, Media: &media, Detail: detail})
			case "input_audio":
				if err := RejectUnknownStrict(object, map[string]bool{"type": true, "input_audio": true}); err != nil {
					return nil, err
				}
				audioObject, err := RawObject(object["input_audio"], "messages.content.input_audio")
				if err != nil {
					return nil, err
				}
				if err := RejectUnknownStrict(audioObject, map[string]bool{"data": true, "format": true}); err != nil {
					return nil, err
				}
				encoded, err := RequireString(audioObject, "data")
				if err != nil {
					return nil, err
				}
				data, err := base64.StdEncoding.DecodeString(encoded)
				if err != nil {
					return nil, contract.Invalid("messages.content.input_audio.data", "must be valid base64")
				}
				format, err := RequireString(audioObject, "format")
				if err != nil {
					return nil, err
				}
				if format != "wav" && format != "mp3" {
					return nil, contract.Unsupported("messages.content.input_audio.format", "only wav and mp3 audio input are supported")
				}
				media := contract.InlineMedia(AudioMIME(format), data)
				media.Format = format
				out = append(out, contract.AudioPart(media))
			case "file":
				if err := RejectUnknownStrict(object, map[string]bool{"type": true, "file": true}); err != nil {
					return nil, err
				}
				fileObject, err := RawObject(object["file"], "messages.content.file")
				if err != nil {
					return nil, err
				}
				media, filename, err := ParseFileMedia(fileObject, "messages.content.file")
				if err != nil {
					return nil, err
				}
				media.Filename = filename
				out = append(out, contract.FilePart(media))
			default:
				return nil, contract.Invalid("messages.content", "unsupported Chat Completions content type "+typeName)
			}
		}
		return out, nil
	})
}

// ParseFileMedia parses a file reference object into contract.Media and an optional filename.
func ParseFileMedia(object map[string]jsontext.Value, param string) (contract.Media, string, error) {
	if err := RejectUnknownStrict(object, map[string]bool{"type": true, "filename": true, "file_data": true, "file_url": true, "file_id": true}); err != nil {
		return contract.Media{}, "", err
	}
	filename := ""
	if value, ok, err := DecodeString(object, "filename"); err != nil {
		return contract.Media{}, "", err
	} else if ok {
		filename = value
	}
	sources := 0
	for _, key := range []string{"file_data", "file_url", "file_id"} {
		if _, ok := object[key]; ok {
			sources++
		}
	}
	if sources != 1 {
		return contract.Media{}, "", contract.Invalid(param, "file requires exactly one of file_data, file_url, or file_id")
	}
	if raw, ok := object["file_data"]; ok {
		encoded, err := RequireString(map[string]jsontext.Value{"file_data": raw}, "file_data")
		if err != nil {
			return contract.Media{}, "", err
		}
		media, err := ParseFileData(encoded, param+".file_data")
		if err != nil {
			return contract.Media{}, "", err
		}
		return media, filename, nil
	}
	if raw, ok := object["file_url"]; ok {
		value, err := RequireString(map[string]jsontext.Value{"file_url": raw}, "file_url")
		if err != nil {
			return contract.Media{}, "", err
		}
		media, err := ParseMediaURL(value, "")
		return media, filename, err
	}
	if raw, ok := object["file_id"]; ok {
		value, err := RequireString(map[string]jsontext.Value{"file_id": raw}, "file_id")
		if err != nil {
			return contract.Media{}, "", err
		}
		return contract.AssetMedia("application/octet-stream", value), filename, nil
	}
	return contract.Media{}, "", contract.Invalid(param, "file requires file_data, file_url, or file_id")
}

// ParseFileData parses base64 or data URL file data into inline contract.Media.
func ParseFileData(value, param string) (contract.Media, error) {
	if strings.HasPrefix(value, "data:") {
		media, err := ParseDataURL(value)
		if err != nil {
			return contract.Media{}, contract.Invalid(param, "must be a valid base64 data URL")
		}
		if media.MIMEType == "" {
			media.MIMEType = "application/octet-stream"
		}
		return media, nil
	}
	data, err := base64.StdEncoding.DecodeString(value)
	if err != nil {
		return contract.Media{}, contract.Invalid(param, "must be valid base64")
	}
	return contract.InlineMedia("application/octet-stream", data), nil
}

// ValidateImageDetail checks that detail is an allowed image detail value.
func ValidateImageDetail(detail, param string) error {
	if detail == "" || detail == "auto" || detail == "low" || detail == "high" {
		return nil
	}
	return contract.Unsupported(param, "supported image detail values are auto, low, and high")
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
func CollectText(parts []contract.Part) string {
	var builder strings.Builder
	for _, part := range parts {
		if part.Type == contract.PartText || part.Type == contract.PartReasoningSummary {
			builder.WriteString(part.Text)
		}
	}
	return builder.String()
}

// OutputTextParts converts text parts into Responses output_text objects.
func OutputTextParts(parts []contract.Part) []map[string]any {
	out := make([]map[string]any, 0, len(parts))
	for _, part := range parts {
		if part.Type == contract.PartText {
			out = append(out, map[string]any{"type": "output_text", "text": part.Text, "annotations": []any{}})
		}
	}
	return out
}

// InputTextParts converts text parts into Responses input_text objects.
func InputTextParts(parts []contract.Part) []map[string]any {
	out := make([]map[string]any, 0, len(parts))
	for _, part := range parts {
		if part.Type == contract.PartText {
			out = append(out, map[string]any{"type": "input_text", "text": part.Text})
		}
	}
	return out
}

// MediaDataURL encodes inline media as a base64 data URL.
func MediaDataURL(media contract.Media) (string, error) {
	if len(media.Data) == 0 {
		return "", errors.New("media has no inline data")
	}
	if media.MIMEType == "" {
		return "", errors.New("inline media MIME type is required")
	}
	return "data:" + media.MIMEType + ";base64," + base64.StdEncoding.EncodeToString(media.Data), nil
}
