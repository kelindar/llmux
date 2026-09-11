// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package wire

import (
	"encoding/base64"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func decodeObject(t *testing.T, raw string) map[string]jsontext.Value {
	t.Helper()
	object, err := DecodeObject([]byte(raw))
	require.NoError(t, err)
	return object
}

func TestDecodeObject(t *testing.T) {
	cases := map[string]struct {
		raw     string
		wantErr string
	}{
		"ok":         {raw: `{"a":"b"}`},
		"notJSON":    {raw: `{`, wantErr: "json"},
		"notObject":  {raw: `[]`, wantErr: "json"},
		"nullObject": {raw: `null`, wantErr: "JSON object"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			object, err := DecodeObject([]byte(tc.raw))
			if tc.wantErr == "" {
				require.NoError(t, err)
				assert.NotNil(t, object)
				return
			}
			require.Error(t, err)
		})
	}
}

func TestDecodeString(t *testing.T) {
	object := decodeObject(t, `{"name":"alice","age":1,"bad":true}`)

	value, ok, err := DecodeString(object, "missing")
	require.NoError(t, err)
	assert.False(t, ok)
	assert.Empty(t, value)

	value, ok, err = DecodeString(object, "name")
	require.NoError(t, err)
	assert.True(t, ok)
	assert.Equal(t, "alice", value)

	_, _, err = DecodeString(object, "age")
	require.Error(t, err)
	var apiErr *chat.Error
	require.True(t, errors.As(err, &apiErr))
	assert.Equal(t, "age", apiErr.Param)
}

func TestDecodeBool(t *testing.T) {
	object := decodeObject(t, `{"flag":true,"bad":"x"}`)

	flag, err := DecodeBool(object, "missing")
	require.NoError(t, err)
	assert.Nil(t, flag)

	flag, err = DecodeBool(object, "flag")
	require.NoError(t, err)
	require.NotNil(t, flag)
	assert.True(t, *flag)

	_, err = DecodeBool(object, "bad")
	require.Error(t, err)
}

func TestDecodeInt(t *testing.T) {
	object := decodeObject(t, `{"count":3,"bad":1.5}`)

	count, err := DecodeInt(object, "missing")
	require.NoError(t, err)
	assert.Nil(t, count)

	count, err = DecodeInt(object, "count")
	require.NoError(t, err)
	require.NotNil(t, count)
	assert.Equal(t, 3, *count)

	_, err = DecodeInt(object, "bad")
	require.Error(t, err)
}

func TestDecodeFloat(t *testing.T) {
	object := decodeObject(t, `{"rate":1.5,"bad":"x"}`)

	rate, err := DecodeFloat(object, "missing")
	require.NoError(t, err)
	assert.Nil(t, rate)

	rate, err = DecodeFloat(object, "rate")
	require.NoError(t, err)
	require.NotNil(t, rate)
	assert.InDelta(t, 1.5, *rate, 0.001)

	_, err = DecodeFloat(object, "bad")
	require.Error(t, err)
}

func TestDecodeStringSlice(t *testing.T) {
	object := decodeObject(t, `{"tags":["a","b"],"bad":"x"}`)

	tags, err := DecodeStringSlice(object, "missing")
	require.NoError(t, err)
	assert.Nil(t, tags)

	tags, err = DecodeStringSlice(object, "tags")
	require.NoError(t, err)
	assert.Equal(t, []string{"a", "b"}, tags)

	_, err = DecodeStringSlice(object, "bad")
	require.Error(t, err)
}

func TestRejectUnknown(t *testing.T) {
	object := decodeObject(t, `{"model":"m","x-custom":1,"vendor:ext":2}`)

	err := RejectUnknown(object, map[string]bool{"model": true})
	require.NoError(t, err)

	object = decodeObject(t, `{"unknown":1}`)
	err = RejectUnknown(object, map[string]bool{"model": true})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not supported")
}

func TestRejectUnknownStrict(t *testing.T) {
	object := decodeObject(t, `{"model":"m"}`)
	require.NoError(t, RejectUnknownStrict(object, map[string]bool{"model": true}))

	object = decodeObject(t, `{"model":"m","extra":1}`)
	err := RejectUnknownStrict(object, map[string]bool{"model": true})
	require.Error(t, err)
}

func TestNamespacedExtensions(t *testing.T) {
	object := decodeObject(t, `{"model":"m","x-a":1,"vendor:b":2,"plain":3}`)
	ext := NamespacedExtensions(object, map[string]bool{"model": true})
	assert.Len(t, ext, 2)
	assert.NotNil(t, ext["x-a"])
	assert.NotNil(t, ext["vendor:b"])
	assert.Nil(t, ext["plain"])
}

func TestRawObjectArray(t *testing.T) {
	obj, err := RawObject(jsontext.Value(`{"k":"v"}`), "param")
	require.NoError(t, err)
	assert.NotNil(t, obj["k"])

	_, err = RawObject(jsontext.Value(`[]`), "param")
	require.Error(t, err)

	arr, err := RawArray(jsontext.Value(`[1,2]`), "param")
	require.NoError(t, err)
	assert.Len(t, arr, 2)

	_, err = RawArray(jsontext.Value(`{}`), "param")
	require.Error(t, err)
}

func TestRequireString(t *testing.T) {
	object := decodeObject(t, `{"name":"bob","empty":"  "}`)

	value, err := RequireString(object, "name")
	require.NoError(t, err)
	assert.Equal(t, "bob", value)

	_, err = RequireString(object, "missing")
	require.Error(t, err)

	_, err = RequireString(object, "empty")
	require.Error(t, err)
}

func TestErrorHelper(t *testing.T) {
	cause := errors.New("cause")
	err := Error("field", "message", cause)
	require.NotNil(t, err)
	assert.Equal(t, 400, err.Status)
	assert.Equal(t, "message", err.Message)
	assert.Equal(t, cause, err.Unwrap())
}

func TestParseDataURL(t *testing.T) {
	data := []byte("hello")
	encoded := base64.StdEncoding.EncodeToString(data)
	url := "data:text/plain;base64," + encoded

	media, err := ParseDataURL(url)
	require.NoError(t, err)
	assert.Equal(t, "text/plain", media.MIMEType)
	assert.Equal(t, data, media.Data)

	cases := map[string]struct {
		value   string
		wantErr string
	}{
		"invalidPrefix": {value: "http://x", wantErr: "invalid data URL"},
		"noComma":       {value: "data:text/plain", wantErr: "invalid data URL"},
		"notBase64":     {value: "data:text/plain,plain", wantErr: "base64 encoding"},
		"badBase64":     {value: "data:text/plain;base64,!!!", wantErr: "invalid base64"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ParseDataURL(tc.value)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestParseMediaURL(t *testing.T) {
	data := base64.StdEncoding.EncodeToString([]byte("x"))
	media, err := ParseMediaURL("data:image/png;base64,"+data, "high")
	require.NoError(t, err)
	assert.Equal(t, "image/png", media.MIMEType)

	media, err = ParseMediaURL("https://example.com/a.png", "")
	require.NoError(t, err)
	assert.Equal(t, "https://example.com/a.png", media.URL)

	cases := map[string]struct {
		value   string
		wantErr string
	}{
		"empty":  {value: "", wantErr: "empty"},
		"badURL": {value: "not-url", wantErr: "absolute HTTP"},
		"ftp":    {value: "ftp://example.com/a", wantErr: "absolute HTTP"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ParseMediaURL(tc.value, "")
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestParseFileMedia(t *testing.T) {
	data := base64.StdEncoding.EncodeToString([]byte("file-bytes"))
	object := decodeObject(t, `{"file_data":"`+data+`","filename":"a.txt"}`)
	media, filename, err := ParseFileMedia(object, "messages.content.file")
	require.NoError(t, err)
	assert.Equal(t, "a.txt", filename)
	assert.Equal(t, []byte("file-bytes"), media.Data)

	object = decodeObject(t, `{"file_url":"https://example.com/f.pdf"}`)
	media, _, err = ParseFileMedia(object, "param")
	require.NoError(t, err)
	assert.Equal(t, "https://example.com/f.pdf", media.URL)

	object = decodeObject(t, `{"file_id":"file-abc"}`)
	media, _, err = ParseFileMedia(object, "param")
	require.NoError(t, err)
	assert.Equal(t, "file-abc", media.Ref)

	object = decodeObject(t, `{"file_data":"`+data+`","file_id":"x"}`)
	_, _, err = ParseFileMedia(object, "param")
	require.Error(t, err)

	object = decodeObject(t, `{"unknown":1}`)
	_, _, err = ParseFileMedia(object, "param")
	require.Error(t, err)
}

func TestParseFileData(t *testing.T) {
	raw := base64.StdEncoding.EncodeToString([]byte("raw"))
	media, err := ParseFileData(raw, "param")
	require.NoError(t, err)
	assert.Equal(t, "application/octet-stream", media.MIMEType)

	dataURL := "data:application/pdf;base64," + raw
	media, err = ParseFileData(dataURL, "param")
	require.NoError(t, err)
	assert.Equal(t, "application/pdf", media.MIMEType)

	_, err = ParseFileData("!!!", "param")
	require.Error(t, err)
}

func TestParseChatContent(t *testing.T) {
	parts, err := ParseChatContent(nil)
	require.NoError(t, err)
	assert.Nil(t, parts)

	parts, err = ParseChatContent(jsontext.Value("null"))
	require.NoError(t, err)
	assert.Nil(t, parts)

	parts, err = ParseChatContent(jsontext.Value(`"hello"`))
	require.NoError(t, err)
	require.Len(t, parts, 1)
	assert.Equal(t, chat.PartText, parts[0].Type)
	assert.Equal(t, "hello", parts[0].Text)

	imageB64 := base64.StdEncoding.EncodeToString([]byte{1})
	content := `[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + imageB64 + `","detail":"low"}}]`
	parts, err = ParseChatContent(jsontext.Value(content))
	require.NoError(t, err)
	require.Len(t, parts, 2)
	assert.Equal(t, chat.PartImage, parts[1].Type)
	assert.Equal(t, "low", parts[1].Detail)

	audioB64 := base64.StdEncoding.EncodeToString([]byte{2})
	audioContent := `[{"type":"input_audio","input_audio":{"data":"` + audioB64 + `","format":"wav"}}]`
	parts, err = ParseChatContent(jsontext.Value(audioContent))
	require.NoError(t, err)
	require.Len(t, parts, 1)
	assert.Equal(t, chat.PartAudio, parts[0].Type)
	assert.Equal(t, "wav", parts[0].Media.Format)

	fileData := base64.StdEncoding.EncodeToString([]byte("f"))
	fileContent := `[{"type":"file","file":{"file_data":"` + fileData + `","filename":"note.txt"}}]`
	parts, err = ParseChatContent(jsontext.Value(fileContent))
	require.NoError(t, err)
	require.Len(t, parts, 1)
	assert.Equal(t, chat.PartFile, parts[0].Type)

	_, err = ParseChatContent(jsontext.Value(`[{"type":"unknown"}]`))
	require.Error(t, err)

	_, err = ParseChatContent(jsontext.Value(`[{"type":"text"}]`))
	require.Error(t, err)

	badDetail := `[{"type":"image_url","image_url":{"url":"https://example.com/a.png","detail":"ultra"}}]`
	_, err = ParseChatContent(jsontext.Value(badDetail))
	require.Error(t, err)

	badAudio := `[{"type":"input_audio","input_audio":{"data":"` + audioB64 + `","format":"ogg"}}]`
	_, err = ParseChatContent(jsontext.Value(badAudio))
	require.Error(t, err)
}

func TestValidateImageDetail(t *testing.T) {
	for _, detail := range []string{"", "auto", "low", "high"} {
		require.NoError(t, ValidateImageDetail(detail, "param"))
	}
	err := ValidateImageDetail("ultra", "param")
	require.Error(t, err)
}

func TestAudioMIME(t *testing.T) {
	cases := map[string]string{
		"wav":  "audio/wav",
		"mp3":  "audio/mpeg",
		"ogg":  "audio/ogg",
		"opus": "audio/ogg",
		"flac": "audio/flac",
		"m4a":  "audio/mp4",
		"mp4":  "audio/mp4",
		"xyz":  "audio/xyz",
	}
	for format, want := range cases {
		t.Run(format, func(t *testing.T) {
			assert.Equal(t, want, AudioMIME(format))
		})
	}
}

func TestCollectText(t *testing.T) {
	parts := []chat.Part{
		chat.TextPart("hello "),
		chat.ImagePart(chat.InlineMedia("image/png", []byte{1})),
		{Type: chat.PartReasoningSummary, Text: "think"},
	}
	assert.Equal(t, "hello think", CollectText(parts))
}

func TestTextParts(t *testing.T) {
	parts := []chat.Part{chat.TextPart("in"), chat.ImagePart(chat.InlineMedia("image/png", []byte{1})), chat.TextPart("out")}

	input := InputTextParts(parts)
	require.Len(t, input, 2)
	assert.Equal(t, "input_text", input[0]["type"])
	assert.Equal(t, "in", input[0]["text"])

	output := OutputTextParts(parts)
	require.Len(t, output, 2)
	assert.Equal(t, "output_text", output[0]["type"])
	assert.Equal(t, []any{}, output[0]["annotations"])
}

func TestMediaDataURL(t *testing.T) {
	media := chat.InlineMedia("text/plain", []byte("abc"))
	url, err := MediaDataURL(media)
	require.NoError(t, err)
	assert.Contains(t, url, "data:text/plain;base64,")

	_, err = MediaDataURL(chat.Media{URL: "https://example.com/x"})
	require.Error(t, err)

	_, err = MediaDataURL(chat.InlineMedia("", []byte{1}))
	require.Error(t, err)
}

func TestWireAliases(t *testing.T) {
	part := chat.TextPart("x")
	assert.Equal(t, chat.TextPart("x"), part)

	media := chat.InlineMedia("image/png", []byte{1})
	assert.Equal(t, chat.InlineMedia("image/png", []byte{1}), media)

	err := chat.Invalid("p", "m")
	require.NotNil(t, err)
	assert.Equal(t, chat.Invalid("p", "m").Code, err.Code)

	err = chat.Unsupported("p", "m")
	require.NotNil(t, err)
	assert.Equal(t, chat.Unsupported("p", "m").Code, err.Code)
}

func TestStringOrContent(t *testing.T) {
	parts, err := ParseStringOrContent(jsontext.Value(`"plain"`), "param", func(jsontext.Value) ([]chat.Part, error) {
		t.Fatal("parser should not run")
		return nil, nil
	})
	require.NoError(t, err)
	require.Len(t, parts, 1)

	_, err = ParseStringOrContent(jsontext.Value(`[1]`), "param", func(raw jsontext.Value) ([]chat.Part, error) {
		var values []jsontext.Value
		require.NoError(t, json.Unmarshal(raw, &values))
		return []chat.Part{chat.TextPart("from-array")}, nil
	})
	require.NoError(t, err)
}
