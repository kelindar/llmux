// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package wire

import (
	"encoding/base64"
	"errors"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
