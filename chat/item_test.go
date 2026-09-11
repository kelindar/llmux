// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
package chat

import (
	"encoding/json/jsontext"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMediaValidate(t *testing.T) {
	cases := map[string]struct {
		media   Media
		max     int64
		wantErr string
	}{
		"inlineOk": {
			media: InlineMedia("image/png", []byte{1, 2}),
		},
		"remoteOk": {
			media: RemoteMedia("image/png", "https://example.com/a.png"),
		},
		"assetOk": {
			media: AssetMedia("image/png", "asset-1"),
		},
		"noSource": {
			media:   Media{MIMEType: "image/png"},
			wantErr: "exactly one",
		},
		"multipleSources": {
			media:   Media{MIMEType: "image/png", Data: []byte{1}, URL: "https://example.com/a.png"},
			wantErr: "exactly one",
		},
		"exceedsLimit": {
			media:   InlineMedia("image/png", []byte{1, 2, 3}),
			max:     2,
			wantErr: "byte limit",
		},
		"missingMime": {
			media:   Media{Data: []byte{1}},
			wantErr: "MIME type is required",
		},
		"badURL": {
			media:   RemoteMedia("image/png", "not-a-url"),
			wantErr: "absolute HTTP",
		},
		"ftpURL": {
			media:   RemoteMedia("image/png", "ftp://example.com/a.png"),
			wantErr: "absolute HTTP",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := ImagePart(tc.media).Validate(tc.max)
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestMediaConstructors(t *testing.T) {
	data := []byte{9}
	inline := InlineMedia("text/plain", data)
	assert.Equal(t, "text/plain", inline.MIMEType)
	assert.Equal(t, []byte{9}, inline.Data)
	data[0] = 0
	assert.Equal(t, []byte{9}, inline.Data)

	remote := RemoteMedia("image/jpeg", "https://example.com/x.jpg")
	assert.Equal(t, "https://example.com/x.jpg", remote.URL)

	asset := AssetMedia("application/pdf", "doc-1")
	assert.Equal(t, "doc-1", asset.Ref)

	cloned := inline.Clone()
	cloned.Data[0] = 7
	assert.Equal(t, byte(9), inline.Data[0])
}

func TestPartConstructors(t *testing.T) {
	text := TextPart("hello")
	assert.Equal(t, PartText, text.Type)
	assert.Equal(t, "hello", text.Text)

	img := ImagePart(InlineMedia("image/png", []byte{1}))
	assert.Equal(t, PartImage, img.Type)
	require.NotNil(t, img.Media)

	audio := AudioPart(InlineMedia("audio/wav", []byte{2}))
	audio.Media.Format = "wav"
	assert.Equal(t, PartAudio, audio.Type)

	file := FilePart(AssetMedia("application/pdf", "f1"))
	assert.Equal(t, PartFile, file.Type)

	jsonPart := JSONPart(jsontext.Value(`{"a":1}`))
	assert.Equal(t, PartJSON, jsonPart.Type)
	assert.True(t, jsonPart.Data.IsValid())

	summary := SummaryPart("thought")
	assert.Equal(t, PartReasoningSummary, summary.Type)

	cloned := img.Clone()
	cloned.Media.Data[0] = 9
	assert.Equal(t, byte(1), img.Media.Data[0])
}

func TestPartValidate(t *testing.T) {
	cases := map[string]struct {
		part    Part
		wantErr string
	}{
		"textOk":    {part: TextPart("hi")},
		"summaryOk": {part: SummaryPart("brief")},
		"textWithMedia": {
			part:    Part{Type: PartText, Media: &Media{MIMEType: "image/png", Data: []byte{1}}},
			wantErr: "cannot contain media",
		},
		"imageMissingMedia": {
			part:    Part{Type: PartImage},
			wantErr: "missing media",
		},
		"audioMissingFormat": {
			part:    AudioPart(InlineMedia("audio/wav", []byte{1})),
			wantErr: "format is required",
		},
		"audioOk": {
			part: func() Part {
				m := InlineMedia("audio/wav", []byte{1})
				m.Format = "wav"
				return AudioPart(m)
			}(),
		},
		"jsonInvalid": {
			part:    Part{Type: PartJSON, Data: jsontext.Value(`{`)},
			wantErr: "not valid JSON",
		},
		"jsonWithMedia": {
			part:    Part{Type: PartJSON, Data: jsontext.Value(`{}`), Media: &Media{}},
			wantErr: "cannot contain media",
		},
		"unknownType": {
			part:    Part{Type: "other"},
			wantErr: "unknown content part type",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := tc.part.Validate(0)
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestItemValidate(t *testing.T) {
	validAudio := func() Part {
		m := InlineMedia("audio/wav", []byte{1})
		m.Format = "wav"
		return AudioPart(m)
	}

	cases := map[string]struct {
		item    Item
		output  bool
		wantErr string
	}{
		"messageOk": {
			item: MessageItem(RoleUser, TextPart("hi")),
		},
		"assistantOutputOk": {
			item:   MessageItem(RoleAssistant, TextPart("reply")),
			output: true,
		},
		"missingType": {
			item:    Item{Role: RoleUser},
			wantErr: "type is required",
		},
		"invalidStatus": {
			item:    Item{Type: ItemMessage, Role: RoleUser, Status: "bogus"},
			wantErr: "invalid item status",
		},
		"messageWrongRole": {
			item:    MessageItem(RoleUser, TextPart("hi")),
			output:  true,
			wantErr: "agent message role",
		},
		"messageExtraFields": {
			item:    Item{Type: ItemMessage, Role: RoleUser, CallID: "x"},
			wantErr: "fields from another item type",
		},
		"functionCallOk": {
			item: FunctionCallItem("c1", "search", `{}`),
		},
		"functionCallBadArgs": {
			item:    FunctionCallItem("c1", "search", `{`),
			wantErr: "valid JSON",
		},
		"functionCallMissingName": {
			item:    Item{Type: ItemFunctionCall, CallID: "c1", Arguments: `{}`},
			wantErr: "call_id and name",
		},
		"functionCallOutputOk": {
			item: FunctionCallOutputItem("c1", TextPart("result")),
		},
		"functionCallOutputMissingID": {
			item:    Item{Type: ItemFunctionCallOutput, Output: []Part{TextPart("x")}},
			wantErr: "call_id",
		},
		"reasoningOk": {
			item: Item{Type: ItemReasoning, Summary: []Part{SummaryPart("think")}},
		},
		"reasoningBadPart": {
			item:    Item{Type: ItemReasoning, Summary: []Part{TextPart("nope")}},
			wantErr: "reasoning_summary",
		},
		"mediaImageOk": {
			item: Item{
				Type:    ItemMedia,
				Content: []Part{ImagePart(InlineMedia("image/png", []byte{1}))},
			},
		},
		"mediaAudioOk": {
			item: Item{Type: ItemMedia, Content: []Part{validAudio()}},
		},
		"mediaWrongCount": {
			item:    Item{Type: ItemMedia, Content: []Part{}},
			wantErr: "exactly one content part",
		},
		"extensionOk": {
			item: Item{Type: ItemExtension, Data: jsontext.Value(`{"k":"v"}`)},
		},
		"extensionInvalidJSON": {
			item:    Item{Type: ItemExtension, Data: jsontext.Value(`{`)},
			wantErr: "valid JSON data",
		},
		"unknownType": {
			item:    Item{Type: "other"},
			wantErr: "unknown item type",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := tc.item.Validate(0, tc.output)
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestItemClone(t *testing.T) {
	orig := MessageItem(RoleUser, ImagePart(InlineMedia("image/png", []byte{1})))
	cloned := orig.Clone()
	cloned.Content[0].Media.Data[0] = 9
	assert.Equal(t, byte(1), orig.Content[0].Media.Data[0])
}

func TestMessageConstructors(t *testing.T) {
	msg := MessageItem(RoleSystem, TextPart("sys"))
	assert.Equal(t, ItemMessage, msg.Type)
	assert.Equal(t, RoleSystem, msg.Role)

	call := FunctionCallItem("id1", "fn", `{"q":"x"}`)
	assert.Equal(t, ItemFunctionCall, call.Type)
	assert.NotEmpty(t, call.ID)
	assert.Equal(t, StatusCompleted, call.Status)

	out := FunctionCallOutputItem("id1", TextPart("done"))
	assert.Equal(t, ItemFunctionCallOutput, out.Type)
	assert.Equal(t, "id1", out.CallID)
}
