// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package agui

import (
	"errors"
	"strings"
	"testing"

	"github.com/kelindar/llmux/chat"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseRequest(t *testing.T) {
	body := []byte(`{"threadId":"thread-1","runId":"run-1","protocolVersion":"1.0","tools":[],"context":[ ],"state":{ },"messages":[{"id":"message-1","role":"user","content":[{"type":"text","text":"look"},{"type":"image","source":{"type":"data","value":"aGVsbG8=","mimeType":"image/png"}}]}],"forwardedProps":{"llmux":{"target":"agent","ui":{"format":"a2ui","version":"0.9.1","catalogId":"catalog-1"}}}}`)
	parsed, err := ParseRequest(body)
	require.NoError(t, err)
	assert.Equal(t, internalprotocol.AGUI, parsed.Kind)
	assert.Equal(t, "thread-1", parsed.Thread)
	assert.Equal(t, "run-1", parsed.RunID)
	assert.True(t, parsed.Stream)
	assert.True(t, parsed.Retain)
	require.NotNil(t, parsed.Store)
	assert.True(t, *parsed.Store)
	assert.Equal(t, "agent", parsed.Request.Target)
	require.Len(t, parsed.Request.Input, 1)
	message := parsed.Request.Input[0]
	assert.Equal(t, chat.ItemMessage, message.Type)
	assert.Equal(t, "message-1", message.ID)
	assert.Equal(t, chat.RoleUser, message.Role)
	require.Len(t, message.Content, 2)
	assert.Equal(t, "look", message.Content[0].Text)
	require.NotNil(t, message.Content[1].Media)
	assert.Equal(t, []byte("hello"), message.Content[1].Media.Data)
	assert.JSONEq(t, `{"format":"a2ui","version":"0.9.1","catalogId":"catalog-1"}`, string(parsed.Request.Controls.Extensions["x-ui"]))
}

func TestParseAction(t *testing.T) {
	body := []byte(`{"threadId":"t","runId":"r","messages":[],"forwardedProps":{"llmux":{"target":"agent","previousResponseId":"response-1","action":{"name":"submit","surfaceId":"surface-1","sourceComponentId":"button-1","context":{"answer":42},"sourceResponseId":"response-1","sourceItemId":"item-1"},"ui":{"format":"a2ui","version":"0.9.1","catalogId":"catalog-1"}}}}`)
	parsed, err := ParseRequest(body)
	require.NoError(t, err)
	require.NotNil(t, parsed.Previous)
	assert.Equal(t, "response-1", *parsed.Previous)
	require.Len(t, parsed.Request.Input, 1)
	assert.Equal(t, chat.ItemExtension, parsed.Request.Input[0].Type)
	assert.JSONEq(t, `{"kind":"ui_action","name":"submit","surfaceId":"surface-1","sourceComponentId":"button-1","context":{"answer":42},"sourceResponseId":"response-1","sourceItemId":"item-1"}`, string(parsed.Request.Input[0].Data))
}

func TestParseOwnership(t *testing.T) {
	body := []byte(`{"threadId":"t","runId":"r","messages":[{"id":"m","role":"user","content":"hello"}],"forwardedProps":{"llmux":{"target":"agent","ui":{"format":"a2ui","version":"0.9.1","catalogId":"catalog-1"}}}}`)
	parsed, err := ParseRequest(body)
	require.NoError(t, err)
	for i := range body {
		body[i] = 'x'
	}
	assert.Equal(t, "t", parsed.Thread)
	assert.Equal(t, "hello", parsed.Request.Input[0].Content[0].Text)
	assert.JSONEq(t, `{"format":"a2ui","version":"0.9.1","catalogId":"catalog-1"}`, string(parsed.Request.Controls.Extensions["x-ui"]))
}

func TestParseRejects(t *testing.T) {
	cases := map[string]string{
		"missingThread":     `{"runId":"r","messages":[],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"missingRun":        `{"threadId":"t","messages":[],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"missingTarget":     `{"threadId":"t","runId":"r","messages":[{"id":"m","role":"user","content":"hi"}],"forwardedProps":{"llmux":{}}}`,
		"history":           `{"threadId":"t","runId":"r","messages":[{"id":"m1","role":"user","content":"a"},{"id":"m2","role":"user","content":"b"}],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"assistantRole":     `{"threadId":"t","runId":"r","messages":[{"id":"m","role":"assistant","content":"x"}],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"actionParent":      `{"threadId":"t","runId":"r","messages":[],"forwardedProps":{"llmux":{"target":"a","previousResponseId":"p1","action":{"name":"n","surfaceId":"s","sourceComponentId":"c","context":{},"sourceResponseId":"p2","sourceItemId":"i"}}}}`,
		"actionWithMessage": `{"threadId":"t","runId":"r","messages":[{"id":"m","role":"user","content":"x"}],"forwardedProps":{"llmux":{"target":"a","previousResponseId":"p","action":{"name":"n","surfaceId":"s","sourceComponentId":"c","context":{},"sourceResponseId":"p","sourceItemId":"i"}}}}`,
		"tools":             `{"threadId":"t","runId":"r","tools":[{"name":"search"}],"messages":[],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"context":           `{"threadId":"t","runId":"r","context":[{"description":"private","value":"x"}],"messages":[],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"state":             `{"threadId":"t","runId":"r","state":{"secret":1},"messages":[],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"resume":            `{"threadId":"t","runId":"r","resume":[{"interruptId":"i","status":"resolved"}],"messages":[],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"parentRun":         `{"threadId":"t","runId":"r","parentRunId":"p","messages":[],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"forwarded":         `{"threadId":"t","runId":"r","messages":[],"forwardedProps":{"other":1,"llmux":{"target":"a"}}}`,
		"video":             `{"threadId":"t","runId":"r","messages":[{"id":"m","role":"user","content":[{"type":"video","source":{"type":"url","value":"https://example.com/v"}}]}],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"duplicate":         `{"threadId":"t","threadId":"other","runId":"r","messages":[],"forwardedProps":{"llmux":{"target":"a"}}}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ParseRequest([]byte(body))
			require.Error(t, err)
			apiErr, ok := errors.AsType[*chat.Error](err)
			require.True(t, ok, "%T", err)
			assert.NotEmpty(t, apiErr.Code)
		})
	}
}

func TestParseImages(t *testing.T) {
	parsed, err := ParseRequest([]byte(requestWithContent(`[{"type":"image","source":{"type":"url","value":"https://example.com/image.png","mimeType":"image/png"}},{"type":"image","source":{"type":"file","value":"file-1","mimeType":"image/jpeg"}},{"type":"text","text":"caption","id":"text-1","metadata":{}}]`)))
	require.NoError(t, err)
	parts := parsed.Request.Input[0].Content
	require.Len(t, parts, 3)
	assert.Equal(t, "https://example.com/image.png", parts[0].Media.URL)
	assert.Equal(t, "image/png", parts[0].Media.MIMEType)
	assert.Equal(t, "file-1", parts[1].Media.Ref)
	assert.Equal(t, "caption", parts[2].Text)

	parsed, err = ParseRequest([]byte(requestWithContent(`[{"type":"text","text":""}]`)))
	require.NoError(t, err)
	require.Len(t, parsed.Request.Input[0].Content, 1)
	assert.Empty(t, parsed.Request.Input[0].Content[0].Text)
}

func TestParseRejectsShapes(t *testing.T) {
	cases := map[string]string{
		"notObject":          `[]`,
		"badJSON":            `{`,
		"unknownTop":         `{"threadId":"t","runId":"r","extra":1,"messages":[],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"badProtocolType":    `{"threadId":"t","runId":"r","protocolVersion":1,"messages":[],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"badProtocolVersion": `{"threadId":"t","runId":"r","protocolVersion":"0.9","messages":[],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"badParentType":      `{"threadId":"t","runId":"r","parentRunId":null,"messages":[],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"emptyMessages":      `{"threadId":"t","runId":"r","messages":[],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"missingMessages":    `{"threadId":"t","runId":"r","forwardedProps":{"llmux":{"target":"a"}}}`,
		"messagesNotArray":   `{"threadId":"t","runId":"r","messages":{},"forwardedProps":{"llmux":{"target":"a"}}}`,
		"messageNotObject":   `{"threadId":"t","runId":"r","messages":[1],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"messageUnknown":     `{"threadId":"t","runId":"r","messages":[{"id":"m","role":"user","content":"x","name":"extra"}],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"contentMissing":     `{"threadId":"t","runId":"r","messages":[{"id":"m","role":"user"}],"forwardedProps":{"llmux":{"target":"a"}}}`,
		"contentType":        requestWithContent(`true`),
		"emptyContentParts":  requestWithContent(`[]`),
		"partNotObject":      requestWithContent(`[1]`),
		"partMissingType":    requestWithContent(`[{"text":"x"}]`),
		"partUnknown":        requestWithContent(`[{"type":"text","text":"x","url":"https://example.com"}]`),
		"textMissing":        requestWithContent(`[{"type":"text"}]`),
		"textNotString":      requestWithContent(`[{"type":"text","text":1}]`),
		"textHasSource":      requestWithContent(`[{"type":"text","text":"x","source":{}}]`),
		"imageHasText":       requestWithContent(`[{"type":"image","text":"x","source":{"type":"file","value":"f"}}]`),
		"imageMissingSource": requestWithContent(`[{"type":"image"}]`),
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ParseRequest([]byte(body))
			assert.Error(t, err)
		})
	}
}

func TestParseRejectsImages(t *testing.T) {
	cases := map[string]string{
		"sourceNotObject":     requestWithContent(`[{"type":"image","source":[]}]`),
		"missingSourceType":   requestWithContent(`[{"type":"image","source":{"value":"x"}}]`),
		"sourceTypeNotString": requestWithContent(`[{"type":"image","source":{"type":1,"value":"x"}}]`),
		"missingSourceValue":  requestWithContent(`[{"type":"image","source":{"type":"file"}}]`),
		"sourceValueEmpty":    requestWithContent(`[{"type":"image","source":{"type":"file","value":""}}]`),
		"mimeNotString":       requestWithContent(`[{"type":"image","source":{"type":"file","value":"f","mimeType":1}}]`),
		"providerNotString":   requestWithContent(`[{"type":"image","source":{"type":"file","value":"f","provider":1}}]`),
		"unknownSourceField":  requestWithContent(`[{"type":"image","source":{"type":"file","value":"f","other":1}}]`),
		"badBase64":           requestWithContent(`[{"type":"image","source":{"type":"data","value":"%%","mimeType":"image/png"}}]`),
		"dataNoMime":          requestWithContent(`[{"type":"image","source":{"type":"data","value":"aGVsbG8="}}]`),
		"dataProvider":        requestWithContent(`[{"type":"image","source":{"type":"data","value":"aGVsbG8=","mimeType":"image/png","provider":"cloud"}}]`),
		"urlProvider":         requestWithContent(`[{"type":"image","source":{"type":"url","value":"https://example.com/a.png","provider":"cloud"}}]`),
		"fileProvider":        requestWithContent(`[{"type":"image","source":{"type":"file","value":"f","provider":"cloud"}}]`),
		"badURL":              requestWithContent(`[{"type":"image","source":{"type":"url","value":"relative/path"}}]`),
		"unknownSourceType":   requestWithContent(`[{"type":"image","source":{"type":"inline","value":"abc"}}]`),
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ParseRequest([]byte(body))
			assert.Error(t, err)
		})
	}
}

func TestParseRejectsForwarded(t *testing.T) {
	base := `{"threadId":"t","runId":"r","messages":[],"forwardedProps":%s}`
	cases := map[string]string{
		"missing":           `{"threadId":"t","runId":"r","messages":[]}`,
		"notObject":         strings.ReplaceAll(base, "%s", `[]`),
		"missingLLMux":      strings.ReplaceAll(base, "%s", `{}`),
		"llmuxNotObject":    strings.ReplaceAll(base, "%s", `{"llmux":[]}`),
		"unknownForward":    strings.ReplaceAll(base, "%s", `{"other":{},"llmux":{"target":"a"}}`),
		"unknownLLMux":      strings.ReplaceAll(base, "%s", `{"llmux":{"target":"a","other":1}}`),
		"targetNotString":   strings.ReplaceAll(base, "%s", `{"llmux":{"target":1}}`),
		"previousNotString": strings.ReplaceAll(base, "%s", `{"llmux":{"target":"a","previousResponseId":1}}`),
		"uiNotObject":       strings.ReplaceAll(base, "%s", `{"llmux":{"target":"a","ui":[]}}`),
		"uiUnknown":         strings.ReplaceAll(base, "%s", `{"llmux":{"target":"a","ui":{"format":"a2ui","version":"0.9.1","catalogId":"c","other":true}}}`),
		"uiWrongFormat":     strings.ReplaceAll(base, "%s", `{"llmux":{"target":"a","ui":{"format":"other","version":"0.9.1","catalogId":"c"}}}`),
		"uiWrongVersion":    strings.ReplaceAll(base, "%s", `{"llmux":{"target":"a","ui":{"format":"a2ui","version":"0.8","catalogId":"c"}}}`),
		"uiMissingCatalog":  strings.ReplaceAll(base, "%s", `{"llmux":{"target":"a","ui":{"format":"a2ui","version":"0.9.1"}}}`),
		"actionNotObject":   strings.ReplaceAll(base, "%s", `{"llmux":{"target":"a","action":[]}}`),
		"actionBadContext":  strings.ReplaceAll(base, "%s", `{"llmux":{"target":"a","action":{"name":"n","surfaceId":"s","sourceComponentId":"c","sourceResponseId":"p","sourceItemId":"i","context":[]}}}`),
		"actionUnknown":     strings.ReplaceAll(base, "%s", `{"llmux":{"target":"a","action":{"name":"n","surfaceId":"s","sourceComponentId":"c","sourceResponseId":"p","sourceItemId":"i","context":{},"kind":"wrong"}}}`),
		"actionMissing":     strings.ReplaceAll(base, "%s", `{"llmux":{"target":"a","action":{"name":"n"}}}`),
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ParseRequest([]byte(body))
			assert.Error(t, err)
		})
	}

	largeAction := `{"name":"n","surfaceId":"s","sourceComponentId":"c","sourceResponseId":"p","sourceItemId":"i","context":{"data":"` + strings.Repeat("x", maxActionBytes) + `"}}`
	large := strings.ReplaceAll(base, "%s", `{"llmux":{"target":"a","action":`+largeAction+`}}`)
	_, err := ParseRequest([]byte(large))
	require.Error(t, err)
	apiErr, ok := errors.AsType[*chat.Error](err)
	require.True(t, ok)
	assert.Equal(t, 413, apiErr.Status)
	assert.Equal(t, "action_too_large", apiErr.Code)
}

func requestWithContent(content string) string {
	return `{"threadId":"t","runId":"r","messages":[{"id":"m","role":"user","content":` + content + `}],"forwardedProps":{"llmux":{"target":"agent"}}}`
}
