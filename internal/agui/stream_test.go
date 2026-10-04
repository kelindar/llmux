// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package agui

import (
	"encoding/json/jsontext"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/kelindar/llmux/chat"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStreamLifecycle(t *testing.T) {
	encoder, response, _ := testStream()
	s := encoder.(*stream)
	text := chat.Item{Type: chat.ItemMessage, ID: "message-1", Role: chat.RoleAssistant, Content: []chat.Part{chat.TextPart("hello")}}
	ui := chat.Item{Type: chat.ItemExtension, ID: "ui-1", Data: jsontext.Value(validUIData)}
	require.NoError(t, s.Event(chat.Event{Type: chat.EventTextDelta, ItemID: "message-1", Delta: "hello"}))
	require.NoError(t, s.Event(chat.Event{Type: chat.EventTextDone, ItemID: "message-1"}))
	require.NoError(t, s.Event(chat.OutputItem(ui)))
	assert.NotContains(t, response.Body.String(), `"type":"CUSTOM"`)

	require.NoError(t, s.Complete(chat.Outcome{Status: chat.StatusCompleted}, []chat.Item{text, ui}))
	body := response.Body.String()
	for _, event := range []string{`"type":"RUN_STARTED"`, `"type":"TEXT_MESSAGE_START"`, `"type":"TEXT_MESSAGE_CONTENT"`, `"type":"TEXT_MESSAGE_END"`, `"type":"CUSTOM"`, `"type":"RUN_FINISHED"`} {
		assert.Contains(t, body, event)
	}
	assert.NotContains(t, body, "[DONE]")
	assert.Less(t, strings.Index(body, `"type":"RUN_STARTED"`), strings.Index(body, `"type":"TEXT_MESSAGE_START"`))
	assert.Less(t, strings.Index(body, `"type":"TEXT_MESSAGE_END"`), strings.Index(body, `"type":"CUSTOM"`))
	assert.Less(t, strings.Index(body, `"type":"CUSTOM"`), strings.Index(body, `"type":"RUN_FINISHED"`))
	assert.Contains(t, body, `"threadId":"stored-thread"`)
	assert.Contains(t, body, `"runId":"run-1"`)
	assert.Contains(t, body, `"responseId":"response-1"`)
	assert.Contains(t, body, `"catalogId":"catalog-1"`)
	assert.Contains(t, body, `"custom":"value"`)
}

func TestStreamTerminal(t *testing.T) {
	for _, status := range []chat.Status{chat.StatusFailed, chat.StatusCancelled, chat.StatusIncomplete} {
		t.Run(string(status), func(t *testing.T) {
			encoder, response, meta := testStream()
			s := encoder.(*stream)
			meta.Response.Status = status
			meta.Response.Error = &chat.Error{Code: "safe_code", Message: "safe message"}
			text := chat.Item{Type: chat.ItemMessage, ID: "message-1", Role: chat.RoleAssistant, Content: []chat.Part{chat.TextPart("partial")}}
			ui := chat.Item{Type: chat.ItemExtension, ID: "ui-1", Data: jsontext.Value(validUIData)}
			require.NoError(t, s.Complete(chat.Outcome{Status: status}, []chat.Item{text, ui}))
			body := response.Body.String()
			assert.Contains(t, body, `"type":"TEXT_MESSAGE_CONTENT"`)
			assert.Contains(t, body, `"type":"RUN_ERROR"`)
			assert.Contains(t, body, `"message":"safe message"`)
			assert.Contains(t, body, `"code":"safe_code"`)
			assert.NotContains(t, body, `"type":"CUSTOM"`)
			assert.NotContains(t, body, `"type":"RUN_FINISHED"`)
		})
	}
}

func TestStreamFailure(t *testing.T) {
	t.Run("before start", func(t *testing.T) {
		encoder, response, _ := testStream()
		require.Error(t, encoder.Fail(chat.Invalid("request", "safe message")))
		assert.Equal(t, "application/json", response.Header().Get("Content-Type"))
		assert.Contains(t, response.Body.String(), `"message":"safe message"`)
		assert.NotContains(t, response.Body.String(), "RUN_ERROR")
	})
	t.Run("after start", func(t *testing.T) {
		encoder, response, _ := testStream()
		s := encoder.(*stream)
		require.NoError(t, s.Event(chat.Activity("progress", jsontext.Value(`{"step":1}`))))
		require.NoError(t, s.Fail(chat.Invalid("run", "safe message")))
		assert.Contains(t, response.Body.String(), `"type":"RUN_ERROR"`)
		assert.Contains(t, response.Body.String(), `"message":"safe message"`)
		assert.NotContains(t, response.Body.String(), `"type":"RUN_FINISHED"`)
	})
}

func TestStreamActivity(t *testing.T) {
	encoder, response, _ := testStream()
	s := encoder.(*stream)
	require.NoError(t, s.Event(chat.Activity("progress", jsontext.Value(`{"step":1}`))))
	body := response.Body.String()
	assert.Contains(t, body, `"type":"ACTIVITY_SNAPSHOT"`)
	assert.Contains(t, body, `"messageId":"run-1"`)
	assert.Contains(t, body, `"activityType":"progress"`)
	assert.Contains(t, body, `"content":{"step":1}`)
}

func TestStreamEdges(t *testing.T) {
	t.Run("text closure", func(t *testing.T) {
		encoder, response, _ := testStream()
		s := encoder.(*stream)
		require.Error(t, s.Event(chat.TextDone("missing")))
		require.NoError(t, s.Event(chat.Event{Type: chat.EventTextDelta, ItemID: "message-1", Delta: "x"}))
		require.NoError(t, s.Event(chat.Event{Type: chat.EventTextDelta, ItemID: "message-1"}))
		assert.Equal(t, 1, strings.Count(response.Body.String(), `"type":"TEXT_MESSAGE_CONTENT"`))
		require.NoError(t, s.Event(chat.TextDone("message-1")))
		require.Error(t, s.Event(chat.Event{Type: chat.EventTextDelta, ItemID: "message-1", Delta: "closed"}))
	})
	t.Run("unsupported events", func(t *testing.T) {
		encoder, response, _ := testStream()
		s := encoder.(*stream)
		require.Error(t, s.Event(chat.ToolStart("call-1", "search")))
		require.Error(t, s.Event(chat.Event{Type: "unknown"}))
		require.NoError(t, s.Event(chat.Activity("progress", jsontext.Value(`{"step":1}`))))
		require.Error(t, s.Event(chat.OutputItem(chat.Item{Type: chat.ItemFunctionCall, CallID: "call-1", Name: "search", Arguments: `{}`})))
		assert.Contains(t, response.Body.String(), `"type":"RUN_STARTED"`)
		assert.NotContains(t, response.Body.String(), `"type":"CUSTOM"`)
	})
	t.Run("duplicate item", func(t *testing.T) {
		encoder, _, _ := testStream()
		s := encoder.(*stream)
		item := chat.Item{Type: chat.ItemMessage, ID: "message-1", Role: chat.RoleAssistant, Content: []chat.Part{chat.TextPart("hello")}}
		require.NoError(t, s.Event(chat.OutputItem(item)))
		require.Error(t, s.Event(chat.OutputItem(item)))
	})
}

func TestStreamCompletion(t *testing.T) {
	t.Run("defaults and fallback", func(t *testing.T) {
		response := httptest.NewRecorder()
		parsed := internalprotocol.ParsedRequest{Thread: "submitted", RunID: "run-1"}
		s := (Adapter{}).Stream(response, parsed, nil, chat.DefaultLimits()).(*stream)
		assert.False(t, s.Started())
		require.NoError(t, s.Complete(chat.Outcome{}, nil))
		body := response.Body.String()
		assert.Contains(t, body, `"threadId":"submitted"`)
		assert.Contains(t, body, `"type":"RUN_FINISHED"`)
		assert.Contains(t, body, `"metadata":{}`)
		assert.True(t, s.Started())
	})
	t.Run("invalid identity", func(t *testing.T) {
		response := httptest.NewRecorder()
		s := (Adapter{}).Stream(response, internalprotocol.ParsedRequest{RunID: "run-1"}, nil, chat.DefaultLimits()).(*stream)
		require.Error(t, s.Complete(chat.Outcome{}, nil))
		assert.False(t, s.Started())
		assert.Empty(t, response.Body.String())
	})
	t.Run("unsupported output", func(t *testing.T) {
		encoder, response, _ := testStream()
		s := encoder.(*stream)
		badMessage := chat.Item{Type: chat.ItemMessage, ID: "message-1", Role: chat.RoleAssistant, Content: []chat.Part{chat.ImagePart(chat.AssetMedia("image/png", "file-1"))}}
		require.Error(t, s.Complete(chat.Outcome{Status: chat.StatusCompleted}, []chat.Item{badMessage}))
		assert.NotContains(t, response.Body.String(), `"type":"RUN_FINISHED"`)
	})
	t.Run("unsupported item", func(t *testing.T) {
		encoder, response, _ := testStream()
		require.Error(t, encoder.Complete(chat.Outcome{Status: chat.StatusCompleted}, []chat.Item{{Type: chat.ItemFunctionCall}}))
		assert.NotContains(t, response.Body.String(), `"type":"RUN_FINISHED"`)
	})
	t.Run("wrong catalog", func(t *testing.T) {
		encoder, response, _ := testStream()
		badUI := strings.ReplaceAll(validUIData, `"catalogId":"catalog-1"`, `"catalogId":"other"`)
		require.Error(t, encoder.Complete(chat.Outcome{Status: chat.StatusCompleted}, []chat.Item{{Type: chat.ItemExtension, ID: "ui-1", Data: jsontext.Value(badUI)}}))
		assert.NotContains(t, response.Body.String(), `"type":"CUSTOM"`)
	})
}

func TestStreamErrors(t *testing.T) {
	cases := []struct {
		status     chat.Status
		apiErr     *chat.Error
		incomplete string
		message    string
		code       string
	}{
		{status: chat.StatusFailed, apiErr: &chat.Error{Message: "safe", Code: "safe_code"}, message: "safe", code: "safe_code"},
		{status: chat.StatusFailed, apiErr: &chat.Error{}, message: "run failed", code: "failed"},
		{status: chat.StatusIncomplete, incomplete: "length", message: "length", code: "incomplete"},
		{status: chat.StatusCancelled, message: "run cancelled", code: "cancelled"},
	}
	for _, tc := range cases {
		t.Run(string(tc.status)+tc.message, func(t *testing.T) {
			encoder, response, meta := testStream()
			meta.Response.Status = tc.status
			meta.Response.Error = tc.apiErr
			meta.Response.Incomplete = tc.incomplete
			require.NoError(t, encoder.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil))
			body := response.Body.String()
			assert.Contains(t, body, `"type":"RUN_ERROR"`)
			assert.Contains(t, body, `"message":"`+tc.message+`"`)
			assert.Contains(t, body, `"code":"`+tc.code+`"`)
		})
	}
}

func TestStreamStartFail(t *testing.T) {
	response := httptest.NewRecorder()
	s := (Adapter{}).Stream(response, internalprotocol.ParsedRequest{}, nil, chat.DefaultLimits()).(*stream)
	require.NoError(t, s.w.Start())
	require.Error(t, s.Fail(chat.Invalid("run", "failed")))
}

func testStream() (internalprotocol.StreamEncoder, *httptest.ResponseRecorder, *internalprotocol.Meta) {
	response := httptest.NewRecorder()
	parsed := internalprotocol.ParsedRequest{
		Thread: "submitted-thread",
		RunID:  "run-1",
		Request: chat.Request{Controls: chat.Controls{Extensions: map[string]jsontext.Value{
			"x-ui": jsontext.Value(`{"format":"a2ui","version":"0.9.1","catalogId":"catalog-1"}`),
		}}},
	}
	meta := &internalprotocol.Meta{Response: chat.Response{
		ID:       "response-1",
		Metadata: map[string]string{"thread_id": "stored-thread", "custom": "value"},
	}}
	return (Adapter{}).Stream(response, parsed, meta, chat.DefaultLimits()), response, meta
}
