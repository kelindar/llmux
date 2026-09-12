// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package anthropic

import (
	"bytes"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStreamStart(t *testing.T) {
	rec := httptest.NewRecorder()
	meta := responseMeta{Response: chat.Response{ID: "m1", Target: "m"}}
	stream := Adapter{}.Stream(rec, parsedRequest{}, &meta, chat.DefaultLimits())
	require.NoError(t, stream.Event(chat.Text("hi")))
	require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("hi"))}))
	assert.Contains(t, rec.Body.String(), "message_start")
}

func TestStreamOutputContracts(t *testing.T) {
	rec := httptest.NewRecorder()
	meta := responseMeta{Response: chat.Response{ID: "m1", Target: "m"}}
	stream := Adapter{}.Stream(rec, parsedRequest{}, &meta, chat.DefaultLimits())

	require.NoError(t, stream.Event(chat.Text("hello")))
	require.NoError(t, stream.Event(chat.Tool("call-1", "lookup", `{"q":"x"}`)))
	require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, []chat.Item{
		chat.FunctionCallItem("call-1", "lookup", `{"q":"x"}`),
	}))

	body := rec.Body.String()
	assert.Contains(t, body, `"type":"content_block_start"`)
	assert.Contains(t, body, `"type":"tool_use"`)
	assert.Contains(t, body, `"name":"lookup"`)
	assert.Contains(t, body, `"type":"input_json_delta"`)
	assert.Contains(t, body, `"partial_json":"{\"q\":\"x\"}"`)
	assert.Contains(t, body, `"stop_reason":"tool_use"`)
	assert.Contains(t, body, `"type":"message_stop"`)
}

type anthropicStreamErrorWriter struct {
	header http.Header
	body   bytes.Buffer
	failAt int
	writes int
}

func (w *anthropicStreamErrorWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}

func (w *anthropicStreamErrorWriter) WriteHeader(int) {}

func (w *anthropicStreamErrorWriter) Write(p []byte) (int, error) {
	w.writes++
	if w.writes == w.failAt {
		return 0, errors.New("client disconnected")
	}
	return w.body.Write(p)
}

func (w *anthropicStreamErrorWriter) Flush() {}

func TestStreamDeliveryErrors(t *testing.T) {
	tests := []struct {
		name   string
		failAt int
		run    func(*testing.T, *anthropicStream) error
	}{
		{
			name:   "message content block start",
			failAt: 2,
			run: func(_ *testing.T, stream *anthropicStream) error {
				return stream.Event(chat.Text("hello"))
			},
		},
		{
			name:   "message content delta",
			failAt: 3,
			run: func(_ *testing.T, stream *anthropicStream) error {
				return stream.Event(chat.Text("hello"))
			},
		},
		{
			name:   "message content block stop",
			failAt: 4,
			run: func(_ *testing.T, stream *anthropicStream) error {
				return stream.Event(chat.Text("hello"))
			},
		},
		{
			name:   "tool content block start",
			failAt: 2,
			run: func(_ *testing.T, stream *anthropicStream) error {
				return stream.Event(chat.ToolStart("call-1", "lookup"))
			},
		},
		{
			name:   "tool input delta",
			failAt: 3,
			run: func(_ *testing.T, stream *anthropicStream) error {
				return stream.Event(chat.Tool("call-1", "lookup", `{"q":"x"}`))
			},
		},
		{
			name:   "tool closes text",
			failAt: 4,
			run: func(t *testing.T, stream *anthropicStream) error {
				require.NoError(t, stream.Event(chat.TextDelta("hello")))
				return stream.Event(chat.ToolStart("call-1", "lookup"))
			},
		},
		{
			name:   "complete starts stream",
			failAt: 1,
			run: func(_ *testing.T, stream *anthropicStream) error {
				return stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil)
			},
		},
		{
			name:   "complete closes text",
			failAt: 4,
			run: func(t *testing.T, stream *anthropicStream) error {
				require.NoError(t, stream.Event(chat.TextDelta("hello")))
				return stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil)
			},
		},
		{
			name:   "complete closes tool",
			failAt: 3,
			run: func(t *testing.T, stream *anthropicStream) error {
				require.NoError(t, stream.Event(chat.ToolStart("call-1", "lookup")))
				return stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil)
			},
		},
		{
			name:   "complete message delta",
			failAt: 5,
			run: func(t *testing.T, stream *anthropicStream) error {
				require.NoError(t, stream.Event(chat.TextDelta("hello")))
				return stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil)
			},
		},
		{
			name:   "fail error event",
			failAt: 4,
			run: func(t *testing.T, stream *anthropicStream) error {
				require.NoError(t, stream.Event(chat.TextDelta("hello")))
				return stream.Fail(errors.New("upstream failed"))
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			writer := &anthropicStreamErrorWriter{failAt: test.failAt}
			meta := responseMeta{Response: chat.Response{ID: "m1", Target: "m"}}
			stream := Adapter{}.Stream(writer, parsedRequest{}, &meta, chat.DefaultLimits()).(*anthropicStream)

			err := test.run(t, stream)
			require.Error(t, err)
			assert.ErrorIs(t, err, chat.ErrDelivery)
		})
	}
}
