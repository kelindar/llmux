// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package completions

import (
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
	meta := responseMeta{Response: chat.Response{ID: "c1", Created: 1, Target: "m"}}
	stream := Adapter{}.Stream(rec, parsedRequest{}, &meta, chat.DefaultLimits())
	require.NoError(t, stream.Event(chat.Text("hi")))
	require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("hi"))}))
	assert.Contains(t, rec.Body.String(), "chat.completion.chunk")
}

func TestStreamOutputContracts(t *testing.T) {
	meta := responseMeta{Response: chat.Response{ID: "c1", Created: 1, Target: "m"}}

	t.Run("toolLifecycle", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := Adapter{}.Stream(rec, parsedRequest{IncludeUsage: true}, &meta, chat.DefaultLimits())
		start := chat.Event{Type: chat.EventToolCallStart, CallID: "call_1", Name: "lookup"}
		require.NoError(t, stream.Event(start))
		require.NoError(t, stream.Event(start))
		require.NoError(t, stream.Event(chat.Event{Type: chat.EventToolCallDelta, CallID: "call_1", Delta: `{"q":"x"}`}))
		require.NoError(t, stream.Event(chat.Event{Type: chat.EventToolCallDone, CallID: "call_1"}))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted, Usage: &chat.Usage{Input: 2, Output: 3, Total: 5}}, nil))
		body := rec.Body.String()
		assert.Contains(t, body, `"tool_calls"`)
		assert.Contains(t, body, `"finish_reason":"tool_calls"`)
		assert.Contains(t, body, `"usage"`)
	})

	t.Run("audioMessage", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := Adapter{}.Stream(rec, parsedRequest{}, &meta, chat.DefaultLimits())
		item := chat.MessageItem(chat.RoleAssistant, chat.AudioPart(chat.InlineMedia("audio/wav", []byte{1, 2})))
		require.NoError(t, stream.Event(chat.OutputItem(item)))
		assert.Contains(t, rec.Body.String(), `"audio":{"data":"AQI="}`)
	})

	t.Run("audioMedia", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := Adapter{}.Stream(rec, parsedRequest{}, &meta, chat.DefaultLimits())
		require.NoError(t, stream.Event(chat.MediaItem(chat.AudioPart(chat.InlineMedia("audio/wav", []byte{3, 4})))))
		assert.Contains(t, rec.Body.String(), `"audio":{"data":"AwQ="}`)
	})

	t.Run("rejectsNonInlineAudio", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := Adapter{}.Stream(rec, parsedRequest{}, &meta, chat.DefaultLimits())
		err := stream.Event(chat.OutputItem(chat.Item{Type: chat.ItemMedia, Content: []chat.Part{chat.AudioPart(chat.RemoteMedia("audio/wav", "https://example.com/a.wav"))}}))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "inline data")
	})
}

type completionStreamErrorWriter struct {
	header http.Header
	failAt int
	writes int
}

func (w *completionStreamErrorWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}

func (w *completionStreamErrorWriter) WriteHeader(int) {}

func (w *completionStreamErrorWriter) Write([]byte) (int, error) {
	w.writes++
	if w.failAt > 0 && w.writes == w.failAt {
		return 0, errors.New("write failed")
	}
	return 1, nil
}

func (w *completionStreamErrorWriter) Flush() {}

func TestStreamDeliveryErrors(t *testing.T) {
	meta := responseMeta{Response: chat.Response{ID: "c1", Created: 1, Target: "m"}}

	writer := &completionStreamErrorWriter{failAt: 1}
	stream := Adapter{}.Stream(writer, parsedRequest{}, &meta, chat.DefaultLimits())
	err := stream.Event(chat.OutputItem(chat.MessageItem(chat.RoleAssistant, chat.TextPart("x"))))
	require.ErrorIs(t, err, chat.ErrDelivery)

	writer = &completionStreamErrorWriter{failAt: 1}
	stream = Adapter{}.Stream(writer, parsedRequest{}, &meta, chat.DefaultLimits())
	err = stream.Event(chat.OutputItem(chat.MessageItem(chat.RoleAssistant, chat.AudioPart(chat.InlineMedia("audio/wav", []byte{1})))))
	require.ErrorIs(t, err, chat.ErrDelivery)

	writer = &completionStreamErrorWriter{failAt: 1}
	stream = Adapter{}.Stream(writer, parsedRequest{IncludeUsage: true}, &meta, chat.DefaultLimits())
	err = stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil)
	require.ErrorIs(t, err, chat.ErrDelivery)

	writer = &completionStreamErrorWriter{failAt: 3}
	stream = Adapter{}.Stream(writer, parsedRequest{IncludeUsage: true}, &meta, chat.DefaultLimits())
	require.NoError(t, stream.Event(chat.TextDelta("x")))
	err = stream.Complete(chat.Outcome{Status: chat.StatusCompleted, Usage: &chat.Usage{Input: 1, Output: 1, Total: 2}}, nil)
	require.ErrorIs(t, err, chat.ErrDelivery)

	writer = &completionStreamErrorWriter{failAt: 2}
	stream = Adapter{}.Stream(writer, parsedRequest{}, &meta, chat.DefaultLimits())
	require.NoError(t, stream.Event(chat.TextDelta("x")))
	err = stream.Fail(errors.New("backend"))
	require.ErrorIs(t, err, chat.ErrDelivery)
}
