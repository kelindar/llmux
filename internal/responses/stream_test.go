// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package responses

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStreamStart(t *testing.T) {
	rec := httptest.NewRecorder()
	meta := responseMeta{Response: chat.Response{ID: "r1", Created: 1, Target: "m"}}
	stream := Adapter{}.Stream(rec, parsedRequest{}, &meta, chat.DefaultLimits())
	require.NoError(t, stream.Event(chat.Text("hi")))
	require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("hi"))}))
	assert.Contains(t, rec.Body.String(), "response.created")
}

func TestStreamOutputPaths(t *testing.T) {
	meta := responseMeta{Response: chat.Response{ID: "r1", Created: 1, Target: "m"}}
	limits := chat.DefaultLimits()

	t.Run("messageParts", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := Adapter{}.Stream(rec, parsedRequest{}, &meta, limits)
		item := chat.MessageItem(chat.RoleAssistant, chat.TextPart("hello"), chat.TextPart(" world"))
		require.NoError(t, stream.Event(chat.OutputItem(item)))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, []chat.Item{item}))
		assert.Contains(t, rec.Body.String(), "response.content_part.added")
		assert.Contains(t, rec.Body.String(), "hello")
	})

	t.Run("emptyMessage", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := Adapter{}.Stream(rec, parsedRequest{}, &meta, limits)
		item := chat.MessageItem(chat.RoleAssistant)
		require.NoError(t, stream.Event(chat.OutputItem(item)))
	})

	t.Run("textEvents", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := Adapter{}.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(chat.Event{Type: chat.EventTextDelta, ItemID: "m1", Delta: "hel"}))
		require.NoError(t, stream.Event(chat.Event{Type: chat.EventTextDelta, ItemID: "m1", Delta: "lo"}))
		require.NoError(t, stream.Event(chat.Event{Type: chat.EventTextDone, ItemID: "m1"}))
		assert.Contains(t, rec.Body.String(), "response.output_text.done")
	})

	t.Run("reasoningWithEncryptedContent", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := Adapter{}.Stream(rec, parsedRequest{}, &meta, limits)
		item := chat.Item{Type: chat.ItemReasoning, ID: "rs1", EncryptedContent: []byte(`"opaque"`), Summary: []chat.Part{chat.SummaryPart("brief")}}
		require.NoError(t, stream.Event(chat.OutputItem(item)))
		assert.Contains(t, rec.Body.String(), "response.reasoning_summary_text.delta")
	})

	t.Run("activity", func(t *testing.T) {
		rec := httptest.NewRecorder()
		active := meta
		active.Activity = true
		stream := Adapter{}.Stream(rec, parsedRequest{}, &active, limits)
		require.NoError(t, stream.Event(chat.Activity("progress", []byte(`{"step":1}`))))
		assert.Contains(t, rec.Body.String(), "response.activity.progress")
	})
}

func TestStreamRejects(t *testing.T) {
	meta := responseMeta{Response: chat.Response{ID: "r1", Created: 1, Target: "m"}}
	limits := chat.DefaultLimits()
	cases := map[string]chat.Event{
		"unknown":          {Type: "unknown"},
		"textDoneUnknown":  {Type: chat.EventTextDone, ItemID: "missing"},
		"toolDeltaUnknown": {Type: chat.EventToolCallDelta, ItemID: "missing", Delta: "{}"},
		"toolDoneUnknown":  {Type: chat.EventToolCallDone, ItemID: "missing", CallID: "missing"},
		"activityDisabled": {Type: chat.EventActivity, Name: "progress", Data: []byte(`{}`)},
		"activityBadName":  {Type: chat.EventActivity, Name: "bad/name", Data: []byte(`{}`)},
		"badMessagePart":   {Type: chat.EventItem, Item: chat.MessageItem(chat.RoleAssistant, chat.ImagePart(chat.InlineMedia("image/png", []byte{1})))},
		"badImage":         {Type: chat.EventItem, Item: chat.Item{Type: chat.ItemMedia, Content: []chat.Part{chat.ImagePart(chat.RemoteMedia("image/png", "https://example.com/a.png"))}}},
	}
	for name, event := range cases {
		t.Run(name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			streamMeta := meta
			if event.Type == chat.EventActivity && name == "activityBadName" {
				streamMeta.Activity = true
			}
			stream := Adapter{}.Stream(rec, parsedRequest{}, &streamMeta, limits)
			require.Error(t, stream.Event(event))
		})
	}
}

func TestIncrementalToolContract(t *testing.T) {
	rec := httptest.NewRecorder()
	meta := responseMeta{Response: chat.Response{ID: "r1", Created: 1, Target: "m"}}
	stream := Adapter{}.Stream(rec, parsedRequest{}, &meta, chat.DefaultLimits())

	require.NoError(t, stream.Event(chat.Event{Type: chat.EventToolCallStart, ItemID: "item-1", CallID: "call-1", Name: "lookup"}))
	require.NoError(t, stream.Event(chat.Event{Type: chat.EventToolCallDelta, ItemID: "item-1", Delta: `{"q":"x"}`}))
	require.NoError(t, stream.Event(chat.Event{Type: chat.EventToolCallDone, ItemID: "item-1"}))

	body := rec.Body.String()
	assert.Contains(t, body, `"type":"response.function_call_arguments.done"`)
	assert.Contains(t, body, `"arguments":"{\"q\":\"x\"}"`)
	assert.Contains(t, body, `"call_id":"call-1"`)
	assert.Contains(t, body, `"name":"lookup"`)
}

type responsesErrorWriter struct {
	header http.Header
	writes int
	failAt int
}

func (w *responsesErrorWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}

func (w *responsesErrorWriter) Write(data []byte) (int, error) {
	w.writes++
	if w.writes == w.failAt {
		return 0, errors.New("write failed")
	}
	return len(data), nil
}

func (w *responsesErrorWriter) WriteHeader(int) {}

func (w *responsesErrorWriter) Flush() {}

func TestStreamDeliveryErrors(t *testing.T) {
	limits := chat.DefaultLimits()
	cases := []struct {
		name     string
		event    chat.Event
		activity bool
		writes   int
	}{
		{name: "message", event: chat.Text("hello"), writes: 8},
		{name: "tool", event: chat.Tool("call_1", "search", `{}`), writes: 6},
		{name: "reasoning", event: chat.Reasoning("think"), writes: 6},
		{name: "image", event: chat.MediaItem(chat.ImagePart(chat.InlineMedia("image/png", []byte{1}))), writes: 4},
		{name: "activity", event: chat.Activity("progress", []byte(`{}`)), activity: true, writes: 3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for failAt := 1; failAt <= tc.writes; failAt++ {
				meta := responseMeta{Response: chat.Response{ID: "r1", Created: 1, Target: "m"}, Activity: tc.activity}
				writer := &responsesErrorWriter{failAt: failAt}
				stream := Adapter{}.Stream(writer, parsedRequest{}, &meta, limits)
				require.Error(t, stream.Event(tc.event), "failed write %d", failAt)
			}
		})
	}

	for _, status := range []chat.Status{chat.StatusFailed, chat.StatusCancelled, chat.StatusIncomplete} {
		t.Run(string(status), func(t *testing.T) {
			meta := responseMeta{Response: chat.Response{ID: "r1", Created: 1, Target: "m"}}
			rec := httptest.NewRecorder()
			stream := Adapter{}.Stream(rec, parsedRequest{}, &meta, limits)
			require.NoError(t, stream.Complete(chat.Outcome{Status: status}, nil))
			assert.Contains(t, rec.Body.String(), "response.")
		})
	}

	for failAt := 1; failAt <= 4; failAt++ {
		meta := responseMeta{Response: chat.Response{ID: "r1", Created: 1, Target: "m"}}
		writer := &responsesErrorWriter{failAt: failAt}
		stream := Adapter{}.Stream(writer, parsedRequest{}, &meta, limits)
		require.Error(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil), "failed write %d", failAt)
	}
}

func TestTextDelivery(t *testing.T) {
	for _, failAt := range []int{3, 4} {
		t.Run(fmt.Sprintf("text delta write %d", failAt), func(t *testing.T) {
			meta := responseMeta{Response: chat.Response{ID: "r1", Created: 1, Target: "m"}}
			writer := &responsesErrorWriter{failAt: failAt}
			stream := Adapter{}.Stream(writer, parsedRequest{}, &meta, chat.DefaultLimits())
			err := stream.Event(chat.Event{Type: chat.EventTextDelta, ItemID: "item-1", Delta: "hello"})
			require.ErrorIs(t, err, chat.ErrDelivery)
		})
	}

	t.Run("text done", func(t *testing.T) {
		meta := responseMeta{Response: chat.Response{ID: "r1", Created: 1, Target: "m"}}
		writer := &responsesErrorWriter{failAt: 6}
		stream := Adapter{}.Stream(writer, parsedRequest{}, &meta, chat.DefaultLimits())
		require.NoError(t, stream.Event(chat.Event{Type: chat.EventTextDelta, ItemID: "item-1", Delta: "hello"}))
		require.ErrorIs(t, stream.Event(chat.Event{Type: chat.EventTextDone, ItemID: "item-1"}), chat.ErrDelivery)
	})
}

func TestFailDeliveryError(t *testing.T) {
	meta := responseMeta{Response: chat.Response{ID: "r1", Created: 1, Target: "m"}}
	writer := &responsesErrorWriter{failAt: 9}
	stream := Adapter{}.Stream(writer, parsedRequest{}, &meta, chat.DefaultLimits())
	require.NoError(t, stream.Event(chat.Text("hello")))
	require.ErrorIs(t, stream.Fail(errors.New("upstream failed")), chat.ErrDelivery)
}
