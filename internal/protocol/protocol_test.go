// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package protocol

import (
	json "encoding/json/v2"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type noFlusherWriter struct {
	header http.Header
	code   int
	body   []byte
}

func (w *noFlusherWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}

func (w *noFlusherWriter) Write(b []byte) (int, error) {
	w.body = append(w.body, b...)
	return len(b), nil
}

func (w *noFlusherWriter) WriteHeader(statusCode int) {
	w.code = statusCode
}

type errorWriter struct {
	header http.Header
	code   int
}

func (w *errorWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}

func (w *errorWriter) Write([]byte) (int, error) {
	return 0, errors.New("write failed")
}

func (w *errorWriter) WriteHeader(statusCode int) {
	w.code = statusCode
}

func (w *errorWriter) Flush() {}

func TestSSEWriter(t *testing.T) {
	limits := chat.DefaultLimits()

	t.Run("startWriteDone", func(t *testing.T) {
		rec := httptest.NewRecorder()
		writer := NewSSEWriter(rec, limits)
		require.NoError(t, writer.Start())
		assert.True(t, writer.Started())
		require.NoError(t, writer.Write("ping", map[string]any{"ok": true}))
		require.NoError(t, writer.Done())
		body := rec.Body.String()
		assert.Equal(t, "text/event-stream", rec.Header().Get("Content-Type"))
		assert.Contains(t, body, "event: ping")
		assert.Contains(t, body, `"ok":true`)
		assert.Contains(t, body, "data: [DONE]")
	})

	t.Run("writeWithoutEventName", func(t *testing.T) {
		rec := httptest.NewRecorder()
		writer := NewSSEWriter(rec, limits)
		require.NoError(t, writer.Write("", map[string]any{"n": 1}))
		assert.Contains(t, rec.Body.String(), "data: {\"n\":1}")
		assert.NotContains(t, rec.Body.String(), "event:")
	})

	t.Run("requiresFlusher", func(t *testing.T) {
		rec := &noFlusherWriter{}
		writer := NewSSEWriter(rec, limits)
		err := writer.Start()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "Flusher")
	})

	t.Run("eventTooLarge", func(t *testing.T) {
		rec := httptest.NewRecorder()
		tiny := chat.Limits{MaxEventBytes: 8}
		writer := NewSSEWriter(rec, tiny)
		err := writer.Write("", map[string]any{"payload": "too large for limit"})
		require.Error(t, err)
		apiErr, ok := errors.AsType[*chat.Error](err)
		require.True(t, ok)
		assert.Equal(t, "event_too_large", apiErr.Code)
	})
}

func TestSSEWriterMore(t *testing.T) {
	limits := chat.DefaultLimits()

	t.Run("marshalError", func(t *testing.T) {
		writer := NewSSEWriter(httptest.NewRecorder(), limits)
		err := writer.Write("", make(chan int))
		require.Error(t, err)
		assert.False(t, writer.Started())
	})

	t.Run("eventWriteError", func(t *testing.T) {
		writer := NewSSEWriter(&errorWriter{}, limits)
		err := writer.Write("event", true)
		assert.ErrorIs(t, err, chat.ErrDelivery)
		assert.True(t, writer.Started())
	})

	t.Run("dataWriteError", func(t *testing.T) {
		writer := NewSSEWriter(&errorWriter{}, limits)
		err := writer.Write("", true)
		assert.ErrorIs(t, err, chat.ErrDelivery)
	})

	t.Run("doneWriteError", func(t *testing.T) {
		writer := NewSSEWriter(&errorWriter{}, limits)
		err := writer.Done()
		assert.ErrorIs(t, err, chat.ErrDelivery)
	})

	t.Run("jsonWriteError", func(t *testing.T) {
		rec := httptest.NewRecorder()
		WriteJSON(rec, http.StatusOK, make(chan int))
		assert.Empty(t, rec.Body.Bytes())
	})
}

func TestAsAPIError(t *testing.T) {
	cases := map[string]struct {
		err     error
		status  int
		typ     string
		code    string
		message string
		param   string
	}{
		"nil": {
			err:     nil,
			status:  http.StatusInternalServerError,
			typ:     "server_error",
			code:    "server_error",
			message: "internal server error",
		},
		"apiErrorPassthrough": {
			err:     chat.Invalid("model", "missing model"),
			status:  http.StatusBadRequest,
			typ:     "invalid_request_error",
			code:    "invalid_request",
			message: "missing model",
			param:   "model",
		},
		"plainErrorWrapped": {
			err:     errors.New("secret"),
			status:  http.StatusInternalServerError,
			typ:     "server_error",
			code:    "server_error",
			message: "internal server error",
		},
		"apiErrorDefaults": {
			err:     &chat.Error{Status: 0, Message: ""},
			status:  http.StatusInternalServerError,
			typ:     "server_error",
			code:    "server_error",
			message: "internal server error",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			apiErr := AsError(tc.err)
			require.NotNil(t, apiErr)
			assert.Equal(t, tc.status, apiErr.Status)
			assert.Equal(t, tc.typ, apiErr.Type)
			assert.Equal(t, tc.code, apiErr.Code)
			assert.Equal(t, tc.message, apiErr.Message)
			if tc.param != "" {
				assert.Equal(t, tc.param, apiErr.Param)
			}
		})
	}
}

func TestAnthropicErrorType(t *testing.T) {
	assert.Equal(t, "invalid_request_error", AnthropicErrorType(chat.Invalid("x", "y")))
	assert.Equal(t, "api_error", AnthropicErrorType(&chat.Error{Type: "server_error"}))
}

func TestWriteError(t *testing.T) {
	cases := map[string]struct {
		kind     Kind
		err      error
		status   int
		contains []string
	}{
		"chat": {
			kind:     Chat,
			err:      chat.Invalid("model", "bad"),
			status:   http.StatusBadRequest,
			contains: []string{`"invalid_request_error"`, `"model"`, `"bad"`},
		},
		"responses": {
			kind:     Responses,
			err:      chat.Unsupported("tools", "nope"),
			status:   http.StatusBadRequest,
			contains: []string{`"unsupported"`, `"tools"`},
		},
		"anthropic": {
			kind:     Anthropic,
			err:      chat.Invalid("max_tokens", "required"),
			status:   http.StatusBadRequest,
			contains: []string{`"type":"error"`, `"invalid_request_error"`, `"required"`},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			WriteError(rec, tc.kind, tc.err)
			assert.Equal(t, tc.status, rec.Code)
			assert.Equal(t, "application/json", rec.Header().Get("Content-Type"))
			body := rec.Body.String()
			for _, part := range tc.contains {
				assert.Contains(t, body, part)
			}
			var decoded map[string]any
			require.NoError(t, json.Unmarshal([]byte(body), &decoded))
		})
	}
}

func TestWriteJSON(t *testing.T) {
	rec := httptest.NewRecorder()
	WriteJSON(rec, http.StatusCreated, map[string]any{"ok": true})
	assert.Equal(t, http.StatusCreated, rec.Code)
	assert.Contains(t, rec.Body.String(), `"ok":true`)
}

func TestParsedRequestKind(t *testing.T) {
	assert.Equal(t, Kind(1), Chat)
	assert.Equal(t, Kind(2), Responses)
	assert.Equal(t, Kind(3), Anthropic)
}

func TestInitialResponse(t *testing.T) {
	previous := "resp-parent"
	parsed := &ParsedRequest{
		Metadata: map[string]string{"tenant": "one"},
		Previous: &previous,
		Retain:   true,
		Request:  chat.Request{Target: "agent", Instructions: "be brief"},
	}
	resp := InitialResponse(chat.Response{}, parsed)
	require.NotEmpty(t, resp.ID)
	assert.NotZero(t, resp.Created)
	assert.Equal(t, "one", resp.Metadata["tenant"])
	assert.True(t, resp.Store)
	assert.Equal(t, "agent", resp.Target)
	assert.Equal(t, "be brief", resp.Instructions)
	require.NotNil(t, resp.Previous)
	assert.Equal(t, previous, *resp.Previous)

	parsed.Metadata["tenant"] = "two"
	assert.Equal(t, "one", resp.Metadata["tenant"])

	seed := chat.Response{ID: "resp-existing", Created: 42, Metadata: map[string]string{"seed": "yes"}}
	resp = InitialResponse(seed, &ParsedRequest{Request: chat.Request{Target: "other"}})
	assert.Equal(t, "resp-existing", resp.ID)
	assert.Equal(t, int64(42), resp.Created)
	assert.Equal(t, "yes", resp.Metadata["seed"])
	assert.Nil(t, resp.Previous)
}

func TestOutputValidation(t *testing.T) {
	text := chat.TextPart("text")
	image := chat.ImagePart(chat.RemoteMedia("image/png", "https://example.com/image.png"))
	audio := chat.AudioPart(chat.InlineMedia("audio/wav", []byte{1}))
	file := chat.FilePart(chat.AssetMedia("application/pdf", "file-1"))

	t.Run("agentOutput", func(t *testing.T) {
		cases := map[string]struct {
			event   chat.Event
			info    chat.Info
			wantErr bool
		}{
			"defaultText": {
				event: chat.OutputItem(chat.MessageItem(chat.RoleAssistant, text)),
			},
			"textRejected": {
				event:   chat.OutputItem(chat.MessageItem(chat.RoleAssistant, text)),
				info:    chat.Info{OutputModalities: chat.ModalityImage},
				wantErr: true,
			},
			"imageAccepted": {
				event: chat.OutputItem(chat.MediaItem(image).Item),
				info:  chat.Info{OutputModalities: chat.ModalityImage},
			},
			"imageRejected": {
				event:   chat.OutputItem(chat.MediaItem(image).Item),
				info:    chat.Info{},
				wantErr: true,
			},
			"audioRejected": {
				event:   chat.OutputItem(chat.MessageItem(chat.RoleAssistant, audio)),
				info:    chat.Info{OutputModalities: chat.ModalityText},
				wantErr: true,
			},
			"fileRejected": {
				event:   chat.OutputItem(chat.MessageItem(chat.RoleAssistant, file)),
				info:    chat.Info{OutputModalities: chat.ModalityText},
				wantErr: true,
			},
			"toolRejected": {
				event:   chat.OutputItem(chat.FunctionCallItem("call-1", "tool", `{}`)),
				info:    chat.Info{},
				wantErr: true,
			},
			"toolAccepted": {
				event: chat.OutputItem(chat.FunctionCallItem("call-1", "tool", `{}`)),
				info:  chat.Info{ClientTools: true},
			},
			"outputRejected": {
				event:   chat.OutputItem(chat.Item{Type: chat.ItemFunctionCallOutput, Output: []chat.Part{image}}),
				info:    chat.Info{OutputModalities: chat.ModalityText},
				wantErr: true,
			},
			"reasoningRejected": {
				event:   chat.Reasoning("thinking"),
				info:    chat.Info{},
				wantErr: true,
			},
			"reasoningAccepted": {
				event: chat.Reasoning("thinking"),
				info:  chat.Info{ReasoningSummary: true},
			},
			"textDeltaRejected": {
				event:   chat.TextDelta("delta"),
				info:    chat.Info{OutputModalities: chat.ModalityImage},
				wantErr: true,
			},
			"toolEventRejected": {
				event:   chat.ToolStart("call-1", "tool"),
				info:    chat.Info{},
				wantErr: true,
			},
			"toolEventAccepted": {
				event: chat.ToolDelta("call-1", `{}`),
				info:  chat.Info{ClientTools: true},
			},
			"unknownEvent": {event: chat.Event{Type: "unknown"}},
		}
		for name, tc := range cases {
			t.Run(name, func(t *testing.T) {
				err := ValidateOutputEvent(tc.event, tc.info)
				if tc.wantErr {
					require.Error(t, err)
					return
				}
				require.NoError(t, err)
			})
		}
	})

	t.Run("requestedOutput", func(t *testing.T) {
		cases := map[string]struct {
			event     chat.Event
			requested chat.Modality
			wantErr   bool
		}{
			"text": {
				event:     chat.OutputItem(chat.MessageItem(chat.RoleAssistant, text)),
				requested: chat.ModalityText,
			},
			"image": {
				event:     chat.OutputItem(chat.MediaItem(image).Item),
				requested: chat.ModalityImage,
			},
			"audioRejected": {
				event:     chat.OutputItem(chat.MessageItem(chat.RoleAssistant, audio)),
				requested: chat.ModalityText,
				wantErr:   true,
			},
			"fileRejected": {
				event:     chat.OutputItem(chat.MessageItem(chat.RoleAssistant, file)),
				requested: chat.ModalityText,
				wantErr:   true,
			},
			"output": {
				event:     chat.OutputItem(chat.Item{Type: chat.ItemFunctionCallOutput, Output: []chat.Part{image}}),
				requested: chat.ModalityImage,
			},
			"deltaRejected": {
				event:     chat.TextDelta("delta"),
				requested: chat.ModalityImage,
				wantErr:   true,
			},
			"toolEvent":    {event: chat.ToolDone("call-1")},
			"unknownEvent": {event: chat.Event{Type: "unknown"}},
		}
		for name, tc := range cases {
			t.Run(name, func(t *testing.T) {
				err := ValidateRequestedOutput(tc.event, tc.requested)
				if tc.wantErr {
					require.Error(t, err)
					return
				}
				require.NoError(t, err)
			})
		}
	})

	assert.True(t, RequiresReasoningSummary(chat.Reasoning("summary")))
	assert.False(t, RequiresReasoningSummary(chat.Text("text")))
}
