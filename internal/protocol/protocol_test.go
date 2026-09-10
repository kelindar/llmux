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
			apiErr := AsAPIError(tc.err)
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
