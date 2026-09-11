package llmux

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCatalogNil(t *testing.T) {
	handler := New(nil)
	req := httptest.NewRequest(http.MethodGet, "/models", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	require.Equal(t, http.StatusOK, rec.Code)
	assert.Empty(t, decodeResponse(t, rec)["data"])

	rec = postJSON(t, handler, "/chat/completions", `{"model":"x","messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusInternalServerError, rec.Code)
}

func TestCatalogLoadOnly(t *testing.T) {
	var lists, loads atomic.Int32
	catalog := &fixedCatalog{
		listFn: func(context.Context) (map[string]chat.Info, error) {
			lists.Add(1)
			return map[string]chat.Info{"agent/a": {Tool: "a"}}, nil
		},
		loadFn: func(_ context.Context, target string) (chat.Agent, chat.Info, error) {
			loads.Add(1)
			assert.Equal(t, "agent/a", target)
			return chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{}, emit(chat.Text("ok"))
			}), chat.Info{Tools: true}, nil
		},
	}
	handler := New(catalog)
	rec := postJSON(t, handler, "/chat/completions", `{"model":"agent/a","messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusOK, rec.Code)
	assert.Equal(t, int32(0), lists.Load(), "chat invocation must not call List")
	assert.Equal(t, int32(1), loads.Load())
}

func TestStoreOptional(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{Continuation: true})
	rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"hi"}`, nil)
	require.Equal(t, http.StatusBadRequest, rec.Code)
	assert.Equal(t, "unsupported", responseError(t, rec)["code"])

	rec = postJSON(t, handler, "/responses", `{"model":"agent/basic","previous_response_id":"x","input":"hi"}`, nil)
	require.Equal(t, http.StatusBadRequest, rec.Code)
}

func TestStoreOptionOrder(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("ok"))
	})
	store := acceptStore{accept: func(context.Context, *chat.TurnRequest) (chat.Acceptance, error) {
		return chat.Acceptance{Response: chat.Response{ID: "resp_1"}}, nil
	}}
	for _, options := range [][]Option{
		{WithStore(store), WithStoreDefault(true), WithMCP()},
		{WithMCP(), WithStoreDefault(true), WithStore(store)},
	} {
		handler := New(&fixedCatalog{agent: agent, info: chat.Info{Continuation: true}}, options...)
		require.NotNil(t, handler.mcp)
		require.NotNil(t, handler.store)
		rec := postJSON(t, handler, "/responses", `{"model":"x","input":"hi"}`, nil)
		require.Equal(t, http.StatusOK, rec.Code)
	}
}
