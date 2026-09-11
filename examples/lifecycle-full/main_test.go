package main

import (
	"context"
	"encoding/json/v2"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLifecycleFull(t *testing.T) {
	store := newStore()
	agent := chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		text := "turn"
		if len(req.Input) > 0 && len(req.Input[len(req.Input)-1].Content) > 0 {
			text = req.Input[len(req.Input)-1].Content[0].Text
		}
		return chat.Outcome{}, emit.Text("echo: " + text)
	})
	resolver := chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
		return agent, chat.Info{Continuation: true}, nil
	})
	mux := http.NewServeMux()
	handler := llmux.New(resolver, llmux.WithLifecycle(store.Accept), llmux.WithContinuationStore(store), llmux.WithStoreDefault(true))
	mux.Handle("/api/v1/", http.StripPrefix("/api/v1", handler))
	mux.HandleFunc("GET /api/v1/responses/{id}", func(w http.ResponseWriter, r *http.Request) {
		rec, ok := store.get(r.PathValue("id"))
		if !ok {
			http.NotFound(w, r)
			return
		}
		body, err := llmux.ResponsesBody(rec.Response)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.MarshalWrite(w, body)
	})

	first := post(t, mux, `{"model":"agent/basic","store":true,"metadata":{"k":"v"},"input":"one"}`, "k1")
	require.Equal(t, http.StatusOK, first.Code)
	id := responseID(t, first)
	assert.Contains(t, first.Body.String(), `"k":"v"`)

	replay := post(t, mux, `{"model":"agent/basic","store":true,"input":"one"}`, "k1")
	require.Equal(t, http.StatusOK, replay.Code)
	assert.Equal(t, id, responseID(t, replay))
	assert.Contains(t, replay.Body.String(), `"k":"v"`)

	second := post(t, mux, `{"model":"agent/basic","store":true,"previous_response_id":"`+id+`","input":"two"}`, "")
	require.Equal(t, http.StatusOK, second.Code)
	assert.Contains(t, second.Body.String(), "echo: two")
	id2 := responseID(t, second)

	store.mu.Lock()
	rec1 := store.byID[id]
	rec2 := store.byID[id2]
	store.mu.Unlock()
	require.Len(t, rec1.Turn, 1)
	require.Len(t, rec2.Turn, 1)
	assert.Equal(t, "one", rec1.Turn[0].Content[0].Text)
	assert.Equal(t, "two", rec2.Turn[0].Content[0].Text)
	require.NotNil(t, rec2.Response.Previous)
	assert.Equal(t, id, *rec2.Response.Previous)
	assert.Equal(t, chat.StatusCompleted, rec1.Response.Status)
	assert.True(t, rec1.Response.Store)

	history, err := store.Load(context.Background(), id2)
	require.NoError(t, err)
	require.Len(t, history, 4) // turn1, out1, turn2, out2

	get := httptest.NewRequest(http.MethodGet, "/api/v1/responses/"+id, nil)
	getRec := httptest.NewRecorder()
	mux.ServeHTTP(getRec, get)
	require.Equal(t, http.StatusOK, getRec.Code)
	assert.Equal(t, id, responseID(t, getRec))
	assert.Contains(t, getRec.Body.String(), `"k":"v"`)
	assert.Contains(t, getRec.Body.String(), "echo: one")
}

func TestIdempotencyKey(t *testing.T) {
	store := newStore()
	calls := 0
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		calls++
		return chat.Outcome{}, emit.Text("ok")
	})
	mux := http.NewServeMux()
	handler := llmux.New(
		chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
			return agent, chat.Info{Continuation: true}, nil
		}),
		llmux.WithLifecycle(store.Accept),
		llmux.WithContinuationStore(store),
		llmux.WithStoreDefault(true),
	)
	mux.Handle("/api/v1/", http.StripPrefix("/api/v1", handler))

	t.Run("reject leaves no reservation", func(t *testing.T) {
		rejected := post(t, mux, `{"model":"agent/basic","store":false,"input":"no"}`, "same-key")
		require.Equal(t, http.StatusBadRequest, rejected.Code)
		assert.Equal(t, 0, calls)
		store.mu.Lock()
		assert.False(t, store.running["same-key"])
		store.mu.Unlock()
	})

	t.Run("same key then accepts", func(t *testing.T) {
		accepted := post(t, mux, `{"model":"agent/basic","store":true,"input":"yes"}`, "same-key")
		require.Equal(t, http.StatusOK, accepted.Code)
		assert.Equal(t, 1, calls)
		assert.Contains(t, accepted.Body.String(), "ok")
	})
}

func TestDurableExample(t *testing.T) {
	store := newStore()
	agent := chat.AgentFunc(func(ctx context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		deadline, ok := ctx.Deadline()
		require.True(t, ok)
		assert.True(t, time.Until(deadline) > 0)
		return chat.Outcome{}, emit.Text("durable")
	})
	mux := http.NewServeMux()
	handler := llmux.New(
		chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
			return agent, chat.Info{Continuation: true, Extensions: map[string]bool{"x-durable": true}}, nil
		}),
		llmux.WithLifecycle(store.Accept),
		llmux.WithContinuationStore(store),
		llmux.WithStoreDefault(true),
	)
	mux.Handle("/api/v1/", http.StripPrefix("/api/v1", handler))

	rec := post(t, mux, `{"model":"agent/basic","store":true,"input":"x","x-durable":true}`, "")
	require.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), "durable")
}

func post(t *testing.T, h http.Handler, body, key string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/responses", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Idempotency-Key", key)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func responseID(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var body map[string]any
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	id, ok := body["id"].(string)
	require.True(t, ok)
	require.NotEmpty(t, id)
	return id
}
