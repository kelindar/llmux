package main

import (
	"context"
	"encoding/json/v2"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLifecycleExample(t *testing.T) {
	store := newStore()
	agent := chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		text := "turn"
		if len(req.Turn) > 0 && len(req.Turn[0].Content) > 0 {
			text = req.Turn[0].Content[0].Text
		}
		return chat.Outcome{}, emit(chat.Text("echo: " + text))
	})
	resolver := chat.Resolver(func(context.Context, string) (chat.Agent, chat.Capabilities, error) {
		return agent, chat.Capabilities{Continuation: true}, nil
	})
	mux := http.NewServeMux()
	handler := llmux.New(resolver, llmux.WithLifecycle(store), llmux.WithContinuationStore(store), llmux.WithStoreDefault(true))
	mux.Handle("/api/v1/", http.StripPrefix("/api/v1", handler))
	mux.HandleFunc("GET /api/v1/responses/{id}", func(w http.ResponseWriter, r *http.Request) {
		rec, ok := store.get(r.PathValue("id"))
		if !ok {
			http.NotFound(w, r)
			return
		}
		body, err := llmux.ResponsesBody(*rec.Request, rec.State, rec.ID, rec.Created)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.MarshalWrite(w, body)
	})

	first := post(t, mux, `{"model":"agent/basic","store":true,"input":"one"}`, "k1")
	require.Equal(t, http.StatusOK, first.Code)
	id := responseID(t, first)

	replay := post(t, mux, `{"model":"agent/basic","store":true,"input":"one"}`, "k1")
	require.Equal(t, http.StatusOK, replay.Code)
	assert.Equal(t, id, responseID(t, replay))

	second := post(t, mux, `{"model":"agent/basic","store":true,"previous_response_id":"`+id+`","input":"two"}`, "")
	require.Equal(t, http.StatusOK, second.Code)
	assert.Contains(t, second.Body.String(), "echo: two")

	get := httptest.NewRequest(http.MethodGet, "/api/v1/responses/"+id, nil)
	getRec := httptest.NewRecorder()
	mux.ServeHTTP(getRec, get)
	require.Equal(t, http.StatusOK, getRec.Code)
	assert.Equal(t, id, responseID(t, getRec))
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
