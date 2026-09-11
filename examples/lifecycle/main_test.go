// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLifecycleMinimal(t *testing.T) {
	store := &store{byID: make(map[string]saved)}
	catalog := &agents{
		echo: chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit.Text("hello")
		}),
	}
	handler := llmux.New(catalog,
		llmux.WithStore(store),
		llmux.WithStoreDefault(true),
	)

	req := httptest.NewRequest(http.MethodPost, "/responses", strings.NewReader(`{"model":"echo","store":true,"input":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	require.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Body.String(), "hello")

	store.mu.Lock()
	defer store.mu.Unlock()
	require.Len(t, store.byID, 1)
	for _, got := range store.byID {
		assert.Equal(t, "echo", got.Response.Target)
		require.Len(t, got.Turn, 1)
		require.NotEmpty(t, got.Response.Output)
		assert.Equal(t, "hello", got.Response.Output[0].Content[0].Text)
	}
}
