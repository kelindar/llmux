// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
package main

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/audio"
	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestServeHelper(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("ok"))
	})
	handler := llmux.New(benchCatalog{agent: agent})
	body := []byte(`{"model":"bench","messages":[{"role":"user","content":"hello"}]}`)
	recorder := serve(handler, "/chat/completions", body, "")
	require.Equal(t, http.StatusOK, recorder.Code)
}

func TestServeModels(t *testing.T) {
	handler := llmux.New(benchCatalog{agent: chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	})})
	recorder := serve(handler, "/models", nil, "GET")
	require.Equal(t, http.StatusOK, recorder.Code)
}

func TestServeTranscription(t *testing.T) {
	handler := llmux.New(benchCatalog{agent: chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	})}, llmux.WithTranscriber(audio.TranscriberFunc(func(context.Context, audio.TranscriptionRequest) (audio.Transcription, error) {
		return audio.Transcription{Text: "ok"}, nil
	})))
	recorder := serveTranscription(handler)
	require.Equal(t, http.StatusOK, recorder.Code)
}

func TestMainPackageBuild(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/chat/completions", bytes.NewReader([]byte(`{"model":"bench","messages":[{"role":"user","content":"hello"}]}`)))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	handler := llmux.New(benchCatalog{agent: chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("ok"))
	})})
	handler.ServeHTTP(recorder, request)
	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Contains(t, recorder.Body.String(), "ok")
}
