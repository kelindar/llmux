// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package main

import (
	"bytes"
	"context"
	"encoding/json/jsontext"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/audio"
	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/anthropic"
	completions "github.com/kelindar/llmux/internal/completions"
	"github.com/kelindar/llmux/internal/execution"
	"github.com/kelindar/llmux/internal/responses"
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

func BenchmarkRequestPaths(b *testing.B) {
	chatBody := []byte(`{"model":"bench","messages":[{"role":"user","content":"hello"}]}`)
	responsesBody := []byte(`{"model":"bench","input":"hello"}`)
	anthropicBody := []byte(`{"model":"bench","max_tokens":32,"messages":[{"role":"user","content":"hello"}]}`)
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("ok"))
	})
	handler := llmux.New(benchCatalog{agent: agent})

	b.Run("chat/raw-parse", func(b *testing.B) {
		b.ReportAllocs()
		var err error
		for i := 0; i < b.N; i++ {
			keep, err = completions.ParseRequest(chatBody)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("chat/http", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			keep = serve(handler, "/chat/completions", chatBody, "")
		}
	})
	b.Run("responses/raw-parse", func(b *testing.B) {
		b.ReportAllocs()
		var err error
		for i := 0; i < b.N; i++ {
			keep, err = responses.ParseRequest(responsesBody)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("responses/http", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			keep = serve(handler, "/responses", responsesBody, "")
		}
	})
	b.Run("anthropic/raw-parse", func(b *testing.B) {
		b.ReportAllocs()
		var err error
		for i := 0; i < b.N; i++ {
			keep, err = anthropic.ParseRequest(anthropicBody)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("anthropic/http", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			keep = serve(handler, "/messages", anthropicBody, "anthropic-version: 2023-06-01")
		}
	})
}

// BenchmarkTransport keeps the existing text path comparable across adapter changes.
func BenchmarkTransport(b *testing.B) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit.Text("ok")
	})
	handler := llmux.New(benchCatalog{agent: agent})
	body := []byte(`{"model":"bench","stream":true,"input":"hello"}`)
	b.Run("text", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			keep = serve(handler, "/responses", body, "")
		}
	})
	ui := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.OutputItem(chat.Item{Type: chat.ItemExtension, Data: jsontext.Value(`{"kind":"ui","format":"a2ui","version":"0.9.1","catalogId":"example/v1","payload":[{"version":"v0.9.1","createSurface":{"surfaceId":"form","catalogId":"example/v1"}},{"version":"v0.9.1","updateComponents":{"surfaceId":"form","components":[{"id":"root","component":"Text","text":"Hello"}]}},{"version":"v0.9.1","updateDataModel":{"surfaceId":"form","value":{}}}]}`)}))
	})
	b.Run("ui-execution", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			result, err := execution.Run(context.Background(), &chat.Request{Target: "bench"}, ui, chat.DefaultLimits(), nil)
			if err != nil {
				b.Fatal(err)
			}
			keep = result
		}
	})
	for _, ui := range []bool{false, true} {
		handler, body := aguiWorkload(ui)
		name := "agui/text"
		if ui {
			name = "agui/ui"
		}
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				keep = serve(handler, "/ag-ui", body, "")
			}
		})
	}
}

func TestAGUIWorkload(t *testing.T) {
	store := benchStore{}
	items, err := store.Load(context.Background(), "response")
	require.NoError(t, err)
	assert.Empty(t, items)
	acceptance, err := store.Accept(context.Background(), &chat.TurnRequest{})
	require.NoError(t, err)
	require.NotNil(t, acceptance.Finish)
	assert.NoError(t, acceptance.Finish(context.Background(), &chat.Response{}, nil))
	for _, ui := range []bool{false, true} {
		handler, body := aguiWorkload(ui)
		rec := serve(handler, "/ag-ui", body, "")
		assert.Contains(t, rec.Body.String(), "RUN_FINISHED")
		assert.Equal(t, ui, bytes.Contains(rec.Body.Bytes(), []byte("llmux.item")))
	}
}
