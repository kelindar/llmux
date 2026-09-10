package main

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestServeHelper(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("ok"))
	})
	handler := llmux.New(chat.Resolver(func(context.Context, string) (chat.Agent, chat.Capabilities, error) {
		return agent, chat.Capabilities{}, nil
	}))
	body := []byte(`{"model":"bench","messages":[{"role":"user","content":"hello"}]}`)
	recorder := serve(handler, "/chat/completions", body, "")
	require.Equal(t, http.StatusOK, recorder.Code)
}

func TestServeModels(t *testing.T) {
	handler := llmux.New(chat.Resolver(func(context.Context, string) (chat.Agent, chat.Capabilities, error) {
		return chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, nil
		}), chat.Capabilities{}, nil
	}), llmux.WithModels(chat.Model{ID: "bench"}))
	recorder := serve(handler, "/models", nil, "GET")
	require.Equal(t, http.StatusOK, recorder.Code)
}

func TestServeTranscription(t *testing.T) {
	handler := llmux.New(chat.Resolver(func(context.Context, string) (chat.Agent, chat.Capabilities, error) {
		return chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, nil
		}), chat.Capabilities{}, nil
	}), llmux.WithTranscriber(llmux.TranscriberFunc(func(context.Context, llmux.TranscriptionRequest) (llmux.Transcription, error) {
		return llmux.Transcription{Text: "ok"}, nil
	})))
	recorder := serveTranscription(handler)
	require.Equal(t, http.StatusOK, recorder.Code)
}

func TestMainPackageBuild(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/chat/completions", bytes.NewReader([]byte(`{"model":"bench","messages":[{"role":"user","content":"hello"}]}`)))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	handler := llmux.New(chat.Resolver(func(context.Context, string) (chat.Agent, chat.Capabilities, error) {
		return chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("ok"))
		}), chat.Capabilities{}, nil
	}))
	handler.ServeHTTP(recorder, request)
	assert.Equal(t, http.StatusOK, recorder.Code)
}
