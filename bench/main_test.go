package main

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kelindar/llmux"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestServeHelper(t *testing.T) {
	agent := llmux.AgentFunc(func(_ context.Context, _ *llmux.Request, emit llmux.Emit) (llmux.Outcome, error) {
		return llmux.Outcome{}, llmux.EmitText(emit, "ok")
	})
	handler := llmux.New(llmux.ResolverFunc(func(context.Context, string) (llmux.Agent, llmux.Capabilities, error) {
		return agent, llmux.Capabilities{}, nil
	}))
	body := []byte(`{"model":"bench","messages":[{"role":"user","content":"hello"}]}`)
	recorder := serve(handler, "/v1/chat/completions", body, "")
	require.Equal(t, http.StatusOK, recorder.Code)
}

func TestServeModels(t *testing.T) {
	handler := llmux.New(llmux.ResolverFunc(func(context.Context, string) (llmux.Agent, llmux.Capabilities, error) {
		return llmux.AgentFunc(func(context.Context, *llmux.Request, llmux.Emit) (llmux.Outcome, error) {
			return llmux.Outcome{}, nil
		}), llmux.Capabilities{}, nil
	}), llmux.WithModels(llmux.Model{ID: "bench"}))
	recorder := serve(handler, "/v1/models", nil, "GET")
	require.Equal(t, http.StatusOK, recorder.Code)
}

func TestServeTranscription(t *testing.T) {
	handler := llmux.New(llmux.ResolverFunc(func(context.Context, string) (llmux.Agent, llmux.Capabilities, error) {
		return llmux.AgentFunc(func(context.Context, *llmux.Request, llmux.Emit) (llmux.Outcome, error) {
			return llmux.Outcome{}, nil
		}), llmux.Capabilities{}, nil
	}), llmux.WithTranscriber(llmux.TranscriberFunc(func(context.Context, llmux.TranscriptionRequest) (llmux.Transcription, error) {
		return llmux.Transcription{Text: "ok"}, nil
	})))
	recorder := serveTranscription(handler)
	require.Equal(t, http.StatusOK, recorder.Code)
}

func TestMainPackageBuild(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{"model":"bench","messages":[{"role":"user","content":"hello"}]}`)))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	handler := llmux.New(llmux.ResolverFunc(func(context.Context, string) (llmux.Agent, llmux.Capabilities, error) {
		return llmux.AgentFunc(func(_ context.Context, _ *llmux.Request, emit llmux.Emit) (llmux.Outcome, error) {
			return llmux.Outcome{}, llmux.EmitText(emit, "ok")
		}), llmux.Capabilities{}, nil
	}))
	handler.ServeHTTP(recorder, request)
	assert.Equal(t, http.StatusOK, recorder.Code)
}
