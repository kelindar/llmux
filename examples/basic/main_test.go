package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/kelindar/llmux"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHandlerBuild(t *testing.T) {
	agent := llmux.AgentFunc(func(_ context.Context, req *llmux.Request, emit llmux.Emit) (llmux.Outcome, error) {
		return llmux.Outcome{}, llmux.EmitText(emit, "received "+req.Input[0].Content[0].Text)
	})
	resolver := llmux.ResolverFunc(func(context.Context, string) (llmux.Agent, llmux.Capabilities, error) {
		return agent, llmux.Capabilities{}, nil
	})
	handler := llmux.New(resolver)

	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Contains(t, recorder.Body.String(), "received hi")
}
