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
		for _, part := range req.Input[0].Content {
			if part.Type == llmux.PartImage && part.Media != nil {
				return llmux.Outcome{}, llmux.EmitText(emit, "received "+part.Media.MIMEType)
			}
		}
		return llmux.Outcome{}, llmux.EmitText(emit, "no image")
	})
	resolver := llmux.ResolverFunc(func(context.Context, string) (llmux.Agent, llmux.Capabilities, error) {
		return agent, llmux.Capabilities{InputModalities: llmux.ModalityText | llmux.ModalityImage}, nil
	})
	handler := llmux.New(resolver)

	body := `{"model":"agent/vision","messages":[{"role":"user","content":[{"type":"text","text":"what?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}]}]}`
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Contains(t, recorder.Body.String(), "received image/png")
}
