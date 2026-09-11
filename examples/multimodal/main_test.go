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

func TestHandlerBuild(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		for _, part := range req.Input[0].Content {
			if part.Type == chat.PartImage && part.Media != nil {
				return chat.Outcome{}, emit(chat.Text("received " + part.Media.MIMEType))
			}
		}
		return chat.Outcome{}, emit(chat.Text("no image"))
	})
	handler := llmux.New(&agents{vision: agent})

	body := `{"model":"vision","messages":[{"role":"user","content":[{"type":"text","text":"what?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}}]}]}`
	request := httptest.NewRequest(http.MethodPost, "/chat/completions", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Contains(t, recorder.Body.String(), "received image/png")
}
