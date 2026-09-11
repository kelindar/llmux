// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package llmux

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRouteBodyErrors(t *testing.T) {
	agent := chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, errors.New("unreachable")
	})
	handler := testHandler(agent, chat.Info{}, WithLimits(chat.Limits{MaxRequestBytes: 8}))

	cases := []struct {
		name    string
		path    string
		body    string
		headers map[string]string
	}{
		{name: "chat", path: "/chat/completions", body: strings.Repeat("x", 32)},
		{name: "responses", path: "/responses", body: strings.Repeat("x", 32)},
		{name: "anthropic", path: "/messages", body: strings.Repeat("x", 32), headers: map[string]string{"anthropic-version": "2023-06-01"}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			recorder := postJSON(t, handler, test.path, test.body, test.headers)
			assert.Equal(t, http.StatusRequestEntityTooLarge, recorder.Code)
			if test.name == "anthropic" {
				body := decodeResponse(t, recorder)
				errorValue := body["error"].(map[string]any)
				assert.Contains(t, errorValue["message"], "limit")
			} else {
				assert.Equal(t, "request_too_large", responseError(t, recorder)["code"])
			}
		})
	}
}

func TestRouteInvalidJSON(t *testing.T) {
	agent := chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, errors.New("unreachable")
	})
	handler := testHandler(agent, chat.Info{})

	cases := []struct {
		name    string
		path    string
		headers map[string]string
	}{
		{name: "chat", path: "/chat/completions"},
		{name: "responses", path: "/responses"},
		{name: "anthropic", path: "/messages", headers: map[string]string{"anthropic-version": "2023-06-01"}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			recorder := postJSON(t, handler, test.path, `{`, test.headers)
			assert.Equal(t, http.StatusBadRequest, recorder.Code)
		})
	}
}

func TestAnthropicVersionRequired(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, errors.New("unreachable")
	}), chat.Info{})
	recorder := postJSON(t, handler, "/messages", `{"model":"agent/basic","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	body := decodeResponse(t, recorder)
	assert.Equal(t, "error", body["type"])
}

func TestDirectMount(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("ok"))
	}), chat.Info{})
	recorder := postJSON(t, handler, "/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
}

func TestStripPrefixMount(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("ok"))
	}), chat.Info{})
	mounted := http.StripPrefix("/v1", handler)
	recorder := postJSON(t, mounted, "/v1/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
}

func TestWrongPrefix404(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{})
	recorder := postJSON(t, handler, "/v1/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusNotFound, recorder.Code)
}

func TestRouteParseErrors(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, errors.New("unreachable")
	}), chat.Info{})

	cases := []struct {
		name    string
		path    string
		body    string
		headers map[string]string
	}{
		{name: "chat", path: "/chat/completions", body: `{"model":"agent/basic"}`},
		{name: "responses", path: "/responses", body: `{"model":"agent/basic"}`},
		{name: "anthropic", path: "/messages", body: `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, headers: map[string]string{"anthropic-version": "2023-06-01"}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			recorder := postJSON(t, handler, test.path, test.body, test.headers)
			assert.Equal(t, http.StatusBadRequest, recorder.Code)
		})
	}
}
