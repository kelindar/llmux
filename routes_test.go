package llmux

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRouteBodyErrors(t *testing.T) {
	agent := AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, errors.New("unreachable")
	})
	handler := testHandler(agent, Capabilities{}, WithLimits(Limits{MaxRequestBytes: 8}))

	cases := []struct {
		name    string
		path    string
		body    string
		headers map[string]string
	}{
		{name: "chat", path: "/v1/chat/completions", body: strings.Repeat("x", 32)},
		{name: "responses", path: "/v1/responses", body: strings.Repeat("x", 32)},
		{name: "anthropic", path: "/v1/messages", body: strings.Repeat("x", 32), headers: map[string]string{"anthropic-version": "2023-06-01"}},
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
	agent := AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, errors.New("unreachable")
	})
	handler := testHandler(agent, Capabilities{})

	cases := []struct {
		name    string
		path    string
		headers map[string]string
	}{
		{name: "chat", path: "/v1/chat/completions"},
		{name: "responses", path: "/v1/responses"},
		{name: "anthropic", path: "/v1/messages", headers: map[string]string{"anthropic-version": "2023-06-01"}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			recorder := postJSON(t, handler, test.path, `{`, test.headers)
			assert.Equal(t, http.StatusBadRequest, recorder.Code)
		})
	}
}

func TestAnthropicVersionRequired(t *testing.T) {
	handler := testHandler(AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, errors.New("unreachable")
	}), Capabilities{})
	recorder := postJSON(t, handler, "/v1/messages", `{"model":"agent/basic","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	body := decodeResponse(t, recorder)
	assert.Equal(t, "error", body["type"])
}

func TestRouteParseErrors(t *testing.T) {
	handler := testHandler(AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, errors.New("unreachable")
	}), Capabilities{})

	cases := []struct {
		name    string
		path    string
		body    string
		headers map[string]string
	}{
		{name: "chat", path: "/v1/chat/completions", body: `{"model":"agent/basic"}`},
		{name: "responses", path: "/v1/responses", body: `{"model":"agent/basic"}`},
		{name: "anthropic", path: "/v1/messages", body: `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, headers: map[string]string{"anthropic-version": "2023-06-01"}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			recorder := postJSON(t, handler, test.path, test.body, test.headers)
			assert.Equal(t, http.StatusBadRequest, recorder.Code)
		})
	}
}
