// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

// Package fixtures exercises llmux through the official OpenAI and Anthropic SDKs.
// JSON request samples in this directory are also used by the main module's protocol tests.
package fixtures

import (
	"context"
	"net/http"
	"net/http/httptest"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/chat"
)

type catalog struct {
	agent   chat.Agent
	info    chat.Info
	entries map[string]chat.Info
	loadFn  func(context.Context, string) (chat.Agent, chat.Info, error)
}

func (c *catalog) List(_ context.Context) (map[string]chat.Info, error) {
	if c.entries != nil {
		return c.entries, nil
	}
	return map[string]chat.Info{"agent/basic": c.info}, nil
}

func (c *catalog) Load(ctx context.Context, target string) (chat.Agent, chat.Info, error) {
	if c.loadFn != nil {
		return c.loadFn(ctx, target)
	}
	return c.agent, c.info, nil
}

func startServer(agent chat.Agent, info chat.Info, opts ...llmux.Option) *httptest.Server {
	return startServerCatalog(&catalog{agent: agent, info: info}, opts...)
}

func startServerEntries(agent chat.Agent, info chat.Info, entries map[string]chat.Info, opts ...llmux.Option) *httptest.Server {
	return startServerCatalog(&catalog{agent: agent, info: info, entries: entries}, opts...)
}

func startServerCatalog(cat llmux.Catalog, opts ...llmux.Option) *httptest.Server {
	return httptest.NewServer(http.StripPrefix("/v1", llmux.New(cat, opts...)))
}

func toolInfo() chat.Info {
	return chat.Info{Tools: true, ClientTools: true}
}

func echoAgent(text string) chat.Agent {
	return chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text(text))
	})
}

func captureAgent(fn func(*chat.Request)) chat.Agent {
	return chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		fn(req)
		return chat.Outcome{}, emit(chat.Text("ok"))
	})
}
