// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
package main

import (
	"context"
	"log"
	"net/http"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/chat"
)

// agents is a minimal Catalog: List for discovery, Load for invocation.
type agents struct {
	echo chat.Agent
}

func (a *agents) List(context.Context) (map[string]chat.Info, error) {
	return map[string]chat.Info{"echo": {}}, nil
}

func (a *agents) Load(_ context.Context, target string) (chat.Agent, chat.Info, error) {
	if target != "echo" {
		return nil, chat.Info{}, chat.NotFound()
	}
	return a.echo, chat.Info{}, nil
}

func main() {
	catalog := &agents{
		echo: chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit.Text("received " + req.Input[0].Content[0].Text)
		}),
	}

	mux := http.NewServeMux()
	mux.Handle("/v1/", http.StripPrefix("/v1", llmux.New(catalog)))

	log.Println("listening on http://127.0.0.1:8080 (POST /v1/chat/completions)")
	if err := http.ListenAndServe("127.0.0.1:8080", mux); err != nil {
		log.Fatal(err)
	}
}
