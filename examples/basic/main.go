package main

import (
	"context"
	"log"
	"net/http"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/chat"
)

func main() {
	agent := chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit.Text("received " + req.Input[0].Content[0].Text)
	})
	resolver := chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
		return agent, chat.Info{}, nil
	})

	mux := http.NewServeMux()
	mux.Handle("/v1/", http.StripPrefix("/v1", llmux.New(resolver)))

	log.Println("listening on http://127.0.0.1:8080 (POST /v1/chat/completions)")
	if err := http.ListenAndServe("127.0.0.1:8080", mux); err != nil {
		log.Fatal(err)
	}
}
