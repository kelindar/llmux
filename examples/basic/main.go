package main

import (
	"context"
	"log"
	"net/http"

	"github.com/kelindar/llmux"
)

func main() {
	agent := llmux.AgentFunc(func(_ context.Context, req *llmux.Request, emit llmux.Emit) (llmux.Outcome, error) {
		return llmux.Outcome{}, llmux.EmitText(emit, "received "+req.Input[0].Content[0].Text)
	})
	resolver := llmux.ResolverFunc(func(context.Context, string) (llmux.Agent, llmux.Capabilities, error) {
		return agent, llmux.Capabilities{}, nil
	})

	log.Println("listening on http://127.0.0.1:8080")
	if err := http.ListenAndServe("127.0.0.1:8080", llmux.New(resolver)); err != nil {
		log.Fatal(err)
	}
}
