package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/kelindar/llmux"
)

func main() {
	agent := llmux.AgentFunc(func(_ context.Context, req *llmux.Request, emit llmux.Emit) (llmux.Outcome, error) {
		for _, part := range req.Input[0].Content {
			if part.Type == llmux.PartImage && part.Media != nil {
				return llmux.Outcome{}, llmux.EmitText(emit, fmt.Sprintf("received %s", part.Media.MIMEType))
			}
		}
		return llmux.Outcome{}, llmux.EmitText(emit, "no image")
	})
	resolver := llmux.ResolverFunc(func(context.Context, string) (llmux.Agent, llmux.Capabilities, error) {
		return agent, llmux.Capabilities{InputModalities: llmux.ModalityText | llmux.ModalityImage}, nil
	})

	log.Println("multimodal example listening on http://127.0.0.1:8080")
	if err := http.ListenAndServe("127.0.0.1:8080", llmux.New(resolver)); err != nil {
		log.Fatal(err)
	}
}
