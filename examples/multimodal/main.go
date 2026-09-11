package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/chat"
)

func main() {
	agent := chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		for _, part := range req.Input[0].Content {
			if part.Type == chat.PartImage && part.Media != nil {
				return chat.Outcome{}, emit.Text(fmt.Sprintf("received %s", part.Media.MIMEType))
			}
		}
		return chat.Outcome{}, emit.Text("no image")
	})
	resolver := chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
		return agent, chat.Info{InputModalities: chat.ModalityText | chat.ModalityImage}, nil
	})

	mux := http.NewServeMux()
	mux.Handle("/v1/", http.StripPrefix("/v1", llmux.New(resolver)))

	log.Println("multimodal example listening on http://127.0.0.1:8080 (POST /v1/chat/completions)")
	if err := http.ListenAndServe("127.0.0.1:8080", mux); err != nil {
		log.Fatal(err)
	}
}
