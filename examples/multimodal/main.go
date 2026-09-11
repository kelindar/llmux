package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/chat"
)

type agents struct {
	vision chat.Agent
}

func (a *agents) List(context.Context) (map[string]chat.Info, error) {
	return map[string]chat.Info{
		"vision": {InputModalities: chat.ModalityText | chat.ModalityImage},
	}, nil
}

func (a *agents) Load(_ context.Context, target string) (chat.Agent, chat.Info, error) {
	if target != "vision" {
		return nil, chat.Info{}, chat.NotFound()
	}
	return a.vision, chat.Info{InputModalities: chat.ModalityText | chat.ModalityImage}, nil
}

func main() {
	catalog := &agents{
		vision: chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			for _, part := range req.Input[0].Content {
				if part.Type == chat.PartImage && part.Media != nil {
					return chat.Outcome{}, emit.Text(fmt.Sprintf("received %s", part.Media.MIMEType))
				}
			}
			return chat.Outcome{}, emit.Text("no image")
		}),
	}

	mux := http.NewServeMux()
	mux.Handle("/v1/", http.StripPrefix("/v1", llmux.New(catalog)))

	log.Println("multimodal example listening on http://127.0.0.1:8080 (POST /v1/chat/completions)")
	if err := http.ListenAndServe("127.0.0.1:8080", mux); err != nil {
		log.Fatal(err)
	}
}
