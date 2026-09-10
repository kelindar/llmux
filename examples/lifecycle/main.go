// Package main shows the smallest Lifecycle: Accept assigns identity, the
// agent runs, and Finish saves the turn-local result.
package main

import (
	"context"
	"log"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/chat"
)

func main() {
	store := &store{byID: make(map[string]saved)}
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit.Text("hello")
	})
	resolver := chat.Resolver(func(context.Context, string) (chat.Agent, chat.Capabilities, error) {
		return agent, chat.Capabilities{Continuation: true}, nil
	})

	mux := http.NewServeMux()
	mux.Handle("/v1/", http.StripPrefix("/v1", llmux.New(resolver,
		llmux.WithLifecycle(store.Accept),
		llmux.WithStoreDefault(true),
	)))

	log.Println("listening on http://127.0.0.1:8080 (POST /v1/responses)")
	if err := http.ListenAndServe("127.0.0.1:8080", mux); err != nil {
		log.Fatal(err)
	}
}

type saved struct {
	ID      string
	Created int64
	Target  string
	Turn    []chat.Item
	State   chat.State
}

type store struct {
	mu   sync.Mutex
	seq  atomic.Int64
	byID map[string]saved
}

func (s *store) Accept(_ context.Context, turn *chat.TurnRequest) (chat.Acceptance, error) {
	if !turn.Request.Retain {
		return chat.Acceptance{}, chat.Unsupported("store", "example requires store")
	}
	n := s.seq.Add(1)
	id := "resp_" + strconv.FormatInt(n, 10)
	target := turn.Request.Target
	turnItems := cloneItems(turn.Request.Turn)
	return chat.Acceptance{
		ID:      id,
		Created: n,
		Finish: func(_ context.Context, result *chat.TurnResult) error {
			if !result.State.Store {
				return nil
			}
			s.mu.Lock()
			defer s.mu.Unlock()
			s.byID[result.ID] = saved{
				ID:      result.ID,
				Created: result.Created,
				Target:  target,
				Turn:    turnItems,
				State:   result.State.Clone(),
			}
			return nil
		},
	}, nil
}

func cloneItems(items []chat.Item) []chat.Item {
	out := make([]chat.Item, len(items))
	for i, item := range items {
		out[i] = item.Clone()
	}
	return out
}
