// Package main shows an in-memory Lifecycle: accept, turn-local continuation,
// idempotency, and GET retrieval. It is illustrative, not production storage.
package main

import (
	"context"
	"encoding/json/v2"
	"errors"
	"log"
	"net/http"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/chat"
)

func main() {
	store := newStore()
	agent := chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		text := "turn"
		if len(req.Turn) > 0 && len(req.Turn[0].Content) > 0 {
			text = req.Turn[0].Content[0].Text
		}
		return chat.Outcome{}, emit(chat.Text("echo: " + text))
	})
	resolver := chat.Resolver(func(context.Context, string) (chat.Agent, chat.Capabilities, error) {
		return agent, chat.Capabilities{Continuation: true}, nil
	})

	mux := http.NewServeMux()
	handler := llmux.New(resolver,
		llmux.WithLifecycle(store),
		llmux.WithContinuationStore(store),
		llmux.WithStoreDefault(true),
	)
	mux.Handle("/api/v1/", http.StripPrefix("/api/v1", handler))
	mux.HandleFunc("GET /api/responses/{id}", func(w http.ResponseWriter, r *http.Request) {
		rec, ok := store.get(r.PathValue("id"))
		if !ok {
			http.NotFound(w, r)
			return
		}
		body, err := llmux.ResponsesBody(*rec.Request, rec.State, rec.ID, rec.Created)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.MarshalWrite(w, body)
	})

	log.Println("listening on http://127.0.0.1:8080 (POST /api/responses)")
	if err := http.ListenAndServe("127.0.0.1:8080", mux); err != nil {
		log.Fatal(err)
	}
}

type turnRecord struct {
	ID      string
	Created int64
	Parent  string
	Turn    []chat.Item
	Request *chat.Request
	State   chat.ResponseState
}

type store struct {
	mu      sync.Mutex
	seq     atomic.Int64
	byID    map[string]turnRecord
	byKey   map[string]string
	pending map[string]string
	running map[string]bool
}

func newStore() *store {
	return &store{
		byID:    make(map[string]turnRecord),
		byKey:   make(map[string]string),
		pending: make(map[string]string),
		running: make(map[string]bool),
	}
}

func (s *store) Load(_ context.Context, id string) ([]chat.Item, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rec, ok := s.byID[id]
	if !ok {
		return nil, errors.New("missing")
	}
	var chain []turnRecord
	for cur := rec; ; {
		chain = append(chain, cur)
		if cur.Parent == "" {
			break
		}
		parent, ok := s.byID[cur.Parent]
		if !ok {
			return nil, errors.New("broken parent chain")
		}
		cur = parent
	}
	var items []chat.Item
	for _, c := range slices.Backward(chain) {
		items = append(items, cloneItems(c.Turn)...)
		items = append(items, cloneItems(c.State.Output)...)
	}
	return items, nil
}

func (s *store) Accept(ctx context.Context, turn *chat.TurnRequest) (chat.Acceptance, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	switch {
	case turn.Request.Store != nil && !*turn.Request.Store:
		return chat.Acceptance{}, chat.Unsupported("store", "example requires store")
	case !turn.Request.Retain:
		return chat.Acceptance{}, chat.Unsupported("store", "example requires store")
	}

	key := turn.IdempotencyKey
	if key != "" {
		if s.running[key] {
			return chat.Acceptance{}, &chat.APIError{
				Status:  http.StatusConflict,
				Type:    "invalid_request_error",
				Code:    "request_in_progress",
				Message: "request already running",
			}
		}
		if id, ok := s.byKey[key]; ok {
			rec := s.byID[id]
			state := rec.State.Clone()
			return chat.Acceptance{
				ID:      rec.ID,
				Created: rec.Created,
				Replay:  &state,
			}, nil
		}
		s.running[key] = true
	}

	n := s.seq.Add(1)
	id := "resp_" + strconv.FormatInt(n, 10)
	acc := chat.Acceptance{ID: id, Created: n}
	if key != "" {
		s.pending[id] = key
	}
	if turn.Request.Controls.Extensions["x-durable"] != nil {
		acc.Durable = true
		acc.Context = context.WithoutCancel(ctx)
	}
	return acc, nil
}

func (s *store) Finalize(ctx context.Context, result *chat.TurnResult) error {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), time.Second)
	defer cancel()

	s.mu.Lock()
	defer s.mu.Unlock()
	key := s.pending[result.ID]
	delete(s.pending, result.ID)
	if key != "" {
		delete(s.running, key)
	}
	if !result.State.Store {
		return nil
	}
	parent := ""
	if result.Request.Previous != nil {
		parent = *result.Request.Previous
	}
	rec := turnRecord{
		ID:      result.ID,
		Created: result.Created,
		Parent:  parent,
		Turn:    cloneItems(result.Request.Turn),
		Request: result.Request,
		State:   result.State.Clone(),
	}
	s.byID[result.ID] = rec
	if key != "" {
		s.byKey[key] = result.ID
	}
	_ = ctx // keep values available for real persistence work
	return nil
}

func (s *store) get(id string) (turnRecord, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rec, ok := s.byID[id]
	return rec, ok
}

func cloneItems(items []chat.Item) []chat.Item {
	out := make([]chat.Item, len(items))
	for i, item := range items {
		out[i] = item.Clone()
	}
	return out
}
