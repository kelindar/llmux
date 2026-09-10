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

	"github.com/kelindar/llmux"
)

func main() {
	store := newStore()
	agent := llmux.AgentFunc(func(_ context.Context, req *llmux.Request, emit llmux.Emit) (llmux.Outcome, error) {
		text := "turn"
		if len(req.Turn) > 0 && len(req.Turn[0].Content) > 0 {
			text = req.Turn[0].Content[0].Text
		}
		return llmux.Outcome{}, llmux.EmitText(emit, "echo: "+text)
	})
	resolver := llmux.ResolverFunc(func(context.Context, string) (llmux.Agent, llmux.Capabilities, error) {
		return agent, llmux.Capabilities{Continuation: true}, nil
	})

	mux := http.NewServeMux()
	handler := llmux.New(resolver, llmux.WithLifecycle(store), llmux.WithContinuationStore(store))
	mux.Handle("/api/", handler)
	mux.HandleFunc("GET /api/v1/responses/{id}", func(w http.ResponseWriter, r *http.Request) {
		rec, ok := store.get(r.PathValue("id"))
		if !ok {
			http.NotFound(w, r)
			return
		}
		body, err := llmux.ResponsesBody(*rec.Request, rec.Outcome, rec.Output, rec.ID, rec.Created)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.MarshalWrite(w, body)
	})

	log.Println("listening on http://127.0.0.1:8080 (POST /api/v1/responses)")
	if err := http.ListenAndServe("127.0.0.1:8080", mux); err != nil {
		log.Fatal(err)
	}
}

type turnRecord struct {
	ID      string
	Created int64
	Parent  string
	Turn    []llmux.Item
	Output  []llmux.Item
	Outcome llmux.Outcome
	Request *llmux.Request
}

type store struct {
	mu      sync.Mutex
	seq     atomic.Int64
	byID    map[string]turnRecord
	byKey   map[string]string
	pending map[string]string // response ID -> idempotency key
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

func (s *store) Load(_ context.Context, id string) ([]llmux.Item, error) {
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
	var items []llmux.Item
	for _, c := range slices.Backward(chain) {
		items = append(items, cloneItems(c.Turn)...)
		items = append(items, cloneItems(c.Output)...)
	}
	return items, nil
}

func (s *store) Accept(_ context.Context, turn *llmux.TurnRequest) (llmux.Acceptance, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	key := turn.IdempotencyKey
	if key != "" {
		if s.running[key] {
			return llmux.Acceptance{}, &llmux.APIError{
				Status:  http.StatusConflict,
				Type:    "invalid_request_error",
				Code:    "request_in_progress",
				Message: "request already running",
			}
		}
		if id, ok := s.byKey[key]; ok {
			rec := s.byID[id]
			return llmux.Acceptance{
				ID:      rec.ID,
				Created: rec.Created,
				Replay:  &llmux.Replay{Outcome: rec.Outcome, Output: cloneItems(rec.Output)},
			}, nil
		}
		s.running[key] = true
	}
	if turn.Request.Controls.Store != nil && !*turn.Request.Controls.Store {
		return llmux.Acceptance{}, llmux.Unsupported("store", "example requires store")
	}
	n := s.seq.Add(1)
	id := "resp_" + strconv.FormatInt(n, 10)
	if key != "" {
		s.pending[id] = key
	}
	return llmux.Acceptance{ID: id, Created: n}, nil
}

func (s *store) Finalize(_ context.Context, result *llmux.TurnResult) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	key := s.pending[result.ID]
	delete(s.pending, result.ID)
	if key != "" {
		delete(s.running, key)
	}
	if result.Err != nil || !result.Store {
		return nil
	}
	parent := ""
	if result.Request.Controls.PreviousResponseID != nil {
		parent = *result.Request.Controls.PreviousResponseID
	}
	rec := turnRecord{
		ID:      result.ID,
		Created: result.Created,
		Parent:  parent,
		Turn:    cloneItems(result.Request.Turn),
		Output:  cloneItems(result.Output),
		Outcome: result.Outcome,
		Request: result.Request,
	}
	s.byID[result.ID] = rec
	if key != "" {
		s.byKey[key] = result.ID
	}
	return nil
}

func (s *store) get(id string) (turnRecord, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rec, ok := s.byID[id]
	return rec, ok
}

func cloneItems(items []llmux.Item) []llmux.Item {
	out := make([]llmux.Item, len(items))
	for i, item := range items {
		out[i] = item.Clone()
	}
	return out
}
