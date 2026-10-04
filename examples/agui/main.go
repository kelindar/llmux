// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

// Package main is a deterministic AG-UI example. Its process-local Store is
// illustrative only: it has no authentication, tenant isolation, or durability.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"embed"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/chat"
)

const (
	target    = "contact"
	catalogID = "https://example.invalid/llmux/agui/catalog/v1"
)

//go:embed catalog.json common_types.json testdata/ui.json
var assets embed.FS

type fixture struct {
	Form   json.RawMessage `json:"form"`
	Result json.RawMessage `json:"result"`
}

type uiEnvelope struct {
	Kind      string            `json:"kind"`
	Format    string            `json:"format"`
	Version   string            `json:"version"`
	CatalogID string            `json:"catalogId"`
	Payload   []json.RawMessage `json:"payload"`
}

type action struct {
	Kind              string            `json:"kind"`
	Name              string            `json:"name"`
	SurfaceID         string            `json:"surfaceId"`
	SourceComponentID string            `json:"sourceComponentId"`
	Context           map[string]string `json:"context"`
	SourceResponseID  string            `json:"sourceResponseId"`
	SourceItemID      string            `json:"sourceItemId"`
}

type counts struct {
	runs    atomic.Int64
	effects atomic.Int64
}

type demo struct {
	handler http.Handler
	store   *memoryStore
	counts  *counts
}

func main() {
	app, err := newDemo()
	if err != nil {
		log.Fatal(err)
	}
	control := http.NewServeMux()
	control.Handle("/", app.handler)
	var server *http.Server
	control.HandleFunc("POST /debug/shutdown", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		go func() { _ = server.Shutdown(context.Background()) }()
	})
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	server = &http.Server{
		Addr:              net.JoinHostPort("127.0.0.1", port),
		Handler:           control,
		ReadHeaderTimeout: 5 * time.Second,
	}
	log.Printf("AG-UI example listening on http://%s", server.Addr)
	if err := server.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func newDemo() (*demo, error) {
	var ui fixture
	raw, err := assets.ReadFile("testdata/ui.json")
	if err != nil {
		return nil, fmt.Errorf("read UI fixture: %w", err)
	}
	if err := json.Unmarshal(raw, &ui); err != nil {
		return nil, fmt.Errorf("decode UI fixture: %w", err)
	}
	for _, payload := range []json.RawMessage{ui.Form, ui.Result} {
		var envelope uiEnvelope
		if err := json.Unmarshal(payload, &envelope); err != nil {
			return nil, fmt.Errorf("decode UI envelope: %w", err)
		}
		if envelope.Kind != "ui" || envelope.Format != "a2ui" || envelope.Version != "0.9.1" || envelope.CatalogID != catalogID || len(envelope.Payload) != 3 {
			return nil, errors.New("UI fixture must contain a complete A2UI 0.9.1 surface")
		}
		for i, operation := range []string{"createSurface", "updateComponents", "updateDataModel"} {
			var message map[string]json.RawMessage
			if err := json.Unmarshal(envelope.Payload[i], &message); err != nil || len(message) != 2 || string(message["version"]) != `"v0.9.1"` || len(message[operation]) == 0 || !json.Valid(message[operation]) {
				return nil, errors.New("UI fixture must contain createSurface, updateComponents, and updateDataModel v0.9.1 messages")
			}
		}
	}
	catalog, err := assets.ReadFile("catalog.json")
	if err != nil {
		return nil, fmt.Errorf("read component catalog: %w", err)
	}
	var metadata struct {
		CatalogID string `json:"catalogId"`
	}
	if err := json.Unmarshal(catalog, &metadata); err != nil || metadata.CatalogID != catalogID {
		return nil, errors.New("component catalog ID does not match the UI fixture")
	}

	counts := &counts{}
	store := newMemoryStore()
	agent := &contactAgent{ui: ui, counts: counts}
	catalogAgent := &agents{agent: agent}
	handler := llmux.New(catalogAgent,
		llmux.WithAGUI(),
		llmux.WithStore(store),
		llmux.WithStoreDefault(true),
	)
	mux := http.NewServeMux()
	mux.Handle("/ag-ui", handler)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusNoContent) })
	mux.HandleFunc("GET /catalog.json", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/schema+json")
		_, _ = w.Write(catalog)
	})
	mux.Handle("GET /common_types.json", http.FileServerFS(assets))
	mux.HandleFunc("GET /debug/state", func(w http.ResponseWriter, _ *http.Request) {
		store.mu.Lock()
		responses := len(store.records)
		store.mu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]int64{
			"agentRuns":       counts.runs.Load(),
			"actionEffects":   counts.effects.Load(),
			"storedResponses": int64(responses),
		})
	})
	return &demo{handler: mux, store: store, counts: counts}, nil
}

type agents struct{ agent chat.Agent }

func (a *agents) List(context.Context) (map[string]chat.Info, error) {
	return map[string]chat.Info{target: info()}, nil
}

func (a *agents) Load(_ context.Context, name string) (chat.Agent, chat.Info, error) {
	if name != target {
		return nil, chat.Info{}, chat.NotFound()
	}
	return a.agent, info(), nil
}

func info() chat.Info {
	return chat.Info{Continuation: true, Extensions: map[string]bool{"x-ui": true}}
}

type contactAgent struct {
	ui     fixture
	counts *counts
}

func (a *contactAgent) Run(ctx context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
	a.counts.runs.Add(1)
	if err := ctx.Err(); err != nil {
		return chat.Outcome{}, err
	}
	var current []chat.Item
	if len(req.Input) > 0 {
		current = req.Input[len(req.Input)-1:]
	}
	selected, err := actionFrom(current)
	if err != nil {
		return chat.Outcome{}, err
	}
	if selected == nil {
		if err := emit.Text("Fill in the contact request and submit it."); err != nil {
			return chat.Outcome{}, err
		}
		return chat.Outcome{}, emit(chat.OutputItem(uiItem("", a.ui.Form)))
	}

	name := strings.TrimSpace(selected.Context["name"])
	email := strings.TrimSpace(selected.Context["email"])
	a.counts.effects.Add(1)
	if err := emit.Text(fmt.Sprintf("Contact request saved for %s (%s).", name, email)); err != nil {
		return chat.Outcome{}, err
	}
	return chat.Outcome{}, emit(chat.OutputItem(uiItem("", a.ui.Result)))
}

func uiItem(id string, data json.RawMessage) chat.Item {
	return chat.Item{Type: chat.ItemExtension, ID: id, Status: chat.StatusCompleted, Data: data}
}

func actionFrom(items []chat.Item) (*action, error) {
	var selected *action
	for _, item := range items {
		if item.Type != chat.ItemExtension {
			continue
		}
		var marker struct {
			Kind string `json:"kind"`
		}
		if err := json.Unmarshal(item.Data, &marker); err != nil || marker.Kind != "ui_action" {
			continue
		}
		if selected != nil {
			return nil, errors.New("only one UI action may be submitted in a turn")
		}
		var value action
		if err := json.Unmarshal(item.Data, &value); err != nil {
			return nil, fmt.Errorf("decode UI action: %w", err)
		}
		selected = &value
	}
	return selected, nil
}

type turnRecord struct {
	thread   string
	turn     []chat.Item
	response chat.Response
}

type conversationState struct {
	latest string
	busy   bool
}

type runRecord struct {
	fingerprint string
	responseID  string
	threadID    string
}

type memoryStore struct {
	mu           sync.Mutex
	sequence     int64
	threadSeq    int64
	records      map[string]turnRecord
	threads      map[string]*conversationState
	correlations map[string]string
	completed    map[string]runRecord
	running      map[string]runRecord
}

func newMemoryStore() *memoryStore {
	return &memoryStore{
		records:      make(map[string]turnRecord),
		threads:      make(map[string]*conversationState),
		correlations: make(map[string]string),
		completed:    make(map[string]runRecord),
		running:      make(map[string]runRecord),
	}
}

func (s *memoryStore) Load(_ context.Context, id string) ([]chat.Item, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	current, ok := s.records[id]
	if !ok {
		return nil, fmt.Errorf("response %q was not found", id)
	}
	chain := []turnRecord{current}
	for current.response.Previous != nil && *current.response.Previous != "" {
		current, ok = s.records[*current.response.Previous]
		if !ok {
			return nil, errors.New("stored response has a missing parent")
		}
		chain = append(chain, current)
	}
	var history []chat.Item
	for _, record := range slices.Backward(chain) {
		history = append(history, cloneItems(record.turn)...)
		history = append(history, cloneItems(record.response.Output)...)
	}
	return history, nil
}

func (s *memoryStore) Accept(_ context.Context, turn *chat.TurnRequest) (chat.Acceptance, error) {
	if turn == nil || turn.Request == nil {
		return chat.Acceptance{}, chat.Invalid("request", "request is required")
	}
	if !turn.Retain {
		return chat.Acceptance{}, chat.Unsupported("store", "the example requires stored turns")
	}
	if _, ok := turn.Request.Controls.Extensions["x-ui"]; !ok {
		return chat.Acceptance{}, chat.Unsupported("forwardedProps", "the example requires the x-ui extension")
	}
	if strings.TrimSpace(turn.Thread) == "" {
		return chat.Acceptance{}, chat.Invalid("threadId", "threadId is required")
	}
	fingerprint, err := turnFingerprint(turn)
	if err != nil {
		return chat.Acceptance{}, fmt.Errorf("fingerprint accepted turn: %w", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// A completed run is checked before thread freshness so an exact retry can
	// replay its result after the thread has advanced.
	if key := turn.IdempotencyKey; key != "" {
		if previous, ok := s.completed[key]; ok {
			if previous.fingerprint != fingerprint {
				return chat.Acceptance{}, conflict("idempotency_conflict", "runId was already used for a different request")
			}
			record, ok := s.records[previous.responseID]
			if !ok {
				return chat.Acceptance{}, errors.New("idempotent response is missing")
			}
			replay := record.response.Clone()
			return chat.Acceptance{Replay: &replay}, nil
		}
		if previous, ok := s.running[key]; ok {
			if previous.fingerprint != fingerprint {
				return chat.Acceptance{}, conflict("idempotency_conflict", "runId was already used for a different request")
			}
			replay := chat.Response{
				ID:       previous.responseID,
				Status:   chat.StatusInProgress,
				Metadata: map[string]string{"thread_id": previous.threadID},
			}
			return chat.Acceptance{Replay: &replay}, nil
		}
	}

	threadID := s.correlations[turn.Thread]
	if threadID == "" {
		threadID = turn.Thread
	}
	threadState := s.threads[threadID]
	if threadState == nil {
		if turn.Previous != nil {
			return chat.Acceptance{}, conflict("stale_parent", "previous response does not belong to this thread")
		}
		if strings.HasPrefix(turn.Thread, "thread_") {
			return chat.Acceptance{}, conflict("stale_parent", "threadId is not a known conversation")
		}
		s.threadSeq++
		threadID = "thread_" + strconv.FormatInt(s.threadSeq, 10)
		threadState = &conversationState{}
		s.threads[threadID] = threadState
		s.correlations[turn.Thread] = threadID
	}
	switch {
	case threadState.busy:
		return chat.Acceptance{}, conflict("thread_in_progress", "another run is being processed for this thread")
	case turn.Previous == nil && threadState.latest != "":
		return chat.Acceptance{}, conflict("stale_parent", "previousResponseId is required for this thread")
	case turn.Previous != nil && *turn.Previous != threadState.latest:
		return chat.Acceptance{}, conflict("stale_parent", "previousResponseId is not the latest response")
	}

	selected, err := actionFrom(turn.Turn)
	if err != nil {
		return chat.Acceptance{}, chat.Invalid("action", err.Error())
	}
	if selected != nil {
		if turn.Previous == nil || selected.SourceResponseID != threadState.latest || selected.SourceResponseID != *turn.Previous {
			return chat.Acceptance{}, conflict("stale_parent", "UI action must reference the latest successful response")
		}
		record := s.records[selected.SourceResponseID]
		if !storedAction(record.response, *selected) {
			return chat.Acceptance{}, chat.Invalid("action", "UI action is not present on its stored source widget")
		}
		if strings.TrimSpace(selected.Context["name"]) == "" || strings.TrimSpace(selected.Context["email"]) == "" {
			return chat.Acceptance{}, chat.Invalid("action", "name and email are required")
		}
	}

	threadState.busy = true
	s.sequence++
	responseID := "resp_" + strconv.FormatInt(s.sequence, 10)
	if key := turn.IdempotencyKey; key != "" {
		s.running[key] = runRecord{fingerprint: fingerprint, responseID: responseID, threadID: threadID}
	}
	items := cloneItems(turn.Turn)
	seed := chat.Response{
		ID: responseID,
		Metadata: map[string]string{
			"thread_id": threadID,
		},
	}
	return chat.Acceptance{
		Response: seed,
		Finish: func(_ context.Context, response *chat.Response, runErr error) error {
			s.mu.Lock()
			defer s.mu.Unlock()
			threadState.busy = false
			if key := turn.IdempotencyKey; key != "" {
				delete(s.running, key)
			}
			if response == nil || !response.Store {
				return nil
			}
			s.records[response.ID] = turnRecord{thread: threadID, turn: items, response: response.Clone()}
			threadState.latest = response.ID
			if key := turn.IdempotencyKey; key != "" {
				s.completed[key] = runRecord{fingerprint: fingerprint, responseID: response.ID}
			}
			return nil
		},
	}, nil
}

func turnFingerprint(turn *chat.TurnRequest) (string, error) {
	identity := struct {
		Thread   string
		Request  *chat.Request
		Turn     []chat.Item
		Previous *string
		Metadata map[string]string
		Store    *bool
		Retain   bool
		Stream   bool
	}{
		Thread: turn.Thread, Request: turn.Request, Turn: turn.Turn, Previous: turn.Previous,
		Metadata: turn.Metadata, Store: turn.Store, Retain: turn.Retain, Stream: turn.Stream,
	}
	data, err := json.Marshal(identity)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:]), nil
}

func conflict(code, message string) *chat.Error {
	return &chat.Error{Status: http.StatusConflict, Type: "invalid_request_error", Code: code, Message: message}
}

func storedAction(response chat.Response, submitted action) bool {
	if response.Status != chat.StatusCompleted {
		return false
	}
	for _, item := range response.Output {
		if item.Type != chat.ItemExtension || item.ID != submitted.SourceItemID {
			continue
		}
		var ui uiEnvelope
		if err := json.Unmarshal(item.Data, &ui); err != nil || ui.Kind != "ui" || ui.Format != "a2ui" || ui.Version != "0.9.1" || ui.CatalogID != catalogID {
			continue
		}
		for _, message := range ui.Payload {
			var update struct {
				UpdateComponents *struct {
					SurfaceID  string `json:"surfaceId"`
					Components []struct {
						ID     string `json:"id"`
						Action struct {
							Event struct {
								Name    string                     `json:"name"`
								Context map[string]json.RawMessage `json:"context"`
							} `json:"event"`
						} `json:"action"`
					} `json:"components"`
				} `json:"updateComponents"`
			}
			if json.Unmarshal(message, &update) != nil || update.UpdateComponents == nil || update.UpdateComponents.SurfaceID != submitted.SurfaceID {
				continue
			}
			for _, component := range update.UpdateComponents.Components {
				if component.ID == submitted.SourceComponentID && component.Action.Event.Name == submitted.Name {
					for key := range submitted.Context {
						if _, declared := component.Action.Event.Context[key]; !declared {
							return false
						}
					}
					return true
				}
			}
		}
	}
	return false
}

func cloneItems(items []chat.Item) []chat.Item {
	cloned := make([]chat.Item, len(items))
	for i, item := range items {
		cloned[i] = item.Clone()
	}
	return cloned
}
