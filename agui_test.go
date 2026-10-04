package llmux

import (
	"context"
	"encoding/json/jsontext"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAGUITransport(t *testing.T) {
	body, err := os.ReadFile("testdata/agui/request.json")
	require.NoError(t, err)
	ui, err := os.ReadFile("testdata/agui/ui.json")
	require.NoError(t, err)
	info := chat.Info{Continuation: true, Extensions: map[string]bool{"x-ui": true}}
	var calls atomic.Int32
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		calls.Add(1)
		if err := emit.Text("hello"); err != nil {
			return chat.Outcome{}, err
		}
		return chat.Outcome{}, emit(chat.OutputItem(chat.Item{Type: chat.ItemExtension, ID: "ui-1", Data: jsontext.Value(ui)}))
	})
	t.Run("configuration", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			options []Option
			status  int
			body    string
		}{
			{"disabled", nil, 404, string(body)},
			{"missing store", []Option{WithAGUI()}, 400, string(body)},
			{"invalid body", []Option{WithAGUI(), WithStore(newMemoryLife())}, 400, "{"},
			{"limit", []Option{WithAGUI(), WithStore(newMemoryLife()), WithLimits(chat.Limits{MaxRequestBytes: 1})}, 413, string(body)},
		} {
			t.Run(tc.name, func(t *testing.T) {
				rec := postJSON(t, testHandler(agent, info, tc.options...), "/ag-ui", tc.body, nil)
				assert.Equal(t, tc.status, rec.Code)
			})
		}
		h := testHandler(agent, info, WithAGUI(), WithStore(newMemoryLife()))
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/ag-ui", nil))
		assert.Equal(t, 405, rec.Code)
		rec = postJSON(t, h, "/ag-ui", string(body), map[string]string{"Idempotency-Key": "different"})
		assert.Equal(t, 400, rec.Code)
		assert.Zero(t, calls.Load())
	})
	t.Run("persist then deliver and replay", func(t *testing.T) {
		var saved *chat.Response
		var finishes int
		rec := httptest.NewRecorder()
		store := storeHook{accept: func(_ context.Context, turn *chat.TurnRequest) (chat.Acceptance, error) {
			assert.Equal(t, "creation-1", turn.Thread)
			assert.Equal(t, "run-1", turn.IdempotencyKey)
			assert.True(t, turn.Retain)
			require.Len(t, turn.Turn, 1)
			if saved != nil {
				return chat.Acceptance{Replay: saved}, nil
			}
			return chat.Acceptance{Response: chat.Response{ID: "response-1", Metadata: map[string]string{"thread_id": "canonical-1", "execution_id": "execution-1"}}, Finish: func(_ context.Context, r *chat.Response, _ error) error {
				finishes++
				assert.NotContains(t, rec.Body.String(), "llmux.item")
				assert.NotContains(t, rec.Body.String(), "RUN_FINISHED")
				clone := r.Clone()
				saved = &clone
				return nil
			}}, nil
		}}
		h := testHandler(agent, info, WithAGUI(), WithStore(store))
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/ag-ui", strings.NewReader(string(body))))
		require.Equal(t, 200, rec.Code)
		assert.Contains(t, rec.Body.String(), `"threadId":"canonical-1"`)
		assert.Contains(t, rec.Body.String(), `"id":"ui-1"`)
		assert.Contains(t, rec.Body.String(), "RUN_FINISHED")
		assert.NotContains(t, rec.Body.String(), "[DONE]")
		replay := postJSON(t, h, "/ag-ui", string(body), nil)
		originalFrames := strings.Split(strings.TrimSpace(rec.Body.String()), "\n\n")
		replayedFrames := strings.Split(strings.TrimSpace(replay.Body.String()), "\n\n")
		require.Len(t, replayedFrames, len(originalFrames))
		for i, frame := range originalFrames {
			assert.JSONEq(t, strings.TrimPrefix(frame, "data: "), strings.TrimPrefix(replayedFrames[i], "data: "))
		}
		assert.Equal(t, 1, finishes)
		assert.Equal(t, int32(1), calls.Load())
	})
	t.Run("persistence failure", func(t *testing.T) {
		store := storeHook{accept: func(context.Context, *chat.TurnRequest) (chat.Acceptance, error) {
			return chat.Acceptance{Finish: func(context.Context, *chat.Response, error) error { return errors.New("private persistence failure") }}, nil
		}}
		rec := postJSON(t, testHandler(agent, info, WithAGUI(), WithStore(store)), "/ag-ui", string(body), nil)
		assert.Contains(t, rec.Body.String(), "RUN_ERROR")
		assert.NotContains(t, rec.Body.String(), "RUN_FINISHED")
		assert.NotContains(t, rec.Body.String(), "llmux.item")
		assert.NotContains(t, rec.Body.String(), "private persistence failure")
	})
	t.Run("terminal replay", func(t *testing.T) {
		for _, status := range []chat.Status{chat.StatusFailed, chat.StatusCancelled, chat.StatusIncomplete, chat.StatusInProgress} {
			t.Run(string(status), func(t *testing.T) {
				store := storeHook{accept: func(context.Context, *chat.TurnRequest) (chat.Acceptance, error) {
					return chat.Acceptance{Replay: &chat.Response{ID: "original", Status: status, Metadata: map[string]string{"thread_id": "thread", "execution_id": "execution"}}}, nil
				}}
				rec := postJSON(t, testHandler(agent, info, WithAGUI(), WithStore(store)), "/ag-ui", string(body), nil)
				assert.NotContains(t, rec.Body.String(), "RUN_FINISHED")
				if status == chat.StatusInProgress {
					assert.Equal(t, 409, rec.Code)
					assert.Contains(t, rec.Body.String(), "run_in_progress")
					assert.Contains(t, rec.Body.String(), `"response_id":"original"`)
				} else {
					assert.Contains(t, rec.Body.String(), "RUN_ERROR")
				}
			})
		}
	})
	t.Run("unsupported adapter", func(t *testing.T) {
		for _, tc := range []struct{ path, body string }{
			{"/responses", `{"model":"agent/basic","input":"hello"}`},
			{"/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hello"}]}`},
			{"/messages", `{"model":"agent/basic","max_tokens":5,"messages":[{"role":"user","content":"hello"}]}`},
		} {
			rec := postJSON(t, testHandler(agent, info), tc.path, tc.body, map[string]string{"anthropic-version": "2023-06-01"})
			assert.Equal(t, 400, rec.Code)
			assert.Contains(t, rec.Body.String(), "UI-enabled agents require the AG-UI endpoint")
		}
	})
	t.Run("missing finish", func(t *testing.T) {
		store := storeHook{accept: func(context.Context, *chat.TurnRequest) (chat.Acceptance, error) { return chat.Acceptance{}, nil }}
		rec := postJSON(t, testHandler(agent, info, WithAGUI(), WithStore(store)), "/ag-ui", string(body), nil)
		assert.Equal(t, 400, rec.Code)
		assert.Contains(t, rec.Body.String(), "Finish")
	})
	t.Run("missing client catalog", func(t *testing.T) {
		var saved chat.Response
		store := storeHook{accept: func(context.Context, *chat.TurnRequest) (chat.Acceptance, error) {
			return chat.Acceptance{Finish: func(_ context.Context, r *chat.Response, _ error) error { saved = r.Clone(); return nil }}, nil
		}}
		agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.OutputItem(chat.Item{Type: chat.ItemExtension, Data: jsontext.Value(ui)}))
		})
		request := `{"threadId":"creation","runId":"run","messages":[{"id":"msg","role":"user","content":"hello"}],"forwardedProps":{"llmux":{"target":"agent/basic"}}}`
		rec := postJSON(t, testHandler(agent, info, WithAGUI(), WithStore(store)), "/ag-ui", request, nil)
		assert.Equal(t, 400, rec.Code)
		assert.Contains(t, rec.Body.String(), "UI output requires declared client catalog support")
		assert.Equal(t, chat.StatusFailed, saved.Status)
		assert.Empty(t, saved.Output)
	})
}

func TestAGUIInvalid(t *testing.T) {
	body, err := os.ReadFile("testdata/agui/request.json")
	require.NoError(t, err)
	validUI, err := os.ReadFile("testdata/agui/ui.json")
	require.NoError(t, err)
	for _, tc := range []struct {
		name, payload string
		declared      bool
	}{
		{"malformed", `{"kind":"ui"}`, true},
		{"oversize", strings.Repeat("x", 256<<10), true},
		{"wrong catalog", strings.ReplaceAll(string(validUI), "example/v1", "wrong"), true},
		{"undeclared UI", string(validUI), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var saved chat.Response
			store := storeHook{accept: func(context.Context, *chat.TurnRequest) (chat.Acceptance, error) {
				return chat.Acceptance{Finish: func(_ context.Context, r *chat.Response, _ error) error { saved = r.Clone(); return nil }}, nil
			}}
			agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				if err := emit.Text("valid text"); err != nil {
					return chat.Outcome{}, err
				}
				// Even an agent that ignores the error cannot turn invalid UI into success.
				_ = emit(chat.OutputItem(chat.Item{Type: chat.ItemExtension, Data: jsontext.Value(tc.payload)}))
				return chat.Outcome{}, nil
			})
			rec := postJSON(t, testHandler(agent, chat.Info{Continuation: true, Extensions: map[string]bool{"x-ui": tc.declared}}, WithAGUI(), WithStore(store)), "/ag-ui", string(body), nil)
			if !tc.declared {
				assert.Equal(t, 400, rec.Code)
				assert.Equal(t, chat.Response{}, saved)
				return
			}
			assert.Equal(t, chat.StatusFailed, saved.Status)
			require.Len(t, saved.Output, 1)
			assert.Equal(t, chat.ItemMessage, saved.Output[0].Type)
			assert.Contains(t, rec.Body.String(), "RUN_ERROR")
			assert.NotContains(t, rec.Body.String(), "llmux.item")
		})
	}
}

func TestAGUIDisconnect(t *testing.T) {
	body, err := os.ReadFile("testdata/agui/request.json")
	require.NoError(t, err)
	started := make(chan struct{})
	continued := make(chan struct{})
	finished := make(chan struct{})
	var accepted atomic.Bool
	var calls atomic.Int32
	agent := chat.AgentFunc(func(ctx context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		calls.Add(1)
		if err := emit.Text("waiting"); err != nil {
			return chat.Outcome{}, err
		}
		close(started)
		select {
		case <-continued:
		case <-ctx.Done():
			return chat.Outcome{}, ctx.Err()
		}
		return chat.Outcome{}, emit.Text("approved")
	})
	store := storeHook{accept: func(context.Context, *chat.TurnRequest) (chat.Acceptance, error) {
		if accepted.Swap(true) {
			return chat.Acceptance{Replay: &chat.Response{ID: "waiting-response", Status: chat.StatusInProgress, Metadata: map[string]string{"thread_id": "waiting-thread", "execution_id": "waiting-execution"}}}, nil
		}
		return chat.Acceptance{RunTimeout: time.Second, Finish: func(_ context.Context, r *chat.Response, err error) error {
			assert.NoError(t, err)
			assert.Equal(t, chat.StatusCompleted, r.Status)
			close(finished)
			return nil
		}}, nil
	}}
	server := httptest.NewServer(testHandler(agent, chat.Info{Continuation: true, Extensions: map[string]bool{"x-ui": true}}, WithAGUI(), WithStore(store)))
	defer server.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/ag-ui", strings.NewReader(string(body)))
	require.NoError(t, err)
	resp, err := server.Client().Do(req)
	require.NoError(t, err)
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("agent did not start")
	}
	duplicate, err := http.Post(server.URL+"/ag-ui", "application/json", strings.NewReader(string(body)))
	require.NoError(t, err)
	assert.Equal(t, http.StatusConflict, duplicate.StatusCode)
	require.NoError(t, duplicate.Body.Close())
	assert.Equal(t, int32(1), calls.Load())
	cancel()
	require.NoError(t, resp.Body.Close())
	close(continued)
	select {
	case <-finished:
	case <-time.After(2 * time.Second):
		t.Fatal("disconnection cancelled accepted work")
	}
}

func TestAGUIErrors(t *testing.T) {
	for _, metadata := range []map[string]string{nil, {"thread_id": "authorized-thread"}} {
		err := activeRun(chat.Response{ID: "original", Metadata: metadata})
		assert.Equal(t, "original", err.Metadata["response_id"])
		assert.NotContains(t, metadata, "response_id")
		for _, kind := range []protocol{protocolAGUI, protocolResponses, protocolChat, protocolAnthropic} {
			rec := httptest.NewRecorder()
			writeProtocolError(rec, kind, err)
			assert.Equal(t, 409, rec.Code)
			if kind == protocolAGUI {
				assert.Contains(t, rec.Body.String(), `"metadata"`)
			} else {
				assert.NotContains(t, rec.Body.String(), `"metadata"`)
			}
		}
	}
}
