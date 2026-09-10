package llmux

import (
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kelindar/llmux/chat"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type replayEntry struct {
	response chat.Response
}

type savedTurn struct {
	Turn     []chat.Item
	Response chat.Response
}

type memoryLife struct {
	mu              sync.Mutex
	accepts         int
	finals          int
	byKey           map[string]replayEntry
	running         map[string]bool
	records         map[string]savedTurn
	history         map[string][]chat.Item
	ctxSeen         atomic.Bool
	failOnce        atomic.Bool
	allowNoStore    bool
	boundCleanup    bool
	negativeTimeout bool
}

func newMemoryLife() *memoryLife {
	return &memoryLife{
		byKey:   make(map[string]replayEntry),
		running: make(map[string]bool),
		records: make(map[string]savedTurn),
		history: make(map[string][]chat.Item),
	}
}

func (m *memoryLife) Load(_ context.Context, id string) ([]chat.Item, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	items, ok := m.history[id]
	if !ok {
		return nil, errors.New("missing")
	}
	return cloneItems(items), nil
}

func (m *memoryLife) Accept(ctx context.Context, turn *chat.TurnRequest) (chat.Acceptance, error) {
	if ctx.Value(ctxKey{}) != nil {
		m.ctxSeen.Store(true)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.accepts++
	key := turn.IdempotencyKey
	if turn.Store != nil && !*turn.Store && !m.allowNoStore {
		return chat.Acceptance{}, chat.Unsupported("store", "application requires store")
	}
	if key != "" {
		if m.running[key] {
			return chat.Acceptance{}, &chat.Error{Status: http.StatusConflict, Type: "invalid_request_error", Code: "request_in_progress", Message: "request already running"}
		}
		if replay, ok := m.byKey[key]; ok {
			r := replay.response.Clone()
			return chat.Acceptance{Replay: &r}, nil
		}
		m.running[key] = true
	}

	turnItems := cloneItems(turn.Turn)
	acc := chat.Acceptance{Response: chat.Response{ID: "resp_app", Created: 99}}
	if turn.Request.Controls.Extensions["x-activity"] != nil {
		acc.Activity = true
	}
	if turn.Request.Controls.Extensions["x-durable"] != nil {
		acc.RunTimeout = 30 * time.Second
	}
	if m.boundCleanup {
		// Bound by RunTimeout; values come from the HTTP request context.
		acc.RunTimeout = 150 * time.Millisecond
	}
	if m.negativeTimeout {
		acc.RunTimeout = -time.Second
	}
	acc.Finish = func(ctx context.Context, resp *chat.Response, err error) error {
		return m.finish(ctx, key, turnItems, resp, err)
	}
	return acc, nil
}

func (m *memoryLife) finish(ctx context.Context, key string, turnItems []chat.Item, resp *chat.Response, runErr error) error {
	if m.boundCleanup {
		cleanupTimeout := 50 * time.Millisecond
		cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), cleanupTimeout)
		defer cancel()
		if cleanupCtx.Value(ctxKey{}) == nil {
			return errors.New("finish context missing value")
		}
		started := time.Now()
		select {
		case <-time.After(cleanupTimeout / 2):
		case <-cleanupCtx.Done():
			return errors.New("cleanup cancelled too early")
		}
		if time.Since(started) < cleanupTimeout/3 {
			return errors.New("cleanup budget not refreshed at finalization")
		}
		<-cleanupCtx.Done()
		if !errors.Is(cleanupCtx.Err(), context.DeadlineExceeded) {
			return errors.New("cleanup cancellation missing")
		}
		ctx = cleanupCtx
	}
	if ctx.Value(ctxKey{}) != nil {
		m.ctxSeen.Store(true)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.finals++
	if m.failOnce.Swap(false) {
		return errors.New("persist failed")
	}
	if key != "" {
		delete(m.running, key)
		if resp.Store {
			m.byKey[key] = replayEntry{response: resp.Clone()}
		}
	}
	if resp.Store && runErr == nil {
		items := append(cloneItems(turnItems), cloneItems(resp.Output)...)
		m.history[resp.ID] = items
		m.records[resp.ID] = savedTurn{Turn: turnItems, Response: resp.Clone()}
	}
	return nil
}

type ctxKey struct{}

func TestLifecycle(t *testing.T) {
	t.Run("supplied identity ordinary", func(t *testing.T) {
		life := newMemoryLife()
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("hi"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept), WithContinuationStore(life))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, nil)
		require.Equal(t, http.StatusOK, rec.Code)
		body := decodeResponse(t, rec)
		assert.Equal(t, "resp_app", body["id"])
		assert.Equal(t, float64(99), body["created_at"])
		assert.Equal(t, 1, life.finals)
	})

	t.Run("supplied identity streaming", func(t *testing.T) {
		life := newMemoryLife()
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("hi"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept), WithContinuationStore(life))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","stream":true,"store":true,"input":"x"}`, nil)
		require.Equal(t, http.StatusOK, rec.Code)
		assert.Contains(t, rec.Header().Get("Content-Type"), "text/event-stream")
		assert.Contains(t, rec.Body.String(), "response.completed")
		assert.Equal(t, 1, life.finals)
	})

	t.Run("accept failure", func(t *testing.T) {
		life := newMemoryLife()
		calls := 0
		handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
			calls++
			return chat.Outcome{}, nil
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":false,"input":"x"}`, nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)
		assert.Equal(t, 0, calls)
		assert.Equal(t, 0, life.finals)
	})

	t.Run("idempotent replay", func(t *testing.T) {
		life := newMemoryLife()
		var calls atomic.Int32
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			calls.Add(1)
			return chat.Outcome{Usage: &chat.Usage{Total: 3}}, emit(chat.Text("once"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept), WithContinuationStore(life))
		headers := map[string]string{"Idempotency-Key": "k1"}
		first := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, headers)
		require.Equal(t, http.StatusOK, first.Code)
		second := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, headers)
		require.Equal(t, http.StatusOK, second.Code)
		assert.Equal(t, int32(1), calls.Load())
		assert.Equal(t, "resp_app", decodeResponse(t, second)["id"])
	})

	t.Run("replay clone isolates encoding", func(t *testing.T) {
		life := newMemoryLife()
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("stable"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		headers := map[string]string{"Idempotency-Key": "replay-iso"}
		first := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, headers)
		require.Equal(t, http.StatusOK, first.Code)
		assert.Contains(t, first.Body.String(), "stable")

		life.mu.Lock()
		entry := life.byKey["replay-iso"]
		require.NotEmpty(t, entry.response.Output)
		entry.response.Output[0].Content[0].Text = "mutated-store"
		before := entry.response.Clone()
		life.byKey["replay-iso"] = entry
		life.mu.Unlock()

		replay := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, headers)
		require.Equal(t, http.StatusOK, replay.Code)
		assert.Contains(t, replay.Body.String(), "mutated-store")

		life.mu.Lock()
		after := life.byKey["replay-iso"].response
		life.mu.Unlock()
		assert.Equal(t, before.Output[0].Content[0].Text, after.Output[0].Content[0].Text)
	})

	t.Run("finalize store clone leaves response", func(t *testing.T) {
		life := newMemoryLife()
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("live"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, map[string]string{"Idempotency-Key": "live"})
		require.Equal(t, http.StatusOK, rec.Code)
		assert.Contains(t, rec.Body.String(), "live")

		life.mu.Lock()
		entry := life.byKey["live"]
		entry.response.Output[0].Content[0].Text = "stored-only"
		life.byKey["live"] = entry
		life.mu.Unlock()
		assert.Contains(t, rec.Body.String(), "live")
		assert.NotContains(t, rec.Body.String(), "stored-only")
	})

	t.Run("in progress conflict", func(t *testing.T) {
		life := newMemoryLife()
		_, err := life.Accept(context.WithValue(context.Background(), ctxKey{}, "v"), &chat.TurnRequest{Request: &chat.Request{Target: "a"}, IdempotencyKey: "busy"})
		require.NoError(t, err)
		_, err = life.Accept(context.WithValue(context.Background(), ctxKey{}, "v"), &chat.TurnRequest{Request: &chat.Request{Target: "a"}, IdempotencyKey: "busy"})
		require.Error(t, err)
		assert.Equal(t, "request_in_progress", err.(*chat.Error).Code)
	})

	t.Run("exactly one finalize", func(t *testing.T) {
		life := newMemoryLife()
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("ok"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, nil)
		require.Equal(t, http.StatusOK, rec.Code)
		assert.Equal(t, 1, life.accepts)
		assert.Equal(t, 1, life.finals)
	})

	t.Run("persist before success ordinary", func(t *testing.T) {
		life := newMemoryLife()
		life.failOnce.Store(true)
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("ok"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)
		assert.Equal(t, 1, life.finals)
	})

	t.Run("persist failure streaming", func(t *testing.T) {
		life := newMemoryLife()
		life.failOnce.Store(true)
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("ok"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","stream":true,"store":true,"input":"x"}`, nil)
		require.Equal(t, http.StatusOK, rec.Code)
		assert.Contains(t, rec.Header().Get("Content-Type"), "text/event-stream")
		assert.NotContains(t, rec.Body.String(), "response.completed")
	})

	t.Run("turn local continuation", func(t *testing.T) {
		life := newMemoryLife()
		var turns []int
		var acceptTurn, acceptInput int
		handler := testHandler(chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			turns = append(turns, len(req.Input))
			return chat.Outcome{}, emit(chat.Text("ok"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(func(ctx context.Context, turn *chat.TurnRequest) (chat.Acceptance, error) {
			acceptTurn = len(turn.Turn)
			acceptInput = len(turn.Request.Input)
			return life.Accept(ctx, turn)
		}), WithContinuationStore(life))
		first := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"a"}`, nil)
		require.Equal(t, http.StatusOK, first.Code)
		id := decodeResponse(t, first)["id"].(string)
		life.mu.Lock()
		firstRec := life.records[id]
		life.mu.Unlock()
		require.Len(t, firstRec.Turn, 1)
		assert.Equal(t, "a", firstRec.Turn[0].Content[0].Text)
		assert.Nil(t, firstRec.Response.Previous)
		assert.Equal(t, "agent/basic", firstRec.Response.Target)

		second := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"previous_response_id":"`+id+`","input":"b"}`, nil)
		require.Equal(t, http.StatusOK, second.Code)
		require.Equal(t, []int{1, 3}, turns)
		assert.Equal(t, 1, acceptTurn)
		assert.Equal(t, 3, acceptInput)
		life.mu.Lock()
		rec := life.records[id]
		life.mu.Unlock()
		// Fixed test ID is reused; final record is the latest turn only (not accumulated Input).
		assert.Len(t, rec.Turn, 1)
		assert.Equal(t, "b", rec.Turn[0].Content[0].Text)
		assert.Len(t, rec.Response.Output, 1)
		require.NotNil(t, rec.Response.Previous)
		assert.Equal(t, id, *rec.Response.Previous)
	})

	t.Run("accept keeps execution options for fingerprinting", func(t *testing.T) {
		life := newMemoryLife()
		var temp *float64
		var tools int
		var ext bool
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("ok"))
		}), chat.Capabilities{
			Continuation:       true,
			GenerationControls: chat.ControlTemperature | chat.ControlMaxOutputTokens,
			Tools:              true,
			Extensions:         map[string]bool{"x-durable": true},
		}, WithLifecycle(func(ctx context.Context, turn *chat.TurnRequest) (chat.Acceptance, error) {
			temp = turn.Request.Controls.Temperature
			tools = len(turn.Request.Controls.Tools)
			ext = turn.Request.Controls.Extensions["x-durable"] != nil
			return life.Accept(ctx, turn)
		}))
		body := `{"model":"agent/basic","store":true,"temperature":0.2,"tools":[{"type":"function","name":"ping","parameters":{"type":"object"}}],"x-durable":true,"input":"x"}`
		rec := postJSON(t, handler, "/responses", body, nil)
		require.Equal(t, http.StatusOK, rec.Code)
		require.NotNil(t, temp)
		assert.InDelta(t, 0.2, *temp, 1e-9)
		assert.Equal(t, 1, tools)
		assert.True(t, ext)
	})

	t.Run("activity events", func(t *testing.T) {
		life := newMemoryLife()
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			require.NoError(t, emit(chat.Activity("progress", jsontext.Value(`{"pct":10}`))))
			return chat.Outcome{}, emit(chat.Text("done"))
		}), chat.Capabilities{Continuation: true, Extensions: map[string]bool{"x-activity": true}}, WithLifecycle(life.Accept))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","stream":true,"store":true,"x-activity":true,"input":"x"}`, nil)
		require.Equal(t, http.StatusOK, rec.Code)
		assert.Contains(t, rec.Body.String(), "response.activity.progress")
		life.mu.Lock()
		out := life.records["resp_app"].Response.Output
		life.mu.Unlock()
		require.Len(t, out, 1)
		assert.Equal(t, chat.ItemMessage, out[0].Type)
	})

	t.Run("activity disabled by default", func(t *testing.T) {
		life := newMemoryLife()
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			require.Error(t, emit(chat.Activity("progress", jsontext.Value(`{"pct":10}`))))
			return chat.Outcome{}, emit(chat.Text("done"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","stream":true,"store":true,"input":"x"}`, nil)
		require.Equal(t, http.StatusOK, rec.Code)
		assert.NotContains(t, rec.Body.String(), "response.activity.")
	})

	t.Run("retrieval envelope", func(t *testing.T) {
		resp := chat.Response{
			ID:          "resp_get",
			Created:     7,
			Target:      "agent/basic",
			Status:      chat.StatusFailed,
			Output:      []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("hi"))},
			Error:       &chat.Error{Status: 400, Type: "invalid_request_error", Code: "test_code", Message: "public msg"},
			Incomplete:  "max_output_tokens",
			CompletedAt: 99,
			Metadata:    map[string]string{"k": "v"},
			Store:       true,
		}
		body, err := ResponsesBody(resp)
		require.NoError(t, err)
		value := jsonObject(t, body)
		assert.Equal(t, "resp_get", value["id"])
		assert.Equal(t, float64(7), value["created_at"])
		assert.Equal(t, "failed", value["status"])
		assert.Equal(t, true, value["store"])
		assert.Equal(t, float64(99), value["completed_at"])
		errObj := value["error"].(map[string]any)
		assert.Equal(t, "test_code", errObj["code"])
		assert.Equal(t, "public msg", errObj["message"])
		incomplete := value["incomplete_details"].(map[string]any)
		assert.Equal(t, "max_output_tokens", incomplete["reason"])
		assert.Equal(t, map[string]any{"k": "v"}, value["metadata"])
	})

	t.Run("failed response retrieval", func(t *testing.T) {
		resp := chat.Response{
			ID:          "resp_fail",
			Created:     10,
			Target:      "agent/basic",
			Status:      chat.StatusFailed,
			Error:       &chat.Error{Status: 503, Type: "server_error", Code: "upstream", Message: "service unavailable"},
			CompletedAt: 123,
			Store:       true,
		}
		body, err := ResponsesBody(resp)
		require.NoError(t, err)
		value := jsonObject(t, body)
		assert.Equal(t, "failed", value["status"])
		errObj := value["error"].(map[string]any)
		assert.Equal(t, "upstream", errObj["code"])
	})

	t.Run("failed replay ordinary", func(t *testing.T) {
		life := newMemoryLife()
		var calls atomic.Int32
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, _ chat.Emit) (chat.Outcome, error) {
			calls.Add(1)
			return chat.Outcome{}, &chat.Error{Status: 503, Type: "server_error", Code: "upstream", Message: "service unavailable"}
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		headers := map[string]string{"Idempotency-Key": "fail-ord"}
		first := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, headers)
		require.Equal(t, http.StatusServiceUnavailable, first.Code)
		second := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, headers)
		require.Equal(t, http.StatusOK, second.Code)
		assert.Equal(t, int32(1), calls.Load())
		body := decodeResponse(t, second)
		assert.Equal(t, "failed", body["status"])
		errObj := body["error"].(map[string]any)
		assert.Equal(t, "upstream", errObj["code"])
	})

	t.Run("failed replay streaming", func(t *testing.T) {
		life := newMemoryLife()
		var calls atomic.Int32
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, _ chat.Emit) (chat.Outcome, error) {
			calls.Add(1)
			return chat.Outcome{}, &chat.Error{Status: 503, Type: "server_error", Code: "upstream", Message: "service unavailable"}
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		headers := map[string]string{"Idempotency-Key": "fail-stream"}
		first := postJSON(t, handler, "/responses", `{"model":"agent/basic","stream":true,"store":true,"input":"x"}`, headers)
		require.Equal(t, http.StatusServiceUnavailable, first.Code)
		second := postJSON(t, handler, "/responses", `{"model":"agent/basic","stream":true,"store":true,"input":"x"}`, headers)
		require.Equal(t, http.StatusOK, second.Code)
		assert.Equal(t, int32(1), calls.Load())
		assert.Contains(t, second.Body.String(), "response.failed")
		assert.NotContains(t, second.Body.String(), `"status":"completed"`)
	})

	t.Run("incomplete retrieval and replay", func(t *testing.T) {
		resp := chat.Response{
			ID:          "resp_inc",
			Created:     5,
			Target:      "agent/basic",
			Status:      chat.StatusIncomplete,
			Incomplete:  "max_output_tokens",
			CompletedAt: 55,
			Store:       true,
		}
		body, err := ResponsesBody(resp)
		require.NoError(t, err)
		value := jsonObject(t, body)
		assert.Equal(t, "incomplete", value["status"])
		incomplete := value["incomplete_details"].(map[string]any)
		assert.Equal(t, "max_output_tokens", incomplete["reason"])

		life := newMemoryLife()
		var calls atomic.Int32
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, _ chat.Emit) (chat.Outcome, error) {
			calls.Add(1)
			return chat.Outcome{Status: chat.StatusIncomplete, StopReason: chat.StopLength}, nil
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		headers := map[string]string{"Idempotency-Key": "inc-replay"}
		first := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, headers)
		require.Equal(t, http.StatusOK, first.Code)
		second := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, headers)
		require.Equal(t, http.StatusOK, second.Code)
		assert.Equal(t, int32(1), calls.Load())
		replayBody := decodeResponse(t, second)
		assert.Equal(t, "incomplete", replayBody["status"])
	})

	t.Run("stable timestamps", func(t *testing.T) {
		life := newMemoryLife()
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("once"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		headers := map[string]string{"Idempotency-Key": "ts"}
		first := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, headers)
		require.Equal(t, http.StatusOK, first.Code)
		firstBody := decodeResponse(t, first)
		created := firstBody["created_at"]
		completed := firstBody["completed_at"]

		second := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, headers)
		require.Equal(t, http.StatusOK, second.Code)
		replayBody := decodeResponse(t, second)
		assert.Equal(t, created, replayBody["created_at"])
		assert.Equal(t, completed, replayBody["completed_at"])

		life.mu.Lock()
		stored := life.byKey["ts"].response
		life.mu.Unlock()
		retrieved, err := ResponsesBody(stored)
		require.NoError(t, err)
		retBody := jsonObject(t, retrieved)
		assert.EqualValues(t, completed, retBody["completed_at"])
	})

	t.Run("in progress retrieval", func(t *testing.T) {
		resp := chat.Response{ID: "resp_run", Created: 1, Target: "agent/basic", Status: chat.StatusInProgress, Store: true}
		body, err := ResponsesBody(resp)
		require.NoError(t, err)
		value := jsonObject(t, body)
		assert.Equal(t, "in_progress", value["status"])
		_, has := value["completed_at"]
		assert.False(t, has)
	})

	t.Run("metadata preserved on replay", func(t *testing.T) {
		life := newMemoryLife()
		var calls atomic.Int32
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			calls.Add(1)
			return chat.Outcome{}, emit(chat.Text("ok"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		headers := map[string]string{"Idempotency-Key": "meta"}
		first := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"metadata":{"original":"1"},"input":"x"}`, headers)
		require.Equal(t, http.StatusOK, first.Code)
		second := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"metadata":{"different":"2"},"input":"x"}`, headers)
		require.Equal(t, http.StatusOK, second.Code)
		assert.Equal(t, int32(1), calls.Load())
		body := decodeResponse(t, second)
		meta := body["metadata"].(map[string]any)
		assert.Equal(t, "1", meta["original"])
		_, ok := meta["different"]
		assert.False(t, ok)
	})

	t.Run("store default false omitted", func(t *testing.T) {
		life := newMemoryLife()
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("ok"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","input":"x"}`, nil)
		require.Equal(t, http.StatusOK, rec.Code)
		body := decodeResponse(t, rec)
		assert.Equal(t, false, body["store"])
		life.mu.Lock()
		defer life.mu.Unlock()
		assert.Empty(t, life.records)
		assert.Empty(t, life.history)
	})

	t.Run("store default true omitted", func(t *testing.T) {
		life := newMemoryLife()
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("ok"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept), WithStoreDefault(true))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","input":"x"}`, nil)
		require.Equal(t, http.StatusOK, rec.Code)
		body := decodeResponse(t, rec)
		assert.Equal(t, true, body["store"])
	})

	t.Run("store explicit false with default true", func(t *testing.T) {
		life := newMemoryLife()
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("ok"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept), WithStoreDefault(true))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":false,"input":"x"}`, nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)
	})

	t.Run("store explicit true with default false", func(t *testing.T) {
		life := newMemoryLife()
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("ok"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, nil)
		require.Equal(t, http.StatusOK, rec.Code)
		body := decodeResponse(t, rec)
		assert.Equal(t, true, body["store"])
	})

	t.Run("retain without lifecycle rejected", func(t *testing.T) {
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("ok"))
		}), chat.Capabilities{})
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)
		assert.Equal(t, "unsupported", responseError(t, rec)["code"])
	})

	t.Run("cleanup context finalize", func(t *testing.T) {
		life := newMemoryLife()
		life.boundCleanup = true
		handler := testHandler(chat.AgentFunc(func(ctx context.Context, _ *chat.Request, _ chat.Emit) (chat.Outcome, error) {
			<-ctx.Done()
			return chat.Outcome{Status: chat.StatusCancelled}, ctx.Err()
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		rec := postJSON(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			r = r.WithContext(context.WithValue(r.Context(), ctxKey{}, "cleanup"))
			handler.ServeHTTP(w, r)
		}), "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)
		assert.Equal(t, 1, life.finals)
	})

	t.Run("auth context", func(t *testing.T) {
		life := newMemoryLife()
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("ok"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		recorder := postJSON(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			r = r.WithContext(context.WithValue(r.Context(), ctxKey{}, "user-1"))
			handler.ServeHTTP(w, r)
		}), "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, nil)
		require.Equal(t, http.StatusOK, recorder.Code)
		assert.True(t, life.ctxSeen.Load())
	})

	t.Run("prefix mount", func(t *testing.T) {
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("ok"))
		}), chat.Capabilities{})
		mounted := http.StripPrefix("/api/v1", handler)
		rec := postJSON(t, mounted, "/api/v1/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
		require.Equal(t, http.StatusOK, rec.Code)
	})

	t.Run("durable disconnect", func(t *testing.T) {
		life := newMemoryLife()
		started := make(chan struct{})
		done := make(chan struct{})
		handler := testHandler(chat.AgentFunc(func(ctx context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			close(started)
			select {
			case <-ctx.Done():
				return chat.Outcome{Status: chat.StatusCancelled}, ctx.Err()
			case <-time.After(80 * time.Millisecond):
			}
			require.NoError(t, emit(chat.Text("late")))
			close(done)
			return chat.Outcome{}, nil
		}), chat.Capabilities{Continuation: true, Extensions: map[string]bool{"x-durable": true}}, WithLifecycle(life.Accept))

		httpCtx, httpCancel := context.WithCancel(context.Background())
		req := httptest.NewRequestWithContext(httpCtx, http.MethodPost, "/responses", strings.NewReader(`{"model":"agent/basic","stream":true,"store":true,"x-durable":true,"input":"x"}`))
		req.Header.Set("Content-Type", "application/json")
		recorder := httptest.NewRecorder()
		go func() {
			<-started
			httpCancel()
		}()
		handler.ServeHTTP(recorder, req)
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("durable agent did not finish")
		}
		assert.Equal(t, 1, life.finals)
	})

	t.Run("default cancel on disconnect", func(t *testing.T) {
		started := make(chan struct{})
		cancelled := make(chan struct{})
		handler := testHandler(chat.AgentFunc(func(ctx context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			close(started)
			<-ctx.Done()
			close(cancelled)
			return chat.Outcome{Status: chat.StatusCancelled}, ctx.Err()
		}), chat.Capabilities{})
		httpCtx, httpCancel := context.WithCancel(context.Background())
		req := httptest.NewRequestWithContext(httpCtx, http.MethodPost, "/chat/completions", strings.NewReader(`{"model":"agent/basic","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
		req.Header.Set("Content-Type", "application/json")
		go func() {
			<-started
			httpCancel()
		}()
		handler.ServeHTTP(httptest.NewRecorder(), req)
		select {
		case <-cancelled:
		case <-time.After(time.Second):
			t.Fatal("agent was not cancelled")
		}
	})

	t.Run("negative run timeout", func(t *testing.T) {
		life := newMemoryLife()
		life.negativeTimeout = true
		handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
			t.Fatal("agent must not run")
			return chat.Outcome{}, nil
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"x"}`, map[string]string{"Idempotency-Key": "neg"})
		require.Equal(t, http.StatusBadRequest, rec.Code)
		assert.Equal(t, 1, life.finals)
		life.mu.Lock()
		assert.False(t, life.running["neg"])
		life.mu.Unlock()
	})

	t.Run("store false skips content", func(t *testing.T) {
		life := newMemoryLife()
		life.allowNoStore = true
		handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("ephemeral"))
		}), chat.Capabilities{Continuation: true}, WithLifecycle(life.Accept))
		rec := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":false,"input":"x"}`, nil)
		require.Equal(t, http.StatusOK, rec.Code)
		assert.Equal(t, 1, life.finals)
		life.mu.Lock()
		defer life.mu.Unlock()
		assert.Empty(t, life.records)
		assert.Empty(t, life.history)
	})
}

func jsonObject(t *testing.T, v any) map[string]any {
	t.Helper()
	data, err := json.Marshal(v)
	require.NoError(t, err)
	var out map[string]any
	require.NoError(t, json.Unmarshal(data, &out))
	return out
}
