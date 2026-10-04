// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package main

import (
	"context"
	"encoding/json"
	"encoding/json/jsontext"
	"errors"
	"net/http/httptest"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStoreActions(t *testing.T) {
	app, err := newDemo()
	require.NoError(t, err)
	store := app.store

	initial := newTurn("client-correlation", "form-run", nil, []chat.Item{
		chat.MessageItem(chat.RoleUser, chat.TextPart("I need to contact the team.")),
	})
	accepted, err := store.Accept(context.Background(), initial)
	require.NoError(t, err)
	threadID := accepted.Response.Metadata["thread_id"]
	require.NotEmpty(t, threadID)
	form := accepted.Response.Clone()
	form.Status = chat.StatusCompleted
	form.Store = true
	form.Output = []chat.Item{uiItem("contact-form", fixtureForTest(t).Form)}
	require.NoError(t, accepted.Finish(context.Background(), &form, nil))

	valid := actionTurn(threadID, "action-run", form.ID, action{
		Kind: "ui_action", Name: "submit_contact", SurfaceID: "contact-form",
		SourceComponentID: "submit", Context: map[string]string{"name": "Ada", "email": "ada@example.test"},
		SourceResponseID: form.ID, SourceItemID: "contact-form",
	})
	forged := *valid
	forged.Request = cloneRequest(valid.Request)
	forged.Turn = cloneItems(valid.Turn)
	forged.Request.Input = cloneItems(valid.Request.Input)
	var forgedAction action
	require.NoError(t, json.Unmarshal(forged.Turn[0].Data, &forgedAction))
	forgedAction.SourceComponentID = "title"
	forged.Turn[0].Data = marshalAction(t, forgedAction)
	forged.Request.Input[0].Data = append(jsontext.Value(nil), forged.Turn[0].Data...)
	_, err = store.Accept(context.Background(), &forged)
	var apiErr *chat.Error
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, "invalid_request", apiErr.Code)
	assert.Contains(t, apiErr.Message, "stored source widget")
	assert.False(t, storedAction(chat.Response{Status: chat.StatusFailed, Output: form.Output}, action{
		SourceItemID: "contact-form", SurfaceID: "contact-form", SourceComponentID: "submit", Name: "submit_contact",
		Context: map[string]string{"name": "Ada", "email": "ada@example.test"},
	}), "a failed response cannot authorize an old widget")

	extraContext := actionTurn(threadID, "extra-context-run", form.ID, action{
		Kind: "ui_action", Name: "submit_contact", SurfaceID: "contact-form",
		SourceComponentID: "submit", Context: map[string]string{"name": "Ada", "email": "ada@example.test", "admin": "true"},
		SourceResponseID: form.ID, SourceItemID: "contact-form",
	})
	_, err = store.Accept(context.Background(), extraContext)
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, "invalid_request", apiErr.Code)
	assert.Contains(t, apiErr.Message, "stored source widget")

	// Reusing the rejected runId with the real stored widget remains admissible.
	accepted, err = store.Accept(context.Background(), valid)
	require.NoError(t, err)
	result := accepted.Response.Clone()
	result.Status = chat.StatusCompleted
	result.Store = true
	result.Previous = stringPointer(form.ID)
	result.Output = []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("saved"))}
	require.NoError(t, accepted.Finish(context.Background(), &result, nil))

	// Exact retries replay before checking the now-stale parent.
	accepted, err = store.Accept(context.Background(), valid)
	require.NoError(t, err)
	require.NotNil(t, accepted.Replay)
	assert.Equal(t, result.ID, accepted.Replay.ID)

	changed := actionTurn(threadID, "action-run", form.ID, action{
		Kind: "ui_action", Name: "submit_contact", SurfaceID: "contact-form",
		SourceComponentID: "submit", Context: map[string]string{"name": "Ada", "email": "changed@example.test"},
		SourceResponseID: form.ID, SourceItemID: "contact-form",
	})
	_, err = store.Accept(context.Background(), changed)
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, "idempotency_conflict", apiErr.Code)

	stale := actionTurn(threadID, "stale-run", form.ID, action{
		Kind: "ui_action", Name: "submit_contact", SurfaceID: "contact-form",
		SourceComponentID: "submit", Context: map[string]string{"name": "Ada", "email": "ada@example.test"},
		SourceResponseID: form.ID, SourceItemID: "contact-form",
	})
	_, err = store.Accept(context.Background(), stale)
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, "stale_parent", apiErr.Code)
}

func TestStoreFreshness(t *testing.T) {
	store := newMemoryStore()
	first := newTurn("client-correlation", "first-run", nil, []chat.Item{
		chat.MessageItem(chat.RoleUser, chat.TextPart("show the contact form")),
	})
	accepted, err := store.Accept(context.Background(), first)
	require.NoError(t, err)
	threadID := accepted.Response.Metadata["thread_id"]
	require.NotEmpty(t, threadID)
	active, err := store.Accept(context.Background(), first)
	require.NoError(t, err)
	require.NotNil(t, active.Replay)
	assert.Equal(t, accepted.Response.ID, active.Replay.ID)
	assert.Equal(t, chat.StatusInProgress, active.Replay.Status)
	assert.Equal(t, threadID, active.Replay.Metadata["thread_id"])

	changed := newTurn("client-correlation", "first-run", nil, []chat.Item{
		chat.MessageItem(chat.RoleUser, chat.TextPart("different request")),
	})
	_, err = store.Accept(context.Background(), changed)
	var apiErr *chat.Error
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, "idempotency_conflict", apiErr.Code)

	otherRun := newTurn("client-correlation", "other-run", nil, first.Turn)
	_, err = store.Accept(context.Background(), otherRun)
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, "thread_in_progress", apiErr.Code)

	form := accepted.Response.Clone()
	form.Status, form.Store = chat.StatusCompleted, true
	form.Output = []chat.Item{uiItem("contact-form", fixtureForTest(t).Form)}
	require.NoError(t, accepted.Finish(context.Background(), &form, nil))

	second := newTurn(threadID, "failed-run", stringPointer(form.ID), []chat.Item{
		chat.MessageItem(chat.RoleUser, chat.TextPart("try again later")),
	})
	next, err := store.Accept(context.Background(), second)
	require.NoError(t, err)
	failed := next.Response.Clone()
	failed.Status, failed.Store = chat.StatusFailed, true
	failed.Error = &chat.Error{Code: "agent_failed", Message: "temporary failure"}
	failed.Previous = stringPointer(form.ID)
	require.NoError(t, next.Finish(context.Background(), &failed, errors.New("private operational error")))

	oldFormAction := actionTurn(threadID, "old-form-run", form.ID, action{
		Kind: "ui_action", Name: "submit_contact", SurfaceID: "contact-form",
		SourceComponentID: "submit", Context: map[string]string{"name": "Ada", "email": "ada@example.test"},
		SourceResponseID: form.ID, SourceItemID: "contact-form",
	})
	_, err = store.Accept(context.Background(), oldFormAction)
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, "stale_parent", apiErr.Code)

	replayed, err := store.Accept(context.Background(), second)
	require.NoError(t, err)
	require.NotNil(t, replayed.Replay)
	assert.Equal(t, chat.StatusFailed, replayed.Replay.Status)
	assert.Equal(t, failed.ID, replayed.Replay.ID)

	_, err = store.Accept(context.Background(), newTurn(threadID, "missing-parent", nil, first.Turn))
	require.ErrorAs(t, err, &apiErr)
	assert.Equal(t, "stale_parent", apiErr.Code)
}

func TestStoreHistory(t *testing.T) {
	store := newMemoryStore()
	first := newTurn("client-correlation", "load-first", nil, []chat.Item{
		chat.MessageItem(chat.RoleUser, chat.TextPart("first")),
	})
	accepted, err := store.Accept(context.Background(), first)
	require.NoError(t, err)
	form := accepted.Response.Clone()
	form.Status, form.Store = chat.StatusCompleted, true
	form.Output = []chat.Item{uiItem("contact-form", fixtureForTest(t).Form)}
	require.NoError(t, accepted.Finish(context.Background(), &form, nil))

	second := newTurn(accepted.Response.Metadata["thread_id"], "load-second", stringPointer(form.ID), []chat.Item{
		chat.MessageItem(chat.RoleUser, chat.TextPart("second")),
	})
	next, err := store.Accept(context.Background(), second)
	require.NoError(t, err)
	result := next.Response.Clone()
	result.Status, result.Store = chat.StatusCompleted, true
	result.Previous = stringPointer(form.ID)
	result.Output = []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("done"))}
	require.NoError(t, next.Finish(context.Background(), &result, nil))

	history, err := store.Load(context.Background(), result.ID)
	require.NoError(t, err)
	assert.Len(t, history, 4)
	history[0].Content[0].Text = "mutated caller copy"
	again, err := store.Load(context.Background(), result.ID)
	require.NoError(t, err)
	assert.Equal(t, "first", again[0].Content[0].Text)
	assert.Equal(t, "done", again[3].Content[0].Text)

	_, err = store.Load(context.Background(), "absent")
	assert.ErrorContains(t, err, "was not found")
	store.mu.Lock()
	broken := store.records[result.ID]
	broken.response.Previous = stringPointer("absent-parent")
	store.records[result.ID] = broken
	store.mu.Unlock()
	_, err = store.Load(context.Background(), result.ID)
	assert.ErrorContains(t, err, "missing parent")
}

func TestStoreInvalid(t *testing.T) {
	store := newMemoryStore()
	_, err := store.Accept(context.Background(), newTurn("thread_999", "unknown-thread", nil, nil))
	var unknownThread *chat.Error
	require.ErrorAs(t, err, &unknownThread)
	assert.Equal(t, "stale_parent", unknownThread.Code)
	assert.Empty(t, store.threads, "an unknown server-issued thread ID cannot create a new conversation")

	cases := []struct {
		name string
		turn *chat.TurnRequest
		code string
	}{
		{name: "nil turn", code: "invalid_request"},
		{name: "nil request", turn: &chat.TurnRequest{}, code: "invalid_request"},
		{name: "not retained", turn: &chat.TurnRequest{Request: &chat.Request{}, Retain: false}, code: "unsupported"},
		{
			name: "missing extension",
			turn: func() *chat.TurnRequest {
				turn := newTurn("thread", "run", nil, nil)
				turn.Request.Controls.Extensions = nil
				return turn
			}(),
			code: "unsupported",
		},
		{name: "missing thread", turn: newTurn("", "run", nil, nil), code: "invalid_request"},
		{
			name: "invalid JSON fingerprint",
			turn: func() *chat.TurnRequest {
				turn := newTurn("thread", "run", nil, []chat.Item{{Type: chat.ItemExtension, Data: jsontext.Value("{")}})
				return turn
			}(),
			code: "error",
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			_, err := store.Accept(context.Background(), test.turn)
			require.Error(t, err)
			if test.code == "error" {
				assert.ErrorContains(t, err, "fingerprint accepted turn")
				return
			}
			var apiErr *chat.Error
			require.ErrorAs(t, err, &apiErr)
			assert.Equal(t, test.code, apiErr.Code)
		})
	}
}

func TestAgentHistory(t *testing.T) {
	ui := fixtureForTest(t)
	counts := &counts{}
	agent := &contactAgent{ui: ui, counts: counts}
	actionItem := chat.Item{Type: chat.ItemExtension, Data: marshalActionValue(action{
		Kind: "ui_action", Name: "submit_contact", SurfaceID: "contact-form", SourceComponentID: "submit",
		Context: map[string]string{"name": "Old", "email": "old@example.test"},
	})}
	request := &chat.Request{Input: []chat.Item{actionItem, chat.MessageItem(chat.RoleUser, chat.TextPart("new request"))}}
	var emitted []chat.Event
	_, err := agent.Run(context.Background(), request, func(event chat.Event) error {
		emitted = append(emitted, event)
		return nil
	})
	require.NoError(t, err)
	assert.EqualValues(t, 0, counts.effects.Load())
	assert.EqualValues(t, 1, counts.runs.Load())
	assert.Equal(t, chat.ItemExtension, emitted[len(emitted)-1].Item.Type)

	selected := marshalActionValue(action{
		Kind: "ui_action", Name: "submit_contact", SurfaceID: "contact-form", SourceComponentID: "submit",
		Context: map[string]string{"name": "Ada", "email": "ada@example.test"},
	})
	emitted = nil
	_, err = agent.Run(context.Background(), &chat.Request{Input: []chat.Item{{Type: chat.ItemExtension, Data: selected}}}, func(event chat.Event) error {
		emitted = append(emitted, event)
		return nil
	})
	require.NoError(t, err)
	assert.EqualValues(t, 1, counts.effects.Load())
	assert.Equal(t, chat.ItemExtension, emitted[len(emitted)-1].Item.Type)

	for _, items := range [][]chat.Item{
		{{Type: chat.ItemExtension, Data: jsontext.Value("{")}},
		{{Type: chat.ItemExtension, Data: jsontext.Value(`{"kind":"other"}`)}},
	} {
		found, err := actionFrom(items)
		assert.NoError(t, err)
		assert.Nil(t, found)
	}
	_, err = actionFrom([]chat.Item{{Type: chat.ItemExtension, Data: jsontext.Value(`{"kind":"ui_action","name":7}`)}})
	assert.ErrorContains(t, err, "decode UI action")

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = agent.Run(cancelled, &chat.Request{}, func(chat.Event) error { return nil })
	assert.ErrorIs(t, err, context.Canceled)

	_, err = agent.Run(context.Background(), &chat.Request{}, func(chat.Event) error { return errors.New("emit failed") })
	assert.ErrorContains(t, err, "emit failed")

	_, err = actionFrom([]chat.Item{
		{Type: chat.ItemExtension, Data: marshalActionValue(action{Kind: "ui_action"})},
		{Type: chat.ItemExtension, Data: marshalActionValue(action{Kind: "ui_action"})},
	})
	assert.ErrorContains(t, err, "only one UI action")
}

func TestDemoEndpoints(t *testing.T) {
	app, err := newDemo()
	require.NoError(t, err)
	for _, test := range []struct {
		path string
		code int
	}{
		{path: "/healthz", code: 204},
		{path: "/catalog.json", code: 200},
		{path: "/common_types.json", code: 200},
		{path: "/debug/state", code: 200},
	} {
		request := httptest.NewRequest("GET", test.path, nil)
		response := httptest.NewRecorder()
		app.handler.ServeHTTP(response, request)
		assert.Equal(t, test.code, response.Code)
	}
}

func TestCatalogReferences(t *testing.T) {
	raw, err := assets.ReadFile("catalog.json")
	require.NoError(t, err)
	var catalog struct {
		Components map[string]struct {
			AllOf []struct {
				Properties map[string]struct {
					Ref string `json:"$ref"`
				} `json:"properties"`
			} `json:"allOf"`
		} `json:"components"`
	}
	require.NoError(t, json.Unmarshal(raw, &catalog))
	for _, link := range []struct{ component, property, kind string }{
		{"Column", "children", "ChildList"},
		{"Card", "child", "ComponentId"},
		{"Button", "child", "ComponentId"},
	} {
		component, ok := catalog.Components[link.component]
		require.True(t, ok)
		require.Len(t, component.AllOf, 2)
		assert.Equal(t, "common_types.json#/$defs/"+link.kind, component.AllOf[1].Properties[link.property].Ref)
	}
	raw, err = assets.ReadFile("common_types.json")
	require.NoError(t, err)
	var common struct {
		Defs map[string]json.RawMessage `json:"$defs"`
	}
	require.NoError(t, json.Unmarshal(raw, &common))
	assert.Contains(t, common.Defs, "ComponentId")
	assert.Contains(t, common.Defs, "ChildList")
}

func newTurn(threadID, runID string, previous *string, items []chat.Item) *chat.TurnRequest {
	return &chat.TurnRequest{
		Thread:         threadID,
		Request:        &chat.Request{Target: target, Input: cloneItems(items), Controls: chat.Controls{Extensions: map[string]jsontext.Value{"x-ui": jsontext.Value(`true`)}}},
		Turn:           cloneItems(items),
		Previous:       previous,
		IdempotencyKey: runID,
		Retain:         true,
		Store:          boolPointer(true),
	}
}

func actionTurn(threadID, runID, previous string, selected action) *chat.TurnRequest {
	item := chat.Item{Type: chat.ItemExtension, ID: "action", Data: marshalActionValue(selected)}
	return newTurn(threadID, runID, stringPointer(previous), []chat.Item{item})
}

func marshalAction(t *testing.T, selected action) jsontext.Value {
	t.Helper()
	data, err := json.Marshal(selected)
	require.NoError(t, err)
	return jsontext.Value(data)
}

func marshalActionValue(selected action) jsontext.Value {
	data, _ := json.Marshal(selected)
	return jsontext.Value(data)
}

func cloneRequest(request *chat.Request) *chat.Request {
	copy := *request
	copy.Input = cloneItems(request.Input)
	return &copy
}

func fixtureForTest(t *testing.T) fixture {
	t.Helper()
	var ui fixture
	raw, err := assets.ReadFile("testdata/ui.json")
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(raw, &ui))
	return ui
}

func boolPointer(value bool) *bool { return &value }

func stringPointer(value string) *string { return &value }
