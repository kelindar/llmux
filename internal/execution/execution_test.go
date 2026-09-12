// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package execution

import (
	"context"
	"encoding/json/jsontext"
	"errors"
	"sync"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type stubAgent struct {
	run func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error)
}

func (a stubAgent) Run(ctx context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
	return a.run(ctx, req, emit)
}

func TestRun(t *testing.T) {
	agentErr := errors.New("agent failed")
	onEventErr := errors.New("handler rejected event")

	cases := map[string]struct {
		agent   stubAgent
		limits  chat.Limits
		onEvent func(chat.Event) error
		wantErr error
		check   func(t *testing.T, result Result, err error)
	}{
		"successTextDelta": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.Event{Type: chat.EventTextDelta, Delta: "hel"}))
				require.NoError(t, emit(chat.Event{Type: chat.EventTextDelta, Delta: "lo"}))
				return chat.Outcome{Status: chat.StatusCompleted, Usage: &chat.Usage{Input: 1, Output: 2, Total: 3}}, nil
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				require.Len(t, result.Items, 1)
				assert.Equal(t, "hello", result.Items[0].Content[0].Text)
				assert.Equal(t, chat.StopNormal, result.Outcome.StopReason)
			},
		},
		"agentError": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, _ chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusFailed}, agentErr
			}},
			wantErr: agentErr,
			check: func(t *testing.T, result Result, err error) {
				require.ErrorIs(t, err, agentErr)
				assert.Equal(t, chat.StatusFailed, result.Outcome.Status)
			},
		},
		"outcomeValidation": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, _ chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusInProgress}, nil
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "in_progress")
			},
		},
		"onEventError": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: chat.EventTextDelta, Delta: "x"})
			}},
			onEvent: func(chat.Event) error { return onEventErr },
			wantErr: onEventErr,
		},
		"toolCallFlow": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.Event{Type: chat.EventToolCallStart, CallID: "call_1", Name: "search"}))
				require.NoError(t, emit(chat.Event{Type: chat.EventToolCallDelta, CallID: "call_1", Delta: `{"q"`}))
				require.NoError(t, emit(chat.Event{Type: chat.EventToolCallDelta, CallID: "call_1", Delta: `:"x"}`}))
				require.NoError(t, emit(chat.Event{Type: chat.EventToolCallDone, CallID: "call_1"}))
				return chat.Outcome{Status: chat.StatusCompleted}, nil
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				require.Len(t, result.Items, 1)
				assert.Equal(t, `{"q":"x"}`, result.Items[0].Arguments)
				assert.Equal(t, chat.StopToolCall, result.Outcome.StopReason)
			},
		},
		"completeMessage": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{
					Type: chat.EventItem,
					Item: chat.MessageItem(chat.RoleAssistant, chat.TextPart("done")),
				})
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, "done", result.Items[0].Content[0].Text)
			},
		},
		"reasoningAndMedia": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.Reasoning("brief")))
				audio := chat.InlineMedia("audio/wav", []byte{1})
				audio.Format = "wav"
				require.NoError(t, emit(chat.MediaItem(chat.AudioPart(audio))))
				return chat.Outcome{Status: chat.StatusCompleted}, nil
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				require.Len(t, result.Items, 2)
				assert.Equal(t, chat.ItemReasoning, result.Items[0].Type)
				assert.Equal(t, chat.ItemMedia, result.Items[1].Type)
			},
		},
		"itemEvent": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{
					Type: chat.EventItem,
					Item: chat.MessageItem(chat.RoleAssistant, chat.TextPart("item")),
				})
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				require.Len(t, result.Items, 1)
			},
		},
		"duplicateItemID": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				item := chat.MessageItem(chat.RoleAssistant, chat.TextPart("one"))
				item.ID = "dup"
				require.NoError(t, emit(chat.Event{Type: chat.EventItem, Item: item}))
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: chat.EventItem, Item: item})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "duplicate")
			},
		},
		"invalidToolDone": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.Event{Type: chat.EventToolCallStart, CallID: "call_1", Name: "search"}))
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: chat.EventToolCallDone, CallID: "call_1"})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "invalid JSON")
			},
		},
		"textDoneClosesAccumulated": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.Event{Type: chat.EventTextDelta, Delta: "abc"}))
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.TextDone(""))
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, "abc", result.Items[0].Content[0].Text)
				assert.Equal(t, chat.StatusCompleted, result.Items[0].Status)
			},
		},
		"textDoneWithoutOpen": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.TextDone("msg_missing"))
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "not open")
			},
		},
		"directToolCall": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Tool("call_1", "search", `{"q":"x"}`))
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, chat.StopToolCall, result.Outcome.StopReason)
			},
		},
		"unknownEvent": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: "bogus"})
			}},
			check: func(t *testing.T, _ Result, err error) { require.Error(t, err) },
		},
		"duplicateToolCall": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.Event{Type: chat.EventToolCallStart, CallID: "call_1", Name: "search"}))
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: chat.EventToolCallStart, CallID: "call_1", Name: "search"})
			}},
			check: func(t *testing.T, _ Result, err error) { require.Error(t, err) },
		},
		"itemWhileToolOpen": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.Event{Type: chat.EventToolCallStart, CallID: "call_1", Name: "search"}))
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: chat.EventItem, Item: chat.MessageItem(chat.RoleAssistant, chat.TextPart("x"))})
			}},
			check: func(t *testing.T, _ Result, err error) { require.Error(t, err) },
		},
		"eventTooLarge": {
			limits: chat.Limits{MaxOutputBytes: 1 << 20, MaxEventBytes: 16, MaxMediaBytes: 1 << 20},
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: chat.EventItem, Item: chat.MessageItem(chat.RoleAssistant, chat.TextPart("too long for event limit"))})
			}},
			check: func(t *testing.T, _ Result, err error) {
				apiErr, ok := errors.AsType[*chat.Error](err)
				require.True(t, ok)
				assert.Equal(t, "event_too_large", apiErr.Code)
			},
		},
		"textDoneExplicitID": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.Event{Type: chat.EventTextDelta, ItemID: "msg_1", Delta: "ok"}))
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: chat.EventTextDone, ItemID: "msg_1"})
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, "ok", result.Items[0].Content[0].Text)
			},
		},
		"toolDoneUsesAccumulated": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.Event{Type: chat.EventToolCallStart, CallID: "call_1", Name: "search"}))
				require.NoError(t, emit(chat.Event{Type: chat.EventToolCallDelta, CallID: "call_1", Delta: `{"q":"x"}`}))
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.ToolDone("call_1"))
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, `{"q":"x"}`, result.Items[0].Arguments)
			},
		},
		"finishOpenTool": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.Event{Type: chat.EventToolCallStart, CallID: "call_1", Name: "search"}))
				require.NoError(t, emit(chat.Event{Type: chat.EventToolCallDelta, CallID: "call_1", Delta: `{"q":"x"}`}))
				return chat.Outcome{Status: chat.StatusCompleted}, nil
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, chat.StatusCompleted, result.Items[0].Status)
			},
		},
		"outputTooLarge": {
			limits: chat.Limits{MaxOutputBytes: 4, MaxEventBytes: 1 << 20, MaxMediaBytes: 1 << 20},
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: chat.EventTextDelta, Delta: "toolong"})
			}},
			check: func(t *testing.T, _ Result, err error) {
				apiErr, ok := errors.AsType[*chat.Error](err)
				require.True(t, ok)
				assert.Equal(t, "output_too_large", apiErr.Code)
			},
		},
		"switchTextItem": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.Event{Type: chat.EventTextDelta, ItemID: "msg_a", Delta: "a"}))
				require.NoError(t, emit(chat.Event{Type: chat.EventTextDelta, ItemID: "msg_b", Delta: "b"}))
				return chat.Outcome{Status: chat.StatusCompleted}, nil
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				require.Len(t, result.Items, 2)
			},
		},
		"badFunctionCallItem": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: chat.EventItem, Item: chat.Item{Type: chat.ItemFunctionCall}})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "tool call requires call_id, name, and valid JSON arguments")
			},
		},
		"badMediaItem": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.MediaItem(chat.TextPart("nope")))
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "media item must contain image or audio")
			},
		},
		"emptyEventType": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "event type is required")
			},
		},
		"finishOnEventError": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.Event{Type: chat.EventTextDelta, Delta: "open"}))
				return chat.Outcome{Status: chat.StatusCompleted}, nil
			}},
			onEvent: func(event chat.Event) error {
				if event.Type == chat.EventTextDone {
					return onEventErr
				}
				return nil
			},
			wantErr: onEventErr,
		},
		"cancelledOutcome": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, _ chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCancelled}, nil
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, chat.StopCancelled, result.Outcome.StopReason)
			},
		},
		"unknownToolDelta": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: chat.EventToolCallDelta, CallID: "missing", Delta: "x"})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "not open")
			},
		},
		"imageMedia": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				img := chat.InlineMedia("image/png", []byte{1, 2, 3})
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.MediaItem(chat.ImagePart(img)))
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, chat.ItemMedia, result.Items[0].Type)
			},
		},
		"reasoningRich": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				item := chat.Item{
					Type:             chat.ItemReasoning,
					Summary:          []chat.Part{chat.SummaryPart("brief")},
					EncryptedContent: []byte("enc"),
					Data:             []byte("raw"),
				}
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: chat.EventItem, Item: item})
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, chat.ItemReasoning, result.Items[0].Type)
			},
		},
		"textItemNotOpen": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				item := chat.MessageItem(chat.RoleAssistant, chat.TextPart("done"))
				item.ID = "msg_closed"
				item.Status = chat.StatusCompleted
				require.NoError(t, emit(chat.Event{Type: chat.EventItem, Item: item}))
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: chat.EventTextDelta, ItemID: "msg_closed", Delta: "x"})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "not open")
			},
		},
		"toolStartMissingID": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: chat.EventToolCallStart, Name: "search"})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "call_id")
			},
		},
		"invalidDirectToolCall": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Tool("call_1", "search", "not-json"))
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "valid JSON arguments")
			},
		},
		"functionOutputItem": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				img := chat.InlineMedia("image/png", []byte{1, 2})
				item := chat.FunctionCallOutputItem("call_1", chat.TextPart("done"), chat.ImagePart(img))
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{Type: chat.EventItem, Item: item})
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, chat.ItemFunctionCallOutput, result.Items[0].Type)
			},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			limits := tc.limits
			if limits.MaxOutputBytes == 0 && limits.MaxEventBytes == 0 {
				limits = chat.DefaultLimits()
			}
			onEvent := tc.onEvent
			result, err := Run(context.Background(), &chat.Request{}, tc.agent, limits, onEvent)
			if tc.wantErr != nil {
				require.ErrorIs(t, err, tc.wantErr)
			}
			if tc.check != nil {
				tc.check(t, result, err)
			}
		})
	}

	t.Run("contextCancel", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		agent := stubAgent{run: func(ctx context.Context, _ *chat.Request, _ chat.Emit) (chat.Outcome, error) {
			<-ctx.Done()
			return chat.Outcome{}, ctx.Err()
		}}
		_, err := Run(ctx, &chat.Request{}, agent, chat.DefaultLimits(), nil)
		require.ErrorIs(t, err, context.Canceled)
	})

	t.Run("emitClosed", func(t *testing.T) {
		var saved chat.Emit
		agent := stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			saved = emit
			return chat.Outcome{Status: chat.StatusCompleted}, nil
		}}
		_, err := Run(context.Background(), &chat.Request{}, agent, chat.DefaultLimits(), nil)
		require.NoError(t, err)
		require.ErrorIs(t, saved(chat.Event{Type: chat.EventTextDelta, Delta: "x"}), chat.ErrEmitClosed)
	})

	t.Run("concurrentEmit", func(t *testing.T) {
		started := make(chan struct{})
		block := make(chan struct{})
		agent := stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			var wg sync.WaitGroup
			wg.Add(1)
			var firstErr error
			go func() {
				defer wg.Done()
				firstErr = emit(chat.Event{Type: chat.EventTextDelta, Delta: "a"})
			}()
			<-started
			secondErr := emit(chat.Event{Type: chat.EventTextDelta, Delta: "b"})
			close(block)
			wg.Wait()
			if secondErr != nil {
				return chat.Outcome{}, secondErr
			}
			return chat.Outcome{Status: chat.StatusCompleted}, firstErr
		}}
		once := sync.Once{}
		_, err := Run(context.Background(), &chat.Request{}, agent, chat.DefaultLimits(), func(chat.Event) error {
			once.Do(func() { started <- struct{}{} })
			<-block
			return nil
		})
		require.ErrorIs(t, err, chat.ErrConcurrentEmit)
	})
}

func TestFunctionCallItem(t *testing.T) {
	item := chat.FunctionCallItem("call_1", "search", `{"q":"x"}`)
	assert.Equal(t, chat.ItemFunctionCall, item.Type)
	assert.Equal(t, "call_1", item.CallID)
}

func TestEventStateEdges(t *testing.T) {
	t.Run("byte limits", func(t *testing.T) {
		state := newEventState(chat.Limits{MaxOutputBytes: 8, MaxEventBytes: 4})
		require.Error(t, state.addBytes(-1))
		require.NoError(t, state.addBytes(2))
		require.Error(t, state.addBytes(7))
		require.Error(t, state.addBytes(5))
	})

	t.Run("item validation", func(t *testing.T) {
		state := newEventState(chat.DefaultLimits())
		_, err := state.apply(chat.OutputItem(chat.MessageItem(chat.Role("invalid"), chat.TextPart("x"))))
		require.Error(t, err)

		_, err = state.apply(chat.OutputItem(chat.MessageItem(chat.RoleUser, chat.TextPart("x"))))
		require.Error(t, err)
	})

	t.Run("unchecked items", func(t *testing.T) {
		state := newEventState(chat.DefaultLimits())
		require.Error(t, state.addItemUnchecked(chat.Item{Type: chat.ItemExtension}))

		item := chat.Item{ID: "item_1", Type: chat.ItemExtension, Data: jsontext.Value(`{"x":1}`)}
		require.NoError(t, state.addItemUnchecked(item))
		require.Error(t, state.addItemUnchecked(item))
	})

	t.Run("text initializes missing content", func(t *testing.T) {
		state := newEventState(chat.DefaultLimits())
		require.NoError(t, state.addItemUnchecked(chat.Item{
			ID:     "msg_1",
			Type:   chat.ItemMessage,
			Status: chat.StatusInProgress,
			Role:   chat.RoleAssistant,
		}))

		events, err := state.apply(chat.Event{Type: chat.EventTextDelta, ItemID: "msg_1", Delta: "hello"})
		require.NoError(t, err)
		require.Len(t, events, 1)
		assert.Equal(t, "hello", state.items[0].Content[0].Text)

		state.textOpen = "msg_1"
		_, err = state.closeText("other")
		require.Error(t, err)
	})

	t.Run("tool lifecycle errors", func(t *testing.T) {
		state := newEventState(chat.DefaultLimits())
		_, err := state.apply(chat.Event{Type: chat.EventToolCallStart, Name: "search"})
		require.Error(t, err)
		_, err = state.apply(chat.Event{Type: chat.EventToolCallStart, CallID: "call_1"})
		require.Error(t, err)

		require.NoError(t, func() error {
			_, err := state.apply(chat.ToolStart("call_1", "search"))
			return err
		}())
		_, err = state.apply(chat.ToolStart("call_1", "search"))
		require.Error(t, err)
		_, err = state.apply(chat.ToolDelta("missing", "{}"))
		require.Error(t, err)
		_, err = state.apply(chat.ToolDone("missing"))
		require.Error(t, err)

		limited := newEventState(chat.Limits{MaxOutputBytes: 1, MaxEventBytes: 1024, MaxMediaBytes: 1024})
		_, err = limited.apply(chat.ToolStart("call_2", "search"))
		require.NoError(t, err)
		_, err = limited.apply(chat.ToolDelta("call_2", "{}"))
		require.Error(t, err)
	})

	t.Run("item shapes", func(t *testing.T) {
		state := newEventState(chat.DefaultLimits())
		for _, content := range [][]chat.Part{
			nil,
			{chat.TextPart("one"), chat.TextPart("two")},
			{{Type: chat.PartImage}},
		} {
			_, err := state.apply(chat.OutputItem(chat.Item{Type: chat.ItemMedia, Content: content}))
			require.Error(t, err)
		}

		extension := chat.Item{Type: chat.ItemExtension, ID: "ext_1", Data: jsontext.Value(`{"x":1}`)}
		events, err := state.apply(chat.OutputItem(extension))
		require.NoError(t, err)
		require.Len(t, events, 1)
		assert.Equal(t, "ext_1", events[0].ItemID)

		_, err = state.apply(chat.OutputItem(chat.Item{Type: chat.ItemType("unknown"), ID: "unknown"}))
		require.Error(t, err)
	})

	t.Run("activity and finish", func(t *testing.T) {
		state := newEventState(chat.DefaultLimits())
		_, err := state.apply(chat.Activity("", jsontext.Value(`{}`)))
		require.Error(t, err)
		_, err = state.apply(chat.Activity("trace", jsontext.Value(`bad`)))
		require.Error(t, err)
		events, err := state.apply(chat.Activity("trace", jsontext.Value(`{"step":1}`)))
		require.NoError(t, err)
		require.Len(t, events, 1)

		limited := newEventState(chat.Limits{MaxOutputBytes: 2, MaxEventBytes: 1024, MaxMediaBytes: 1024})
		_, err = limited.apply(chat.Activity("trace", jsontext.Value(`{"step":1}`)))
		require.Error(t, err)

		open := newEventState(chat.DefaultLimits())
		_, err = open.apply(chat.ToolStart("call_3", "search"))
		require.NoError(t, err)
		_, err = open.finish()
		require.Error(t, err)

		valid := newEventState(chat.DefaultLimits())
		_, err = valid.apply(chat.ToolStart("call_4", "search"))
		require.NoError(t, err)
		_, err = valid.apply(chat.ToolDelta("call_4", `{}`))
		require.NoError(t, err)
		events, err = valid.finish()
		require.NoError(t, err)
		require.Len(t, events, 1)
		events, err = valid.finish()
		require.NoError(t, err)
		assert.Empty(t, events)
	})
}

func TestClosure(t *testing.T) {
	tests := map[string]struct {
		agent stubAgent
		check func(t *testing.T, result Result, err error)
	}{
		"empty output": {
			agent: stubAgent{run: func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, nil
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Empty(t, result.Items)
				assert.Equal(t, chat.StopNormal, result.Outcome.StopReason)
			},
		},
		"interleaved text ids": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.Event{Type: chat.EventTextDelta, ItemID: "msg_a", Delta: "A"}))
				require.NoError(t, emit(chat.Event{Type: chat.EventTextDelta, ItemID: "msg_b", Delta: "B"}))
				require.NoError(t, emit(chat.ToolStart("call_1", "search")))
				require.NoError(t, emit(chat.ToolDelta("call_1", `{}`)))
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.ToolDone("call_1"))
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				require.Len(t, result.Items, 3)
				assert.Equal(t, "msg_a", result.Items[0].ID)
				assert.Equal(t, "A", result.Items[0].Content[0].Text)
				assert.Equal(t, chat.StatusCompleted, result.Items[0].Status)
				assert.Equal(t, "msg_b", result.Items[1].ID)
				assert.Equal(t, "B", result.Items[1].Content[0].Text)
				assert.Equal(t, chat.StatusCompleted, result.Items[1].Status)
				assert.Equal(t, chat.ItemFunctionCall, result.Items[2].Type)
				assert.Equal(t, `{}`, result.Items[2].Arguments)
			},
		},
		"automatic text closure": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.TextDelta("open")))
				return chat.Outcome{Status: chat.StatusCompleted}, nil
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				require.Len(t, result.Items, 1)
				assert.Equal(t, "open", result.Items[0].Content[0].Text)
				assert.Equal(t, chat.StatusCompleted, result.Items[0].Status)
			},
		},
		"malformed tool arguments": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				require.NoError(t, emit(chat.ToolStart("call_1", "search")))
				require.NoError(t, emit(chat.ToolDelta("call_1", "{bad")))
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.ToolDone("call_1"))
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "invalid JSON")
			},
		},
		"canonical reasoning": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Reasoning("think"))
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				require.Len(t, result.Items, 1)
				assert.Equal(t, chat.ItemReasoning, result.Items[0].Type)
				require.Len(t, result.Items[0].Summary, 1)
				assert.Equal(t, "think", result.Items[0].Summary[0].Text)
			},
		},
		"reasoning without summary": {
			agent: stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Event{
					Type: chat.EventItem,
					Item: chat.Item{Type: chat.ItemReasoning, Status: chat.StatusCompleted},
				})
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				require.Len(t, result.Items, 1)
				assert.Equal(t, chat.ItemReasoning, result.Items[0].Type)
				assert.Empty(t, result.Items[0].Summary)
			},
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			result, err := Run(context.Background(), &chat.Request{}, tc.agent, chat.DefaultLimits(), nil)
			tc.check(t, result, err)
		})
	}
}

func TestOutputOwnership(t *testing.T) {
	t.Run("agent mutates after emit", func(t *testing.T) {
		media := chat.InlineMedia("image/png", []byte{1, 2, 3})
		agent := stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			item := chat.Item{
				Type:    chat.ItemMedia,
				Status:  chat.StatusCompleted,
				Role:    chat.RoleAssistant,
				Content: []chat.Part{chat.ImagePart(media)},
			}
			require.NoError(t, emit(chat.Event{Type: chat.EventItem, Item: item}))
			item.Content[0].Media.Data[0] = 9
			return chat.Outcome{Status: chat.StatusCompleted}, nil
		}}
		result, err := Run(context.Background(), &chat.Request{}, agent, chat.DefaultLimits(), nil)
		require.NoError(t, err)
		require.Len(t, result.Items, 1)
		assert.Equal(t, byte(1), result.Items[0].Content[0].Media.Data[0])
	})

	t.Run("result owns transferred items", func(t *testing.T) {
		agent := stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Text("owned"))
		}}
		result, err := Run(context.Background(), &chat.Request{}, agent, chat.DefaultLimits(), nil)
		require.NoError(t, err)
		require.Len(t, result.Items, 1)
		result.Items[0].Content[0].Text = "caller"
		assert.Equal(t, "caller", result.Items[0].Content[0].Text)
	})

	t.Run("event item delivery is a borrow", func(t *testing.T) {
		agent := stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{Status: chat.StatusCompleted}, emit(chat.Text("hello"))
		}}
		result, err := Run(context.Background(), &chat.Request{}, agent, chat.DefaultLimits(), func(event chat.Event) error {
			if event.Type == chat.EventItem && len(event.Item.Content) > 0 {
				event.Item.Content[0].Text = "borrowed"
			}
			return nil
		})
		require.NoError(t, err)
		require.Len(t, result.Items, 1)
		assert.Equal(t, "borrowed", result.Items[0].Content[0].Text)
	})
}

func BenchmarkEventApply(b *testing.B) {
	agent := stubAgent{run: func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		require.NoError(b, emit(chat.ToolStart("call_1", "search")))
		require.NoError(b, emit(chat.ToolDelta("call_1", `{"q":"`)))
		require.NoError(b, emit(chat.ToolDelta("call_1", `x"}`)))
		require.NoError(b, emit(chat.ToolDone("call_1")))
		require.NoError(b, emit(chat.TextDelta("hello")))
		require.NoError(b, emit(chat.TextDone("")))
		return chat.Outcome{Status: chat.StatusCompleted}, nil
	}}
	b.ReportAllocs()
	for b.Loop() {
		_, err := Run(context.Background(), &chat.Request{}, agent, chat.DefaultLimits(), nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}
