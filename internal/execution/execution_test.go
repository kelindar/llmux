package execution

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/kelindar/llmux/contract"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type stubAgent struct {
	run func(context.Context, *contract.Request, contract.Emit) (contract.Outcome, error)
}

func (a stubAgent) Run(ctx context.Context, req *contract.Request, emit contract.Emit) (contract.Outcome, error) {
	return a.run(ctx, req, emit)
}

func TestRun(t *testing.T) {
	agentErr := errors.New("agent failed")
	onEventErr := errors.New("handler rejected event")

	cases := map[string]struct {
		agent   stubAgent
		limits  contract.Limits
		onEvent func(contract.Event) error
		wantErr error
		check   func(t *testing.T, result Result, err error)
	}{
		"successTextDelta": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				require.NoError(t, emit(contract.Event{Type: contract.EventTextDelta, Delta: "hel"}))
				require.NoError(t, emit(contract.Event{Type: contract.EventTextDelta, Delta: "lo"}))
				return contract.Outcome{Status: contract.StatusCompleted, Usage: &contract.Usage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3}}, nil
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				require.Len(t, result.Items, 1)
				assert.Equal(t, "hello", result.Items[0].Content[0].Text)
				assert.Equal(t, contract.StopStop, result.Outcome.StopReason)
			},
		},
		"agentError": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, _ contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusFailed}, agentErr
			}},
			wantErr: agentErr,
			check: func(t *testing.T, result Result, err error) {
				require.ErrorIs(t, err, agentErr)
				assert.Equal(t, contract.StatusFailed, result.Outcome.Status)
			},
		},
		"outcomeValidation": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, _ contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusInProgress}, nil
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "in_progress")
			},
		},
		"onEventError": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventTextDelta, Delta: "x"})
			}},
			onEvent: func(contract.Event) error { return onEventErr },
			wantErr: onEventErr,
		},
		"toolCallFlow": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				require.NoError(t, emit(contract.Event{Type: contract.EventToolCallStart, CallID: "call_1", Name: "search"}))
				require.NoError(t, emit(contract.Event{Type: contract.EventToolCallDelta, CallID: "call_1", Delta: `{"q"`}))
				require.NoError(t, emit(contract.Event{Type: contract.EventToolCallDelta, CallID: "call_1", Delta: `:"x"}`}))
				require.NoError(t, emit(contract.Event{Type: contract.EventToolCallDone, CallID: "call_1"}))
				return contract.Outcome{Status: contract.StatusCompleted}, nil
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				require.Len(t, result.Items, 1)
				assert.Equal(t, `{"q":"x"}`, result.Items[0].Arguments)
				assert.Equal(t, contract.StopToolCall, result.Outcome.StopReason)
			},
		},
		"completeMessage": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{
					Type: contract.EventMessage,
					Item: contract.MessageItem(contract.RoleAssistant, contract.TextPart("done")),
				})
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, "done", result.Items[0].Content[0].Text)
			},
		},
		"reasoningAndMedia": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				require.NoError(t, emit(contract.Event{Type: contract.EventReasoning, Text: "brief"}))
				audio := contract.InlineMedia("audio/wav", []byte{1})
				audio.Format = "wav"
				require.NoError(t, emit(contract.Event{Type: contract.EventMedia, Part: contract.AudioPart(audio)}))
				return contract.Outcome{Status: contract.StatusCompleted}, nil
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				require.Len(t, result.Items, 2)
				assert.Equal(t, contract.ItemReasoning, result.Items[0].Type)
				assert.Equal(t, contract.ItemMedia, result.Items[1].Type)
			},
		},
		"itemEvent": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{
					Type: contract.EventItem,
					Item: contract.MessageItem(contract.RoleAssistant, contract.TextPart("item")),
				})
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				require.Len(t, result.Items, 1)
			},
		},
		"duplicateItemID": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				item := contract.MessageItem(contract.RoleAssistant, contract.TextPart("one"))
				item.ID = "dup"
				require.NoError(t, emit(contract.Event{Type: contract.EventItem, Item: item}))
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventItem, Item: item})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "duplicate")
			},
		},
		"invalidToolDone": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				require.NoError(t, emit(contract.Event{Type: contract.EventToolCallStart, CallID: "call_1", Name: "search"}))
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventToolCallDone, CallID: "call_1"})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "invalid JSON")
			},
		},
		"textDoneMismatch": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				require.NoError(t, emit(contract.Event{Type: contract.EventTextDelta, Delta: "abc"}))
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventTextDone, Text: "wrong"})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "does not match")
			},
		},
		"directToolCall": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventToolCall, CallID: "call_1", Name: "search", Arguments: `{"q":"x"}`})
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, contract.StopToolCall, result.Outcome.StopReason)
			},
		},
		"unknownEvent": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: "bogus"})
			}},
			check: func(t *testing.T, _ Result, err error) { require.Error(t, err) },
		},
		"duplicateToolCall": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				require.NoError(t, emit(contract.Event{Type: contract.EventToolCallStart, CallID: "call_1", Name: "search"}))
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventToolCallStart, CallID: "call_1", Name: "search"})
			}},
			check: func(t *testing.T, _ Result, err error) { require.Error(t, err) },
		},
		"itemWhileToolOpen": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				require.NoError(t, emit(contract.Event{Type: contract.EventToolCallStart, CallID: "call_1", Name: "search"}))
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventItem, Item: contract.MessageItem(contract.RoleAssistant, contract.TextPart("x"))})
			}},
			check: func(t *testing.T, _ Result, err error) { require.Error(t, err) },
		},
		"eventTooLarge": {
			limits: contract.Limits{MaxOutputBytes: 1 << 20, MaxEventBytes: 16, MaxMediaBytes: 1 << 20},
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventMessage, Item: contract.MessageItem(contract.RoleAssistant, contract.TextPart("too long for event limit"))})
			}},
			check: func(t *testing.T, _ Result, err error) {
				apiErr, ok := errors.AsType[*contract.APIError](err)
				require.True(t, ok)
				assert.Equal(t, "event_too_large", apiErr.Code)
			},
		},
		"textDoneExplicitID": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				require.NoError(t, emit(contract.Event{Type: contract.EventTextDelta, ItemID: "msg_1", Delta: "ok"}))
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventTextDone, ItemID: "msg_1"})
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, "ok", result.Items[0].Content[0].Text)
			},
		},
		"toolArgsMismatch": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				require.NoError(t, emit(contract.Event{Type: contract.EventToolCallStart, CallID: "call_1", Name: "search"}))
				require.NoError(t, emit(contract.Event{Type: contract.EventToolCallDelta, CallID: "call_1", Delta: `{"q":"x"}`}))
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventToolCallDone, CallID: "call_1", Arguments: `{"q":"y"}`})
			}},
			check: func(t *testing.T, _ Result, err error) { require.Error(t, err) },
		},
		"finishOpenTool": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				require.NoError(t, emit(contract.Event{Type: contract.EventToolCallStart, CallID: "call_1", Name: "search"}))
				require.NoError(t, emit(contract.Event{Type: contract.EventToolCallDelta, CallID: "call_1", Delta: `{"q":"x"}`}))
				return contract.Outcome{Status: contract.StatusCompleted}, nil
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, contract.StatusCompleted, result.Items[0].Status)
			},
		},
		"outputTooLarge": {
			limits: contract.Limits{MaxOutputBytes: 4, MaxEventBytes: 1 << 20, MaxMediaBytes: 1 << 20},
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventTextDelta, Delta: "toolong"})
			}},
			check: func(t *testing.T, _ Result, err error) {
				apiErr, ok := errors.AsType[*contract.APIError](err)
				require.True(t, ok)
				assert.Equal(t, "output_too_large", apiErr.Code)
			},
		},
		"switchTextItem": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				require.NoError(t, emit(contract.Event{Type: contract.EventTextDelta, ItemID: "msg_a", Delta: "a"}))
				require.NoError(t, emit(contract.Event{Type: contract.EventTextDelta, ItemID: "msg_b", Delta: "b"}))
				return contract.Outcome{Status: contract.StatusCompleted}, nil
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				require.Len(t, result.Items, 2)
			},
		},
		"badMessageEvent": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventMessage, Item: contract.Item{Type: contract.ItemFunctionCall}})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "message event")
			},
		},
		"badMediaEvent": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventMedia, Part: contract.TextPart("nope")})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "media event")
			},
		},
		"emptyEventType": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "event type is required")
			},
		},
		"finishOnEventError": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				require.NoError(t, emit(contract.Event{Type: contract.EventTextDelta, Delta: "open"}))
				return contract.Outcome{Status: contract.StatusCompleted}, nil
			}},
			onEvent: func(event contract.Event) error {
				if event.Type == contract.EventTextDone {
					return onEventErr
				}
				return nil
			},
			wantErr: onEventErr,
		},
		"cancelledOutcome": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, _ contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusCancelled}, nil
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, contract.StopCancelled, result.Outcome.StopReason)
			},
		},
		"unknownToolDelta": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventToolCallDelta, CallID: "missing", Delta: "x"})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "not open")
			},
		},
		"imageMedia": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				img := contract.InlineMedia("image/png", []byte{1, 2, 3})
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventMedia, Part: contract.ImagePart(img)})
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, contract.ItemMedia, result.Items[0].Type)
			},
		},
		"reasoningRich": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				item := contract.Item{
					Type:             contract.ItemReasoning,
					Summary:          []contract.Part{contract.SummaryPart("brief")},
					EncryptedContent: []byte("enc"),
					Data:             []byte("raw"),
				}
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventReasoning, Item: item})
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, contract.ItemReasoning, result.Items[0].Type)
			},
		},
		"textItemNotOpen": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				item := contract.MessageItem(contract.RoleAssistant, contract.TextPart("done"))
				item.ID = "msg_closed"
				item.Status = contract.StatusCompleted
				require.NoError(t, emit(contract.Event{Type: contract.EventItem, Item: item}))
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventTextDelta, ItemID: "msg_closed", Delta: "x"})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "not open")
			},
		},
		"toolStartMissingID": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventToolCallStart, Name: "search"})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "call_id")
			},
		},
		"invalidDirectToolCall": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventToolCall, CallID: "call_1", Name: "search"})
			}},
			check: func(t *testing.T, _ Result, err error) {
				require.Error(t, err)
				assert.Contains(t, err.Error(), "valid JSON arguments")
			},
		},
		"functionOutputItem": {
			agent: stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
				img := contract.InlineMedia("image/png", []byte{1, 2})
				item := contract.FunctionCallOutputItem("call_1", contract.TextPart("done"), contract.ImagePart(img))
				return contract.Outcome{Status: contract.StatusCompleted}, emit(contract.Event{Type: contract.EventItem, Item: item})
			}},
			check: func(t *testing.T, result Result, err error) {
				require.NoError(t, err)
				assert.Equal(t, contract.ItemFunctionCallOutput, result.Items[0].Type)
			},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			limits := tc.limits
			if limits.MaxOutputBytes == 0 && limits.MaxEventBytes == 0 {
				limits = contract.DefaultLimits()
			}
			onEvent := tc.onEvent
			result, err := Run(context.Background(), &contract.Request{}, tc.agent, limits, onEvent)
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
		agent := stubAgent{run: func(ctx context.Context, _ *contract.Request, _ contract.Emit) (contract.Outcome, error) {
			<-ctx.Done()
			return contract.Outcome{}, ctx.Err()
		}}
		_, err := Run(ctx, &contract.Request{}, agent, contract.DefaultLimits(), nil)
		require.ErrorIs(t, err, context.Canceled)
	})

	t.Run("emitClosed", func(t *testing.T) {
		var saved contract.Emit
		agent := stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
			saved = emit
			return contract.Outcome{Status: contract.StatusCompleted}, nil
		}}
		_, err := Run(context.Background(), &contract.Request{}, agent, contract.DefaultLimits(), nil)
		require.NoError(t, err)
		require.ErrorIs(t, saved(contract.Event{Type: contract.EventTextDelta, Delta: "x"}), contract.ErrEmitClosed)
	})

	t.Run("concurrentEmit", func(t *testing.T) {
		started := make(chan struct{})
		block := make(chan struct{})
		agent := stubAgent{run: func(_ context.Context, _ *contract.Request, emit contract.Emit) (contract.Outcome, error) {
			var wg sync.WaitGroup
			wg.Add(1)
			var firstErr error
			go func() {
				defer wg.Done()
				firstErr = emit(contract.Event{Type: contract.EventTextDelta, Delta: "a"})
			}()
			<-started
			secondErr := emit(contract.Event{Type: contract.EventTextDelta, Delta: "b"})
			close(block)
			wg.Wait()
			if secondErr != nil {
				return contract.Outcome{}, secondErr
			}
			return contract.Outcome{Status: contract.StatusCompleted}, firstErr
		}}
		once := sync.Once{}
		_, err := Run(context.Background(), &contract.Request{}, agent, contract.DefaultLimits(), func(contract.Event) error {
			once.Do(func() { started <- struct{}{} })
			<-block
			return nil
		})
		require.ErrorIs(t, err, contract.ErrConcurrentEmit)
	})
}

func TestFunctionCallItem(t *testing.T) {
	item := contract.FunctionCallItem("call_1", "search", `{"q":"x"}`)
	assert.Equal(t, "call_1", item.CallID)
}
