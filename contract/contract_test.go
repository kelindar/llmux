package contract

import (
	"context"
	"encoding/json/jsontext"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMediaValidate(t *testing.T) {
	cases := map[string]struct {
		media   Media
		max     int64
		wantErr string
	}{
		"inlineOk": {
			media: InlineMedia("image/png", []byte{1, 2}),
		},
		"remoteOk": {
			media: RemoteMedia("image/png", "https://example.com/a.png"),
		},
		"assetOk": {
			media: AssetMedia("image/png", "asset-1"),
		},
		"noSource": {
			media:   Media{MIMEType: "image/png"},
			wantErr: "exactly one",
		},
		"multipleSources": {
			media:   Media{MIMEType: "image/png", Data: []byte{1}, URL: "https://example.com/a.png"},
			wantErr: "exactly one",
		},
		"exceedsLimit": {
			media:   InlineMedia("image/png", []byte{1, 2, 3}),
			max:     2,
			wantErr: "byte limit",
		},
		"missingMime": {
			media:   Media{Data: []byte{1}},
			wantErr: "MIME type is required",
		},
		"badURL": {
			media:   RemoteMedia("image/png", "not-a-url"),
			wantErr: "absolute HTTP",
		},
		"ftpURL": {
			media:   RemoteMedia("image/png", "ftp://example.com/a.png"),
			wantErr: "absolute HTTP",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := ImagePart(tc.media).Validate(tc.max)
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestMediaConstructors(t *testing.T) {
	data := []byte{9}
	inline := InlineMedia("text/plain", data)
	assert.Equal(t, "text/plain", inline.MIMEType)
	assert.Equal(t, []byte{9}, inline.Data)
	data[0] = 0
	assert.Equal(t, []byte{9}, inline.Data)

	remote := RemoteMedia("image/jpeg", "https://example.com/x.jpg")
	assert.Equal(t, "https://example.com/x.jpg", remote.URL)

	asset := AssetMedia("application/pdf", "doc-1")
	assert.Equal(t, "doc-1", asset.Ref)

	cloned := inline.Clone()
	cloned.Data[0] = 7
	assert.Equal(t, byte(9), inline.Data[0])
}

func TestPartConstructors(t *testing.T) {
	text := TextPart("hello")
	assert.Equal(t, PartText, text.Type)
	assert.Equal(t, "hello", text.Text)

	img := ImagePart(InlineMedia("image/png", []byte{1}))
	assert.Equal(t, PartImage, img.Type)
	require.NotNil(t, img.Media)

	audio := AudioPart(InlineMedia("audio/wav", []byte{2}))
	audio.Media.Format = "wav"
	assert.Equal(t, PartAudio, audio.Type)

	file := FilePart(AssetMedia("application/pdf", "f1"))
	assert.Equal(t, PartFile, file.Type)

	jsonPart := JSONPart(jsontext.Value(`{"a":1}`))
	assert.Equal(t, PartJSON, jsonPart.Type)
	assert.True(t, jsonPart.Data.IsValid())

	summary := SummaryPart("thought")
	assert.Equal(t, PartReasoningSummary, summary.Type)

	cloned := img.Clone()
	cloned.Media.Data[0] = 9
	assert.Equal(t, byte(1), img.Media.Data[0])
}

func TestPartValidate(t *testing.T) {
	cases := map[string]struct {
		part    Part
		wantErr string
	}{
		"textOk":    {part: TextPart("hi")},
		"summaryOk": {part: SummaryPart("brief")},
		"textWithMedia": {
			part:    Part{Type: PartText, Media: &Media{MIMEType: "image/png", Data: []byte{1}}},
			wantErr: "cannot contain media",
		},
		"imageMissingMedia": {
			part:    Part{Type: PartImage},
			wantErr: "missing media",
		},
		"audioMissingFormat": {
			part:    AudioPart(InlineMedia("audio/wav", []byte{1})),
			wantErr: "format is required",
		},
		"audioOk": {
			part: func() Part {
				m := InlineMedia("audio/wav", []byte{1})
				m.Format = "wav"
				return AudioPart(m)
			}(),
		},
		"jsonInvalid": {
			part:    Part{Type: PartJSON, Data: jsontext.Value(`{`)},
			wantErr: "not valid JSON",
		},
		"jsonWithMedia": {
			part:    Part{Type: PartJSON, Data: jsontext.Value(`{}`), Media: &Media{}},
			wantErr: "cannot contain media",
		},
		"unknownType": {
			part:    Part{Type: "other"},
			wantErr: "unknown content part type",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := tc.part.Validate(0)
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestItemValidate(t *testing.T) {
	validAudio := func() Part {
		m := InlineMedia("audio/wav", []byte{1})
		m.Format = "wav"
		return AudioPart(m)
	}

	cases := map[string]struct {
		item    Item
		output  bool
		wantErr string
	}{
		"messageOk": {
			item: MessageItem(RoleUser, TextPart("hi")),
		},
		"assistantOutputOk": {
			item:   MessageItem(RoleAssistant, TextPart("reply")),
			output: true,
		},
		"missingType": {
			item:    Item{Role: RoleUser},
			wantErr: "type is required",
		},
		"invalidStatus": {
			item:    Item{Type: ItemMessage, Role: RoleUser, Status: "bogus"},
			wantErr: "invalid item status",
		},
		"messageWrongRole": {
			item:    MessageItem(RoleUser, TextPart("hi")),
			output:  true,
			wantErr: "agent message role",
		},
		"messageExtraFields": {
			item:    Item{Type: ItemMessage, Role: RoleUser, CallID: "x"},
			wantErr: "fields from another item type",
		},
		"functionCallOk": {
			item: FunctionCallItem("c1", "search", `{}`),
		},
		"functionCallBadArgs": {
			item:    FunctionCallItem("c1", "search", `{`),
			wantErr: "valid JSON",
		},
		"functionCallMissingName": {
			item:    Item{Type: ItemFunctionCall, CallID: "c1", Arguments: `{}`},
			wantErr: "call_id and name",
		},
		"functionCallOutputOk": {
			item: FunctionCallOutputItem("c1", TextPart("result")),
		},
		"functionCallOutputMissingID": {
			item:    Item{Type: ItemFunctionCallOutput, Output: []Part{TextPart("x")}},
			wantErr: "call_id",
		},
		"reasoningOk": {
			item: Item{Type: ItemReasoning, Summary: []Part{SummaryPart("think")}},
		},
		"reasoningBadPart": {
			item:    Item{Type: ItemReasoning, Summary: []Part{TextPart("nope")}},
			wantErr: "reasoning_summary",
		},
		"mediaImageOk": {
			item: Item{
				Type:    ItemMedia,
				Content: []Part{ImagePart(InlineMedia("image/png", []byte{1}))},
			},
		},
		"mediaAudioOk": {
			item: Item{Type: ItemMedia, Content: []Part{validAudio()}},
		},
		"mediaWrongCount": {
			item:    Item{Type: ItemMedia, Content: []Part{}},
			wantErr: "exactly one content part",
		},
		"extensionOk": {
			item: Item{Type: ItemExtension, Data: jsontext.Value(`{"k":"v"}`)},
		},
		"extensionInvalidJSON": {
			item:    Item{Type: ItemExtension, Data: jsontext.Value(`{`)},
			wantErr: "valid JSON data",
		},
		"unknownType": {
			item:    Item{Type: "other"},
			wantErr: "unknown item type",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := tc.item.Validate(0, tc.output)
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestItemClone(t *testing.T) {
	orig := MessageItem(RoleUser, ImagePart(InlineMedia("image/png", []byte{1})))
	cloned := orig.Clone()
	cloned.Content[0].Media.Data[0] = 9
	assert.Equal(t, byte(1), orig.Content[0].Media.Data[0])
}

func TestMessageConstructors(t *testing.T) {
	msg := MessageItem(RoleSystem, TextPart("sys"))
	assert.Equal(t, ItemMessage, msg.Type)
	assert.Equal(t, RoleSystem, msg.Role)

	call := FunctionCallItem("id1", "fn", `{"q":"x"}`)
	assert.Equal(t, ItemFunctionCall, call.Type)
	assert.NotEmpty(t, call.ID)
	assert.Equal(t, StatusCompleted, call.Status)

	out := FunctionCallOutputItem("id1", TextPart("done"))
	assert.Equal(t, ItemFunctionCallOutput, out.Type)
	assert.Equal(t, "id1", out.CallID)
}

func TestOutcomeValidate(t *testing.T) {
	cases := map[string]struct {
		outcome Outcome
		wantErr string
	}{
		"emptyOk": {},
		"completedOk": {
			outcome: Outcome{Status: StatusCompleted, StopReason: StopStop},
		},
		"usageOk": {
			outcome: Outcome{
				Status: StatusCompleted,
				Usage:  &Usage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3},
			},
		},
		"invalidStatus": {
			outcome: Outcome{Status: "bad"},
			wantErr: "invalid outcome status",
		},
		"inProgress": {
			outcome: Outcome{Status: StatusInProgress},
			wantErr: "cannot be in_progress",
		},
		"negativeUsage": {
			outcome: Outcome{Status: StatusCompleted, Usage: &Usage{InputTokens: -1}},
			wantErr: "negative values",
		},
		"invalidStopReason": {
			outcome: Outcome{Status: StatusCompleted, StopReason: "nope"},
			wantErr: "invalid outcome stop reason",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := tc.outcome.Validate()
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestAPIError(t *testing.T) {
	var nilErr *APIError
	assert.Equal(t, "", nilErr.Error())

	withMessage := NewAPIError(400, "invalid_request_error", "invalid_request", "field", "bad field")
	assert.Equal(t, "bad field", withMessage.Error())
	assert.Equal(t, 400, withMessage.Status)

	wrapped := &APIError{Err: errors.New("underlying")}
	assert.Equal(t, "underlying", wrapped.Error())
	assert.Equal(t, errors.New("underlying"), wrapped.Unwrap())

	defaultMsg := &APIError{}
	assert.Equal(t, "llmux API error", defaultMsg.Error())

	invalid := Invalid("p", "msg")
	assert.Equal(t, 400, invalid.Status)
	assert.Equal(t, "invalid_request", invalid.Code)

	unsupported := Unsupported("p", "msg")
	assert.Equal(t, "unsupported", unsupported.Code)
}

func TestLimitsNormalize(t *testing.T) {
	def := DefaultLimits()
	assert.Equal(t, int64(8<<20), def.MaxRequestBytes)
	assert.Equal(t, int64(16<<20), def.MaxMediaBytes)
	assert.Equal(t, 32, def.MaxAssets)

	norm := Limits{}.Normalize()
	assert.Equal(t, def, norm)

	partial := Limits{MaxAssets: 10}.Normalize()
	assert.Equal(t, 10, partial.MaxAssets)
	assert.Equal(t, def.MaxRequestBytes, partial.MaxRequestBytes)
}

func TestCapabilitiesNormalize(t *testing.T) {
	zero := Capabilities{}.Normalize()
	assert.True(t, zero.InputModalities.Has(ModalityText))
	assert.True(t, zero.OutputModalities.Has(ModalityText))
	assert.True(t, zero.GenerationControls.Has(ControlMaxOutputTokens))
	assert.True(t, zero.GenerationControls.Has(ControlTemperature))

	withImage := Capabilities{ImageGeneration: true}.Normalize()
	assert.True(t, withImage.OutputModalities.Has(ModalityImage))

	withTools := Capabilities{Tools: true}.Normalize()
	assert.True(t, withTools.GenerationControls.Has(ControlParallelToolCalls))

	withReasoning := Capabilities{ReasoningSummary: true}.Normalize()
	assert.True(t, withReasoning.GenerationControls.Has(ControlReasoning))

	withAudio := Capabilities{OutputModalities: ModalityAudio}.Normalize()
	assert.True(t, withAudio.GenerationControls.Has(ControlAudio))
}

func TestModalityHas(t *testing.T) {
	m := ModalityText | ModalityImage
	assert.True(t, m.Has(ModalityText))
	assert.True(t, m.Has(ModalityImage))
	assert.False(t, m.Has(ModalityAudio))
	assert.True(t, Modality(0).Has(Modality(0)))
}

func TestToolValidate(t *testing.T) {
	cases := map[string]struct {
		tool    Tool
		wantErr string
	}{
		"minimalOk": {tool: Tool{Name: "search"}},
		"paramsOk":  {tool: Tool{Name: "search", Parameters: jsontext.Value(`{"type":"object"}`)}},
		"missingName": {
			tool:    Tool{},
			wantErr: "name is required",
		},
		"badParams": {
			tool:    Tool{Name: "search", Parameters: jsontext.Value(`[]`)},
			wantErr: "must be a JSON object",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := tc.tool.Validate()
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestToolChoiceValidate(t *testing.T) {
	modes := []string{"auto", "none", "required", "any"}
	for _, mode := range modes {
		t.Run(mode, func(t *testing.T) {
			require.NoError(t, ToolChoice{Mode: mode}.Validate())
		})
	}

	cases := map[string]struct {
		choice  ToolChoice
		wantErr string
	}{
		"autoWithName": {
			choice:  ToolChoice{Mode: "auto", Name: "x"},
			wantErr: "cannot have a name",
		},
		"functionOk": {
			choice: ToolChoice{Mode: "function", Name: "search"},
		},
		"functionMissingName": {
			choice:  ToolChoice{Mode: "function", Name: "  "},
			wantErr: "requires a name",
		},
		"unsupportedMode": {
			choice:  ToolChoice{Mode: "weird"},
			wantErr: "unsupported tool choice mode",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := tc.choice.Validate()
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestStructuredOutputValidate(t *testing.T) {
	cases := map[string]struct {
		out     StructuredOutput
		wantErr string
	}{
		"ok": {
			out: StructuredOutput{Name: "out", Schema: jsontext.Value(`{"type":"object"}`)},
		},
		"missingName": {
			out:     StructuredOutput{Schema: jsontext.Value(`{}`)},
			wantErr: "requires a name",
		},
		"invalidSchema": {
			out:     StructuredOutput{Name: "out", Schema: jsontext.Value(`{`)},
			wantErr: "requires a name",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := tc.out.Validate()
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestReasoningValidate(t *testing.T) {
	assert.NoError(t, Reasoning{Effort: "high"}.Validate())
	assert.NoError(t, Reasoning{Summary: true}.Validate())
	err := Reasoning{}.Validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "effort or summary")
}

func TestAudioControlsValidate(t *testing.T) {
	formats := []string{"wav", "aac", "mp3", "flac", "opus", "pcm16", "WAV"}
	for _, format := range formats {
		t.Run(format, func(t *testing.T) {
			require.NoError(t, AudioControls{Voice: "alloy", Format: format}.Validate())
		})
	}

	cases := map[string]struct {
		ctrl    AudioControls
		wantErr string
	}{
		"missingVoice": {
			ctrl:    AudioControls{Format: "mp3"},
			wantErr: "voice is required",
		},
		"badFormat": {
			ctrl:    AudioControls{Voice: "alloy", Format: "wma"},
			wantErr: "unsupported Chat Completions audio format",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := tc.ctrl.Validate()
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestValidRoleStatus(t *testing.T) {
	assert.True(t, ValidRole(RoleUser))
	assert.False(t, ValidRole(Role("guest")))
	assert.True(t, ValidStatus(StatusCompleted))
	assert.False(t, ValidStatus(Status("pending")))
}

func TestAgentFunc(t *testing.T) {
	var nilAgent AgentFunc
	_, err := nilAgent.Run(context.Background(), &Request{}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nil agent function")

	var captured Event
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		require.NoError(t, emit(CompleteText("hi")))
		return Outcome{Status: StatusCompleted}, nil
	})
	out, err := agent.Run(context.Background(), &Request{}, func(ev Event) error {
		captured = ev
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, StatusCompleted, out.Status)
	assert.Equal(t, EventMessage, captured.Type)
}

func TestResolverFunc(t *testing.T) {
	var nilResolver ResolverFunc
	_, _, err := nilResolver.Resolve(context.Background(), "target")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nil resolver function")

	agent := AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, nil
	})
	caps := Capabilities{Tools: true}
	resolver := ResolverFunc(func(_ context.Context, target string) (Agent, Capabilities, error) {
		assert.Equal(t, "my-agent", target)
		return agent, caps, nil
	})
	gotAgent, gotCaps, err := resolver.Resolve(context.Background(), "my-agent")
	require.NoError(t, err)
	require.NotNil(t, gotAgent)
	_, ok := gotAgent.(AgentFunc)
	assert.True(t, ok)
	assert.Equal(t, caps, gotCaps)
}

func TestAssetResolverFunc(t *testing.T) {
	var nilResolver AssetResolverFunc
	_, err := nilResolver.Resolve(context.Background(), Media{}, 0)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nil asset resolver function")

	resolver := AssetResolverFunc(func(_ context.Context, media Media, max int64) (Media, error) {
		assert.Equal(t, "ref-1", media.Ref)
		assert.Equal(t, int64(100), max)
		return InlineMedia("text/plain", []byte("data")), nil
	})
	got, err := resolver.Resolve(context.Background(), AssetMedia("text/plain", "ref-1"), 100)
	require.NoError(t, err)
	assert.Equal(t, []byte("data"), got.Data)
}

func TestEmitHelpers(t *testing.T) {
	var events []Event
	emit := func(ev Event) error {
		events = append(events, ev)
		return nil
	}

	require.NoError(t, EmitText(emit, "full"))
	require.NoError(t, EmitTextDelta(emit, "part"))
	require.NoError(t, EmitToolCall(emit, "c1", "fn", `{}`))
	require.NoError(t, EmitToolCallStart(emit, "c1", "fn"))
	require.NoError(t, EmitToolCallDelta(emit, "c1", `{"a"`))
	require.NoError(t, EmitToolCallDone(emit, "c1"))
	require.NoError(t, EmitMedia(emit, ImagePart(InlineMedia("image/png", []byte{1}))))
	require.NoError(t, EmitItem(emit, MessageItem(RoleAssistant, TextPart("item"))))

	assert.Len(t, events, 8)
	assert.Equal(t, EventMessage, events[0].Type)
	assert.Equal(t, EventTextDelta, events[1].Type)
	assert.Equal(t, EventToolCall, events[2].Type)
	assert.Equal(t, EventToolCallStart, events[3].Type)
	assert.Equal(t, EventToolCallDelta, events[4].Type)
	assert.Equal(t, EventToolCallDone, events[5].Type)
	assert.Equal(t, EventMedia, events[6].Type)
	assert.Equal(t, EventItem, events[7].Type)

	reasoning := ReasoningSummary("brief")
	assert.Equal(t, EventReasoning, reasoning.Type)
	assert.Equal(t, ItemReasoning, reasoning.Item.Type)
}

func TestGenerationControlHas(t *testing.T) {
	c := ControlMaxOutputTokens | ControlStop
	assert.True(t, c.Has(ControlStop))
	assert.False(t, c.Has(ControlAudio))
}
