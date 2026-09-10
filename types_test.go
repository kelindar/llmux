package llmux

import (
	"context"
	"encoding/json/jsontext"
	"testing"

	internalexecution "github.com/kelindar/llmux/internal/execution"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMediaConstructors(t *testing.T) {
	remote := RemoteMedia("image/png", "https://example.com/a.png")
	assert.Equal(t, "image/png", remote.MIMEType)
	assert.Equal(t, "https://example.com/a.png", remote.URL)

	asset := AssetMedia("application/pdf", "file-abc")
	assert.Equal(t, "application/pdf", asset.MIMEType)
	assert.Equal(t, "file-abc", asset.Ref)
}

func TestPartConstructors(t *testing.T) {
	media := InlineMedia("application/pdf", []byte{1})
	media.Filename = "doc.pdf"
	file := FilePart(media)
	assert.Equal(t, PartFile, file.Type)
	require.NotNil(t, file.Media)
	assert.Equal(t, "doc.pdf", file.Media.Filename)

	jsonPart := JSONPart(jsontext.Value(`{"ok":true}`))
	assert.Equal(t, PartJSON, jsonPart.Type)
	assert.True(t, jsonPart.Data.IsValid())
}

func TestItemConstructors(t *testing.T) {
	message := MessageItem(RoleUser, TextPart("hi"))
	assert.Equal(t, ItemMessage, message.Type)
	assert.Equal(t, RoleUser, message.Role)
	require.Len(t, message.Content, 1)

	call := FunctionCallItem("call_1", "lookup", `{"q":"go"}`)
	assert.Equal(t, ItemFunctionCall, call.Type)
	assert.Equal(t, "call_1", call.CallID)
	assert.Equal(t, "lookup", call.Name)

	output := FunctionCallOutputItem("call_1", TextPart("done"))
	assert.Equal(t, ItemFunctionCallOutput, output.Type)
	assert.Equal(t, "call_1", output.CallID)
	require.Len(t, output.Output, 1)
}

func TestEventConstructors(t *testing.T) {
	text := CompleteText("hello")
	assert.Equal(t, EventMessage, text.Type)
	assert.Equal(t, RoleAssistant, text.Item.Role)

	tool := ToolCall("call_1", "lookup", `{}`)
	assert.Equal(t, EventToolCall, tool.Type)
	assert.Equal(t, "call_1", tool.CallID)

	start := ToolCallStart("call_1", "lookup")
	assert.Equal(t, EventToolCallStart, start.Type)

	delta := ToolCallDelta("call_1", `{"q"`)
	assert.Equal(t, EventToolCallDelta, delta.Type)

	done := ToolCallDone("call_1")
	assert.Equal(t, EventToolCallDone, done.Type)

	media := MediaOutput(ImagePart(InlineMedia("image/png", []byte{1})))
	assert.Equal(t, EventMedia, media.Type)
	assert.Equal(t, PartImage, media.Part.Type)

	reasoning := ReasoningSummary("brief")
	assert.Equal(t, EventReasoning, reasoning.Type)
	assert.Equal(t, ItemReasoning, reasoning.Item.Type)

	item := OutputItem(MessageItem(RoleAssistant, TextPart("ok")))
	assert.Equal(t, EventItem, item.Type)

	var got Event
	emit := func(event Event) error {
		got = event
		return nil
	}
	require.NoError(t, EmitToolCallDone(emit, "call_1"))
	assert.Equal(t, EventToolCallDone, got.Type)
	assert.Equal(t, "call_1", got.CallID)
}

func TestAPIErrorConstructors(t *testing.T) {
	err := NewAPIError(418, "teapot", "brew_failed", "water", "too hot")
	require.NotNil(t, err)
	assert.Equal(t, 418, err.Status)
	assert.Equal(t, "teapot", err.Type)
	assert.Equal(t, "brew_failed", err.Code)
	assert.Equal(t, "water", err.Param)
	assert.Equal(t, "too hot", err.Message)

	invalid := Invalid("model", "model is required")
	assert.Equal(t, 400, invalid.Status)
	assert.Equal(t, "invalid_request", invalid.Code)
}

func TestValidRole(t *testing.T) {
	cases := []struct {
		role Role
		ok   bool
	}{
		{role: RoleUser, ok: true},
		{role: RoleAssistant, ok: true},
		{role: RoleSystem, ok: true},
		{role: RoleDeveloper, ok: true},
		{role: RoleTool, ok: true},
		{role: Role("unknown"), ok: false},
	}
	for _, test := range cases {
		t.Run(string(test.role), func(t *testing.T) {
			assert.Equal(t, test.ok, validRole(test.role))
		})
	}
}

func TestNewHandlerAlias(t *testing.T) {
	agent := AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, nil
	})
	handler := NewHandler(ResolverFunc(func(context.Context, string) (Agent, Capabilities, error) {
		return agent, Capabilities{}, nil
	}))
	require.NotNil(t, handler)
}

func TestExecutionEvents(t *testing.T) {
	var events []Event
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		require.NoError(t, emit(ReasoningSummary("thinking")))
		return Outcome{StopReason: StopToolCall}, EmitToolCall(emit, "call_1", "lookup", `{}`)
	})
	_, err := internalexecution.Run(context.Background(), &Request{Target: "agent/basic"}, agent, DefaultLimits(), func(event Event) error {
		events = append(events, event)
		return nil
	})
	require.NoError(t, err)
	require.Len(t, events, 2)
	assert.Equal(t, EventReasoning, events[0].Type)
	assert.Equal(t, EventToolCall, events[1].Type)
}
