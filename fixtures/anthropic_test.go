// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package fixtures

import (
	"context"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAnthropic(t *testing.T) {
	server := startServer(echoAgent("hello"), chat.Info{})
	defer server.Close()
	client := anthropicClient(server)
	ctx := context.Background()

	t.Run("text", func(t *testing.T) {
		message, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     "agent/basic",
			MaxTokens: 32,
			Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock("hi"))},
		})
		require.NoError(t, err)
		require.NotNil(t, message)
		assert.Equal(t, "message", string(message.Type))
		require.Len(t, message.Content, 1)
		assert.Equal(t, "hello", message.Content[0].AsText().Text)
	})
}

func TestAnthropicStream(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		if err := emit(chat.TextDelta("hel")); err != nil {
			return chat.Outcome{}, err
		}
		return chat.Outcome{}, emit(chat.TextDelta("lo"))
	})
	server := startServer(agent, chat.Info{})
	defer server.Close()
	client := anthropicClient(server)

	stream := client.Messages.NewStreaming(context.Background(), anthropic.MessageNewParams{
		Model:     "agent/basic",
		MaxTokens: 32,
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock("hi"))},
	})
	var text strings.Builder
	for stream.Next() {
		event := stream.Current()
		if event.Type == "content_block_delta" {
			text.WriteString(event.Delta.Text)
		}
	}
	require.NoError(t, stream.Err())
	assert.Equal(t, "hello", text.String())
}

func TestAnthropicToolUse(t *testing.T) {
	ctx := context.Background()
	toolAgent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{StopReason: chat.StopToolCall}, emit(chat.Tool("call_1", "lookup", `{"q":"x"}`))
	})
	streamAgent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		if err := emit(chat.ToolStart("call_1", "lookup")); err != nil {
			return chat.Outcome{}, err
		}
		if err := emit(chat.ToolDelta("call_1", `{"q":`)); err != nil {
			return chat.Outcome{}, err
		}
		if err := emit(chat.ToolDelta("call_1", `"x"}`)); err != nil {
			return chat.Outcome{}, err
		}
		return chat.Outcome{StopReason: chat.StopToolCall}, emit(chat.ToolDone("call_1"))
	})

	t.Run("toolUse", func(t *testing.T) {
		server := startServer(toolAgent, toolInfo())
		defer server.Close()
		client := anthropicClient(server)

		message, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     "agent/basic",
			MaxTokens: 32,
			Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock("find"))},
			Tools:     []anthropic.ToolUnionParam{lookupToolAnthropic()},
		})
		require.NoError(t, err)
		require.Len(t, message.Content, 1)
		tool := message.Content[0].AsToolUse()
		assert.Equal(t, "tool_use", string(tool.Type))
		assert.Equal(t, "call_1", tool.ID)
		assert.Equal(t, "lookup", tool.Name)
		assert.JSONEq(t, `{"q":"x"}`, string(tool.Input))
		assert.Equal(t, anthropic.StopReasonToolUse, message.StopReason)
	})

	t.Run("streamToolUse", func(t *testing.T) {
		server := startServer(streamAgent, toolInfo())
		defer server.Close()
		client := anthropicClient(server)

		stream := client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
			Model:     "agent/basic",
			MaxTokens: 32,
			Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock("find"))},
			Tools:     []anthropic.ToolUnionParam{lookupToolAnthropic()},
		})
		var args strings.Builder
		var stopReason anthropic.StopReason
		for stream.Next() {
			event := stream.Current()
			switch event.Type {
			case "content_block_delta":
				args.WriteString(event.Delta.PartialJSON)
			case "message_delta":
				stopReason = event.Delta.StopReason
			}
		}
		require.NoError(t, stream.Err())
		assert.JSONEq(t, `{"q":"x"}`, args.String())
		assert.Equal(t, anthropic.StopReasonToolUse, stopReason)
	})
}

func TestAnthropicValidation(t *testing.T) {
	ctx := context.Background()
	server := startServer(echoAgent("unused"), chat.Info{})
	defer server.Close()
	client := anthropicClient(server)

	t.Run("invalidMaxTokens", func(t *testing.T) {
		_, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     "agent/basic",
			MaxTokens: 0,
			Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock("hi"))},
		})
		require.Error(t, err)
	})

	t.Run("emptyMessages", func(t *testing.T) {
		_, err := client.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     "agent/basic",
			MaxTokens: 32,
			Messages:  []anthropic.MessageParam{},
		})
		require.Error(t, err)
	})

	t.Run("multiTurn", func(t *testing.T) {
		var captured chat.Request
		capture := startServer(captureAgent(func(req *chat.Request) { captured = *req }), toolInfo())
		defer capture.Close()
		captureClient := anthropicClient(capture)

		_, err := captureClient.Messages.New(ctx, anthropic.MessageNewParams{
			Model:     "agent/basic",
			MaxTokens: 32,
			Messages: []anthropic.MessageParam{
				{
					Role: anthropic.MessageParamRoleAssistant,
					Content: []anthropic.ContentBlockParamUnion{{
						OfToolUse: &anthropic.ToolUseBlockParam{
							ID:    "tu_1",
							Name:  "lookup",
							Input: map[string]any{"q": "x"},
						},
					}},
				},
				{
					Role: anthropic.MessageParamRoleUser,
					Content: []anthropic.ContentBlockParamUnion{{
						OfToolResult: &anthropic.ToolResultBlockParam{
							ToolUseID: "tu_1",
							Content: []anthropic.ToolResultBlockParamContentUnion{
								{OfText: &anthropic.TextBlockParam{Text: "found"}},
							},
						},
					}},
				},
			},
		})
		require.NoError(t, err)
		require.GreaterOrEqual(t, len(captured.Input), 2)
		assert.Equal(t, chat.ItemFunctionCall, captured.Input[0].Type)
		assert.Equal(t, "tu_1", captured.Input[0].CallID)
		assert.Equal(t, chat.ItemFunctionCallOutput, captured.Input[1].Type)
		assert.Equal(t, "tu_1", captured.Input[1].CallID)
	})
}
