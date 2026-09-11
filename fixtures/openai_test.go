// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package fixtures

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAI(t *testing.T) {
	server := startServer(echoAgent("hello"), chat.Info{})
	defer server.Close()
	client := openaiClient(server)
	ctx := context.Background()

	t.Run("chatText", func(t *testing.T) {
		completion, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:    "agent/basic",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
		})
		require.NoError(t, err)
		require.NotNil(t, completion)
		require.Len(t, completion.Choices, 1)
		assert.Equal(t, "hello", completion.Choices[0].Message.Content)
	})

	t.Run("responsesText", func(t *testing.T) {
		response, err := client.Responses.New(ctx, responses.ResponseNewParams{
			Model: "agent/basic",
			Input: responses.ResponseNewParamsInputUnion{OfString: param.NewOpt("hi")},
		})
		require.NoError(t, err)
		require.NotNil(t, response)
		assert.Equal(t, "response", string(response.Object))
		assert.Equal(t, "hello", response.OutputText())
	})
}

func TestOpenAIStream(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		if err := emit(chat.TextDelta("hel")); err != nil {
			return chat.Outcome{}, err
		}
		return chat.Outcome{}, emit(chat.TextDelta("lo"))
	})
	server := startServer(agent, chat.Info{})
	defer server.Close()
	client := openaiClient(server)
	ctx := context.Background()

	t.Run("chatDeltas", func(t *testing.T) {
		stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
			Model:    "agent/basic",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
		})
		var text strings.Builder
		for stream.Next() {
			for _, choice := range stream.Current().Choices {
				text.WriteString(choice.Delta.Content)
			}
		}
		require.NoError(t, stream.Err())
		assert.Equal(t, "hello", text.String())
	})

	t.Run("responsesDeltas", func(t *testing.T) {
		stream := client.Responses.NewStreaming(ctx, responses.ResponseNewParams{
			Model: "agent/basic",
			Input: responses.ResponseNewParamsInputUnion{OfString: param.NewOpt("hi")},
		})
		var text strings.Builder
		for stream.Next() {
			event := stream.Current()
			if event.Type == "response.output_text.delta" {
				text.WriteString(event.Delta)
			}
		}
		require.NoError(t, stream.Err())
		assert.Equal(t, "hello", text.String())
	})
}

func TestOpenAITools(t *testing.T) {
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

	t.Run("chatCompletion", func(t *testing.T) {
		server := startServer(toolAgent, toolInfo())
		defer server.Close()
		client := openaiClient(server)

		completion, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:    "agent/basic",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("find")},
			Tools:    []openai.ChatCompletionToolUnionParam{lookupToolOpenAI()},
		})
		require.NoError(t, err)
		require.Len(t, completion.Choices, 1)
		assert.Equal(t, "tool_calls", completion.Choices[0].FinishReason)

		calls := completion.Choices[0].Message.ToolCalls
		require.Len(t, calls, 1)
		call := calls[0].AsFunction()
		assert.Equal(t, "call_1", call.ID)
		assert.Equal(t, "lookup", call.Function.Name)
		assert.Equal(t, `{"q":"x"}`, call.Function.Arguments)
	})

	t.Run("responsesAPI", func(t *testing.T) {
		server := startServer(toolAgent, toolInfo())
		defer server.Close()
		client := openaiClient(server)

		response, err := client.Responses.New(ctx, responses.ResponseNewParams{
			Model: "agent/basic",
			Input: responses.ResponseNewParamsInputUnion{OfString: param.NewOpt("find")},
			Tools: []responses.ToolUnionParam{{
				OfFunction: &responses.FunctionToolParam{
					Name:       "lookup",
					Parameters: map[string]any{"type": "object"},
				},
			}},
		})
		require.NoError(t, err)
		require.NotEmpty(t, response.Output)

		var call responses.ResponseFunctionToolCall
		for _, item := range response.Output {
			if fn := item.AsAny(); fn != nil {
				if typed, ok := fn.(responses.ResponseFunctionToolCall); ok {
					call = typed
					break
				}
			}
		}
		assert.Equal(t, "call_1", call.CallID)
		assert.Equal(t, "lookup", call.Name)
		assert.Equal(t, `{"q":"x"}`, call.Arguments)
	})

	t.Run("chatStream", func(t *testing.T) {
		server := startServer(streamAgent, toolInfo())
		defer server.Close()
		client := openaiClient(server)

		stream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
			Model:    "agent/basic",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("find")},
			Tools:    []openai.ChatCompletionToolUnionParam{lookupToolOpenAI()},
		})
		var args strings.Builder
		var finishReason string
		for stream.Next() {
			for _, choice := range stream.Current().Choices {
				if choice.FinishReason != "" {
					finishReason = choice.FinishReason
				}
				for _, call := range choice.Delta.ToolCalls {
					args.WriteString(call.Function.Arguments)
				}
			}
		}
		require.NoError(t, stream.Err())
		assert.Equal(t, `{"q":"x"}`, args.String())
		assert.Equal(t, "tool_calls", finishReason)
	})
}

func TestOpenAIValidation(t *testing.T) {
	ctx := context.Background()

	t.Run("emptyMessages", func(t *testing.T) {
		server := startServer(echoAgent("unused"), chat.Info{})
		defer server.Close()
		client := openaiClient(server)

		_, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:    "agent/basic",
			Messages: []openai.ChatCompletionMessageParamUnion{},
		})
		require.Error(t, err)
	})

	t.Run("missingModel", func(t *testing.T) {
		server := startServer(echoAgent("unused"), chat.Info{})
		defer server.Close()
		client := openaiClient(server)

		_, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
		})
		require.Error(t, err)
	})

	t.Run("toolsUnsupported", func(t *testing.T) {
		server := startServer(echoAgent("unused"), chat.Info{})
		defer server.Close()
		client := openaiClient(server)

		_, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:    "agent/basic",
			Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
			Tools:    []openai.ChatCompletionToolUnionParam{lookupToolOpenAI()},
		})
		require.Error(t, err)
		var apiErr *openai.Error
		require.True(t, errors.As(err, &apiErr))
		assert.Equal(t, "unsupported", apiErr.Code)
	})

	t.Run("systemAndUser", func(t *testing.T) {
		var captured chat.Request
		server := startServer(captureAgent(func(req *chat.Request) { captured = *req }), chat.Info{})
		defer server.Close()
		client := openaiClient(server)

		_, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model: "agent/basic",
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("be helpful"),
				openai.UserMessage("hi"),
			},
		})
		require.NoError(t, err)
		require.Len(t, captured.Input, 2)
		assert.Equal(t, chat.RoleSystem, captured.Input[0].Role)
		require.Len(t, captured.Input[0].Content, 1)
		assert.Equal(t, "be helpful", captured.Input[0].Content[0].Text)
		assert.Equal(t, chat.RoleUser, captured.Input[1].Role)
		require.Len(t, captured.Input[1].Content, 1)
		assert.Equal(t, "hi", captured.Input[1].Content[0].Text)
	})

	t.Run("generationControls", func(t *testing.T) {
		var captured chat.Request
		info := chat.Info{
			GenerationControls: chat.ControlMaxOutputTokens | chat.ControlTemperature,
		}
		server := startServer(captureAgent(func(req *chat.Request) { captured = *req }), info)
		defer server.Close()
		client := openaiClient(server)

		_, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:       "agent/basic",
			Messages:    []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
			Temperature: openai.Float(0.7),
			MaxTokens:   openai.Int(16),
		})
		require.NoError(t, err)
		require.NotNil(t, captured.Controls.Temperature)
		assert.InDelta(t, 0.7, *captured.Controls.Temperature, 0.001)
		require.NotNil(t, captured.Controls.MaxOutputTokens)
		assert.Equal(t, 16, *captured.Controls.MaxOutputTokens)
	})
}

func TestOpenAIModels(t *testing.T) {
	server := startServerEntries(echoAgent("unused"), chat.Info{}, map[string]chat.Info{
		"agent/basic": {Created: 42, OwnedBy: "test"},
		"agent/other": {},
	})
	defer server.Close()
	client := openaiClient(server)

	page, err := client.Models.List(context.Background())
	require.NoError(t, err)
	require.NotNil(t, page)
	assert.Equal(t, "list", page.Object)
	require.Len(t, page.Data, 2)
	assert.Equal(t, "agent/basic", page.Data[0].ID)
	assert.Equal(t, int64(42), page.Data[0].Created)
	assert.Equal(t, "test", page.Data[0].OwnedBy)
	assert.Equal(t, "agent/other", page.Data[1].ID)
}

func TestOpenAIRoute(t *testing.T) {
	server := startServer(echoAgent("unused"), chat.Info{})
	defer server.Close()

	t.Run("unknownRoute", func(t *testing.T) {
		resp, err := http.Get(server.URL + "/v1/unknown")
		require.NoError(t, err)
		defer resp.Body.Close()
		assert.Equal(t, http.StatusNotFound, resp.StatusCode)
	})

	t.Run("methodNotAllowed", func(t *testing.T) {
		req, err := http.NewRequest(http.MethodPut, server.URL+"/v1/chat/completions", nil)
		require.NoError(t, err)
		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		defer resp.Body.Close()
		assert.Equal(t, http.StatusMethodNotAllowed, resp.StatusCode)
	})
}
