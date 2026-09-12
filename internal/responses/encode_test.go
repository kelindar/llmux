// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package responses

import (
	"encoding/json/jsontext"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/execution"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEncodeResponse(t *testing.T) {
	body, err := Adapter{}.Response(chat.Request{Target: "m"}, execution.Result{
		Items:   []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("hi"))},
		Outcome: chat.Outcome{Status: chat.StatusCompleted},
	}, responseMeta{Response: chat.Response{ID: "id", Created: 1, Target: "m"}})
	require.NoError(t, err)
	assert.Equal(t, "response", body.(wireResponse).Object)
}

func TestRenderResponse(t *testing.T) {
	previous := "resp_parent"
	response := chat.Response{
		ID:           "resp_1",
		Created:      100,
		CompletedAt:  200,
		Status:       chat.StatusIncomplete,
		Incomplete:   "max_output_tokens",
		Error:        &chat.Error{Code: "server_error", Message: "backend failed"},
		Instructions: "be concise",
		Metadata:     map[string]string{"trace": "t1"},
		Store:        true,
		Target:       "gpt-4.1",
		Previous:     &previous,
		Usage:        &chat.Usage{Input: 3, Output: 4, Total: 7, Cached: 1, Reasoning: 2},
		Output: []chat.Item{
			chat.MessageItem(chat.RoleAssistant, chat.TextPart("answer")),
			chat.FunctionCallItem("call_1", "lookup", `{"q":"x"}`),
			chat.FunctionCallOutputItem("call_1", chat.TextPart("found"), chat.ImagePart(chat.InlineMedia("image/png", []byte{1, 2}))),
			{Type: chat.ItemReasoning, Summary: []chat.Part{chat.SummaryPart("because")}, EncryptedContent: jsontext.Value(`"opaque"`)},
			{Type: chat.ItemMedia, Content: []chat.Part{chat.ImagePart(chat.InlineMedia("image/png", []byte{3, 4}))}},
		},
	}

	value, err := Render(response)
	require.NoError(t, err)
	body := value.(wireResponse)
	assert.Equal(t, "resp_1", body.ID)
	assert.Equal(t, "incomplete", body.Status)
	assert.Equal(t, wireIncomplete{Reason: "max_output_tokens"}, body.IncompleteDetails)
	assert.Equal(t, "be concise", body.Instructions)
	assert.Equal(t, "resp_parent", body.PreviousResponseID)
	assert.Equal(t, map[string]string{"trace": "t1"}, body.Metadata)
	require.Len(t, body.Output, 5)
	assert.Equal(t, "message", body.Output[0].(wireMessageItem).Type)
	assert.Equal(t, "function_call", body.Output[1].(wireFunctionCallItem).Type)
	assert.Equal(t, "function_call_output", body.Output[2].(wireFunctionOutputItem).Type)
	assert.Equal(t, "reasoning", body.Output[3].(wireReasoningItem).Type)
	assert.Equal(t, "image_generation_call", body.Output[4].(wireImageItem).Type)
	require.NotNil(t, body.Usage)
	assert.Equal(t, 2, body.Usage.OutputTokensDetails.ReasoningTokens)
}

func TestResponseObjectOptions(t *testing.T) {
	maxTokens, temperature, topP, parallel := 32, 0.5, 0.7, false
	value, err := (Adapter{}).Response(chat.Request{
		Target:       "gpt-4.1",
		Instructions: "respond",
		Controls: chat.Controls{
			MaxOutputTokens:  &maxTokens,
			Temperature:      &temperature,
			TopP:             &topP,
			ParallelToolCall: &parallel,
			ToolChoice:       &chat.ToolChoice{Mode: "function", Name: "lookup"},
			Tools:            []chat.FunctionTool{{Name: "lookup", Parameters: jsontext.Value(`{"type":"object"}`), Strict: new(true)}},
		},
		Output: chat.OutputSpec{Format: chat.OutputFormat{Kind: chat.FormatJSONObject}},
	}, execution.Result{}, responseMeta{})
	require.NoError(t, err)
	body := value.(wireResponse)
	assert.Equal(t, "gpt-4.1", body.Model)
	assert.Equal(t, "respond", body.Instructions)
	assert.Equal(t, 32, body.MaxOutputTokens)
	assert.Equal(t, 0.5, body.Temperature)
	assert.Equal(t, 0.7, body.TopP)
	assert.False(t, body.ParallelToolCalls)
	assert.Equal(t, "json_object", body.Text.Format.Type)
	assert.Equal(t, map[string]any{"type": "function", "name": "lookup"}, body.ToolChoice)
	require.Len(t, body.Tools, 1)
}

func TestRenderErrors(t *testing.T) {
	t.Run("unsupported output", func(t *testing.T) {
		_, err := (Adapter{}).Response(chat.Request{}, execution.Result{
			Items: []chat.Item{{Type: chat.ItemExtension}},
		}, responseMeta{})
		require.Error(t, err)
		assert.Equal(t, "unsupported", err.(*chat.Error).Code)
	})

	t.Run("opaque reasoning", func(t *testing.T) {
		_, err := Render(chat.Response{Output: []chat.Item{{Type: chat.ItemReasoning, Data: jsontext.Value(`{"x":1}`)}}})
		require.Error(t, err)
		assert.Equal(t, "unsupported", err.(*chat.Error).Code)
	})

	t.Run("invalid function output", func(t *testing.T) {
		_, err := Render(chat.Response{Output: []chat.Item{{Type: chat.ItemFunctionCallOutput, Output: []chat.Part{{Type: chat.PartImage}}}}})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "missing")
	})
}
