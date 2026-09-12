// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package completions

import (
	json "encoding/json/v2"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/buger/jsonparser"
	chat "github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/execution"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/kelindar/llmux/internal/wire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func decodeObject(t *testing.T, raw string) []byte {
	t.Helper()
	data := []byte(raw)
	require.NoError(t, wire.ValidateObject(data))
	return data
}

func decodeField(t *testing.T, raw string) field {
	t.Helper()
	value, typ, _, err := jsonparser.Get([]byte(raw))
	require.NoError(t, err)
	return field{Raw: value, Type: typ}
}

func requireAPIError(t *testing.T, err error, code, param string) {
	t.Helper()
	require.Error(t, err)
	apiErr, ok := errors.AsType[*chat.Error](err)
	require.True(t, ok, "expected chat.APIError, got %T", err)
	assert.Equal(t, code, apiErr.Code)
	if param != "" {
		assert.Equal(t, param, apiErr.Param)
	}
}

func TestParseRequest(t *testing.T) {
	cases := map[string]struct {
		body    string
		wantErr bool
		code    string
		param   string
		check   func(t *testing.T, parsed parsedRequest)
	}{
		"minimalUserMessage": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, internalprotocol.Chat, parsed.Kind)
				assert.Equal(t, "gpt-4", parsed.Request.Target)
				assert.False(t, parsed.Stream)
				require.Len(t, parsed.Request.Input, 1)
				assert.Equal(t, chat.RoleUser, parsed.Request.Input[0].Role)
			},
		},
		"streamWithTools": {
			body: `{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"search","description":"find","parameters":{"type":"object"}}}],"tool_choice":"auto","stop":"END","max_tokens":128,"temperature":0.5,"top_p":0.9}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.True(t, parsed.Stream)
				require.Len(t, parsed.Request.Controls.Tools, 1)
				assert.Equal(t, "search", parsed.Request.Controls.Tools[0].Name)
				require.NotNil(t, parsed.Request.Controls.ToolChoice)
				assert.Equal(t, "auto", parsed.Request.Controls.ToolChoice.Mode)
				assert.Equal(t, []string{"END"}, parsed.Request.Controls.Stop)
			},
		},
		"assistantToolCalls": {
			body: `{"model":"gpt-4","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"fn","arguments":"{}"}}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.Len(t, parsed.Request.Input, 1)
				assert.Equal(t, chat.ItemFunctionCall, parsed.Request.Input[0].Type)
				assert.Equal(t, "call_1", parsed.Request.Input[0].CallID)
			},
		},
		"toolRoleOutput": {
			body: `{"model":"gpt-4","messages":[{"role":"tool","tool_call_id":"call_1","content":"result"}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.Len(t, parsed.Request.Input, 1)
				assert.Equal(t, chat.ItemFunctionCallOutput, parsed.Request.Input[0].Type)
			},
		},
		"responseFormatJsonSchema": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema","json_schema":{"name":"out","schema":{"type":"object","properties":{"a":{"type":"string"}}}}}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.FormatJSONSchema, parsed.Request.Output.Format.Kind)
				assert.Equal(t, "out", parsed.Request.Output.Format.Name)
			},
		},
		"modalitiesWithAudio": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"speak"}],"modalities":["text","audio"],"audio":{"voice":"alloy","format":"wav"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.True(t, parsed.Request.Output.Modalities.Has(chat.ModalityText))
				assert.True(t, parsed.Request.Output.Modalities.Has(chat.ModalityAudio))
				require.NotNil(t, parsed.Request.Controls.Audio)
				assert.Equal(t, "alloy", parsed.Request.Controls.Audio.Voice)
			},
		},
		"streamOptionsUsage": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"stream_options":{"include_usage":true}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.True(t, parsed.IncludeUsage)
			},
		},
		"stopArray": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"stop":["a","b"]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, []string{"a", "b"}, parsed.Request.Controls.Stop)
			},
		},
		"toolChoiceFunction": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"function","function":{"name":"search"}}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.NotNil(t, parsed.Request.Controls.ToolChoice)
				assert.Equal(t, "function", parsed.Request.Controls.ToolChoice.Mode)
				assert.Equal(t, "search", parsed.Request.Controls.ToolChoice.Name)
			},
		},
		"xExtensionAllowed": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"x-custom":"value"}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.NotNil(t, parsed.Request.Controls.Extensions)
				assert.Contains(t, parsed.Request.Controls.Extensions, "x-custom")
			},
		},
		"unknownField": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"bogus":1}`,
			wantErr: true,
			code:    "unsupported",
			param:   "bogus",
		},
		"missingModel": {
			body:    `{"messages":[{"role":"user","content":"x"}]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "model",
		},
		"missingMessages": {
			body:    `{"model":"gpt-4"}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "messages",
		},
		"emptyMessages": {
			body:    `{"model":"gpt-4","messages":[]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "messages",
		},
		"invalidToolChoice": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"tool_choice":"maybe"}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "tool_choice",
		},
		"audioWithoutControls": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"modalities":["text","audio"]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "audio",
		},
		"unsupportedModality": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"modalities":["video"]}`,
			wantErr: true,
			code:    "unsupported",
			param:   "modalities",
		},
		"invalidStop": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"stop":123}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "stop",
		},
		"streamOptionsUnknown": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"stream_options":{"bogus":true}}`,
			wantErr: true,
			code:    "unsupported",
			param:   "bogus",
		},
		"invalidToolArguments": {
			body:    `{"model":"gpt-4","messages":[{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"not-json"}}]}]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "messages.tool_calls.function.arguments",
		},
		"unsupportedToolType": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"tools":[{"type":"code_interpreter"}]}`,
			wantErr: true,
			code:    "unsupported",
			param:   "tools",
		},
		"bothMaxTokens": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"max_tokens":10,"max_completion_tokens":20}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "max_completion_tokens",
		},
		"unsupportedUserField": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"user":"u1"}`,
			wantErr: true,
			code:    "unsupported",
			param:   "user",
		},
		"unsupportedLogprobs": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"logprobs":true}`,
			wantErr: true,
			code:    "unsupported",
			param:   "logprobs",
		},
		"invalidRole": {
			body:    `{"model":"gpt-4","messages":[{"role":"bogus","content":"x"}]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "messages.role",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			parsed, err := ParseRequest(decodeObject(t, tc.body))
			if tc.wantErr {
				requireAPIError(t, err, tc.code, tc.param)
				return
			}
			require.NoError(t, err)
			if tc.check != nil {
				tc.check(t, parsed)
			}
		})
	}
}

func TestAdapterValidateEvent(t *testing.T) {
	adapter := Adapter{}
	cases := map[string]struct {
		event   chat.Event
		wantErr bool
	}{
		"textMessage": {
			event: chat.Event{Type: chat.EventItem, Item: chat.MessageItem(chat.RoleAssistant, chat.TextPart("hi"))},
		},
		"toolCall": {
			event: chat.Tool("c1", "fn", `{}`),
		},
		"reasoningRejected": {
			event:   chat.Event{Type: chat.EventItem, Item: chat.Item{Type: chat.ItemReasoning}},
			wantErr: true,
		},
		"imageMediaRejected": {
			event:   chat.MediaItem(chat.ImagePart(chat.InlineMedia("image/png", []byte{1}))),
			wantErr: true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := adapter.ValidateEvent(tc.event)
			if tc.wantErr {
				requireAPIError(t, err, "unsupported", "output")
				return
			}
			require.NoError(t, err)
		})
	}
}

func TestAdapterResponse(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{Response: chat.Response{ID: "chatcmpl-1", Created: 100, Target: "gpt-4"}}
	req := chat.Request{Target: "gpt-4"}

	t.Run("textOnly", func(t *testing.T) {
		result := execution.Result{
			Items:   []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("hello"))},
			Outcome: chat.Outcome{Status: chat.StatusCompleted},
		}
		value, err := adapter.Response(req, result, meta)
		require.NoError(t, err)
		response := value.(completionResponse)
		require.Len(t, response.Choices, 1)
		assert.Equal(t, "hello", response.Choices[0].Message.Content)
		assert.Equal(t, "stop", response.Choices[0].FinishReason)
	})

	t.Run("toolCall", func(t *testing.T) {
		result := execution.Result{
			Items:   []chat.Item{chat.FunctionCallItem("call_1", "search", `{"q":"x"}`)},
			Outcome: chat.Outcome{Status: chat.StatusCompleted, StopReason: chat.StopToolCall},
		}
		value, err := adapter.Response(req, result, meta)
		require.NoError(t, err)
		data, err := json.Marshal(value)
		require.NoError(t, err)
		assert.Contains(t, string(data), `"tool_calls"`)
		assert.Contains(t, string(data), `"call_1"`)
	})

	t.Run("mediaItem", func(t *testing.T) {
		audio := chat.InlineMedia("audio/wav", []byte{1})
		audio.Format = "wav"
		result := execution.Result{
			Items:   []chat.Item{chat.Item{Type: chat.ItemMedia, Content: []chat.Part{chat.AudioPart(audio)}}},
			Outcome: chat.Outcome{Status: chat.StatusCompleted},
		}
		value, err := adapter.Response(chat.Request{}, result, meta)
		require.NoError(t, err)
		data, err := json.Marshal(value)
		require.NoError(t, err)
		assert.Contains(t, string(data), `"audio"`)
	})

	t.Run("reasoningRejected", func(t *testing.T) {
		result := execution.Result{
			Items:   []chat.Item{chat.Item{Type: chat.ItemReasoning}},
			Outcome: chat.Outcome{Status: chat.StatusCompleted},
		}
		_, err := adapter.Response(chat.Request{}, result, meta)
		requireAPIError(t, err, "unsupported", "output")
	})
}

func TestAdapterStream(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{Response: chat.Response{ID: "chatcmpl-1", Created: 100, Target: "gpt-4"}}
	req := chat.Request{Target: "gpt-4"}
	limits := chat.DefaultLimits()

	t.Run("textDelta", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{Request: req, IncludeUsage: true}, &meta, limits)
		require.NoError(t, stream.Event(chat.TextDelta("hel")))
		require.NoError(t, stream.Event(chat.TextDelta("lo")))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted, Usage: &chat.Usage{Input: 1, Output: 2, Total: 3}}, nil))
		assert.Contains(t, rec.Body.String(), "chat.completion.chunk")
		assert.Contains(t, rec.Body.String(), "[DONE]")
		assert.Contains(t, rec.Body.String(), `"usage"`)
	})

	t.Run("toolCallFlow", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{Request: req}, &meta, limits)
		require.NoError(t, stream.Event(chat.ToolStart("call_1", "search")))
		require.NoError(t, stream.Event(chat.ToolDelta("call_1", `{"q"`)))
		require.NoError(t, stream.Event(chat.ToolDelta("call_1", `:"x"}`)))
		require.NoError(t, stream.Event(chat.ToolDone("call_1")))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted, StopReason: chat.StopToolCall}, nil))
		assert.Contains(t, rec.Body.String(), "tool_calls")
		assert.Equal(t, "tool_calls", finishReasonFromStream(t, rec.Body.String()))
	})
}

func finishReasonFromStream(t *testing.T, body string) string {
	t.Helper()
	for _, line := range splitLines(body) {
		if len(line) < 6 || line[:5] != "data:" {
			continue
		}
		data := line[6:]
		if data == "[DONE]" {
			continue
		}
		var chunk map[string]any
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		choices, ok := chunk["choices"].([]any)
		if !ok || len(choices) == 0 {
			continue
		}
		if reason, ok := choices[0].(map[string]any)["finish_reason"]; ok && reason != nil {
			return reason.(string)
		}
	}
	t.Fatal("finish_reason not found")
	return ""
}

func splitLines(s string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			lines = append(lines, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}

func InlineMedia(mime string, data []byte) chat.Media {
	return chat.InlineMedia(mime, data)
}

func ImagePart(media chat.Media) chat.Part {
	return chat.ImagePart(media)
}

func TextDelta(text string) chat.Event {
	return chat.Event{Type: chat.EventTextDelta, Delta: text}
}

func ToolCallStart(callID, name string) chat.Event {
	return chat.Event{Type: chat.EventToolCallStart, CallID: callID, Name: name}
}

func ToolCallDelta(callID, delta string) chat.Event {
	return chat.Event{Type: chat.EventToolCallDelta, CallID: callID, Delta: delta}
}

func ToolCallDone(callID string) chat.Event {
	return chat.Event{Type: chat.EventToolCallDone, CallID: callID}
}

func TestParseRequestExtra(t *testing.T) {
	cases := map[string]struct {
		body    string
		wantErr bool
		code    string
		param   string
		check   func(t *testing.T, parsed parsedRequest)
	}{
		"jsonObjectFormat": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_object"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.FormatJSONObject, parsed.Request.Output.Format.Kind)
			},
		},
		"textFormat": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"response_format":{"type":"text"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.FormatText, parsed.Request.Output.Format.Kind)
			},
		},
		"toolChoiceByName": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"function","name":"search"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "search", parsed.Request.Controls.ToolChoice.Name)
			},
		},
		"parallelToolCalls": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"parallel_tool_calls":false}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.NotNil(t, parsed.Request.Controls.ParallelToolCall)
				assert.False(t, *parsed.Request.Controls.ParallelToolCall)
			},
		},
		"storeAndMetadata": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"store":true,"metadata":{"k":"v"},"reasoning_effort":"medium"}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.NotNil(t, parsed.Store)
				assert.True(t, *parsed.Store)
				assert.Equal(t, "v", parsed.Metadata["k"])
				require.NotNil(t, parsed.Request.Controls.Reasoning)
			},
		},
		"imageContent": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.PartImage, parsed.Request.Input[0].Content[0].Type)
			},
		},
		"audioInput": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"YQ==","format":"wav"}}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.PartAudio, parsed.Request.Input[0].Content[0].Type)
			},
		},
		"assistantEmptyContent": {
			body: `{"model":"gpt-4","messages":[{"role":"assistant","content":""}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.Len(t, parsed.Request.Input, 1)
			},
		},
		"unsupportedResponseFormat": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"response_format":{"type":"grammar"}}`,
			wantErr: true, code: "unsupported", param: "response_format",
		},
		"invalidJsonSchema": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema","json_schema":{"name":"x"}}}`,
			wantErr: true, code: "invalid_request", param: "response_format.json_schema.schema",
		},
		"unsupportedN": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"n":2}`,
			wantErr: true, code: "unsupported", param: "n",
		},
		"topLogprobs": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"top_logprobs":5}`,
			wantErr: true, code: "unsupported", param: "top_logprobs",
		},
		"invalidMaxTokens": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"max_tokens":0}`,
			wantErr: true, code: "invalid_request", param: "max_tokens",
		},
		"invalidTemperature": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"temperature":3}`,
			wantErr: true, code: "invalid_request", param: "temperature",
		},
		"unsupportedToolCallType": {
			body:    `{"model":"gpt-4","messages":[{"role":"assistant","tool_calls":[{"id":"c1","type":"custom","function":{"name":"f","arguments":"{}"}}]}]}`,
			wantErr: true, code: "unsupported", param: "messages.tool_calls.type",
		},
		"emptyModalities": {
			body:    `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"modalities":[]}`,
			wantErr: true, code: "invalid_request", param: "modalities",
		},
		"fileContent": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":[{"type":"file","file":{"file_data":"YQ==","filename":"a.txt"}}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.PartFile, parsed.Request.Input[0].Content[0].Type)
			},
		},
		"toolChoiceNone": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"tool_choice":"none"}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "none", parsed.Request.Controls.ToolChoice.Mode)
			},
		},
		"toolChoiceLegacy": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"function","name":"search"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "search", parsed.Request.Controls.ToolChoice.Name)
			},
		},
		"toolChoiceRequired": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"tool_choice":"required"}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "required", parsed.Request.Controls.ToolChoice.Mode)
			},
		},
		"jsonSchemaFull": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema","json_schema":{"name":"out","description":"desc","schema":{"type":"object"},"strict":true}}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "desc", parsed.Request.Output.Format.Description)
				assert.True(t, parsed.Request.Output.Format.Strict)
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			parsed, err := ParseRequest(decodeObject(t, tc.body))
			if tc.wantErr {
				requireAPIError(t, err, tc.code, tc.param)
				return
			}
			require.NoError(t, err)
			if tc.check != nil {
				tc.check(t, parsed)
			}
		})
	}
}

func TestParseContentCoverage(t *testing.T) {
	cases := map[string]struct {
		raw     string
		wantErr bool
		check   func(t *testing.T, parts []chat.Part)
	}{
		"null": {raw: `null`},
		"plainText": {
			raw: `"hello"`,
			check: func(t *testing.T, parts []chat.Part) {
				require.Len(t, parts, 1)
				assert.Equal(t, chat.PartText, parts[0].Type)
			},
		},
		"wrongType":       {raw: `1`, wantErr: true},
		"emptyArray":      {raw: `[]`},
		"partWrongType":   {raw: `[1]`, wantErr: true},
		"partMissingType": {raw: `[{}]`, wantErr: true},
		"textPart": {
			raw: `[{"type":"text","text":"hello"}]`,
			check: func(t *testing.T, parts []chat.Part) {
				require.Len(t, parts, 1)
				assert.Equal(t, "hello", parts[0].Text)
			},
		},
		"textMissing":      {raw: `[{"type":"text"}]`, wantErr: true},
		"textWrongType":    {raw: `[{"type":"text","text":1}]`, wantErr: true},
		"textMixed":        {raw: `[{"type":"text","text":"x","file":{}}]`, wantErr: true},
		"unknownPartField": {raw: `[{"type":"text","text":"x","extra":true}]`, wantErr: true},
		"imageURL": {
			raw: `[{"type":"image_url","image_url":{"url":"https://example.com/a.png","detail":"high"}}]`,
			check: func(t *testing.T, parts []chat.Part) {
				require.Len(t, parts, 1)
				assert.Equal(t, chat.PartImage, parts[0].Type)
				assert.Equal(t, "high", parts[0].Detail)
			},
		},
		"imageWrongObject": {raw: `[{"type":"image_url","image_url":"url"}]`, wantErr: true},
		"imageMissingURL":  {raw: `[{"type":"image_url","image_url":{}}]`, wantErr: true},
		"imageBadDetail":   {raw: `[{"type":"image_url","image_url":{"url":"https://example.com/a.png","detail":"bad"}}]`, wantErr: true},
		"imageBadURL":      {raw: `[{"type":"image_url","image_url":{"url":"ftp://example.com/a.png"}}]`, wantErr: true},
		"audioMP3": {
			raw: `[{"type":"input_audio","input_audio":{"data":"YQ==","format":"mp3"}}]`,
			check: func(t *testing.T, parts []chat.Part) {
				require.Len(t, parts, 1)
				assert.Equal(t, chat.PartAudio, parts[0].Type)
				assert.Equal(t, "mp3", parts[0].Media.Format)
			},
		},
		"audioWrongObject":  {raw: `[{"type":"input_audio","input_audio":"audio"}]`, wantErr: true},
		"audioUnknownField": {raw: `[{"type":"input_audio","input_audio":{"data":"YQ==","format":"wav","extra":true}}]`, wantErr: true},
		"audioBadData":      {raw: `[{"type":"input_audio","input_audio":{"data":"!!!","format":"wav"}}]`, wantErr: true},
		"audioBadFormat":    {raw: `[{"type":"input_audio","input_audio":{"data":"YQ==","format":"flac"}}]`, wantErr: true},
		"fileURL": {
			raw: `[{"type":"file","file":{"file_url":"https://example.com/a.txt","filename":"a.txt"}}]`,
			check: func(t *testing.T, parts []chat.Part) {
				require.Len(t, parts, 1)
				assert.Equal(t, chat.PartFile, parts[0].Type)
				assert.Equal(t, "https://example.com/a.txt", parts[0].Media.URL)
			},
		},
		"fileID": {
			raw: `[{"type":"file","file":{"file_id":"file-1"}}]`,
			check: func(t *testing.T, parts []chat.Part) {
				require.Len(t, parts, 1)
				assert.Equal(t, "file-1", parts[0].Media.Ref)
			},
		},
		"fileMissingObject":   {raw: `[{"type":"file"}]`, wantErr: true},
		"fileMultipleSources": {raw: `[{"type":"file","file":{"file_id":"file-1","file_url":"https://example.com/a"}}]`, wantErr: true},
		"fileUnknownField":    {raw: `[{"type":"file","file":{"file_id":"file-1","extra":true}}]`, wantErr: true},
		"unknownType":         {raw: `[{"type":"video"}]`, wantErr: true},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			parts, err := parseChatContent(decodeField(t, tc.raw))
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			if tc.check != nil {
				tc.check(t, parts)
			}
		})
	}
}

func TestAdapterConstruct(t *testing.T) {
	assert.NotNil(t, NewAdapter())
}

func TestAdapterValidateMore(t *testing.T) {
	adapter := Adapter{}
	audio := chat.InlineMedia("audio/wav", []byte{1, 2, 3})
	audio.Format = "wav"
	cases := map[string]struct {
		event   chat.Event
		wantErr bool
	}{
		"audioMessagePart": {
			event: chat.Event{Type: chat.EventItem, Item: chat.MessageItem(chat.RoleAssistant, chat.AudioPart(audio))},
		},
		"audioMediaOutput": {
			event: chat.MediaItem(chat.AudioPart(audio)),
		},
		"audioItemMedia": {
			event: chat.Event{Type: chat.EventItem, Item: chat.Item{Type: chat.ItemMedia, Content: []chat.Part{chat.AudioPart(audio)}}},
		},
		"functionCallItem": {
			event: chat.Event{Type: chat.EventItem, Item: chat.FunctionCallItem("c1", "fn", `{}`)},
		},
		"unsupportedItem": {
			event:   chat.Event{Type: chat.EventItem, Item: chat.Item{Type: chat.ItemReasoning}},
			wantErr: true,
		},
		"messageImagePart": {
			event:   chat.Event{Type: chat.EventItem, Item: chat.MessageItem(chat.RoleAssistant, chat.ImagePart(chat.InlineMedia("image/png", []byte{1})))},
			wantErr: true,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := adapter.ValidateEvent(tc.event)
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
		})
	}
}

func TestAdapterResponseErrors(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{Response: chat.Response{ID: "chatcmpl-1", Created: 100, Target: "gpt-4"}}
	audio := chat.InlineMedia("audio/wav", []byte{1})
	audio.Format = "wav"
	audio2 := chat.InlineMedia("audio/wav", []byte{2})
	audio2.Format = "wav"
	result := execution.Result{
		Items:   []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.AudioPart(audio), chat.AudioPart(audio2))},
		Outcome: chat.Outcome{Status: chat.StatusCompleted},
	}
	_, err := adapter.Response(chat.Request{}, result, meta)
	requireAPIError(t, err, "unsupported", "output")
}

func TestAdapterResponseAudio(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{Response: chat.Response{ID: "chatcmpl-1", Created: 100, Target: "gpt-4"}}
	audio := chat.InlineMedia("audio/wav", []byte{1, 2, 3})
	audio.Format = "wav"
	audio.Ref = "audio_ref"
	result := execution.Result{
		Items:   []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.AudioPart(audio))},
		Outcome: chat.Outcome{Status: chat.StatusCompleted},
	}
	value, err := adapter.Response(chat.Request{}, result, meta)
	require.NoError(t, err)
	data, err := json.Marshal(value)
	require.NoError(t, err)
	assert.Contains(t, string(data), `"audio"`)
}

func TestParseRequestRejects(t *testing.T) {
	cases := map[string]string{
		"invalidMetadata":     `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"metadata":{"k":1}}`,
		"invalidStream":       `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"stream":"yes"}`,
		"invalidStore":        `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"store":"yes"}`,
		"invalidParallel":     `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"parallel_tool_calls":"yes"}`,
		"invalidTopP":         `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"top_p":"bad"}`,
		"invalidMessagesType": `{"model":"gpt-4","messages":"x"}`,
		"toolChoiceBadType":   `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"custom","name":"x"}}`,
		"toolChoiceBadFunc":   `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"function","function":{"name":"x","extra":1}}}`,
		"audioMissingVoice":   `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"modalities":["text","audio"],"audio":{"format":"wav"}}`,
		"strictTools":         `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f","parameters":{},"strict":"yes"}}]}`,
		"invalidToolCallType": `{"model":"gpt-4","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"custom","function":{"name":"f","arguments":"{}"}}]}]}`,
		"invalidToolCallArgs": `{"model":"gpt-4","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"not-json"}}]}]}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ParseRequest(decodeObject(t, body))
			require.Error(t, err)
		})
	}
}

func TestParseControls(t *testing.T) {
	cases := map[string]struct {
		body  string
		check func(t *testing.T, parsed parsedRequest)
	}{
		"maxCompletionTokens": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"max_completion_tokens":12}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.NotNil(t, parsed.Request.Controls.MaxOutputTokens)
				assert.Equal(t, 12, *parsed.Request.Controls.MaxOutputTokens)
			},
		},
		"toolWithoutWrapper": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"tools":[{"function":{"name":"lookup"}}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.Len(t, parsed.Request.Controls.Tools, 1)
				assert.Equal(t, "lookup", parsed.Request.Controls.Tools[0].Name)
			},
		},
		"strictTool": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"lookup","strict":true}}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.Len(t, parsed.Request.Controls.Tools, 1)
				require.NotNil(t, parsed.Request.Controls.Tools[0].Strict)
				assert.True(t, *parsed.Request.Controls.Tools[0].Strict)
			},
		},
		"streamOptionsWithoutUsage": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"stream_options":{}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.False(t, parsed.IncludeUsage)
			},
		},
		"nullControls": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"stop":null,"store":null,"stream":null}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Nil(t, parsed.Request.Controls.Stop)
				require.NotNil(t, parsed.Store)
				assert.False(t, *parsed.Store)
				assert.False(t, parsed.Stream)
			},
		},
		"jsonSchemaDefaults": {
			body: `{"model":"gpt-4","messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_schema","json_schema":{"name":"result","schema":{"type":"object"}}}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.FormatJSONSchema, parsed.Request.Output.Format.Kind)
				assert.False(t, parsed.Request.Output.Format.Strict)
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			parsed, err := ParseRequest(decodeObject(t, tc.body))
			require.NoError(t, err)
			tc.check(t, parsed)
		})
	}
}

func TestParseControlErrors(t *testing.T) {
	base := `"model":"gpt-4","messages":[{"role":"user","content":"x"}]`
	cases := map[string]string{
		"invalidMaxCompletionTokens":  `{` + base + `,"max_completion_tokens":"12"}`,
		"negativeMaxCompletionTokens": `{` + base + `,"max_completion_tokens":0}`,
		"invalidTopPRange":            `{` + base + `,"top_p":2}`,
		"stopArrayValue":              `{` + base + `,"stop":["ok",1]}`,
		"toolsNotArray":               `{` + base + `,"tools":{}}`,
		"toolNotObject":               `{` + base + `,"tools":[1]}`,
		"toolUnknownField":            `{` + base + `,"tools":[{"function":{"name":"f"},"extra":true}]}`,
		"toolMissingFunction":         `{` + base + `,"tools":[{}]}`,
		"toolFunctionNotObject":       `{` + base + `,"tools":[{"function":"f"}]}`,
		"toolMissingName":             `{` + base + `,"tools":[{"function":{}}]}`,
		"toolBadDescription":          `{` + base + `,"tools":[{"function":{"name":"f","description":1}}]}`,
		"toolBadStrict":               `{` + base + `,"tools":[{"function":{"name":"f","strict":"yes"}}]}`,
		"toolCallNotObject":           `{"model":"gpt-4","messages":[{"role":"assistant","tool_calls":[1]}]}`,
		"toolCallMissingFunction":     `{"model":"gpt-4","messages":[{"role":"assistant","tool_calls":[{"id":"c1"}]}]}`,
		"toolCallFunctionNotObject":   `{"model":"gpt-4","messages":[{"role":"assistant","tool_calls":[{"id":"c1","function":"f"}]}]}`,
		"toolCallMissingName":         `{"model":"gpt-4","messages":[{"role":"assistant","tool_calls":[{"id":"c1","function":{}}]}]}`,
		"toolCallMissingArguments":    `{"model":"gpt-4","messages":[{"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"f"}}]}]}`,
		"toolCallBadType":             `{"model":"gpt-4","messages":[{"role":"assistant","tool_calls":[{"id":"c1","type":1,"function":{"name":"f","arguments":"{}"}}]}]}`,
		"toolCallUnknownField":        `{"model":"gpt-4","messages":[{"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"f","arguments":"{}","extra":true}}]}]}`,
		"toolChoiceFunctionField":     `{` + base + `,"tool_choice":{"type":"function","function":{"name":"f"},"name":"legacy"}}`,
		"toolChoiceFunctionNotObject": `{` + base + `,"tool_choice":{"type":"function","function":"f"}}`,
		"toolChoiceMissingType":       `{` + base + `,"tool_choice":{}}`,
		"toolChoiceBadType":           `{` + base + `,"tool_choice":{"type":1,"name":"f"}}`,
		"toolChoiceMissingName":       `{` + base + `,"tool_choice":{"type":"function"}}`,
		"invalidModalitiesType":       `{` + base + `,"modalities":"text"}`,
		"invalidModalityElement":      `{` + base + `,"modalities":[1]}`,
		"audioNotObject":              `{` + base + `,"audio":"wav"}`,
		"audioMissingFormat":          `{` + base + `,"audio":{"voice":"alloy"}}`,
		"streamOptionsNotObject":      `{` + base + `,"stream_options":true}`,
		"streamOptionsBadUsage":       `{` + base + `,"stream_options":{"include_usage":"yes"}}`,
		"responseFormatNotObject":     `{` + base + `,"response_format":"json_object"}`,
		"responseFormatMissingType":   `{` + base + `,"response_format":{}}`,
		"schemaNotObject":             `{` + base + `,"response_format":{"type":"json_schema","json_schema":true}}`,
		"schemaUnknownField":          `{` + base + `,"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{},"extra":true}}}`,
		"schemaBadDescription":        `{` + base + `,"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{},"description":1}}}`,
		"schemaBadStrict":             `{` + base + `,"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{},"strict":"yes"}}}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ParseRequest(decodeObject(t, body))
			require.Error(t, err)
		})
	}
}

func TestWireDelegates(t *testing.T) {
	media, err := parseMediaURL("https://example.com/a.png", "auto")
	require.NoError(t, err)
	assert.Equal(t, "https://example.com/a.png", media.URL)

	fileMedia, name, err := parseFileMedia([]byte(`{"file_data":"YQ==","filename":"a.txt"}`), "file")
	require.NoError(t, err)
	assert.Equal(t, "a.txt", name)
	assert.NotEmpty(t, fileMedia.Data)

	assert.Equal(t, "audio/wav", audioMIME("wav"))
	parts := outputTextParts([]chat.Part{chat.TextPart("x")})
	require.Len(t, parts, 1)
	url, err := mediaDataURL(chat.InlineMedia("image/png", []byte{1}))
	require.NoError(t, err)
	assert.Contains(t, url, "data:image/png")
	assert.NotEmpty(t, newID())
}

func TestAdapterResponseCombined(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{Response: chat.Response{ID: "chatcmpl-1", Created: 100, Target: "gpt-4"}}
	audio := chat.InlineMedia("audio/wav", []byte{1})
	audio.Format = "wav"
	result := execution.Result{
		Items: []chat.Item{
			chat.MessageItem(chat.RoleAssistant, chat.TextPart("hello"), chat.AudioPart(audio)),
			chat.FunctionCallItem("call_1", "search", `{}`),
		},
		Outcome: chat.Outcome{Status: chat.StatusCompleted, StopReason: chat.StopToolCall, Usage: &chat.Usage{Input: 1, Output: 2, Total: 3}},
	}
	value, err := adapter.Response(chat.Request{}, result, meta)
	require.NoError(t, err)
	data, err := json.Marshal(value)
	require.NoError(t, err)
	assert.Contains(t, string(data), `"tool_calls"`)
	assert.Contains(t, string(data), `"usage"`)
}

func TestAdapterStreamErrors(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{Response: chat.Response{ID: "chatcmpl-1", Created: 100, Target: "gpt-4"}}
	limits := chat.DefaultLimits()
	rec := httptest.NewRecorder()
	stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
	require.NoError(t, stream.Event(chat.ToolStart("call_1", "search")))
	err := stream.Event(ToolCallDelta("missing", `{}`))
	require.Error(t, err)
	assert.True(t, stream.Started())
	rec2 := httptest.NewRecorder()
	stream2 := adapter.Stream(rec2, parsedRequest{}, &meta, limits)
	require.NoError(t, stream2.Event(TextDelta("x")))
	assert.True(t, stream2.Started())
}

func TestAdapterStreamMore(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{Response: chat.Response{ID: "chatcmpl-1", Created: 100, Target: "gpt-4"}}
	limits := chat.DefaultLimits()

	t.Run("completeToolCall", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(chat.Tool("call_1", "search", `{"q":"x"}`)))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted, StopReason: chat.StopToolCall}, nil))
		assert.Contains(t, rec.Body.String(), "search")
	})

	t.Run("audioStream", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		audio := chat.InlineMedia("audio/wav", []byte{9, 8})
		audio.Format = "wav"
		require.NoError(t, stream.Event(chat.MediaItem(chat.AudioPart(audio))))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil))
		assert.Contains(t, rec.Body.String(), "audio")
	})

	t.Run("eventItemMessage", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(chat.Event{Type: chat.EventItem, Item: chat.MessageItem(chat.RoleAssistant, chat.TextPart("via item"))}))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil))
	})

	t.Run("eventItemToolCall", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(chat.Event{Type: chat.EventItem, Item: chat.FunctionCallItem("call_1", "search", `{}`)}))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted, StopReason: chat.StopToolCall}, nil))
	})

	t.Run("lengthFinish", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(TextDelta("x")))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusIncomplete, StopReason: chat.StopLength}, nil))
		assert.Equal(t, "length", finishReasonFromStream(t, rec.Body.String()))
	})

	t.Run("unsupportedEvent", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		err := stream.Event(chat.Event{Type: "nope"})
		require.Error(t, err)
	})

	t.Run("reasoningStreamError", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		err := stream.Event(chat.Event{Type: chat.EventItem})
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("badMediaItem", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		err := stream.Event(chat.Event{Type: chat.EventItem, Item: chat.Item{Type: chat.ItemMedia, Content: []chat.Part{}}})
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("completeUsage", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{IncludeUsage: true}, &meta, limits)
		require.NoError(t, stream.Event(TextDelta("x")))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted, Usage: &chat.Usage{Input: 1, Output: 2, Total: 3}}, nil))
		assert.Contains(t, rec.Body.String(), `"usage"`)
	})

	t.Run("messageAudio", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		audio := chat.InlineMedia("audio/wav", []byte{1, 2})
		audio.Format = "wav"
		item := chat.MessageItem(chat.RoleAssistant, chat.AudioPart(audio))
		require.NoError(t, stream.Event(chat.Event{Type: chat.EventItem, Item: item}))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil))
		assert.Contains(t, rec.Body.String(), `"audio"`)
	})

	t.Run("mediaItemEvent", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		audio := chat.InlineMedia("audio/wav", []byte{3})
		audio.Format = "wav"
		item := chat.Item{Type: chat.ItemMedia, Content: []chat.Part{chat.AudioPart(audio)}}
		require.NoError(t, stream.Event(chat.Event{Type: chat.EventItem, Item: item}))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil))
	})
}

func TestSSEWrite(t *testing.T) {
	rec := httptest.NewRecorder()
	sw := &sseWriter{w: rec, limits: chat.DefaultLimits()}
	require.NoError(t, sw.start())
	require.NoError(t, sw.write("", map[string]any{"ok": true}))
	require.NoError(t, sw.done())
	assert.True(t, sw.started)
}

func TestAdapterStreamFail(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{Response: chat.Response{ID: "chatcmpl-1", Created: 100, Target: "gpt-4"}}
	limits := chat.DefaultLimits()

	t.Run("beforeStart", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		err := stream.Fail(chat.Invalid("model", "bad model"))
		requireAPIError(t, err, "invalid_request", "model")
		assert.Equal(t, http.StatusBadRequest, rec.Code)
	})

	t.Run("afterStart", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(TextDelta("x")))
		require.NoError(t, stream.Fail(errors.New("boom")))
		assert.Contains(t, rec.Body.String(), "error")
	})

	t.Run("afterStartWithParam", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(TextDelta("x")))
		require.NoError(t, stream.Fail(chat.Invalid("model", "bad")))
		assert.Contains(t, rec.Body.String(), `"param":"model"`)
	})
}

func TestParseRawContract(t *testing.T) {
	body := []byte(`{"model":"m\u006fdel","messages":[{"role":"user","content":"hello \u2603"}],"max_tokens":3,"metadata":{"k":"v"},"x-meta":{"n":1}}`)
	parsed, err := ParseRequest(body)
	require.NoError(t, err)
	assert.Equal(t, "model", parsed.Request.Target)
	assert.Equal(t, "hello ☃", parsed.Request.Input[0].Content[0].Text)
	require.NotNil(t, parsed.Request.Controls.MaxOutputTokens)
	assert.Equal(t, 3, *parsed.Request.Controls.MaxOutputTokens)
	assert.Equal(t, "v", parsed.Metadata["k"])
	assert.Equal(t, `{"n":1}`, string(parsed.Request.Controls.Extensions["x-meta"]))

	for i := range body {
		body[i] = 'x'
	}
	assert.Equal(t, "model", parsed.Request.Target)
	assert.Equal(t, "hello ☃", parsed.Request.Input[0].Content[0].Text)
	assert.Equal(t, "v", parsed.Metadata["k"])
	assert.Equal(t, `{"n":1}`, string(parsed.Request.Controls.Extensions["x-meta"]))

	cases := map[string]struct {
		body  string
		param string
	}{
		"missing":         {`{"model":"m"}`, "messages"},
		"null":            {`{"model":"m","messages":null}`, "messages"},
		"wrongType":       {`{"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":1.0}`, "max_tokens"},
		"duplicate":       {`{"model":"m","messages":[],"model":"m2"}`, "body"},
		"trailing":        {`{"model":"m","messages":[]} {}`, "body"},
		"malformedNested": {`{"model":"m","messages":[{"role":"user","content":[}`, "body"},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ParseRequest([]byte(test.body))
			requireAPIError(t, err, "invalid_request", test.param)
		})
	}
}
