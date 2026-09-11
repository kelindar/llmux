// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
package anthropic

import (
	"encoding/base64"
	"encoding/json/jsontext"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/execution"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/kelindar/llmux/internal/wire"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func decodeObject(t *testing.T, raw string) map[string]jsontext.Value {
	t.Helper()
	object, err := wire.DecodeObject([]byte(raw))
	require.NoError(t, err)
	return object
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
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"hello"}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, internalprotocol.Anthropic, parsed.Kind)
				assert.Equal(t, "claude-3", parsed.Request.Target)
				require.Len(t, parsed.Request.Input, 1)
				assert.Equal(t, chat.RoleUser, parsed.Request.Input[0].Role)
				require.NotNil(t, parsed.Request.Controls.MaxOutputTokens)
				assert.Equal(t, 64, *parsed.Request.Controls.MaxOutputTokens)
			},
		},
		"streamWithSystem": {
			body: `{"model":"claude-3","max_tokens":32,"stream":true,"system":"be helpful","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.True(t, parsed.Stream)
				assert.Equal(t, "be helpful", parsed.Request.Instructions)
			},
		},
		"toolUseAndResult": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"search","input":{"q":"x"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"found"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.Len(t, parsed.Request.Input, 2)
				assert.Equal(t, chat.ItemFunctionCall, parsed.Request.Input[0].Type)
				assert.Equal(t, chat.ItemFunctionCallOutput, parsed.Request.Input[1].Type)
			},
		},
		"imageBase64": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + base64.StdEncoding.EncodeToString([]byte{1, 2, 3}) + `"}}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.Len(t, parsed.Request.Input[0].Content, 1)
				assert.Equal(t, chat.PartImage, parsed.Request.Input[0].Content[0].Type)
			},
		},
		"toolsAndChoice": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"tools":[{"name":"search","description":"find","input_schema":{"type":"object"}}],"tool_choice":{"type":"auto"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.Len(t, parsed.Request.Controls.Tools, 1)
				assert.Equal(t, "search", parsed.Request.Controls.Tools[0].Name)
				require.NotNil(t, parsed.Request.Controls.ToolChoice)
				assert.Equal(t, "auto", parsed.Request.Controls.ToolChoice.Mode)
			},
		},
		"stopSequences": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"stop_sequences":["END"]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, []string{"END"}, parsed.Request.Controls.Stop)
			},
		},
		"thinkingDisabled": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"thinking":{"type":"disabled"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Nil(t, parsed.Request.Controls.Reasoning)
			},
		},
		"unknownField": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"bogus":1}`,
			wantErr: true,
			code:    "unsupported",
			param:   "bogus",
		},
		"missingModel": {
			body:    `{"max_tokens":64,"messages":[{"role":"user","content":"x"}]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "model",
		},
		"missingMaxTokens": {
			body:    `{"model":"claude-3","messages":[{"role":"user","content":"x"}]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "max_tokens",
		},
		"zeroMaxTokens": {
			body:    `{"model":"claude-3","max_tokens":0,"messages":[{"role":"user","content":"x"}]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "max_tokens",
		},
		"missingMessages": {
			body:    `{"model":"claude-3","max_tokens":64}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "messages",
		},
		"emptyMessages": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "messages",
		},
		"missingContent": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user"}]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "messages.content",
		},
		"invalidRole": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"tool","content":"x"}]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "messages.role",
		},
		"unsupportedThinking": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"thinking":{"type":"enabled"}}`,
			wantErr: true,
			code:    "unsupported",
			param:   "thinking",
		},
		"unsupportedServiceTier": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"service_tier":"fast"}`,
			wantErr: true,
			code:    "unsupported",
			param:   "service_tier",
		},
		"invalidToolChoiceName": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"auto","name":"search"}}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "tool_choice.name",
		},
		"invalidToolSchema": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"tools":[{"name":"search"}]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "tools.input_schema",
		},
		"invalidTemperature": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"temperature":2}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "temperature",
		},
		"unsupportedContentType": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"video","source":{}}]}]}`,
			wantErr: true,
			code:    "unsupported",
			param:   "messages.content.type",
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
		"mediaItemRejected": {
			event:   chat.Event{Type: chat.EventItem, Item: chat.Item{Type: chat.ItemMedia}},
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
	meta := responseMeta{Response: chat.Response{ID: "msg_1", Created: 100, Target: "claude-3"}}
	req := chat.Request{Target: "claude-3"}

	t.Run("textOnly", func(t *testing.T) {
		result := execution.Result{
			Items:   []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("hello"))},
			Outcome: chat.Outcome{Status: chat.StatusCompleted},
		}
		value, err := adapter.Response(req, result, meta)
		require.NoError(t, err)
		response := value.(messageResponse)
		require.Len(t, response.Content, 1)
		assert.Equal(t, "text", response.Content[0].(textBlock).Type)
		assert.Equal(t, "end_turn", response.StopReason)
	})

	t.Run("toolUse", func(t *testing.T) {
		result := execution.Result{
			Items:   []chat.Item{chat.FunctionCallItem("tu_1", "search", `{"q":"x"}`)},
			Outcome: chat.Outcome{Status: chat.StatusCompleted, StopReason: chat.StopToolCall},
		}
		value, err := adapter.Response(req, result, meta)
		require.NoError(t, err)
		response := value.(messageResponse)
		assert.Equal(t, "tool_use", response.Content[0].(toolUseBlock).Type)
		assert.Equal(t, "tool_use", response.StopReason)
	})

	t.Run("withUsage", func(t *testing.T) {
		result := execution.Result{
			Items:   []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("hi"))},
			Outcome: chat.Outcome{Status: chat.StatusCompleted, Usage: &chat.Usage{Input: 3, Output: 5}},
		}
		value, err := adapter.Response(req, result, meta)
		require.NoError(t, err)
		require.NotNil(t, value.(messageResponse).Usage)
		assert.Equal(t, 3, value.(messageResponse).Usage.InputTokens)
	})

	t.Run("unsupportedItem", func(t *testing.T) {
		result := execution.Result{
			Items:   []chat.Item{{Type: chat.ItemMedia}},
			Outcome: chat.Outcome{Status: chat.StatusCompleted},
		}
		_, err := adapter.Response(req, result, meta)
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("nonTextMessage", func(t *testing.T) {
		img := chat.InlineMedia("image/png", []byte{1})
		result := execution.Result{
			Items:   []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.ImagePart(img))},
			Outcome: chat.Outcome{Status: chat.StatusCompleted},
		}
		_, err := adapter.Response(req, result, meta)
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("invalidToolArguments", func(t *testing.T) {
		result := execution.Result{
			Items:   []chat.Item{chat.FunctionCallItem("tu_1", "search", "not-json")},
			Outcome: chat.Outcome{Status: chat.StatusCompleted},
		}
		_, err := adapter.Response(req, result, meta)
		require.Error(t, err)
	})
}

func TestAdapterStream(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{Response: chat.Response{ID: "msg_1", Created: 100, Target: "claude-3"}}
	req := chat.Request{Target: "claude-3"}
	limits := chat.DefaultLimits()

	t.Run("textDelta", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{Request: req}, &meta, limits)
		assert.False(t, stream.Started())
		require.NoError(t, stream.Event(TextDelta("hel")))
		assert.True(t, stream.Started())
		require.NoError(t, stream.Event(TextDelta("lo")))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil))
		body := rec.Body.String()
		assert.Contains(t, body, "message_start")
		assert.Contains(t, body, "text_delta")
		assert.Contains(t, body, "message_stop")
	})

	t.Run("toolCallFlow", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{Request: req}, &meta, limits)
		require.NoError(t, stream.Event(chat.Tool("call_1", "search", `{"q":"x"}`)))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted, StopReason: chat.StopToolCall}, nil))
		assert.Contains(t, rec.Body.String(), "tool_use")
	})
}

func TextDelta(text string) chat.Event {
	return chat.Event{Type: chat.EventTextDelta, Delta: text}
}

func ToolCallDelta(callID, delta string) chat.Event {
	return chat.Event{Type: chat.EventToolCallDelta, CallID: callID, Delta: delta}
}

func TestParseRequestExtra(t *testing.T) {
	cases := map[string]struct {
		body    string
		wantErr bool
		code    string
		param   string
		check   func(t *testing.T, parsed parsedRequest)
	}{
		"systemBlocks": {
			body: `{"model":"claude-3","max_tokens":64,"system":[{"type":"text","text":"a"},{"type":"text","text":"b"}],"messages":[{"role":"user","content":"x"}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "ab", parsed.Request.Instructions)
			},
		},
		"systemString": {
			body: `{"model":"claude-3","max_tokens":64,"system":"be helpful","messages":[{"role":"user","content":"x"}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "be helpful", parsed.Request.Instructions)
			},
		},
		"imageURL": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.PartImage, parsed.Request.Input[0].Content[0].Type)
			},
		},
		"documentBase64": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","data":"YQ=="}}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.PartFile, parsed.Request.Input[0].Content[0].Type)
			},
		},
		"toolChoiceByName": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"tool","name":"search"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "function", parsed.Request.Controls.ToolChoice.Mode)
				assert.Equal(t, "search", parsed.Request.Controls.ToolChoice.Name)
			},
		},
		"thinkingBlock": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":"sig"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.ItemReasoning, parsed.Request.Input[0].Type)
			},
		},
		"redactedThinking": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"assistant","content":[{"type":"redacted_thinking","data":"opaque"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.ItemReasoning, parsed.Request.Input[0].Type)
			},
		},
		"toolResultImage": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":[{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.ItemFunctionCallOutput, parsed.Request.Input[0].Type)
			},
		},
		"metadata": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"metadata":{"user_id":"u1"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "u1", parsed.Metadata["user_id"])
			},
		},
		"invalidImageBase64": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"!!!"}}]}]}`,
			wantErr: true, code: "invalid_request", param: "messages.content.source.data",
		},
		"invalidImageMime": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"text/plain","data":"YQ=="}}]}]}`,
			wantErr: true, code: "invalid_request", param: "messages.content.source.media_type",
		},
		"unsupportedSystemBlock": {
			body:    `{"model":"claude-3","max_tokens":64,"system":[{"type":"image","source":{}}],"messages":[{"role":"user","content":"x"}]}`,
			wantErr: true, code: "unsupported", param: "system",
		},
		"unsupportedToolChoice": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"custom"}}`,
			wantErr: true, code: "unsupported", param: "tool_choice.type",
		},
		"invalidToolInput": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"search"}]}]}`,
			wantErr: true, code: "invalid_request", param: "messages.content.input",
		},
		"documentURL": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"url","url":"https://example.com/doc.pdf"}}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.PartFile, parsed.Request.Input[0].Content[0].Type)
			},
		},
		"stringContent": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"plain"}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "plain", parsed.Request.Input[0].Content[0].Text)
			},
		},
		"toolChoiceAny": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"any"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "any", parsed.Request.Controls.ToolChoice.Mode)
			},
		},
		"toolChoiceAuto": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"auto"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "auto", parsed.Request.Controls.ToolChoice.Mode)
			},
		},
		"toolChoiceNone": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"none"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "none", parsed.Request.Controls.ToolChoice.Mode)
			},
		},
		"assistantTextAndTool": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"assistant","content":[{"type":"text","text":"hi"},{"type":"tool_use","id":"tu_1","name":"search","input":{"q":"x"}}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.Len(t, parsed.Request.Input, 2)
				assert.Equal(t, chat.ItemFunctionCall, parsed.Request.Input[1].Type)
			},
		},
		"toolResultString": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"ok"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.ItemFunctionCallOutput, parsed.Request.Input[0].Type)
			},
		},
		"toolResultTextArray": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":[{"type":"text","text":"ok"}]}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "ok", parsed.Request.Input[0].Output[0].Text)
			},
		},
		"documentWithMime": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"YQ=="}}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "application/pdf", parsed.Request.Input[0].Content[0].Media.MIMEType)
			},
		},
		"unsupportedToolResultType": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":[{"type":"document","source":{}}]}]}]}`,
			wantErr: true, code: "unsupported", param: "tool_result.content",
		},
		"toolChoiceNameWithAuto": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"auto","name":"search"}}`,
			wantErr: true, code: "invalid_request", param: "tool_choice.name",
		},
		"imageBase64Input": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"YQ=="}}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.PartImage, parsed.Request.Input[0].Content[0].Type)
				assert.Equal(t, "image/png", parsed.Request.Input[0].Content[0].Media.MIMEType)
			},
		},
		"toolResultImagePart": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":[{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.PartImage, parsed.Request.Input[0].Output[0].Type)
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

func TestNewAdapter(t *testing.T) {
	assert.NotNil(t, NewAdapter())
}

func TestAdapterResponseLength(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{Response: chat.Response{ID: "msg_1", Created: 100, Target: "claude-3"}}
	result := execution.Result{
		Items:   []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("truncated"))},
		Outcome: chat.Outcome{Status: chat.StatusIncomplete, StopReason: chat.StopLength},
	}
	value, err := adapter.Response(chat.Request{}, result, meta)
	require.NoError(t, err)
	assert.Equal(t, "max_tokens", value.(messageResponse).StopReason)
}

func TestParseRequestRejects(t *testing.T) {
	cases := map[string]string{
		"invalidMetadata":           `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"metadata":{"k":1}}`,
		"invalidStream":             `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"stream":"yes"}`,
		"invalidStop":               `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"stop_sequences":"END"}`,
		"invalidTopP":               `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"top_p":"bad"}`,
		"invalidMessages":           `{"model":"claude-3","max_tokens":64,"messages":"x"}`,
		"unsupportedImage":          `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"ftp","url":"x"}}]}]}`,
		"toolUnknownField":          `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":"x"}],"tools":[{"name":"f","input_schema":{},"extra":1}]}`,
		"invalidDocumentData":       `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","data":"!!!"}}]}]}`,
		"unsupportedDocumentSource": `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"ftp","url":"x"}}]}]}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ParseRequest(decodeObject(t, body))
			require.Error(t, err)
		})
	}
}

func TestWireDelegates(t *testing.T) {
	assert.True(t, validRole(chat.RoleUser))
	_, err := decodeStringSlice(map[string]jsontext.Value{"stop_sequences": jsontext.Value(`["END"]`)}, "stop_sequences")
	require.NoError(t, err)
	assert.NotEmpty(t, newID())
	_ = chat.AudioPart(chat.InlineMedia("audio/wav", []byte{1}))
}

func TestAdapterStreamMore(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{Response: chat.Response{ID: "msg_1", Created: 100, Target: "claude-3"}}
	limits := chat.DefaultLimits()

	t.Run("streamedToolDeltas", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(chat.ToolStart("call_1", "search")))
		require.NoError(t, stream.Event(ToolCallDelta("call_1", `{"q"`)))
		require.NoError(t, stream.Event(ToolCallDelta("call_1", `:"x"}`)))
		require.NoError(t, stream.Event(chat.ToolDone("call_1")))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted, StopReason: chat.StopToolCall}, nil))
		assert.Contains(t, rec.Body.String(), "input_json_delta")
	})

	t.Run("textDone", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(TextDelta("a")))
		require.NoError(t, stream.Event(chat.Event{Type: chat.EventTextDone}))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil))
	})

	t.Run("messageOutputItem", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(chat.OutputItem(chat.MessageItem(chat.RoleAssistant, chat.TextPart("via item")))))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil))
	})

	t.Run("completeOpenTool", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(chat.ToolStart("call_1", "search")))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted, StopReason: chat.StopToolCall}, []chat.Item{chat.FunctionCallItem("call_1", "search", `{}`)}))
	})

	t.Run("outputItem", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(chat.OutputItem(chat.FunctionCallItem("call_1", "search", `{"q":"x"}`))))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, []chat.Item{chat.FunctionCallItem("call_1", "search", `{"q":"x"}`)}))
	})

	t.Run("completeUsage", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(TextDelta("x")))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted, Usage: &chat.Usage{Input: 1, Output: 2}}, []chat.Item{chat.FunctionCallItem("call_1", "search", `{}`)}))
		assert.Contains(t, rec.Body.String(), "message_delta")
	})

	t.Run("unsupportedEvent", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		err := stream.Event(chat.Event{Type: "nope"})
		require.Error(t, err)
	})

	t.Run("nonTextMessage", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		item := chat.MessageItem(chat.RoleAssistant, chat.ImagePart(chat.InlineMedia("image/png", []byte{1})))
		err := stream.Event(chat.Event{Type: chat.EventItem, Item: item})
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("unknownToolDelta", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		err := stream.Event(ToolCallDelta("missing", `{}`))
		require.Error(t, err)
	})

	t.Run("unsupportedOutputItem", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		err := stream.Event(chat.OutputItem(chat.Item{Type: chat.ItemReasoning}))
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("failAfterToolStart", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(chat.ToolStart("call_1", "search")))
		require.NoError(t, stream.Fail(errors.New("boom")))
		assert.Contains(t, rec.Body.String(), "error")
	})

	t.Run("completeToolItemsOnly", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(TextDelta("x")))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted, StopReason: chat.StopToolCall, Usage: &chat.Usage{Input: 1, Output: 2}}, []chat.Item{chat.FunctionCallItem("call_1", "search", `{}`)}))
		assert.Contains(t, rec.Body.String(), "tool_use")
	})

	t.Run("noFlusher", func(t *testing.T) {
		w := &plainResponseWriter{header: make(http.Header)}
		stream := adapter.Stream(w, parsedRequest{}, &meta, limits)
		err := stream.Event(TextDelta("x"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "Flusher")
	})

	t.Run("emptyToolCallArgs", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(chat.Tool("call_1", "search", "")))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted, StopReason: chat.StopToolCall}, nil))
	})

	t.Run("stopMaxTokens", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(TextDelta("x")))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusIncomplete, StopReason: chat.StopLength}, nil))
		assert.Contains(t, rec.Body.String(), "max_tokens")
	})
}

type plainResponseWriter struct {
	header http.Header
	code   int
	body   []byte
}

func (w *plainResponseWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}
func (w *plainResponseWriter) Write(p []byte) (int, error) {
	w.body = append(w.body, p...)
	return len(p), nil
}
func (w *plainResponseWriter) WriteHeader(statusCode int) { w.code = statusCode }

func TestAdapterResponseMore(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{Response: chat.Response{ID: "msg_1", Created: 100, Target: "claude-3"}}
	req := chat.Request{Target: "claude-3"}

	t.Run("maxTokens", func(t *testing.T) {
		value, err := adapter.Response(req, execution.Result{
			Items:   []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("cut"))},
			Outcome: chat.Outcome{Status: chat.StatusIncomplete, StopReason: chat.StopLength},
		}, meta)
		require.NoError(t, err)
		assert.Equal(t, "max_tokens", value.(messageResponse).StopReason)
	})

	t.Run("mixedOutput", func(t *testing.T) {
		value, err := adapter.Response(req, execution.Result{
			Items: []chat.Item{
				chat.MessageItem(chat.RoleAssistant, chat.TextPart("hi")),
				chat.FunctionCallItem("tu_1", "search", `{}`),
			},
			Outcome: chat.Outcome{Status: chat.StatusCompleted, StopReason: chat.StopToolCall},
		}, meta)
		require.NoError(t, err)
		require.Len(t, value.(messageResponse).Content, 2)
	})

	t.Run("unsupportedItem", func(t *testing.T) {
		_, err := adapter.Response(req, execution.Result{
			Items:   []chat.Item{{Type: chat.ItemMedia}},
			Outcome: chat.Outcome{Status: chat.StatusCompleted},
		}, meta)
		requireAPIError(t, err, "unsupported", "output")
	})
}

func TestValidateMore(t *testing.T) {
	adapter := Adapter{}
	err := adapter.ValidateEvent(chat.MediaItem(chat.ImagePart(chat.InlineMedia("image/png", []byte{1}))))
	requireAPIError(t, err, "unsupported", "output")
	err = adapter.ValidateEvent(chat.Event{Type: chat.EventItem, Item: chat.Item{Type: chat.ItemReasoning}})
	requireAPIError(t, err, "unsupported", "output")
	err = adapter.ValidateEvent(chat.Event{Type: chat.EventItem, Item: chat.Item{Type: chat.ItemMedia}})
	requireAPIError(t, err, "unsupported", "output")
	err = adapter.ValidateEvent(chat.Event{Type: chat.EventItem, Item: chat.MessageItem(chat.RoleAssistant, chat.ImagePart(chat.InlineMedia("image/png", []byte{1})))})
	requireAPIError(t, err, "unsupported", "output")
}

func TestSSEWrite(t *testing.T) {
	rec := httptest.NewRecorder()
	sw := &sseWriter{w: rec, limits: chat.DefaultLimits()}
	require.NoError(t, sw.start())
	require.NoError(t, sw.write("evt", map[string]any{"ok": true}))
	require.NoError(t, sw.done())
	assert.True(t, sw.started)
}

func TestAdapterStreamFail(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{Response: chat.Response{ID: "msg_1", Created: 100, Target: "claude-3"}}
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

	t.Run("permissionError", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(TextDelta("x")))
		require.NoError(t, stream.Fail(&chat.Error{Status: 403, Type: "permission_error", Code: "forbidden", Message: "denied"}))
		assert.Contains(t, rec.Body.String(), "permission_error")
	})
}

func TestStreamCoverage(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{Response: chat.Response{ID: "msg_1", Created: 100, Target: "claude-3"}}
	limits := chat.DefaultLimits()

	t.Run("startedAndCompleteOnly", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		assert.False(t, stream.Started())
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, []chat.Item{chat.FunctionCallItem("call_1", "search", `{}`)}))
		assert.True(t, stream.Started())
		assert.Contains(t, rec.Body.String(), "message_stop")
	})

	t.Run("oneShotToolCall", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(chat.Tool("call_1", "search", `{"q":"x"}`)))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted, StopReason: chat.StopToolCall}, nil))
		assert.Contains(t, rec.Body.String(), "tool_use")
	})

	t.Run("openTextTwice", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, parsedRequest{}, &meta, limits)
		require.NoError(t, stream.Event(TextDelta("a")))
		require.NoError(t, stream.Event(TextDelta("b")))
		require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, nil))
	})
}

func TestParseMediaMore(t *testing.T) {
	cases := map[string]struct {
		body    string
		wantErr bool
		code    string
		param   string
		check   func(t *testing.T, parsed parsedRequest)
	}{
		"imageBase64": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"YQ=="}}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, []byte("a"), parsed.Request.Input[0].Content[0].Media.Data)
			},
		},
		"documentMime": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"YQ=="}}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "application/pdf", parsed.Request.Input[0].Content[0].Media.MIMEType)
			},
		},
		"badDocumentBase64": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","data":"!!!"}}]}]}`,
			wantErr: true, code: "invalid_request", param: "messages.content.source.data",
		},
		"badImageSource": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"file","url":"x"}}]}]}`,
			wantErr: true, code: "unsupported", param: "messages.content.source.type",
		},
		"contentPartsArray": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.Len(t, parsed.Request.Input[0].Content, 2)
			},
		},
		"toolResultText": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"done"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.ItemFunctionCallOutput, parsed.Request.Input[0].Type)
			},
		},
		"toolResultImageParts": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":[{"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}]}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.PartImage, parsed.Request.Input[0].Output[0].Type)
			},
		},
		"toolResultUnsupported": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":[{"type":"document","source":{"type":"url","url":"https://example.com/a.pdf"}}]}]}]}`,
			wantErr: true, code: "unsupported", param: "tool_result.content",
		},
		"documentBadSource": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"file"}}]}]}`,
			wantErr: true, code: "unsupported", param: "messages.content.source.type",
		},
		"documentURLExtra": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"url","url":"https://example.com/a.pdf","extra":1}}]}]}`,
			wantErr: true, code: "unsupported", param: "extra",
		},
		"imageURLExtra": {
			body:    `{"model":"claude-3","max_tokens":64,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.com/a.png","extra":1}}]}]}`,
			wantErr: true, code: "unsupported", param: "extra",
		},
		"emptyToolCallArgs": {
			body: `{"model":"claude-3","max_tokens":64,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"search","input":{}}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, chat.ItemFunctionCall, parsed.Request.Input[0].Type)
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
