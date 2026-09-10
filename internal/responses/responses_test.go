package responses

import (
	"encoding/base64"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/kelindar/llmux/contract"
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
	apiErr, ok := errors.AsType[*contract.APIError](err)
	require.True(t, ok, "expected contract.APIError, got %T", err)
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
		"stringInput": {
			body: `{"model":"gpt-4.1","input":"hello"}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, internalprotocol.Responses, parsed.Kind)
				assert.Equal(t, "gpt-4.1", parsed.Request.Target)
				require.Len(t, parsed.Request.Input, 1)
				assert.Equal(t, contract.RoleUser, parsed.Request.Input[0].Role)
			},
		},
		"messageInput": {
			body: `{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.Len(t, parsed.Request.Input, 1)
				assert.Equal(t, "hi", parsed.Request.Input[0].Content[0].Text)
			},
		},
		"streamWithInstructions": {
			body: `{"model":"gpt-4.1","stream":true,"instructions":"be brief","input":"x"}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.True(t, parsed.Stream)
				assert.Equal(t, "be brief", parsed.Request.Instructions)
			},
		},
		"functionCallItems": {
			body: `{"model":"gpt-4.1","input":[{"type":"function_call","call_id":"c1","name":"search","arguments":"{}"},{"type":"function_call_output","call_id":"c1","output":"done"}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.Len(t, parsed.Request.Input, 2)
				assert.Equal(t, contract.ItemFunctionCall, parsed.Request.Input[0].Type)
				assert.Equal(t, contract.ItemFunctionCallOutput, parsed.Request.Input[1].Type)
			},
		},
		"toolsAndChoice": {
			body: `{"model":"gpt-4.1","input":"x","tools":[{"type":"function","name":"search","parameters":{"type":"object"}}],"tool_choice":"required"}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.Len(t, parsed.Request.Controls.Tools, 1)
				require.NotNil(t, parsed.Request.Controls.ToolChoice)
				assert.Equal(t, "required", parsed.Request.Controls.ToolChoice.Mode)
			},
		},
		"imageGenerationTool": {
			body: `{"model":"gpt-4.1","input":"draw","tools":[{"type":"image_generation"}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.True(t, parsed.Request.Controls.ImageGeneration)
				assert.True(t, parsed.Request.Output.Modalities.Has(contract.ModalityText|contract.ModalityImage))
			},
		},
		"textJsonSchema": {
			body: `{"model":"gpt-4.1","input":"x","text":{"format":{"type":"json_schema","name":"out","schema":{"type":"object"}}}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "json_schema", parsed.Request.Output.Format)
				require.NotNil(t, parsed.Request.Controls.Structured)
				assert.Equal(t, "out", parsed.Request.Controls.Structured.Name)
			},
		},
		"reasoningControls": {
			body: `{"model":"gpt-4.1","input":"think","reasoning":{"effort":"medium","summary":"auto"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.NotNil(t, parsed.Request.Controls.Reasoning)
				assert.Equal(t, "medium", parsed.Request.Controls.Reasoning.Effort)
				assert.True(t, parsed.Request.Controls.Reasoning.Summary)
			},
		},
		"reasoningItem": {
			body: `{"model":"gpt-4.1","input":[{"type":"reasoning","summary":[{"type":"summary_text","text":"brief"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.Len(t, parsed.Request.Input, 1)
				assert.Equal(t, contract.ItemReasoning, parsed.Request.Input[0].Type)
			},
		},
		"unknownField": {
			body:    `{"model":"gpt-4.1","input":"x","bogus":1}`,
			wantErr: true,
			code:    "unsupported",
			param:   "bogus",
		},
		"missingModel": {
			body:    `{"input":"x"}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "model",
		},
		"missingInput": {
			body:    `{"model":"gpt-4.1"}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "input",
		},
		"emptyInputArray": {
			body:    `{"model":"gpt-4.1","input":[]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "input",
		},
		"invalidToolChoice": {
			body:    `{"model":"gpt-4.1","input":"x","tool_choice":"maybe"}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "tool_choice",
		},
		"invalidFunctionArguments": {
			body:    `{"model":"gpt-4.1","input":[{"type":"function_call","call_id":"c1","name":"f","arguments":"not-json"}]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "input.arguments",
		},
		"unsupportedBackground": {
			body:    `{"model":"gpt-4.1","input":"x","background":true}`,
			wantErr: true,
			code:    "unsupported",
			param:   "background",
		},
		"unsupportedConversation": {
			body:    `{"model":"gpt-4.1","input":"x","conversation":"c1"}`,
			wantErr: true,
			code:    "unsupported",
			param:   "conversation",
		},
		"duplicateImageGeneration": {
			body:    `{"model":"gpt-4.1","input":"x","tools":[{"type":"image_generation"},{"type":"image_generation"}]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "tools",
		},
		"invalidMaxOutputTokens": {
			body:    `{"model":"gpt-4.1","input":"x","max_output_tokens":0}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "max_output_tokens",
		},
		"streamOptionsUnknown": {
			body:    `{"model":"gpt-4.1","input":"x","stream_options":{"include_usage":true}}`,
			wantErr: true,
			code:    "unsupported",
			param:   "include_usage",
		},
		"unsupportedInputAudio": {
			body:    `{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_audio","audio":{}}]}]}`,
			wantErr: true,
			code:    "unsupported",
			param:   "input.content",
		},
		"invalidInputRole": {
			body:    `{"model":"gpt-4.1","input":[{"type":"message","role":"bogus","content":"x"}]}`,
			wantErr: true,
			code:    "invalid_request",
			param:   "input.role",
		},
		"fullControls": {
			body: `{"model":"gpt-4.1","input":"x","max_output_tokens":128,"top_p":0.9,"parallel_tool_calls":true,"metadata":{"k":"v"},"stream_options":{}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.NotNil(t, parsed.Request.Controls.MaxOutputTokens)
				require.NotNil(t, parsed.Request.Controls.TopP)
				require.NotNil(t, parsed.Request.Controls.ParallelToolCall)
				assert.Equal(t, "v", parsed.Request.Controls.Metadata["k"])
			},
		},
		"unsupportedInclude": {
			body:    `{"model":"gpt-4.1","input":"x","include":["reasoning.encrypted_content"]}`,
			wantErr: true, code: "unsupported", param: "include",
		},
		"unsupportedTruncation": {
			body:    `{"model":"gpt-4.1","input":"x","truncation":"auto"}`,
			wantErr: true, code: "unsupported", param: "truncation",
		},
		"invalidTopP": {
			body:    `{"model":"gpt-4.1","input":"x","top_p":2}`,
			wantErr: true, code: "invalid_request", param: "top_p",
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
		event   contract.Event
		wantErr bool
	}{
		"textMessage": {
			event: contract.Event{Type: contract.EventMessage, Item: contract.MessageItem(contract.RoleAssistant, contract.TextPart("hi"))},
		},
		"toolCall": {
			event: contract.ToolCall("c1", "fn", `{}`),
		},
		"audioMediaRejected": {
			event:   contract.Event{Type: contract.EventMedia, Part: contract.AudioPart(contract.InlineMedia("audio/wav", []byte{1}))},
			wantErr: true,
		},
		"reasoningSummary": {
			event: contract.Event{Type: contract.EventReasoning, Item: contract.Item{Type: contract.ItemReasoning, Summary: []contract.Part{contract.SummaryPart("brief")}}},
		},
		"unsupportedItem": {
			event:   contract.Event{Type: contract.EventItem, Item: contract.Item{Type: contract.ItemFunctionCallOutput}},
			wantErr: true,
		},
		"messageImagePart": {
			event:   contract.Event{Type: contract.EventMessage, Item: contract.MessageItem(contract.RoleAssistant, contract.ImagePart(contract.InlineMedia("image/png", []byte{1})))},
			wantErr: true,
		},
		"mediaItemBadContent": {
			event:   contract.Event{Type: contract.EventItem, Item: contract.Item{Type: contract.ItemMedia, Content: []contract.Part{}}},
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
	meta := responseMeta{ID: "resp_1", Created: 100, Model: "gpt-4.1"}
	req := contract.Request{Target: "gpt-4.1"}

	t.Run("textMessage", func(t *testing.T) {
		item := contract.MessageItem(contract.RoleAssistant, contract.TextPart("hello"))
		item.ID = "msg_1"
		result := execution.Result{
			Items:   []contract.Item{item},
			Outcome: contract.Outcome{Status: contract.StatusCompleted},
		}
		value, err := adapter.Response(req, result, meta)
		require.NoError(t, err)
		response := value.(map[string]any)
		output := response["output"].([]any)
		require.Len(t, output, 1)
		assert.Equal(t, "message", output[0].(map[string]any)["type"])
	})

	t.Run("functionCall", func(t *testing.T) {
		item := contract.FunctionCallItem("call_1", "search", `{"q":"x"}`)
		item.ID = "fc_1"
		result := execution.Result{Items: []contract.Item{item}, Outcome: contract.Outcome{Status: contract.StatusCompleted}}
		value, err := adapter.Response(req, result, meta)
		require.NoError(t, err)
		output := value.(map[string]any)["output"].([]any)
		assert.Equal(t, "function_call", output[0].(map[string]any)["type"])
	})
}

func TestAdapterStream(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{ID: "resp_1", Created: 100, Model: "gpt-4.1"}
	req := contract.Request{Target: "gpt-4.1"}
	limits := contract.DefaultLimits()

	t.Run("textDelta", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, req, meta, limits)
		assert.False(t, stream.Started())
		item := contract.MessageItem(contract.RoleAssistant, contract.TextPart("hello"))
		item.ID = "msg_1"
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventMessage, Item: item}))
		assert.True(t, stream.Started())
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted}, []contract.Item{item}))
		body := rec.Body.String()
		assert.Contains(t, body, "response.created")
		assert.Contains(t, body, "response.completed")
		assert.Contains(t, body, "[DONE]")
	})

	t.Run("toolCallFlow", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, req, meta, limits)
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventToolCallStart, ItemID: "fc_1", CallID: "call_1", Name: "search"}))
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventToolCallDelta, ItemID: "fc_1", CallID: "call_1", Delta: `{"q"`}))
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventToolCall, ItemID: "fc_1", CallID: "call_1", Name: "search", Arguments: `{"q":"x"}`}))
		item := contract.FunctionCallItem("call_1", "search", `{"q":"x"}`)
		item.ID = "fc_1"
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted, StopReason: contract.StopToolCall}, []contract.Item{item}))
		assert.Contains(t, rec.Body.String(), "function_call")
	})
}

func TestAdapterStreamFail(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{ID: "resp_1", Created: 100, Model: "gpt-4.1"}
	limits := contract.DefaultLimits()

	t.Run("beforeStart", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		err := stream.Fail(contract.Invalid("model", "bad model"))
		requireAPIError(t, err, "invalid_request", "model")
		assert.Equal(t, http.StatusBadRequest, rec.Code)
	})

	t.Run("afterStart", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		item := contract.MessageItem(contract.RoleAssistant, contract.TextPart("x"))
		item.ID = "msg_1"
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventMessage, Item: item}))
		require.NoError(t, stream.Fail(errors.New("boom")))
		assert.Contains(t, rec.Body.String(), "response.failed")
	})
}

func TestParseRequestExtra(t *testing.T) {
	cases := map[string]struct {
		body    string
		wantErr bool
		code    string
		param   string
		check   func(t *testing.T, parsed parsedRequest)
	}{
		"previousResponseID": {
			body: `{"model":"gpt-4.1","input":"x","previous_response_id":"resp_prev"}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.NotNil(t, parsed.Request.Controls.PreviousResponseID)
				assert.Equal(t, "resp_prev", *parsed.Request.Controls.PreviousResponseID)
			},
		},
		"storeFalse": {
			body: `{"model":"gpt-4.1","input":"x","store":false}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.NotNil(t, parsed.Request.Controls.Store)
				assert.False(t, *parsed.Request.Controls.Store)
			},
		},
		"jsonObjectFormat": {
			body: `{"model":"gpt-4.1","input":"x","text":{"format":{"type":"json_object"}}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "json_object", parsed.Request.Output.Format)
			},
		},
		"toolChoiceFunction": {
			body: `{"model":"gpt-4.1","input":"x","tool_choice":{"type":"function","function":{"name":"search"}}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "search", parsed.Request.Controls.ToolChoice.Name)
			},
		},
		"reasoningNone": {
			body: `{"model":"gpt-4.1","input":"x","reasoning":{"effort":"low","summary":"none"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.False(t, parsed.Request.Controls.Reasoning.Summary)
			},
		},
		"inputImageURL": {
			body: `{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"https://example.com/a.png"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, contract.PartImage, parsed.Request.Input[0].Content[0].Type)
			},
		},
		"inputImageFileID": {
			body: `{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_image","file_id":"file_1"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "file_1", parsed.Request.Input[0].Content[0].Media.Ref)
			},
		},
		"inputFileData": {
			body: `{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_file","file_data":"YQ==","filename":"a.txt"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, contract.PartFile, parsed.Request.Input[0].Content[0].Type)
			},
		},
		"unsupportedInputType": {
			body:    `{"model":"gpt-4.1","input":[{"type":"custom","role":"user","content":"x"}]}`,
			wantErr: true, code: "unsupported", param: "input.type",
		},
		"bothImageSources": {
			body:    `{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"https://x","file_id":"f"}]}]}`,
			wantErr: true, code: "invalid_request", param: "input.content",
		},
		"invalidTextFormat": {
			body:    `{"model":"gpt-4.1","input":"x","text":{"format":{"type":"grammar"}}}`,
			wantErr: true, code: "unsupported", param: "text.format",
		},
		"unsupportedReasoningSummary": {
			body:    `{"model":"gpt-4.1","input":"x","reasoning":{"summary":"verbose"}}`,
			wantErr: true, code: "unsupported", param: "reasoning.summary",
		},
		"toolChoiceAuto": {
			body: `{"model":"gpt-4.1","input":"x","tool_choice":"auto"}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "auto", parsed.Request.Controls.ToolChoice.Mode)
			},
		},
		"textFormatSchema": {
			body: `{"model":"gpt-4.1","input":"x","text":{"format":{"type":"json_schema","name":"out","schema":{"type":"object"},"strict":true}}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.NotNil(t, parsed.Request.Controls.Structured)
				assert.True(t, parsed.Request.Controls.Structured.Strict)
			},
		},
		"functionOutputInput": {
			body: `{"model":"gpt-4.1","input":[{"type":"function_call_output","call_id":"c1","output":[{"type":"input_text","text":"done"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, contract.ItemFunctionCallOutput, parsed.Request.Input[0].Type)
			},
		},
		"toolChoiceRequired": {
			body: `{"model":"gpt-4.1","input":"x","tool_choice":"required"}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "required", parsed.Request.Controls.ToolChoice.Mode)
			},
		},
		"toolChoiceNone": {
			body: `{"model":"gpt-4.1","input":"x","tool_choice":"none"}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "none", parsed.Request.Controls.ToolChoice.Mode)
			},
		},
		"toolChoiceByName": {
			body: `{"model":"gpt-4.1","input":"x","tool_choice":{"type":"function","name":"search"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "search", parsed.Request.Controls.ToolChoice.Name)
			},
		},
		"toolChoiceInvalid": {
			body:    `{"model":"gpt-4.1","input":"x","tool_choice":"maybe"}`,
			wantErr: true, code: "invalid_request", param: "tool_choice",
		},
		"toolChoiceBadType": {
			body:    `{"model":"gpt-4.1","input":"x","tool_choice":{"type":"custom","name":"x"}}`,
			wantErr: true, code: "invalid_request", param: "tool_choice",
		},
		"textFormatText": {
			body: `{"model":"gpt-4.1","input":"x","text":{"format":{"type":"text"}}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "text", parsed.Request.Output.Format)
			},
		},
		"textFormatJsonObject": {
			body: `{"model":"gpt-4.1","input":"x","text":{"format":{"type":"json_object"}}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "json_object", parsed.Request.Output.Format)
			},
		},
		"textFormatSchemaDesc": {
			body: `{"model":"gpt-4.1","input":"x","text":{"format":{"type":"json_schema","name":"out","description":"desc","schema":{"type":"object"}}}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				require.NotNil(t, parsed.Request.Controls.Structured)
				assert.Equal(t, "desc", parsed.Request.Controls.Structured.Description)
			},
		},
		"inputFunctionCall": {
			body: `{"model":"gpt-4.1","input":[{"type":"function_call","call_id":"c1","name":"search","arguments":"{\"q\":\"x\"}"}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, contract.ItemFunctionCall, parsed.Request.Input[0].Type)
			},
		},
		"inputReasoning": {
			body: `{"model":"gpt-4.1","input":[{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"brief"}],"encrypted_content":"opaque"}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, contract.ItemReasoning, parsed.Request.Input[0].Type)
				assert.Equal(t, "brief", parsed.Request.Input[0].Summary[0].Text)
			},
		},
		"toolChoiceBadFuncType": {
			body:    `{"model":"gpt-4.1","input":"x","tool_choice":{"type":"custom","function":{"name":"search"}}}`,
			wantErr: true, code: "unsupported", param: "tool_choice.type",
		},
		"inputStringContent": {
			body: `{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":"plain"}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "plain", parsed.Request.Input[0].Content[0].Text)
			},
		},
		"reasoningSummaryAuto": {
			body: `{"model":"gpt-4.1","input":"x","reasoning":{"effort":"low","summary":"auto"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.True(t, parsed.Request.Controls.Reasoning.Summary)
			},
		},
		"outputTextContent": {
			body: `{"model":"gpt-4.1","input":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"out"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "out", parsed.Request.Input[0].Content[0].Text)
			},
		},
		"functionOutputString": {
			body: `{"model":"gpt-4.1","input":[{"type":"function_call_output","call_id":"c1","output":"plain"}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "plain", parsed.Request.Input[0].Output[0].Text)
			},
		},
		"inputFileURL": {
			body: `{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_file","file_url":"https://example.com/a.txt"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, contract.PartFile, parsed.Request.Input[0].Content[0].Type)
			},
		},
		"invalidFunctionCallArgs": {
			body:    `{"model":"gpt-4.1","input":[{"type":"function_call","call_id":"c1","name":"search","arguments":"bad"}]}`,
			wantErr: true, code: "invalid_request", param: "input.arguments",
		},
		"invalidTextSchema": {
			body:    `{"model":"gpt-4.1","input":"x","text":{"format":{"type":"json_schema","name":"out"}}}`,
			wantErr: true, code: "invalid_request", param: "text.format.schema",
		},
		"toolWithDescription": {
			body: `{"model":"gpt-4.1","input":"x","tools":[{"type":"function","name":"search","description":"find","parameters":{"type":"object"},"strict":true}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "find", parsed.Request.Controls.Tools[0].Description)
				assert.True(t, parsed.Request.Controls.Tools[0].Strict != nil && *parsed.Request.Controls.Tools[0].Strict)
			},
		},
		"inputImageDetail": {
			body: `{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"https://example.com/a.png","detail":"high"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "high", parsed.Request.Input[0].Content[0].Detail)
			},
		},
		"reasoningSummaryConcise": {
			body: `{"model":"gpt-4.1","input":"x","reasoning":{"summary":"concise"}}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.True(t, parsed.Request.Controls.Reasoning.Summary)
			},
		},
		"unsupportedContentType": {
			body:    `{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_video","video":{}}]}]}`,
			wantErr: true, code: "unsupported", param: "input.content",
		},
		"inputFileByID": {
			body: `{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_file","file_id":"file_1"}]}]}`,
			check: func(t *testing.T, parsed parsedRequest) {
				assert.Equal(t, "file_1", parsed.Request.Input[0].Content[0].Media.Ref)
			},
		},
		"unsupportedToolType": {
			body:    `{"model":"gpt-4.1","input":"x","tools":[{"type":"custom","name":"x"}]}`,
			wantErr: true, code: "unsupported", param: "tools",
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

func TestAdapterResponseMore(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{ID: "resp_1", Created: 100, Model: "gpt-4.1"}
	req := contract.Request{
		Target:       "gpt-4.1",
		Instructions: "help",
		Controls: contract.Controls{
			MaxOutputTokens: new(128),
			Temperature:     new(0.2),
			ToolChoice:      &contract.ToolChoice{Mode: "auto"},
			Tools:           []contract.Tool{{Name: "search"}},
			ImageGeneration: true,
			Structured:      &contract.StructuredOutput{Name: "out", Schema: jsontext.Value(`{"type":"object"}`), Strict: true},
		},
	}

	t.Run("reasoningOutput", func(t *testing.T) {
		item := contract.Item{Type: contract.ItemReasoning, ID: "rs_1", Summary: []contract.Part{contract.SummaryPart("brief")}}
		result := execution.Result{Items: []contract.Item{item}, Outcome: contract.Outcome{Status: contract.StatusCompleted, Usage: &contract.Usage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3, CachedTokens: 1, ReasoningTokens: 1}}}
		value, err := adapter.Response(req, result, meta)
		require.NoError(t, err)
		data, err := json.Marshal(value)
		require.NoError(t, err)
		assert.Contains(t, string(data), `"reasoning"`)
		assert.Contains(t, string(data), `"summary_text"`)
	})

	t.Run("defaultStatus", func(t *testing.T) {
		result := execution.Result{Outcome: contract.Outcome{}}
		value, err := adapter.Response(contract.Request{}, result, meta)
		require.NoError(t, err)
		assert.Equal(t, string(contract.StatusCompleted), value.(map[string]any)["status"])
	})
}

func TestParseRequestRejects(t *testing.T) {
	cases := map[string]string{
		"invalidMetadata":       `{"model":"gpt-4.1","input":"x","metadata":{"k":1}}`,
		"invalidStream":         `{"model":"gpt-4.1","input":"x","stream":"yes"}`,
		"invalidTemperature":    `{"model":"gpt-4.1","input":"x","temperature":"hot"}`,
		"invalidInputArray":     `{"model":"gpt-4.1","input":[123]}`,
		"missingMessageContent": `{"model":"gpt-4.1","input":[{"type":"message","role":"user"}]}`,
		"invalidToolStrict":     `{"model":"gpt-4.1","input":"x","tools":[{"type":"function","name":"f","strict":"yes"}]}`,
		"imageGenWithOptions":   `{"model":"gpt-4.1","input":"x","tools":[{"type":"image_generation","size":"large"}]}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := ParseRequest(decodeObject(t, body))
			require.Error(t, err)
		})
	}
}

func TestWireDelegates(t *testing.T) {
	media, err := parseMediaURL("https://example.com/a.png", "high")
	require.NoError(t, err)
	assert.Equal(t, "https://example.com/a.png", media.URL)
	fileObject, err := rawObject(jsontext.Value(`{"file_data":"YQ=="}`), "file")
	require.NoError(t, err)
	_, _, err = parseFileMedia(fileObject, "file")
	require.NoError(t, err)
	assert.Equal(t, "x", collectText([]contract.Part{contract.TextPart("x")}))
	parts := inputTextParts([]contract.Part{contract.TextPart("in")})
	require.Len(t, parts, 1)
	url, err := mediaDataURL(contract.InlineMedia("text/plain", []byte("a")))
	require.NoError(t, err)
	assert.Contains(t, url, "data:text/plain")
	assert.NotEmpty(t, newID("resp_"))
}

func TestResponseFunctionOutput(t *testing.T) {
	text, err := responseFunctionOutput([]contract.Part{contract.TextPart("plain")})
	require.NoError(t, err)
	assert.Equal(t, "plain", text)

	media := contract.InlineMedia("image/png", []byte{1, 2})
	values, err := responseFunctionOutput([]contract.Part{contract.ImagePart(media), contract.TextPart("note")})
	require.NoError(t, err)
	list := values.([]any)
	require.Len(t, list, 2)

	file := contract.InlineMedia("text/plain", []byte("data"))
	file.Filename = "a.txt"
	_, err = responseFunctionOutput([]contract.Part{contract.FilePart(file)})
	require.NoError(t, err)
}

func TestResponseItemMore(t *testing.T) {
	item := contract.FunctionCallOutputItem("call_1", contract.TextPart("done"))
	item.ID = "fco_1"
	value, err := responseItem(item)
	require.NoError(t, err)
	assert.Equal(t, "function_call_output", value["type"])

	_, err = responseItem(contract.Item{Type: contract.ItemReasoning, Data: jsontext.Value(`{"x":1}`)})
	requireAPIError(t, err, "unsupported", "output")
}

func TestAdapterValidateMore(t *testing.T) {
	adapter := Adapter{}
	err := adapter.ValidateEvent(contract.Event{Type: contract.EventItem, Item: contract.Item{Type: contract.ItemMedia, Content: []contract.Part{contract.ImagePart(contract.InlineMedia("image/png", []byte{1}))}}})
	require.NoError(t, err)
	err = adapter.ValidateEvent(contract.Event{Type: contract.EventItem, Item: contract.Item{Type: contract.ItemFunctionCallOutput}})
	requireAPIError(t, err, "unsupported", "output")
	err = adapter.ValidateEvent(contract.Event{Type: contract.EventMessage, Item: contract.MessageItem(contract.RoleAssistant, contract.ImagePart(contract.InlineMedia("image/png", []byte{1})))})
	requireAPIError(t, err, "unsupported", "output")
	err = adapter.ValidateEvent(contract.Event{Type: contract.EventMedia, Part: contract.ImagePart(contract.InlineMedia("image/png", []byte{1}))})
	require.NoError(t, err)
}

func TestResponseObject(t *testing.T) {
	req := contract.Request{
		Target: "gpt-4.1",
		Controls: contract.Controls{
			ParallelToolCall:   new(false),
			PreviousResponseID: new("prev"),
			ToolChoice:         &contract.ToolChoice{Mode: "function", Name: "search"},
		},
	}
	value := responseObject(req, contract.Outcome{Status: contract.StatusIncomplete}, responseMeta{Model: "gpt-4.1"}, []any{})
	assert.Equal(t, false, value["parallel_tool_calls"])
	assert.Equal(t, "prev", value["previous_response_id"])
}

//go:fix inline
func boolPtr(v bool) *bool { return new(v) }

//go:fix inline
func strPtr(v string) *string { return new(v) }

func TestAdapterStreamMore(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{ID: "resp_1", Created: 100, Model: "gpt-4.1"}
	limits := contract.DefaultLimits()

	t.Run("textDeltaDone", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventTextDelta, ItemID: "msg_1", Delta: "hel"}))
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventTextDelta, ItemID: "msg_1", Delta: "lo"}))
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventTextDone, ItemID: "msg_1", Text: "hello"}))
		item := contract.MessageItem(contract.RoleAssistant, contract.TextPart("hello"))
		item.ID = "msg_1"
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted}, []contract.Item{item}))
		assert.Contains(t, rec.Body.String(), "response.output_text.delta")
	})

	t.Run("reasoningStream", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		item := contract.Item{Type: contract.ItemReasoning, Summary: []contract.Part{contract.SummaryPart("thinking")}}
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventReasoning, Item: item}))
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted}, []contract.Item{item}))
		assert.Contains(t, rec.Body.String(), "reasoning")
	})

	t.Run("toolCallDonePath", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventToolCallStart, ItemID: "fc_1", CallID: "call_1", Name: "search"}))
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventToolCallDelta, ItemID: "fc_1", CallID: "call_1", Delta: `{"q":"x"}`}))
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventToolCallDone, ItemID: "fc_1", CallID: "call_1", Arguments: `{"q":"x"}`}))
		item := contract.FunctionCallItem("call_1", "search", `{"q":"x"}`)
		item.ID = "fc_1"
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted}, []contract.Item{item}))
	})

	t.Run("textDoneUnknown", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventTextDelta, ItemID: "msg_1", Delta: "x"}))
		err := stream.Event(contract.Event{Type: contract.EventTextDone, ItemID: "missing"})
		require.Error(t, err)
	})

	t.Run("outputItemPaths", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		require.NoError(t, stream.Event(contract.OutputItem(contract.FunctionCallItem("call_1", "search", `{}`))))
		require.NoError(t, stream.Event(contract.OutputItem(contract.MessageItem(contract.RoleAssistant, contract.TextPart("via item")))))
	})

	t.Run("imageStream", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		part := contract.ImagePart(contract.InlineMedia("image/png", []byte{1, 2}))
		require.NoError(t, stream.Event(contract.MediaOutput(part)))
		mediaItem := contract.Item{Type: contract.ItemMedia, ID: "img_1", Content: []contract.Part{part}}
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted}, []contract.Item{mediaItem}))
		assert.Contains(t, rec.Body.String(), "image_generation_call")
	})

	t.Run("emptyMessage", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		item := contract.MessageItem(contract.RoleAssistant)
		item.ID = "msg_empty"
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventMessage, Item: item}))
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted}, []contract.Item{item}))
	})

	t.Run("unsupportedEvent", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		err := stream.Event(contract.Event{Type: "nope"})
		require.Error(t, err)
	})

	t.Run("reasoningOpaqueData", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		item := contract.Item{Type: contract.ItemReasoning, Data: jsontext.Value(`{"x":1}`)}
		err := stream.Event(contract.Event{Type: contract.EventReasoning, Item: item})
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("completeDirect", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		item := contract.MessageItem(contract.RoleAssistant, contract.TextPart("done"))
		item.ID = "msg_1"
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted, Usage: &contract.Usage{InputTokens: 1, OutputTokens: 2}}, []contract.Item{item}))
		assert.Contains(t, rec.Body.String(), "response.completed")
	})

	t.Run("audioRejected", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		audio := contract.InlineMedia("audio/wav", []byte{1})
		err := stream.Event(contract.Event{Type: contract.EventMedia, Part: contract.AudioPart(audio)})
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("textDoneImplicit", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventTextDelta, ItemID: "msg_1", Delta: "hel"}))
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventTextDelta, ItemID: "msg_1", Delta: "lo"}))
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventTextDone, ItemID: "msg_1"}))
		item := contract.MessageItem(contract.RoleAssistant, contract.TextPart("hello"))
		item.ID = "msg_1"
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted}, []contract.Item{item}))
		assert.Contains(t, rec.Body.String(), "response.output_text.done")
	})

	t.Run("reasoningEncrypted", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		item := contract.Item{Type: contract.ItemReasoning, ID: "rs_1", Summary: []contract.Part{contract.SummaryPart("think")}, EncryptedContent: jsontext.Value(`"enc"`)}
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventReasoning, Item: item}))
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted}, []contract.Item{item}))
		assert.Contains(t, rec.Body.String(), "reasoning")
	})

	t.Run("duplicateToolStart", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventToolCallStart, ItemID: "fc_1", CallID: "call_1", Name: "search"}))
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventToolCallStart, ItemID: "fc_1", CallID: "call_1", Name: "search"}))
	})

	t.Run("completeMixedItems", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		msg := contract.MessageItem(contract.RoleAssistant, contract.TextPart("hi"))
		msg.ID = "msg_1"
		call := contract.FunctionCallItem("call_1", "search", `{}`)
		call.ID = "fc_1"
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted, Usage: &contract.Usage{InputTokens: 1, OutputTokens: 2}}, []contract.Item{msg, call}))
		assert.Contains(t, rec.Body.String(), "response.completed")
	})

	t.Run("multiPartMessage", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		item := contract.MessageItem(contract.RoleAssistant, contract.TextPart("a"), contract.TextPart("b"))
		item.ID = "msg_1"
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventMessage, Item: item}))
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted}, []contract.Item{item}))
		assert.Contains(t, rec.Body.String(), "response.content_part.done")
	})

	t.Run("toolCallDoneLookup", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventToolCallStart, ItemID: "fc_1", CallID: "call_1", Name: "search"}))
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventToolCallDelta, ItemID: "fc_1", CallID: "call_1", Delta: `{"q":"x"}`}))
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventToolCallDone, ItemID: "fc_1", CallID: "call_1", Arguments: `{"q":"x"}`}))
	})

	t.Run("duplicateMessageAdd", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		item := contract.MessageItem(contract.RoleAssistant, contract.TextPart("once"))
		item.ID = "msg_1"
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventMessage, Item: item}))
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventMessage, Item: item}))
	})

	t.Run("unsupportedOutputItem", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		err := stream.Event(contract.OutputItem(contract.Item{Type: contract.ItemFunctionCallOutput}))
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("completeBadItem", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		bad := contract.Item{Type: contract.ItemReasoning, Data: jsontext.Value(`{"x":1}`)}
		err := stream.Complete(contract.Outcome{Status: contract.StatusCompleted}, []contract.Item{bad})
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("reasoningNoSummary", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		item := contract.Item{Type: contract.ItemReasoning, ID: "rs_1"}
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventReasoning, Item: item}))
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted}, []contract.Item{item}))
	})

	t.Run("mediaOutputEvent", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		part := contract.ImagePart(contract.InlineMedia("image/png", []byte{9}))
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventMedia, ItemID: "img_1", Part: part}))
	})
}

func TestSSEWrite(t *testing.T) {
	rec := httptest.NewRecorder()
	sw := &sseWriter{w: rec, limits: contract.DefaultLimits()}
	require.NoError(t, sw.start())
	require.NoError(t, sw.write("evt", map[string]any{"ok": true}))
	require.NoError(t, sw.done())
	assert.True(t, sw.started)
}

//go:fix inline
func intPtr(v int) *int { return new(v) }

//go:fix inline
func floatPtr(v float64) *float64 { return new(v) }

func TestResponseItemImage(t *testing.T) {
	item := contract.Item{Type: contract.ItemMedia, ID: "img_1", Status: contract.StatusCompleted, Content: []contract.Part{contract.ImagePart(contract.InlineMedia("image/png", []byte{1, 2, 3}))}}
	value, err := responseItem(item)
	require.NoError(t, err)
	assert.Equal(t, "image_generation_call", value["type"])
	assert.Equal(t, base64.StdEncoding.EncodeToString([]byte{1, 2, 3}), value["result"])
}

func TestCoverageMore(t *testing.T) {
	adapter := Adapter{}
	meta := responseMeta{ID: "resp_1", Created: 100, Model: "gpt-4.1"}
	limits := contract.DefaultLimits()

	t.Run("startedFlag", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		assert.False(t, stream.Started())
		require.NoError(t, stream.Complete(contract.Outcome{}, []contract.Item{contract.MessageItem(contract.RoleAssistant, contract.TextPart("hi"))}))
		assert.True(t, stream.Started())
		assert.Contains(t, rec.Body.String(), `"status":"completed"`)
	})

	t.Run("oneShotToolCall", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventToolCall, ItemID: "fc_1", CallID: "call_1", Name: "search", Arguments: `{"q":"x"}`}))
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted}, nil))
	})

	t.Run("unknownToolDelta", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		err := stream.Event(contract.Event{Type: contract.EventToolCallDelta, ItemID: "missing", CallID: "c", Delta: `{}`})
		require.Error(t, err)
	})

	t.Run("unknownToolDone", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		err := stream.Event(contract.Event{Type: contract.EventToolCallDone, ItemID: "missing", CallID: "c", Arguments: `{}`})
		require.Error(t, err)
	})

	t.Run("messageNonTextStream", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		item := contract.MessageItem(contract.RoleAssistant, contract.ImagePart(contract.InlineMedia("image/png", []byte{1})))
		item.ID = "msg_1"
		err := stream.Event(contract.Event{Type: contract.EventMessage, Item: item})
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("functionOutputImage", func(t *testing.T) {
		out, err := responseFunctionOutput([]contract.Part{contract.ImagePart(contract.InlineMedia("image/png", []byte{1, 2}))})
		require.NoError(t, err)
		require.IsType(t, []any{}, out)
	})

	t.Run("functionOutputBadPart", func(t *testing.T) {
		_, err := responseFunctionOutput([]contract.Part{{Type: contract.PartReasoningSummary}})
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("functionOutputMissingMedia", func(t *testing.T) {
		_, err := responseFunctionOutput([]contract.Part{{Type: contract.PartImage}})
		require.Error(t, err)
	})

	t.Run("functionOutputRemoteMedia", func(t *testing.T) {
		_, err := responseFunctionOutput([]contract.Part{contract.ImagePart(contract.RemoteMedia("image/png", "https://example.com/a.png"))})
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("emptyTextObject", func(t *testing.T) {
		parsed, err := ParseRequest(decodeObject(t, `{"model":"gpt-4.1","input":"x","text":{}}`))
		require.NoError(t, err)
		assert.Equal(t, "text", parsed.Request.Output.Format)
	})

	t.Run("schemaMissing", func(t *testing.T) {
		_, err := ParseRequest(decodeObject(t, `{"model":"gpt-4.1","input":"x","text":{"format":{"type":"json_schema","name":"out"}}}`))
		requireAPIError(t, err, "invalid_request", "text.format.schema")
	})

	t.Run("toolChoiceFunctionTypeMismatch", func(t *testing.T) {
		_, err := ParseRequest(decodeObject(t, `{"model":"gpt-4.1","input":"x","tool_choice":{"type":"custom","function":{"name":"search"}}}`))
		requireAPIError(t, err, "unsupported", "tool_choice.type")
	})

	t.Run("imageOutputNoData", func(t *testing.T) {
		_, err := responseItem(contract.Item{Type: contract.ItemMedia, Content: []contract.Part{contract.ImagePart(contract.RemoteMedia("image/png", "https://example.com/a.png"))}})
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("failAfterStart", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventTextDelta, ItemID: "msg_1", Delta: "x"}))
		require.NoError(t, stream.Fail(errors.New("boom")))
		assert.Contains(t, rec.Body.String(), "response.failed")
	})

	t.Run("noFlusher", func(t *testing.T) {
		w := &plainResponseWriter{header: make(http.Header)}
		stream := adapter.Stream(w, contract.Request{}, meta, limits)
		err := stream.Event(contract.Event{Type: contract.EventTextDelta, ItemID: "msg_1", Delta: "x"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "Flusher")
	})

	t.Run("emptyToolCallArgs", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		require.NoError(t, stream.Event(contract.Event{Type: contract.EventToolCall, ItemID: "fc_1", CallID: "call_1", Name: "search"}))
		require.NoError(t, stream.Complete(contract.Outcome{Status: contract.StatusCompleted}, nil))
	})

	t.Run("imageMediaBad", func(t *testing.T) {
		rec := httptest.NewRecorder()
		stream := adapter.Stream(rec, contract.Request{}, meta, limits)
		err := stream.Event(contract.OutputItem(contract.Item{Type: contract.ItemMedia, Content: []contract.Part{contract.ImagePart(contract.RemoteMedia("image/png", "https://example.com/a.png"))}}))
		requireAPIError(t, err, "unsupported", "output")
	})

	t.Run("responseEmptyStatus", func(t *testing.T) {
		value, err := adapter.Response(contract.Request{Target: "gpt-4.1"}, execution.Result{
			Items:   []contract.Item{contract.MessageItem(contract.RoleAssistant, contract.TextPart("hi"))},
			Outcome: contract.Outcome{},
		}, meta)
		require.NoError(t, err)
		assert.Equal(t, "completed", value.(map[string]any)["status"])
	})

	t.Run("functionOutputItem", func(t *testing.T) {
		item := contract.FunctionCallOutputItem("call_1", contract.TextPart("done"), contract.ImagePart(contract.InlineMedia("image/png", []byte{1})))
		item.ID = "fco_1"
		value, err := responseItem(item)
		require.NoError(t, err)
		assert.Equal(t, "function_call_output", value["type"])
	})

	t.Run("toolsAndImageGen", func(t *testing.T) {
		parsed, err := ParseRequest(decodeObject(t, `{"model":"gpt-4.1","input":"x","tools":[{"type":"image_generation"},{"type":"function","name":"search","parameters":{"type":"object"}}]}`))
		require.NoError(t, err)
		assert.True(t, parsed.Request.Controls.ImageGeneration)
		require.Len(t, parsed.Request.Controls.Tools, 1)
	})

	t.Run("reasoningConcise", func(t *testing.T) {
		parsed, err := ParseRequest(decodeObject(t, `{"model":"gpt-4.1","input":"x","reasoning":{"effort":"high","summary":"concise"}}`))
		require.NoError(t, err)
		assert.True(t, parsed.Request.Controls.Reasoning.Summary)
		assert.Equal(t, "high", parsed.Request.Controls.Reasoning.Effort)
	})

	t.Run("parseRejects", func(t *testing.T) {
		cases := map[string]struct {
			body  string
			code  string
			param string
		}{
			"badRole":            {`{"model":"gpt-4.1","input":[{"type":"message","role":"narrator","content":"x"}]}`, "invalid_request", "input.role"},
			"missingContent":     {`{"model":"gpt-4.1","input":[{"type":"message","role":"user"}]}`, "invalid_request", "input.content"},
			"badArgumentsJSON":   {`{"model":"gpt-4.1","input":[{"type":"function_call","call_id":"c1","name":"search","arguments":"{"}]}`, "invalid_request", "input.arguments"},
			"missingOutput":      {`{"model":"gpt-4.1","input":[{"type":"function_call_output","call_id":"c1"}]}`, "invalid_request", "input.output"},
			"stringOutput":       {`{"model":"gpt-4.1","input":[{"type":"function_call_output","call_id":"c1","output":"done"}]}`, "", ""},
			"badSummaryType":     {`{"model":"gpt-4.1","input":[{"type":"reasoning","summary":[{"type":"other","text":"x"}]}]}`, "unsupported", "input.summary.type"},
			"noImageSource":      {`{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_image"}]}]}`, "invalid_request", "input.content"},
			"badImageDetail":     {`{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"https://example.com/a.png","detail":"ultra"}]}]}`, "unsupported", "input.content.detail"},
			"inputAudio":         {`{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_audio"}]}]}`, "unsupported", "input.content"},
			"badContentType":     {`{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"widget"}]}]}`, "unsupported", "input.content"},
			"outputTextPart":     {`{"model":"gpt-4.1","input":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`, "", ""},
			"imageGenOpts":       {`{"model":"gpt-4.1","input":"x","tools":[{"type":"image_generation","size":"1024"}]}`, "unsupported", "size"},
			"dupImageGen":        {`{"model":"gpt-4.1","input":"x","tools":[{"type":"image_generation"},{"type":"image_generation"}]}`, "invalid_request", "tools"},
			"badToolType":        {`{"model":"gpt-4.1","input":"x","tools":[{"type":"code_interpreter"}]}`, "unsupported", "tools"},
			"textExtraField":     {`{"model":"gpt-4.1","input":"x","text":{"format":{"type":"text"},"verbosity":1}}`, "unsupported", "verbosity"},
			"formatExtra":        {`{"model":"gpt-4.1","input":"x","text":{"format":{"type":"text","foo":1}}}`, "unsupported", "foo"},
			"toolChoiceMissing":  {`{"model":"gpt-4.1","input":"x","tool_choice":{"type":"function","function":{}}}`, "invalid_request", "name"},
			"toolChoiceNoName":   {`{"model":"gpt-4.1","input":"x","tool_choice":{"type":"function"}}`, "invalid_request", "name"},
			"reasoningDetailed":  {`{"model":"gpt-4.1","input":"x","reasoning":{"summary":"detailed"}}`, "", ""},
			"fileURL":            {`{"model":"gpt-4.1","input":[{"type":"message","role":"user","content":[{"type":"input_file","file_url":"https://example.com/a.txt"}]}]}`, "", ""},
			"defaultMessageType": {`{"model":"gpt-4.1","input":[{"role":"user","content":"hi"}]}`, "", ""},
			"messageWithID":      {`{"model":"gpt-4.1","input":[{"type":"message","id":"m1","role":"user","content":"hi"}]}`, "", ""},
			"developerRole":      {`{"model":"gpt-4.1","input":[{"type":"message","role":"developer","content":"sys"}]}`, "", ""},
		}
		for name, tc := range cases {
			t.Run(name, func(t *testing.T) {
				parsed, err := ParseRequest(decodeObject(t, tc.body))
				if tc.code == "" {
					require.NoError(t, err)
					assert.NotEmpty(t, parsed.Request.Target)
					return
				}
				requireAPIError(t, err, tc.code, tc.param)
			})
		}
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
