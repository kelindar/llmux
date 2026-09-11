// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
package llmux

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"time"

	"github.com/kelindar/llmux/audio"
	"github.com/kelindar/llmux/chat"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	internalexecution "github.com/kelindar/llmux/internal/execution"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/openai/openai-go/v3"
	openaioption "github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatText(t *testing.T) {
	var calls atomic.Int32
	agent := chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		calls.Add(1)
		require.Equal(t, "agent/basic", req.Target)
		require.Len(t, req.Input, 1)
		return chat.Outcome{Usage: &chat.Usage{Input: 3, Output: 2, Total: 5}}, emit(chat.Text("hello"))
	})
	recorder := postJSON(t, testHandler(agent, chat.Info{}), "/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	body := decodeResponse(t, recorder)
	assert.Equal(t, "chat.completion", body["object"])
	choices := body["choices"].([]any)
	message := choices[0].(map[string]any)["message"].(map[string]any)
	assert.Equal(t, "assistant", message["role"])
	assert.Equal(t, "hello", message["content"])
	assert.Equal(t, "stop", choices[0].(map[string]any)["finish_reason"])
	assert.Equal(t, float64(5), body["usage"].(map[string]any)["total_tokens"])
	assert.Equal(t, int32(1), calls.Load())
}

func TestResponsesText(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		if err := emit(chat.Text("hello")); err != nil {
			return chat.Outcome{}, err
		}
		return chat.Outcome{}, nil
	})
	recorder := postJSON(t, testHandler(agent, chat.Info{}), "/responses", `{"model":"agent/basic","input":"hi"}`, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	body := decodeResponse(t, recorder)
	assert.Equal(t, "response", body["object"])
	assert.Equal(t, "completed", body["status"])
	output := body["output"].([]any)
	assert.Equal(t, "message", output[0].(map[string]any)["type"])
	content := output[0].(map[string]any)["content"].([]any)
	assert.Equal(t, "hello", content[0].(map[string]any)["text"])
}

func TestAnthropicText(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("hello"))
	})
	recorder := postJSON(t, testHandler(agent, chat.Info{}), "/messages", `{"model":"agent/basic","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}`, map[string]string{"anthropic-version": "2023-06-01"})
	require.Equal(t, http.StatusOK, recorder.Code)
	body := decodeResponse(t, recorder)
	assert.Equal(t, "message", body["type"])
	assert.Equal(t, "assistant", body["role"])
	content := body["content"].([]any)
	assert.Equal(t, "text", content[0].(map[string]any)["type"])
	assert.Equal(t, "hello", content[0].(map[string]any)["text"])
	assert.Equal(t, "end_turn", body["stop_reason"])
}

func TestStreamText(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		if err := emit(chat.TextDelta("hel")); err != nil {
			return chat.Outcome{}, err
		}
		return chat.Outcome{}, emit(chat.TextDelta("lo"))
	})
	cases := []struct {
		name    string
		path    string
		body    string
		headers map[string]string
		want    string
	}{
		{name: "chat", path: "/chat/completions", body: `{"model":"agent/basic","stream":true,"messages":[{"role":"user","content":"hi"}]}`, want: "hello"},
		{name: "responses", path: "/responses", body: `{"model":"agent/basic","stream":true,"input":"hi"}`, want: "hello"},
		{name: "anthropic", path: "/messages", body: `{"model":"agent/basic","stream":true,"max_tokens":32,"messages":[{"role":"user","content":"hi"}]}`, headers: map[string]string{"anthropic-version": "2023-06-01"}, want: "hello"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			recorder := postJSON(t, testHandler(agent, chat.Info{}), test.path, test.body, test.headers)
			require.Equal(t, http.StatusOK, recorder.Code)
			require.Contains(t, recorder.Header().Get("Content-Type"), "text/event-stream")
			records := readSSE(t, recorder.Result().Body)
			require.NotEmpty(t, records)
			if test.name == "anthropic" {
				var terminal map[string]any
				require.NoError(t, json.Unmarshal([]byte(records[len(records)-1].Data), &terminal))
				assert.Equal(t, "message_stop", terminal["type"])
			} else {
				assert.Equal(t, "[DONE]", records[len(records)-1].Data)
			}
			var builder strings.Builder
			for _, record := range records {
				if record.Data == "[DONE]" {
					continue
				}
				var value map[string]any
				require.NoError(t, json.Unmarshal([]byte(record.Data), &value))
				switch test.name {
				case "chat":
					choices, ok := value["choices"].([]any)
					if !ok || len(choices) == 0 {
						continue
					}
					delta := choices[0].(map[string]any)["delta"].(map[string]any)
					if text, ok := delta["content"].(string); ok {
						builder.WriteString(text)
					}
				case "responses":
					if value["type"] == "response.output_text.delta" {
						builder.WriteString(value["delta"].(string))
					}
				case "anthropic":
					if value["type"] == "content_block_delta" {
						delta := value["delta"].(map[string]any)
						if text, ok := delta["text"].(string); ok {
							builder.WriteString(text)
						}
					}
				}
			}
			assert.Equal(t, test.want, builder.String())
		})
	}
}

func TestToolStream(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		if err := emit(chat.ToolStart("call_1", "lookup")); err != nil {
			return chat.Outcome{}, err
		}
		if err := emit(chat.ToolDelta("call_1", `{"q":`)); err != nil {
			return chat.Outcome{}, err
		}
		if err := emit(chat.ToolDelta("call_1", `"go"}`)); err != nil {
			return chat.Outcome{}, err
		}
		return chat.Outcome{StopReason: chat.StopToolCall}, emit(chat.ToolDone("call_1"))
	})
	caps := chat.Info{Tools: true, ClientTools: true}
	body := `{"model":"agent/basic","stream":true,"messages":[{"role":"user","content":"find"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]}`
	recorder := postJSON(t, testHandler(agent, caps), "/chat/completions", body, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	records := readSSE(t, recorder.Result().Body)
	var arguments strings.Builder
	for _, record := range records {
		if record.Data == "[DONE]" {
			continue
		}
		var value map[string]any
		require.NoError(t, json.Unmarshal([]byte(record.Data), &value))
		choices, ok := value["choices"].([]any)
		if !ok || len(choices) == 0 {
			continue
		}
		delta := choices[0].(map[string]any)["delta"].(map[string]any)
		calls, ok := delta["tool_calls"].([]any)
		if !ok {
			continue
		}
		function := calls[0].(map[string]any)["function"].(map[string]any)
		if piece, ok := function["arguments"].(string); ok {
			arguments.WriteString(piece)
		}
	}
	assert.Equal(t, `{"q":"go"}`, arguments.String())
	var finishReason string
	for _, record := range records {
		if record.Data == "[DONE]" {
			continue
		}
		var value map[string]any
		require.NoError(t, json.Unmarshal([]byte(record.Data), &value))
		choices, ok := value["choices"].([]any)
		if !ok || len(choices) == 0 {
			continue
		}
		if reason, ok := choices[0].(map[string]any)["finish_reason"].(string); ok && reason != "" {
			finishReason = reason
		}
	}
	assert.Equal(t, "tool_calls", finishReason)
}

func TestToolCallAdapters(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{StopReason: chat.StopToolCall}, emit(chat.Tool("call_1", "lookup", `{"q":"go"}`))
	})
	cases := []struct {
		name    string
		path    string
		body    string
		headers map[string]string
	}{
		{
			name: "chat",
			path: "/chat/completions",
			body: `{"model":"agent/basic","messages":[{"role":"user","content":"find"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]}`,
		},
		{
			name: "responses",
			path: "/responses",
			body: `{"model":"agent/basic","input":"find","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}`,
		},
		{
			name:    "anthropic",
			path:    "/messages",
			body:    `{"model":"agent/basic","max_tokens":32,"messages":[{"role":"user","content":"find"}],"tools":[{"name":"lookup","input_schema":{"type":"object"}}]}`,
			headers: map[string]string{"anthropic-version": "2023-06-01"},
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			recorder := postJSON(t, testHandler(agent, chat.Info{Tools: true, ClientTools: true}), test.path, test.body, test.headers)
			require.Equal(t, http.StatusOK, recorder.Code)
			body := decodeResponse(t, recorder)
			switch test.name {
			case "chat":
				message := body["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
				call := message["tool_calls"].([]any)[0].(map[string]any)
				assert.Equal(t, "call_1", call["id"])
			case "responses":
				call := body["output"].([]any)[0].(map[string]any)
				assert.Equal(t, "function_call", call["type"])
				assert.Equal(t, "call_1", call["call_id"])
			case "anthropic":
				call := body["content"].([]any)[0].(map[string]any)
				assert.Equal(t, "tool_use", call["type"])
				assert.Equal(t, "call_1", call["id"])
			}
		})
	}
}

func TestImageOutput(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.MediaItem(chat.ImagePart(chat.InlineMedia("image/png", []byte{1, 2, 3}))))
	})
	recorder := postJSON(t, testHandler(agent, chat.Info{OutputModalities: chat.ModalityText | chat.ModalityImage, ImageGeneration: true}), "/responses", `{"model":"agent/image","tools":[{"type":"image_generation"}],"input":"draw"}`, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	body := decodeResponse(t, recorder)
	output := body["output"].([]any)[0].(map[string]any)
	assert.Equal(t, "image_generation_call", output["type"])
	assert.Equal(t, "AQID", output["result"])
}

func TestAudioOutput(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		media := chat.InlineMedia("audio/wav", []byte{1, 2})
		media.Format = "wav"
		return chat.Outcome{}, emit(chat.MediaItem(chat.AudioPart(media)))
	})
	recorder := postJSON(t, testHandler(agent, chat.Info{OutputModalities: chat.ModalityText | chat.ModalityAudio}), "/chat/completions", `{"model":"agent/audio","modalities":["audio"],"audio":{"voice":"alloy","format":"wav"},"messages":[{"role":"user","content":"speak"}]}`, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	body := decodeResponse(t, recorder)
	message := body["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	audio := message["audio"].(map[string]any)
	assert.Equal(t, "AQI=", audio["data"])
}

func TestMediaInput(t *testing.T) {
	var got chat.Item
	agent := chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		got = req.Input[0]
		return chat.Outcome{}, emit(chat.Text("seen"))
	})
	data := "AQID"
	body := `{"model":"agent/vision","messages":[{"role":"user","content":[{"type":"text","text":"what?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + data + `"}}]}]}`
	recorder := postJSON(t, testHandler(agent, chat.Info{InputModalities: chat.ModalityText | chat.ModalityImage}), "/chat/completions", body, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	require.Len(t, got.Content, 2)
	assert.Equal(t, chat.PartImage, got.Content[1].Type)
	require.NotNil(t, got.Content[1].Media)
	assert.Equal(t, []byte{1, 2, 3}, got.Content[1].Media.Data)
}

func TestFileDataURL(t *testing.T) {
	var got chat.Part
	agent := chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		got = req.Input[0].Content[0]
		return chat.Outcome{}, emit(chat.Text("seen"))
	})
	body := `{"model":"agent/file","messages":[{"role":"user","content":[{"type":"file","file":{"filename":"sample.pdf","file_data":"data:application/pdf;base64,AQID"}}]}]}`
	recorder := postJSON(t, testHandler(agent, chat.Info{InputModalities: chat.ModalityText | chat.ModalityFile}), "/chat/completions", body, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, chat.PartFile, got.Type)
	require.NotNil(t, got.Media)
	assert.Equal(t, "application/pdf", got.Media.MIMEType)
	assert.Equal(t, []byte{1, 2, 3}, got.Media.Data)
	assert.Equal(t, "sample.pdf", got.Media.Filename)
}

func TestParallelRequests(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text(req.Input[0].Content[0].Text))
	})
	server := httptest.NewServer(http.StripPrefix("/v1", testHandler(agent, chat.Info{})))
	defer server.Close()
	type responseResult struct {
		value string
		err   error
	}
	results := make(chan responseResult, 4)
	var group sync.WaitGroup
	for _, value := range []string{"one", "two", "three", "four"} {
		group.Go(func() {
			response, err := http.Post(server.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"agent/basic","messages":[{"role":"user","content":"`+value+`"}]}`))
			if err != nil {
				results <- responseResult{err: err}
				return
			}
			defer response.Body.Close()
			data, err := io.ReadAll(response.Body)
			if err != nil {
				results <- responseResult{err: err}
				return
			}
			var body map[string]any
			if err := json.Unmarshal(data, &body); err != nil {
				results <- responseResult{err: err}
				return
			}
			message := body["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
			results <- responseResult{value: message["content"].(string)}
		})
	}
	group.Wait()
	close(results)
	counts := make(map[string]int, 4)
	for result := range results {
		require.NoError(t, result.err)
		counts[result.value]++
	}
	assert.Equal(t, map[string]int{"one": 1, "two": 1, "three": 1, "four": 1}, counts)
}

func TestCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	agent := chat.AgentFunc(func(ctx context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		if err := emit(chat.TextDelta("first")); err != nil {
			return chat.Outcome{}, err
		}
		cancel()
		select {
		case <-ctx.Done():
			return chat.Outcome{}, ctx.Err()
		case <-time.After(time.Second):
			return chat.Outcome{}, errors.New("context was not cancelled")
		}
	})
	_, err := internalexecution.Run(ctx, &chat.Request{Target: "agent/basic"}, agent, chat.DefaultLimits(), nil)
	assert.ErrorIs(t, err, context.Canceled)
}

func TestOutputLimit(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("too long"))
	})
	recorder := postJSON(t, testHandler(agent, chat.Info{}, WithLimits(chat.Limits{MaxOutputBytes: 3})), "/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
	assert.Equal(t, http.StatusRequestEntityTooLarge, recorder.Code)
	assert.Equal(t, "output_too_large", responseError(t, recorder)["code"])
}

type sseRecord struct {
	Event string
	Data  string
}

// fixedCatalog is a test Catalog with independent List and Load behavior.
type fixedCatalog struct {
	agent   chat.Agent
	info    chat.Info
	entries map[string]chat.Info
	listFn  func(context.Context) (map[string]chat.Info, error)
	loadFn  func(context.Context, string) (chat.Agent, chat.Info, error)
}

func (c *fixedCatalog) List(ctx context.Context) (map[string]chat.Info, error) {
	if c.listFn != nil {
		return c.listFn(ctx)
	}
	if c.entries != nil {
		return c.entries, nil
	}
	return map[string]chat.Info{}, nil
}

func (c *fixedCatalog) Load(ctx context.Context, target string) (chat.Agent, chat.Info, error) {
	if c.loadFn != nil {
		return c.loadFn(ctx, target)
	}
	return c.agent, c.info, nil
}

func testHandler(agent chat.Agent, caps chat.Info, options ...Option) *Handler {
	return New(&fixedCatalog{agent: agent, info: caps}, options...)
}

func postJSON(t *testing.T, handler http.Handler, path string, body string, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, req)
	return recorder
}

func decodeResponse(t *testing.T, recorder *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var value map[string]any
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &value), recorder.Body.String())
	return value
}

func readSSE(t *testing.T, body io.Reader) []sseRecord {
	t.Helper()
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 1024), 8<<20)
	var records []sseRecord
	var current sseRecord
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			current.Event = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			current.Data = strings.TrimPrefix(line, "data: ")
		case line == "":
			if current.Event != "" || current.Data != "" {
				records = append(records, current)
				current = sseRecord{}
			}
		}
	}
	require.NoError(t, scanner.Err())
	return records
}

func postRaw(t *testing.T, handler http.Handler, path string, body []byte, contentType string, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, req)
	return recorder
}

func responseError(t *testing.T, recorder *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	value := decodeResponse(t, recorder)
	errorValue, ok := value["error"].(map[string]any)
	_assert := assert.True(t, ok, "response did not contain an error: %v", value)
	if !_assert {
		return nil
	}
	return errorValue
}

func TestOpenAI(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("hello"))
	})
	server := httptest.NewServer(http.StripPrefix("/v1", testHandler(agent, chat.Info{})))
	defer server.Close()
	client := openai.NewClient(
		openaioption.WithAPIKey("test"),
		openaioption.WithBaseURL(server.URL+"/v1/"),
		openaioption.WithMaxRetries(0),
	)
	ctx := context.Background()

	chat, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:    "agent/basic",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	})
	require.NoError(t, err)
	require.NotNil(t, chat)
	require.Len(t, chat.Choices, 1)
	assert.Equal(t, "hello", chat.Choices[0].Message.Content)

	response, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model: "agent/basic",
		Input: responses.ResponseNewParamsInputUnion{OfString: param.NewOpt("hi")},
	})
	require.NoError(t, err)
	require.NotNil(t, response)
	assert.Equal(t, "response", string(response.Object))
	assert.Equal(t, "hello", response.OutputText())
}

func TestOpenAIStream(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		if err := emit(chat.TextDelta("hel")); err != nil {
			return chat.Outcome{}, err
		}
		return chat.Outcome{}, emit(chat.TextDelta("lo"))
	})
	server := httptest.NewServer(http.StripPrefix("/v1", testHandler(agent, chat.Info{})))
	defer server.Close()
	client := openai.NewClient(openaioption.WithAPIKey("test"), openaioption.WithBaseURL(server.URL+"/v1/"), openaioption.WithMaxRetries(0))
	ctx := context.Background()

	chatStream := client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
		Model:    "agent/basic",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	})
	var chatText string
	for chatStream.Next() {
		for _, choice := range chatStream.Current().Choices {
			chatText += choice.Delta.Content
		}
	}
	require.NoError(t, chatStream.Err())
	assert.Equal(t, "hello", chatText)

	responseStream := client.Responses.NewStreaming(ctx, responses.ResponseNewParams{
		Model: "agent/basic",
		Input: responses.ResponseNewParamsInputUnion{OfString: param.NewOpt("hi")},
	})
	var responseText string
	for responseStream.Next() {
		event := responseStream.Current()
		if event.Type == "response.output_text.delta" {
			responseText += event.Delta
		}
	}
	require.NoError(t, responseStream.Err())
	assert.Equal(t, "hello", responseText)
}

func TestAnthropic(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("hello"))
	})
	server := httptest.NewServer(http.StripPrefix("/v1", testHandler(agent, chat.Info{})))
	defer server.Close()
	client := anthropic.NewClient(
		anthropicoption.WithoutEnvironmentDefaults(),
		anthropicoption.WithAPIKey("test"),
		anthropicoption.WithBaseURL(server.URL+"/"),
		anthropicoption.WithMaxRetries(0),
	)
	message, err := client.Messages.New(context.Background(), anthropic.MessageNewParams{
		Model:     "agent/basic",
		MaxTokens: 32,
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock("hi"))},
	})
	require.NoError(t, err)
	require.NotNil(t, message)
	assert.Equal(t, "message", string(message.Type))
	require.Len(t, message.Content, 1)
	assert.Equal(t, "hello", message.Content[0].Text)
}

func TestAnthropicStream(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		if err := emit(chat.TextDelta("hel")); err != nil {
			return chat.Outcome{}, err
		}
		return chat.Outcome{}, emit(chat.TextDelta("lo"))
	})
	server := httptest.NewServer(http.StripPrefix("/v1", testHandler(agent, chat.Info{})))
	defer server.Close()
	client := anthropic.NewClient(
		anthropicoption.WithoutEnvironmentDefaults(),
		anthropicoption.WithAPIKey("test"),
		anthropicoption.WithBaseURL(server.URL+"/"),
		anthropicoption.WithMaxRetries(0),
	)
	stream := client.Messages.NewStreaming(context.Background(), anthropic.MessageNewParams{
		Model:     "agent/basic",
		MaxTokens: 32,
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock("hi"))},
	})
	var text string
	for stream.Next() {
		event := stream.Current()
		if event.Type == "content_block_delta" {
			text += event.Delta.Text
		}
	}
	require.NoError(t, stream.Err())
	assert.Equal(t, "hello", text)
}

func TestOpenAIAudio(t *testing.T) {
	server := httptest.NewServer(http.StripPrefix("/v1", New(
		&fixedCatalog{agent: chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("unused"))
		})},
		WithTranscriber(audio.TranscriberFunc(func(_ context.Context, req audio.TranscriptionRequest) (audio.Transcription, error) {
			assert.Equal(t, "gpt-4o-transcribe", req.Model)
			assert.Equal(t, []byte("audio"), req.Data)
			return audio.Transcription{Text: "hello"}, nil
		})),
		WithSpeaker(audio.SpeakerFunc(func(_ context.Context, req audio.SpeechRequest) (audio.Speech, error) {
			assert.Equal(t, "say hello", req.Input)
			return audio.Speech{Data: []byte("audio"), MIMEType: "audio/mpeg"}, nil
		})),
	)))
	defer server.Close()
	client := openai.NewClient(openaioption.WithAPIKey("test"), openaioption.WithBaseURL(server.URL+"/v1/"), openaioption.WithMaxRetries(0))

	speech, err := client.Audio.Speech.New(context.Background(), openai.AudioSpeechNewParams{
		Input:          "say hello",
		Model:          openai.SpeechModelTTS1,
		Voice:          openai.AudioSpeechNewParamsVoiceUnion{OfAudioSpeechNewsVoiceString2: openai.String("alloy")},
		ResponseFormat: openai.AudioSpeechNewParamsResponseFormatMP3,
	})
	require.NoError(t, err)
	speechData, err := io.ReadAll(speech.Body)
	require.NoError(t, err)
	require.NoError(t, speech.Body.Close())
	assert.Equal(t, []byte("audio"), speechData)

	transcription, err := client.Audio.Transcriptions.New(context.Background(), openai.AudioTranscriptionNewParams{
		File:           bytes.NewReader([]byte("audio")),
		Model:          openai.AudioModelGPT4oTranscribe,
		ResponseFormat: openai.AudioResponseFormatJSON,
	})
	require.NoError(t, err)
	assert.Equal(t, "hello", transcription.AsTranscription().Text)
}

func TestModelsList(t *testing.T) {
	handler := New(&fixedCatalog{
		entries: map[string]chat.Info{
			"agent/other": {},
			"agent/basic": {Created: 42, OwnedBy: "test"},
		},
		agent: chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, nil
		}),
	})
	req := httptest.NewRequest(http.MethodGet, "/models", nil)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, req)
	require.Equal(t, http.StatusOK, recorder.Code)
	body := decodeResponse(t, recorder)
	assert.Equal(t, "list", body["object"])
	data := body["data"].([]any)
	require.Len(t, data, 2)
	for i, entry := range data {
		obj, ok := entry.(map[string]any)
		require.True(t, ok, "entry %d", i)
		assert.Equal(t, "model", obj["object"], "entry %d", i)
	}
	assert.Equal(t, "agent/basic", data[0].(map[string]any)["id"])
	assert.Equal(t, float64(42), data[0].(map[string]any)["created"])
	assert.Equal(t, "test", data[0].(map[string]any)["owned_by"])
	assert.Equal(t, "agent/other", data[1].(map[string]any)["id"])
}

func TestModelsEmpty(t *testing.T) {
	handler := New(nil)
	req := httptest.NewRequest(http.MethodGet, "/models", nil)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, req)
	require.Equal(t, http.StatusOK, recorder.Code)
	body := decodeResponse(t, recorder)
	assert.Equal(t, "list", body["object"])
	assert.Empty(t, body["data"])
}

func TestMethodNotAllowed(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{})
	paths := []string{
		"/chat/completions", "/responses", "/messages",
		"/audio/transcriptions", "/audio/speech", "/models",
	}
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPut, path, nil)
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, req)
			require.Equal(t, http.StatusMethodNotAllowed, recorder.Code)
			if path == "/messages" {
				body := decodeResponse(t, recorder)
				assert.Equal(t, "error", body["type"])
			} else {
				assert.Equal(t, "method_not_allowed", responseError(t, recorder)["code"])
			}
		})
	}
}

func TestRouteNotFound(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{})
	req := httptest.NewRequest(http.MethodGet, "/unknown", nil)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, req)
	require.Equal(t, http.StatusNotFound, recorder.Code)
}

func TestResolveErrors(t *testing.T) {
	t.Run("missing resolver", func(t *testing.T) {
		handler := New(nil)
		recorder := postJSON(t, handler, "/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
		require.Equal(t, http.StatusInternalServerError, recorder.Code)
	})

	t.Run("load error", func(t *testing.T) {
		var logged error
		handler := New(&fixedCatalog{loadFn: func(context.Context, string) (chat.Agent, chat.Info, error) {
			return nil, chat.Info{}, errors.New("lookup failed")
		}}, WithErrorLog(func(_ context.Context, err error) { logged = err }))
		recorder := postJSON(t, handler, "/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
		require.Equal(t, http.StatusInternalServerError, recorder.Code)
		require.Error(t, logged)
	})

	t.Run("nil agent", func(t *testing.T) {
		handler := New(&fixedCatalog{loadFn: func(context.Context, string) (chat.Agent, chat.Info, error) {
			return nil, chat.Info{}, nil
		}})
		recorder := postJSON(t, handler, "/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
		require.Equal(t, http.StatusInternalServerError, recorder.Code)
	})
}

func TestAssetResolution(t *testing.T) {
	var resolved string
	agent := chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		resolved = req.Input[0].Content[0].Media.MIMEType
		return chat.Outcome{}, emit(chat.Text("ok"))
	})
	handler := testHandler(agent, chat.Info{InputModalities: chat.ModalityText | chat.ModalityFile},
		WithAssetResolver(chat.AssetResolver(func(_ context.Context, media chat.Media, _ int64) (chat.Media, error) {
			assert.Equal(t, "file-abc", media.Ref)
			return chat.InlineMedia("application/pdf", []byte{9}), nil
		})),
	)
	body := `{"model":"agent/file","messages":[{"role":"user","content":[{"type":"file","file":{"file_id":"file-abc"}}]}]}`
	recorder := postJSON(t, handler, "/chat/completions", body, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, "application/pdf", resolved)
}

func TestAssetResolutionFailure(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{InputModalities: chat.ModalityText | chat.ModalityFile},
		WithAssetResolver(chat.AssetResolver(func(context.Context, chat.Media, int64) (chat.Media, error) {
			return chat.Media{}, errors.New("missing asset")
		})),
	)
	body := `{"model":"agent/file","messages":[{"role":"user","content":[{"type":"file","file":{"file_id":"missing"}}]}]}`
	recorder := postJSON(t, handler, "/chat/completions", body, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "asset_resolution_failed", responseError(t, recorder)["code"])
}

func TestCapabilityRejects(t *testing.T) {
	agent := chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, errors.New("unreachable")
	})
	baseCaps := chat.Info{
		InputModalities:    chat.ModalityText,
		OutputModalities:   chat.ModalityText,
		GenerationControls: chat.ControlMaxOutputTokens | chat.ControlTemperature | chat.ControlTopP | chat.ControlStop,
	}
	cases := []struct {
		name string
		caps chat.Info
		body string
		code string
	}{
		{name: "temperature", caps: chat.Info{InputModalities: chat.ModalityText, OutputModalities: chat.ModalityText, GenerationControls: chat.ControlMaxOutputTokens | chat.ControlTopP | chat.ControlStop}, body: `{"model":"agent/basic","temperature":0.5,"messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
		{name: "top_p", caps: chat.Info{InputModalities: chat.ModalityText, OutputModalities: chat.ModalityText, GenerationControls: chat.ControlMaxOutputTokens | chat.ControlTemperature | chat.ControlStop}, body: `{"model":"agent/basic","top_p":0.5,"messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
		{name: "stop", caps: chat.Info{InputModalities: chat.ModalityText, OutputModalities: chat.ModalityText, GenerationControls: chat.ControlMaxOutputTokens | chat.ControlTemperature | chat.ControlTopP}, body: `{"model":"agent/basic","stop":["END"],"messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
		{name: "parallel tools", caps: baseCaps, body: `{"model":"agent/basic","parallel_tool_calls":false,"messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
		{name: "image input", caps: baseCaps, body: `{"model":"agent/basic","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]}`, code: "unsupported"},
		{name: "tools", caps: baseCaps, body: `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]}`, code: "unsupported"},
		{name: "structured output", caps: baseCaps, body: `{"model":"agent/basic","response_format":{"type":"json_schema","json_schema":{"name":"out","schema":{"type":"object"}}},"messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
		{name: "max tokens", caps: chat.Info{InputModalities: chat.ModalityText, OutputModalities: chat.ModalityText, GenerationControls: chat.ControlTemperature | chat.ControlTopP | chat.ControlStop}, body: `{"model":"agent/basic","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
		{name: "reasoning", caps: baseCaps, body: `{"model":"agent/basic","reasoning_effort":"medium","messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
		{name: "extension", caps: baseCaps, body: `{"model":"agent/basic","x-vendor-trace":{"enabled":true},"messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			recorder := postJSON(t, testHandler(agent, test.caps), "/chat/completions", test.body, nil)
			require.Equal(t, http.StatusBadRequest, recorder.Code)
			assert.Equal(t, test.code, responseError(t, recorder)["code"])
		})
	}
}

func TestValidateRequestDirect(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{})
	parsed := parsedRequest{Request: chat.Request{Target: "agent/basic", Output: chat.OutputSpec{Modalities: chat.ModalityText | chat.ModalityAudio}}}
	err := handler.validateParsed(&parsed, chat.Info{InputModalities: chat.ModalityText, OutputModalities: chat.ModalityText})
	require.Error(t, err)
	assert.Equal(t, "unsupported", err.(*chat.Error).Code)
}

func TestOutputEvent(t *testing.T) {
	cases := []struct {
		name  string
		event chat.Event
		caps  chat.Info
	}{
		{name: "file output", event: chat.MediaItem(chat.FilePart(chat.InlineMedia("application/pdf", []byte{1}))), caps: chat.Info{OutputModalities: chat.ModalityText}},
		{name: "audio output", event: chat.MediaItem(chat.AudioPart(chat.InlineMedia("audio/wav", []byte{1}))), caps: chat.Info{OutputModalities: chat.ModalityText}},
		{name: "tool call", event: chat.Tool("call_1", "lookup", `{}`), caps: chat.Info{OutputModalities: chat.ModalityText, Tools: true}},
		{name: "reasoning", event: chat.Reasoning("brief"), caps: chat.Info{OutputModalities: chat.ModalityText}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			err := internalprotocol.ValidateOutputEvent(test.event, test.caps)
			require.Error(t, err)
		})
	}
}

func TestDecodeHelpers(t *testing.T) {
	object := map[string]jsontext.Value{
		"flag":  jsontext.Value(`true`),
		"count": jsontext.Value(`3`),
		"tags":  jsontext.Value(`["a","b"]`),
		"ratio": jsontext.Value(`1.5`),
	}
	flag, err := decodeBool(object, "flag")
	require.NoError(t, err)
	require.NotNil(t, flag)
	assert.True(t, *flag)

	count, err := decodeInt(object, "count")
	require.NoError(t, err)
	require.NotNil(t, count)
	assert.Equal(t, 3, *count)

	tags, err := decodeStringSlice(object, "tags")
	require.NoError(t, err)
	assert.Equal(t, []string{"a", "b"}, tags)

	ratio, err := decodeFloat(object, "ratio")
	require.NoError(t, err)
	require.NotNil(t, ratio)

	bad := map[string]jsontext.Value{"flag": jsontext.Value(`"nope"`), "count": jsontext.Value(`"nope"`), "tags": jsontext.Value(`1`)}
	_, err = decodeBool(bad, "flag")
	require.Error(t, err)
	_, err = decodeInt(bad, "count")
	require.Error(t, err)
	_, err = decodeStringSlice(bad, "tags")
	require.Error(t, err)
}

func TestContinuationMissingStore(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{Continuation: true}, WithStore(&testContinuationStore{items: make(map[string][]chat.Item)}))
	recorder := postJSON(t, handler, "/responses", `{"model":"agent/basic","previous_response_id":"resp_missing","input":"hi"}`, nil)
	require.Equal(t, http.StatusNotFound, recorder.Code)
}

func TestStoreWithoutContinuation(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{Continuation: true})
	recorder := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"hi"}`, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "unsupported", responseError(t, recorder)["code"])
}

func TestAgentRunError(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, errors.New("agent failed")
	}), chat.Info{})
	recorder := postJSON(t, handler, "/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusInternalServerError, recorder.Code)
}

func TestValidateRequest(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{})
	maxTokens := 16
	temperature := 0.5
	parallel := false
	store := true
	cases := []struct {
		name   string
		parsed parsedRequest
		caps   chat.Info
	}{
		{name: "empty model", parsed: parsedRequest{Request: chat.Request{Target: " ", Output: chat.OutputSpec{Modalities: chat.ModalityText}}}, caps: chat.Info{InputModalities: chat.ModalityText, OutputModalities: chat.ModalityText}},
		{name: "audio input", parsed: func() parsedRequest {
			media := chat.InlineMedia("audio/wav", []byte{1})
			media.Format = "wav"
			return parsedRequest{Request: chat.Request{Target: "agent", Input: []chat.Item{chat.MessageItem(chat.RoleUser, chat.AudioPart(media))}, Output: chat.OutputSpec{Modalities: chat.ModalityText}}}
		}(), caps: chat.Info{InputModalities: chat.ModalityText, OutputModalities: chat.ModalityText}},
		{name: "max output tokens", parsed: parsedRequest{Request: chat.Request{Target: "agent", Controls: chat.Controls{MaxOutputTokens: &maxTokens}, Output: chat.OutputSpec{Modalities: chat.ModalityText}}}, caps: chat.Info{InputModalities: chat.ModalityText, OutputModalities: chat.ModalityText, GenerationControls: chat.ControlTemperature | chat.ControlTopP | chat.ControlStop}},
		{name: "parallel tools", parsed: parsedRequest{Request: chat.Request{Target: "agent", Controls: chat.Controls{ParallelToolCall: &parallel}, Output: chat.OutputSpec{Modalities: chat.ModalityText}}}, caps: chat.Info{InputModalities: chat.ModalityText, OutputModalities: chat.ModalityText, GenerationControls: chat.ControlMaxOutputTokens | chat.ControlTemperature | chat.ControlTopP | chat.ControlStop}},
		{name: "store without continuation", parsed: parsedRequest{Request: chat.Request{Target: "agent", Controls: chat.Controls{}, Output: chat.OutputSpec{Modalities: chat.ModalityText}}, Store: &store}, caps: chat.Info{InputModalities: chat.ModalityText, OutputModalities: chat.ModalityText, Continuation: false}},
		{name: "image generation", parsed: parsedRequest{Request: chat.Request{Target: "agent", Controls: chat.Controls{ImageGeneration: true}, Output: chat.OutputSpec{Modalities: chat.ModalityText}}}, caps: chat.Info{InputModalities: chat.ModalityText, OutputModalities: chat.ModalityText}},
		{name: "reasoning summary", parsed: parsedRequest{Request: chat.Request{Target: "agent", Controls: chat.Controls{Reasoning: &chat.ReasoningControl{Summary: true}}, Output: chat.OutputSpec{Modalities: chat.ModalityText}}}, caps: chat.Info{InputModalities: chat.ModalityText, OutputModalities: chat.ModalityText, GenerationControls: chat.ControlMaxOutputTokens | chat.ControlTemperature | chat.ControlTopP | chat.ControlStop | chat.ControlReasoning}},
		{name: "temperature chat.Unsupported", parsed: parsedRequest{Request: chat.Request{Target: "agent", Controls: chat.Controls{Temperature: &temperature}, Output: chat.OutputSpec{Modalities: chat.ModalityText}}}, caps: chat.Info{InputModalities: chat.ModalityText, OutputModalities: chat.ModalityText, GenerationControls: chat.ControlMaxOutputTokens | chat.ControlTopP | chat.ControlStop}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			err := handler.validateParsed(&test.parsed, test.caps)
			require.Error(t, err)
		})
	}
}

func TestResolveItemOutput(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{InputModalities: chat.ModalityText | chat.ModalityFile},
		WithAssetResolver(chat.AssetResolver(func(_ context.Context, _ chat.Media, _ int64) (chat.Media, error) {
			return chat.InlineMedia("application/pdf", []byte{1}), nil
		})),
	)
	item := chat.FunctionCallOutputItem("call_1", chat.FilePart(chat.AssetMedia("application/pdf", "file-abc")))
	count := 0
	require.NoError(t, handler.resolveItemMedia(context.Background(), &item, &count))
	require.NotNil(t, item.Output[0].Media)
	assert.Equal(t, []byte{1}, item.Output[0].Media.Data)
}

func TestReasoningSummaryGate(t *testing.T) {
	assert.True(t, internalprotocol.RequiresReasoningSummary(chat.Reasoning("brief")))
	assert.True(t, internalprotocol.RequiresReasoningSummary(chat.OutputItem(chat.Item{Type: chat.ItemReasoning, Summary: []chat.Part{chat.SummaryPart("brief")}})))
}
