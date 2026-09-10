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

	"github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
	internalexecution "github.com/kelindar/llmux/internal/execution"
	"github.com/openai/openai-go/v3"
	openaioption "github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatText(t *testing.T) {
	var calls atomic.Int32
	agent := AgentFunc(func(_ context.Context, req *Request, emit Emit) (Outcome, error) {
		calls.Add(1)
		require.Equal(t, "agent/basic", req.Target)
		require.Len(t, req.Input, 1)
		return Outcome{Usage: &Usage{InputTokens: 3, OutputTokens: 2, TotalTokens: 5}}, EmitText(emit, "hello")
	})
	recorder := postJSON(t, testHandler(agent, Capabilities{}), "/v1/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
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
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		if err := EmitText(emit, "hello"); err != nil {
			return Outcome{}, err
		}
		return Outcome{}, nil
	})
	recorder := postJSON(t, testHandler(agent, Capabilities{}), "/v1/responses", `{"model":"agent/basic","input":"hi"}`, nil)
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
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitText(emit, "hello")
	})
	recorder := postJSON(t, testHandler(agent, Capabilities{}), "/v1/messages", `{"model":"agent/basic","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}`, map[string]string{"anthropic-version": "2023-06-01"})
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
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		if err := EmitTextDelta(emit, "hel"); err != nil {
			return Outcome{}, err
		}
		return Outcome{}, EmitTextDelta(emit, "lo")
	})
	cases := []struct {
		name    string
		path    string
		body    string
		headers map[string]string
		want    string
	}{
		{name: "chat", path: "/v1/chat/completions", body: `{"model":"agent/basic","stream":true,"messages":[{"role":"user","content":"hi"}]}`, want: "hello"},
		{name: "responses", path: "/v1/responses", body: `{"model":"agent/basic","stream":true,"input":"hi"}`, want: "hello"},
		{name: "anthropic", path: "/v1/messages", body: `{"model":"agent/basic","stream":true,"max_tokens":32,"messages":[{"role":"user","content":"hi"}]}`, headers: map[string]string{"anthropic-version": "2023-06-01"}, want: "hello"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			recorder := postJSON(t, testHandler(agent, Capabilities{}), test.path, test.body, test.headers)
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
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		if err := EmitToolCallStart(emit, "call_1", "lookup"); err != nil {
			return Outcome{}, err
		}
		if err := EmitToolCallDelta(emit, "call_1", `{"q":`); err != nil {
			return Outcome{}, err
		}
		return Outcome{StopReason: StopToolCall}, EmitToolCallDelta(emit, "call_1", `"go"}`)
	})
	caps := Capabilities{Tools: true, ClientTools: true}
	body := `{"model":"agent/basic","stream":true,"messages":[{"role":"user","content":"find"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]}`
	recorder := postJSON(t, testHandler(agent, caps), "/v1/chat/completions", body, nil)
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
	last := records[len(records)-2]
	var terminal map[string]any
	require.NoError(t, json.Unmarshal([]byte(last.Data), &terminal))
	assert.Equal(t, "tool_calls", terminal["choices"].([]any)[0].(map[string]any)["finish_reason"])
}

func TestToolCallAdapters(t *testing.T) {
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{StopReason: StopToolCall}, EmitToolCall(emit, "call_1", "lookup", `{"q":"go"}`)
	})
	cases := []struct {
		name    string
		path    string
		body    string
		headers map[string]string
	}{
		{
			name: "chat",
			path: "/v1/chat/completions",
			body: `{"model":"agent/basic","messages":[{"role":"user","content":"find"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]}`,
		},
		{
			name: "responses",
			path: "/v1/responses",
			body: `{"model":"agent/basic","input":"find","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}`,
		},
		{
			name:    "anthropic",
			path:    "/v1/messages",
			body:    `{"model":"agent/basic","max_tokens":32,"messages":[{"role":"user","content":"find"}],"tools":[{"name":"lookup","input_schema":{"type":"object"}}]}`,
			headers: map[string]string{"anthropic-version": "2023-06-01"},
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			recorder := postJSON(t, testHandler(agent, Capabilities{Tools: true, ClientTools: true}), test.path, test.body, test.headers)
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
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitMedia(emit, ImagePart(InlineMedia("image/png", []byte{1, 2, 3})))
	})
	recorder := postJSON(t, testHandler(agent, Capabilities{OutputModalities: ModalityText | ModalityImage, ImageGeneration: true}), "/v1/responses", `{"model":"agent/image","tools":[{"type":"image_generation"}],"input":"draw"}`, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	body := decodeResponse(t, recorder)
	output := body["output"].([]any)[0].(map[string]any)
	assert.Equal(t, "image_generation_call", output["type"])
	assert.Equal(t, "AQID", output["result"])
}

func TestAudioOutput(t *testing.T) {
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		media := InlineMedia("audio/wav", []byte{1, 2})
		media.Format = "wav"
		return Outcome{}, EmitMedia(emit, AudioPart(media))
	})
	recorder := postJSON(t, testHandler(agent, Capabilities{OutputModalities: ModalityText | ModalityAudio}), "/v1/chat/completions", `{"model":"agent/audio","modalities":["audio"],"audio":{"voice":"alloy","format":"wav"},"messages":[{"role":"user","content":"speak"}]}`, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	body := decodeResponse(t, recorder)
	message := body["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	audio := message["audio"].(map[string]any)
	assert.Equal(t, "AQI=", audio["data"])
}

func TestMediaInput(t *testing.T) {
	var got Item
	agent := AgentFunc(func(_ context.Context, req *Request, emit Emit) (Outcome, error) {
		got = req.Input[0]
		return Outcome{}, EmitText(emit, "seen")
	})
	data := "AQID"
	body := `{"model":"agent/vision","messages":[{"role":"user","content":[{"type":"text","text":"what?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + data + `"}}]}]}`
	recorder := postJSON(t, testHandler(agent, Capabilities{InputModalities: ModalityText | ModalityImage}), "/v1/chat/completions", body, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	require.Len(t, got.Content, 2)
	assert.Equal(t, PartImage, got.Content[1].Type)
	require.NotNil(t, got.Content[1].Media)
	assert.Equal(t, []byte{1, 2, 3}, got.Content[1].Media.Data)
}

func TestFileDataURL(t *testing.T) {
	var got Part
	agent := AgentFunc(func(_ context.Context, req *Request, emit Emit) (Outcome, error) {
		got = req.Input[0].Content[0]
		return Outcome{}, EmitText(emit, "seen")
	})
	body := `{"model":"agent/file","messages":[{"role":"user","content":[{"type":"file","file":{"filename":"sample.pdf","file_data":"data:application/pdf;base64,AQID"}}]}]}`
	recorder := postJSON(t, testHandler(agent, Capabilities{InputModalities: ModalityText | ModalityFile}), "/v1/chat/completions", body, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, PartFile, got.Type)
	require.NotNil(t, got.Media)
	assert.Equal(t, "application/pdf", got.Media.MIMEType)
	assert.Equal(t, []byte{1, 2, 3}, got.Media.Data)
	assert.Equal(t, "sample.pdf", got.Media.Filename)
}

func TestParallelRequests(t *testing.T) {
	agent := AgentFunc(func(_ context.Context, req *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitText(emit, req.Input[0].Content[0].Text)
	})
	server := httptest.NewServer(testHandler(agent, Capabilities{}))
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
	agent := AgentFunc(func(ctx context.Context, _ *Request, emit Emit) (Outcome, error) {
		if err := EmitTextDelta(emit, "first"); err != nil {
			return Outcome{}, err
		}
		cancel()
		select {
		case <-ctx.Done():
			return Outcome{}, ctx.Err()
		case <-time.After(time.Second):
			return Outcome{}, errors.New("context was not cancelled")
		}
	})
	_, err := internalexecution.Run(ctx, &Request{Target: "agent/basic"}, agent, DefaultLimits(), nil)
	assert.ErrorIs(t, err, context.Canceled)
}

func TestOutputLimit(t *testing.T) {
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitText(emit, "too long")
	})
	recorder := postJSON(t, testHandler(agent, Capabilities{}, WithLimits(Limits{MaxOutputBytes: 3})), "/v1/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
	assert.Equal(t, http.StatusRequestEntityTooLarge, recorder.Code)
	assert.Equal(t, "output_too_large", responseError(t, recorder)["code"])
}

type sseRecord struct {
	Event string
	Data  string
}

func testHandler(agent Agent, caps Capabilities, options ...Option) *Handler {
	resolver := ResolverFunc(func(_ context.Context, target string) (Agent, Capabilities, error) {
		return agent, caps, nil
	})
	return New(resolver, options...)
}

func postJSON(t *testing.T, handler http.Handler, path string, body string, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		request.Header.Set(key, value)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
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
	request := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
	if contentType != "" {
		request.Header.Set("Content-Type", contentType)
	}
	for key, value := range headers {
		request.Header.Set(key, value)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
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
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitText(emit, "hello")
	})
	server := httptest.NewServer(testHandler(agent, Capabilities{}))
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
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		if err := EmitTextDelta(emit, "hel"); err != nil {
			return Outcome{}, err
		}
		return Outcome{}, EmitTextDelta(emit, "lo")
	})
	server := httptest.NewServer(testHandler(agent, Capabilities{}))
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
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitText(emit, "hello")
	})
	server := httptest.NewServer(testHandler(agent, Capabilities{}))
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
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		if err := EmitTextDelta(emit, "hel"); err != nil {
			return Outcome{}, err
		}
		return Outcome{}, EmitTextDelta(emit, "lo")
	})
	server := httptest.NewServer(testHandler(agent, Capabilities{}))
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
	server := httptest.NewServer(New(
		ResolverFunc(func(context.Context, string) (Agent, Capabilities, error) {
			return AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
				return Outcome{}, EmitText(emit, "unused")
			}), Capabilities{}, nil
		}),
		WithTranscriber(TranscriberFunc(func(_ context.Context, request TranscriptionRequest) (Transcription, error) {
			assert.Equal(t, "gpt-4o-transcribe", request.Model)
			assert.Equal(t, []byte("audio"), request.Data)
			return Transcription{Text: "hello"}, nil
		})),
		WithSpeaker(SpeakerFunc(func(_ context.Context, request SpeechRequest) (Speech, error) {
			assert.Equal(t, "say hello", request.Input)
			return Speech{Data: []byte("audio"), MIMEType: "audio/mpeg"}, nil
		})),
	))
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
	handler := NewHandler(ResolverFunc(func(context.Context, string) (Agent, Capabilities, error) {
		return AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
			return Outcome{}, nil
		}), Capabilities{}, nil
	}), WithModels(
		Model{ID: "agent/basic", OwnedBy: "test"},
		Model{ID: "agent/other"},
	))
	request := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	require.Equal(t, http.StatusOK, recorder.Code)
	body := decodeResponse(t, recorder)
	assert.Equal(t, "list", body["object"])
	data := body["data"].([]any)
	require.Len(t, data, 2)
	assert.Equal(t, "model", data[1].(map[string]any)["object"])
}

func TestMethodNotAllowed(t *testing.T) {
	handler := testHandler(AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, nil
	}), Capabilities{})
	paths := []string{
		"/v1/chat/completions", "/v1/responses", "/v1/messages",
		"/v1/audio/transcriptions", "/v1/audio/speech", "/v1/models",
	}
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPut, path, nil)
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, request)
			require.Equal(t, http.StatusMethodNotAllowed, recorder.Code)
			if path == "/v1/messages" {
				body := decodeResponse(t, recorder)
				assert.Equal(t, "error", body["type"])
			} else {
				assert.Equal(t, "method_not_allowed", responseError(t, recorder)["code"])
			}
		})
	}
}

func TestRouteNotFound(t *testing.T) {
	handler := testHandler(AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, nil
	}), Capabilities{})
	request := httptest.NewRequest(http.MethodGet, "/v1/unknown", nil)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	require.Equal(t, http.StatusNotFound, recorder.Code)
}

func TestResolveErrors(t *testing.T) {
	t.Run("missing resolver", func(t *testing.T) {
		handler := New(nil)
		recorder := postJSON(t, handler, "/v1/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
		require.Equal(t, http.StatusInternalServerError, recorder.Code)
	})

	t.Run("resolver error", func(t *testing.T) {
		var logged error
		handler := New(ResolverFunc(func(context.Context, string) (Agent, Capabilities, error) {
			return nil, Capabilities{}, errors.New("lookup failed")
		}), WithErrorLog(func(_ context.Context, err error) { logged = err }))
		recorder := postJSON(t, handler, "/v1/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
		require.Equal(t, http.StatusInternalServerError, recorder.Code)
		require.Error(t, logged)
	})

	t.Run("nil agent", func(t *testing.T) {
		handler := New(ResolverFunc(func(context.Context, string) (Agent, Capabilities, error) {
			return nil, Capabilities{}, nil
		}))
		recorder := postJSON(t, handler, "/v1/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
		require.Equal(t, http.StatusInternalServerError, recorder.Code)
	})
}

func TestAssetResolution(t *testing.T) {
	var resolved string
	agent := AgentFunc(func(_ context.Context, req *Request, emit Emit) (Outcome, error) {
		resolved = req.Input[0].Content[0].Media.MIMEType
		return Outcome{}, EmitText(emit, "ok")
	})
	handler := testHandler(agent, Capabilities{InputModalities: ModalityText | ModalityFile},
		WithAssetResolver(AssetResolverFunc(func(_ context.Context, media Media, _ int64) (Media, error) {
			assert.Equal(t, "file-abc", media.Ref)
			return InlineMedia("application/pdf", []byte{9}), nil
		})),
	)
	body := `{"model":"agent/file","messages":[{"role":"user","content":[{"type":"file","file":{"file_id":"file-abc"}}]}]}`
	recorder := postJSON(t, handler, "/v1/chat/completions", body, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, "application/pdf", resolved)
}

func TestAssetResolutionFailure(t *testing.T) {
	handler := testHandler(AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, nil
	}), Capabilities{InputModalities: ModalityText | ModalityFile},
		WithAssetResolver(AssetResolverFunc(func(context.Context, Media, int64) (Media, error) {
			return Media{}, errors.New("missing asset")
		})),
	)
	body := `{"model":"agent/file","messages":[{"role":"user","content":[{"type":"file","file":{"file_id":"missing"}}]}]}`
	recorder := postJSON(t, handler, "/v1/chat/completions", body, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "asset_resolution_failed", responseError(t, recorder)["code"])
}

func TestCapabilityRejects(t *testing.T) {
	agent := AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, errors.New("unreachable")
	})
	baseCaps := Capabilities{
		InputModalities:    ModalityText,
		OutputModalities:   ModalityText,
		GenerationControls: ControlMaxOutputTokens | ControlTemperature | ControlTopP | ControlStop,
	}
	cases := []struct {
		name string
		caps Capabilities
		body string
		code string
	}{
		{name: "temperature", caps: Capabilities{InputModalities: ModalityText, OutputModalities: ModalityText, GenerationControls: ControlMaxOutputTokens | ControlTopP | ControlStop}, body: `{"model":"agent/basic","temperature":0.5,"messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
		{name: "top_p", caps: Capabilities{InputModalities: ModalityText, OutputModalities: ModalityText, GenerationControls: ControlMaxOutputTokens | ControlTemperature | ControlStop}, body: `{"model":"agent/basic","top_p":0.5,"messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
		{name: "stop", caps: Capabilities{InputModalities: ModalityText, OutputModalities: ModalityText, GenerationControls: ControlMaxOutputTokens | ControlTemperature | ControlTopP}, body: `{"model":"agent/basic","stop":["END"],"messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
		{name: "parallel tools", caps: baseCaps, body: `{"model":"agent/basic","parallel_tool_calls":false,"messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
		{name: "image input", caps: baseCaps, body: `{"model":"agent/basic","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]}`, code: "unsupported"},
		{name: "tools", caps: baseCaps, body: `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]}`, code: "unsupported"},
		{name: "structured output", caps: baseCaps, body: `{"model":"agent/basic","response_format":{"type":"json_schema","json_schema":{"name":"out","schema":{"type":"object"}}},"messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
		{name: "max tokens", caps: Capabilities{InputModalities: ModalityText, OutputModalities: ModalityText, GenerationControls: ControlTemperature | ControlTopP | ControlStop}, body: `{"model":"agent/basic","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
		{name: "reasoning", caps: baseCaps, body: `{"model":"agent/basic","reasoning_effort":"medium","messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
		{name: "extension", caps: baseCaps, body: `{"model":"agent/basic","x-vendor-trace":{"enabled":true},"messages":[{"role":"user","content":"hi"}]}`, code: "unsupported"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			recorder := postJSON(t, testHandler(agent, test.caps), "/v1/chat/completions", test.body, nil)
			require.Equal(t, http.StatusBadRequest, recorder.Code)
			assert.Equal(t, test.code, responseError(t, recorder)["code"])
		})
	}
}

func TestValidateRequestDirect(t *testing.T) {
	handler := testHandler(AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, nil
	}), Capabilities{})
	req := &Request{Target: "agent/basic", Output: OutputSpec{Modalities: ModalityText | ModalityAudio}}
	err := handler.validateRequest(req, Capabilities{InputModalities: ModalityText, OutputModalities: ModalityText})
	require.Error(t, err)
	assert.Equal(t, "unsupported", err.(*APIError).Code)
}

func TestOutputEvent(t *testing.T) {
	cases := []struct {
		name  string
		event Event
		caps  Capabilities
	}{
		{name: "file output", event: MediaOutput(FilePart(InlineMedia("application/pdf", []byte{1}))), caps: Capabilities{OutputModalities: ModalityText}},
		{name: "audio output", event: MediaOutput(AudioPart(InlineMedia("audio/wav", []byte{1}))), caps: Capabilities{OutputModalities: ModalityText}},
		{name: "tool call", event: ToolCall("call_1", "lookup", `{}`), caps: Capabilities{OutputModalities: ModalityText, Tools: true}},
		{name: "reasoning", event: ReasoningSummary("brief"), caps: Capabilities{OutputModalities: ModalityText}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			err := validateOutputEvent(test.event, test.caps)
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
	handler := testHandler(AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, nil
	}), Capabilities{Continuation: true}, WithContinuationStore(&testContinuationStore{items: make(map[string][]Item)}))
	recorder := postJSON(t, handler, "/v1/responses", `{"model":"agent/basic","previous_response_id":"resp_missing","input":"hi"}`, nil)
	require.Equal(t, http.StatusNotFound, recorder.Code)
}

func TestStoreWithoutContinuation(t *testing.T) {
	handler := testHandler(AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, nil
	}), Capabilities{Continuation: true})
	recorder := postJSON(t, handler, "/v1/responses", `{"model":"agent/basic","store":true,"input":"hi"}`, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "unsupported", responseError(t, recorder)["code"])
}

func TestAgentRunError(t *testing.T) {
	handler := testHandler(AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, errors.New("agent failed")
	}), Capabilities{})
	recorder := postJSON(t, handler, "/v1/chat/completions", `{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusInternalServerError, recorder.Code)
}

func TestValidateRequest(t *testing.T) {
	handler := testHandler(AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, nil
	}), Capabilities{})
	maxTokens := 16
	temperature := 0.5
	parallel := false
	store := true
	cases := []struct {
		name string
		req  Request
		caps Capabilities
	}{
		{name: "empty model", req: Request{Target: " ", Output: OutputSpec{Modalities: ModalityText}}, caps: Capabilities{InputModalities: ModalityText, OutputModalities: ModalityText}},
		{name: "audio input", req: func() Request {
			media := InlineMedia("audio/wav", []byte{1})
			media.Format = "wav"
			return Request{Target: "agent", Input: []Item{MessageItem(RoleUser, AudioPart(media))}, Output: OutputSpec{Modalities: ModalityText}}
		}(), caps: Capabilities{InputModalities: ModalityText, OutputModalities: ModalityText}},
		{name: "max output tokens", req: Request{Target: "agent", Controls: Controls{MaxOutputTokens: &maxTokens}, Output: OutputSpec{Modalities: ModalityText}}, caps: Capabilities{InputModalities: ModalityText, OutputModalities: ModalityText, GenerationControls: ControlTemperature | ControlTopP | ControlStop}},
		{name: "parallel tools", req: Request{Target: "agent", Controls: Controls{ParallelToolCall: &parallel}, Output: OutputSpec{Modalities: ModalityText}}, caps: Capabilities{InputModalities: ModalityText, OutputModalities: ModalityText, GenerationControls: ControlMaxOutputTokens | ControlTemperature | ControlTopP | ControlStop}},
		{name: "store without continuation", req: Request{Target: "agent", Controls: Controls{Store: &store}, Output: OutputSpec{Modalities: ModalityText}}, caps: Capabilities{InputModalities: ModalityText, OutputModalities: ModalityText, Continuation: false}},
		{name: "image generation", req: Request{Target: "agent", Controls: Controls{ImageGeneration: true}, Output: OutputSpec{Modalities: ModalityText}}, caps: Capabilities{InputModalities: ModalityText, OutputModalities: ModalityText}},
		{name: "reasoning summary", req: Request{Target: "agent", Controls: Controls{Reasoning: &Reasoning{Summary: true}}, Output: OutputSpec{Modalities: ModalityText}}, caps: Capabilities{InputModalities: ModalityText, OutputModalities: ModalityText, GenerationControls: ControlMaxOutputTokens | ControlTemperature | ControlTopP | ControlStop | ControlReasoning}},
		{name: "temperature unsupported", req: Request{Target: "agent", Controls: Controls{Temperature: &temperature}, Output: OutputSpec{Modalities: ModalityText}}, caps: Capabilities{InputModalities: ModalityText, OutputModalities: ModalityText, GenerationControls: ControlMaxOutputTokens | ControlTopP | ControlStop}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			err := handler.validateRequest(&test.req, test.caps)
			require.Error(t, err)
		})
	}
}

func TestResolveItemOutput(t *testing.T) {
	handler := testHandler(AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, nil
	}), Capabilities{InputModalities: ModalityText | ModalityFile},
		WithAssetResolver(AssetResolverFunc(func(_ context.Context, _ Media, _ int64) (Media, error) {
			return InlineMedia("application/pdf", []byte{1}), nil
		})),
	)
	item := FunctionCallOutputItem("call_1", FilePart(AssetMedia("application/pdf", "file-abc")))
	count := 0
	require.NoError(t, handler.resolveItemMedia(context.Background(), &item, &count))
	require.NotNil(t, item.Output[0].Media)
	assert.Equal(t, []byte{1}, item.Output[0].Media.Data)
}

func TestReasoningSummaryGate(t *testing.T) {
	assert.True(t, requiresReasoningSummary(ReasoningSummary("brief")))
	assert.True(t, requiresReasoningSummary(OutputItem(Item{Type: ItemReasoning, Summary: []Part{SummaryPart("brief")}})))
}
