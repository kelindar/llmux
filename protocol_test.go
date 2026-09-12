// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package llmux

import (
	"bytes"
	"context"
	json "encoding/json/v2"
	"errors"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/kelindar/llmux/audio"
	"github.com/kelindar/llmux/chat"

	internalexecution "github.com/kelindar/llmux/internal/execution"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStreamFailureLifecycle(t *testing.T) {
	cases := []struct {
		name    string
		path    string
		body    string
		headers map[string]string
	}{
		{name: "chat", path: "/chat/completions", body: `{"model":"agent/basic","stream":true,"messages":[{"role":"user","content":"hi"}]}`},
		{name: "responses", path: "/responses", body: `{"model":"agent/basic","stream":true,"input":"hi"}`},
		{name: "anthropic", path: "/messages", body: `{"model":"agent/basic","stream":true,"max_tokens":32,"messages":[{"role":"user","content":"hi"}]}`, headers: map[string]string{"anthropic-version": "2023-06-01"}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
				if err := emit(chat.TextDelta("partial")); err != nil {
					return chat.Outcome{}, err
				}
				return chat.Outcome{}, errors.New("backend secret")
			})
			recorder := postJSON(t, testHandler(agent, chat.Info{}), test.path, test.body, test.headers)
			require.Equal(t, http.StatusOK, recorder.Code)
			assert.NotContains(t, recorder.Body.String(), "backend secret")
			records := readSSE(t, recorder.Result().Body)
			require.NotEmpty(t, records)
			if test.name == "anthropic" {
				var terminal map[string]any
				require.NoError(t, json.Unmarshal([]byte(records[len(records)-1].Data), &terminal))
				assert.Equal(t, "message_stop", terminal["type"])
			} else {
				assert.Equal(t, "[DONE]", records[len(records)-1].Data)
			}
		})
	}
}

func TestStreamValidation(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.MediaItem(chat.ImagePart(chat.InlineMedia("image/png", []byte{1}))))
	})
	recorder := postJSON(t, testHandler(agent, chat.Info{}), "/chat/completions", `{"model":"agent/basic","stream":true,"messages":[{"role":"user","content":"draw"}]}`, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "unsupported", responseError(t, recorder)["code"])
	assert.NotContains(t, recorder.Body.String(), "text/event-stream")
}

func TestImageToolRequired(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.MediaItem(chat.ImagePart(chat.InlineMedia("image/png", []byte{1}))))
	})
	recorder := postJSON(t, testHandler(agent, chat.Info{OutputModalities: chat.ModalityText | chat.ModalityImage, ImageGeneration: true}), "/responses", `{"model":"agent/image","input":"draw"}`, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "unsupported", responseError(t, recorder)["code"])
}

func TestReasoningRequestGate(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.OutputItem(chat.Item{Type: chat.ItemReasoning, Summary: []chat.Part{chat.SummaryPart("brief")}}))
	})
	caps := chat.Info{ReasoningSummary: true}
	denied := postJSON(t, testHandler(agent, caps), "/responses", `{"model":"agent/reasoning","input":"think"}`, nil)
	require.Equal(t, http.StatusBadRequest, denied.Code)
	assert.Equal(t, "unsupported", responseError(t, denied)["code"])

	allowed := postJSON(t, testHandler(agent, caps), "/responses", `{"model":"agent/reasoning","reasoning":{"effort":"medium","summary":"auto"},"input":"think"}`, nil)
	require.Equal(t, http.StatusOK, allowed.Code)
	assert.Equal(t, "reasoning", decodeResponse(t, allowed)["output"].([]any)[0].(map[string]any)["type"])
}

type testContinuationStore struct {
	mu    sync.Mutex
	items map[string][]chat.Item
}

func (s *testContinuationStore) Load(_ context.Context, id string) ([]chat.Item, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	items, ok := s.items[id]
	if !ok {
		return nil, errors.New("missing continuation")
	}
	cloned := make([]chat.Item, len(items))
	for n, item := range items {
		cloned[n] = item.Clone()
	}
	return cloned, nil
}

func (s *testContinuationStore) Accept(context.Context, *chat.TurnRequest) (chat.Acceptance, error) {
	return chat.Acceptance{}, nil
}

func (s *testContinuationStore) put(id string, items []chat.Item) {
	s.mu.Lock()
	defer s.mu.Unlock()
	cloned := make([]chat.Item, len(items))
	for n, item := range items {
		cloned[n] = item.Clone()
	}
	s.items[id] = cloned
}

type savedFinish struct {
	Input    []chat.Item
	Turn     []chat.Item
	Response chat.Response
}

type testLifecycle struct {
	store   *testContinuationStore
	fail    bool
	finals  int
	accepts int
	last    *savedFinish
}

func (l *testLifecycle) Load(ctx context.Context, id string) ([]chat.Item, error) {
	if l.store == nil {
		return nil, errors.New("missing continuation")
	}
	return l.store.Load(ctx, id)
}

func (l *testLifecycle) Accept(_ context.Context, turn *chat.TurnRequest) (chat.Acceptance, error) {
	l.accepts++
	input := make([]chat.Item, len(turn.Request.Input))
	for i, item := range turn.Request.Input {
		input[i] = item.Clone()
	}
	turnItems := make([]chat.Item, len(turn.Turn))
	for i, item := range turn.Turn {
		turnItems[i] = item.Clone()
	}
	return chat.Acceptance{
		Finish: func(_ context.Context, resp *chat.Response, err error) error {
			return l.finish(input, turnItems, resp, err)
		},
	}, nil
}

func (l *testLifecycle) finish(input, turnItems []chat.Item, resp *chat.Response, runErr error) error {
	l.finals++
	l.last = &savedFinish{
		Input:    input,
		Turn:     turnItems,
		Response: resp.Clone(),
	}
	if l.fail {
		return errors.New("finalize failed")
	}
	if !resp.Store || l.store == nil {
		return nil
	}
	items := make([]chat.Item, 0, len(input)+len(resp.Output))
	for _, item := range input {
		items = append(items, item.Clone())
	}
	for _, item := range resp.Output {
		items = append(items, item.Clone())
	}
	l.store.put(resp.ID, items)
	return nil
}

func TestContinuation(t *testing.T) {
	store := &testContinuationStore{items: make(map[string][]chat.Item)}
	life := &testLifecycle{store: store}
	agent := chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text(strconv.Itoa(len(req.Input))))
	})
	handler := testHandler(agent, chat.Info{Continuation: true}, WithStore(life))

	first := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"first"}`, nil)
	require.Equal(t, http.StatusOK, first.Code)
	firstBody := decodeResponse(t, first)
	responseID, ok := firstBody["id"].(string)
	require.True(t, ok)
	require.Equal(t, 1, life.finals)
	require.NotNil(t, life.last)
	assert.Len(t, life.last.Input, 1)
	assert.Len(t, life.last.Response.Output, 1)

	second := postJSON(t, handler, "/responses", `{"model":"agent/basic","previous_response_id":"`+responseID+`","input":"second"}`, nil)
	require.Equal(t, http.StatusOK, second.Code)
	secondBody := decodeResponse(t, second)
	output := secondBody["output"].([]any)
	message := output[0].(map[string]any)
	content := message["content"].([]any)
	assert.Equal(t, "3", content[0].(map[string]any)["text"])
}

func TestNamespacedExtension(t *testing.T) {
	var got string
	agent := chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		got = string(req.Controls.Extensions["x-vendor-trace"])
		return chat.Outcome{}, emit(chat.Text("ok"))
	})
	caps := chat.Info{Extensions: map[string]bool{"x-vendor-trace": true}}
	recorder := postJSON(t, testHandler(agent, caps), "/chat/completions", `{"model":"agent/basic","x-vendor-trace":{"enabled":true},"messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	assert.JSONEq(t, `{"enabled":true}`, got)
}

func TestUnsupportedChatUser(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("unreachable"))
	})
	recorder := postJSON(t, testHandler(agent, chat.Info{}), "/chat/completions", `{"model":"agent/basic","user":"end-user","messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "unsupported", responseError(t, recorder)["code"])
}

func TestUnsupportedChatContinuation(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("unreachable"))
	})
	recorder := postJSON(t, testHandler(agent, chat.Info{}), "/chat/completions", `{"model":"agent/basic","previous_response_id":"resp_1","messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "unsupported", responseError(t, recorder)["code"])
}

func TestJSONV2Duplicates(t *testing.T) {
	bodies := []string{
		`{"model":"agent/basic","model":"agent/other","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]} {}`,
	}
	for _, body := range bodies {
		recorder := postJSON(t, testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("unreachable"))
		}), chat.Info{}), "/chat/completions", body, nil)
		assert.Equal(t, http.StatusBadRequest, recorder.Code)
	}
}

func TestMultipartMalformed(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("unreachable"))
	}), chat.Info{}, WithTranscriber(audio.TranscriberFunc(func(context.Context, audio.TranscriptionRequest) (audio.Transcription, error) {
		return audio.Transcription{}, nil
	})))
	recorder := postRaw(t, handler, "/audio/transcriptions", []byte("--broken\r\nContent-Disposition: form-data; name=\"file\"\r\n\r\nx"), "multipart/form-data; boundary=broken", nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "invalid_multipart", responseError(t, recorder)["code"])
}

func TestSpeechSSE(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("unreachable"))
	}), chat.Info{}, WithSpeaker(audio.SpeakerFunc(func(context.Context, audio.SpeechRequest) (audio.Speech, error) {
		return audio.Speech{Data: []byte{1, 2, 3}, Format: "wav"}, nil
	})))
	recorder := postJSON(t, handler, "/audio/speech", `{"model":"tts","input":"say hi","voice":"alloy","response_format":"wav","stream_format":"sse"}`, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	records := readSSE(t, recorder.Result().Body)
	require.Len(t, records, 2)
	var delta map[string]any
	require.NoError(t, json.Unmarshal([]byte(records[0].Data), &delta))
	assert.Equal(t, "speech.audio.delta", delta["type"])
	assert.Equal(t, "AQID", delta["audio"])
	var done map[string]any
	require.NoError(t, json.Unmarshal([]byte(records[1].Data), &done))
	assert.Equal(t, "speech.audio.done", done["type"])
}

func TestVerboseTranscription(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("unreachable"))
	}), chat.Info{}, WithTranscriber(audio.TranscriberFunc(func(_ context.Context, req audio.TranscriptionRequest) (audio.Transcription, error) {
		assert.Equal(t, "sample.wav", req.Filename)
		return audio.Transcription{Text: "hello", Language: "en", Duration: 1.5, Segments: []audio.TranscriptSegment{{ID: 0, Start: 0, End: 1.5, Text: "hello"}}}, nil
	})))
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	require.NoError(t, writer.WriteField("model", "whisper"))
	require.NoError(t, writer.WriteField("response_format", "verbose_json"))
	file, err := writer.CreateFormFile("file", "sample.wav")
	require.NoError(t, err)
	_, err = file.Write([]byte("audio"))
	require.NoError(t, err)
	require.NoError(t, writer.Close())

	recorder := postRaw(t, handler, "/audio/transcriptions", body.Bytes(), writer.FormDataContentType(), nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	value := decodeResponse(t, recorder)
	assert.Equal(t, "transcribe", value["task"])
	assert.Equal(t, "en", value["language"])
	assert.Equal(t, float64(1.5), value["duration"])
	assert.Len(t, value["segments"].([]any), 1)
}

func TestEmit(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	firstErr := make(chan error, 1)
	secondErr := make(chan error, 1)
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		go func() { firstErr <- emit(chat.TextDelta("first")) }()
		<-entered
		err := emit(chat.TextDelta("second"))
		secondErr <- err
		close(release)
		<-firstErr
		return chat.Outcome{}, err
	})
	onEvent := func(chat.Event) error {
		select {
		case <-entered:
		default:
			close(entered)
		}
		<-release
		return nil
	}
	_, err := internalexecution.Run(context.Background(), &chat.Request{Target: "agent/basic"}, agent, chat.DefaultLimits(), onEvent)
	require.ErrorIs(t, err, chat.ErrConcurrentEmit)
	require.ErrorIs(t, <-secondErr, chat.ErrConcurrentEmit)
}

func TestOutcomeDefaults(t *testing.T) {
	cases := []struct {
		status chat.Status
		stop   chat.StopReason
	}{
		{status: chat.StatusCompleted, stop: chat.StopNormal},
		{status: chat.StatusIncomplete, stop: chat.StopLength},
		{status: chat.StatusFailed, stop: chat.StopError},
		{status: chat.StatusCancelled, stop: chat.StopCancelled},
	}
	for _, test := range cases {
		t.Run(string(test.status), func(t *testing.T) {
			agent := chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
				return chat.Outcome{Status: test.status}, nil
			})
			result, err := internalexecution.Run(context.Background(), &chat.Request{Target: "agent/basic"}, agent, chat.DefaultLimits(), nil)
			require.NoError(t, err)
			assert.Equal(t, test.stop, result.Outcome.StopReason)
		})
	}

	agent := chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{Status: chat.StatusInProgress}, nil
	})
	_, err := internalexecution.Run(context.Background(), &chat.Request{Target: "agent/basic"}, agent, chat.DefaultLimits(), nil)
	assert.Error(t, err)
}

func TestMultipartSizeLimit(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("unreachable"))
	}), chat.Info{}, WithTranscriber(audio.TranscriberFunc(func(context.Context, audio.TranscriptionRequest) (audio.Transcription, error) {
		return audio.Transcription{}, nil
	})), WithLimits(chat.Limits{MaxMultipartBytes: 32}))
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	require.NoError(t, writer.WriteField("model", "whisper"))
	file, err := writer.CreateFormFile("file", "sample.wav")
	require.NoError(t, err)
	_, err = file.Write([]byte(strings.Repeat("a", 128)))
	require.NoError(t, err)
	require.NoError(t, writer.Close())
	recorder := postRaw(t, handler, "/audio/transcriptions", body.Bytes(), writer.FormDataContentType(), nil)
	assert.Equal(t, http.StatusRequestEntityTooLarge, recorder.Code)
}

func TestProtocolFixtures(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("fixture"))
	})
	handler := testHandler(agent, chat.Info{})
	cases := []struct {
		name    string
		file    string
		path    string
		headers map[string]string
	}{
		{name: "chat", file: "fixtures/chat-completion.json", path: "/chat/completions"},
		{name: "responses", file: "fixtures/responses.json", path: "/responses"},
		{name: "anthropic", file: "fixtures/messages.json", path: "/messages", headers: map[string]string{"anthropic-version": "2023-06-01"}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			body, err := os.ReadFile(test.file)
			require.NoError(t, err)
			recorder := postRaw(t, handler, test.path, body, "application/json", test.headers)
			assert.Equal(t, 200, recorder.Code)
			assert.NotEmpty(t, recorder.Body.Bytes())
		})
	}
}

func FuzzProtocolBodies(f *testing.F) {
	for _, seed := range []string{
		`{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"agent/basic","input":"hi"}`,
		`{"model":"agent/basic","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}`,
		`null`,
		`[]`,
		`{"model":1}`,
	} {
		f.Add([]byte(seed))
	}

	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("ok"))
	})
	handler := testHandler(agent, chat.Info{})
	f.Fuzz(func(t *testing.T, body []byte) {
		cases := []struct {
			path    string
			headers map[string]string
		}{
			{path: "/chat/completions"},
			{path: "/responses"},
			{path: "/messages", headers: map[string]string{"anthropic-version": "2023-06-01"}},
		}
		for _, test := range cases {
			req := httptest.NewRequest(http.MethodPost, test.path, bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			for key, value := range test.headers {
				req.Header.Set(key, value)
			}
			recorder := httptest.NewRecorder()
			require.NotPanics(t, func() { handler.ServeHTTP(recorder, req) })
			assert.GreaterOrEqual(t, recorder.Code, http.StatusOK)
			assert.LessOrEqual(t, recorder.Code, 599)
		}
	})
}

func TestAPIErrorMapping(t *testing.T) {
	apiErr := asAPIError(chat.Invalid("model", "required"))
	require.NotNil(t, apiErr)
	assert.Equal(t, "invalid_request_error", apiErr.Type)
	assert.Equal(t, "invalid_request_error", anthropicErrorType(apiErr))

	generic := asAPIError(errors.New("backend"))
	assert.Equal(t, "server_error", generic.Code)
	assert.Equal(t, "api_error", anthropicErrorType(generic))

	nilErr := asAPIError(nil)
	assert.Equal(t, http.StatusInternalServerError, nilErr.Status)
}

func TestSSEWriterLifecycle(t *testing.T) {
	recorder := httptest.NewRecorder()
	stream := &sseWriter{w: recorder, limits: chat.DefaultLimits()}
	require.NoError(t, stream.start())
	assert.True(t, stream.Started())
	require.NoError(t, stream.write("", map[string]string{"type": "ping"}))
	require.NoError(t, stream.done())
	assert.Contains(t, recorder.Body.String(), "[DONE]")
}

func TestFinalizeError(t *testing.T) {
	var logged error
	life := &testLifecycle{fail: true}
	handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("ok"))
	}), chat.Info{Continuation: true},
		WithStore(life),
		WithErrorLog(func(_ context.Context, err error) { logged = err }),
	)
	recorder := postJSON(t, handler, "/responses", `{"model":"agent/basic","store":true,"input":"first"}`, nil)
	require.Equal(t, http.StatusInternalServerError, recorder.Code)
	require.Error(t, logged)
	assert.Equal(t, 1, life.finals)
}

func TestStreamImmediateFailure(t *testing.T) {
	var logged error
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, errors.New("immediate")
	}), chat.Info{}, WithErrorLog(func(_ context.Context, err error) { logged = err }))
	recorder := postJSON(t, handler, "/chat/completions", `{"model":"agent/basic","stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusInternalServerError, recorder.Code)
	assert.NotContains(t, recorder.Header().Get("Content-Type"), "text/event-stream")
	require.Error(t, logged)
}

func TestAdapterResponseError(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		media := chat.InlineMedia("audio/wav", []byte{1})
		media.Format = "wav"
		return chat.Outcome{}, emit(chat.MediaItem(chat.AudioPart(media)))
	})
	recorder := postJSON(t, testHandler(agent, chat.Info{OutputModalities: chat.ModalityText | chat.ModalityAudio}),
		"/chat/completions", `{"model":"agent/audio","messages":[{"role":"user","content":"speak"}]}`, nil)
	assert.GreaterOrEqual(t, recorder.Code, http.StatusBadRequest)
}

func TestImageWithoutTool(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.MediaItem(chat.ImagePart(chat.InlineMedia("image/png", []byte{1}))))
	})
	recorder := postJSON(t, testHandler(agent, chat.Info{OutputModalities: chat.ModalityText | chat.ModalityImage, ImageGeneration: true}),
		"/responses", `{"model":"agent/image","stream":true,"input":"draw"}`, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
}

func TestFileOutputReject(t *testing.T) {
	agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.MediaItem(chat.FilePart(chat.InlineMedia("application/pdf", []byte{1}))))
	})
	recorder := postJSON(t, testHandler(agent, chat.Info{OutputModalities: chat.ModalityText}),
		"/chat/completions", `{"model":"agent/file","messages":[{"role":"user","content":"pdf"}]}`, nil)
	assert.GreaterOrEqual(t, recorder.Code, http.StatusBadRequest)
}

func TestResponseStatusMapping(t *testing.T) {
	cases := []struct {
		name       string
		status     chat.Status
		stop       chat.StopReason
		runErr     error
		wantStatus chat.Status
		wantReason string
	}{
		{name: "completed", wantStatus: chat.StatusCompleted},
		{name: "incompleteLength", status: chat.StatusIncomplete, stop: chat.StopLength, wantStatus: chat.StatusIncomplete, wantReason: "max_output_tokens"},
		{name: "incompleteUnknown", status: chat.StatusIncomplete, wantStatus: chat.StatusIncomplete, wantReason: "incomplete"},
		{name: "cancelled", status: chat.StatusCancelled, runErr: errors.New("cancelled"), wantStatus: chat.StatusCancelled},
		{name: "failed", runErr: errors.New("backend"), wantStatus: chat.StatusFailed},
		{name: "inProgress", status: chat.StatusInProgress, wantStatus: chat.StatusInProgress},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp := buildResponse(chat.Response{ID: "resp_1"}, internalexecution.Result{Outcome: chat.Outcome{Status: tc.status, StopReason: tc.stop}}, tc.runErr)
			assert.Equal(t, tc.wantStatus, resp.Status)
			assert.Equal(t, tc.wantReason, resp.Incomplete)
			if tc.name == "inProgress" {
				assert.Zero(t, resp.CompletedAt)
			}
			if tc.runErr != nil {
				require.NotNil(t, resp.Error)
				assert.Empty(t, resp.Error.Err)
			}
		})
	}

	assert.Equal(t, chat.StopError, outcomeFromResponse(chat.Response{Status: chat.StatusFailed}).StopReason)
	assert.Equal(t, chat.StopCancelled, outcomeFromResponse(chat.Response{Status: chat.StatusCancelled}).StopReason)
	assert.Equal(t, chat.StopLength, outcomeFromResponse(chat.Response{Status: chat.StatusIncomplete, Incomplete: "max_output_tokens"}).StopReason)
}

type deliveryFailureWriter struct {
	header http.Header
	code   int
	writes int
}

func (w *deliveryFailureWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}

func (w *deliveryFailureWriter) WriteHeader(code int) { w.code = code }

func (w *deliveryFailureWriter) Write([]byte) (int, error) {
	w.writes++
	return 0, errors.New("client disconnected")
}

func (w *deliveryFailureWriter) Flush() {}

func TestDeliveryFinalization(t *testing.T) {
	life := newMemoryLife()
	life.boundCleanup = true
	handler := testHandler(chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("ok"))
	}), chat.Info{Continuation: true}, WithStore(life))

	recorder := &deliveryFailureWriter{}
	request := httptest.NewRequest(http.MethodPost, "/responses", strings.NewReader(`{"model":"agent/basic","stream":true,"store":true,"input":"x"}`))
	request = request.WithContext(context.WithValue(request.Context(), ctxKey{}, "delivery-test"))
	request.Header.Set("Content-Type", "application/json")
	handler.ServeHTTP(recorder, request)

	assert.Equal(t, http.StatusOK, recorder.code)
	assert.Equal(t, 1, recorder.writes)
	assert.Equal(t, 1, life.finals, "execution must still be finalized after a bounded delivery failure")
}

func TestReplayOutputContract(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(strconv.FormatBool(stream), func(t *testing.T) {
			life := newMemoryLife()
			life.byKey["replay"] = replayEntry{response: chat.Response{
				ID:     "resp_replay",
				Target: "agent/basic",
				Output: []chat.Item{{Type: chat.ItemExtension}},
			}}
			handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
				t.Fatal("replayed requests must not execute the agent")
				return chat.Outcome{}, nil
			}), chat.Info{Continuation: true}, WithStore(life))
			body := `{"model":"agent/basic","stream":` + strconv.FormatBool(stream) + `,"store":true,"input":"x"}`
			recorder := postJSON(t, handler, "/responses", body, map[string]string{"Idempotency-Key": "replay"})
			if stream {
				require.Equal(t, http.StatusOK, recorder.Code)
				assert.Contains(t, recorder.Body.String(), "response.failed")
				return
			}
			require.Equal(t, http.StatusBadRequest, recorder.Code)
			assert.Equal(t, "unsupported", responseError(t, recorder)["code"])
		})
	}
}

func TestNewID(t *testing.T) {
	first := newID()
	second := newID()
	require.NotEmpty(t, first)
	assert.NotEqual(t, first, second)
}
