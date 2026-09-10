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
		{name: "chat", path: "/v1/chat/completions", body: `{"model":"agent/basic","stream":true,"messages":[{"role":"user","content":"hi"}]}`},
		{name: "responses", path: "/v1/responses", body: `{"model":"agent/basic","stream":true,"input":"hi"}`},
		{name: "anthropic", path: "/v1/messages", body: `{"model":"agent/basic","stream":true,"max_tokens":32,"messages":[{"role":"user","content":"hi"}]}`, headers: map[string]string{"anthropic-version": "2023-06-01"}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
				if err := EmitTextDelta(emit, "partial"); err != nil {
					return Outcome{}, err
				}
				return Outcome{}, errors.New("backend secret")
			})
			recorder := postJSON(t, testHandler(agent, Capabilities{}), test.path, test.body, test.headers)
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
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitMedia(emit, ImagePart(InlineMedia("image/png", []byte{1})))
	})
	recorder := postJSON(t, testHandler(agent, Capabilities{}), "/v1/chat/completions", `{"model":"agent/basic","stream":true,"messages":[{"role":"user","content":"draw"}]}`, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "unsupported", responseError(t, recorder)["code"])
	assert.NotContains(t, recorder.Body.String(), "text/event-stream")
}

func TestImageToolRequired(t *testing.T) {
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitMedia(emit, ImagePart(InlineMedia("image/png", []byte{1})))
	})
	recorder := postJSON(t, testHandler(agent, Capabilities{OutputModalities: ModalityText | ModalityImage, ImageGeneration: true}), "/v1/responses", `{"model":"agent/image","input":"draw"}`, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "unsupported", responseError(t, recorder)["code"])
}

func TestReasoningRequestGate(t *testing.T) {
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitItem(emit, Item{Type: ItemReasoning, Summary: []Part{SummaryPart("brief")}})
	})
	caps := Capabilities{ReasoningSummary: true}
	denied := postJSON(t, testHandler(agent, caps), "/v1/responses", `{"model":"agent/reasoning","input":"think"}`, nil)
	require.Equal(t, http.StatusBadRequest, denied.Code)
	assert.Equal(t, "unsupported", responseError(t, denied)["code"])

	allowed := postJSON(t, testHandler(agent, caps), "/v1/responses", `{"model":"agent/reasoning","reasoning":{"effort":"medium","summary":"auto"},"input":"think"}`, nil)
	require.Equal(t, http.StatusOK, allowed.Code)
	assert.Equal(t, "reasoning", decodeResponse(t, allowed)["output"].([]any)[0].(map[string]any)["type"])
}

type testContinuationStore struct {
	mu    sync.Mutex
	items map[string][]Item
}

func (s *testContinuationStore) Load(_ context.Context, id string) ([]Item, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	items, ok := s.items[id]
	if !ok {
		return nil, errors.New("missing continuation")
	}
	cloned := make([]Item, len(items))
	for n, item := range items {
		cloned[n] = item.Clone()
	}
	return cloned, nil
}

func (s *testContinuationStore) put(id string, items []Item) {
	s.mu.Lock()
	defer s.mu.Unlock()
	cloned := make([]Item, len(items))
	for n, item := range items {
		cloned[n] = item.Clone()
	}
	s.items[id] = cloned
}

type testLifecycle struct {
	store   *testContinuationStore
	fail    bool
	finals  int
	accepts int
	last    *TurnResult
}

func (l *testLifecycle) Accept(_ context.Context, _ *TurnRequest) (Acceptance, error) {
	l.accepts++
	return Acceptance{}, nil
}

func (l *testLifecycle) Finalize(_ context.Context, result *TurnResult) error {
	l.finals++
	clone := *result
	if result.Request != nil {
		req := *result.Request
		clone.Request = &req
	}
	clone.State = result.State.Clone()
	l.last = &clone
	if l.fail {
		return errors.New("finalize failed")
	}
	if !result.State.Store || l.store == nil {
		return nil
	}
	items := make([]Item, 0, len(result.Request.Input)+len(result.State.Output))
	for _, item := range result.Request.Input {
		items = append(items, item.Clone())
	}
	for _, item := range result.State.Output {
		items = append(items, item.Clone())
	}
	l.store.put(result.ID, items)
	return nil
}

func TestContinuation(t *testing.T) {
	store := &testContinuationStore{items: make(map[string][]Item)}
	life := &testLifecycle{store: store}
	agent := AgentFunc(func(_ context.Context, req *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitText(emit, strconv.Itoa(len(req.Input)))
	})
	handler := testHandler(agent, Capabilities{Continuation: true}, WithContinuationStore(store), WithLifecycle(life))

	first := postJSON(t, handler, "/v1/responses", `{"model":"agent/basic","store":true,"input":"first"}`, nil)
	require.Equal(t, http.StatusOK, first.Code)
	firstBody := decodeResponse(t, first)
	responseID, ok := firstBody["id"].(string)
	require.True(t, ok)
	require.Equal(t, 1, life.finals)
	require.NotNil(t, life.last)
	assert.Len(t, life.last.Request.Turn, 1)
	assert.Len(t, life.last.State.Output, 1)

	second := postJSON(t, handler, "/v1/responses", `{"model":"agent/basic","previous_response_id":"`+responseID+`","input":"second"}`, nil)
	require.Equal(t, http.StatusOK, second.Code)
	secondBody := decodeResponse(t, second)
	output := secondBody["output"].([]any)
	message := output[0].(map[string]any)
	content := message["content"].([]any)
	assert.Equal(t, "3", content[0].(map[string]any)["text"])
}

func TestNamespacedExtension(t *testing.T) {
	var got string
	agent := AgentFunc(func(_ context.Context, req *Request, emit Emit) (Outcome, error) {
		got = string(req.Controls.Extensions["x-vendor-trace"])
		return Outcome{}, EmitText(emit, "ok")
	})
	caps := Capabilities{Extensions: map[string]bool{"x-vendor-trace": true}}
	recorder := postJSON(t, testHandler(agent, caps), "/v1/chat/completions", `{"model":"agent/basic","x-vendor-trace":{"enabled":true},"messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	assert.JSONEq(t, `{"enabled":true}`, got)
}

func TestUnsupportedChatUser(t *testing.T) {
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitText(emit, "unreachable")
	})
	recorder := postJSON(t, testHandler(agent, Capabilities{}), "/v1/chat/completions", `{"model":"agent/basic","user":"end-user","messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "unsupported", responseError(t, recorder)["code"])
}

func TestUnsupportedChatContinuation(t *testing.T) {
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitText(emit, "unreachable")
	})
	recorder := postJSON(t, testHandler(agent, Capabilities{}), "/v1/chat/completions", `{"model":"agent/basic","previous_response_id":"resp_1","messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "unsupported", responseError(t, recorder)["code"])
}

func TestJSONV2Duplicates(t *testing.T) {
	bodies := []string{
		`{"model":"agent/basic","model":"agent/other","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"agent/basic","messages":[{"role":"user","content":"hi"}]} {}`,
	}
	for _, body := range bodies {
		recorder := postJSON(t, testHandler(AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
			return Outcome{}, EmitText(emit, "unreachable")
		}), Capabilities{}), "/v1/chat/completions", body, nil)
		assert.Equal(t, http.StatusBadRequest, recorder.Code)
	}
}

func TestMultipartMalformed(t *testing.T) {
	handler := testHandler(AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitText(emit, "unreachable")
	}), Capabilities{}, WithTranscriber(TranscriberFunc(func(context.Context, TranscriptionRequest) (Transcription, error) {
		return Transcription{}, nil
	})))
	recorder := postRaw(t, handler, "/v1/audio/transcriptions", []byte("--broken\r\nContent-Disposition: form-data; name=\"file\"\r\n\r\nx"), "multipart/form-data; boundary=broken", nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "invalid_multipart", responseError(t, recorder)["code"])
}

func TestSpeechSSE(t *testing.T) {
	handler := testHandler(AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitText(emit, "unreachable")
	}), Capabilities{}, WithSpeaker(SpeakerFunc(func(context.Context, SpeechRequest) (Speech, error) {
		return Speech{Data: []byte{1, 2, 3}, Format: "wav"}, nil
	})))
	recorder := postJSON(t, handler, "/v1/audio/speech", `{"model":"tts","input":"say hi","voice":"alloy","response_format":"wav","stream_format":"sse"}`, nil)
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
	handler := testHandler(AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitText(emit, "unreachable")
	}), Capabilities{}, WithTranscriber(TranscriberFunc(func(_ context.Context, req TranscriptionRequest) (Transcription, error) {
		assert.Equal(t, "sample.wav", req.Filename)
		return Transcription{Text: "hello", Language: "en", Duration: 1.5, Segments: []TranscriptSegment{{ID: 0, Start: 0, End: 1.5, Text: "hello"}}}, nil
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

	recorder := postRaw(t, handler, "/v1/audio/transcriptions", body.Bytes(), writer.FormDataContentType(), nil)
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
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		go func() { firstErr <- emit(TextDelta("first")) }()
		<-entered
		err := emit(TextDelta("second"))
		secondErr <- err
		close(release)
		<-firstErr
		return Outcome{}, err
	})
	onEvent := func(Event) error {
		select {
		case <-entered:
		default:
			close(entered)
		}
		<-release
		return nil
	}
	_, err := internalexecution.Run(context.Background(), &Request{Target: "agent/basic"}, agent, DefaultLimits(), onEvent)
	require.ErrorIs(t, err, ErrConcurrentEmit)
	require.ErrorIs(t, <-secondErr, ErrConcurrentEmit)
}

func TestOutcomeDefaults(t *testing.T) {
	cases := []struct {
		status Status
		stop   StopReason
	}{
		{status: StatusCompleted, stop: StopStop},
		{status: StatusIncomplete, stop: StopLength},
		{status: StatusFailed, stop: StopError},
		{status: StatusCancelled, stop: StopCancelled},
	}
	for _, test := range cases {
		t.Run(string(test.status), func(t *testing.T) {
			agent := AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
				return Outcome{Status: test.status}, nil
			})
			result, err := internalexecution.Run(context.Background(), &Request{Target: "agent/basic"}, agent, DefaultLimits(), nil)
			require.NoError(t, err)
			assert.Equal(t, test.stop, result.Outcome.StopReason)
		})
	}

	agent := AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{Status: StatusInProgress}, nil
	})
	_, err := internalexecution.Run(context.Background(), &Request{Target: "agent/basic"}, agent, DefaultLimits(), nil)
	assert.Error(t, err)
}

func TestMultipartSizeLimit(t *testing.T) {
	handler := testHandler(AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitText(emit, "unreachable")
	}), Capabilities{}, WithTranscriber(TranscriberFunc(func(context.Context, TranscriptionRequest) (Transcription, error) {
		return Transcription{}, nil
	})), WithLimits(Limits{MaxMultipartBytes: 32}))
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	require.NoError(t, writer.WriteField("model", "whisper"))
	file, err := writer.CreateFormFile("file", "sample.wav")
	require.NoError(t, err)
	_, err = file.Write([]byte(strings.Repeat("a", 128)))
	require.NoError(t, err)
	require.NoError(t, writer.Close())
	recorder := postRaw(t, handler, "/v1/audio/transcriptions", body.Bytes(), writer.FormDataContentType(), nil)
	assert.Equal(t, http.StatusRequestEntityTooLarge, recorder.Code)
}

func TestProtocolFixtures(t *testing.T) {
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitText(emit, "fixture")
	})
	handler := testHandler(agent, Capabilities{})
	cases := []struct {
		name    string
		file    string
		path    string
		headers map[string]string
	}{
		{name: "chat", file: "fixtures/chat-completion.json", path: "/v1/chat/completions"},
		{name: "responses", file: "fixtures/responses.json", path: "/v1/responses"},
		{name: "anthropic", file: "fixtures/messages.json", path: "/v1/messages", headers: map[string]string{"anthropic-version": "2023-06-01"}},
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

	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		return Outcome{}, EmitText(emit, "ok")
	})
	handler := testHandler(agent, Capabilities{})
	f.Fuzz(func(t *testing.T, body []byte) {
		cases := []struct {
			path    string
			headers map[string]string
		}{
			{path: "/v1/chat/completions"},
			{path: "/v1/responses"},
			{path: "/v1/messages", headers: map[string]string{"anthropic-version": "2023-06-01"}},
		}
		for _, test := range cases {
			request := httptest.NewRequest(http.MethodPost, test.path, bytes.NewReader(body))
			request.Header.Set("Content-Type", "application/json")
			for key, value := range test.headers {
				request.Header.Set(key, value)
			}
			recorder := httptest.NewRecorder()
			require.NotPanics(t, func() { handler.ServeHTTP(recorder, request) })
			assert.GreaterOrEqual(t, recorder.Code, http.StatusOK)
			assert.LessOrEqual(t, recorder.Code, 599)
		}
	})
}

func TestAPIErrorMapping(t *testing.T) {
	apiErr := asAPIError(Invalid("model", "required"))
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
	stream := &sseWriter{w: recorder, limits: DefaultLimits()}
	require.NoError(t, stream.start())
	assert.True(t, stream.Started())
	require.NoError(t, stream.write("", map[string]string{"type": "ping"}))
	require.NoError(t, stream.done())
	assert.Contains(t, recorder.Body.String(), "[DONE]")
}

func TestFinalizeError(t *testing.T) {
	var logged error
	life := &testLifecycle{fail: true}
	handler := testHandler(AgentFunc(func(_ context.Context, _ *Request, emitFn Emit) (Outcome, error) {
		return Outcome{}, EmitText(emitFn, "ok")
	}), Capabilities{Continuation: true},
		WithLifecycle(life),
		WithErrorLog(func(_ context.Context, err error) { logged = err }),
	)
	recorder := postJSON(t, handler, "/v1/responses", `{"model":"agent/basic","store":true,"input":"first"}`, nil)
	require.Equal(t, http.StatusInternalServerError, recorder.Code)
	require.Error(t, logged)
	assert.Equal(t, 1, life.finals)
}

func TestStreamImmediateFailure(t *testing.T) {
	var logged error
	handler := testHandler(AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, errors.New("immediate")
	}), Capabilities{}, WithErrorLog(func(_ context.Context, err error) { logged = err }))
	recorder := postJSON(t, handler, "/v1/chat/completions", `{"model":"agent/basic","stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	require.Equal(t, http.StatusInternalServerError, recorder.Code)
	assert.NotContains(t, recorder.Header().Get("Content-Type"), "text/event-stream")
	require.Error(t, logged)
}

func TestAdapterResponseError(t *testing.T) {
	agent := AgentFunc(func(_ context.Context, _ *Request, emitFn Emit) (Outcome, error) {
		media := InlineMedia("audio/wav", []byte{1})
		media.Format = "wav"
		return Outcome{}, EmitMedia(emitFn, AudioPart(media))
	})
	recorder := postJSON(t, testHandler(agent, Capabilities{OutputModalities: ModalityText | ModalityAudio}),
		"/v1/chat/completions", `{"model":"agent/audio","messages":[{"role":"user","content":"speak"}]}`, nil)
	assert.GreaterOrEqual(t, recorder.Code, http.StatusBadRequest)
}

func TestImageWithoutTool(t *testing.T) {
	agent := AgentFunc(func(_ context.Context, _ *Request, emitFn Emit) (Outcome, error) {
		return Outcome{}, EmitMedia(emitFn, ImagePart(InlineMedia("image/png", []byte{1})))
	})
	recorder := postJSON(t, testHandler(agent, Capabilities{OutputModalities: ModalityText | ModalityImage, ImageGeneration: true}),
		"/v1/responses", `{"model":"agent/image","stream":true,"input":"draw"}`, nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
}

func TestFileOutputReject(t *testing.T) {
	agent := AgentFunc(func(_ context.Context, _ *Request, emitFn Emit) (Outcome, error) {
		return Outcome{}, EmitMedia(emitFn, FilePart(InlineMedia("application/pdf", []byte{1})))
	})
	recorder := postJSON(t, testHandler(agent, Capabilities{OutputModalities: ModalityText}),
		"/v1/chat/completions", `{"model":"agent/file","messages":[{"role":"user","content":"pdf"}]}`, nil)
	assert.GreaterOrEqual(t, recorder.Code, http.StatusBadRequest)
}
