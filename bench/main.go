// Command bench measures the contract, protocol adapters, and end-to-end
// handler paths. Run it with `go run ./bench`.
package main

import (
	"bytes"
	"context"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"time"

	"github.com/kelindar/bench"
	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/audio"
	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/anthropic"
	completions "github.com/kelindar/llmux/internal/completions"
	"github.com/kelindar/llmux/internal/execution"
	"github.com/kelindar/llmux/internal/responses"
	"github.com/kelindar/llmux/internal/wire"
)

var keep any

type benchCatalog struct {
	agent chat.Agent
}

func (c benchCatalog) List(context.Context) (map[string]chat.Info, error) {
	return map[string]chat.Info{"bench": {}}, nil
}

func (c benchCatalog) Load(context.Context, string) (chat.Agent, chat.Info, error) {
	return c.agent, chat.Info{}, nil
}

func main() {
	chatBody := []byte(`{"model":"bench","messages":[{"role":"user","content":"hello"}]}`)
	responsesBody := []byte(`{"model":"bench","input":"hello"}`)
	anthropicBody := []byte(`{"model":"bench","max_tokens":32,"messages":[{"role":"user","content":"hello"}]}`)

	chatObject, err := wire.DecodeObject(chatBody)
	if err != nil {
		panic(err)
	}
	responsesObject, err := wire.DecodeObject(responsesBody)
	if err != nil {
		panic(err)
	}
	anthropicObject, err := wire.DecodeObject(anthropicBody)
	if err != nil {
		panic(err)
	}

	agent := chat.AgentFunc(func(ctx context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit(chat.Text("ok"))
	})
	handler := llmux.New(benchCatalog{agent: agent},
		llmux.WithTranscriber(audio.TranscriberFunc(func(context.Context, audio.TranscriptionRequest) (audio.Transcription, error) {
			return audio.Transcription{Text: "ok"}, nil
		})),
		llmux.WithSpeaker(audio.SpeakerFunc(func(context.Context, audio.SpeechRequest) (audio.Speech, error) {
			return audio.Speech{Data: []byte("audio"), MIMEType: "audio/mpeg"}, nil
		})),
	)

	bench.Run(func(b *bench.B) {
		b.Run("wire/decode", func(int) {
			keep, _ = wire.DecodeObject(chatBody)
		})

		b.Run("exec/run", func(int) {
			result, err := execution.Run(context.Background(), &chat.Request{Target: "bench"}, agent, chat.DefaultLimits(), nil)
			if err != nil {
				panic(err)
			}
			keep = result
		})
		b.Run("chat/models", func(int) { keep = serve(handler, "/models", nil, "GET") })
		b.Run("chat/parse", func(int) {
			keep, _ = completions.ParseRequest(chatObject)
		})
		b.Run("chat/completion", func(int) { keep = serve(handler, "/chat/completions", chatBody, "") })
		b.Run("chat/stream", func(int) {
			keep = serve(handler, "/chat/completions", []byte(`{"model":"bench","stream":true,"messages":[{"role":"user","content":"hello"}]}`), "")
		})
		b.Run("responses/parse", func(int) {
			keep, _ = responses.ParseRequest(responsesObject)
		})
		b.Run("responses/http", func(int) { keep = serve(handler, "/responses", responsesBody, "") })
		b.Run("responses/stream", func(int) {
			keep = serve(handler, "/responses", []byte(`{"model":"bench","stream":true,"input":"hello"}`), "")
		})
		b.Run("anthropic/parse", func(int) {
			keep, _ = anthropic.ParseRequest(anthropicObject)
		})
		b.Run("anthropic/http", func(int) { keep = serve(handler, "/messages", anthropicBody, "anthropic-version: 2023-06-01") })
		b.Run("anthropic/stream", func(int) {
			keep = serve(handler, "/messages", []byte(`{"model":"bench","stream":true,"max_tokens":32,"messages":[{"role":"user","content":"hello"}]}`), "anthropic-version: 2023-06-01")
		})
		b.Run("audio/speech", func(int) {
			keep = serve(handler, "/audio/speech", []byte(`{"model":"bench","input":"hello","voice":"alloy"}`), "")
		})
		b.Run("audio/transcribe", func(int) { keep = serveTranscription(handler) })
	}, bench.WithSamples(50), bench.WithDuration(10*time.Millisecond))
}

func serve(handler http.Handler, path string, body []byte, header string) *httptest.ResponseRecorder {
	method := http.MethodPost
	if header == "GET" {
		method = http.MethodGet
	}
	request := httptest.NewRequest(method, path, bytes.NewReader(body))
	if header != "" && header != "GET" {
		key, value, _ := bytes.Cut([]byte(header), []byte(":"))
		request.Header.Set(string(key), string(bytes.TrimSpace(value)))
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		panic(response.Code)
	}
	return response
}

func serveTranscription(handler http.Handler) *httptest.ResponseRecorder {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	if err := writer.WriteField("model", "bench"); err != nil {
		panic(err)
	}
	file, err := writer.CreateFormFile("file", "sample.wav")
	if err != nil {
		panic(err)
	}
	if _, err := file.Write([]byte("audio")); err != nil {
		panic(err)
	}
	if err := writer.Close(); err != nil {
		panic(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/audio/transcriptions", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		panic(response.Code)
	}
	return response
}
