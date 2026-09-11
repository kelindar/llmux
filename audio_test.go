package llmux

import (
	"bytes"
	"context"
	"errors"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/kelindar/llmux/audio"
	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSpeechMIME(t *testing.T) {
	cases := map[string]string{
		"mp3": "audio/mpeg", "opus": "audio/opus", "aac": "audio/aac",
		"flac": "audio/flac", "wav": "audio/wav", "pcm": "audio/pcm",
		"unknown": "",
	}
	for format, want := range cases {
		t.Run(format, func(t *testing.T) {
			assert.Equal(t, want, speechMIME(format))
		})
	}
}

func TestValidSpeechFormat(t *testing.T) {
	for _, format := range []string{"mp3", "opus", "aac", "flac", "wav", "pcm"} {
		assert.True(t, validSpeechFormat(format))
	}
	assert.False(t, validSpeechFormat("ogg"))
}

func TestParseSpeechVoice(t *testing.T) {
	voice, err := parseSpeechVoice([]byte(`"alloy"`))
	require.NoError(t, err)
	assert.Equal(t, "alloy", voice)

	voice, err = parseSpeechVoice([]byte(`{"id":"custom-voice"}`))
	require.NoError(t, err)
	assert.Equal(t, "custom-voice", voice)

	_, err = parseSpeechVoice([]byte(`{"name":"bad"}`))
	require.Error(t, err)

	_, err = parseSpeechVoice([]byte(`123`))
	require.Error(t, err)
}

func TestSpeechNotConfigured(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{})
	recorder := postJSON(t, handler, "/audio/speech", `{"model":"tts","input":"hi","voice":"alloy"}`, nil)
	require.Equal(t, http.StatusNotFound, recorder.Code)
	assert.Equal(t, "not_found", responseError(t, recorder)["code"])
}

func TestTranscriptionNotConfigured(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{})
	recorder := postRaw(t, handler, "/audio/transcriptions", []byte("x"), "application/json", nil)
	require.Equal(t, http.StatusNotFound, recorder.Code)
}

func TestTranscriptionContentType(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{}, WithTranscriber(audio.TranscriberFunc(func(context.Context, audio.TranscriptionRequest) (audio.Transcription, error) {
		return audio.Transcription{}, nil
	})))
	recorder := postRaw(t, handler, "/audio/transcriptions", []byte("x"), "application/json", nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "invalid_request", responseError(t, recorder)["code"])
}

func TestTranscriptionMissingFields(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{}, WithTranscriber(audio.TranscriberFunc(func(context.Context, audio.TranscriptionRequest) (audio.Transcription, error) {
		return audio.Transcription{}, nil
	})))

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	require.NoError(t, writer.Close())
	recorder := postRaw(t, handler, "/audio/transcriptions", body.Bytes(), writer.FormDataContentType(), nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "file", responseError(t, recorder)["param"])
}

func TestTranscriptionUnsupportedField(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{}, WithTranscriber(audio.TranscriberFunc(func(context.Context, audio.TranscriptionRequest) (audio.Transcription, error) {
		return audio.Transcription{}, nil
	})))
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	require.NoError(t, writer.WriteField("model", "whisper"))
	require.NoError(t, writer.WriteField("timestamp_granularities", "word"))
	file, err := writer.CreateFormFile("file", "sample.wav")
	require.NoError(t, err)
	_, err = file.Write([]byte("audio"))
	require.NoError(t, err)
	require.NoError(t, writer.Close())
	recorder := postRaw(t, handler, "/audio/transcriptions", body.Bytes(), writer.FormDataContentType(), nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
	assert.Equal(t, "unsupported", responseError(t, recorder)["code"])
}

func TestTranscriptionFormats(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{}, WithTranscriber(audio.TranscriberFunc(func(_ context.Context, req audio.TranscriptionRequest) (audio.Transcription, error) {
		return audio.Transcription{Text: "hello"}, nil
	})))

	t.Run("text", func(t *testing.T) {
		recorder := postTranscription(t, handler, map[string]string{"model": "whisper", "response_format": "text"})
		require.Equal(t, http.StatusOK, recorder.Code)
		assert.Equal(t, "hello", recorder.Body.String())
	})

	t.Run("json", func(t *testing.T) {
		recorder := postTranscription(t, handler, map[string]string{"model": "whisper", "response_format": "json"})
		require.Equal(t, http.StatusOK, recorder.Code)
		assert.Equal(t, "hello", decodeResponse(t, recorder)["text"])
	})

	t.Run("unsupported", func(t *testing.T) {
		recorder := postTranscription(t, handler, map[string]string{"model": "whisper", "response_format": "srt"})
		require.Equal(t, http.StatusBadRequest, recorder.Code)
		assert.Equal(t, "unsupported", responseError(t, recorder)["code"])
	})
}

func TestTranscriptionDuplicateFile(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{}, WithTranscriber(audio.TranscriberFunc(func(context.Context, audio.TranscriptionRequest) (audio.Transcription, error) {
		return audio.Transcription{}, nil
	})))
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	require.NoError(t, writer.WriteField("model", "whisper"))
	fileA, err := writer.CreateFormFile("file", "a.wav")
	require.NoError(t, err)
	_, err = fileA.Write([]byte("audio"))
	require.NoError(t, err)
	fileB, err := writer.CreateFormFile("file", "b.wav")
	require.NoError(t, err)
	_, err = fileB.Write([]byte("audio"))
	require.NoError(t, err)
	require.NoError(t, writer.Close())
	recorder := postRaw(t, handler, "/audio/transcriptions", body.Bytes(), writer.FormDataContentType(), nil)
	require.Equal(t, http.StatusBadRequest, recorder.Code)
}

func TestTranscriptionFailure(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{}, WithTranscriber(audio.TranscriberFunc(func(context.Context, audio.TranscriptionRequest) (audio.Transcription, error) {
		return audio.Transcription{}, chat.Invalid("file", "bad audio")
	})))
	recorder := postTranscription(t, handler, map[string]string{"model": "whisper"})
	require.Equal(t, http.StatusBadRequest, recorder.Code)
}

func TestTranscriptionTemperature(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{}, WithTranscriber(audio.TranscriberFunc(func(context.Context, audio.TranscriptionRequest) (audio.Transcription, error) {
		return audio.Transcription{}, nil
	})))

	recorder := postTranscription(t, handler, map[string]string{"model": "whisper", "temperature": "bad"})
	require.Equal(t, http.StatusBadRequest, recorder.Code)

	recorder = postTranscription(t, handler, map[string]string{"model": "whisper", "temperature": "2"})
	require.Equal(t, http.StatusBadRequest, recorder.Code)
}

func TestSpeechValidation(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{}, WithSpeaker(audio.SpeakerFunc(func(context.Context, audio.SpeechRequest) (audio.Speech, error) {
		return audio.Speech{Data: []byte("audio"), MIMEType: "audio/mpeg"}, nil
	})))

	cases := []struct {
		name string
		body string
	}{
		{name: "missing voice", body: `{"model":"tts","input":"hi"}`},
		{name: "long input", body: `{"model":"tts","input":"` + strings.Repeat("a", 4097) + `","voice":"alloy"}`},
		{name: "bad format", body: `{"model":"tts","input":"hi","voice":"alloy","response_format":"ogg"}`},
		{name: "bad speed", body: `{"model":"tts","input":"hi","voice":"alloy","speed":10}`},
		{name: "bad stream", body: `{"model":"tts","input":"hi","voice":"alloy","stream_format":"json"}`},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			recorder := postJSON(t, handler, "/audio/speech", test.body, nil)
			assert.Equal(t, http.StatusBadRequest, recorder.Code)
		})
	}
}

func TestSpeechObjectVoice(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{}, WithSpeaker(audio.SpeakerFunc(func(_ context.Context, req audio.SpeechRequest) (audio.Speech, error) {
		assert.Equal(t, "custom", req.Voice)
		return audio.Speech{Data: []byte("audio"), MIMEType: "audio/mpeg"}, nil
	})))
	recorder := postJSON(t, handler, "/audio/speech", `{"model":"tts","input":"hi","voice":{"id":"custom"}}`, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
}

func TestSpeechBinaryResponse(t *testing.T) {
	handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, nil
	}), chat.Info{}, WithSpeaker(audio.SpeakerFunc(func(context.Context, audio.SpeechRequest) (audio.Speech, error) {
		return audio.Speech{Data: []byte("audio"), Format: "wav"}, nil
	})))
	recorder := postJSON(t, handler, "/audio/speech", `{"model":"tts","input":"hi","voice":"alloy","response_format":"wav"}`, nil)
	require.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, "audio/wav", recorder.Header().Get("Content-Type"))
}

func TestSpeechErrors(t *testing.T) {
	t.Run("empty audio", func(t *testing.T) {
		handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, nil
		}), chat.Info{}, WithSpeaker(audio.SpeakerFunc(func(context.Context, audio.SpeechRequest) (audio.Speech, error) {
			return audio.Speech{}, nil
		})))
		recorder := postJSON(t, handler, "/audio/speech", `{"model":"tts","input":"hi","voice":"alloy"}`, nil)
		require.Equal(t, http.StatusInternalServerError, recorder.Code)
	})

	t.Run("speaker failure", func(t *testing.T) {
		var logged error
		handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, nil
		}), chat.Info{},
			WithSpeaker(audio.SpeakerFunc(func(context.Context, audio.SpeechRequest) (audio.Speech, error) {
				return audio.Speech{}, errors.New("speaker down")
			})),
			WithErrorLog(func(_ context.Context, err error) { logged = err }),
		)
		recorder := postJSON(t, handler, "/audio/speech", `{"model":"tts","input":"hi","voice":"alloy"}`, nil)
		require.NotEqual(t, http.StatusOK, recorder.Code)
		require.Error(t, logged)
	})

	t.Run("output too large", func(t *testing.T) {
		handler := testHandler(chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, nil
		}), chat.Info{},
			WithSpeaker(audio.SpeakerFunc(func(context.Context, audio.SpeechRequest) (audio.Speech, error) {
				return audio.Speech{Data: []byte("0123456789")}, nil
			})),
			WithLimits(chat.Limits{MaxOutputBytes: 4}),
		)
		recorder := postJSON(t, handler, "/audio/speech", `{"model":"tts","input":"hi","voice":"alloy"}`, nil)
		require.Equal(t, http.StatusRequestEntityTooLarge, recorder.Code)
	})
}

func postTranscription(t *testing.T, handler http.Handler, fields map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	for key, value := range fields {
		require.NoError(t, writer.WriteField(key, value))
	}
	file, err := writer.CreateFormFile("file", "sample.wav")
	require.NoError(t, err)
	_, err = file.Write([]byte("audio"))
	require.NoError(t, err)
	require.NoError(t, writer.Close())
	return postRaw(t, handler, "/audio/transcriptions", body.Bytes(), writer.FormDataContentType(), nil)
}
