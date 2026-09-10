package audio

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTranscriberFunc(t *testing.T) {
	var nilTranscriber TranscriberFunc
	_, err := nilTranscriber.Transcribe(context.Background(), TranscriptionRequest{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nil transcriber function")

	transcriber := TranscriberFunc(func(_ context.Context, req TranscriptionRequest) (Transcription, error) {
		assert.Equal(t, "whisper", req.Model)
		assert.Equal(t, []byte("audio"), req.Data)
		return Transcription{Text: "hello", Language: "en", Duration: 1.5}, nil
	})
	got, err := transcriber.Transcribe(context.Background(), TranscriptionRequest{
		Model: "whisper",
		Data:  []byte("audio"),
	})
	require.NoError(t, err)
	assert.Equal(t, "hello", got.Text)
	assert.Equal(t, "en", got.Language)
	assert.InDelta(t, 1.5, got.Duration, 0.001)
}

func TestSpeakerFunc(t *testing.T) {
	var nilSpeaker SpeakerFunc
	_, err := nilSpeaker.Speak(context.Background(), SpeechRequest{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nil speaker function")

	speaker := SpeakerFunc(func(_ context.Context, req SpeechRequest) (Speech, error) {
		assert.Equal(t, "tts-1", req.Model)
		assert.Equal(t, "say hi", req.Input)
		return Speech{Data: []byte{1, 2}, MIMEType: "audio/mpeg", Format: "mp3"}, nil
	})
	got, err := speaker.Speak(context.Background(), SpeechRequest{
		Model: "tts-1",
		Input: "say hi",
	})
	require.NoError(t, err)
	assert.Equal(t, []byte{1, 2}, got.Data)
	assert.Equal(t, "audio/mpeg", got.MIMEType)
	assert.Equal(t, "mp3", got.Format)
}
