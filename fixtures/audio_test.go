// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package fixtures

import (
	"bytes"
	"context"
	"io"
	"testing"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/audio"
	"github.com/kelindar/llmux/chat"
	"github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAIAudio(t *testing.T) {
	ctx := context.Background()

	t.Run("speechAndTranscription", func(t *testing.T) {
		server := startServer(echoAgent("unused"), chat.Info{},
			llmux.WithTranscriber(audio.TranscriberFunc(func(_ context.Context, req audio.TranscriptionRequest) (audio.Transcription, error) {
				assert.Equal(t, "gpt-4o-transcribe", req.Model)
				assert.Equal(t, []byte("audio"), req.Data)
				return audio.Transcription{Text: "hello"}, nil
			})),
			llmux.WithSpeaker(audio.SpeakerFunc(func(_ context.Context, req audio.SpeechRequest) (audio.Speech, error) {
				assert.Equal(t, "say hello", req.Input)
				assert.Equal(t, "mp3", req.ResponseFormat)
				return audio.Speech{Data: []byte("audio"), MIMEType: "audio/mpeg"}, nil
			})),
		)
		defer server.Close()
		client := openaiClient(server)

		speech, err := client.Audio.Speech.New(ctx, openai.AudioSpeechNewParams{
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

		transcription, err := client.Audio.Transcriptions.New(ctx, openai.AudioTranscriptionNewParams{
			File:           bytes.NewReader([]byte("audio")),
			Model:          openai.AudioModelGPT4oTranscribe,
			ResponseFormat: openai.AudioResponseFormatJSON,
		})
		require.NoError(t, err)
		assert.Equal(t, "hello", transcription.AsTranscription().Text)
	})

	t.Run("speechWAV", func(t *testing.T) {
		server := startServer(echoAgent("unused"), chat.Info{},
			llmux.WithSpeaker(audio.SpeakerFunc(func(_ context.Context, req audio.SpeechRequest) (audio.Speech, error) {
				assert.Equal(t, "wav", req.ResponseFormat)
				return audio.Speech{Data: []byte("wav-bytes"), MIMEType: "audio/wav"}, nil
			})),
		)
		defer server.Close()
		client := openaiClient(server)

		speech, err := client.Audio.Speech.New(ctx, openai.AudioSpeechNewParams{
			Input:          "say hello",
			Model:          openai.SpeechModelTTS1,
			Voice:          openai.AudioSpeechNewParamsVoiceUnion{OfAudioSpeechNewsVoiceString2: openai.String("alloy")},
			ResponseFormat: openai.AudioSpeechNewParamsResponseFormatWAV,
		})
		require.NoError(t, err)
		data, err := io.ReadAll(speech.Body)
		require.NoError(t, err)
		require.NoError(t, speech.Body.Close())
		assert.Equal(t, []byte("wav-bytes"), data)
	})

	t.Run("speechWithoutSpeaker", func(t *testing.T) {
		server := startServer(echoAgent("unused"), chat.Info{})
		defer server.Close()
		client := openaiClient(server)

		_, err := client.Audio.Speech.New(ctx, openai.AudioSpeechNewParams{
			Input: "say hello",
			Model: openai.SpeechModelTTS1,
			Voice: openai.AudioSpeechNewParamsVoiceUnion{OfAudioSpeechNewsVoiceString2: openai.String("alloy")},
		})
		require.Error(t, err)
	})

	t.Run("transcriptionWithoutTranscriber", func(t *testing.T) {
		server := startServer(echoAgent("unused"), chat.Info{})
		defer server.Close()
		client := openaiClient(server)

		_, err := client.Audio.Transcriptions.New(ctx, openai.AudioTranscriptionNewParams{
			File:  bytes.NewReader([]byte("audio")),
			Model: openai.AudioModelGPT4oTranscribe,
		})
		require.Error(t, err)
	})
}
