package llmux

import (
	"github.com/kelindar/llmux/audio"
)

// Transcriber is the optional service behind POST /audio/transcriptions.
type Transcriber = audio.Transcriber

// TranscriberFunc adapts a function to Transcriber.
type TranscriberFunc = audio.TranscriberFunc

// TranscriptionRequest is the normalized multipart input for a Transcriber.
type TranscriptionRequest = audio.TranscriptionRequest

// Transcription is the normalized result of a transcription service.
type Transcription = audio.Transcription

// TranscriptSegment is a timed portion of a verbose transcription.
type TranscriptSegment = audio.TranscriptSegment

// Speaker is the optional service behind POST /audio/speech.
type Speaker = audio.Speaker

// SpeakerFunc adapts a function to Speaker.
type SpeakerFunc = audio.SpeakerFunc

// SpeechRequest is the normalized JSON input for a Speaker.
type SpeechRequest = audio.SpeechRequest

// Speech is the complete audio returned by a Speaker.
type Speech = audio.Speech
