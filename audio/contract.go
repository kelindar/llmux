// Package audio contains the optional service contracts used by llmux's
// transcription and speech endpoints.
package audio

import (
	"context"
	"errors"
)

// Transcriber is the optional service behind POST /v1/audio/transcriptions.
// It is deliberately separate from Agent because transcription does not have
// conversational event semantics.
type Transcriber interface {
	// Transcribe turns uploaded audio into a transcript.
	Transcribe(context.Context, TranscriptionRequest) (Transcription, error)
}

// TranscriberFunc adapts a function to Transcriber.
type TranscriberFunc func(context.Context, TranscriptionRequest) (Transcription, error)

// Transcribe implements Transcriber.
func (f TranscriberFunc) Transcribe(ctx context.Context, req TranscriptionRequest) (Transcription, error) {
	if f == nil {
		return Transcription{}, errors.New("llmux: nil transcriber function")
	}
	return f(ctx, req)
}

// TranscriptionRequest is the normalized multipart input for a Transcriber.
// Data is owned by the request and must be treated as immutable by the
// service after Transcribe returns.
type TranscriptionRequest struct {
	Model          string   // Target model or voice name from the multipart form.
	Filename       string   // Original uploaded filename, when provided.
	MIMEType       string   // Declared content type of the uploaded file.
	Data           []byte   // Raw audio bytes read from the uploaded file.
	Prompt         string   // Optional transcription prompt.
	Language       string   // Optional source language hint.
	ResponseFormat string   // Requested wire format: json, text, or verbose_json.
	Temperature    *float64 // Optional sampling temperature between 0 and 1.
}

// Transcription is the normalized result of a transcription service.
type Transcription struct {
	Text     string              // Full transcript text.
	Language string              // Detected or declared source language.
	Duration float64             // Audio duration in seconds for verbose responses.
	Segments []TranscriptSegment // Timed segments for verbose_json responses.
}

// TranscriptSegment is the portion of a verbose transcription with timing.
type TranscriptSegment struct {
	ID    int     `json:"id"`    // Segment index in the verbose transcript.
	Start float64 `json:"start"` // Segment start time in seconds.
	End   float64 `json:"end"`   // Segment end time in seconds.
	Text  string  `json:"text"`  // Transcript text for the segment.
}

// Speaker is the optional service behind POST /v1/audio/speech.
type Speaker interface {
	// Speak synthesizes speech audio for the request text.
	Speak(context.Context, SpeechRequest) (Speech, error)
}

// SpeakerFunc adapts a function to Speaker.
type SpeakerFunc func(context.Context, SpeechRequest) (Speech, error)

// Speak implements Speaker.
func (f SpeakerFunc) Speak(ctx context.Context, req SpeechRequest) (Speech, error) {
	if f == nil {
		return Speech{}, errors.New("llmux: nil speaker function")
	}
	return f(ctx, req)
}

// SpeechRequest is the normalized JSON input for a Speaker.
type SpeechRequest struct {
	Model          string  // Target model or voice backend from the JSON body.
	Input          string  // Text to synthesize.
	Voice          string  // Selected voice identifier.
	Instructions   string  // Optional delivery instructions.
	ResponseFormat string  // Audio container format, such as mp3 or wav.
	StreamFormat   string  // Delivery mode: audio or sse.
	Speed          float64 // Playback speed multiplier between 0.25 and 4.
}

// Speech is the complete audio returned by a Speaker. The handler applies the
// configured output limit before writing it to the client.
type Speech struct {
	Data     []byte // Encoded audio bytes.
	MIMEType string // Declared content type when known.
	Format   string // Container format when MIMEType is empty.
}
