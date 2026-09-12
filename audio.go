// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package llmux

import (
	"bytes"
	"cmp"
	"encoding/base64"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/buger/jsonparser"
	"github.com/kelindar/llmux/audio"
	"github.com/kelindar/llmux/chat"
	internalwire "github.com/kelindar/llmux/internal/wire"
)

func (h *Handler) serveTranscription(w http.ResponseWriter, r *http.Request) {
	switch {
	case h.transcriber == nil:
		writeProtocolError(w, protocolChat, &chat.Error{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "not_found", Message: "audio transcription is not configured"})
		return
	case !strings.HasPrefix(strings.ToLower(r.Header.Get("Content-Type")), "multipart/form-data;"):
		writeProtocolError(w, protocolChat, chat.Invalid("content_type", "audio transcription requires multipart/form-data"))
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, h.limits.MaxMultipartBytes)
	if err := r.ParseMultipartForm(h.limits.MaxMultipartBytes); err != nil {
		status := http.StatusBadRequest
		code := "invalid_multipart"
		message := "multipart request is malformed"
		if _, ok := errors.AsType[*http.MaxBytesError](err); ok {
			status = http.StatusRequestEntityTooLarge
			code = "request_too_large"
			message = "multipart request exceeds the configured limit"
		}
		writeProtocolError(w, protocolChat, &chat.Error{Status: status, Type: "invalid_request_error", Code: code, Message: message, Err: err})
		return
	}
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}
	if err := rejectMultipartFields(r.MultipartForm); err != nil {
		writeProtocolError(w, protocolChat, err)
		return
	}

	model := strings.TrimSpace(r.FormValue("model"))
	if model == "" {
		writeProtocolError(w, protocolChat, chat.Invalid("model", "model is required"))
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		writeProtocolError(w, protocolChat, chat.Invalid("file", "file is required"))
		return
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, h.limits.MaxMediaBytes+1))
	switch {
	case err != nil:
		writeProtocolError(w, protocolChat, &chat.Error{Status: http.StatusBadRequest, Type: "invalid_request_error", Code: "file_read_failed", Param: "file", Message: "could not read audio file", Err: err})
		return
	case int64(len(data)) > h.limits.MaxMediaBytes:
		writeProtocolError(w, protocolChat, &chat.Error{Status: http.StatusRequestEntityTooLarge, Type: "invalid_request_error", Code: "media_too_large", Param: "file", Message: "audio file exceeds the configured media limit"})
		return
	}

	responseFormat := cmp.Or(strings.ToLower(strings.TrimSpace(r.FormValue("response_format"))), "json")
	switch responseFormat {
	case "json", "text", "verbose_json":
	default:
		writeProtocolError(w, protocolChat, chat.Unsupported("response_format", "only json, text, and verbose_json are supported"))
		return
	}
	temperature, err := parseFormFloat(r, "temperature")
	switch {
	case err != nil:
		writeProtocolError(w, protocolChat, err)
		return
	case temperature != nil && (*temperature < 0 || *temperature > 1):
		writeProtocolError(w, protocolChat, chat.Invalid("temperature", "temperature must be between 0 and 1"))
		return
	}
	request := audio.TranscriptionRequest{
		Model:          model,
		Filename:       header.Filename,
		MIMEType:       header.Header.Get("Content-Type"),
		Data:           data,
		Prompt:         r.FormValue("prompt"),
		Language:       r.FormValue("language"),
		ResponseFormat: responseFormat,
		Temperature:    temperature,
	}
	result, err := h.transcriber.Transcribe(r.Context(), request)
	if err != nil {
		h.logError(r.Context(), err)
		writeProtocolError(w, protocolChat, err)
		return
	}
	switch responseFormat {
	case "text":
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, result.Text)
	case "verbose_json":
		body := map[string]any{
			"task":     "transcribe",
			"language": result.Language,
			"duration": result.Duration,
			"text":     result.Text,
			"segments": result.Segments,
		}
		writeJSON(w, http.StatusOK, body)
	default:
		writeJSON(w, http.StatusOK, map[string]string{"text": result.Text})
	}
}

func rejectMultipartFields(form *multipart.Form) error {
	if form == nil {
		return chat.Invalid("body", "multipart form is required")
	}
	allowed := map[string]bool{
		"model": true, "file": true, "prompt": true, "language": true,
		"response_format": true, "temperature": true,
	}
	for key, values := range form.Value {
		switch {
		case !allowed[key]:
			return chat.Unsupported(key, key+" is not supported")
		case len(values) != 1:
			return chat.Invalid(key, key+" must be provided once")
		}
	}
	for key := range form.File {
		switch key {
		case "file":
		default:
			return chat.Unsupported(key, key+" is not supported")
		}
	}
	if len(form.File["file"]) != 1 {
		return chat.Invalid("file", "file must be provided once")
	}
	return nil
}

func parseFormFloat(r *http.Request, key string) (*float64, error) {
	value := strings.TrimSpace(r.FormValue(key))
	if value == "" {
		return nil, nil
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return nil, fmtError(key, "must be a number", err)
	}
	return &parsed, nil
}

func (h *Handler) serveSpeech(w http.ResponseWriter, r *http.Request) {
	if h.speaker == nil {
		writeProtocolError(w, protocolChat, &chat.Error{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "not_found", Message: "speech generation is not configured"})
		return
	}
	body, err := h.readBody(w, r, h.limits.MaxRequestBytes)
	if err != nil {
		writeProtocolError(w, protocolChat, err)
		return
	}
	request, err := parseSpeechRequest(body)
	if err != nil {
		writeProtocolError(w, protocolChat, err)
		return
	}
	result, err := h.speaker.Speak(r.Context(), request)
	if err != nil {
		h.logError(r.Context(), err)
		writeProtocolError(w, protocolChat, err)
		return
	}
	switch {
	case len(result.Data) == 0:
		writeProtocolError(w, protocolChat, &chat.Error{Status: http.StatusInternalServerError, Type: "server_error", Code: "empty_audio", Message: "speech service returned no audio"})
		return
	case int64(len(result.Data)) > h.limits.MaxOutputBytes:
		writeProtocolError(w, protocolChat, &chat.Error{Status: http.StatusRequestEntityTooLarge, Type: "invalid_request_error", Code: "output_too_large", Message: "speech output exceeds the configured limit"})
		return
	}
	if request.StreamFormat == "sse" {
		stream := &sseWriter{w: w, limits: h.limits}
		if err := writeSpeechStream(r, result, stream); err != nil {
			h.logError(r.Context(), err)
			if !stream.Started() {
				writeProtocolError(w, protocolChat, err)
			}
		}
		return
	}
	mime := result.MIMEType
	if mime == "" {
		mime = speechMIME(result.Format)
	}
	if mime == "" {
		mime = speechMIME(request.ResponseFormat)
	}
	if mime == "" {
		mime = "application/octet-stream"
	}
	w.Header().Set("Content-Type", mime)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(result.Data); err != nil {
		h.logError(r.Context(), err)
	}
}

type speechDecoder struct {
	model          internalwire.Value
	input          internalwire.Value
	voice          internalwire.Value
	instructions   internalwire.Value
	responseFormat internalwire.Value
	speed          internalwire.Value
	streamFormat   internalwire.Value
}

func (d *speechDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	value := internalwire.Value{Raw: raw, Type: typ}
	switch {
	case bytes.Equal(key, []byte("model")):
		d.model = value
	case bytes.Equal(key, []byte("input")):
		d.input = value
	case bytes.Equal(key, []byte("voice")):
		d.voice = value
	case bytes.Equal(key, []byte("instructions")):
		d.instructions = value
	case bytes.Equal(key, []byte("response_format")):
		d.responseFormat = value
	case bytes.Equal(key, []byte("speed")):
		d.speed = value
	case bytes.Equal(key, []byte("stream_format")):
		d.streamFormat = value
	default:
		return chat.Unsupported(string(key), "request field is not supported by llmux")
	}
	return nil
}

func parseSpeechRequest(data []byte) (audio.SpeechRequest, error) {
	if err := internalwire.ValidateObject(data); err != nil {
		return audio.SpeechRequest{}, fmtError("body", "request body must be valid JSON", err)
	}
	var decoder speechDecoder
	if err := jsonparser.ObjectEach(data, decoder.field); err != nil {
		if _, ok := err.(*chat.Error); ok {
			return audio.SpeechRequest{}, err
		}
		return audio.SpeechRequest{}, fmtError("body", "request body must be valid JSON", err)
	}
	model, err := speechRequiredString(decoder.model, "model")
	if err != nil {
		return audio.SpeechRequest{}, err
	}
	input, err := speechRequiredString(decoder.input, "input")
	if err != nil {
		return audio.SpeechRequest{}, err
	}
	if utf8.RuneCountInString(input) > 4096 {
		return audio.SpeechRequest{}, chat.Invalid("input", "input exceeds the 4096 character limit")
	}
	voice, err := parseSpeechVoice(decoder.voice)
	if err != nil {
		return audio.SpeechRequest{}, err
	}
	if voice == "" {
		return audio.SpeechRequest{}, chat.Invalid("voice", "voice is required")
	}
	request := audio.SpeechRequest{Model: model, Input: input, Voice: voice, Speed: 1, ResponseFormat: "mp3", StreamFormat: "audio"}
	switch value, ok, err := speechString(decoder.instructions, "instructions"); {
	case err != nil:
		return audio.SpeechRequest{}, err
	case ok:
		request.Instructions = value
	}
	switch value, ok, err := speechString(decoder.responseFormat, "response_format"); {
	case err != nil:
		return audio.SpeechRequest{}, err
	case ok:
		request.ResponseFormat = strings.ToLower(value)
	}
	if !validSpeechFormat(request.ResponseFormat) {
		return audio.SpeechRequest{}, chat.Unsupported("response_format", "supported formats are mp3, opus, aac, flac, wav, and pcm")
	}
	switch value, err := speechFloat(decoder.speed, "speed"); {
	case err != nil:
		return audio.SpeechRequest{}, err
	case value != nil:
		if *value < 0.25 || *value > 4 {
			return audio.SpeechRequest{}, chat.Invalid("speed", "speed must be between 0.25 and 4")
		}
		request.Speed = *value
	}
	switch value, ok, err := speechString(decoder.streamFormat, "stream_format"); {
	case err != nil:
		return audio.SpeechRequest{}, err
	case ok:
		request.StreamFormat = strings.ToLower(value)
	}
	switch request.StreamFormat {
	case "audio", "sse":
	default:
		return audio.SpeechRequest{}, chat.Unsupported("stream_format", "supported stream formats are audio and sse")
	}
	return request, nil
}

func parseSpeechVoice(value internalwire.Value) (string, error) {
	if !value.Present() {
		return "", chat.Invalid("voice", "voice is required")
	}
	if value.Type == jsonparser.String || value.Type == jsonparser.Null {
		voice, err := internalwire.String(value.Raw, value.Type)
		if err != nil {
			return "", chat.Invalid("voice", "voice must be a string or an object with an id")
		}
		return strings.TrimSpace(voice), nil
	}
	if value.Type != jsonparser.Object {
		return "", chat.Invalid("voice", "voice must be a string or an object with an id")
	}
	var decoder voiceDecoder
	if err := jsonparser.ObjectEach(value.Raw, decoder.field); err != nil {
		if _, ok := err.(*chat.Error); ok {
			return "", err
		}
		return "", chat.Invalid("voice", "voice must be a string or an object with an id")
	}
	return speechRequiredString(decoder.id, "id")
}

type voiceDecoder struct{ id internalwire.Value }

func (d *voiceDecoder) field(key, raw []byte, typ jsonparser.ValueType, _ int) error {
	if bytes.Equal(key, []byte("id")) {
		d.id = internalwire.Value{Raw: raw, Type: typ}
		return nil
	}
	return chat.Unsupported(string(key), "request field is not supported by llmux")
}

func speechRequiredString(value internalwire.Value, key string) (string, error) {
	decoded, ok, err := speechString(value, key)
	if err != nil {
		return "", err
	}
	if !ok || strings.TrimSpace(decoded) == "" {
		return "", chat.Invalid(key, key+" is required")
	}
	return decoded, nil
}

func speechString(value internalwire.Value, key string) (string, bool, error) {
	if !value.Present() {
		return "", false, nil
	}
	decoded, err := internalwire.String(value.Raw, value.Type)
	if err != nil {
		return "", true, chat.Invalid(key, "must be a string")
	}
	return decoded, true, nil
}

func speechFloat(value internalwire.Value, key string) (*float64, error) {
	if !value.Present() {
		return nil, nil
	}
	decoded, err := internalwire.Float(value.Raw, value.Type)
	if err != nil {
		return nil, chat.Invalid(key, "must be a number")
	}
	return &decoded, nil
}

func validSpeechFormat(format string) bool {
	switch format {
	case "mp3", "opus", "aac", "flac", "wav", "pcm":
		return true
	default:
		return false
	}
}

func speechMIME(format string) string {
	switch strings.ToLower(format) {
	case "mp3":
		return "audio/mpeg"
	case "opus":
		return "audio/opus"
	case "aac":
		return "audio/aac"
	case "flac":
		return "audio/flac"
	case "wav":
		return "audio/wav"
	case "pcm":
		return "audio/pcm"
	default:
		return ""
	}
}

func writeSpeechStream(r *http.Request, speech audio.Speech, stream *sseWriter) error {
	chunkSize := 3072
	for offset := 0; offset < len(speech.Data); offset += chunkSize {
		end := min(offset+chunkSize, len(speech.Data))
		if err := stream.write("", map[string]any{
			"type":  "speech.audio.delta",
			"audio": base64.StdEncoding.EncodeToString(speech.Data[offset:end]),
		}); err != nil {
			return err
		}
		if err := r.Context().Err(); err != nil {
			return err
		}
	}
	return stream.write("", map[string]any{"type": "speech.audio.done"})
}
