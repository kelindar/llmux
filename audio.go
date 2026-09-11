package llmux

import (
	"cmp"
	"encoding/base64"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
	"unicode/utf8"

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
	object, err := decodeObject(body)
	if err != nil {
		writeProtocolError(w, protocolChat, fmtError("body", "request body must be valid JSON", err))
		return
	}
	request, err := parseSpeechRequest(object)
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

func parseSpeechRequest(object map[string]jsontext.Value) (audio.SpeechRequest, error) {
	allowed := map[string]bool{
		"model": true, "input": true, "voice": true, "instructions": true,
		"response_format": true, "speed": true, "stream_format": true,
	}
	if err := internalwire.RejectUnknownStrict(object, allowed); err != nil {
		return audio.SpeechRequest{}, err
	}
	model, err := internalwire.RequireString(object, "model")
	if err != nil {
		return audio.SpeechRequest{}, err
	}
	input, err := internalwire.RequireString(object, "input")
	if err != nil {
		return audio.SpeechRequest{}, err
	}
	if utf8.RuneCountInString(input) > 4096 {
		return audio.SpeechRequest{}, chat.Invalid("input", "input exceeds the 4096 character limit")
	}
	voice, err := parseSpeechVoice(object["voice"])
	if err != nil {
		return audio.SpeechRequest{}, err
	}
	if voice == "" {
		return audio.SpeechRequest{}, chat.Invalid("voice", "voice is required")
	}
	request := audio.SpeechRequest{Model: model, Input: input, Voice: voice, Speed: 1, ResponseFormat: "mp3", StreamFormat: "audio"}
	if value, ok, err := decodeString(object, "instructions"); err != nil {
		return audio.SpeechRequest{}, err
	} else if ok {
		request.Instructions = value
	}
	if value, ok, err := decodeString(object, "response_format"); err != nil {
		return audio.SpeechRequest{}, err
	} else if ok {
		request.ResponseFormat = strings.ToLower(value)
	}
	if !validSpeechFormat(request.ResponseFormat) {
		return audio.SpeechRequest{}, chat.Unsupported("response_format", "supported formats are mp3, opus, aac, flac, wav, and pcm")
	}
	if value, err := decodeFloat(object, "speed"); err != nil {
		return audio.SpeechRequest{}, err
	} else if value != nil {
		if *value < 0.25 || *value > 4 {
			return audio.SpeechRequest{}, chat.Invalid("speed", "speed must be between 0.25 and 4")
		}
		request.Speed = *value
	}
	if value, ok, err := decodeString(object, "stream_format"); err != nil {
		return audio.SpeechRequest{}, err
	} else if ok {
		request.StreamFormat = strings.ToLower(value)
	}
	switch request.StreamFormat {
	case "audio", "sse":
	default:
		return audio.SpeechRequest{}, chat.Unsupported("stream_format", "supported stream formats are audio and sse")
	}
	return request, nil
}

func parseSpeechVoice(raw jsontext.Value) (string, error) {
	if len(raw) == 0 {
		return "", chat.Invalid("voice", "voice is required")
	}
	var voice string
	if err := json.Unmarshal(raw, &voice); err == nil {
		return strings.TrimSpace(voice), nil
	}
	object, err := internalwire.RawObject(raw, "voice")
	if err != nil {
		return "", chat.Invalid("voice", "voice must be a string or an object with an id")
	}
	if err := internalwire.RejectUnknownStrict(object, map[string]bool{"id": true}); err != nil {
		return "", err
	}
	return internalwire.RequireString(object, "id")
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
