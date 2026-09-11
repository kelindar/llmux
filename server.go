package llmux

import (
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/mcp"
)

// Option configures Handler. Options are applied once by New.
type Option func(*Handler)

// Handler exposes configured agents through standard HTTP endpoints. It does
// not create a listener and is safe for concurrent requests.
type Handler struct {
	resolver     chat.Resolver
	catalog      []chat.Model
	limits       chat.Limits
	assets       chat.AssetResolver
	continuation chat.ContinuationStore
	lifecycle    chat.Lifecycle
	storeDefault bool
	transcriber  Transcriber
	speaker      Speaker
	errorLog     func(context.Context, error)
	mcp          *mcp.Transport
}

// New builds a Handler with the given resolver and options.
func New(resolver chat.Resolver, options ...Option) *Handler {
	h := &Handler{resolver: resolver, limits: chat.DefaultLimits()}
	for _, option := range options {
		if option != nil {
			option(h)
		}
	}
	h.limits = h.limits.Normalize()
	return h
}

// WithModels registers catalog entries for GET /models.
func WithModels(models ...chat.Model) Option {
	return func(h *Handler) {
		h.catalog = append([]chat.Model(nil), models...)
	}
}

// WithLimits sets request, media, and output size limits for the handler.
func WithLimits(limits chat.Limits) Option { return func(h *Handler) { h.limits = limits } }

// WithAssetResolver enables resolution of application-owned media references.
func WithAssetResolver(resolver chat.AssetResolver) Option {
	return func(h *Handler) { h.assets = resolver }
}

// WithContinuationStore enables previous_response_id history loading.
// Persistence of new turns is owned by Lifecycle.
func WithContinuationStore(store chat.ContinuationStore) Option {
	return func(h *Handler) { h.continuation = store }
}

// WithLifecycle enables application-owned response identity, idempotent
// acceptance, and terminal persistence via Acceptance.Finish.
// Pass a Lifecycle function such as store.Accept.
func WithLifecycle(life chat.Lifecycle) Option {
	return func(h *Handler) { h.lifecycle = life }
}

// WithStoreDefault sets the content-retention policy when the request omits
// store. The zero option default is false (current behavior). Explicit
// store:true or store:false always overrides this default. Effective
// retention (Retain) requires a configured Lifecycle.
func WithStoreDefault(retain bool) Option {
	return func(h *Handler) { h.storeDefault = retain }
}

// WithTranscriber enables POST /audio/transcriptions.
func WithTranscriber(transcriber Transcriber) Option {
	return func(h *Handler) { h.transcriber = transcriber }
}

// WithSpeaker enables POST /audio/speech.
func WithSpeaker(speaker Speaker) Option { return func(h *Handler) { h.speaker = speaker } }

// WithErrorLog lets an application observe operational failures without
// exposing raw backend or prompt data to clients. The hook is not called for
// ordinary client validation errors.
func WithErrorLog(logf func(context.Context, error)) Option {
	return func(h *Handler) { h.errorLog = logf }
}

// ServeHTTP routes supported protocol and audio endpoints at exact paths.
// Mount under an application prefix with http.StripPrefix; authentication
// remains outside llmux.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/chat/completions":
		if r.Method != http.MethodPost {
			writeProtocolError(w, protocolChat, methodError(r.Method))
			return
		}
		h.serveChat(w, r)
	case "/responses":
		if r.Method != http.MethodPost {
			writeProtocolError(w, protocolResponses, methodError(r.Method))
			return
		}
		h.serveResponses(w, r)
	case "/messages":
		if r.Method != http.MethodPost {
			writeProtocolError(w, protocolAnthropic, methodError(r.Method))
			return
		}
		h.serveMessages(w, r)
	case "/audio/transcriptions":
		if r.Method != http.MethodPost {
			writeProtocolError(w, protocolChat, methodError(r.Method))
			return
		}
		h.serveTranscription(w, r)
	case "/audio/speech":
		if r.Method != http.MethodPost {
			writeProtocolError(w, protocolChat, methodError(r.Method))
			return
		}
		h.serveSpeech(w, r)
	case "/models":
		if r.Method != http.MethodGet {
			writeProtocolError(w, protocolChat, methodError(r.Method))
			return
		}
		h.serveModels(w, r)
	case "/mcp":
		if h.mcp == nil {
			writeProtocolError(w, protocolChat, notFoundError())
			return
		}
		h.serveMCP(w, r)
	default:
		writeProtocolError(w, protocolChat, notFoundError())
	}
}

func notFoundError() *chat.Error {
	return &chat.Error{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "not_found", Message: "not found"}
}

func methodError(method string) *chat.Error {
	return &chat.Error{Status: http.StatusMethodNotAllowed, Type: "invalid_request_error", Code: "method_not_allowed", Message: "method " + method + " is not allowed"}
}

func (h *Handler) serveModels(w http.ResponseWriter, _ *http.Request) {
	type catalogEntry struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	data := make([]catalogEntry, len(h.catalog))
	for i, model := range h.catalog {
		data[i] = catalogEntry{
			ID:      model.ID,
			Object:  "model",
			Created: model.Created,
			OwnedBy: model.OwnedBy,
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func (h *Handler) readBody(w http.ResponseWriter, r *http.Request, limit int64) ([]byte, error) {
	if limit <= 0 {
		limit = h.limits.MaxRequestBytes
	}
	data, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if err != nil {
		return nil, &chat.Error{Status: http.StatusBadRequest, Type: "invalid_request_error", Code: "body_read_failed", Message: "could not read request body", Err: err}
	}
	if int64(len(data)) > limit {
		return nil, &chat.Error{Status: http.StatusRequestEntityTooLarge, Type: "invalid_request_error", Code: "request_too_large", Message: "request body exceeds the configured limit"}
	}
	return data, nil
}

func decodeObject(data []byte) (map[string]jsontext.Value, error) {
	var object map[string]jsontext.Value
	if err := json.Unmarshal(data, &object); err != nil {
		return nil, err
	}
	if object == nil {
		return nil, errors.New("request body must be a JSON object")
	}
	return object, nil
}

func decodeString(object map[string]jsontext.Value, key string) (string, bool, error) {
	raw, ok := object[key]
	if !ok {
		return "", false, nil
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", true, fmtError(key, "must be a string", err)
	}
	return value, true, nil
}

func decodeBool(object map[string]jsontext.Value, key string) (*bool, error) {
	raw, ok := object[key]
	if !ok {
		return nil, nil
	}
	var value bool
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, fmtError(key, "must be a boolean", err)
	}
	return &value, nil
}

func decodeInt(object map[string]jsontext.Value, key string) (*int, error) {
	raw, ok := object[key]
	if !ok {
		return nil, nil
	}
	var value int
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, fmtError(key, "must be an integer", err)
	}
	return &value, nil
}

func decodeFloat(object map[string]jsontext.Value, key string) (*float64, error) {
	raw, ok := object[key]
	if !ok {
		return nil, nil
	}
	var value float64
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, fmtError(key, "must be a number", err)
	}
	return &value, nil
}

func decodeStringSlice(object map[string]jsontext.Value, key string) ([]string, error) {
	raw, ok := object[key]
	if !ok {
		return nil, nil
	}
	var values []string
	if err := json.Unmarshal(raw, &values); err != nil {
		return nil, fmtError(key, "must be an array of strings", err)
	}
	return values, nil
}

func fmtError(param, message string, err error) *chat.Error {
	return &chat.Error{Status: http.StatusBadRequest, Type: "invalid_request_error", Code: "invalid_request", Param: param, Message: message, Err: err}
}

func (h *Handler) resolve(ctx context.Context, target string) (chat.Agent, chat.Info, error) {
	if h.resolver == nil {
		return nil, chat.Info{}, errors.New("llmux: no agent resolver configured")
	}
	agent, info, err := h.resolver(ctx, target)
	switch {
	case err != nil:
		return nil, chat.Info{}, err
	case agent == nil:
		return nil, chat.Info{}, errors.New("llmux: resolver returned a nil agent")
	}
	return agent, info.Normalize(), nil
}

func (h *Handler) prepareParsed(ctx context.Context, parsed *parsedRequest) error {
	if parsed.Turn == nil {
		parsed.Turn = cloneItems(parsed.Request.Input)
	}
	if parsed.Previous != nil {
		if h.continuation == nil {
			return chat.Unsupported("previous_response_id", "continuation is not configured")
		}
		prior, err := h.continuation.Load(ctx, *parsed.Previous)
		if err != nil {
			return &chat.Error{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "previous_response_not_found", Param: "previous_response_id", Message: "previous response was not found", Err: err}
		}
		input := make([]chat.Item, 0, len(prior)+len(parsed.Turn))
		for _, item := range prior {
			input = append(input, item.Clone())
		}
		input = append(input, cloneItems(parsed.Turn)...)
		parsed.Request.Input = input
	}
	h.applyStorePolicy(parsed)
	switch {
	case parsed.Retain && h.lifecycle == nil:
		return chat.Unsupported("store", "response persistence requires a Lifecycle")
	case h.assets == nil:
		return nil
	}
	count := 0
	for item := range parsed.Request.Input {
		if err := h.resolveItemMedia(ctx, &parsed.Request.Input[item], &count); err != nil {
			return err
		}
	}
	return nil
}

// applyStorePolicy sets Retain from the wire store field and the configured
// default without mutating Store.
func (h *Handler) applyStorePolicy(parsed *parsedRequest) {
	switch {
	case parsed.Store != nil:
		parsed.Retain = *parsed.Store
	default:
		parsed.Retain = h.storeDefault
	}
}

func (h *Handler) resolveItemMedia(ctx context.Context, item *chat.Item, count *int) error {
	resolve := func(part *chat.Part) error {
		if part.Media == nil || (part.Media.URL == "" && part.Media.Ref == "") {
			return nil
		}
		(*count)++
		if *count > h.limits.MaxAssets {
			return fmtError("input", "too many media assets", nil)
		}
		media, err := h.assets(ctx, *part.Media, h.limits.MaxMediaBytes)
		if err != nil {
			return &chat.Error{Status: http.StatusBadRequest, Type: "invalid_request_error", Code: "asset_resolution_failed", Param: "input", Message: "could not resolve media asset", Err: err}
		}
		part.Media = &media
		return nil
	}
	for n := range item.Content {
		if err := resolve(&item.Content[n]); err != nil {
			return err
		}
	}
	for n := range item.Output {
		if err := resolve(&item.Output[n]); err != nil {
			return err
		}
	}
	return nil
}

func (h *Handler) validateParsed(parsed *parsedRequest, caps chat.Info) error {
	req := &parsed.Request
	if strings.TrimSpace(req.Target) == "" {
		return chat.Invalid("model", "model is required")
	}
	caps = caps.Normalize()
	switch {
	case req.Controls.MaxOutputTokens != nil && !caps.GenerationControls.Has(chat.ControlMaxOutputTokens):
		return chat.Unsupported("max_output_tokens", "selected agent does not support max output tokens")
	case req.Controls.Temperature != nil && !caps.GenerationControls.Has(chat.ControlTemperature):
		return chat.Unsupported("temperature", "selected agent does not support temperature")
	case req.Controls.TopP != nil && !caps.GenerationControls.Has(chat.ControlTopP):
		return chat.Unsupported("top_p", "selected agent does not support top_p")
	case req.Controls.Stop != nil && !caps.GenerationControls.Has(chat.ControlStop):
		return chat.Unsupported("stop", "selected agent does not support stop sequences")
	case req.Controls.ParallelToolCall != nil && !caps.GenerationControls.Has(chat.ControlParallelToolCalls):
		return chat.Unsupported("parallel_tool_calls", "selected agent does not support parallel tool calls")
	case parsed.Previous != nil && !caps.Continuation:
		return chat.Unsupported("previous_response_id", "selected agent does not support continuation")
	case parsed.Store != nil && *parsed.Store && !caps.Continuation:
		return chat.Unsupported("store", "selected agent does not support continuation")
	}
	assets := 0
	checkPart := func(part chat.Part) error {
		if part.Media != nil {
			assets++
			if assets > h.limits.MaxAssets {
				return chat.Invalid("input", "too many media assets")
			}
		}
		switch {
		case part.Type == chat.PartImage && !caps.InputModalities.Has(chat.ModalityImage):
			return chat.Unsupported("input", "selected agent does not accept image input")
		case part.Type == chat.PartAudio && !caps.InputModalities.Has(chat.ModalityAudio):
			return chat.Unsupported("input", "selected agent does not accept audio input")
		case part.Type == chat.PartFile && !caps.InputModalities.Has(chat.ModalityFile):
			return chat.Unsupported("input", "selected agent does not accept file input")
		}
		return nil
	}
	for _, item := range req.Input {
		if err := item.Validate(h.limits.MaxMediaBytes, false); err != nil {
			return chat.Invalid("input", err.Error())
		}
		for _, part := range item.Content {
			if err := checkPart(part); err != nil {
				return err
			}
		}
		for _, part := range item.Output {
			if err := checkPart(part); err != nil {
				return err
			}
		}
	}
	if len(req.Controls.Tools) > 0 {
		if !caps.Tools {
			return chat.Unsupported("tools", "selected agent does not accept tools")
		}
		for _, tool := range req.Controls.Tools {
			if err := tool.Validate(); err != nil {
				return chat.Invalid("tools", err.Error())
			}
		}
	}
	if req.Controls.ToolChoice != nil {
		if err := req.Controls.ToolChoice.Validate(); err != nil {
			return chat.Invalid("tool_choice", err.Error())
		}
		if len(req.Controls.Tools) == 0 {
			switch req.Controls.ToolChoice.Mode {
			case "auto", "none":
			default:
				return chat.Invalid("tool_choice", "tool_choice requires tools")
			}
		} else if req.Controls.ToolChoice.Mode == "function" {
			found := false
			for _, tool := range req.Controls.Tools {
				if tool.Name == req.Controls.ToolChoice.Name {
					found = true
					break
				}
			}
			if !found {
				return chat.Invalid("tool_choice", "selected function is not declared in tools")
			}
		}
	}
	if req.Output.Format.IsStructured() {
		if !caps.StructuredOutput {
			return chat.Unsupported("text.format", "selected agent does not support structured output")
		}
		if err := req.Output.Format.Validate(); err != nil {
			return chat.Invalid("text.format", err.Error())
		}
	}
	if req.Controls.Reasoning != nil {
		if err := req.Controls.Reasoning.Validate(); err != nil {
			return chat.Invalid("reasoning", err.Error())
		}
		switch {
		case !caps.GenerationControls.Has(chat.ControlReasoning):
			return chat.Unsupported("reasoning", "selected agent does not support reasoning controls")
		case req.Controls.Reasoning.Summary && !caps.ReasoningSummary:
			return chat.Unsupported("reasoning.summary", "selected agent does not provide reasoning summaries")
		}
	}
	if req.Controls.Audio != nil {
		if err := req.Controls.Audio.Validate(); err != nil {
			return chat.Invalid("audio", err.Error())
		}
		if !caps.GenerationControls.Has(chat.ControlAudio) {
			return chat.Unsupported("audio", "selected agent does not support audio controls")
		}
	}
	for key := range req.Controls.Extensions {
		if !caps.Extensions[key] {
			return chat.Unsupported(key, "selected agent does not support this extension")
		}
	}
	if req.Output.Modalities == 0 {
		req.Output.Modalities = chat.ModalityText
	}
	switch {
	case req.Controls.ImageGeneration && !caps.ImageGeneration:
		return chat.Unsupported("tools", "selected agent does not support image generation")
	case !caps.OutputModalities.Has(req.Output.Modalities):
		return chat.Unsupported("modalities", "selected agent does not provide the requested output modalities")
	case req.Output.Format.Kind == chat.FormatJSONSchema && len(req.Output.Format.Schema) > 0 && !req.Output.Format.Schema.IsValid():
		return chat.Invalid("response_format", "schema must be valid JSON")
	}
	return nil
}

func requiresImageGeneration(event chat.Event) bool {
	return event.Type == chat.EventItem &&
		event.Item.Type == chat.ItemMedia &&
		len(event.Item.Content) == 1 &&
		event.Item.Content[0].Type == chat.PartImage
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	data, err := json.Marshal(value)
	if err != nil {
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(append(data, '\n'))
}
