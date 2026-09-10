package llmux

import (
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"io"
	"net/http"
	"strings"
)

// Model is one optional catalog entry returned by GET /v1/models.
type Model struct {
	ID      string `json:"id"`       // Model identifier returned to clients.
	Object  string `json:"object"`   // Object type; defaults to "model" when empty.
	Created int64  `json:"created"`  // Unix creation time in seconds.
	OwnedBy string `json:"owned_by"` // Owner label shown in the models list.
}

// Option configures Handler. Options are applied once by New.
type Option func(*Handler)

// Handler exposes configured agents through standard HTTP endpoints. It does
// not create a listener and is safe for concurrent requests.
type Handler struct {
	resolver     Resolver
	catalog      []Model
	limits       Limits
	assets       AssetResolver
	continuation ContinuationStore
	lifecycle    Lifecycle
	storeDefault bool
	transcriber  Transcriber
	speaker      Speaker
	errorLog     func(context.Context, error)
}

// New builds a Handler with the given resolver and options.
func New(resolver Resolver, options ...Option) *Handler {
	h := &Handler{resolver: resolver, limits: DefaultLimits()}
	for _, option := range options {
		if option != nil {
			option(h)
		}
	}
	h.limits = h.limits.Normalize()
	return h
}

// NewHandler is an explicit spelling for callers that prefer constructor
// names which describe the returned value.
func NewHandler(resolver Resolver, options ...Option) *Handler { return New(resolver, options...) }

// WithModels registers catalog entries for GET /v1/models.
func WithModels(models ...Model) Option {
	return func(h *Handler) {
		h.catalog = append([]Model(nil), models...)
	}
}

// WithLimits sets request, media, and output size limits for the handler.
func WithLimits(limits Limits) Option { return func(h *Handler) { h.limits = limits } }

// WithAssetResolver enables resolution of application-owned media references.
func WithAssetResolver(resolver AssetResolver) Option {
	return func(h *Handler) { h.assets = resolver }
}

// WithContinuationStore enables previous_response_id history loading.
// Persistence of new turns is owned by Lifecycle.
func WithContinuationStore(store ContinuationStore) Option {
	return func(h *Handler) { h.continuation = store }
}

// WithLifecycle enables application-owned response identity, idempotent
// acceptance, and terminal persistence.
func WithLifecycle(life Lifecycle) Option {
	return func(h *Handler) { h.lifecycle = life }
}

// WithStoreDefault sets the content-retention policy when the request omits
// store. The zero option default is false (current behavior). Explicit
// store:true or store:false always overrides this default. Effective
// retention (Retain) requires a configured Lifecycle.
func WithStoreDefault(retain bool) Option {
	return func(h *Handler) { h.storeDefault = retain }
}

// WithTranscriber enables POST /v1/audio/transcriptions.
func WithTranscriber(transcriber Transcriber) Option {
	return func(h *Handler) { h.transcriber = transcriber }
}

// WithSpeaker enables POST /v1/audio/speech.
func WithSpeaker(speaker Speaker) Option { return func(h *Handler) { h.speaker = speaker } }

// WithErrorLog lets an application observe operational failures without
// exposing raw backend or prompt data to clients. The hook is not called for
// ordinary client validation errors.
func WithErrorLog(logf func(context.Context, error)) Option {
	return func(h *Handler) { h.errorLog = logf }
}

// ServeHTTP routes supported protocol and audio endpoints. Paths may be mounted
// under an application prefix (for example /api/v1/responses) without changing
// protocol parsing; authentication remains outside llmux.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch routePath(r.URL.Path) {
	case "/v1/chat/completions":
		if r.Method != http.MethodPost {
			writeProtocolError(w, protocolChat, methodError(r.Method))
			return
		}
		h.serveChat(w, r)
	case "/v1/responses":
		if r.Method != http.MethodPost {
			writeProtocolError(w, protocolResponses, methodError(r.Method))
			return
		}
		h.serveResponses(w, r)
	case "/v1/messages":
		if r.Method != http.MethodPost {
			writeProtocolError(w, protocolAnthropic, methodError(r.Method))
			return
		}
		h.serveMessages(w, r)
	case "/v1/audio/transcriptions":
		if r.Method != http.MethodPost {
			writeProtocolError(w, protocolChat, methodError(r.Method))
			return
		}
		h.serveTranscription(w, r)
	case "/v1/audio/speech":
		if r.Method != http.MethodPost {
			writeProtocolError(w, protocolChat, methodError(r.Method))
			return
		}
		h.serveSpeech(w, r)
	case "/v1/models":
		if r.Method != http.MethodGet {
			writeProtocolError(w, protocolChat, methodError(r.Method))
			return
		}
		h.serveModels(w, r)
	default:
		writeProtocolError(w, protocolChat, &APIError{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "not_found", Message: "not found"})
	}
}

func routePath(path string) string {
	endpoints := []string{
		"/v1/chat/completions",
		"/v1/responses",
		"/v1/messages",
		"/v1/audio/transcriptions",
		"/v1/audio/speech",
		"/v1/models",
	}
	for _, ep := range endpoints {
		// Allow mounts under an application prefix (for example /api/v1/responses).
		// Endpoints begin with '/', so a suffix match already enforces a path boundary.
		if path == ep || strings.HasSuffix(path, ep) {
			return ep
		}
	}
	return path
}

func methodError(method string) *APIError {
	return &APIError{Status: http.StatusMethodNotAllowed, Type: "invalid_request_error", Code: "method_not_allowed", Message: "method " + method + " is not allowed"}
}

func (h *Handler) serveModels(w http.ResponseWriter, _ *http.Request) {
	models := make([]Model, len(h.catalog))
	copy(models, h.catalog)
	for i := range models {
		if models[i].Object == "" {
			models[i].Object = "model"
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": models})
}

func (h *Handler) readBody(w http.ResponseWriter, r *http.Request, limit int64) ([]byte, error) {
	if limit <= 0 {
		limit = h.limits.MaxRequestBytes
	}
	data, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if err != nil {
		return nil, &APIError{Status: http.StatusBadRequest, Type: "invalid_request_error", Code: "body_read_failed", Message: "could not read request body", Err: err}
	}
	if int64(len(data)) > limit {
		return nil, &APIError{Status: http.StatusRequestEntityTooLarge, Type: "invalid_request_error", Code: "request_too_large", Message: "request body exceeds the configured limit"}
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

func fmtError(param, message string, err error) *APIError {
	return &APIError{Status: http.StatusBadRequest, Type: "invalid_request_error", Code: "invalid_request", Param: param, Message: message, Err: err}
}

func (h *Handler) resolve(ctx context.Context, target string) (Agent, Capabilities, error) {
	if h.resolver == nil {
		return nil, Capabilities{}, errors.New("llmux: no agent resolver configured")
	}
	agent, capabilities, err := h.resolver.Resolve(ctx, target)
	if err != nil {
		return nil, Capabilities{}, err
	}
	if agent == nil {
		return nil, Capabilities{}, errors.New("llmux: resolver returned a nil agent")
	}
	return agent, capabilities.Normalize(), nil
}

func (h *Handler) prepareRequest(ctx context.Context, req *Request) error {
	if req.Turn == nil {
		req.Turn = cloneItems(req.Input)
	}
	if req.Controls.PreviousResponseID != nil {
		if h.continuation == nil {
			return Unsupported("previous_response_id", "continuation is not configured")
		}
		prior, err := h.continuation.Load(ctx, *req.Controls.PreviousResponseID)
		if err != nil {
			return &APIError{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "previous_response_not_found", Param: "previous_response_id", Message: "previous response was not found", Err: err}
		}
		input := make([]Item, 0, len(prior)+len(req.Turn))
		for _, item := range prior {
			input = append(input, item.Clone())
		}
		input = append(input, cloneItems(req.Turn)...)
		req.Input = input
	}
	h.applyStorePolicy(req)
	if req.Retain && h.lifecycle == nil {
		return Unsupported("store", "response persistence requires a Lifecycle")
	}
	if h.assets == nil {
		return nil
	}
	count := 0
	for item := range req.Input {
		if err := h.resolveItemMedia(ctx, &req.Input[item], &count); err != nil {
			return err
		}
	}
	return nil
}

// applyStorePolicy sets Request.Retain from the wire store field and the
// configured default without mutating Controls.Store.
func (h *Handler) applyStorePolicy(req *Request) {
	switch {
	case req.Controls.Store != nil:
		req.Retain = *req.Controls.Store
	default:
		req.Retain = h.storeDefault
	}
}

func (h *Handler) resolveItemMedia(ctx context.Context, item *Item, count *int) error {
	resolve := func(part *Part) error {
		if part.Media == nil || (part.Media.URL == "" && part.Media.Ref == "") {
			return nil
		}
		(*count)++
		if *count > h.limits.MaxAssets {
			return fmtError("input", "too many media assets", nil)
		}
		media, err := h.assets.Resolve(ctx, *part.Media, h.limits.MaxMediaBytes)
		if err != nil {
			return &APIError{Status: http.StatusBadRequest, Type: "invalid_request_error", Code: "asset_resolution_failed", Param: "input", Message: "could not resolve media asset", Err: err}
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

func (h *Handler) validateRequest(req *Request, caps Capabilities) error {
	if strings.TrimSpace(req.Target) == "" {
		return Invalid("model", "model is required")
	}
	caps = caps.Normalize()
	switch {
	case req.Controls.MaxOutputTokens != nil && !caps.GenerationControls.Has(ControlMaxOutputTokens):
		return Unsupported("max_output_tokens", "selected agent does not support max output tokens")
	case req.Controls.Temperature != nil && !caps.GenerationControls.Has(ControlTemperature):
		return Unsupported("temperature", "selected agent does not support temperature")
	case req.Controls.TopP != nil && !caps.GenerationControls.Has(ControlTopP):
		return Unsupported("top_p", "selected agent does not support top_p")
	case req.Controls.Stop != nil && !caps.GenerationControls.Has(ControlStop):
		return Unsupported("stop", "selected agent does not support stop sequences")
	case req.Controls.ParallelToolCall != nil && !caps.GenerationControls.Has(ControlParallelToolCalls):
		return Unsupported("parallel_tool_calls", "selected agent does not support parallel tool calls")
	case req.Controls.PreviousResponseID != nil && !caps.Continuation:
		return Unsupported("previous_response_id", "selected agent does not support continuation")
	case req.Controls.Store != nil && *req.Controls.Store && !caps.Continuation:
		return Unsupported("store", "selected agent does not support continuation")
	}
	assets := 0
	checkPart := func(part Part) error {
		if part.Media != nil {
			assets++
			if assets > h.limits.MaxAssets {
				return Invalid("input", "too many media assets")
			}
		}
		switch part.Type {
		case PartImage:
			if !caps.InputModalities.Has(ModalityImage) {
				return Unsupported("input", "selected agent does not accept image input")
			}
		case PartAudio:
			if !caps.InputModalities.Has(ModalityAudio) {
				return Unsupported("input", "selected agent does not accept audio input")
			}
		case PartFile:
			if !caps.InputModalities.Has(ModalityFile) {
				return Unsupported("input", "selected agent does not accept file input")
			}
		}
		return nil
	}
	for _, item := range req.Input {
		if err := item.Validate(h.limits.MaxMediaBytes, false); err != nil {
			return Invalid("input", err.Error())
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
			return Unsupported("tools", "selected agent does not accept tools")
		}
		for _, tool := range req.Controls.Tools {
			if err := tool.Validate(); err != nil {
				return Invalid("tools", err.Error())
			}
		}
	}
	if req.Controls.ToolChoice != nil {
		if err := req.Controls.ToolChoice.Validate(); err != nil {
			return Invalid("tool_choice", err.Error())
		}
		if len(req.Controls.Tools) == 0 {
			switch req.Controls.ToolChoice.Mode {
			case "auto", "none":
			default:
				return Invalid("tool_choice", "tool_choice requires tools")
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
				return Invalid("tool_choice", "selected function is not declared in tools")
			}
		}
	}
	if req.Controls.Structured != nil {
		if !caps.StructuredOutput {
			return Unsupported("text.format", "selected agent does not support structured output")
		}
		if err := req.Controls.Structured.Validate(); err != nil {
			return Invalid("text.format", err.Error())
		}
	}
	if req.Controls.Reasoning != nil {
		if err := req.Controls.Reasoning.Validate(); err != nil {
			return Invalid("reasoning", err.Error())
		}
		switch {
		case !caps.GenerationControls.Has(ControlReasoning):
			return Unsupported("reasoning", "selected agent does not support reasoning controls")
		case req.Controls.Reasoning.Summary && !caps.ReasoningSummary:
			return Unsupported("reasoning.summary", "selected agent does not provide reasoning summaries")
		}
	}
	if req.Controls.Audio != nil {
		if err := req.Controls.Audio.Validate(); err != nil {
			return Invalid("audio", err.Error())
		}
		if !caps.GenerationControls.Has(ControlAudio) {
			return Unsupported("audio", "selected agent does not support audio controls")
		}
	}
	for key := range req.Controls.Extensions {
		if !caps.Extensions[key] {
			return Unsupported(key, "selected agent does not support this extension")
		}
	}
	if req.Output.Modalities == 0 {
		req.Output.Modalities = ModalityText
	}
	switch {
	case req.Controls.ImageGeneration && !caps.ImageGeneration:
		return Unsupported("tools", "selected agent does not support image generation")
	case !caps.OutputModalities.Has(req.Output.Modalities):
		return Unsupported("modalities", "selected agent does not provide the requested output modalities")
	case len(req.Output.Schema) > 0 && !req.Output.Schema.IsValid():
		return Invalid("response_format", "schema must be valid JSON")
	}
	return nil
}

func validateOutputEvent(event Event, caps Capabilities) error {
	caps = caps.Normalize()
	checkPart := func(part Part) error {
		switch part.Type {
		case PartText:
			if !caps.OutputModalities.Has(ModalityText) {
				return Unsupported("output", "selected agent does not provide text output")
			}
		case PartImage:
			if !caps.OutputModalities.Has(ModalityImage) {
				return Unsupported("output", "selected agent does not provide image output")
			}
		case PartAudio:
			if !caps.OutputModalities.Has(ModalityAudio) {
				return Unsupported("output", "selected agent does not provide audio output")
			}
		case PartFile:
			if !caps.OutputModalities.Has(ModalityFile) {
				return Unsupported("output", "selected agent does not provide file output")
			}
		}
		return nil
	}
	checkItem := func(item Item) error {
		switch item.Type {
		case ItemMessage, ItemMedia:
			for _, part := range item.Content {
				if err := checkPart(part); err != nil {
					return err
				}
			}
		case ItemFunctionCall:
			if !caps.ClientTools {
				return Unsupported("tools", "selected agent does not hand tool calls to the client")
			}
		case ItemFunctionCallOutput:
			for _, part := range item.Output {
				if err := checkPart(part); err != nil {
					return err
				}
			}
		case ItemReasoning:
			if !caps.ReasoningSummary {
				return Unsupported("output", "selected agent does not provide reasoning summaries")
			}
		}
		return nil
	}
	switch event.Type {
	case EventMessage, EventMedia, EventItem, EventReasoning:
		if event.Type == EventMedia {
			return checkPart(event.Part)
		}
		return checkItem(event.Item)
	case EventTextDelta:
		if !caps.OutputModalities.Has(ModalityText) {
			return Unsupported("output", "selected agent does not provide text output")
		}
	case EventToolCall, EventToolCallStart, EventToolCallDelta, EventToolCallDone:
		if !caps.ClientTools {
			return Unsupported("tools", "selected agent does not hand tool calls to the client")
		}
	}
	return nil
}

func requiresImageGeneration(event Event) bool {
	if event.Type == EventMedia {
		return event.Part.Type == PartImage
	}
	if event.Type == EventItem {
		return event.Item.Type == ItemMedia && len(event.Item.Content) == 1 && event.Item.Content[0].Type == PartImage
	}
	return false
}

func requiresReasoningSummary(event Event) bool {
	if event.Type == EventReasoning {
		return true
	}
	return event.Type == EventItem && event.Item.Type == ItemReasoning
}

func validateRequestedOutput(event Event, modalities Modality) error {
	checkPart := func(part Part) error {
		var modality Modality
		switch part.Type {
		case PartText:
			modality = ModalityText
		case PartImage:
			modality = ModalityImage
		case PartAudio:
			modality = ModalityAudio
		case PartFile:
			modality = ModalityFile
		default:
			return nil
		}
		if !modalities.Has(modality) {
			return Unsupported("modalities", "agent emitted an output modality that was not requested")
		}
		return nil
	}
	checkItem := func(item Item) error {
		switch item.Type {
		case ItemMessage, ItemMedia:
			for _, part := range item.Content {
				if err := checkPart(part); err != nil {
					return err
				}
			}
		case ItemFunctionCallOutput:
			for _, part := range item.Output {
				if err := checkPart(part); err != nil {
					return err
				}
			}
		}
		return nil
	}
	switch event.Type {
	case EventMessage, EventItem:
		return checkItem(event.Item)
	case EventMedia:
		return checkPart(event.Part)
	case EventTextDelta:
		return checkPart(TextPart(event.Delta))
	}
	return nil
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
