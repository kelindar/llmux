package anthropic

import (
	"encoding/base64"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"fmt"
	"net/http"
	"strings"

	"github.com/kelindar/llmux/contract"
	"github.com/kelindar/llmux/internal/execution"
	internalidentity "github.com/kelindar/llmux/internal/identity"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/kelindar/llmux/internal/wire"
)

type parsedRequest = internalprotocol.ParsedRequest
type responseMeta = internalprotocol.Meta
type streamEncoder = internalprotocol.StreamEncoder
type protocol = internalprotocol.Kind

const protocolAnthropic = internalprotocol.Anthropic

func newID(prefix string) string { return internalidentity.New(prefix) }

func rejectUnknown(object map[string]jsontext.Value, allowed map[string]bool) error {
	return wire.RejectUnknown(object, allowed)
}
func rejectUnknownStrict(object map[string]jsontext.Value, allowed map[string]bool) error {
	return wire.RejectUnknownStrict(object, allowed)
}
func namespacedExtensions(object map[string]jsontext.Value, allowed map[string]bool) map[string]jsontext.Value {
	return wire.NamespacedExtensions(object, allowed)
}
func rawObject(raw jsontext.Value, param string) (map[string]jsontext.Value, error) {
	return wire.RawObject(raw, param)
}
func rawArray(raw jsontext.Value, param string) ([]jsontext.Value, error) {
	return wire.RawArray(raw, param)
}
func requireString(object map[string]jsontext.Value, key string) (string, error) {
	return wire.RequireString(object, key)
}
func fmtError(param, message string, err error) *contract.APIError {
	return wire.Error(param, message, err)
}
func validRole(role contract.Role) bool { return contract.ValidRole(role) }
func decodeString(object map[string]jsontext.Value, key string) (string, bool, error) {
	return wire.DecodeString(object, key)
}
func decodeBool(object map[string]jsontext.Value, key string) (*bool, error) {
	return wire.DecodeBool(object, key)
}
func decodeFloat(object map[string]jsontext.Value, key string) (*float64, error) {
	return wire.DecodeFloat(object, key)
}
func decodeInt(object map[string]jsontext.Value, key string) (*int, error) {
	return wire.DecodeInt(object, key)
}
func decodeStringSlice(object map[string]jsontext.Value, key string) ([]string, error) {
	return wire.DecodeStringSlice(object, key)
}
func optionalString(object map[string]jsontext.Value, key string) (string, error) {
	value, _, err := decodeString(object, key)
	return value, err
}

type sseWriter struct {
	w       http.ResponseWriter
	limits  contract.Limits
	inner   *internalprotocol.SSEWriter
	started bool
}

func (s *sseWriter) writer() *internalprotocol.SSEWriter {
	if s.inner == nil {
		s.inner = internalprotocol.NewSSEWriter(s.w, s.limits)
	}
	return s.inner
}

func (s *sseWriter) start() error {
	if err := s.writer().Start(); err != nil {
		return err
	}
	s.started = true
	return nil
}

func (s *sseWriter) write(event string, value any) error {
	if err := s.writer().Write(event, value); err != nil {
		return err
	}
	s.started = true
	return nil
}

func (s *sseWriter) done() error {
	if err := s.writer().Done(); err != nil {
		return err
	}
	s.started = true
	return nil
}

func asAPIError(err error) *contract.APIError { return internalprotocol.AsAPIError(err) }
func anthropicErrorType(err *contract.APIError) string {
	return internalprotocol.AnthropicErrorType(err)
}
func writeProtocolErrorAndReturn(w http.ResponseWriter, p protocol, err error) error {
	internalprotocol.WriteError(w, p, err)
	return err
}

// ParseRequest decodes an Anthropic Messages request into a canonical request.
func ParseRequest(object map[string]jsontext.Value) (parsedRequest, error) {
	allowed := map[string]bool{
		"model": true, "max_tokens": true, "messages": true, "stream": true,
		"system": true, "temperature": true, "top_p": true, "stop_sequences": true,
		"tools": true, "tool_choice": true, "metadata": true, "thinking": true,
		"service_tier": true, "container": true, "output_config": true,
		"mcp_servers": true,
	}
	if err := rejectUnknown(object, allowed); err != nil {
		return parsedRequest{}, err
	}
	target, err := requireString(object, "model")
	if err != nil {
		return parsedRequest{}, err
	}
	maxTokens, err := decodeInt(object, "max_tokens")
	if err != nil {
		return parsedRequest{}, err
	}
	if maxTokens == nil || *maxTokens < 1 {
		return parsedRequest{}, contract.Invalid("max_tokens", "max_tokens must be positive")
	}
	rawMessages, ok := object["messages"]
	if !ok {
		return parsedRequest{}, contract.Invalid("messages", "messages is required")
	}
	messageValues, err := rawArray(rawMessages, "messages")
	if err != nil {
		return parsedRequest{}, err
	}
	if len(messageValues) == 0 {
		return parsedRequest{}, contract.Invalid("messages", "messages must not be empty")
	}
	input := make([]contract.Item, 0, len(messageValues))
	for _, value := range messageValues {
		items, err := parseAnthropicMessage(value)
		if err != nil {
			return parsedRequest{}, err
		}
		input = append(input, items...)
	}
	controls := contract.Controls{MaxOutputTokens: maxTokens, Extensions: namespacedExtensions(object, allowed)}
	if value, err := decodeFloat(object, "temperature"); err != nil {
		return parsedRequest{}, err
	} else if value != nil {
		if *value < 0 || *value > 1 {
			return parsedRequest{}, contract.Invalid("temperature", "temperature must be between 0 and 1")
		}
		controls.Temperature = value
	}
	if value, err := decodeFloat(object, "top_p"); err != nil {
		return parsedRequest{}, err
	} else if value != nil {
		if *value < 0 || *value > 1 {
			return parsedRequest{}, contract.Invalid("top_p", "top_p must be between 0 and 1")
		}
		controls.TopP = value
	}
	if _, ok := object["stop_sequences"]; ok {
		controls.Stop, err = decodeStringSlice(object, "stop_sequences")
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if raw, ok := object["tools"]; ok {
		controls.Tools, err = parseAnthropicTools(raw)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if raw, ok := object["tool_choice"]; ok {
		controls.ToolChoice, err = parseAnthropicToolChoice(raw)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if raw, ok := object["metadata"]; ok {
		var metadata map[string]string
		if err := json.Unmarshal(raw, &metadata); err != nil {
			return parsedRequest{}, fmtError("metadata", "metadata must be an object of strings", err)
		}
		controls.Metadata = metadata
	}
	if raw, ok := object["thinking"]; ok {
		thinking, err := rawObject(raw, "thinking")
		if err != nil {
			return parsedRequest{}, err
		}
		if err := rejectUnknownStrict(thinking, map[string]bool{"type": true}); err != nil {
			return parsedRequest{}, err
		}
		typeName, err := requireString(thinking, "type")
		if err != nil {
			return parsedRequest{}, err
		}
		if typeName != "disabled" {
			return parsedRequest{}, contract.Unsupported("thinking", "internal thinking is not exposed by the canonical agent contract")
		}
	}
	for _, key := range []string{"service_tier", "container", "output_config", "mcp_servers"} {
		if _, ok := object[key]; ok {
			return parsedRequest{}, contract.Unsupported(key, key+" is not supported")
		}
	}
	instructions := ""
	if raw, ok := object["system"]; ok {
		instructions, err = parseAnthropicSystem(raw)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	stream := false
	if value, err := decodeBool(object, "stream"); err != nil {
		return parsedRequest{}, err
	} else if value != nil {
		stream = *value
	}
	return parsedRequest{Kind: protocolAnthropic, Request: contract.Request{Target: target, Instructions: instructions, Input: input, Controls: controls, Output: contract.OutputSpec{Modalities: contract.ModalityText}}, Stream: stream}, nil
}

func parseAnthropicMessage(raw jsontext.Value) ([]contract.Item, error) {
	object, err := rawObject(raw, "messages")
	if err != nil {
		return nil, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"role": true, "content": true}); err != nil {
		return nil, err
	}
	roleValue, err := requireString(object, "role")
	if err != nil {
		return nil, err
	}
	switch roleValue {
	case "user", "assistant":
	default:
		return nil, contract.Invalid("messages.role", "Anthropic messages only support user and assistant roles")
	}
	rawContent, ok := object["content"]
	if !ok {
		return nil, contract.Invalid("messages.content", "message content is required")
	}
	values, err := anthropicContent(rawContent)
	if err != nil {
		return nil, err
	}
	items := make([]contract.Item, 0, len(values)+1)
	var text []contract.Part
	for _, value := range values {
		if value.item.Type == contract.ItemMessage {
			text = append(text, value.item.Content...)
			continue
		}
		if len(text) > 0 {
			items = append(items, contract.MessageItem(contract.Role(roleValue), text...))
			text = nil
		}
		items = append(items, value.item)
	}
	if len(text) > 0 || len(items) == 0 {
		items = append(items, contract.MessageItem(contract.Role(roleValue), text...))
	}
	return items, nil
}

type anthropicContentValue struct{ item contract.Item }

func anthropicContent(raw jsontext.Value) ([]anthropicContentValue, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []anthropicContentValue{{item: contract.MessageItem(contract.RoleUser, contract.TextPart(text))}}, nil
	}
	values, err := rawArray(raw, "messages.content")
	if err != nil {
		return nil, err
	}
	out := make([]anthropicContentValue, 0, len(values))
	for _, value := range values {
		object, err := rawObject(value, "messages.content")
		if err != nil {
			return nil, err
		}
		typeName, err := requireString(object, "type")
		if err != nil {
			return nil, err
		}
		switch typeName {
		case "text":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "text": true}); err != nil {
				return nil, err
			}
			text, err := requireString(object, "text")
			if err != nil {
				return nil, err
			}
			out = append(out, anthropicContentValue{item: contract.MessageItem(contract.RoleUser, contract.TextPart(text))})
		case "image":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "source": true}); err != nil {
				return nil, err
			}
			media, err := parseAnthropicImage(object)
			if err != nil {
				return nil, err
			}
			out = append(out, anthropicContentValue{item: contract.MessageItem(contract.RoleUser, contract.ImagePart(media))})
		case "document":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "source": true}); err != nil {
				return nil, err
			}
			media, err := parseAnthropicDocument(object)
			if err != nil {
				return nil, err
			}
			out = append(out, anthropicContentValue{item: contract.MessageItem(contract.RoleUser, contract.FilePart(media))})
		case "tool_use":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "id": true, "name": true, "input": true}); err != nil {
				return nil, err
			}
			callID, err := requireString(object, "id")
			if err != nil {
				return nil, err
			}
			name, err := requireString(object, "name")
			if err != nil {
				return nil, err
			}
			input, ok := object["input"]
			if !ok || !input.IsValid() {
				return nil, contract.Invalid("messages.content.input", "tool input must be valid JSON")
			}
			call := contract.FunctionCallItem(callID, name, string(input))
			id, err := optionalString(object, "id")
			if err != nil {
				return nil, err
			}
			if id != "" {
				call.ID = id
			}
			out = append(out, anthropicContentValue{item: call})
		case "tool_result":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "tool_use_id": true, "content": true}); err != nil {
				return nil, err
			}
			callID, err := requireString(object, "tool_use_id")
			if err != nil {
				return nil, err
			}
			parts := []contract.Part{}
			if rawContent, ok := object["content"]; ok {
				content, err := anthropicContentParts(rawContent)
				if err != nil {
					return nil, err
				}
				parts = content
			}
			if len(parts) == 0 {
				parts = []contract.Part{contract.TextPart("")}
			}
			out = append(out, anthropicContentValue{item: contract.FunctionCallOutputItem(callID, parts...)})
		case "thinking":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "thinking": true, "signature": true}); err != nil {
				return nil, err
			}
			item := contract.Item{Type: contract.ItemReasoning, Data: append(jsontext.Value(nil), value...), Status: contract.StatusCompleted}
			out = append(out, anthropicContentValue{item: item})
		case "redacted_thinking":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "data": true}); err != nil {
				return nil, err
			}
			item := contract.Item{Type: contract.ItemReasoning, Data: append(jsontext.Value(nil), value...), Status: contract.StatusCompleted}
			out = append(out, anthropicContentValue{item: item})
		default:
			return nil, contract.Unsupported("messages.content.type", "unsupported Anthropic content type "+typeName)
		}
	}
	return out, nil
}

func anthropicContentParts(raw jsontext.Value) ([]contract.Part, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []contract.Part{contract.TextPart(text)}, nil
	}
	values, err := rawArray(raw, "tool_result.content")
	if err != nil {
		return nil, err
	}
	parts := make([]contract.Part, 0, len(values))
	for _, value := range values {
		object, err := rawObject(value, "tool_result.content")
		if err != nil {
			return nil, err
		}
		typeName, err := requireString(object, "type")
		if err != nil {
			return nil, err
		}
		switch typeName {
		case "text":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "text": true}); err != nil {
				return nil, err
			}
			text, err := requireString(object, "text")
			if err != nil {
				return nil, err
			}
			parts = append(parts, contract.TextPart(text))
		case "image":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true, "source": true}); err != nil {
				return nil, err
			}
			media, err := parseAnthropicImage(object)
			if err != nil {
				return nil, err
			}
			parts = append(parts, contract.ImagePart(media))
		default:
			return nil, contract.Unsupported("tool_result.content", "unsupported tool result type "+typeName)
		}
	}
	return parts, nil
}

func parseAnthropicImage(object map[string]jsontext.Value) (contract.Media, error) {
	source, err := rawObject(object["source"], "messages.content.source")
	if err != nil {
		return contract.Media{}, err
	}
	typeName, err := requireString(source, "type")
	if err != nil {
		return contract.Media{}, err
	}
	switch typeName {
	case "base64":
		if err := rejectUnknownStrict(source, map[string]bool{"type": true, "media_type": true, "data": true}); err != nil {
			return contract.Media{}, err
		}
		mime, err := requireString(source, "media_type")
		if err != nil {
			return contract.Media{}, err
		}
		encoded, err := requireString(source, "data")
		if err != nil {
			return contract.Media{}, err
		}
		data, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return contract.Media{}, fmtError("messages.content.source.data", "must be valid base64", err)
		}
		if !strings.HasPrefix(mime, "image/") {
			return contract.Media{}, contract.Invalid("messages.content.source.media_type", "Anthropic image media_type must be an image MIME type")
		}
		return contract.InlineMedia(mime, data), nil
	case "url":
		if err := rejectUnknownStrict(source, map[string]bool{"type": true, "url": true}); err != nil {
			return contract.Media{}, err
		}
		value, err := requireString(source, "url")
		if err != nil {
			return contract.Media{}, err
		}
		return contract.RemoteMedia("image/*", value), nil
	default:
		return contract.Media{}, contract.Unsupported("messages.content.source.type", "unsupported Anthropic image source "+typeName)
	}
}

func parseAnthropicDocument(object map[string]jsontext.Value) (contract.Media, error) {
	source, err := rawObject(object["source"], "messages.content.source")
	if err != nil {
		return contract.Media{}, err
	}
	typeName, err := requireString(source, "type")
	if err != nil {
		return contract.Media{}, err
	}
	switch typeName {
	case "base64":
		if err := rejectUnknownStrict(source, map[string]bool{"type": true, "media_type": true, "data": true}); err != nil {
			return contract.Media{}, err
		}
		mime, err := optionalString(source, "media_type")
		if err != nil {
			return contract.Media{}, err
		}
		if mime == "" {
			mime = "application/octet-stream"
		}
		encoded, err := requireString(source, "data")
		if err != nil {
			return contract.Media{}, err
		}
		data, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return contract.Media{}, fmtError("messages.content.source.data", "must be valid base64", err)
		}
		return contract.InlineMedia(mime, data), nil
	case "url":
		if err := rejectUnknownStrict(source, map[string]bool{"type": true, "url": true}); err != nil {
			return contract.Media{}, err
		}
		value, err := requireString(source, "url")
		if err != nil {
			return contract.Media{}, err
		}
		return contract.RemoteMedia("application/octet-stream", value), nil
	default:
		return contract.Media{}, contract.Unsupported("messages.content.source.type", "unsupported Anthropic document source "+typeName)
	}
}

func parseAnthropicSystem(raw jsontext.Value) (string, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}
	values, err := rawArray(raw, "system")
	if err != nil {
		return "", err
	}
	var builder strings.Builder
	for _, value := range values {
		object, err := rawObject(value, "system")
		if err != nil {
			return "", err
		}
		typeName, err := requireString(object, "type")
		if err != nil {
			return "", err
		}
		if typeName != "text" {
			return "", contract.Unsupported("system", "only text system blocks are supported")
		}
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "text": true}); err != nil {
			return "", err
		}
		value, err := requireString(object, "text")
		if err != nil {
			return "", err
		}
		builder.WriteString(value)
	}
	return builder.String(), nil
}

func parseAnthropicTools(raw jsontext.Value) ([]contract.Tool, error) {
	values, err := rawArray(raw, "tools")
	if err != nil {
		return nil, err
	}
	tools := make([]contract.Tool, 0, len(values))
	for _, value := range values {
		object, err := rawObject(value, "tools")
		if err != nil {
			return nil, err
		}
		if err := rejectUnknownStrict(object, map[string]bool{"name": true, "description": true, "input_schema": true}); err != nil {
			return nil, err
		}
		name, err := requireString(object, "name")
		if err != nil {
			return nil, err
		}
		inputSchema, ok := object["input_schema"]
		if !ok || !inputSchema.IsValid() {
			return nil, contract.Invalid("tools.input_schema", "input_schema must be valid JSON")
		}
		description, err := optionalString(object, "description")
		if err != nil {
			return nil, err
		}
		tools = append(tools, contract.Tool{Name: name, Description: description, Parameters: append(jsontext.Value(nil), inputSchema...)})
	}
	return tools, nil
}

func parseAnthropicToolChoice(raw jsontext.Value) (*contract.ToolChoice, error) {
	object, err := rawObject(raw, "tool_choice")
	if err != nil {
		return nil, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"type": true, "name": true}); err != nil {
		return nil, err
	}
	typeName, err := requireString(object, "type")
	if err != nil {
		return nil, err
	}
	switch typeName {
	case "auto", "none", "any":
		if _, ok := object["name"]; ok {
			return nil, contract.Invalid("tool_choice.name", "tool_choice name is only valid for type tool")
		}
		return &contract.ToolChoice{Mode: typeName}, nil
	case "tool":
		name, err := requireString(object, "name")
		if err != nil {
			return nil, err
		}
		return &contract.ToolChoice{Mode: "function", Name: name}, nil
	default:
		return nil, contract.Unsupported("tool_choice.type", "unsupported Anthropic tool choice "+typeName)
	}
}

// Adapter encodes canonical events as Anthropic Messages.
type Adapter struct{}

// NewAdapter returns the Anthropic Messages protocol adapter.
func NewAdapter() internalprotocol.Adapter { return Adapter{} }

// ValidateEvent checks whether event can be encoded for Anthropic Messages.
func (Adapter) ValidateEvent(event contract.Event) error {
	switch {
	case event.Type == contract.EventMedia || event.Type == contract.EventReasoning || event.Type == contract.EventActivity:
		return contract.Unsupported("output", "Anthropic Messages output supports text and tool_use only")
	case event.Type == contract.EventItem && event.Item.Type != contract.ItemMessage && event.Item.Type != contract.ItemFunctionCall:
		return contract.Unsupported("output", "Anthropic Messages output supports text and tool_use only")
	}
	if event.Type == contract.EventMessage {
		for _, part := range event.Item.Content {
			if part.Type != contract.PartText {
				return contract.Unsupported("output", "Anthropic Messages output supports text and tool_use only")
			}
		}
	}
	return nil
}

// Response builds a non-streaming Anthropic Messages response body.
func (Adapter) Response(req contract.Request, result execution.Result, meta responseMeta) (any, error) {
	content, tools, err := anthropicOutput(result.Items)
	if err != nil {
		return nil, err
	}
	value := map[string]any{
		"id":            meta.ID,
		"type":          "message",
		"role":          "assistant",
		"model":         meta.Model,
		"content":       content,
		"stop_reason":   anthropicStopReason(result.Outcome, tools),
		"stop_sequence": nil,
		"usage":         nil,
	}
	if result.Outcome.Usage != nil {
		value["usage"] = map[string]any{"input_tokens": result.Outcome.Usage.InputTokens, "output_tokens": result.Outcome.Usage.OutputTokens}
	}
	return value, nil
}

// Stream returns an SSE encoder for Anthropic Messages streaming events.
func (Adapter) Stream(w http.ResponseWriter, req contract.Request, meta responseMeta, limits contract.Limits) streamEncoder {
	return &anthropicStream{
		writer:     &sseWriter{w: w, limits: limits},
		request:    req,
		meta:       meta,
		toolBlocks: make(map[string]int),
		toolNames:  make(map[string]string),
		toolCalls:  make(map[string]string),
	}
}

func anthropicOutput(items []contract.Item) ([]any, bool, error) {
	content := make([]any, 0, len(items))
	hasTools := false
	for _, item := range items {
		switch item.Type {
		case contract.ItemMessage:
			for _, part := range item.Content {
				if part.Type != contract.PartText {
					return nil, false, contract.Unsupported("output", "Anthropic output supports text only")
				}
				content = append(content, map[string]any{"type": "text", "text": part.Text})
			}
		case contract.ItemFunctionCall:
			var input any
			if err := json.Unmarshal([]byte(item.Arguments), &input); err != nil {
				return nil, false, err
			}
			content = append(content, map[string]any{"type": "tool_use", "id": item.CallID, "name": item.Name, "input": input})
			hasTools = true
		default:
			return nil, false, contract.Unsupported("output", "unsupported Anthropic output item")
		}
	}
	return content, hasTools, nil
}

func anthropicStopReason(outcome contract.Outcome, tools bool) string {
	if tools || outcome.StopReason == contract.StopToolCall {
		return "tool_use"
	}
	if outcome.Status == contract.StatusIncomplete || outcome.StopReason == contract.StopLength {
		return "max_tokens"
	}
	return "end_turn"
}

type anthropicStream struct {
	writer     *sseWriter
	request    contract.Request
	meta       responseMeta
	started    bool
	block      int
	textOpen   bool
	textID     string
	text       strings.Builder
	toolBlocks map[string]int
	toolNames  map[string]string
	toolCalls  map[string]string
	hadTool    bool
}

// Started reports whether the Anthropic SSE stream has begun.
func (s *anthropicStream) Started() bool { return s.writer.started }

func (s *anthropicStream) emit(eventType string, value map[string]any) error {
	value["type"] = eventType
	return s.writer.write(eventType, value)
}

func (s *anthropicStream) start() error {
	if s.started {
		return nil
	}
	if err := s.emit("message_start", map[string]any{"message": map[string]any{"id": s.meta.ID, "type": "message", "role": "assistant", "model": s.meta.Model, "content": []any{}, "stop_reason": nil, "stop_sequence": nil, "usage": map[string]any{}}}); err != nil {
		return err
	}
	s.started = true
	return nil
}

func (s *anthropicStream) openText() error {
	if s.textOpen {
		return nil
	}
	s.textOpen = true
	s.textID = newID("text_")
	return s.emit("content_block_start", map[string]any{"index": s.block, "content_block": map[string]any{"type": "text", "text": ""}})
}

func (s *anthropicStream) closeText() error {
	if !s.textOpen {
		return nil
	}
	if err := s.emit("content_block_stop", map[string]any{"index": s.block}); err != nil {
		return err
	}
	s.block++
	s.textOpen = false
	s.textID = ""
	s.text.Reset()
	return nil
}

// Event encodes one canonical event as Anthropic SSE frames.
func (s *anthropicStream) Event(event contract.Event) error {
	if err := s.start(); err != nil {
		return err
	}
	switch event.Type {
	case contract.EventMessage:
		for _, part := range event.Item.Content {
			if part.Type != contract.PartText {
				return contract.Unsupported("output", "Anthropic stream supports text only")
			}
			if err := s.openText(); err != nil {
				return err
			}
			s.text.WriteString(part.Text)
			if err := s.emit("content_block_delta", map[string]any{"index": s.block, "delta": map[string]any{"type": "text_delta", "text": part.Text}}); err != nil {
				return err
			}
		}
		return s.closeText()
	case contract.EventTextDelta:
		if err := s.openText(); err != nil {
			return err
		}
		s.text.WriteString(event.Delta)
		return s.emit("content_block_delta", map[string]any{"index": s.block, "delta": map[string]any{"type": "text_delta", "text": event.Delta}})
	case contract.EventTextDone:
		return s.closeText()
	case contract.EventToolCallStart:
		if err := s.closeText(); err != nil {
			return err
		}
		s.hadTool = true
		s.toolBlocks[event.CallID] = s.block
		s.toolNames[event.CallID] = event.Name
		s.toolCalls[event.CallID] = event.CallID
		if err := s.emit("content_block_start", map[string]any{"index": s.block, "content_block": map[string]any{"type": "tool_use", "id": event.CallID, "name": event.Name, "input": map[string]any{}}}); err != nil {
			return err
		}
		s.block++
		return nil
	case contract.EventToolCall:
		if err := s.Event(contract.ToolCallStart(event.CallID, event.Name)); err != nil {
			return err
		}
		if event.Arguments != "" {
			if err := s.emit("content_block_delta", map[string]any{"index": s.toolBlocks[event.CallID], "delta": map[string]any{"type": "input_json_delta", "partial_json": event.Arguments}}); err != nil {
				return err
			}
		}
		return s.Event(contract.ToolCallDone(event.CallID))
	case contract.EventToolCallDelta:
		index, ok := s.toolBlocks[event.CallID]
		if !ok {
			return fmt.Errorf("unknown tool call %q", event.CallID)
		}
		return s.emit("content_block_delta", map[string]any{"index": index, "delta": map[string]any{"type": "input_json_delta", "partial_json": event.Delta}})
	case contract.EventToolCallDone:
		index, ok := s.toolBlocks[event.CallID]
		if !ok {
			return fmt.Errorf("unknown tool call %q", event.CallID)
		}
		delete(s.toolBlocks, event.CallID)
		return s.emit("content_block_stop", map[string]any{"index": index})
	case contract.EventItem:
		if event.Item.Type == contract.ItemMessage {
			return s.Event(contract.Event{Type: contract.EventMessage, Item: event.Item})
		}
		if event.Item.Type == contract.ItemFunctionCall {
			return s.Event(contract.ToolCall(event.Item.CallID, event.Item.Name, event.Item.Arguments))
		}
		return contract.Unsupported("output", "unsupported Anthropic output item")
	default:
		return fmt.Errorf("unsupported Anthropic event %q", event.Type)
	}
}

// Complete emits terminal Anthropic message_delta and message_stop events.
func (s *anthropicStream) Complete(outcome contract.Outcome, items []contract.Item) error {
	if err := s.start(); err != nil {
		return err
	}
	if err := s.closeText(); err != nil {
		return err
	}
	for callID, index := range s.toolBlocks {
		if err := s.emit("content_block_stop", map[string]any{"index": index}); err != nil {
			return err
		}
		delete(s.toolBlocks, callID)
	}
	stop := anthropicStopReason(outcome, s.hadTool || hasFunctionCall(items))
	usage := map[string]any{}
	if outcome.Usage != nil {
		usage["input_tokens"] = outcome.Usage.InputTokens
		usage["output_tokens"] = outcome.Usage.OutputTokens
	}
	if err := s.emit("message_delta", map[string]any{"delta": map[string]any{"stop_reason": stop, "stop_sequence": nil}, "usage": usage}); err != nil {
		return err
	}
	if err := s.emit("message_stop", map[string]any{}); err != nil {
		return err
	}
	return nil
}

func hasFunctionCall(items []contract.Item) bool {
	for _, item := range items {
		if item.Type == contract.ItemFunctionCall {
			return true
		}
	}
	return false
}

// Fail writes an Anthropic error event or HTTP error envelope.
func (s *anthropicStream) Fail(err error) error {
	apiErr := asAPIError(err)
	if !s.writer.started {
		return writeProtocolErrorAndReturn(s.writer.w, protocolAnthropic, apiErr)
	}
	if startErr := s.start(); startErr != nil {
		return startErr
	}
	if err := s.emit("error", map[string]any{"error": map[string]any{"type": anthropicErrorType(apiErr), "message": apiErr.Message}}); err != nil {
		return err
	}
	return s.emit("message_stop", map[string]any{})
}
