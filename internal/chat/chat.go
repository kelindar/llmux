package chat

import (
	"encoding/base64"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
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

const protocolChat = internalprotocol.Chat

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
func parseChatContent(raw jsontext.Value) ([]contract.Part, error) { return wire.ParseChatContent(raw) }
func parseMediaURL(value, detail string) (contract.Media, error) {
	return wire.ParseMediaURL(value, detail)
}
func parseFileMedia(object map[string]jsontext.Value, param string) (contract.Media, string, error) {
	return wire.ParseFileMedia(object, param)
}
func audioMIME(format string) string                         { return wire.AudioMIME(format) }
func outputTextParts(parts []contract.Part) []map[string]any { return wire.OutputTextParts(parts) }
func mediaDataURL(media contract.Media) (string, error)      { return wire.MediaDataURL(media) }
func fmtError(param, message string, err error) *contract.APIError {
	return wire.Error(param, message, err)
}
func validRole(role contract.Role) bool { return contract.ValidRole(role) }

func decodeBool(object map[string]jsontext.Value, key string) (*bool, error) {
	return wire.DecodeBool(object, key)
}

func decodeString(object map[string]jsontext.Value, key string) (string, bool, error) {
	return wire.DecodeString(object, key)
}
func decodeInt(object map[string]jsontext.Value, key string) (*int, error) {
	return wire.DecodeInt(object, key)
}
func decodeFloat(object map[string]jsontext.Value, key string) (*float64, error) {
	return wire.DecodeFloat(object, key)
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

// ParseRequest decodes a Chat Completions request into a canonical request.
func ParseRequest(object map[string]jsontext.Value) (parsedRequest, error) {
	allowed := map[string]bool{
		"model": true, "messages": true, "stream": true, "stream_options": true,
		"max_tokens": true, "max_completion_tokens": true, "temperature": true,
		"top_p": true, "stop": true, "tools": true, "tool_choice": true,
		"parallel_tool_calls": true, "response_format": true, "modalities": true,
		"audio": true, "n": true, "logprobs": true, "top_logprobs": true,
		"user": true, "store": true, "metadata": true, "reasoning_effort": true,
	}
	if err := rejectUnknown(object, allowed); err != nil {
		return parsedRequest{}, err
	}
	target, err := requireString(object, "model")
	if err != nil {
		return parsedRequest{}, err
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
		items, err := parseChatMessage(value)
		if err != nil {
			return parsedRequest{}, err
		}
		input = append(input, items...)
	}
	controls := contract.Controls{Extensions: namespacedExtensions(object, allowed)}
	if maxTokens, err := decodeInt(object, "max_tokens"); err != nil {
		return parsedRequest{}, err
	} else if maxTokens != nil {
		controls.MaxOutputTokens = maxTokens
	}
	if maxTokens, err := decodeInt(object, "max_completion_tokens"); err != nil {
		return parsedRequest{}, err
	} else if maxTokens != nil {
		if controls.MaxOutputTokens != nil {
			return parsedRequest{}, contract.Invalid("max_completion_tokens", "max_tokens and max_completion_tokens cannot both be set")
		}
		controls.MaxOutputTokens = maxTokens
	}
	if controls.MaxOutputTokens != nil && *controls.MaxOutputTokens < 1 {
		return parsedRequest{}, contract.Invalid("max_tokens", "max_tokens must be positive")
	}
	if value, err := decodeFloat(object, "temperature"); err != nil {
		return parsedRequest{}, err
	} else if value != nil {
		if *value < 0 || *value > 2 {
			return parsedRequest{}, contract.Invalid("temperature", "temperature must be between 0 and 2")
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
	stop, err := parseStop(object["stop"])
	if err != nil {
		return parsedRequest{}, err
	}
	controls.Stop = stop
	if raw, ok := object["tools"]; ok {
		controls.Tools, err = parseChatTools(raw)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if raw, ok := object["tool_choice"]; ok {
		controls.ToolChoice, err = parseToolChoice(raw)
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if _, ok := object["parallel_tool_calls"]; ok {
		controls.ParallelToolCall, err = decodeBool(object, "parallel_tool_calls")
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if raw, ok := object["stream_options"]; ok {
		streamOptions, err := rawObject(raw, "stream_options")
		if err != nil {
			return parsedRequest{}, err
		}
		if err := rejectUnknownStrict(streamOptions, map[string]bool{"include_usage": true}); err != nil {
			return parsedRequest{}, err
		}
		includeUsage, err := decodeBool(streamOptions, "include_usage")
		if err != nil {
			return parsedRequest{}, err
		}
		controls.IncludeUsage = includeUsage != nil && *includeUsage
	}
	output := contract.OutputSpec{Modalities: contract.ModalityText}
	modalitiesProvided := false
	if raw, ok := object["modalities"]; ok {
		modalitiesProvided = true
		output.Modalities, err = parseModalities(raw, "modalities")
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if raw, ok := object["audio"]; ok {
		controls.Audio, err = parseAudioControls(raw)
		if err != nil {
			return parsedRequest{}, err
		}
		output.Modalities |= contract.ModalityAudio
	}
	if controls.Audio != nil && modalitiesProvided && !output.Modalities.Has(contract.ModalityAudio) {
		return parsedRequest{}, contract.Invalid("modalities", "audio output requires audio in modalities")
	}
	if output.Modalities.Has(contract.ModalityAudio) && controls.Audio == nil {
		return parsedRequest{}, contract.Invalid("audio", "audio controls are required for audio output")
	}
	if raw, ok := object["response_format"]; ok {
		format, structured, err := parseResponseFormat(raw)
		if err != nil {
			return parsedRequest{}, err
		}
		output.Format = format
		controls.Structured = structured
	}
	if raw, ok := object["n"]; ok {
		var n int
		if err := json.Unmarshal(raw, &n); err != nil {
			return parsedRequest{}, fmtError("n", "must be an integer", err)
		}
		if n != 1 {
			return parsedRequest{}, contract.Unsupported("n", "only n=1 is supported")
		}
	}
	if raw, ok := object["logprobs"]; ok {
		var enabled bool
		if err := json.Unmarshal(raw, &enabled); err != nil {
			return parsedRequest{}, fmtError("logprobs", "must be a boolean", err)
		}
		if enabled {
			return parsedRequest{}, contract.Unsupported("logprobs", "log probabilities are not supported")
		}
	}
	if _, ok := object["top_logprobs"]; ok {
		return parsedRequest{}, contract.Unsupported("top_logprobs", "log probabilities are not supported")
	}
	if _, ok := object["store"]; ok {
		controls.Store, err = decodeBool(object, "store")
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if raw, ok := object["reasoning_effort"]; ok {
		effort, err := requireString(map[string]jsontext.Value{"reasoning_effort": raw}, "reasoning_effort")
		if err != nil {
			return parsedRequest{}, err
		}
		controls.Reasoning = &contract.Reasoning{Effort: effort}
	}
	if raw, ok := object["metadata"]; ok {
		if err := json.Unmarshal(raw, &controls.Metadata); err != nil {
			return parsedRequest{}, fmtError("metadata", "must be an object of strings", err)
		}
	}
	if raw, ok := object["user"]; ok {
		if _, err := requireString(map[string]jsontext.Value{"user": raw}, "user"); err != nil {
			return parsedRequest{}, err
		}
		return parsedRequest{}, contract.Unsupported("user", "the user field is not supported")
	}
	stream := false
	if value, err := decodeBool(object, "stream"); err != nil {
		return parsedRequest{}, err
	} else if value != nil {
		stream = *value
	}
	return parsedRequest{Kind: protocolChat, Request: contract.Request{Target: target, Input: input, Controls: controls, Output: output}, Stream: stream}, nil
}

func parseChatMessage(raw jsontext.Value) ([]contract.Item, error) {
	object, err := rawObject(raw, "messages")
	if err != nil {
		return nil, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"role": true, "content": true, "tool_calls": true, "tool_call_id": true}); err != nil {
		return nil, err
	}
	roleValue, err := requireString(object, "role")
	if err != nil {
		return nil, err
	}
	role := contract.Role(roleValue)
	if !validRole(role) {
		return nil, contract.Invalid("messages.role", "unsupported message role "+roleValue)
	}
	var content []contract.Part
	if rawContent, ok := object["content"]; ok {
		content, err = parseChatContent(rawContent)
		if err != nil {
			return nil, err
		}
	}
	if role == contract.RoleTool {
		callID, err := requireString(object, "tool_call_id")
		if err != nil {
			return nil, err
		}
		return []contract.Item{contract.FunctionCallOutputItem(callID, content...)}, nil
	}
	items := make([]contract.Item, 0, 2)
	if len(content) > 0 || role != contract.RoleAssistant {
		items = append(items, contract.MessageItem(role, content...))
	}
	if rawCalls, ok := object["tool_calls"]; ok {
		calls, err := rawArray(rawCalls, "messages.tool_calls")
		if err != nil {
			return nil, err
		}
		for _, rawCall := range calls {
			call, err := parseChatToolCall(rawCall)
			if err != nil {
				return nil, err
			}
			items = append(items, call)
		}
	}
	if len(items) == 0 {
		items = append(items, contract.MessageItem(role))
	}
	return items, nil
}

func parseChatToolCall(raw jsontext.Value) (contract.Item, error) {
	object, err := rawObject(raw, "messages.tool_calls")
	if err != nil {
		return contract.Item{}, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"id": true, "type": true, "function": true}); err != nil {
		return contract.Item{}, err
	}
	if typeName, ok, err := decodeString(object, "type"); err != nil {
		return contract.Item{}, err
	} else if ok && typeName != "function" {
		return contract.Item{}, contract.Unsupported("messages.tool_calls.type", "only function tool calls are supported")
	}
	callID, err := requireString(object, "id")
	if err != nil {
		return contract.Item{}, err
	}
	function, err := rawObject(object["function"], "messages.tool_calls.function")
	if err != nil {
		return contract.Item{}, err
	}
	if err := rejectUnknownStrict(function, map[string]bool{"name": true, "arguments": true}); err != nil {
		return contract.Item{}, err
	}
	name, err := requireString(function, "name")
	if err != nil {
		return contract.Item{}, err
	}
	arguments, err := requireString(function, "arguments")
	if err != nil {
		return contract.Item{}, err
	}
	if !jsontext.Value(arguments).IsValid() {
		return contract.Item{}, contract.Invalid("messages.tool_calls.function.arguments", "arguments must be valid JSON")
	}
	return contract.FunctionCallItem(callID, name, arguments), nil
}

func parseChatTools(raw jsontext.Value) ([]contract.Tool, error) {
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
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "function": true}); err != nil {
			return nil, err
		}
		typeName := "function"
		if value, ok, err := decodeString(object, "type"); err != nil {
			return nil, err
		} else if ok {
			typeName = value
		}
		if typeName != "function" {
			return nil, contract.Unsupported("tools", "only function tools are supported")
		}
		function, err := rawObject(object["function"], "tools.function")
		if err != nil {
			return nil, err
		}
		if err := rejectUnknownStrict(function, map[string]bool{"name": true, "description": true, "parameters": true, "strict": true}); err != nil {
			return nil, err
		}
		name, err := requireString(function, "name")
		if err != nil {
			return nil, err
		}
		tool := contract.Tool{Name: name}
		if value, ok, err := decodeString(function, "description"); err != nil {
			return nil, err
		} else if ok {
			tool.Description = value
		}
		if parameters, ok := function["parameters"]; ok {
			tool.Parameters = append(jsontext.Value(nil), parameters...)
		}
		if strict, err := decodeBool(function, "strict"); err != nil {
			return nil, err
		} else {
			tool.Strict = strict
		}
		tools = append(tools, tool)
	}
	return tools, nil
}

func parseToolChoice(raw jsontext.Value) (*contract.ToolChoice, error) {
	var value string
	if err := json.Unmarshal(raw, &value); err == nil {
		switch value {
		case "auto", "none", "required":
			return &contract.ToolChoice{Mode: value}, nil
		default:
			return nil, contract.Invalid("tool_choice", "unsupported tool_choice "+value)
		}
	}
	object, err := rawObject(raw, "tool_choice")
	if err != nil {
		return nil, err
	}
	if functionRaw, ok := object["function"]; ok {
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "function": true}); err != nil {
			return nil, err
		}
		if typeName, ok, err := decodeString(object, "type"); err != nil {
			return nil, err
		} else if ok && typeName != "function" {
			return nil, contract.Unsupported("tool_choice.type", "only function tool choices are supported")
		}
		function, err := rawObject(functionRaw, "tool_choice.function")
		if err != nil {
			return nil, err
		}
		if err := rejectUnknownStrict(function, map[string]bool{"name": true}); err != nil {
			return nil, err
		}
		name, err := requireString(function, "name")
		if err != nil {
			return nil, err
		}
		return &contract.ToolChoice{Mode: "function", Name: name}, nil
	}
	if err := rejectUnknownStrict(object, map[string]bool{"type": true, "name": true}); err != nil {
		return nil, err
	}
	mode, err := requireString(object, "type")
	if err != nil {
		return nil, err
	}
	if mode != "function" {
		return nil, contract.Invalid("tool_choice", "unsupported tool_choice type "+mode)
	}
	name, err := requireString(object, "name")
	if err != nil {
		return nil, err
	}
	return &contract.ToolChoice{Mode: "function", Name: name}, nil
}

func parseStop(raw jsontext.Value) ([]string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		return []string{one}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err != nil {
		return nil, fmtError("stop", "must be a string or array of strings", err)
	}
	return many, nil
}

func parseModalities(raw jsontext.Value, param string) (contract.Modality, error) {
	values, err := rawArray(raw, param)
	if err != nil {
		return 0, err
	}
	var modalities contract.Modality
	for _, value := range values {
		var name string
		if err := json.Unmarshal(value, &name); err != nil {
			return 0, fmtError(param, "must contain strings", err)
		}
		switch name {
		case "text":
			modalities |= contract.ModalityText
		case "audio":
			modalities |= contract.ModalityAudio
		default:
			return 0, contract.Unsupported(param, "unsupported output modality "+name)
		}
	}
	if modalities == 0 {
		return 0, contract.Invalid(param, "at least one output modality is required")
	}
	return modalities, nil
}

func parseAudioControls(raw jsontext.Value) (*contract.AudioControls, error) {
	object, err := rawObject(raw, "audio")
	if err != nil {
		return nil, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"voice": true, "format": true}); err != nil {
		return nil, err
	}
	voice, err := requireString(object, "voice")
	if err != nil {
		return nil, err
	}
	format, err := requireString(object, "format")
	if err != nil {
		return nil, err
	}
	return &contract.AudioControls{Voice: voice, Format: format}, nil
}

func parseResponseFormat(raw jsontext.Value) (string, *contract.StructuredOutput, error) {
	object, err := rawObject(raw, "response_format")
	if err != nil {
		return "", nil, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"type": true, "json_schema": true}); err != nil {
		return "", nil, err
	}
	typeName, err := requireString(object, "type")
	if err != nil {
		return "", nil, err
	}
	switch typeName {
	case "text":
		return typeName, nil, nil
	case "json_object":
		return typeName, &contract.StructuredOutput{Name: "response", Schema: jsontext.Value(`{"type":"object"}`)}, nil
	case "json_schema":
		schemaObject, err := rawObject(object["json_schema"], "response_format.json_schema")
		if err != nil {
			return "", nil, err
		}
		if err := rejectUnknownStrict(schemaObject, map[string]bool{"name": true, "description": true, "schema": true, "strict": true}); err != nil {
			return "", nil, err
		}
		name, err := requireString(schemaObject, "name")
		if err != nil {
			return "", nil, err
		}
		schema := append(jsontext.Value(nil), schemaObject["schema"]...)
		if len(schema) == 0 || !schema.IsValid() {
			return "", nil, contract.Invalid("response_format.json_schema.schema", "schema must be valid JSON")
		}
		description := ""
		if value, ok, err := decodeString(schemaObject, "description"); err != nil {
			return "", nil, err
		} else if ok {
			description = value
		}
		strict, err := decodeBool(schemaObject, "strict")
		if err != nil {
			return "", nil, err
		}
		return typeName, &contract.StructuredOutput{Name: name, Description: description, Schema: schema, Strict: strict != nil && *strict}, nil
	default:
		return "", nil, contract.Unsupported("response_format", "unsupported response format "+typeName)
	}
}

// Adapter encodes canonical events as OpenAI Chat Completions.
type Adapter struct{}

// NewAdapter returns the Chat Completions protocol adapter.
func NewAdapter() internalprotocol.Adapter { return Adapter{} }

// ValidateEvent checks whether event can be encoded for Chat Completions.
func (Adapter) ValidateEvent(event contract.Event) error {
	switch {
	case event.Type == contract.EventReasoning:
		return contract.Unsupported("output", "Chat Completions has no public reasoning-summary output mapping")
	case event.Type == contract.EventActivity:
		return contract.Unsupported("output", "activity events are not supported on Chat Completions")
	case event.Type == contract.EventMedia && event.Part.Type != contract.PartAudio:
		return contract.Unsupported("output", "Chat Completions supports audio output here, not generated image output")
	}
	if event.Type == contract.EventItem {
		switch event.Item.Type {
		case contract.ItemMedia:
			if len(event.Item.Content) != 1 || event.Item.Content[0].Type != contract.PartAudio {
				return contract.Unsupported("output", "Chat Completions supports audio output here, not generated image output")
			}
		case contract.ItemMessage:
			for _, part := range event.Item.Content {
				if part.Type != contract.PartText && part.Type != contract.PartAudio {
					return contract.Unsupported("output", "Chat Completions supports text and audio output here")
				}
			}
		case contract.ItemFunctionCall:
		default:
			return contract.Unsupported("output", "unsupported Chat Completions output item")
		}
	}
	if event.Type == contract.EventMessage {
		for _, part := range event.Item.Content {
			if part.Type != contract.PartText && part.Type != contract.PartAudio {
				return contract.Unsupported("output", "Chat Completions supports text and audio output here")
			}
		}
	}
	return nil
}

// Response builds a non-streaming Chat Completions response body.
func (Adapter) Response(req contract.Request, result execution.Result, meta responseMeta) (any, error) {
	message, finishReason, err := chatMessage(result.Items, result.Outcome)
	if err != nil {
		return nil, err
	}
	response := map[string]any{
		"id":      meta.ID,
		"object":  "chat.completion",
		"created": meta.Created,
		"model":   meta.Model,
		"choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": finishReason}},
	}
	if result.Outcome.Usage != nil {
		response["usage"] = chatUsage(result.Outcome.Usage)
	}
	return response, nil
}

// Stream returns an SSE encoder for Chat Completions streaming chunks.
func (Adapter) Stream(w http.ResponseWriter, req contract.Request, meta responseMeta, limits contract.Limits) streamEncoder {
	return &chatStream{writer: &sseWriter{w: w, limits: limits}, request: req, meta: meta}
}

func chatMessage(items []contract.Item, outcome contract.Outcome) (map[string]any, string, error) {
	message := map[string]any{"role": "assistant"}
	var text strings.Builder
	var calls []map[string]any
	var audio *map[string]any
	for _, item := range items {
		switch item.Type {
		case contract.ItemMessage:
			for _, part := range item.Content {
				switch part.Type {
				case contract.PartText:
					text.WriteString(part.Text)
				case contract.PartAudio:
					value, err := chatAudio(part)
					if err != nil {
						return nil, "", err
					}
					if audio != nil {
						return nil, "", contract.Unsupported("output", "Chat Completions supports one audio output")
					}
					audio = &value
				default:
					return nil, "", contract.Unsupported("output", "unsupported Chat Completions output part")
				}
			}
		case contract.ItemFunctionCall:
			calls = append(calls, map[string]any{"id": item.CallID, "type": "function", "function": map[string]any{"name": item.Name, "arguments": item.Arguments}})
		case contract.ItemReasoning:
			return nil, "", contract.Unsupported("output", "Chat Completions has no public reasoning-summary output mapping")
		case contract.ItemMedia:
			if len(item.Content) != 1 || item.Content[0].Type != contract.PartAudio {
				return nil, "", contract.Unsupported("output", "generated image output is not part of Chat Completions")
			}
			value, err := chatAudio(item.Content[0])
			if err != nil {
				return nil, "", err
			}
			if audio != nil {
				return nil, "", contract.Unsupported("output", "Chat Completions supports one audio output")
			}
			audio = &value
		default:
			return nil, "", contract.Unsupported("output", "unsupported output item")
		}
	}
	textValue := text.String()
	switch {
	case textValue != "":
		message["content"] = textValue
	case len(calls) > 0 || audio != nil:
		message["content"] = nil
	default:
		message["content"] = ""
	}
	if len(calls) > 0 {
		message["tool_calls"] = calls
	}
	if audio != nil {
		message["audio"] = *audio
	}
	finishReason := chatFinishReason(outcome, len(calls) > 0)
	return message, finishReason, nil
}

func chatFinishReason(outcome contract.Outcome, tools bool) string {
	if tools || outcome.StopReason == contract.StopToolCall {
		return "tool_calls"
	}
	if outcome.Status == contract.StatusIncomplete || outcome.StopReason == contract.StopLength {
		return "length"
	}
	return "stop"
}

func chatUsage(usage *contract.Usage) map[string]any {
	return map[string]any{"prompt_tokens": usage.InputTokens, "completion_tokens": usage.OutputTokens, "total_tokens": usage.TotalTokens}
}

func chatAudio(part contract.Part) (map[string]any, error) {
	if part.Media == nil || len(part.Media.Data) == 0 {
		return nil, errors.New("Chat Completions audio output requires inline data")
	}
	id := part.Media.Ref
	if id == "" {
		id = newID("audio_")
	}
	return map[string]any{
		"id":         id,
		"data":       base64.StdEncoding.EncodeToString(part.Media.Data),
		"expires_at": 0,
		"transcript": part.Text,
	}, nil
}

type chatStream struct {
	writer      *sseWriter
	request     contract.Request
	meta        responseMeta
	started     bool
	roleSent    bool
	toolIndexes map[string]int
	nextTool    int
	hadTool     bool
}

// Started reports whether the Chat Completions SSE stream has begun.
func (s *chatStream) Started() bool { return s.writer.started }

// Event encodes one canonical event as a Chat Completions chunk.
func (s *chatStream) Event(event contract.Event) error {
	if s.toolIndexes == nil {
		s.toolIndexes = make(map[string]int)
	}
	if event.Type == contract.EventTextDone || event.Type == contract.EventToolCallDone {
		return nil
	}
	chunk := map[string]any{"id": s.meta.ID, "object": "chat.completion.chunk", "created": s.meta.Created, "model": s.meta.Model, "choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": nil}}}
	if s.request.Controls.IncludeUsage {
		chunk["usage"] = nil
	}
	delta := chunk["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	switch event.Type {
	case contract.EventMessage:
		for _, part := range event.Item.Content {
			switch part.Type {
			case contract.PartText:
				if !s.roleSent {
					delta["role"] = "assistant"
					s.roleSent = true
				}
				delta["content"] = part.Text
				if err := s.writer.write("", chunk); err != nil {
					return err
				}
			case contract.PartAudio:
				if part.Media == nil || len(part.Media.Data) == 0 {
					return errors.New("streamed audio output requires inline data")
				}
				if !s.roleSent {
					delta["role"] = "assistant"
					s.roleSent = true
				}
				delta["audio"] = map[string]any{"data": base64.StdEncoding.EncodeToString(part.Media.Data)}
				if err := s.writer.write("", chunk); err != nil {
					return err
				}
			default:
				return contract.Unsupported("output", "Chat Completions stream supports text and audio only")
			}
		}
	case contract.EventTextDelta:
		if !s.roleSent {
			delta["role"] = "assistant"
			s.roleSent = true
		}
		delta["content"] = event.Delta
		return s.writer.write("", chunk)
	case contract.EventToolCallStart, contract.EventToolCall:
		s.hadTool = true
		if !s.roleSent {
			delta["role"] = "assistant"
			s.roleSent = true
		}
		index := s.nextTool
		if existing, ok := s.toolIndexes[event.CallID]; ok {
			index = existing
		} else {
			s.toolIndexes[event.CallID] = index
			s.nextTool++
		}
		function := map[string]any{"name": event.Name}
		if event.Type == contract.EventToolCall {
			function["arguments"] = event.Arguments
		} else {
			function["arguments"] = ""
		}
		delta["tool_calls"] = []any{map[string]any{"index": index, "id": event.CallID, "type": "function", "function": function}}
		return s.writer.write("", chunk)
	case contract.EventToolCallDelta:
		s.hadTool = true
		index, ok := s.toolIndexes[event.CallID]
		if !ok {
			return fmt.Errorf("unknown tool call %q", event.CallID)
		}
		delta["tool_calls"] = []any{map[string]any{"index": index, "function": map[string]any{"arguments": event.Delta}}}
		return s.writer.write("", chunk)
	case contract.EventReasoning:
		return contract.Unsupported("output", "Chat Completions has no public reasoning-summary output mapping")
	case contract.EventMedia:
		if event.Part.Type != contract.PartAudio {
			return contract.Unsupported("output", "Chat Completions stream only supports audio media")
		}
		if event.Part.Media == nil || len(event.Part.Media.Data) == 0 {
			return errors.New("streamed audio output requires inline data")
		}
		if !s.roleSent {
			delta["role"] = "assistant"
			s.roleSent = true
		}
		delta["audio"] = map[string]any{"data": base64.StdEncoding.EncodeToString(event.Part.Media.Data)}
		return s.writer.write("", chunk)
	case contract.EventItem:
		return s.eventItem(event.Item)
	default:
		return fmt.Errorf("unsupported Chat Completions event %q", event.Type)
	}
	return nil
}

func (s *chatStream) eventItem(item contract.Item) error {
	switch item.Type {
	case contract.ItemMessage:
		return s.Event(contract.Event{Type: contract.EventMessage, Item: item})
	case contract.ItemFunctionCall:
		return s.Event(contract.ToolCall(item.CallID, item.Name, item.Arguments))
	case contract.ItemMedia:
		if len(item.Content) == 1 {
			return s.Event(contract.MediaOutput(item.Content[0]))
		}
	}
	return contract.Unsupported("output", "unsupported Chat Completions output item")
}

// Complete emits the final chunk and optional usage chunk for the stream.
func (s *chatStream) Complete(outcome contract.Outcome, items []contract.Item) error {
	finish := chatFinishReason(outcome, s.hadTool)
	chunk := map[string]any{"id": s.meta.ID, "object": "chat.completion.chunk", "created": s.meta.Created, "model": s.meta.Model, "choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": finish}}}
	if s.request.Controls.IncludeUsage {
		chunk["usage"] = nil
	}
	if err := s.writer.write("", chunk); err != nil {
		return err
	}
	if outcome.Usage != nil && s.request.Controls.IncludeUsage {
		usageChunk := map[string]any{"id": s.meta.ID, "object": "chat.completion.chunk", "created": s.meta.Created, "model": s.meta.Model, "choices": []any{}, "usage": chatUsage(outcome.Usage)}
		if err := s.writer.write("", usageChunk); err != nil {
			return err
		}
	}
	return s.writer.done()
}

// Fail writes a Chat Completions error chunk or HTTP error envelope.
func (s *chatStream) Fail(err error) error {
	apiErr := internalprotocol.AsAPIError(err)
	if !s.writer.started {
		return writeProtocolErrorAndReturn(s.writer.w, protocolChat, apiErr)
	}
	payload := map[string]any{"error": map[string]any{"message": apiErr.Message, "type": apiErr.Type, "code": apiErr.Code}}
	if apiErr.Param != "" {
		payload["error"].(map[string]any)["param"] = apiErr.Param
	}
	if err := s.writer.write("", payload); err != nil {
		return err
	}
	return s.writer.done()
}

func writeProtocolErrorAndReturn(w http.ResponseWriter, p protocol, err error) error {
	internalprotocol.WriteError(w, p, err)
	return err
}
