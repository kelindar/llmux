package responses

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

const protocolResponses = internalprotocol.Responses

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
func parseStringOrContent(raw jsontext.Value, param string, parser func(jsontext.Value) ([]contract.Part, error)) ([]contract.Part, error) {
	return wire.ParseStringOrContent(raw, param, parser)
}
func parseMediaURL(value, detail string) (contract.Media, error) {
	return wire.ParseMediaURL(value, detail)
}
func parseFileMedia(object map[string]jsontext.Value, param string) (contract.Media, string, error) {
	return wire.ParseFileMedia(object, param)
}
func inputTextParts(parts []contract.Part) []map[string]any  { return wire.InputTextParts(parts) }
func outputTextParts(parts []contract.Part) []map[string]any { return wire.OutputTextParts(parts) }
func collectText(parts []contract.Part) string               { return wire.CollectText(parts) }
func mediaDataURL(media contract.Media) (string, error)      { return wire.MediaDataURL(media) }
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
func decodeInt(object map[string]jsontext.Value, key string) (*int, error) {
	return wire.DecodeInt(object, key)
}
func decodeFloat(object map[string]jsontext.Value, key string) (*float64, error) {
	return wire.DecodeFloat(object, key)
}

func optionalString(object map[string]jsontext.Value, key string) (string, error) {
	value, _, err := decodeString(object, key)
	return value, err
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

func writeProtocolErrorAndReturn(w http.ResponseWriter, p protocol, err error) error {
	internalprotocol.WriteError(w, p, err)
	return err
}

// ParseRequest decodes a Responses API request into a canonical request.
func ParseRequest(object map[string]jsontext.Value) (parsedRequest, error) {
	allowed := map[string]bool{
		"model": true, "input": true, "instructions": true, "stream": true,
		"stream_options": true, "tools": true, "tool_choice": true,
		"parallel_tool_calls": true, "temperature": true, "top_p": true,
		"max_output_tokens": true, "text": true, "reasoning": true, "store": true,
		"previous_response_id": true, "metadata": true, "background": true,
		"conversation": true, "include": true, "service_tier": true,
		"truncation": true, "user": true, "prompt": true, "max_tool_calls": true,
	}
	if err := rejectUnknown(object, allowed); err != nil {
		return parsedRequest{}, err
	}
	target, err := requireString(object, "model")
	if err != nil {
		return parsedRequest{}, err
	}
	rawInput, ok := object["input"]
	if !ok {
		return parsedRequest{}, contract.Invalid("input", "input is required")
	}
	input, err := parseResponsesInput(rawInput)
	if err != nil {
		return parsedRequest{}, err
	}
	controls := contract.Controls{Extensions: namespacedExtensions(object, allowed)}
	if value, err := decodeInt(object, "max_output_tokens"); err != nil {
		return parsedRequest{}, err
	} else if value != nil {
		if *value < 1 {
			return parsedRequest{}, contract.Invalid("max_output_tokens", "max_output_tokens must be positive")
		}
		controls.MaxOutputTokens = value
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
	if raw, ok := object["tools"]; ok {
		controls.Tools, controls.ImageGeneration, err = parseResponsesTools(raw)
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
		if err := rejectUnknownStrict(streamOptions, map[string]bool{"include_obfuscation": true}); err != nil {
			return parsedRequest{}, err
		}
	}
	output := contract.OutputSpec{Modalities: contract.ModalityText}
	if controls.ImageGeneration {
		output.Modalities |= contract.ModalityImage
	}
	if raw, ok := object["text"]; ok {
		format, structured, err := parseResponsesText(raw)
		if err != nil {
			return parsedRequest{}, err
		}
		output.Format = format
		controls.Structured = structured
	}
	if raw, ok := object["reasoning"]; ok {
		reasoning, err := parseResponsesReasoning(raw)
		if err != nil {
			return parsedRequest{}, err
		}
		controls.Reasoning = reasoning
	}
	if _, ok := object["store"]; ok {
		controls.Store, err = decodeBool(object, "store")
		if err != nil {
			return parsedRequest{}, err
		}
	}
	if raw, ok := object["previous_response_id"]; ok {
		value, err := requireString(map[string]jsontext.Value{"previous_response_id": raw}, "previous_response_id")
		if err != nil {
			return parsedRequest{}, err
		}
		controls.PreviousResponseID = &value
	}
	if raw, ok := object["metadata"]; ok {
		if err := json.Unmarshal(raw, &controls.Metadata); err != nil {
			return parsedRequest{}, fmtError("metadata", "must be an object of strings", err)
		}
	}
	if _, ok := object["background"]; ok {
		return parsedRequest{}, contract.Unsupported("background", "background execution is not supported")
	}
	for _, key := range []string{"conversation", "include", "service_tier", "truncation", "user", "prompt", "max_tool_calls"} {
		if _, ok := object[key]; ok {
			return parsedRequest{}, contract.Unsupported(key, key+" is not supported")
		}
	}
	stream := false
	if value, err := decodeBool(object, "stream"); err != nil {
		return parsedRequest{}, err
	} else if value != nil {
		stream = *value
	}
	instructions := ""
	if _, ok := object["instructions"]; ok {
		instructions, _, err = decodeString(object, "instructions")
		if err != nil {
			return parsedRequest{}, err
		}
	}
	return parsedRequest{Kind: protocolResponses, Request: contract.Request{Target: target, Instructions: instructions, Input: input, Controls: controls, Output: output}, Stream: stream}, nil
}

func parseResponsesInput(raw jsontext.Value) ([]contract.Item, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []contract.Item{contract.MessageItem(contract.RoleUser, contract.TextPart(text))}, nil
	}
	values, err := rawArray(raw, "input")
	if err != nil {
		return nil, err
	}
	if len(values) == 0 {
		return nil, contract.Invalid("input", "input must not be empty")
	}
	items := make([]contract.Item, 0, len(values))
	for _, value := range values {
		item, err := parseResponsesItem(value)
		if err != nil {
			return nil, err
		}
		items = append(items, item...)
	}
	return items, nil
}

func parseResponsesItem(raw jsontext.Value) ([]contract.Item, error) {
	object, err := rawObject(raw, "input")
	if err != nil {
		return nil, err
	}
	typeName, ok, err := decodeString(object, "type")
	if err != nil {
		return nil, err
	}
	if !ok {
		typeName = "message"
	}
	switch typeName {
	case "message":
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "role": true, "content": true, "id": true}); err != nil {
			return nil, err
		}
		roleValue, err := requireString(object, "role")
		if err != nil {
			return nil, err
		}
		role := contract.Role(roleValue)
		if !validRole(role) {
			return nil, contract.Invalid("input.role", "unsupported message role "+roleValue)
		}
		rawContent, ok := object["content"]
		if !ok {
			return nil, contract.Invalid("input.content", "message content is required")
		}
		content, err := parseResponsesContent(rawContent, "input.content")
		if err != nil {
			return nil, err
		}
		item := contract.MessageItem(role, content...)
		item.ID, err = optionalString(object, "id")
		if err != nil {
			return nil, err
		}
		return []contract.Item{item}, nil
	case "function_call":
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "id": true, "call_id": true, "name": true, "arguments": true}); err != nil {
			return nil, err
		}
		callID, err := requireString(object, "call_id")
		if err != nil {
			return nil, err
		}
		name, err := requireString(object, "name")
		if err != nil {
			return nil, err
		}
		arguments, err := requireString(object, "arguments")
		if err != nil {
			return nil, err
		}
		if !jsontext.Value(arguments).IsValid() {
			return nil, contract.Invalid("input.arguments", "function call arguments must be valid JSON")
		}
		item := contract.FunctionCallItem(callID, name, arguments)
		item.ID, err = optionalString(object, "id")
		if err != nil {
			return nil, err
		}
		return []contract.Item{item}, nil
	case "function_call_output":
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "id": true, "call_id": true, "output": true}); err != nil {
			return nil, err
		}
		callID, err := requireString(object, "call_id")
		if err != nil {
			return nil, err
		}
		rawOutput, ok := object["output"]
		if !ok {
			return nil, contract.Invalid("input.output", "function call output is required")
		}
		parts, err := parseResponsesOutput(rawOutput, "input.output")
		if err != nil {
			return nil, err
		}
		item := contract.FunctionCallOutputItem(callID, parts...)
		item.ID, err = optionalString(object, "id")
		if err != nil {
			return nil, err
		}
		return []contract.Item{item}, nil
	case "reasoning":
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "id": true, "summary": true, "encrypted_content": true}); err != nil {
			return nil, err
		}
		itemID, err := optionalString(object, "id")
		if err != nil {
			return nil, err
		}
		item := contract.Item{Type: contract.ItemReasoning, ID: itemID, Status: contract.StatusCompleted}
		if rawSummary, ok := object["summary"]; ok {
			values, err := rawArray(rawSummary, "input.summary")
			if err != nil {
				return nil, err
			}
			for _, value := range values {
				summaryObject, err := rawObject(value, "input.summary")
				if err != nil {
					return nil, err
				}
				if err := rejectUnknownStrict(summaryObject, map[string]bool{"type": true, "text": true}); err != nil {
					return nil, err
				}
				typeName, ok, err := decodeString(summaryObject, "type")
				if err != nil {
					return nil, err
				}
				if ok && typeName != "summary_text" {
					return nil, contract.Unsupported("input.summary.type", "unsupported reasoning summary type "+typeName)
				}
				text, err := requireString(summaryObject, "text")
				if err != nil {
					return nil, err
				}
				item.Summary = append(item.Summary, contract.SummaryPart(text))
			}
		}
		if raw, ok := object["encrypted_content"]; ok {
			item.EncryptedContent = append(jsontext.Value(nil), raw...)
		}
		return []contract.Item{item}, nil
	default:
		return nil, contract.Unsupported("input.type", "input item type "+typeName+" is not supported")
	}
}

func parseResponsesContent(raw jsontext.Value, param string) ([]contract.Part, error) {
	return parseStringOrContent(raw, param, func(value jsontext.Value) ([]contract.Part, error) {
		values, err := rawArray(value, param)
		if err != nil {
			return nil, err
		}
		parts := make([]contract.Part, 0, len(values))
		for _, value := range values {
			object, err := rawObject(value, param)
			if err != nil {
				return nil, err
			}
			typeName, err := requireString(object, "type")
			if err != nil {
				return nil, err
			}
			switch typeName {
			case "input_text", "output_text":
				if err := rejectUnknownStrict(object, map[string]bool{"type": true, "text": true}); err != nil {
					return nil, err
				}
				text, err := requireString(object, "text")
				if err != nil {
					return nil, err
				}
				parts = append(parts, contract.TextPart(text))
			case "input_image":
				if err := rejectUnknownStrict(object, map[string]bool{"type": true, "image_url": true, "file_id": true, "detail": true}); err != nil {
					return nil, err
				}
				imageURL, ok, err := decodeString(object, "image_url")
				if err != nil {
					return nil, err
				}
				var media contract.Media
				fileID, fileIDOK, fileIDErr := decodeString(object, "file_id")
				if fileIDErr != nil {
					return nil, fileIDErr
				}
				if ok && fileIDOK {
					return nil, contract.Invalid(param, "input_image accepts only one of image_url or file_id")
				}
				detail, err := optionalString(object, "detail")
				if err != nil {
					return nil, err
				}
				if err := wire.ValidateImageDetail(detail, param+".detail"); err != nil {
					return nil, err
				}
				if ok {
					media, err = parseMediaURL(imageURL, detail)
				} else if fileIDOK {
					media = contract.AssetMedia("image/*", fileID)
				} else {
					return nil, contract.Invalid(param, "input_image requires image_url or file_id")
				}
				if err != nil {
					return nil, err
				}
				parts = append(parts, contract.Part{Type: contract.PartImage, Media: &media, Detail: detail})
			case "input_file":
				if err := rejectUnknownStrict(object, map[string]bool{"type": true, "file_data": true, "file_url": true, "file_id": true, "filename": true}); err != nil {
					return nil, err
				}
				media, filename, err := parseFileMedia(object, param)
				if err != nil {
					return nil, err
				}
				media.Filename = filename
				parts = append(parts, contract.FilePart(media))
			case "input_audio", "input_video":
				return nil, contract.Unsupported(param, typeName+" is not supported")
			default:
				return nil, contract.Unsupported(param, "unsupported Responses content type "+typeName)
			}
		}
		return parts, nil
	})
}

func parseResponsesOutput(raw jsontext.Value, param string) ([]contract.Part, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []contract.Part{contract.TextPart(text)}, nil
	}
	return parseResponsesContent(raw, param)
}

func parseResponsesTools(raw jsontext.Value) ([]contract.Tool, bool, error) {
	values, err := rawArray(raw, "tools")
	if err != nil {
		return nil, false, err
	}
	tools := make([]contract.Tool, 0, len(values))
	imageGeneration := false
	for _, value := range values {
		object, err := rawObject(value, "tools")
		if err != nil {
			return nil, false, err
		}
		typeName, ok, err := decodeString(object, "type")
		if err != nil {
			return nil, false, err
		}
		if !ok {
			typeName = "function"
		}
		switch typeName {
		case "image_generation":
			if err := rejectUnknownStrict(object, map[string]bool{"type": true}); err != nil {
				return nil, false, err
			}
			switch {
			case len(object) != 1:
				return nil, false, contract.Unsupported("tools", "image_generation does not accept options in this compatibility profile")
			case imageGeneration:
				return nil, false, contract.Invalid("tools", "only one image_generation tool is supported")
			}
			imageGeneration = true
			continue
		case "function":
		default:
			return nil, false, contract.Unsupported("tools", "only function and image_generation tools are supported")
		}
		if err := rejectUnknownStrict(object, map[string]bool{"type": true, "name": true, "description": true, "parameters": true, "strict": true}); err != nil {
			return nil, false, err
		}
		name, err := requireString(object, "name")
		if err != nil {
			return nil, false, err
		}
		description, err := optionalString(object, "description")
		if err != nil {
			return nil, false, err
		}
		tool := contract.Tool{Name: name, Description: description}
		if parameters, ok := object["parameters"]; ok {
			tool.Parameters = append(jsontext.Value(nil), parameters...)
		}
		if strict, err := decodeBool(object, "strict"); err != nil {
			return nil, false, err
		} else {
			tool.Strict = strict
		}
		tools = append(tools, tool)
	}
	return tools, imageGeneration, nil
}

func parseResponsesText(raw jsontext.Value) (string, *contract.StructuredOutput, error) {
	object, err := rawObject(raw, "text")
	if err != nil {
		return "", nil, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"format": true}); err != nil {
		return "", nil, err
	}
	if rawFormat, ok := object["format"]; ok {
		format, err := rawObject(rawFormat, "text.format")
		if err != nil {
			return "", nil, err
		}
		if err := rejectUnknownStrict(format, map[string]bool{"type": true, "name": true, "description": true, "schema": true, "strict": true}); err != nil {
			return "", nil, err
		}
		typeName, err := requireString(format, "type")
		if err != nil {
			return "", nil, err
		}
		switch typeName {
		case "text":
			return typeName, nil, nil
		case "json_object":
			return typeName, &contract.StructuredOutput{Name: "response", Schema: jsontext.Value(`{"type":"object"}`)}, nil
		case "json_schema":
			name, err := requireString(format, "name")
			if err != nil {
				return "", nil, err
			}
			schema := append(jsontext.Value(nil), format["schema"]...)
			if len(schema) == 0 || !schema.IsValid() {
				return "", nil, contract.Invalid("text.format.schema", "schema must be valid JSON")
			}
			description, err := optionalString(format, "description")
			if err != nil {
				return "", nil, err
			}
			strict, err := decodeBool(format, "strict")
			if err != nil {
				return "", nil, err
			}
			return typeName, &contract.StructuredOutput{Name: name, Description: description, Schema: schema, Strict: strict != nil && *strict}, nil
		default:
			return "", nil, contract.Unsupported("text.format", "unsupported text format "+typeName)
		}
	}
	return "text", nil, nil
}

func parseResponsesReasoning(raw jsontext.Value) (*contract.Reasoning, error) {
	object, err := rawObject(raw, "reasoning")
	if err != nil {
		return nil, err
	}
	if err := rejectUnknownStrict(object, map[string]bool{"effort": true, "summary": true}); err != nil {
		return nil, err
	}
	reasoning := &contract.Reasoning{}
	if value, ok, err := decodeString(object, "effort"); err != nil {
		return nil, err
	} else if ok {
		reasoning.Effort = value
	}
	if raw, ok := object["summary"]; ok && string(raw) != "null" {
		value, _, err := decodeString(object, "summary")
		if err != nil {
			return nil, err
		}
		switch value {
		case "none":
			reasoning.Summary = false
		case "auto", "concise", "detailed":
			reasoning.Summary = true
		default:
			return nil, contract.Unsupported("reasoning.summary", "supported values are none, auto, concise, and detailed")
		}
	}
	return reasoning, nil
}

// Adapter encodes canonical events as OpenAI Responses.
type Adapter struct{}

// NewAdapter returns the Responses API protocol adapter.
func NewAdapter() internalprotocol.Adapter { return Adapter{} }

// ValidateEvent checks whether event can be encoded for the Responses API.
func (Adapter) ValidateEvent(event contract.Event) error {
	if event.Type == contract.EventActivity {
		switch {
		case strings.TrimSpace(event.Name) == "" || strings.ContainsAny(event.Name, "./ \t\r\n"):
			return contract.Invalid("activity", "activity name must be a non-empty token without separators")
		case len(event.Data) == 0 || !event.Data.IsValid():
			return contract.Invalid("activity", "activity data must be valid JSON")
		default:
			return nil
		}
	}
	switch {
	case event.Type == contract.EventMedia && event.Part.Type != contract.PartImage:
		return contract.Unsupported("output", "OpenAI Responses compatibility exposes image results, not audio results")
	case event.Type == contract.EventItem && event.Item.Type == contract.ItemMedia && (len(event.Item.Content) != 1 || event.Item.Content[0].Type != contract.PartImage):
		return contract.Unsupported("output", "OpenAI Responses compatibility exposes image results, not audio results")
	}
	if event.Type == contract.EventItem {
		switch event.Item.Type {
		case contract.ItemMessage, contract.ItemFunctionCall, contract.ItemReasoning, contract.ItemMedia:
			if event.Item.Type == contract.ItemMessage {
				for _, part := range event.Item.Content {
					if part.Type != contract.PartText {
						return contract.Unsupported("output", "Responses message output supports text parts only")
					}
				}
			}
		default:
			return contract.Unsupported("output", "unsupported Responses output item")
		}
	}
	if event.Type == contract.EventMessage {
		for _, part := range event.Item.Content {
			if part.Type != contract.PartText {
				return contract.Unsupported("output", "Responses message streaming only supports text parts")
			}
		}
	}
	return nil
}

// Response builds a non-streaming Responses API response body.
func (Adapter) Response(req contract.Request, result execution.Result, meta responseMeta) (any, error) {
	output := make([]any, 0, len(result.Items))
	for _, item := range result.Items {
		value, err := responseItem(item)
		if err != nil {
			return nil, err
		}
		output = append(output, value)
	}
	return responseObject(req, result.Outcome, meta, output), nil
}

// Render builds the same Responses envelope used for creation, replay, and
// application-owned GET retrieval.
func Render(req contract.Request, outcome contract.Outcome, items []contract.Item, id string, created int64) (any, error) {
	return Adapter{}.Response(req, execution.Result{Items: items, Outcome: outcome}, responseMeta{
		ID:      id,
		Created: created,
		Model:   req.Target,
	})
}

// Stream returns an SSE encoder for Responses API streaming events.
func (Adapter) Stream(w http.ResponseWriter, req contract.Request, meta responseMeta, limits contract.Limits) streamEncoder {
	return &responsesStream{
		writer:    &sseWriter{w: w, limits: limits},
		request:   req,
		meta:      meta,
		indexes:   make(map[string]int),
		text:      make(map[string]string),
		toolNames: make(map[string]string),
		toolCalls: make(map[string]string),
	}
}

func responseStatus(outcome contract.Outcome) string {
	if outcome.Status == "" {
		return string(contract.StatusCompleted)
	}
	return string(outcome.Status)
}

func responseObject(req contract.Request, outcome contract.Outcome, meta responseMeta, output []any) map[string]any {
	store := false
	if req.Controls.Store != nil {
		store = *req.Controls.Store
	}
	value := map[string]any{
		"id":                   meta.ID,
		"object":               "response",
		"created_at":           meta.Created,
		"status":               responseStatus(outcome),
		"error":                nil,
		"incomplete_details":   nil,
		"instructions":         nil,
		"model":                meta.Model,
		"output":               output,
		"parallel_tool_calls":  true,
		"previous_response_id": nil,
		"reasoning":            map[string]any{"effort": nil, "summary": nil},
		"store":                store,
		"temperature":          nil,
		"text":                 map[string]any{"format": map[string]any{"type": "text"}},
		"tool_choice":          "auto",
		"tools":                []any{},
		"top_p":                nil,
	}
	if req.Instructions != "" {
		value["instructions"] = req.Instructions
	}
	if req.Controls.MaxOutputTokens != nil {
		value["max_output_tokens"] = *req.Controls.MaxOutputTokens
	} else {
		value["max_output_tokens"] = nil
	}
	if req.Controls.Temperature != nil {
		value["temperature"] = *req.Controls.Temperature
	}
	if req.Controls.TopP != nil {
		value["top_p"] = *req.Controls.TopP
	}
	if req.Controls.ParallelToolCall != nil {
		value["parallel_tool_calls"] = *req.Controls.ParallelToolCall
	}
	if req.Controls.PreviousResponseID != nil {
		value["previous_response_id"] = *req.Controls.PreviousResponseID
	}
	if req.Controls.Structured != nil {
		value["text"] = map[string]any{"format": map[string]any{"type": "json_schema", "name": req.Controls.Structured.Name, "description": req.Controls.Structured.Description, "schema": jsontext.Value(req.Controls.Structured.Schema), "strict": req.Controls.Structured.Strict}}
	}
	if len(req.Controls.Tools) > 0 {
		tools := make([]any, 0, len(req.Controls.Tools))
		for _, tool := range req.Controls.Tools {
			parameters := jsontext.Value(tool.Parameters)
			if len(parameters) == 0 {
				parameters = jsontext.Value(`{"type":"object"}`)
			}
			tools = append(tools, map[string]any{"type": "function", "name": tool.Name, "description": tool.Description, "parameters": parameters, "strict": boolPointerValue(tool.Strict)})
		}
		value["tools"] = tools
	}
	if req.Controls.ImageGeneration {
		tools := value["tools"].([]any)
		value["tools"] = append(tools, map[string]any{"type": "image_generation"})
	}
	if req.Controls.ToolChoice != nil {
		if req.Controls.ToolChoice.Mode == "function" {
			value["tool_choice"] = map[string]any{"type": "function", "name": req.Controls.ToolChoice.Name}
		} else {
			value["tool_choice"] = req.Controls.ToolChoice.Mode
		}
	}
	if outcome.Usage != nil {
		value["usage"] = responseUsage(outcome.Usage)
	}
	return value
}

func boolPointerValue(value *bool) bool { return value != nil && *value }

func responseUsage(usage *contract.Usage) map[string]any {
	return map[string]any{
		"input_tokens":          usage.InputTokens,
		"output_tokens":         usage.OutputTokens,
		"total_tokens":          usage.TotalTokens,
		"input_tokens_details":  map[string]any{"cached_tokens": usage.CachedTokens},
		"output_tokens_details": map[string]any{"reasoning_tokens": usage.ReasoningTokens},
	}
}

func responseItem(item contract.Item) (map[string]any, error) {
	status := item.Status
	if status == "" {
		status = contract.StatusCompleted
	}
	switch item.Type {
	case contract.ItemMessage:
		for _, part := range item.Content {
			if part.Type != contract.PartText {
				return nil, contract.Unsupported("output", "Responses message output supports text parts only")
			}
		}
		return map[string]any{"id": item.ID, "type": "message", "status": status, "role": item.Role, "content": outputTextParts(item.Content)}, nil
	case contract.ItemFunctionCall:
		return map[string]any{"id": item.ID, "type": "function_call", "status": status, "call_id": item.CallID, "name": item.Name, "arguments": item.Arguments}, nil
	case contract.ItemFunctionCallOutput:
		output, err := responseFunctionOutput(item.Output)
		if err != nil {
			return nil, err
		}
		return map[string]any{"id": item.ID, "type": "function_call_output", "status": status, "call_id": item.CallID, "output": output}, nil
	case contract.ItemReasoning:
		if len(item.Data) > 0 {
			return nil, contract.Unsupported("output", "Responses reasoning output cannot carry opaque data")
		}
		summary := make([]any, 0, len(item.Summary))
		for _, part := range item.Summary {
			summary = append(summary, map[string]any{"type": "summary_text", "text": part.Text})
		}
		value := map[string]any{"id": item.ID, "type": "reasoning", "status": status, "summary": summary}
		if len(item.EncryptedContent) > 0 {
			value["encrypted_content"] = jsontext.Value(item.EncryptedContent)
		}
		return value, nil
	case contract.ItemMedia:
		if len(item.Content) != 1 || item.Content[0].Type != contract.PartImage || item.Content[0].Media == nil {
			return nil, contract.Unsupported("output", "Responses image output requires one image media part")
		}
		if len(item.Content[0].Media.Data) == 0 {
			return nil, contract.Unsupported("output", "Responses image output requires inline image data")
		}
		return map[string]any{"id": item.ID, "type": "image_generation_call", "status": status, "result": base64.StdEncoding.EncodeToString(item.Content[0].Media.Data)}, nil
	default:
		return nil, contract.Unsupported("output", "unsupported Responses output item")
	}
}

func responseFunctionOutput(parts []contract.Part) (any, error) {
	if len(parts) == 1 && parts[0].Type == contract.PartText {
		return parts[0].Text, nil
	}
	values := make([]any, 0, len(parts))
	for _, part := range parts {
		switch part.Type {
		case contract.PartText:
			values = append(values, map[string]any{"type": "input_text", "text": part.Text})
		case contract.PartImage, contract.PartFile:
			if part.Media == nil {
				return nil, errors.New("function output media is missing")
			}
			if len(part.Media.Data) > 0 {
				data, err := mediaDataURL(*part.Media)
				if err != nil {
					return nil, err
				}
				if part.Type == contract.PartImage {
					values = append(values, map[string]any{"type": "input_image", "image_url": data, "detail": "auto"})
				} else {
					values = append(values, map[string]any{"type": "input_file", "file_data": base64.StdEncoding.EncodeToString(part.Media.Data), "filename": part.Media.Filename})
				}
			} else {
				return nil, contract.Unsupported("output", "function output media must be inline")
			}
		default:
			return nil, contract.Unsupported("output", "unsupported function output part")
		}
	}
	return values, nil
}

type responsesStream struct {
	writer    *sseWriter
	request   contract.Request
	meta      responseMeta
	sequence  int
	indexes   map[string]int
	text      map[string]string
	toolNames map[string]string
	toolCalls map[string]string
	next      int
	started   bool
}

// Started reports whether the Responses SSE stream has begun.
func (s *responsesStream) Started() bool { return s.writer.started }

func (s *responsesStream) emit(eventType string, value map[string]any) error {
	value["type"] = eventType
	value["sequence_number"] = s.sequence
	s.sequence++
	return s.writer.write(eventType, value)
}

func (s *responsesStream) start() error {
	if s.started {
		return nil
	}
	base := responseObject(s.request, contract.Outcome{Status: contract.StatusInProgress}, s.meta, []any{})
	if err := s.emit("response.created", map[string]any{"response": base}); err != nil {
		return err
	}
	if err := s.emit("response.in_progress", map[string]any{"response": base}); err != nil {
		return err
	}
	s.started = true
	return nil
}

func (s *responsesStream) addMessage(id string, index int) error {
	if _, ok := s.indexes[id]; ok {
		return nil
	}
	s.indexes[id] = index
	if index >= s.next {
		s.next = index + 1
	}
	return s.emit("response.output_item.added", map[string]any{"output_index": index, "item": map[string]any{"id": id, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{}}})
}

// Event encodes one canonical event as a Responses SSE frame.
func (s *responsesStream) Event(event contract.Event) error {
	if err := s.start(); err != nil {
		return err
	}
	switch event.Type {
	case contract.EventMessage:
		index := s.next
		if err := s.addMessage(event.Item.ID, index); err != nil {
			return err
		}
		if len(event.Item.Content) == 0 {
			return s.finishMessage(event.Item.ID, index, nil)
		}
		parts := make([]any, 0, len(event.Item.Content))
		for contentIndex, part := range event.Item.Content {
			if part.Type != contract.PartText {
				return contract.Unsupported("output", "Responses message streaming only supports text parts")
			}
			parts = append(parts, map[string]any{"type": "output_text", "text": part.Text, "annotations": []any{}})
			if err := s.emit("response.content_part.added", map[string]any{"item_id": event.Item.ID, "output_index": index, "content_index": contentIndex, "part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}}}); err != nil {
				return err
			}
			if part.Text != "" {
				if err := s.emit("response.output_text.delta", map[string]any{"item_id": event.Item.ID, "output_index": index, "content_index": contentIndex, "delta": part.Text}); err != nil {
					return err
				}
			}
			if err := s.finishContentPart(event.Item.ID, index, contentIndex, part.Text); err != nil {
				return err
			}
		}
		return s.finishMessage(event.Item.ID, index, parts)
	case contract.EventTextDelta:
		index, ok := s.indexes[event.ItemID]
		if !ok {
			index = s.next
			if err := s.addMessage(event.ItemID, index); err != nil {
				return err
			}
			if err := s.emit("response.content_part.added", map[string]any{"item_id": event.ItemID, "output_index": index, "content_index": 0, "part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}}}); err != nil {
				return err
			}
		}
		s.text[event.ItemID] += event.Delta
		return s.emit("response.output_text.delta", map[string]any{"item_id": event.ItemID, "output_index": index, "content_index": 0, "delta": event.Delta})
	case contract.EventTextDone:
		index, ok := s.indexes[event.ItemID]
		if !ok {
			return fmt.Errorf("unknown text item %q", event.ItemID)
		}
		textValue := event.Text
		if textValue == "" {
			textValue = s.text[event.ItemID]
		}
		if err := s.finishContentPart(event.ItemID, index, 0, textValue); err != nil {
			return err
		}
		return s.finishMessage(event.ItemID, index, []any{map[string]any{"type": "output_text", "text": textValue, "annotations": []any{}}})
	case contract.EventToolCallStart:
		return s.startTool(event.ItemID, event.CallID, event.Name, s.next)
	case contract.EventToolCall:
		if err := s.startTool(event.ItemID, event.CallID, event.Name, s.next); err != nil {
			return err
		}
		index := s.indexes[event.ItemID]
		if event.Arguments != "" {
			if err := s.emit("response.function_call_arguments.delta", map[string]any{"item_id": event.ItemID, "output_index": index, "delta": event.Arguments}); err != nil {
				return err
			}
		}
		return s.finishTool(event.ItemID, event.CallID, event.Name, event.Arguments, index)
	case contract.EventToolCallDelta:
		index, ok := s.indexes[event.ItemID]
		if !ok {
			return fmt.Errorf("unknown tool item %q", event.ItemID)
		}
		return s.emit("response.function_call_arguments.delta", map[string]any{"item_id": event.ItemID, "output_index": index, "delta": event.Delta})
	case contract.EventToolCallDone:
		index, ok := s.indexes[event.ItemID]
		if !ok {
			return fmt.Errorf("unknown tool item %q", event.ItemID)
		}
		return s.finishTool(event.ItemID, event.CallID, "", event.Arguments, index)
	case contract.EventReasoning:
		item := event.Item
		if len(item.Data) > 0 {
			return contract.Unsupported("output", "Responses reasoning output cannot carry opaque data")
		}
		if item.ID == "" {
			item.ID = event.ItemID
		}
		if item.ID == "" {
			item.ID = newID("rs_")
		}
		index := s.next
		s.indexes[item.ID] = index
		s.next++
		reasoning := map[string]any{"id": item.ID, "type": "reasoning", "status": "in_progress", "summary": []any{}}
		if len(item.EncryptedContent) > 0 {
			reasoning["encrypted_content"] = jsontext.Value(item.EncryptedContent)
		}
		if err := s.emit("response.output_item.added", map[string]any{"output_index": index, "item": reasoning}); err != nil {
			return err
		}
		if len(item.Summary) > 0 {
			if err := s.emit("response.reasoning_summary_part.added", map[string]any{"item_id": item.ID, "output_index": index, "summary_index": 0}); err != nil {
				return err
			}
			for _, part := range item.Summary {
				if err := s.emit("response.reasoning_summary_text.delta", map[string]any{"item_id": item.ID, "output_index": index, "summary_index": 0, "delta": part.Text}); err != nil {
					return err
				}
			}
		}
		reasoning["status"] = "completed"
		reasoning["summary"] = reasoningSummary(item.Summary)
		return s.emit("response.output_item.done", map[string]any{"output_index": index, "item": reasoning})
	case contract.EventMedia:
		item := contract.Item{Type: contract.ItemMedia, ID: event.ItemID, Status: contract.StatusCompleted, Content: []contract.Part{event.Part}}
		return s.eventItem(item)
	case contract.EventItem:
		return s.eventItem(event.Item)
	case contract.EventActivity:
		if !s.meta.Activity {
			return contract.Unsupported("output", "activity events were not enabled for this request")
		}
		name := strings.TrimSpace(event.Name)
		if name == "" || strings.ContainsAny(name, "./ \t\r\n") {
			return contract.Invalid("activity", "activity name must be a non-empty token without separators")
		}
		return s.emit("response.activity."+name, map[string]any{
			"activity": map[string]any{"name": name, "data": jsontext.Value(event.Data)},
		})
	default:
		return fmt.Errorf("unsupported Responses event %q", event.Type)
	}
}

func (s *responsesStream) eventItem(item contract.Item) error {
	switch item.Type {
	case contract.ItemMessage:
		return s.Event(contract.Event{Type: contract.EventMessage, Item: item})
	case contract.ItemFunctionCall:
		return s.Event(contract.Event{Type: contract.EventToolCall, ItemID: item.ID, CallID: item.CallID, Name: item.Name, Arguments: item.Arguments})
	case contract.ItemReasoning:
		return s.Event(contract.Event{Type: contract.EventReasoning, Item: item})
	case contract.ItemMedia:
		if len(item.Content) != 1 || item.Content[0].Type != contract.PartImage || item.Content[0].Media == nil || len(item.Content[0].Media.Data) == 0 {
			return contract.Unsupported("output", "Responses image output requires inline image data")
		}
		index := s.next
		s.next++
		image := map[string]any{"id": item.ID, "type": "image_generation_call", "status": "in_progress"}
		if err := s.emit("response.output_item.added", map[string]any{"output_index": index, "item": image}); err != nil {
			return err
		}
		image["status"] = "completed"
		image["result"] = base64.StdEncoding.EncodeToString(item.Content[0].Media.Data)
		return s.emit("response.output_item.done", map[string]any{"output_index": index, "item": image})
	default:
		return contract.Unsupported("output", "unsupported Responses output item")
	}
}

func (s *responsesStream) startTool(itemID, callID, name string, index int) error {
	if _, ok := s.indexes[itemID]; ok {
		return nil
	}
	s.indexes[itemID] = index
	s.toolNames[itemID] = name
	s.toolCalls[itemID] = callID
	s.next = index + 1
	return s.emit("response.output_item.added", map[string]any{"output_index": index, "item": map[string]any{"id": itemID, "type": "function_call", "status": "in_progress", "call_id": callID, "name": name, "arguments": ""}})
}

func (s *responsesStream) finishTool(itemID, callID, name, arguments string, index int) error {
	if callID == "" {
		callID = s.toolCalls[itemID]
	}
	if name == "" {
		name = s.toolNames[itemID]
	}
	if err := s.emit("response.function_call_arguments.done", map[string]any{"item_id": itemID, "output_index": index, "arguments": arguments}); err != nil {
		return err
	}
	if err := s.emit("response.output_item.done", map[string]any{"output_index": index, "item": map[string]any{"id": itemID, "type": "function_call", "status": "completed", "call_id": callID, "name": name, "arguments": arguments}}); err != nil {
		return err
	}
	delete(s.indexes, itemID)
	delete(s.toolNames, itemID)
	delete(s.toolCalls, itemID)
	return nil
}

func (s *responsesStream) finishMessage(itemID string, index int, parts []any) error {
	return s.emit("response.output_item.done", map[string]any{"output_index": index, "item": map[string]any{"id": itemID, "type": "message", "status": "completed", "role": "assistant", "content": parts}})
}

func (s *responsesStream) finishContentPart(itemID string, index, contentIndex int, textValue string) error {
	if err := s.emit("response.output_text.done", map[string]any{"item_id": itemID, "output_index": index, "content_index": contentIndex, "text": textValue}); err != nil {
		return err
	}
	if err := s.emit("response.content_part.done", map[string]any{"item_id": itemID, "output_index": index, "content_index": contentIndex, "part": map[string]any{"type": "output_text", "text": textValue, "annotations": []any{}}}); err != nil {
		return err
	}
	return nil
}

func reasoningSummary(parts []contract.Part) []any {
	values := make([]any, 0, len(parts))
	for _, part := range parts {
		values = append(values, map[string]any{"type": "summary_text", "text": part.Text})
	}
	return values
}

// Complete emits the response.completed event and closes the SSE stream.
func (s *responsesStream) Complete(outcome contract.Outcome, items []contract.Item) error {
	output := make([]any, 0, len(items))
	for _, item := range items {
		value, err := responseItem(item)
		if err != nil {
			return err
		}
		output = append(output, value)
	}
	if err := s.start(); err != nil {
		return err
	}
	if err := s.emit("response.completed", map[string]any{"response": responseObject(s.request, outcome, s.meta, output)}); err != nil {
		return err
	}
	return s.writer.done()
}

// Fail writes a Responses error event or HTTP error envelope.
func (s *responsesStream) Fail(err error) error {
	apiErr := internalprotocol.AsAPIError(err)
	if !s.writer.started {
		return writeProtocolErrorAndReturn(s.writer.w, protocolResponses, apiErr)
	}
	if startErr := s.start(); startErr != nil {
		return startErr
	}
	response := responseObject(s.request, contract.Outcome{Status: contract.StatusFailed, StopReason: contract.StopError}, s.meta, []any{})
	response["error"] = map[string]any{"type": apiErr.Code, "message": apiErr.Message}
	if err := s.emit("response.failed", map[string]any{"response": response}); err != nil {
		return err
	}
	return s.writer.done()
}
