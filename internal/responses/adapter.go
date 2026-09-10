package responses

import (
	"encoding/json/jsontext"
	"net/http"
	"strings"

	"github.com/kelindar/llmux/chat"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/kelindar/llmux/internal/wire"
	"github.com/rs/xid"
)

type parsedRequest = internalprotocol.ParsedRequest
type responseMeta = internalprotocol.Meta
type streamEncoder = internalprotocol.StreamEncoder
type protocol = internalprotocol.Kind

const protocolResponses = internalprotocol.Responses

func newID() string { return xid.New().String() }

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
func parseStringOrContent(raw jsontext.Value, param string, parser func(jsontext.Value) ([]chat.Part, error)) ([]chat.Part, error) {
	return wire.ParseStringOrContent(raw, param, parser)
}
func parseMediaURL(value, detail string) (chat.Media, error) {
	return wire.ParseMediaURL(value, detail)
}
func parseFileMedia(object map[string]jsontext.Value, param string) (chat.Media, string, error) {
	return wire.ParseFileMedia(object, param)
}
func inputTextParts(parts []chat.Part) []map[string]any { return wire.InputTextParts(parts) }
func outputTextParts(parts []chat.Part) []wireOutputText {
	out := make([]wireOutputText, 0, len(parts))
	for _, part := range parts {
		if part.Type == chat.PartText {
			out = append(out, wireOutputText{Type: "output_text", Text: part.Text, Annotations: []any{}})
		}
	}
	return out
}
func collectText(parts []chat.Part) string          { return wire.CollectText(parts) }
func mediaDataURL(media chat.Media) (string, error) { return wire.MediaDataURL(media) }
func fmtError(param, message string, err error) *chat.Error {
	return wire.Error(param, message, err)
}
func validRole(role chat.Role) bool { return chat.ValidRole(role) }
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

type sseWriter struct {
	w       http.ResponseWriter
	limits  chat.Limits
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

// Adapter encodes canonical events as OpenAI Responses.
type Adapter struct{}

// NewAdapter returns the Responses API protocol adapter.
func NewAdapter() internalprotocol.Adapter { return Adapter{} }

// ValidateEvent checks whether event can be encoded for the Responses API.
func (Adapter) ValidateEvent(event chat.Event) error {
	if event.Type == chat.EventActivity {
		switch {
		case strings.TrimSpace(event.Name) == "" || strings.ContainsAny(event.Name, "./ \t\r\n"):
			return chat.Invalid("activity", "activity name must be a non-empty token without separators")
		case len(event.Data) == 0 || !event.Data.IsValid():
			return chat.Invalid("activity", "activity data must be valid JSON")
		default:
			return nil
		}
	}
	if event.Type != chat.EventItem {
		return nil
	}
	switch event.Item.Type {
	case chat.ItemMedia:
		if len(event.Item.Content) != 1 || event.Item.Content[0].Type != chat.PartImage {
			return chat.Unsupported("output", "OpenAI Responses compatibility exposes image results, not audio results")
		}
	case chat.ItemMessage:
		for _, part := range event.Item.Content {
			if part.Type != chat.PartText {
				return chat.Unsupported("output", "Responses message output supports text parts only")
			}
		}
	case chat.ItemFunctionCall, chat.ItemReasoning:
	default:
		return chat.Unsupported("output", "unsupported Responses output item")
	}
	return nil
}

// Stream returns an SSE encoder for Responses API streaming events.
func (Adapter) Stream(w http.ResponseWriter, parsed parsedRequest, meta *responseMeta, limits chat.Limits) streamEncoder {
	return &responsesStream{
		writer:    &sseWriter{w: w, limits: limits},
		request:   parsed.Request,
		meta:      meta,
		indexes:   make(map[string]int),
		text:      make(map[string]string),
		toolArgs:  make(map[string]string),
		toolNames: make(map[string]string),
		toolCalls: make(map[string]string),
	}
}
