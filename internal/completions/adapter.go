// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
package completions

import (
	"encoding/json/jsontext"
	"net/http"

	chat "github.com/kelindar/llmux/chat"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/kelindar/llmux/internal/wire"
	"github.com/rs/xid"
)

type parsedRequest = internalprotocol.ParsedRequest
type responseMeta = internalprotocol.Meta
type streamEncoder = internalprotocol.StreamEncoder
type protocol = internalprotocol.Kind

const protocolChat = internalprotocol.Chat

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
func parseChatContent(raw jsontext.Value) ([]chat.Part, error) {
	return wire.ParseChatContent(raw)
}
func parseMediaURL(value, detail string) (chat.Media, error) {
	return wire.ParseMediaURL(value, detail)
}
func parseFileMedia(object map[string]jsontext.Value, param string) (chat.Media, string, error) {
	return wire.ParseFileMedia(object, param)
}
func audioMIME(format string) string                     { return wire.AudioMIME(format) }
func outputTextParts(parts []chat.Part) []map[string]any { return wire.OutputTextParts(parts) }
func mediaDataURL(media chat.Media) (string, error)      { return wire.MediaDataURL(media) }
func fmtError(param, message string, err error) *chat.Error {
	return wire.Error(param, message, err)
}
func validRole(role chat.Role) bool { return chat.ValidRole(role) }

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

// Adapter encodes canonical events as OpenAI Chat Completions.
type Adapter struct{}

// NewAdapter returns the Chat Completions protocol adapter.
func NewAdapter() internalprotocol.Adapter { return Adapter{} }

// ValidateEvent checks whether event can be encoded for Chat Completions.
func (Adapter) ValidateEvent(event chat.Event) error {
	switch event.Type {
	case chat.EventActivity:
		return chat.Unsupported("output", "activity events are not supported on Chat Completions")
	case chat.EventItem:
		switch event.Item.Type {
		case chat.ItemReasoning:
			return chat.Unsupported("output", "Chat Completions has no public reasoning-summary output mapping")
		case chat.ItemMedia:
			if len(event.Item.Content) != 1 || event.Item.Content[0].Type != chat.PartAudio {
				return chat.Unsupported("output", "Chat Completions supports audio output here, not generated image output")
			}
		case chat.ItemMessage:
			for _, part := range event.Item.Content {
				if part.Type != chat.PartText && part.Type != chat.PartAudio {
					return chat.Unsupported("output", "Chat Completions supports text and audio output here")
				}
			}
		case chat.ItemFunctionCall:
		default:
			return chat.Unsupported("output", "unsupported Chat Completions output item")
		}
	}
	return nil
}

// Stream returns an SSE encoder for Chat Completions streaming chunks.
func (Adapter) Stream(w http.ResponseWriter, parsed parsedRequest, meta *responseMeta, limits chat.Limits) streamEncoder {
	return &chatStream{writer: &sseWriter{w: w, limits: limits}, request: parsed.Request, meta: meta, includeUsage: parsed.IncludeUsage}
}
