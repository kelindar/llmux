// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package anthropic

import (
	"encoding/json/jsontext"
	"net/http"

	"github.com/kelindar/llmux/chat"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/kelindar/llmux/internal/wire"
	"github.com/rs/xid"
)

type parsedRequest = internalprotocol.ParsedRequest
type responseMeta = internalprotocol.Meta
type streamEncoder = internalprotocol.StreamEncoder
type protocol = internalprotocol.Kind

const protocolAnthropic = internalprotocol.Anthropic

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

func asAPIError(err error) *chat.Error { return internalprotocol.AsError(err) }
func anthropicErrorType(err *chat.Error) string {
	return internalprotocol.AnthropicErrorType(err)
}
func writeProtocolErrorAndReturn(w http.ResponseWriter, p protocol, err error) error {
	internalprotocol.WriteError(w, p, err)
	return err
}

// Adapter encodes canonical events as Anthropic Messages.
type Adapter struct{}

// NewAdapter returns the Anthropic Messages protocol adapter.
func NewAdapter() internalprotocol.Adapter { return Adapter{} }

// ValidateEvent checks whether event can be encoded for Anthropic Messages.
func (Adapter) ValidateEvent(event chat.Event) error {
	switch event.Type {
	case chat.EventActivity:
		return chat.Unsupported("output", "Anthropic Messages output supports text and tool_use only")
	case chat.EventItem:
		switch event.Item.Type {
		case chat.ItemMessage:
			for _, part := range event.Item.Content {
				if part.Type != chat.PartText {
					return chat.Unsupported("output", "Anthropic Messages output supports text and tool_use only")
				}
			}
		case chat.ItemFunctionCall:
		default:
			return chat.Unsupported("output", "Anthropic Messages output supports text and tool_use only")
		}
	}
	return nil
}

// Stream returns an SSE encoder for Anthropic Messages streaming events.
func (Adapter) Stream(w http.ResponseWriter, parsed parsedRequest, meta *responseMeta, limits chat.Limits) streamEncoder {
	return &anthropicStream{
		writer:     &sseWriter{w: w, limits: limits},
		request:    parsed.Request,
		meta:       meta,
		toolBlocks: make(map[string]int),
		toolNames:  make(map[string]string),
		toolCalls:  make(map[string]string),
	}
}
