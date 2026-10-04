// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package agui

import (
	"cmp"
	"encoding/json/jsontext"
	"net/http"
	"strings"

	"github.com/kelindar/llmux/chat"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
)

type stream struct {
	w        *internalprotocol.SSEWriter
	response http.ResponseWriter
	parsed   internalprotocol.ParsedRequest
	meta     *internalprotocol.Meta
	started  bool
	openText string
	open     map[string]bool
	emitted  map[string]bool
}

type event struct {
	Type            string         `json:"type"`
	ThreadID        string         `json:"threadId,omitempty"`
	RunID           string         `json:"runId,omitempty"`
	ProtocolVersion string         `json:"protocolVersion,omitempty"`
	MessageID       string         `json:"messageId,omitempty"`
	Role            string         `json:"role,omitempty"`
	Delta           *string        `json:"delta,omitempty"`
	ActivityType    string         `json:"activityType,omitempty"`
	Content         jsontext.Value `json:"content,omitempty"`
	Name            string         `json:"name,omitempty"`
	Value           any            `json:"value,omitempty"`
	Message         string         `json:"message,omitempty"`
	Code            string         `json:"code,omitempty"`
	Result          any            `json:"result,omitempty"`
}

type runResult struct {
	ResponseID string            `json:"responseId"`
	Metadata   map[string]string `json:"metadata"`
}

type uiValue struct {
	ID        string         `json:"id"`
	Format    string         `json:"format"`
	Version   string         `json:"version"`
	CatalogID string         `json:"catalogId"`
	Payload   jsontext.Value `json:"payload"`
}

// Started reports whether SSE output has begun.
func (s *stream) Started() bool { return s.w.Started() }

func (s *stream) emit(value any) error { return s.w.Write("", value) }

func (s *stream) threadID() string {
	if s.meta == nil {
		return s.parsed.Thread
	}
	return cmp.Or(s.meta.Response.Metadata["thread_id"], s.parsed.Thread)
}

func (s *stream) start() error {
	if s.started {
		return nil
	}
	threadID := s.threadID()
	if threadID == "" || s.parsed.RunID == "" {
		return chat.Invalid("run", "threadId and runId are required for AG-UI streaming")
	}
	if err := s.emit(event{Type: "RUN_STARTED", ThreadID: threadID, RunID: s.parsed.RunID, ProtocolVersion: "1.0"}); err != nil {
		return err
	}
	s.started = true
	return nil
}

// Event encodes a canonical event as AG-UI 1.0 SSE data.
func (s *stream) Event(value chat.Event) error {
	if value.Type != chat.EventItem || value.Item.Type != chat.ItemExtension {
		if err := (Adapter{}).ValidateEvent(value); err != nil {
			return err
		}
	}
	if err := s.start(); err != nil {
		return err
	}
	switch value.Type {
	case chat.EventTextDelta:
		id := cmp.Or(value.ItemID, s.openText)
		if id == "" {
			return chat.Invalid("output.id", "text message ID is required")
		}
		if s.emitted[id] {
			return chat.Invalid("output", "text message is already closed")
		}
		if !s.open[id] {
			if err := s.emit(event{Type: "TEXT_MESSAGE_START", MessageID: id, Role: "assistant"}); err != nil {
				return err
			}
			s.open[id] = true
		}
		s.openText = id
		return s.textContent(id, value.Delta)
	case chat.EventTextDone:
		id := cmp.Or(value.ItemID, s.openText)
		if id == "" || !s.open[id] {
			return chat.Invalid("output", "text message is not open")
		}
		if err := s.emit(event{Type: "TEXT_MESSAGE_END", MessageID: id}); err != nil {
			return err
		}
		delete(s.open, id)
		s.emitted[id] = true
		if s.openText == id {
			s.openText = ""
		}
		return nil
	case chat.EventActivity:
		id := cmp.Or(value.ItemID, s.parsed.RunID)
		return s.emit(event{Type: "ACTIVITY_SNAPSHOT", MessageID: id, ActivityType: value.Name, Content: jsontext.Value(value.Data)})
	case chat.EventItem:
		switch value.Item.Type {
		case chat.ItemMessage:
			return s.message(value.Item)
		case chat.ItemExtension:
			// UI items publish only after the finalized response has persisted.
			return nil
		default:
			return chat.Unsupported("output", "AG-UI does not expose client tools or this output item")
		}
	default:
		return chat.Unsupported("output", "AG-UI does not expose client tools or this event")
	}
}

func (s *stream) message(item chat.Item) error {
	if err := (Adapter{}).ValidateEvent(chat.OutputItem(item)); err != nil {
		return err
	}
	id := item.ID
	if id == "" {
		return chat.Invalid("output.id", "text message ID is required")
	}
	if s.open[id] || s.emitted[id] {
		return chat.Invalid("output", "duplicate text message ID")
	}
	if err := s.emit(event{Type: "TEXT_MESSAGE_START", MessageID: id, Role: "assistant"}); err != nil {
		return err
	}
	s.open[id] = true
	for _, part := range item.Content {
		if err := s.textContent(id, part.Text); err != nil {
			return err
		}
	}
	if err := s.emit(event{Type: "TEXT_MESSAGE_END", MessageID: id}); err != nil {
		return err
	}
	delete(s.open, id)
	s.emitted[id] = true
	return nil
}

func (s *stream) textContent(id, delta string) error {
	if delta == "" {
		return nil
	}
	return s.emit(event{Type: "TEXT_MESSAGE_CONTENT", MessageID: id, Delta: &delta})
}

func (s *stream) ui(item chat.Item) error {
	ui, err := parseUIItem(item)
	if err != nil {
		return err
	}
	if ui.ID == "" {
		return chat.Invalid("output.id", "UI item ID is required for streaming")
	}
	support := s.parsed.Request.Controls.Extensions["x-ui"]
	if err := matchesOutput(ui, support); err != nil {
		return err
	}
	return s.emit(event{
		Type: "CUSTOM", Name: "llmux.item",
		Value: uiValue{ID: ui.ID, Format: ui.Format, Version: ui.Version, CatalogID: ui.CatalogID, Payload: ui.Payload},
	})
}

// Complete emits finalized text and UI items, then the matching terminal event.
func (s *stream) Complete(outcome chat.Outcome, items []chat.Item) error {
	if err := s.start(); err != nil {
		return err
	}
	status := outcome.Status
	if s.meta != nil {
		status = cmp.Or(s.meta.Response.Status, status)
	}
	status = cmp.Or(status, chat.StatusCompleted)
	for _, item := range items {
		switch item.Type {
		case chat.ItemMessage:
			switch status {
			case chat.StatusCompleted, chat.StatusFailed, chat.StatusCancelled, chat.StatusIncomplete:
				if !s.emitted[item.ID] {
					if err := s.message(item); err != nil {
						return err
					}
				}
			}
		case chat.ItemExtension:
			if status == chat.StatusCompleted {
				if err := s.ui(item); err != nil {
					return err
				}
			}
		default:
			return chat.Unsupported("output", "AG-UI does not expose client tools or this output item")
		}
	}
	if status != chat.StatusCompleted {
		message, code := s.terminalError(status)
		return s.emit(event{Type: "RUN_ERROR", Message: message, Code: code})
	}
	metadata := map[string]string{}
	responseID := ""
	if s.meta != nil {
		responseID = s.meta.Response.ID
		if s.meta.Response.Metadata != nil {
			metadata = s.meta.Response.Metadata
		}
	}
	return s.emit(event{Type: "RUN_FINISHED", ThreadID: s.threadID(), RunID: s.parsed.RunID, Result: runResult{ResponseID: responseID, Metadata: metadata}})
}

func (s *stream) terminalError(status chat.Status) (string, string) {
	if s.meta != nil && s.meta.Response.Error != nil {
		err := s.meta.Response.Error
		message := err.Message
		code := err.Code
		return cmp.Or(message, "run failed"), cmp.Or(code, string(status))
	}
	if s.meta != nil && status == chat.StatusIncomplete && s.meta.Response.Incomplete != "" {
		return s.meta.Response.Incomplete, string(status)
	}
	return "run " + string(status), string(status)
}

// Fail writes a protocol error before SSE starts or a sanitized RUN_ERROR after.
func (s *stream) Fail(err error) error {
	if !s.w.Started() {
		internalprotocol.WriteError(s.response, internalprotocol.AGUI, err)
		return err
	}
	if startErr := s.start(); startErr != nil {
		return startErr
	}
	apiErr := internalprotocol.AsError(err)
	message, code := apiErr.Message, apiErr.Code
	if strings.TrimSpace(message) == "" {
		message = "run failed"
	}
	return s.emit(event{Type: "RUN_ERROR", Message: message, Code: code})
}
