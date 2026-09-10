package llmux

import (
	"net/http"
	"strings"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/anthropic"
	completions "github.com/kelindar/llmux/internal/completions"
	"github.com/kelindar/llmux/internal/responses"
)

func (h *Handler) serveChat(w http.ResponseWriter, r *http.Request) {
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
	parsed, err := completions.ParseRequest(object)
	if err != nil {
		writeProtocolError(w, protocolChat, err)
		return
	}
	h.serveParsed(w, r, parsed)
}

func (h *Handler) serveResponses(w http.ResponseWriter, r *http.Request) {
	body, err := h.readBody(w, r, h.limits.MaxRequestBytes)
	if err != nil {
		writeProtocolError(w, protocolResponses, err)
		return
	}
	object, err := decodeObject(body)
	if err != nil {
		writeProtocolError(w, protocolResponses, fmtError("body", "request body must be valid JSON", err))
		return
	}
	parsed, err := responses.ParseRequest(object)
	if err != nil {
		writeProtocolError(w, protocolResponses, err)
		return
	}
	h.serveParsed(w, r, parsed)
}

func (h *Handler) serveMessages(w http.ResponseWriter, r *http.Request) {
	if strings.TrimSpace(r.Header.Get("anthropic-version")) == "" {
		writeProtocolError(w, protocolAnthropic, chat.Invalid("anthropic-version", "anthropic-version header is required"))
		return
	}
	body, err := h.readBody(w, r, h.limits.MaxRequestBytes)
	if err != nil {
		writeProtocolError(w, protocolAnthropic, err)
		return
	}
	object, err := decodeObject(body)
	if err != nil {
		writeProtocolError(w, protocolAnthropic, fmtError("body", "request body must be valid JSON", err))
		return
	}
	parsed, err := anthropic.ParseRequest(object)
	if err != nil {
		writeProtocolError(w, protocolAnthropic, err)
		return
	}
	h.serveParsed(w, r, parsed)
}
