// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package llmux

import (
	"context"
	"net/http"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/mcp"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
)

// mcpProtocolVersion is the MCP protocol revision served at /mcp. Requests
// at older revisions are rejected; llmux does not maintain legacy handshakes
// on this endpoint. See the README for the supported-revision policy.
const mcpProtocolVersion = mcp.ProtocolVersion

// WithMCP enables the optional MCP endpoint at the exact path /mcp (over
// stateless Streamable HTTP). It takes no arguments: the exposed tools come
// from Catalog.List, filtered to entries whose Info.Tool is nonempty. A nil
// catalog yields an empty tool catalog, never implicit enumeration.
//
// The transport is constructed after all options are applied, so option
// ordering does not matter.
func WithMCP() Option {
	return func(h *Handler) { h.mcpEnabled = true }
}

// serveMCP delegates to the internal MCP transport.
func (h *Handler) serveMCP(w http.ResponseWriter, r *http.Request) {
	h.mcp.ServeHTTP(w, r)
}

// hostAdapter implements the internal/mcp Host seam on Handler. MCP calls the
// same prepareTurn/runTurn lifecycle as ordinary chat requests.
type hostAdapter struct{ h *Handler }

func (a hostAdapter) List(ctx context.Context) (map[string]chat.Info, error) {
	return a.h.projectCatalog(ctx)
}

func (a hostAdapter) Run(ctx context.Context, idempotencyKey string, parsed *internalprotocol.ParsedRequest) (chat.Response, error) {
	return a.h.runTurn(ctx, idempotencyKey, parsed)
}

func (a hostAdapter) Limits() chat.Limits { return a.h.limits }

func (a hostAdapter) LogError(ctx context.Context, err error) { a.h.logError(ctx, err) }
