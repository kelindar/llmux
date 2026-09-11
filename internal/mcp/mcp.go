// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
// Package mcp implements the optional MCP endpoint for llmux. It exposes
// application-owned agents as MCP tools over stateless Streamable HTTP,
// reusing the shared agent execution and lifecycle machinery through the
// Host seam. MCP SDK types never cross back into the public agent contract.
package mcp

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"runtime/debug"
	"slices"
	"strings"
	"sync"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/protocol"
	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

// ProtocolVersion is the MCP protocol revision served by Transport. Requests
// at older revisions are rejected; legacy initialize handshakes cannot
// proceed against this endpoint.
const ProtocolVersion = "2026-07-28"

// toolInputSchema is the fixed internal input schema shared by every MCP
// agent tool. Tools accept exactly one non-empty string message; the schema
// is not configurable and no other arguments are supported.
const toolInputSchema = `{"type":"object","properties":{"message":{"type":"string"}},"required":["message"],"additionalProperties":false}`

// Host is the execution seam provided by the llmux Handler. It keeps the MCP
// adapter from duplicating lifecycle machinery: resolution, validation,
// acceptance (including CatalogAgent), preparation, execution, and Finish.
type Host interface {
	// List returns the caller-visible unified catalog for this request.
	// Only entries with a nonempty Info.Tool become MCP tools. A nil or empty
	// result yields an empty tool catalog rather than a hidden failure.
	List(ctx context.Context) (map[string]chat.Info, error)
	// Run executes one canonical non-streaming request through the shared
	// lifecycle coordinator. Replays return without preparation or execution.
	// Accepted work that fails preparation is finalized exactly once.
	Run(ctx context.Context, idempotencyKey string, parsed *protocol.ParsedRequest) (chat.Response, error)
	// Limits returns the handler request and output limits.
	Limits() chat.Limits
	// LogError observes operational failures without exposing them.
	LogError(ctx context.Context, err error)
}

// Transport owns the stateless Streamable HTTP endpoint for one Handler.
type Transport struct {
	host    Host
	handler *sdkmcp.StreamableHTTPHandler
	cache   *sdkmcp.SchemaCache
}

// serverKey pins the per-request server to the request context.
type serverKey struct{}

// NewTransport builds the MCP endpoint transport.
func NewTransport(host Host) *Transport {
	t := &Transport{host: host, cache: sdkmcp.NewSchemaCache()}
	t.handler = sdkmcp.NewStreamableHTTPHandler(func(r *http.Request) *sdkmcp.Server {
		server, _ := r.Context().Value(serverKey{}).(*sdkmcp.Server)
		return server
	}, &sdkmcp.StreamableHTTPOptions{
		// Stateless Streamable HTTP: no session storage, no sticky sessions,
		// no Mcp-Session-Id handling. GET and DELETE return 405.
		Stateless: true,
		// Completed tool results as one application/json body.
		JSONResponse: true,
		// Tie tool handler contexts to the HTTP request so client
		// cancellations propagate into execution.
		PropagateRequestCancellation: true,
	})
	return t
}

// ServeHTTP serves one MCP request. Host.List runs exactly once with the
// authenticated request context; the projected tool set is pinned to this
// request and never shared across callers or identities.
func (t *Transport) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	catalog, err := t.host.List(r.Context())
	if err != nil {
		t.host.LogError(r.Context(), err)
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	server, err := t.buildServer(catalog)
	if err != nil {
		t.host.LogError(r.Context(), err)
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	ctx := context.WithValue(r.Context(), serverKey{}, server)
	t.handler.ServeHTTP(w, r.WithContext(ctx))
}

// entry is one projected tool: the catalog target bound to its public tool
// name and description source.
type entry struct {
	tool   string
	target string
	info   chat.Info
}

// projectTool filters the unified catalog down to MCP-exposed agents. The
// application's map and nested values are only read, never mutated.
func projectTool(target string, info chat.Info) (entry, bool) {
	if strings.TrimSpace(info.Tool) == "" {
		return entry{}, false
	}
	return entry{tool: info.Tool, target: target, info: info}, true
}

// buildServer validates the projected catalog and constructs one
// revision-pinned server exposing the entries as tools. Tools are added in
// sorted name order so listing is deterministic.
func (t *Transport) buildServer(catalog map[string]chat.Info) (*sdkmcp.Server, error) {
	entries := make([]entry, 0, len(catalog))
	for target, info := range catalog {
		if projected, ok := projectTool(target, info); ok {
			entries = append(entries, projected)
		}
	}
	slices.SortFunc(entries, func(a, b entry) int { return strings.Compare(a.tool, b.tool) })
	seen := make(map[string]struct{}, len(entries))
	for _, e := range entries {
		if err := validateToolName(e.tool); err != nil {
			return nil, err
		}
		_, exists := seen[e.tool]
		switch {
		case strings.TrimSpace(e.target) == "":
			return nil, fmt.Errorf("llmux: mcp tool %q has an empty target", e.tool)
		case exists:
			return nil, fmt.Errorf("llmux: duplicate mcp tool name %q", e.tool)
		}
		seen[e.tool] = struct{}{}
	}
	server := sdkmcp.NewServer(&sdkmcp.Implementation{Name: "llmux", Version: version()}, &sdkmcp.ServerOptions{
		// Serve exactly one revision. server/discover advertises this version
		// and requests identifying excluded revisions are rejected.
		SupportedProtocolVersions: []string{ProtocolVersion},
		// The catalog is fixed per request; never advertise listChanged.
		Capabilities: &sdkmcp.ServerCapabilities{Tools: &sdkmcp.ToolCapabilities{}},
		// Listings are caller-specific: mark every cacheable result
		// (server/discover, tools/list) private and immediately stale so
		// caller-specific listings cannot leak across identities.
		SetCacheable: func(_ context.Context, _ sdkmcp.Request, c *sdkmcp.Cacheable) {
			c.TTLMs = 0
			c.CacheScope = "private"
		},
		SchemaCache: t.cache,
	})
	for _, e := range entries {
		tool := &sdkmcp.Tool{
			Name:        e.tool,
			Description: e.info.Description,
			InputSchema: json.RawMessage(toolInputSchema),
		}
		server.AddTool(tool, t.callAgentTool(e))
	}
	return server, nil
}

// ValidateToolName mirrors the tool name rule of the selected SDK and the
// protocol schema: 1-128 characters from [A-Za-z0-9._-]. Names are never
// derived from targets through lossy sanitization; applications choose them.
func validateToolName(name string) error {
	if name == "" {
		return errors.New("llmux: mcp tool name cannot be empty")
	}
	if len(name) > 128 {
		return fmt.Errorf("llmux: mcp tool name %q exceeds the maximum length of 128 characters", name)
	}
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-', r == '.':
		default:
			return fmt.Errorf("llmux: mcp tool name %q contains invalid characters", name)
		}
	}
	return nil
}

// callAgentTool returns the SDK tool handler for one entry. Arguments are
// validated strictly against the fixed schema, then the entry target is
// resolved and executed through the shared Host machinery.
func (t *Transport) callAgentTool(e entry) sdkmcp.ToolHandler {
	return func(ctx context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
		if int64(len(req.Params.Arguments)) > t.host.Limits().MaxRequestBytes {
			// Malformed requests are protocol errors, not tool results.
			return nil, &jsonrpc.Error{
				Code:    jsonrpc.CodeInvalidParams,
				Message: "arguments exceed the configured request limit",
			}
		}
		message, err := decodeMessage(req.Params.Arguments)
		if err != nil {
			return nil, &jsonrpc.Error{Code: jsonrpc.CodeInvalidParams, Message: err.Error()}
		}
		result, runErr := t.runAgentTool(ctx, e, message)
		if runErr != nil {
			t.host.LogError(ctx, runErr)
			return &sdkmcp.CallToolResult{
				Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: publicToolMessage(runErr)}},
				IsError: true,
			}, nil
		}
		return result, nil
	}
}

// runAgentTool maps one validated tool invocation onto the shared lifecycle
// coordinator. The returned response is already finalized through Finish when
// acceptance reserved new work; replays skip preparation and execution.
func (t *Transport) runAgentTool(ctx context.Context, e entry, message string) (*sdkmcp.CallToolResult, error) {
	parsed := protocol.ParsedRequest{
		Request: chat.Request{
			Target: e.target,
			Input:  []chat.Item{chat.MessageItem(chat.RoleUser, chat.TextPart(message))},
			Output: chat.OutputSpec{Modalities: chat.ModalityText},
		},
	}
	parsed.Turn = parsed.Request.Input

	resp, err := t.host.Run(ctx, "", &parsed)
	if err != nil {
		return nil, err
	}
	return toolResultFromResponse(resp)
}

// toolResultFromResponse maps a finalized response to a completed MCP tool
// result. Text output maps in order; output without a supported MCP
// representation is rejected explicitly rather than discarded.
func toolResultFromResponse(resp chat.Response) (*sdkmcp.CallToolResult, error) {
	content := make([]sdkmcp.Content, 0, len(resp.Output))
	mapPart := func(part chat.Part) error {
		switch part.Type {
		case chat.PartText:
			if part.Text != "" {
				content = append(content, &sdkmcp.TextContent{Text: part.Text})
			}
			return nil
		case chat.PartImage, chat.PartAudio, chat.PartFile:
			// Unreachable for well-formed runs: MCP requests declare
			// text-only output and execution rejects other modalities
			// during the run. Kept explicit so no output is ever silently
			// discarded.
			return fmt.Errorf("agent produced %s output, which has no MCP tool representation", part.Type)
		default:
			return fmt.Errorf("agent produced %s output, which has no MCP tool representation", part.Type)
		}
	}
	for _, item := range resp.Output {
		switch item.Type {
		case chat.ItemMessage, chat.ItemMedia:
			for _, part := range item.Content {
				if err := mapPart(part); err != nil {
					return nil, err
				}
			}
		case chat.ItemFunctionCall:
			// Internal agent tool calls are never exposed as client tool
			// requests.
			return nil, errors.New("agent produced an internal tool call, which has no MCP tool representation")
		default:
			return nil, fmt.Errorf("agent produced %s output, which has no MCP tool representation", item.Type)
		}
	}
	result := &sdkmcp.CallToolResult{Content: content}
	switch resp.Status {
	case "", chat.StatusCompleted:
		return result, nil
	case chat.StatusIncomplete:
		// Truncated output is returned, but flagged as an error so partial
		// content is never mistaken for a complete answer.
		result.Content = append(result.Content, &sdkmcp.TextContent{Text: "output incomplete: " + cmp.Or(resp.Incomplete, "unknown")})
		result.IsError = true
		return result, nil
	case chat.StatusCancelled:
		return nil, errors.New("execution was cancelled")
	case chat.StatusFailed:
		msg := "the agent could not complete this request"
		if resp.Error != nil {
			msg = cmp.Or(resp.Error.Message, msg)
		}
		return nil, errors.New(msg)
	default:
		return nil, fmt.Errorf("agent returned unsupported status %q", resp.Status)
	}
}

// PublicToolMessage sanitizes an operational error for a tool result.
// Application-declared chat.Error messages pass through; raw operational
// errors never do.
func publicToolMessage(err error) string {
	var apiErr *chat.Error
	if errors.As(err, &apiErr) && apiErr.Message != "" {
		return apiErr.Message
	}
	return "the agent could not complete this request"
}

// decodeMessage strictly validates tool arguments against the fixed internal
// schema: one object with exactly one non-empty string "message".
func decodeMessage(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", errors.New(`arguments must be an object with a "message" string property`)
	}
	var arguments map[string]json.RawMessage
	if err := json.Unmarshal(raw, &arguments); err != nil {
		return "", errors.New(`arguments must be an object with a "message" string property`)
	}
	names := make([]string, 0, len(arguments))
	for name := range arguments {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		if name != "message" {
			return "", fmt.Errorf("unknown argument %q; only %q is supported", name, "message")
		}
	}
	rawMessage, ok := arguments["message"]
	if !ok {
		return "", errors.New(`"message" is required`)
	}
	var message string
	if err := json.Unmarshal(rawMessage, &message); err != nil {
		return "", errors.New(`"message" must be a string`)
	}
	if strings.TrimSpace(message) == "" {
		return "", errors.New(`"message" must not be empty`)
	}
	return message, nil
}

var (
	versionOnce sync.Once
	versionStr  string
)

func version() string {
	versionOnce.Do(func() {
		if info, ok := debug.ReadBuildInfo(); ok && info.Main.Version != "" && info.Main.Version != "(devel)" {
			versionStr = info.Main.Version
			return
		}
		versionStr = "dev"
	})
	return versionStr
}
