// Package mcp implements the optional MCP endpoint for llmux. It exposes
// application-owned agents as MCP tools over stateless Streamable HTTP,
// reusing the shared agent execution and lifecycle machinery through the
// Host seam. MCP SDK types never cross back into the public agent contract.
package mcp

import (
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

// Entry is one agent exposed as an MCP tool.
type Entry struct {
	// Tool is the stable MCP tool name, validated at catalog build time.
	Tool string
	// Target is the agent target resolved independently at invocation.
	Target string
	// Info is reused as the tool description source; no other representation
	// exists. Info.Description becomes the MCP tool description.
	Info chat.Info
}

// Lister returns the entries exposed to the authenticated caller. It runs
// once per MCP request with the request context; authorization belongs
// inside the callback.
type Lister func(context.Context) ([]Entry, error)

// Config configures a Transport.
type Config struct {
	// List returns the caller-specific agent catalog. Required.
	List Lister
}

// Host is the execution seam provided by the llmux Handler. It keeps the MCP
// adapter from duplicating the execution engine, the capability validation,
// or the lifecycle machinery.
type Host interface {
	// Resolve selects an agent and its Info for a target name.
	Resolve(ctx context.Context, target string) (chat.Agent, chat.Info, error)
	// Validate checks a parsed request against agent Info.
	Validate(parsed *protocol.ParsedRequest, info chat.Info) error
	// Prepare completes a parsed request (store policy, media resolution).
	Prepare(ctx context.Context, parsed *protocol.ParsedRequest) error
	// Accept applies lifecycle acceptance. Idempotency keys come from the
	// calling protocol; MCP passes none, so JSON-RPC request IDs are never
	// idempotency keys.
	Accept(ctx context.Context, idempotencyKey string, parsed *protocol.ParsedRequest) (chat.Acceptance, bool, error)
	// Execute runs a fully validated request to completion without streaming
	// delivery, applying bounded execution and lifecycle Finish exactly once.
	// The response is persisted before a successful result is returned.
	Execute(ctx context.Context, parsed protocol.ParsedRequest, agent chat.Agent, meta *protocol.Meta, acceptance chat.Acceptance, validateEvent func(chat.Event) error) (chat.Response, error)
	// Limits returns the handler request and output limits.
	Limits() chat.Limits
	// LogError observes operational failures without exposing them.
	LogError(ctx context.Context, err error)
}

// Transport owns the stateless Streamable HTTP endpoint for one Handler.
type Transport struct {
	host    Host
	list    Lister
	handler *sdkmcp.StreamableHTTPHandler
	cache   *sdkmcp.SchemaCache
}

// serverKey pins the per-request server to the request context.
type serverKey struct{}

// NewTransport builds the MCP endpoint transport.
func NewTransport(host Host, config Config) *Transport {
	t := &Transport{host: host, list: config.List, cache: sdkmcp.NewSchemaCache()}
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

// ServeHTTP serves one MCP request. The listing callback runs exactly once
// with the authenticated request context; the resolved catalog is pinned to
// this request and never shared across callers or identities.
func (t *Transport) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	entries, err := t.list(r.Context())
	if err != nil {
		t.host.LogError(r.Context(), err)
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	server, err := t.buildServer(entries)
	if err != nil {
		t.host.LogError(r.Context(), err)
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	ctx := context.WithValue(r.Context(), serverKey{}, server)
	t.handler.ServeHTTP(w, r.WithContext(ctx))
}

// buildServer validates the catalog and constructs one revision-pinned
// server exposing the entries as tools. Tools are added in sorted name order
// so listing is deterministic.
func (t *Transport) buildServer(entries []Entry) (*sdkmcp.Server, error) {
	sorted := slices.Clone(entries)
	slices.SortFunc(sorted, func(a, b Entry) int { return strings.Compare(a.Tool, b.Tool) })
	seen := make(map[string]struct{}, len(sorted))
	for _, entry := range sorted {
		if err := validateToolName(entry.Tool); err != nil {
			return nil, err
		}
		if strings.TrimSpace(entry.Target) == "" {
			return nil, fmt.Errorf("llmux: mcp tool %q has an empty target", entry.Tool)
		}
		if _, exists := seen[entry.Tool]; exists {
			return nil, fmt.Errorf("llmux: duplicate mcp tool name %q", entry.Tool)
		}
		seen[entry.Tool] = struct{}{}
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
	for _, entry := range sorted {
		tool := &sdkmcp.Tool{
			Name:        entry.Tool,
			Description: entry.Info.Description,
			InputSchema: json.RawMessage(toolInputSchema),
		}
		server.AddTool(tool, t.callAgentTool(entry))
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
func (t *Transport) callAgentTool(entry Entry) sdkmcp.ToolHandler {
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
		result, runErr := t.runAgentTool(ctx, entry, message)
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

// runAgentTool maps one validated tool invocation onto the canonical request
// path: resolve, validate, prepare, accept, execute. The response returned by
// Execute has already been finalized through lifecycle Finish.
func (t *Transport) runAgentTool(ctx context.Context, entry Entry, message string) (*sdkmcp.CallToolResult, error) {
	parsed := protocol.ParsedRequest{
		Request: chat.Request{
			Target: entry.Target,
			Input:  []chat.Item{chat.MessageItem(chat.RoleUser, chat.TextPart(message))},
			Output: chat.OutputSpec{Modalities: chat.ModalityText},
		},
	}
	parsed.Turn = parsed.Request.Input

	agent, info, err := t.host.Resolve(ctx, entry.Target)
	if err != nil {
		return nil, err
	}
	if err := t.host.Validate(&parsed, info); err != nil {
		return nil, err
	}
	if err := t.host.Prepare(ctx, &parsed); err != nil {
		return nil, err
	}
	if err := t.host.Validate(&parsed, info); err != nil {
		return nil, err
	}
	acceptance, _, err := t.host.Accept(ctx, "", &parsed)
	if err != nil {
		return nil, err
	}

	meta := protocol.Meta{
		Response: protocol.InitialResponse(acceptance.Response, &parsed),
		Activity: acceptance.Activity,
	}
	acceptance.Response = meta.Response

	// Replays skip execution and map the stored response like any other.
	if acceptance.Replay != nil {
		return toolResultFromResponse(acceptance.Replay.Clone())
	}

	validateEvent := func(event chat.Event) error {
		switch {
		case protocol.RequiresReasoningSummary(event) && (parsed.Request.Controls.Reasoning == nil || !parsed.Request.Controls.Reasoning.Summary):
			return chat.Unsupported("reasoning.summary", "reasoning summary output was not requested")
		case event.Type == chat.EventActivity && !acceptance.Activity:
			return chat.Unsupported("output", "activity events were not enabled for this request")
		}
		if event.Type == chat.EventActivity {
			return nil
		}
		if err := protocol.ValidateOutputEvent(event, info); err != nil {
			return err
		}
		return protocol.ValidateRequestedOutput(event, parsed.Request.Output.Modalities)
	}

	resp, err := t.host.Execute(ctx, parsed, agent, &meta, acceptance, validateEvent)
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
		reason := resp.Incomplete
		if reason == "" {
			reason = "unknown"
		}
		result.Content = append(result.Content, &sdkmcp.TextContent{Text: "output incomplete: " + reason})
		result.IsError = true
		return result, nil
	case chat.StatusCancelled:
		return nil, errors.New("execution was cancelled")
	case chat.StatusFailed:
		if resp.Error != nil && resp.Error.Message != "" {
			return nil, errors.New(resp.Error.Message)
		}
		return nil, errors.New("the agent could not complete this request")
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
