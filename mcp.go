package llmux

import (
	"context"
	"net/http"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/mcp"
)

// mcpProtocolVersion is the MCP protocol revision served at /mcp. Requests
// at older revisions are rejected; llmux does not maintain legacy handshakes
// on this endpoint. See the README for the supported-revision policy.
const mcpProtocolVersion = mcp.ProtocolVersion

// MCPEntry is one agent exposed as a tool through /mcp.
type MCPEntry struct {
	// Tool is the stable MCP tool name. It must be 1-128 characters from
	// [A-Za-z0-9._-] and unique across the catalog. Names are never derived
	// from Target; applications choose and own them.
	Tool string
	// Target is the agent target passed to the Resolver when the tool is
	// invoked. Resolution at invocation time is independent of listing.
	Target string
	// Info is the agent information returned by the Resolver. Info.Description
	// becomes the MCP tool description; no other description source exists.
	Info chat.Info
}

// MCPConfig configures the optional /mcp endpoint (exact path /mcp, served
// only when WithMCP is applied; applications own any prefix mounting). The
// zero config never routes /mcp, so existing chat-only applications are
// unaffected.
type MCPConfig struct {
	// List returns the agents exposed to the authenticated caller. It runs
	// once per /mcp request with the request context, so authorization
	// belongs inside the callback. Returned entries are copied; mutating
	// them afterward has no effect.
	List func(context.Context) ([]MCPEntry, error)
}

// WithMCP enables the /mcp endpoint. See MCPConfig for details.
func WithMCP(config MCPConfig) Option {
	return func(h *Handler) {
		if config.List == nil {
			panic("llmux: WithMCP requires MCPConfig.List")
		}
		h.mcp = mcp.NewTransport(hostAdapter{h: h}, mcp.Config{
			List: func(ctx context.Context) ([]mcp.Entry, error) {
				entries, err := config.List(ctx)
				if err != nil {
					return nil, err
				}
				out := make([]mcp.Entry, len(entries))
				for i, entry := range entries {
					out[i] = mcp.Entry{Tool: entry.Tool, Target: entry.Target, Info: entry.Info}
				}
				return out, nil
			},
		})
	}
}

// serveMCP delegates to the internal MCP transport.
func (h *Handler) serveMCP(w http.ResponseWriter, r *http.Request) {
	h.mcp.ServeHTTP(w, r)
}

// hostAdapter implements the internal/mcp execution seam on Handler. It
// reuses the exact machinery behind the chat endpoints: resolver-based
// authorization, capability validation, bounded execution, and lifecycle
// Finish.
type hostAdapter struct{ h *Handler }

func (a hostAdapter) Resolve(ctx context.Context, target string) (chat.Agent, chat.Info, error) {
	return a.h.resolve(ctx, target)
}

func (a hostAdapter) Validate(parsed *parsedRequest, info chat.Info) error {
	return a.h.validateParsed(parsed, info)
}

func (a hostAdapter) Prepare(ctx context.Context, parsed *parsedRequest) error {
	return a.h.prepareParsed(ctx, parsed)
}

func (a hostAdapter) Accept(ctx context.Context, idempotencyKey string, parsed *parsedRequest) (chat.Acceptance, bool, error) {
	return a.h.acceptContext(ctx, idempotencyKey, parsed)
}

func (a hostAdapter) Execute(
	ctx context.Context,
	parsed parsedRequest,
	agent chat.Agent,
	meta *responseMeta,
	acceptance chat.Acceptance,
	validateEvent func(chat.Event) error,
) (chat.Response, error) {
	return a.h.executeOrdinary(ctx, parsed, agent, meta, acceptance, validateEvent)
}

func (a hostAdapter) Limits() chat.Limits { return a.h.limits }

func (a hostAdapter) LogError(ctx context.Context, err error) { a.h.logError(ctx, err) }
