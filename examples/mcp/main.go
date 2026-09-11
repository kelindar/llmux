// Command mcp demonstrates llmux's optional MCP endpoint: one authenticated
// catalog, projected into GET /models and (when enabled) the MCP tools at
// /mcp.
//
// The authentication wrapper below is ILLUSTRATIVE ONLY. In production the
// token check must be real OAuth validation (or equivalent), and the MCP
// spec's OAuth responsibilities — token audience/resource validation, scopes,
// authorization-server discovery, protected-resource metadata — stay with
// the host application. llmux never sees or validates tokens.
package main

import (
	"context"
	"log"
	"net/http"
	"strings"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/chat"
)

type callerKey struct{}

func caller(ctx context.Context) string {
	identity, _ := ctx.Value(callerKey{}).(string)
	return identity
}

// echo is a minimal application-owned agent. It runs through the same
// execution path for MCP tool calls and for the chat endpoints.
var echo = chat.AgentFunc(func(_ context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
	message := req.Input[len(req.Input)-1].Content[0].Text
	return chat.Outcome{}, emit(chat.Text("echo: " + message))
})

// resolve selects agents for invocation. It runs independently of the
// catalog, so listing visibility never replaces authorization here.
func resolve(_ context.Context, target string) (chat.Agent, chat.Info, error) {
	switch target {
	case "agent/echo":
		return echo, chat.Info{Description: "Echoes the caller's message back as assistant text."}, nil
	default:
		return nil, chat.Info{}, &chat.Error{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "not_found", Message: "unknown agent"}
	}
}

// catalog is the single authenticated listing callback. It receives the
// request context produced by the authentication wrapper, so visibility is
// an application decision per caller. Keys are resolver targets; Info.Tool
// (nonempty) is the public MCP tool name, Info.Description its description,
// and the remaining fields flow into GET /models.
func catalog(ctx context.Context) (map[string]chat.Info, error) {
	info := chat.Info{
		Description: "Echoes a message back as assistant text.",
		Tool:        "echo",
		Created:     1735689600,
		OwnedBy:     "acme-agents",
	}
	switch caller(ctx) {
	case "alice":
		return map[string]chat.Info{"agent/echo": info}, nil
	default:
		// Visible in /models, but not exposed through MCP (no Tool).
		info.Tool = ""
		return map[string]chat.Info{"agent/echo": info}, nil
	}
}

// newHandler builds the llmux handler with the unified catalog and MCP
// enabled. Both are opt-in: removing WithCatalog yields empty listings, and
// removing WithMCP removes the /mcp route while chat endpoints are
// unaffected. Option ordering does not matter.
func newHandler() *llmux.Handler {
	return llmux.New(resolve,
		llmux.WithCatalog(catalog),
		llmux.WithMCP(),
	)
}

// withAuth is an ILLUSTRATIVE authentication wrapper: a fixed token map
// standing in for real OAuth bearer-token validation. Replace it with your
// middleware; the shape is what matters — authenticate outside llmux, reject
// with 401 and a WWW-Authenticate challenge, then forward the authenticated
// context so listing, resolution, acceptance, execution, and finish all see
// the caller identity.
func withAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		var identity string
		switch token {
		case "alice-token":
			identity = "alice"
		default:
			w.Header().Set("WWW-Authenticate", `Bearer realm="llmux-mcp"`)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), callerKey{}, identity)))
	})
}

// newServer mounts the handler under an application-owned prefix. llmux
// serves exact paths beneath its root, so /v1/mcp is the MCP endpoint and
// /v1/models the model catalog.
func newServer() http.Handler {
	return http.StripPrefix("/v1", withAuth(newHandler()))
}

func main() {
	log.Println("listening on http://127.0.0.1:8080")
	log.Println("  MCP:      POST /v1/mcp        (stateless Streamable HTTP)")
	log.Println("  models:   GET  /v1/models")
	log.Println("  chat:     POST /v1/chat/completions")
	if err := http.ListenAndServe("127.0.0.1:8080", newServer()); err != nil {
		log.Fatal(err)
	}
}
