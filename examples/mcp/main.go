// Command mcp demonstrates llmux's optional MCP endpoint: an agent exposed
// as MCP tools over stateless Streamable HTTP at /mcp.
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

// resolve selects agents and declares their Info. Info.Description becomes
// the MCP tool description when the agent is listed.
func resolve(_ context.Context, target string) (chat.Agent, chat.Info, error) {
	switch target {
	case "agent/echo":
		return echo, chat.Info{Description: "Echoes the caller's message back as assistant text."}, nil
	default:
		return nil, chat.Info{}, &chat.Error{Status: http.StatusNotFound, Type: "invalid_request_error", Code: "not_found", Message: "unknown agent"}
	}
}

// list is the authenticated listing callback. It receives the request
// context produced by the authentication wrapper, so visibility is an
// application decision per caller. Tool names are stable and chosen here —
// never derived from targets.
func list(ctx context.Context) ([]llmux.MCPEntry, error) {
	switch caller(ctx) {
	case "alice":
		return []llmux.MCPEntry{
			{Tool: "echo_alice", Target: "agent/echo", Info: chat.Info{Description: "Echoes a message (for Alice)."}},
		}, nil
	default:
		return nil, nil
	}
}

// newHandler builds the llmux handler with MCP enabled. MCP is opt-in:
// removing the WithMCP option removes the /mcp route and leaves the chat
// endpoints untouched.
func newHandler() *llmux.Handler {
	return llmux.New(resolve,
		llmux.WithMCP(llmux.MCPConfig{List: list}),
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
// serves exact paths beneath its root, so /v1/mcp is the MCP endpoint.
func newServer() http.Handler {
	return http.StripPrefix("/v1", withAuth(newHandler()))
}

func main() {
	log.Println("listening on http://127.0.0.1:8080")
	log.Println("  MCP:      POST /v1/mcp        (stateless Streamable HTTP)")
	log.Println("  chat:     POST /v1/chat/completions")
	if err := http.ListenAndServe("127.0.0.1:8080", newServer()); err != nil {
		log.Fatal(err)
	}
}
