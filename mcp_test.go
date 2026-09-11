package llmux

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/kelindar/llmux/chat"
	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mcpTestAgent returns an agent that echoes the single user message and
// records each run.
type mcpTestAgent struct {
	runs  atomic.Int32
	seen  chan *chat.Request
	block chan chan struct{}
	logs  []string
}

func newMCPTestAgent() *mcpTestAgent {
	return &mcpTestAgent{seen: make(chan *chat.Request, 16)}
}

func (a *mcpTestAgent) Run(ctx context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
	a.runs.Add(1)
	select {
	case a.seen <- req:
	default:
	}
	if a.block != nil {
		release := make(chan struct{})
		select {
		case a.block <- release:
		case <-ctx.Done():
			return chat.Outcome{Status: chat.StatusCancelled}, ctx.Err()
		}
		select {
		case <-ctx.Done():
			return chat.Outcome{Status: chat.StatusCancelled}, ctx.Err()
		case <-release:
		}
	}
	text := req.Input[len(req.Input)-1].Content[0].Text
	return chat.Outcome{}, emit(chat.Text("echo: " + text))
}

// mcpAuth is an illustrative authentication wrapper. Production applications
// replace it with real OAuth token validation; llmux never sees the token.
type mcpAuth struct {
	next     http.Handler
	identity map[string]string // bearer token → caller identity
}

func (a *mcpAuth) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	identity, ok := a.identity[token]
	if !ok {
		w.Header().Set("WWW-Authenticate", `Bearer realm="llmux-mcp", resource_metadata="/.well-known/oauth-protected-resource"`)
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	a.next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), mcpCallerKey{}, identity)))
}

type mcpCallerKey struct{}

func mcpCaller(ctx context.Context) string {
	identity, _ := ctx.Value(mcpCallerKey{}).(string)
	return identity
}

type mcpFixture struct {
	handler *Handler
	http    *httptest.Server
	auth    *mcpAuth
}

// newMCPFixture wires a handler behind the illustrative auth wrapper with
// caller-specific listings and mounts it under an application prefix.
func newMCPFixture(t *testing.T, agent chat.Agent, options ...Option) *mcpFixture {
	t.Helper()
	handler := New(chat.Resolver(func(ctx context.Context, target string) (chat.Agent, chat.Info, error) {
		if mcpCaller(ctx) == "" {
			return nil, chat.Info{}, &chat.Error{Status: http.StatusUnauthorized, Type: "authentication_error", Code: "missing_identity", Message: "request identity is required"}
		}
		return agent, chat.Info{Description: "test agent for " + target}, nil
	}), options...)
	auth := &mcpAuth{next: handler, identity: map[string]string{"token-a": "user-a", "token-b": "user-b"}}
	fixture := &mcpFixture{handler: handler, auth: auth}
	fixture.http = httptest.NewServer(http.StripPrefix("/v1", auth))
	t.Cleanup(fixture.http.Close)
	return fixture
}

func (f *mcpFixture) endpoint() string { return f.http.URL + "/v1/mcp" }

// connect opens an official-SDK client session against the fixture as the
// given caller.
func (f *mcpFixture) connect(t *testing.T, token string) *sdkmcp.ClientSession {
	t.Helper()
	client := sdkmcp.NewClient(&sdkmcp.Implementation{Name: "mcp-test", Version: "v0.0.1"}, nil)
	transport := &sdkmcp.StreamableClientTransport{
		Endpoint:   f.endpoint(),
		HTTPClient: &http.Client{Transport: bearerTransport{token: token, base: f.http.Client().Transport}},
	}
	session, err := client.Connect(context.Background(), transport, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = session.Close() })
	return session
}

type bearerTransport struct {
	token string
	base  http.RoundTripper
}

func (t bearerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.Header.Set("Authorization", "Bearer "+t.token)
	return t.base.RoundTrip(req)
}

func TestMCPDiscoveryListAndCall(t *testing.T) {
	agent := newMCPTestAgent()
	listing := func(ctx context.Context) ([]MCPEntry, error) {
		require.Equal(t, "user-a", mcpCaller(ctx), "listing must receive the authenticated context")
		return []MCPEntry{
			{Tool: "zeta", Target: "agent/zeta", Info: chat.Info{Description: "last"}},
			{Tool: "alpha", Target: "agent/alpha", Info: chat.Info{Description: "first"}},
		}, nil
	}
	fixture := newMCPFixture(t, agent, WithMCP(MCPConfig{List: listing}))
	session := fixture.connect(t, "token-a")

	// server/discover runs inside Connect; a stateless session is live
	// immediately afterward without initialize.
	result, err := session.ListTools(context.Background(), nil)
	require.NoError(t, err)
	require.Len(t, result.Tools, 2)
	// Deterministic order: sorted by tool name, not listing order.
	assert.Equal(t, "alpha", result.Tools[0].Name)
	assert.Equal(t, "zeta", result.Tools[1].Name)
	assert.Equal(t, "first", result.Tools[0].Description)
	// Caller-specific listings must be marked private and immediately stale.
	assert.Equal(t, "private", result.GetCacheScope())
	assert.Equal(t, 0, result.GetTTLMs())

	call, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{
		Name:      "alpha",
		Arguments: map[string]any{"message": "hello"},
	})
	require.NoError(t, err)
	require.False(t, call.IsError)
	require.Len(t, call.Content, 1)
	text, ok := call.Content[0].(*sdkmcp.TextContent)
	require.True(t, ok, "expected text content, got %T", call.Content[0])
	assert.Equal(t, "echo: hello", text.Text)
	assert.Equal(t, int32(1), agent.runs.Load())

	// The invocation resolved the listed target through the Resolver.
	select {
	case req := <-agent.seen:
		require.Len(t, req.Input, 1)
		item := req.Input[0]
		assert.Equal(t, chat.ItemMessage, item.Type)
		assert.Equal(t, chat.RoleUser, item.Role)
		require.Len(t, item.Content, 1)
		assert.Equal(t, "hello", item.Content[0].Text)
		assert.Equal(t, "agent/alpha", req.Target)
	default:
		t.Fatal("agent was not invoked")
	}
}

func TestMCPCallerSpecificListingAndCacheIsolation(t *testing.T) {
	agent := newMCPTestAgent()
	listing := func(ctx context.Context) ([]MCPEntry, error) {
		switch mcpCaller(ctx) {
		case "user-a":
			return []MCPEntry{{Tool: "tool-a", Target: "agent/a", Info: chat.Info{Description: "only for a"}}}, nil
		case "user-b":
			return []MCPEntry{{Tool: "tool-b", Target: "agent/b", Info: chat.Info{Description: "only for b"}}}, nil
		default:
			return nil, nil
		}
	}
	fixture := newMCPFixture(t, agent, WithMCP(MCPConfig{List: listing}))

	for _, test := range []struct{ token, tool string }{
		{"token-a", "tool-a"},
		{"token-b", "tool-b"},
	} {
		session := fixture.connect(t, test.token)
		result, err := session.ListTools(context.Background(), nil)
		require.NoError(t, err)
		require.Len(t, result.Tools, 1)
		assert.Equal(t, test.tool, result.Tools[0].Name)

		// A caller cannot invoke a tool they are not allowed to list.
		other := "tool-b"
		if test.tool == "tool-b" {
			other = "tool-a"
		}
		_, err = session.CallTool(context.Background(), &sdkmcp.CallToolParams{
			Name:      other,
			Arguments: map[string]any{"message": "hi"},
		})
		require.Error(t, err, "unlisted tool must not be invocable")
		var wireErr *jsonrpc.Error
		require.ErrorAs(t, err, &wireErr)
		assert.EqualValues(t, jsonrpc.CodeInvalidParams, wireErr.Code)
	}

	// cacheScope "private" is on the wire so shared caches cannot reuse one
	// caller's listing for another.
	for _, test := range []struct{ token, tool string }{{"token-a", "tool-a"}, {"token-b", "tool-b"}} {
		result := fixture.rawToolsList(t, test.token)
		assert.Equal(t, "private", result["cacheScope"], "listing for %s must be private", test.token)
	}
}

// rawMCPMeta is the per-request _meta triple required by the 2026-07-28
// revision on every call.
const rawMCPMeta = `"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}`

// rawMCPRequest issues one compliant stateless 2026-07-28 JSON-RPC request
// and returns the HTTP response plus the decoded result (if successful).
func (f *mcpFixture) rawMCPRequest(t *testing.T, id int, method, params string) (*http.Response, map[string]any) {
	t.Helper()
	body := fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"method":%q,"params":{"_meta":{%s}}%s}`, id, method, rawMCPMeta, params)
	req, err := http.NewRequest(http.MethodPost, f.endpoint(), strings.NewReader(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("Mcp-Protocol-Version", mcpProtocolVersion)
	req.Header.Set("Mcp-Method", method)
	resp, err := f.http.Client().Do(req)
	require.NoError(t, err)
	t.Cleanup(func() { resp.Body.Close() })
	require.Equal(t, http.StatusOK, resp.StatusCode, resp.Status)
	var envelope struct {
		Result map[string]any `json:"result"`
		Error  map[string]any `json:"error"`
	}
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&envelope))
	require.Empty(t, envelope.Error, "raw request failed: %v", envelope.Error)
	return resp, envelope.Result
}

// rawToolsList issues a raw stateless tools/list and decodes the JSON result.
func (f *mcpFixture) rawToolsList(t *testing.T, token string) map[string]any {
	t.Helper()
	auth := bearerTransport{token: token, base: f.http.Client().Transport}
	client := &http.Client{Transport: auth}
	body := fmt.Sprintf(`{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"_meta":{%s}}}`, rawMCPMeta)
	req, err := http.NewRequest(http.MethodPost, f.endpoint(), strings.NewReader(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("Mcp-Protocol-Version", mcpProtocolVersion)
	req.Header.Set("Mcp-Method", "tools/list")
	resp, err := client.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	require.Equal(t, http.StatusOK, resp.StatusCode, resp.Status)
	var envelope struct {
		Result map[string]any `json:"result"`
	}
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&envelope))
	return envelope.Result
}

func TestMCPFixedInputSchema(t *testing.T) {
	agent := newMCPTestAgent()
	fixture := newMCPFixture(t, agent, WithMCP(MCPConfig{List: func(context.Context) ([]MCPEntry, error) {
		return []MCPEntry{{Tool: "echo", Target: "agent/echo", Info: chat.Info{Description: "echo"}}}, nil
	}}))
	session := fixture.connect(t, "token-a")

	result, err := session.ListTools(context.Background(), nil)
	require.NoError(t, err)
	require.Len(t, result.Tools, 1)
	schema, err := json.Marshal(result.Tools[0].InputSchema)
	require.NoError(t, err)
	var decoded map[string]any
	require.NoError(t, json.Unmarshal(schema, &decoded))
	assert.Equal(t, "object", decoded["type"])
	assert.Equal(t, false, decoded["additionalProperties"])
	properties, ok := decoded["properties"].(map[string]any)
	require.True(t, ok)
	message, ok := properties["message"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "string", message["type"])
	assert.ElementsMatch(t, []any{"message"}, decoded["required"])

	for _, test := range []struct {
		name      string
		arguments map[string]any
		fragment  string
	}{
		{name: "missing message", arguments: map[string]any{}, fragment: `"message" is required`},
		{name: "unknown argument", arguments: map[string]any{"message": "hi", "session": "x"}, fragment: `unknown argument "session"`},
		{name: "non-string message", arguments: map[string]any{"message": 42}, fragment: `"message" must be a string`},
		{name: "empty message", arguments: map[string]any{"message": "  "}, fragment: `"message" must not be empty`},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: test.arguments})
			require.Error(t, err)
			var wireErr *jsonrpc.Error
			require.ErrorAs(t, err, &wireErr, "malformed arguments must be protocol errors")
			assert.EqualValues(t, jsonrpc.CodeInvalidParams, wireErr.Code)
			assert.Contains(t, wireErr.Message, test.fragment)
		})
	}
	assert.Equal(t, int32(0), agent.runs.Load(), "no malformed request may reach the agent")
}

func TestMCPUnknownAndUnauthorizedTools(t *testing.T) {
	agent := newMCPTestAgent()
	fixture := newMCPFixture(t, agent, WithMCP(MCPConfig{List: func(ctx context.Context) ([]MCPEntry, error) {
		if mcpCaller(ctx) != "user-a" {
			return nil, nil
		}
		return []MCPEntry{{Tool: "listed", Target: "agent/ok", Info: chat.Info{Description: "ok"}}}, nil
	}}))
	session := fixture.connect(t, "token-a")

	t.Run("unknown tool", func(t *testing.T) {
		_, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "nope", Arguments: map[string]any{"message": "hi"}})
		require.Error(t, err)
		var wireErr *jsonrpc.Error
		require.ErrorAs(t, err, &wireErr)
		assert.EqualValues(t, jsonrpc.CodeInvalidParams, wireErr.Code)
	})
	t.Run("listed but resolver denies target", func(t *testing.T) {
		denying := New(chat.Resolver(func(_ context.Context, target string) (chat.Agent, chat.Info, error) {
			return nil, chat.Info{}, &chat.Error{Status: http.StatusForbidden, Type: "permission_error", Code: "target_forbidden", Message: "you may not call this agent"}
		}), WithMCP(MCPConfig{List: func(context.Context) ([]MCPEntry, error) {
			return []MCPEntry{{Tool: "listed", Target: "agent/denied", Info: chat.Info{Description: "ok"}}}, nil
		}}))
		server := httptest.NewServer(denying)
		t.Cleanup(server.Close)
		client := sdkmcp.NewClient(&sdkmcp.Implementation{Name: "mcp-test", Version: "v0.0.1"}, nil)
		session, err := client.Connect(context.Background(), &sdkmcp.StreamableClientTransport{Endpoint: server.URL + "/mcp"}, nil)
		require.NoError(t, err)
		t.Cleanup(func() { _ = session.Close() })

		result, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "listed", Arguments: map[string]any{"message": "hi"}})
		require.NoError(t, err)
		require.True(t, result.IsError)
		text := result.Content[0].(*sdkmcp.TextContent)
		assert.Equal(t, "you may not call this agent", text.Text)
		assert.Equal(t, int32(0), agent.runs.Load())
	})
	t.Run("resolver failure without public error is sanitized", func(t *testing.T) {
		broken := New(chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
			return nil, chat.Info{}, errors.New("database connection string leaked")
		}), WithMCP(MCPConfig{List: func(context.Context) ([]MCPEntry, error) {
			return []MCPEntry{{Tool: "listed", Target: "agent/x", Info: chat.Info{Description: "ok"}}}, nil
		}}), WithErrorLog(func(context.Context, error) {}))
		server := httptest.NewServer(broken)
		t.Cleanup(server.Close)
		client := sdkmcp.NewClient(&sdkmcp.Implementation{Name: "mcp-test", Version: "v0.0.1"}, nil)
		session, err := client.Connect(context.Background(), &sdkmcp.StreamableClientTransport{Endpoint: server.URL + "/mcp"}, nil)
		require.NoError(t, err)
		t.Cleanup(func() { _ = session.Close() })

		result, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "listed", Arguments: map[string]any{"message": "hi"}})
		require.NoError(t, err)
		require.True(t, result.IsError)
		text := result.Content[0].(*sdkmcp.TextContent)
		assert.NotContains(t, text.Text, "database", "raw operational errors must never surface")
		assert.Contains(t, text.Text, "could not complete")
	})
}

func TestMCPUnauthenticatedChallenge(t *testing.T) {
	agent := newMCPTestAgent()
	fixture := newMCPFixture(t, agent, WithMCP(MCPConfig{List: func(context.Context) ([]MCPEntry, error) {
		return []MCPEntry{{Tool: "echo", Target: "agent/echo"}}, nil
	}}))
	resp, err := http.Post(fixture.endpoint(), "application/json", strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}`))
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusUnauthorized, resp.StatusCode)
	assert.Equal(t, `Bearer realm="llmux-mcp", resource_metadata="/.well-known/oauth-protected-resource"`, resp.Header.Get("WWW-Authenticate"))
	assert.Equal(t, int32(0), agent.runs.Load())
}

func TestMCPAuthenticatedContextPropagation(t *testing.T) {
	agent := newMCPTestAgent()
	var mu sync.Mutex
	var identities []string
	fixture := newMCPFixture(t, agent, WithMCP(MCPConfig{List: func(ctx context.Context) ([]MCPEntry, error) {
		mu.Lock()
		identities = append(identities, "list:"+mcpCaller(ctx))
		mu.Unlock()
		return []MCPEntry{{Tool: "echo", Target: "agent/echo"}}, nil
	}}))
	// The resolver records the identity observed at invocation time.
	observing := New(chat.Resolver(func(ctx context.Context, target string) (chat.Agent, chat.Info, error) {
		mu.Lock()
		identities = append(identities, "resolve:"+mcpCaller(ctx))
		mu.Unlock()
		return agent, chat.Info{}, nil
	}), WithMCP(MCPConfig{List: func(ctx context.Context) ([]MCPEntry, error) {
		mu.Lock()
		identities = append(identities, "list:"+mcpCaller(ctx))
		mu.Unlock()
		return []MCPEntry{{Tool: "echo", Target: "agent/echo"}}, nil
	}}))
	_ = fixture // fixture guarantees the auth wrapper pattern; observing is wrapped the same way
	auth := &mcpAuth{next: observing, identity: map[string]string{"token-a": "user-a"}}
	server := httptest.NewServer(auth)
	t.Cleanup(server.Close)

	client := sdkmcp.NewClient(&sdkmcp.Implementation{Name: "mcp-test", Version: "v0.0.1"}, nil)
	session, err := client.Connect(context.Background(), &sdkmcp.StreamableClientTransport{
		Endpoint:   server.URL + "/mcp",
		HTTPClient: &http.Client{Transport: bearerTransport{token: "token-a", base: http.DefaultTransport}},
	}, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = session.Close() })

	_, err = session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": "hi"}})
	require.NoError(t, err)
	mu.Lock()
	defer mu.Unlock()
	var sawList, sawResolve bool
	for _, identity := range identities {
		switch identity {
		case "list:user-a":
			sawList = true
		case "resolve:user-a":
			sawResolve = true
		default:
			t.Fatalf("unexpected identity observation %q", identity)
		}
	}
	assert.True(t, sawList, "listing must observe the authenticated identity")
	assert.True(t, sawResolve, "invocation must observe the same authenticated identity")
}

func TestMCPDuplicateAndInvalidCatalogNames(t *testing.T) {
	agent := newMCPTestAgent()
	for _, test := range []struct {
		name     string
		entries  []MCPEntry
		fragment string
	}{
		{
			name:     "duplicate names",
			entries:  []MCPEntry{{Tool: "dup", Target: "a"}, {Tool: "dup", Target: "b"}},
			fragment: "duplicate",
		},
		{
			name:     "invalid characters",
			entries:  []MCPEntry{{Tool: "bad name!", Target: "a"}},
			fragment: "invalid characters",
		},
		{
			name:     "empty name",
			entries:  []MCPEntry{{Tool: "", Target: "a"}},
			fragment: "empty",
		},
		{
			name:     "empty target",
			entries:  []MCPEntry{{Tool: "ok", Target: "  "}},
			fragment: "empty target",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			var logged atomic.Int32
			handler := New(chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
				return agent, chat.Info{}, nil
			}), WithMCP(MCPConfig{List: func(context.Context) ([]MCPEntry, error) {
				return test.entries, nil
			}}), WithErrorLog(func(context.Context, error) { logged.Add(1) }))
			server := httptest.NewServer(handler)
			t.Cleanup(server.Close)

			body := `{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}`
			req, err := http.NewRequest(http.MethodPost, server.URL+"/mcp", strings.NewReader(body))
			require.NoError(t, err)
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Accept", "application/json, text/event-stream")
			req.Header.Set("Mcp-Protocol-Version", mcpProtocolVersion)
			resp, err := http.DefaultClient.Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()
			assert.Equal(t, http.StatusInternalServerError, resp.StatusCode)
			assert.Equal(t, int32(1), logged.Load(), "catalog failures must be logged, never exposed")
		})
	}
	// No listing failure may ever reach the agent.
	assert.Equal(t, int32(0), agent.runs.Load())
}

func TestMCPStatelessIndependentRequests(t *testing.T) {
	agent := newMCPTestAgent()
	fixture := newMCPFixture(t, agent, WithMCP(MCPConfig{List: func(context.Context) ([]MCPEntry, error) {
		return []MCPEntry{{Tool: "echo", Target: "agent/echo"}}, nil
	}}))
	// Every response is a completed tool result with no session affinity.
	result := fixture.rawToolsList(t, "token-a")
	require.NotNil(t, result["tools"])

	req, err := http.NewRequest(http.MethodGet, fixture.endpoint(), nil)
	require.NoError(t, err)
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Mcp-Protocol-Version", mcpProtocolVersion)
	req.Header.Set("Authorization", "Bearer token-a")
	resp, err := f_clientDo(t, fixture, req)
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusMethodNotAllowed, resp.StatusCode)

	// Two fully independent sessions each observe the same single-run state.
	session1 := fixture.connect(t, "token-a")
	session2 := fixture.connect(t, "token-a")
	for _, session := range []*sdkmcp.ClientSession{session1, session2} {
		call, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": "hi"}})
		require.NoError(t, err)
		require.False(t, call.IsError)
	}
	assert.Equal(t, int32(2), agent.runs.Load())
}

func f_clientDo(t *testing.T, f *mcpFixture, req *http.Request) (*http.Response, error) {
	t.Helper()
	req.Header.Set("Authorization", "Bearer token-a")
	return f.http.Client().Do(req)
}

func TestMCPRoutingAndPrefixMounting(t *testing.T) {
	agent := newMCPTestAgent()
	handler := New(chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
		return agent, chat.Info{}, nil
	}), WithMCP(MCPConfig{List: func(context.Context) ([]MCPEntry, error) {
		return []MCPEntry{{Tool: "echo", Target: "agent/echo"}}, nil
	}}))

	t.Run("disabled by default", func(t *testing.T) {
		plain := New(chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
			return agent, chat.Info{}, nil
		}))
		recorder := postJSON(t, plain, "/mcp", `{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}`, nil)
		assert.Equal(t, http.StatusNotFound, recorder.Code)
	})
	t.Run("prefixed mounting", func(t *testing.T) {
		server := httptest.NewServer(http.StripPrefix("/v1", handler))
		t.Cleanup(server.Close)

		// Exact path inside the prefix works.
		body := fmt.Sprintf(`{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"_meta":{%s}}}`, rawMCPMeta)
		req, err := http.NewRequest(http.MethodPost, server.URL+"/v1/mcp", strings.NewReader(body))
		require.NoError(t, err)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json, text/event-stream")
		req.Header.Set("Mcp-Protocol-Version", mcpProtocolVersion)
		req.Header.Set("Mcp-Method", "tools/list")
		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		defer resp.Body.Close()
		assert.Equal(t, http.StatusOK, resp.StatusCode)

		// The un-prefixed path is not routed.
		resp2, err := http.Post(server.URL+"/mcp", "application/json", strings.NewReader(body))
		require.NoError(t, err)
		defer resp2.Body.Close()
		assert.Equal(t, http.StatusNotFound, resp2.StatusCode)
	})
	t.Run("chat endpoints unaffected", func(t *testing.T) {
		recorder := postJSON(t, handler, "/chat/completions", `{"model":"agent/echo","messages":[{"role":"user","content":"hi"}]}`, nil)
		assert.Equal(t, http.StatusOK, recorder.Code)
	})
}

func TestMCPOldRevisionRejected(t *testing.T) {
	agent := newMCPTestAgent()
	handler := New(chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
		return agent, chat.Info{}, nil
	}), WithMCP(MCPConfig{List: func(context.Context) ([]MCPEntry, error) {
		return []MCPEntry{{Tool: "echo", Target: "agent/echo"}}, nil
	}}))
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	// A request identifying a retired revision is rejected at the transport.
	body := `{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}`
	req, err := http.NewRequest(http.MethodPost, server.URL+"/mcp", strings.NewReader(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("Mcp-Protocol-Version", "2025-11-25")
	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
	assert.Contains(t, resp.Header.Get("Content-Type"), "text/plain")
}

func TestMCPJSONRPCIDIsNotIdempotencyKey(t *testing.T) {
	agent := newMCPTestAgent()
	handler := New(chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
		return agent, chat.Info{}, nil
	}), WithMCP(MCPConfig{List: func(context.Context) ([]MCPEntry, error) {
		return []MCPEntry{{Tool: "echo", Target: "agent/echo"}}, nil
	}}))
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	rawCall := func() {
		t.Helper()
		body := fmt.Sprintf(`{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"_meta":{%s},"name":"echo","arguments":{"message":"hi"}}}`, rawMCPMeta)
		req, err := http.NewRequest(http.MethodPost, server.URL+"/mcp", strings.NewReader(body))
		require.NoError(t, err)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json, text/event-stream")
		req.Header.Set("Mcp-Protocol-Version", mcpProtocolVersion)
		req.Header.Set("Mcp-Method", "tools/call")
		req.Header.Set("Mcp-Name", "echo")
		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		defer resp.Body.Close()
		require.Equal(t, http.StatusOK, resp.StatusCode)
	}
	rawCall()
	rawCall()
	assert.Equal(t, int32(2), agent.runs.Load(), "the same JSON-RPC request ID must execute twice")
}

func TestMCPOutputMappingAndLimits(t *testing.T) {
	t.Run("output order preserved", func(t *testing.T) {
		agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			if err := emit(chat.TextDelta("one ")); err != nil {
				return chat.Outcome{}, err
			}
			if err := emit(chat.TextDelta("two")); err != nil {
				return chat.Outcome{}, err
			}
			if err := emit(chat.Text("three")); err != nil {
				return chat.Outcome{}, err
			}
			return chat.Outcome{}, nil
		})
		session := newPlainMCPSession(t, agent)
		call, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": "hi"}})
		require.NoError(t, err)
		require.False(t, call.IsError)
		require.Len(t, call.Content, 2)
		assert.Equal(t, "one two", call.Content[0].(*sdkmcp.TextContent).Text)
		assert.Equal(t, "three", call.Content[1].(*sdkmcp.TextContent).Text)
	})
	t.Run("output limit is sanitized", func(t *testing.T) {
		agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Text("this output is far too long"))
		})
		session := newPlainMCPSession(t, agent, WithLimits(chat.Limits{MaxOutputBytes: 16}))
		call, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": "hi"}})
		require.NoError(t, err)
		require.True(t, call.IsError)
		text := call.Content[0].(*sdkmcp.TextContent).Text
		assert.Equal(t, "agent output exceeds the configured limit", text)
	})
	t.Run("agent failure is sanitized", func(t *testing.T) {
		agent := chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, errors.New("upstream 500 with stack trace and api key sk-123")
		})
		session := newPlainMCPSession(t, agent, WithErrorLog(func(context.Context, error) {}))
		call, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": "hi"}})
		require.NoError(t, err)
		require.True(t, call.IsError)
		text := call.Content[0].(*sdkmcp.TextContent).Text
		assert.NotContains(t, text, "sk-123")
		assert.Equal(t, "the agent could not complete this request", text)
	})
	t.Run("agent-declared failure keeps public message", func(t *testing.T) {
		agent := chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{Status: chat.StatusFailed}, &chat.Error{Status: 429, Type: "rate_limit_error", Code: "rate_limited", Message: "quota exceeded"}
		})
		session := newPlainMCPSession(t, agent)
		call, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": "hi"}})
		require.NoError(t, err)
		require.True(t, call.IsError)
		assert.Equal(t, "quota exceeded", call.Content[0].(*sdkmcp.TextContent).Text)
	})
	t.Run("non-text output rejected explicitly", func(t *testing.T) {
		agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.MediaItem(chat.ImagePart(chat.InlineMedia("image/png", []byte{1}))))
		})
		session := newPlainMCPSession(t, agent, WithLimits(chat.Limits{MaxOutputBytes: 1 << 20}))
		call, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": "hi"}})
		require.NoError(t, err)
		require.True(t, call.IsError)
		text := call.Content[0].(*sdkmcp.TextContent).Text
		assert.Contains(t, text, "image output")
	})
	t.Run("internal tool calls never become client tool requests", func(t *testing.T) {
		agent := chat.AgentFunc(func(_ context.Context, _ *chat.Request, emit chat.Emit) (chat.Outcome, error) {
			return chat.Outcome{}, emit(chat.Tool("call_1", "lookup", `{"q":"x"}`))
		})
		session := newPlainMCPSession(t, agent, WithLimits(chat.Limits{MaxOutputBytes: 1 << 20}))
		call, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": "hi"}})
		require.NoError(t, err)
		require.True(t, call.IsError)
		text := call.Content[0].(*sdkmcp.TextContent).Text
		assert.Contains(t, text, "tool calls")
	})
}

// newPlainMCPSession starts an unauthenticated MCP server with a single echo
// tool bound to agent and returns a connected official-SDK session.
func newPlainMCPSession(t *testing.T, agent chat.Agent, options ...Option) *sdkmcp.ClientSession {
	t.Helper()
	handler := New(chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
		return agent, chat.Info{}, nil
	}), append(options, WithMCP(MCPConfig{List: func(context.Context) ([]MCPEntry, error) {
		return []MCPEntry{{Tool: "echo", Target: "agent/echo", Info: chat.Info{Description: "echo"}}}, nil
	}}))...)
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	client := sdkmcp.NewClient(&sdkmcp.Implementation{Name: "mcp-test", Version: "v0.0.1"}, nil)
	session, err := client.Connect(context.Background(), &sdkmcp.StreamableClientTransport{Endpoint: server.URL + "/mcp"}, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = session.Close() })
	return session
}

func TestMCPCancellation(t *testing.T) {
	agent := newMCPTestAgent()
	agent.block = make(chan chan struct{}, 1)
	session := newPlainMCPSession(t, agent)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := session.CallTool(ctx, &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": "hi"}})
		done <- err
	}()

	var release chan struct{}
	select {
	case release = <-agent.block:
	case <-time.After(5 * time.Second):
		t.Fatal("agent was not started")
	}
	cancel()
	select {
	case err := <-done:
		require.Error(t, err, "cancelled tool calls must not return results")
	case <-time.After(5 * time.Second):
		t.Fatal("cancellation did not reach the tool call")
	}
	select {
	case <-release:
	default:
	}
}

func TestMCPLifecycle(t *testing.T) {
	type event struct {
		kind string
		id   string
	}
	var mu sync.Mutex
	var events []event
	store := make(map[string]chat.Response)

	life := chat.Lifecycle(func(_ context.Context, tr *chat.TurnRequest) (chat.Acceptance, error) {
		require.NotNil(t, tr.Request)
		key := tr.Request.Target + "|" + tr.Turn[0].Content[0].Text
		mu.Lock()
		if prior, ok := store[key]; ok {
			clone := prior.Clone()
			mu.Unlock()
			return chat.Acceptance{Replay: &clone}, nil
		}
		mu.Unlock()
		finished := false
		return chat.Acceptance{
			Response: chat.Response{ID: "resp_mcp_identity"},
			Finish: func(_ context.Context, resp *chat.Response, _ error) error {
				require.False(t, finished, "Finish must run exactly once")
				finished = true
				mu.Lock()
				events = append(events, event{kind: "finish", id: resp.ID})
				store[key] = resp.Clone()
				mu.Unlock()
				return nil
			},
		}, nil
	})

	agent := newMCPTestAgent()
	handler := New(chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
		return agent, chat.Info{}, nil
	}), WithLifecycle(life), WithMCP(MCPConfig{List: func(context.Context) ([]MCPEntry, error) {
		return []MCPEntry{{Tool: "echo", Target: "agent/echo"}}, nil
	}}))
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	client := sdkmcp.NewClient(&sdkmcp.Implementation{Name: "mcp-test", Version: "v0.0.1"}, nil)
	session, err := client.Connect(context.Background(), &sdkmcp.StreamableClientTransport{Endpoint: server.URL + "/mcp"}, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = session.Close() })

	t.Run("acceptance identity and persistence before result", func(t *testing.T) {
		call, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": "one"}})
		require.NoError(t, err)
		require.False(t, call.IsError)
		mu.Lock()
		defer mu.Unlock()
		require.NotEmpty(t, events)
		assert.Equal(t, "finish", events[len(events)-1].kind)
		assert.Equal(t, "resp_mcp_identity", events[len(events)-1].id)
		assert.Contains(t, store, "agent/echo|one")
		assert.Equal(t, int32(1), agent.runs.Load())
	})
	t.Run("replay skips execution and maps stored output", func(t *testing.T) {
		call, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": "one"}})
		require.NoError(t, err)
		require.False(t, call.IsError)
		text := call.Content[0].(*sdkmcp.TextContent).Text
		assert.Equal(t, "echo: one", text)
		assert.Equal(t, int32(1), agent.runs.Load(), "replay must not re-run the agent")
	})
	t.Run("finalization failure maps to sanitized tool error", func(t *testing.T) {
		failing := chat.Lifecycle(func(context.Context, *chat.TurnRequest) (chat.Acceptance, error) {
			return chat.Acceptance{Response: chat.Response{ID: "resp_x"}, Finish: func(context.Context, *chat.Response, error) error {
				return errors.New("persistence backend exploded")
			}}, nil
		})
		session := newPlainMCPSession(t, newMCPTestAgent(), WithLifecycle(failing), WithErrorLog(func(context.Context, error) {}))
		call, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": "hi"}})
		require.NoError(t, err)
		require.True(t, call.IsError, "a failed Finish must not produce a successful tool result")
		text := call.Content[0].(*sdkmcp.TextContent).Text
		assert.NotContains(t, text, "exploded")
		assert.Contains(t, text, "could not complete")
	})
}

func TestMCPExecutionLimits(t *testing.T) {
	t.Run("empty listing is valid", func(t *testing.T) {
		session := newPlainMCPSessionWithEntries(t, newMCPTestAgent(), nil)
		result, err := session.ListTools(context.Background(), nil)
		require.NoError(t, err)
		assert.Empty(t, result.Tools)
	})
	t.Run("oversized message", func(t *testing.T) {
		agent := newMCPTestAgent()
		handler := New(chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
			return agent, chat.Info{}, nil
		}), WithLimits(chat.Limits{MaxRequestBytes: 64}), WithMCP(MCPConfig{List: func(context.Context) ([]MCPEntry, error) {
			return []MCPEntry{{Tool: "echo", Target: "agent/echo"}}, nil
		}}))
		server := httptest.NewServer(handler)
		t.Cleanup(server.Close)
		client := sdkmcp.NewClient(&sdkmcp.Implementation{Name: "mcp-test", Version: "v0.0.1"}, nil)
		session, err := client.Connect(context.Background(), &sdkmcp.StreamableClientTransport{Endpoint: server.URL + "/mcp"}, nil)
		require.NoError(t, err)
		t.Cleanup(func() { _ = session.Close() })
		big := strings.Repeat("x", 4096)
		_, err = session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": big}})
		require.Error(t, err, "requests beyond the configured limit must be rejected")
	})
}

func newPlainMCPSessionWithEntries(t *testing.T, agent chat.Agent, entries []MCPEntry) *sdkmcp.ClientSession {
	t.Helper()
	handler := New(chat.Resolver(func(context.Context, string) (chat.Agent, chat.Info, error) {
		return agent, chat.Info{}, nil
	}), WithMCP(MCPConfig{List: func(context.Context) ([]MCPEntry, error) { return entries, nil }}))
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	client := sdkmcp.NewClient(&sdkmcp.Implementation{Name: "mcp-test", Version: "v0.0.1"}, nil)
	session, err := client.Connect(context.Background(), &sdkmcp.StreamableClientTransport{Endpoint: server.URL + "/mcp"}, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = session.Close() })
	return session
}

func TestMCPStructuredFailureShape(t *testing.T) {
	// Every tool-execution error must be a completed tool result with
	// IsError=true, never a transport-level failure and never raw JSON.
	agent := chat.AgentFunc(func(context.Context, *chat.Request, chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, errors.New("boom")
	})
	session := newPlainMCPSession(t, agent, WithErrorLog(func(context.Context, error) {}))
	call, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": "hi"}})
	require.NoError(t, err)
	require.True(t, call.IsError)
	require.NotEmpty(t, call.Content)
	for _, content := range call.Content {
		_, ok := content.(*sdkmcp.TextContent)
		assert.True(t, ok, "failure content must be typed text")
	}
}

var _ = bearerTransport{}
var _ = bytes.MinRead
