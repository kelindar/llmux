// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
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

// acceptStore implements Store with a custom Accept hook for lifecycle tests.
type acceptStore struct {
	accept func(context.Context, *chat.TurnRequest) (chat.Acceptance, error)
}

func (s acceptStore) Load(context.Context, string) ([]chat.Item, error) {
	return nil, errors.New("unsupported")
}

func (s acceptStore) Accept(ctx context.Context, tr *chat.TurnRequest) (chat.Acceptance, error) {
	return s.accept(ctx, tr)
}

// mcpAuthCatalog requires an authenticated caller identity before Load.
type mcpAuthCatalog struct {
	inner Catalog
}

func (c *mcpAuthCatalog) List(ctx context.Context) (map[string]chat.Info, error) {
	return c.inner.List(ctx)
}

func (c *mcpAuthCatalog) Load(ctx context.Context, target string) (chat.Agent, chat.Info, error) {
	if mcpCaller(ctx) == "" {
		return nil, chat.Info{}, &chat.Error{Status: http.StatusUnauthorized, Type: "authentication_error", Code: "missing_identity", Message: "request identity is required"}
	}
	return c.inner.Load(ctx, target)
}

// catalogWithTools builds a fixedCatalog with static entries for MCP tests.
func catalogWithTools(agent chat.Agent, entries map[string]chat.Info) *fixedCatalog {
	return &fixedCatalog{agent: agent, entries: entries}
}

// newMCPFixture wires a handler behind the illustrative auth wrapper with
// caller-specific listings and mounts it under an application prefix.
func newMCPFixture(t *testing.T, catalog Catalog, options ...Option) *mcpFixture {
	t.Helper()
	handler := New(&mcpAuthCatalog{inner: catalog}, options...)
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

// withTools enables the MCP endpoint. Catalog entries belong on the Catalog
// passed to New.
func withTools(_ map[string]chat.Info) []Option {
	return []Option{WithMCP()}
}

func TestMCPDiscovery(t *testing.T) {
	agent := newMCPTestAgent()
	catalog := &fixedCatalog{
		agent: agent,
		listFn: func(ctx context.Context) (map[string]chat.Info, error) {
			require.Equal(t, "user-a", mcpCaller(ctx), "listing must receive the authenticated context")
			return map[string]chat.Info{
				"agent/zeta":  {Description: "last", Tool: "zeta"},
				"agent/alpha": {Description: "first", Tool: "alpha"},
			}, nil
		},
		loadFn: func(_ context.Context, target string) (chat.Agent, chat.Info, error) {
			return agent, chat.Info{Description: "test agent for " + target}, nil
		},
	}
	fixture := newMCPFixture(t, catalog, WithMCP())
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

	// The invocation resolved the listed target through Catalog.Load.
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

func TestMCPCallerListing(t *testing.T) {
	agent := newMCPTestAgent()
	catalog := &fixedCatalog{
		agent: agent,
		listFn: func(ctx context.Context) (map[string]chat.Info, error) {
			switch mcpCaller(ctx) {
			case "user-a":
				return map[string]chat.Info{"agent/a": {Description: "only for a", Tool: "tool-a"}}, nil
			case "user-b":
				return map[string]chat.Info{"agent/b": {Description: "only for b", Tool: "tool-b"}}, nil
			default:
				return nil, nil
			}
		},
		loadFn: func(_ context.Context, target string) (chat.Agent, chat.Info, error) {
			return agent, chat.Info{Description: "test agent for " + target}, nil
		},
	}
	fixture := newMCPFixture(t, catalog, WithMCP())

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

func TestMCPInputSchema(t *testing.T) {
	agent := newMCPTestAgent()
	fixture := newMCPFixture(t, catalogWithTools(agent, map[string]chat.Info{
		"agent/echo": {Description: "echo", Tool: "echo"},
	}), withTools(nil)...)
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

func TestMCPUnauthorized(t *testing.T) {
	agent := newMCPTestAgent()
	fixture := newMCPFixture(t, &fixedCatalog{
		agent: agent,
		listFn: func(ctx context.Context) (map[string]chat.Info, error) {
			if mcpCaller(ctx) != "user-a" {
				return nil, nil
			}
			return map[string]chat.Info{"agent/ok": {Description: "ok", Tool: "listed"}}, nil
		},
		loadFn: func(_ context.Context, target string) (chat.Agent, chat.Info, error) {
			return agent, chat.Info{}, nil
		},
	}, WithMCP())
	session := fixture.connect(t, "token-a")

	t.Run("unknown tool", func(t *testing.T) {
		_, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "nope", Arguments: map[string]any{"message": "hi"}})
		require.Error(t, err)
		var wireErr *jsonrpc.Error
		require.ErrorAs(t, err, &wireErr)
		assert.EqualValues(t, jsonrpc.CodeInvalidParams, wireErr.Code)
	})
	t.Run("listed but load denies target", func(t *testing.T) {
		denying := New(&fixedCatalog{
			entries: map[string]chat.Info{"agent/denied": {Description: "ok", Tool: "listed"}},
			loadFn: func(_ context.Context, target string) (chat.Agent, chat.Info, error) {
				return nil, chat.Info{}, &chat.Error{Status: http.StatusForbidden, Type: "permission_error", Code: "target_forbidden", Message: "you may not call this agent"}
			},
		}, withTools(nil)...)
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
	t.Run("load failure without public error is sanitized", func(t *testing.T) {
		broken := New(&fixedCatalog{
			entries: map[string]chat.Info{"agent/x": {Description: "ok", Tool: "listed"}},
			loadFn: func(context.Context, string) (chat.Agent, chat.Info, error) {
				return nil, chat.Info{}, errors.New("database connection string leaked")
			},
		}, append(withTools(nil), WithErrorLog(func(context.Context, error) {}))...)
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
	fixture := newMCPFixture(t, catalogWithTools(agent, map[string]chat.Info{"agent/echo": {Tool: "echo"}}), withTools(nil)...)
	resp, err := http.Post(fixture.endpoint(), "application/json", strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}`))
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusUnauthorized, resp.StatusCode)
	assert.Equal(t, `Bearer realm="llmux-mcp", resource_metadata="/.well-known/oauth-protected-resource"`, resp.Header.Get("WWW-Authenticate"))
	assert.Equal(t, int32(0), agent.runs.Load())
}

func TestMCPAuthContext(t *testing.T) {
	agent := newMCPTestAgent()
	var mu sync.Mutex
	var identities []string
	observingCatalog := &fixedCatalog{
		agent:   agent,
		entries: map[string]chat.Info{"agent/echo": {Tool: "echo"}},
		listFn: func(ctx context.Context) (map[string]chat.Info, error) {
			mu.Lock()
			identities = append(identities, "list:"+mcpCaller(ctx))
			mu.Unlock()
			return map[string]chat.Info{"agent/echo": {Tool: "echo"}}, nil
		},
		loadFn: func(ctx context.Context, target string) (chat.Agent, chat.Info, error) {
			mu.Lock()
			identities = append(identities, "load:"+mcpCaller(ctx))
			mu.Unlock()
			return agent, chat.Info{}, nil
		},
	}
	observing := New(&mcpAuthCatalog{inner: observingCatalog}, WithMCP())
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
	var sawList, sawLoad bool
	for _, identity := range identities {
		switch identity {
		case "list:user-a":
			sawList = true
		case "load:user-a":
			sawLoad = true
		default:
			t.Fatalf("unexpected identity observation %q", identity)
		}
	}
	assert.True(t, sawList, "listing must observe the authenticated identity")
	assert.True(t, sawLoad, "invocation must observe the same authenticated identity")
}

func TestMCPCatalogNames(t *testing.T) {
	agent := newMCPTestAgent()
	for _, test := range []struct {
		name     string
		catalog  map[string]chat.Info
		fragment string
	}{
		{
			name:     "duplicate names",
			catalog:  map[string]chat.Info{"a": {Tool: "dup"}, "b": {Tool: "dup"}},
			fragment: "duplicate",
		},
		{
			name:     "invalid characters",
			catalog:  map[string]chat.Info{"a": {Tool: "bad name!"}},
			fragment: "invalid characters",
		},
		{
			name:     "empty target",
			catalog:  map[string]chat.Info{"  ": {Tool: "ok"}},
			fragment: "empty target",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			var logged atomic.Int32
			handler := New(&fixedCatalog{
				agent: agent,
				listFn: func(context.Context) (map[string]chat.Info, error) {
					return test.catalog, nil
				},
			}, WithMCP(), WithErrorLog(func(context.Context, error) { logged.Add(1) }))
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

func TestMCPStateless(t *testing.T) {
	agent := newMCPTestAgent()
	fixture := newMCPFixture(t, catalogWithTools(agent, map[string]chat.Info{"agent/echo": {Tool: "echo"}}), withTools(nil)...)
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

func TestMCPRouting(t *testing.T) {
	agent := newMCPTestAgent()
	handler := New(catalogWithTools(agent, map[string]chat.Info{"agent/echo": {Tool: "echo"}}), withTools(nil)...)

	t.Run("disabled by default", func(t *testing.T) {
		plain := New(catalogWithTools(agent, nil))
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

func TestMCPOldRevision(t *testing.T) {
	agent := newMCPTestAgent()
	handler := New(catalogWithTools(agent, map[string]chat.Info{"agent/echo": {Tool: "echo"}}), withTools(nil)...)
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

func TestMCPRequestID(t *testing.T) {
	agent := newMCPTestAgent()
	handler := New(catalogWithTools(agent, map[string]chat.Info{"agent/echo": {Tool: "echo"}}), withTools(nil)...)
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

func TestMCPOutputLimits(t *testing.T) {
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
	handler := New(catalogWithTools(agent, map[string]chat.Info{
		"agent/echo": {Description: "echo", Tool: "echo"},
	}), append(options, WithMCP())...)
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
	persisted := make(map[string]chat.Response)

	life := acceptStore{accept: func(_ context.Context, tr *chat.TurnRequest) (chat.Acceptance, error) {
		require.NotNil(t, tr.Request)
		key := tr.Request.Target + "|" + tr.Turn[0].Content[0].Text
		mu.Lock()
		if prior, ok := persisted[key]; ok {
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
				persisted[key] = resp.Clone()
				mu.Unlock()
				return nil
			},
		}, nil
	}}

	agent := newMCPTestAgent()
	handler := New(catalogWithTools(agent, map[string]chat.Info{"agent/echo": {Tool: "echo"}}), WithStore(life), WithMCP())
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
		assert.Contains(t, persisted, "agent/echo|one")
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
		failing := acceptStore{accept: func(context.Context, *chat.TurnRequest) (chat.Acceptance, error) {
			return chat.Acceptance{Response: chat.Response{ID: "resp_x"}, Finish: func(context.Context, *chat.Response, error) error {
				return errors.New("persistence backend exploded")
			}}, nil
		}}
		session := newPlainMCPSession(t, newMCPTestAgent(), WithStore(failing), WithErrorLog(func(context.Context, error) {}))
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
		session := newPlainMCPSessionWithCatalog(t, newMCPTestAgent(), map[string]chat.Info{})
		result, err := session.ListTools(context.Background(), nil)
		require.NoError(t, err)
		assert.Empty(t, result.Tools)
	})
	t.Run("nil catalog yields empty tools", func(t *testing.T) {
		session := newPlainMCPSessionWithCatalog(t, newMCPTestAgent(), nil)
		result, err := session.ListTools(context.Background(), nil)
		require.NoError(t, err)
		assert.Empty(t, result.Tools)
	})
	t.Run("oversized message", func(t *testing.T) {
		agent := newMCPTestAgent()
		handler := New(catalogWithTools(agent, map[string]chat.Info{"agent/echo": {Tool: "echo"}}), append([]Option{WithLimits(chat.Limits{MaxRequestBytes: 64})}, WithMCP())...)
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

func newPlainMCPSessionWithCatalog(t *testing.T, agent chat.Agent, entries map[string]chat.Info) *sdkmcp.ClientSession {
	t.Helper()
	var catalog Catalog = &fixedCatalog{agent: agent, entries: entries}
	if entries == nil {
		catalog = nil
	}
	handler := New(catalog, WithMCP())
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	client := sdkmcp.NewClient(&sdkmcp.Implementation{Name: "mcp-test", Version: "v0.0.1"}, nil)
	session, err := client.Connect(context.Background(), &sdkmcp.StreamableClientTransport{Endpoint: server.URL + "/mcp"}, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = session.Close() })
	return session
}

func TestMCPFailureShape(t *testing.T) {
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

func TestCatalogProjection(t *testing.T) {
	agent := newMCPTestAgent()
	owned := map[string]chat.Info{
		"agent/zeta": {
			Description: "listed in models only",
			Created:     100,
			OwnedBy:     "team-z",
		},
		"agent/alpha": {
			Description: "public echo",
			Tool:        "public_echo",
			Created:     50,
			OwnedBy:     "team-a",
			Extensions:  map[string]bool{"keep": true},
		},
	}
	fixture := newMCPFixture(t, &fixedCatalog{agent: agent, entries: owned}, WithMCP())

	t.Run("models includes all targets in deterministic order", func(t *testing.T) {
		req, err := http.NewRequest(http.MethodGet, fixture.http.URL+"/v1/models", nil)
		require.NoError(t, err)
		req.Header.Set("Authorization", "Bearer token-a")
		resp, err := fixture.http.Client().Do(req)
		require.NoError(t, err)
		defer resp.Body.Close()
		require.Equal(t, http.StatusOK, resp.StatusCode)
		var body struct {
			Object string `json:"object"`
			Data   []struct {
				ID      string `json:"id"`
				Object  string `json:"object"`
				Created int64  `json:"created"`
				OwnedBy string `json:"owned_by"`
			} `json:"data"`
		}
		require.NoError(t, json.NewDecoder(resp.Body).Decode(&body))
		require.Len(t, body.Data, 2)
		assert.Equal(t, "agent/alpha", body.Data[0].ID)
		assert.Equal(t, int64(50), body.Data[0].Created)
		assert.Equal(t, "team-a", body.Data[0].OwnedBy)
		assert.Equal(t, "agent/zeta", body.Data[1].ID)
		assert.Equal(t, int64(100), body.Data[1].Created)
		assert.Equal(t, "team-z", body.Data[1].OwnedBy)
	})

	t.Run("mcp exposes only nonempty Tool names differing from targets", func(t *testing.T) {
		session := fixture.connect(t, "token-a")
		listed, err := session.ListTools(context.Background(), nil)
		require.NoError(t, err)
		require.Len(t, listed.Tools, 1)
		assert.Equal(t, "public_echo", listed.Tools[0].Name)
		assert.Equal(t, "public echo", listed.Tools[0].Description)

		call, err := session.CallTool(context.Background(), &sdkmcp.CallToolParams{
			Name:      "public_echo",
			Arguments: map[string]any{"message": "hi"},
		})
		require.NoError(t, err)
		require.False(t, call.IsError)
		select {
		case req := <-agent.seen:
			assert.Equal(t, "agent/alpha", req.Target)
		default:
			t.Fatal("agent was not invoked through the catalog target")
		}
	})

	t.Run("application catalog map and nested values are not mutated", func(t *testing.T) {
		require.Len(t, owned, 2)
		assert.Equal(t, chat.Info{
			Description: "listed in models only",
			Created:     100,
			OwnedBy:     "team-z",
		}, owned["agent/zeta"])
		assert.Equal(t, "public_echo", owned["agent/alpha"].Tool)
		assert.Equal(t, map[string]bool{"keep": true}, owned["agent/alpha"].Extensions)
	})
}

func TestCatalogCallerModels(t *testing.T) {
	agent := newMCPTestAgent()
	fixture := newMCPFixture(t, &fixedCatalog{
		agent: agent,
		listFn: func(ctx context.Context) (map[string]chat.Info, error) {
			switch mcpCaller(ctx) {
			case "user-a":
				return map[string]chat.Info{"agent/a": {OwnedBy: "a", Tool: "tool-a"}}, nil
			case "user-b":
				return map[string]chat.Info{"agent/b": {OwnedBy: "b"}}, nil
			default:
				return nil, nil
			}
		},
		loadFn: func(_ context.Context, target string) (chat.Agent, chat.Info, error) {
			return agent, chat.Info{}, nil
		},
	}, WithMCP())

	for _, test := range []struct {
		token   string
		modelID string
		tools   int
	}{
		{"token-a", "agent/a", 1},
		{"token-b", "agent/b", 0},
	} {
		t.Run(test.token, func(t *testing.T) {
			req, err := http.NewRequest(http.MethodGet, fixture.http.URL+"/v1/models", nil)
			require.NoError(t, err)
			req.Header.Set("Authorization", "Bearer "+test.token)
			resp, err := fixture.http.Client().Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()
			require.Equal(t, http.StatusOK, resp.StatusCode)
			var body struct {
				Data []struct {
					ID string `json:"id"`
				} `json:"data"`
			}
			require.NoError(t, json.NewDecoder(resp.Body).Decode(&body))
			require.Len(t, body.Data, 1)
			assert.Equal(t, test.modelID, body.Data[0].ID)

			session := fixture.connect(t, test.token)
			listed, err := session.ListTools(context.Background(), nil)
			require.NoError(t, err)
			assert.Len(t, listed.Tools, test.tools)
		})
	}
}

func TestCatalogOptionOrder(t *testing.T) {
	agent := newMCPTestAgent()
	catalog := catalogWithTools(agent, map[string]chat.Info{"agent/echo": {Tool: "echo", Description: "echo"}})
	store := acceptStore{accept: func(context.Context, *chat.TurnRequest) (chat.Acceptance, error) {
		return chat.Acceptance{}, nil
	}}
	for _, name := range []string{"mcp then store", "store then mcp"} {
		t.Run(name, func(t *testing.T) {
			var options []Option
			if name == "mcp then store" {
				options = []Option{WithMCP(), WithStore(store)}
			} else {
				options = []Option{WithStore(store), WithMCP()}
			}
			handler := New(catalog, options...)
			require.NotNil(t, handler.mcp)
			server := httptest.NewServer(handler)
			t.Cleanup(server.Close)
			client := sdkmcp.NewClient(&sdkmcp.Implementation{Name: "mcp-test", Version: "v0.0.1"}, nil)
			session, err := client.Connect(context.Background(), &sdkmcp.StreamableClientTransport{Endpoint: server.URL + "/mcp"}, nil)
			require.NoError(t, err)
			t.Cleanup(func() { _ = session.Close() })
			listed, err := session.ListTools(context.Background(), nil)
			require.NoError(t, err)
			require.Len(t, listed.Tools, 1)
			assert.Equal(t, "echo", listed.Tools[0].Name)
		})
	}
}

func TestCatalogToolFilter(t *testing.T) {
	session := newPlainMCPSessionWithCatalog(t, newMCPTestAgent(), map[string]chat.Info{
		"agent/hidden": {Description: "models only"},
		"agent/shown":  {Description: "mcp too", Tool: "shown"},
	})
	listed, err := session.ListTools(context.Background(), nil)
	require.NoError(t, err)
	require.Len(t, listed.Tools, 1)
	assert.Equal(t, "shown", listed.Tools[0].Name)
}

func TestCatalogMCPEmpty(t *testing.T) {
	handler := New(nil, WithMCP())
	require.NotNil(t, handler.mcp)
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	client := sdkmcp.NewClient(&sdkmcp.Implementation{Name: "mcp-test", Version: "v0.0.1"}, nil)
	session, err := client.Connect(context.Background(), &sdkmcp.StreamableClientTransport{Endpoint: server.URL + "/mcp"}, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = session.Close() })
	listed, err := session.ListTools(context.Background(), nil)
	require.NoError(t, err)
	assert.Empty(t, listed.Tools)
}

func TestMCPLoadAuth(t *testing.T) {
	agent := newMCPTestAgent()
	catalog := &fixedCatalog{
		agent: agent,
		listFn: func(context.Context) (map[string]chat.Info, error) {
			return map[string]chat.Info{"agent/alpha": {Tool: "alpha", Description: "visible to all callers"}}, nil
		},
		loadFn: func(ctx context.Context, target string) (chat.Agent, chat.Info, error) {
			if mcpCaller(ctx) != "user-a" {
				return nil, chat.Info{}, &chat.Error{Status: http.StatusForbidden, Type: "permission_error", Code: "target_forbidden", Message: "load denied for this caller"}
			}
			return agent, chat.Info{Description: "test agent for " + target}, nil
		},
	}
	fixture := newMCPFixture(t, catalog, WithMCP())

	sessionA := fixture.connect(t, "token-a")
	call, err := sessionA.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "alpha", Arguments: map[string]any{"message": "ok"}})
	require.NoError(t, err)
	require.False(t, call.IsError)

	sessionB := fixture.connect(t, "token-b")
	listed, err := sessionB.ListTools(context.Background(), nil)
	require.NoError(t, err)
	require.Len(t, listed.Tools, 1, "listing visibility is independent of load authorization")

	call, err = sessionB.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "alpha", Arguments: map[string]any{"message": "nope"}})
	require.NoError(t, err)
	require.True(t, call.IsError)
	text := call.Content[0].(*sdkmcp.TextContent)
	assert.Equal(t, "load denied for this caller", text.Text)
	assert.Equal(t, int32(1), agent.runs.Load(), "denied load must not run the agent")
}

func TestMCPLoadOnce(t *testing.T) {
	var listCalls, loadCalls atomic.Int32
	agent := newMCPTestAgent()
	catalog := &fixedCatalog{
		agent: agent,
		listFn: func(context.Context) (map[string]chat.Info, error) {
			listCalls.Add(1)
			return map[string]chat.Info{"agent/echo": {Tool: "echo"}}, nil
		},
		loadFn: func(context.Context, string) (chat.Agent, chat.Info, error) {
			loadCalls.Add(1)
			return agent, chat.Info{}, nil
		},
	}
	handler := New(catalog, WithMCP())
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	client := sdkmcp.NewClient(&sdkmcp.Implementation{Name: "mcp-test", Version: "v0.0.1"}, nil)
	session, err := client.Connect(context.Background(), &sdkmcp.StreamableClientTransport{Endpoint: server.URL + "/mcp"}, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = session.Close() })

	listBefore := listCalls.Load()
	_, err = session.CallTool(context.Background(), &sdkmcp.CallToolParams{Name: "echo", Arguments: map[string]any{"message": "hi"}})
	require.NoError(t, err)
	assert.Equal(t, listBefore+1, listCalls.Load(), "each tools/call request lists once to pin name→target")
	assert.Equal(t, int32(1), loadCalls.Load(), "invocation resolves the pinned target through Load")
}

var _ = bearerTransport{}
var _ = bytes.MinRead
