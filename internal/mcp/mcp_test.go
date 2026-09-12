// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/protocol"
	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type testHost struct {
	catalog  map[string]chat.Info
	listErr  error
	runErr   error
	response chat.Response
	limits   chat.Limits
	runs     int
	last     protocol.ParsedRequest
	logged   []error
}

func (h *testHost) List(context.Context) (map[string]chat.Info, error) {
	return h.catalog, h.listErr
}

func (h *testHost) Run(_ context.Context, _ string, parsed *protocol.ParsedRequest) (chat.Response, error) {
	h.runs++
	h.last = *parsed
	if h.runErr != nil {
		return chat.Response{}, h.runErr
	}
	return h.response, nil
}

func (h *testHost) Limits() chat.Limits {
	return h.limits
}

func (h *testHost) LogError(_ context.Context, err error) {
	h.logged = append(h.logged, err)
}

func TestProjectTool(t *testing.T) {
	t.Run("empty tool is filtered", func(t *testing.T) {
		_, ok := projectTool("agent/a", chat.Info{Description: "x"})
		assert.False(t, ok)
		_, ok = projectTool("agent/a", chat.Info{Tool: "   "})
		assert.False(t, ok)
	})
	t.Run("nonempty tool keeps target", func(t *testing.T) {
		entry, ok := projectTool("agent/a", chat.Info{Tool: "public", Description: "desc"})
		require.True(t, ok)
		assert.Equal(t, "public", entry.tool)
		assert.Equal(t, "agent/a", entry.target)
		assert.Equal(t, "desc", entry.info.Description)
	})
}

func TestValidateToolName(t *testing.T) {
	assert.NoError(t, validateToolName("echo"))
	assert.NoError(t, validateToolName("Echo_1.2-3"))
	assert.Error(t, validateToolName(""))
	assert.Error(t, validateToolName("bad name!"))
	assert.Error(t, validateToolName(strings.Repeat("a", 129)))
}

func TestDecodeMessage(t *testing.T) {
	cases := map[string]struct {
		raw     json.RawMessage
		want    string
		wantErr string
	}{
		"empty":        {wantErr: "arguments must be"},
		"malformed":    {raw: json.RawMessage(`{`), wantErr: "arguments must be"},
		"array":        {raw: json.RawMessage(`[]`), wantErr: "arguments must be"},
		"unknown":      {raw: json.RawMessage(`{"extra":"x"}`), wantErr: "unknown argument"},
		"missing":      {raw: json.RawMessage(`{}`), wantErr: `"message" is required`},
		"wrongType":    {raw: json.RawMessage(`{"message":1}`), wantErr: `"message" must be a string`},
		"emptyMessage": {raw: json.RawMessage(`{"message":"  "}`), wantErr: "must not be empty"},
		"valid":        {raw: json.RawMessage(`{"message":" hello "}`), want: " hello "},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			message, err := decodeMessage(tc.raw)
			if tc.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.want, message)
		})
	}
}

func TestToolResult(t *testing.T) {
	text := chat.TextPart("hello")
	image := chat.ImagePart(chat.RemoteMedia("image/png", "https://example.com/image.png"))

	t.Run("text", func(t *testing.T) {
		resp, err := toolResultFromResponse(chat.Response{Output: []chat.Item{
			chat.MessageItem(chat.RoleAssistant, chat.TextPart(""), text),
		}})
		require.NoError(t, err)
		require.Len(t, resp.Content, 1)
		assert.Equal(t, "hello", resp.Content[0].(*sdkmcp.TextContent).Text)
	})

	cases := map[string]struct {
		response chat.Response
		wantErr  string
		check    func(t *testing.T, result *sdkmcp.CallToolResult)
	}{
		"image": {
			response: chat.Response{Output: []chat.Item{chat.MessageItem(chat.RoleAssistant, image)}},
			wantErr:  "image output",
		},
		"functionCall": {
			response: chat.Response{Output: []chat.Item{chat.FunctionCallItem("call-1", "tool", `{}`)}},
			wantErr:  "internal tool call",
		},
		"unknownItem": {
			response: chat.Response{Output: []chat.Item{{Type: "unknown"}}},
			wantErr:  "unknown output",
		},
		"incomplete": {
			response: chat.Response{Status: chat.StatusIncomplete, Incomplete: "max_output_tokens"},
			check: func(t *testing.T, result *sdkmcp.CallToolResult) {
				assert.True(t, result.IsError)
				assert.Equal(t, "output incomplete: max_output_tokens", result.Content[0].(*sdkmcp.TextContent).Text)
			},
		},
		"incompleteUnknown": {
			response: chat.Response{Status: chat.StatusIncomplete},
			check: func(t *testing.T, result *sdkmcp.CallToolResult) {
				assert.Equal(t, "output incomplete: unknown", result.Content[0].(*sdkmcp.TextContent).Text)
			},
		},
		"cancelled": {
			response: chat.Response{Status: chat.StatusCancelled},
			wantErr:  "cancelled",
		},
		"failed": {
			response: chat.Response{Status: chat.StatusFailed, Error: chat.Invalid("model", "agent failed")},
			wantErr:  "agent failed",
		},
		"failedDefault": {
			response: chat.Response{Status: chat.StatusFailed},
			wantErr:  "could not complete",
		},
		"unsupportedStatus": {
			response: chat.Response{Status: "in_progress"},
			wantErr:  "unsupported status",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			result, err := toolResultFromResponse(tc.response)
			if tc.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.wantErr)
				return
			}
			require.NoError(t, err)
			if tc.check != nil {
				tc.check(t, result)
			}
		})
	}
}

func TestTransport(t *testing.T) {
	limits := chat.DefaultLimits()

	t.Run("buildServer", func(t *testing.T) {
		host := &testHost{limits: limits}
		transport := NewTransport(host)
		server, err := transport.buildServer(map[string]chat.Info{
			"agent/z": {Tool: "z", Description: "z tool"},
			"agent/a": {Tool: "a", Description: "a tool"},
			"hidden":  {Description: "not exposed"},
		})
		require.NoError(t, err)
		assert.NotNil(t, server)
	})

	for name, catalog := range map[string]map[string]chat.Info{
		"invalidName": {"agent": {Tool: "bad name"}},
		"emptyTarget": {"": {Tool: "echo"}},
		"duplicate":   {"one": {Tool: "echo"}, "two": {Tool: "echo"}},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := NewTransport(&testHost{limits: limits}).buildServer(catalog)
			require.Error(t, err)
		})
	}

	t.Run("serveListError", func(t *testing.T) {
		host := &testHost{listErr: errors.New("catalog failed"), limits: limits}
		rec := httptest.NewRecorder()
		NewTransport(host).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/mcp", nil))
		assert.Equal(t, http.StatusInternalServerError, rec.Code)
		require.Len(t, host.logged, 1)
	})

	t.Run("serveBuildError", func(t *testing.T) {
		host := &testHost{catalog: map[string]chat.Info{"agent": {Tool: "bad name"}}, limits: limits}
		rec := httptest.NewRecorder()
		NewTransport(host).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/mcp", nil))
		assert.Equal(t, http.StatusInternalServerError, rec.Code)
		require.Len(t, host.logged, 1)
	})

	t.Run("serveHandler", func(t *testing.T) {
		host := &testHost{catalog: map[string]chat.Info{"agent": {Tool: "echo"}}, limits: limits}
		rec := httptest.NewRecorder()
		NewTransport(host).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/mcp", nil))
		assert.Equal(t, http.StatusMethodNotAllowed, rec.Code)
	})

	t.Run("callTool", func(t *testing.T) {
		host := &testHost{
			limits:   limits,
			response: chat.Response{Output: []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("done"))}},
		}
		transport := NewTransport(host)
		handler := transport.callAgentTool(entry{tool: "echo", target: "agent"})
		request := &sdkmcp.CallToolRequest{Params: &sdkmcp.CallToolParamsRaw{Arguments: json.RawMessage(`{"message":"hello"}`)}}
		result, err := handler(context.Background(), request)
		require.NoError(t, err)
		require.Len(t, result.Content, 1)
		assert.Equal(t, "done", result.Content[0].(*sdkmcp.TextContent).Text)
		assert.Equal(t, 1, host.runs)
		assert.Equal(t, "agent", host.last.Request.Target)
		assert.Equal(t, "hello", host.last.Request.Input[0].Content[0].Text)
	})

	t.Run("callErrors", func(t *testing.T) {
		tooSmall := limits
		tooSmall.MaxRequestBytes = 1
		host := &testHost{limits: tooSmall}
		handler := NewTransport(host).callAgentTool(entry{target: "agent"})
		request := func(raw string) *sdkmcp.CallToolRequest {
			return &sdkmcp.CallToolRequest{Params: &sdkmcp.CallToolParamsRaw{Arguments: json.RawMessage(raw)}}
		}

		_, err := handler(context.Background(), request(`{"message":"hello"}`))
		var rpcErr *jsonrpc.Error
		require.Error(t, err)
		require.True(t, errors.As(err, &rpcErr))
		assert.Equal(t, int64(jsonrpc.CodeInvalidParams), rpcErr.Code)

		host.limits = limits
		_, err = handler(context.Background(), request(`{"message":1}`))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "must be a string")

		host.runErr = errors.New("private failure")
		result, err := handler(context.Background(), request(`{"message":"hello"}`))
		require.NoError(t, err)
		assert.True(t, result.IsError)
		assert.Equal(t, "the agent could not complete this request", result.Content[0].(*sdkmcp.TextContent).Text)
		require.Len(t, host.logged, 1)
	})
}

func TestToolMessage(t *testing.T) {
	assert.Equal(t, "visible", publicToolMessage(chat.Invalid("x", "visible")))
	assert.Equal(t, "the agent could not complete this request", publicToolMessage(errors.New("secret")))
	assert.Equal(t, "the agent could not complete this request", publicToolMessage(&chat.Error{}))
}
