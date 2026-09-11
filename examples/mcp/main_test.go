// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMCPEndpoint(t *testing.T) {
	server := httptest.NewServer(newServer())
	t.Cleanup(server.Close)

	t.Run("unauthenticated challenge", func(t *testing.T) {
		resp, err := http.Post(server.URL+"/v1/mcp", "application/json", strings.NewReader("{}"))
		require.NoError(t, err)
		defer resp.Body.Close()
		assert.Equal(t, http.StatusUnauthorized, resp.StatusCode)
		assert.NotEmpty(t, resp.Header.Get("WWW-Authenticate"))
	})

	t.Run("authenticated listing and call", func(t *testing.T) {
		client := mcp.NewClient(&mcp.Implementation{Name: "example-test", Version: "v0.0.1"}, nil)
		transport := &mcp.StreamableClientTransport{
			Endpoint:   server.URL + "/v1/mcp",
			HTTPClient: &http.Client{Transport: tokenTransport{token: "alice-token", base: http.DefaultTransport}},
		}
		session, err := client.Connect(context.Background(), transport, nil)
		require.NoError(t, err)
		t.Cleanup(func() { _ = session.Close() })

		listed, err := session.ListTools(context.Background(), nil)
		require.NoError(t, err)
		require.Len(t, listed.Tools, 1)
		assert.Equal(t, "echo", listed.Tools[0].Name)
		assert.Equal(t, "Echoes a message back as assistant text.", listed.Tools[0].Description)

		call, err := session.CallTool(context.Background(), &mcp.CallToolParams{
			Name:      "echo",
			Arguments: map[string]any{"message": "hello"},
		})
		require.NoError(t, err)
		require.False(t, call.IsError)
		require.Len(t, call.Content, 1)
		assert.Equal(t, "echo: hello", call.Content[0].(*mcp.TextContent).Text)

		models, err := http.Get(server.URL + "/v1/models")
		require.NoError(t, err)
		defer models.Body.Close()
		assert.Equal(t, http.StatusUnauthorized, models.StatusCode)

		req, err := http.NewRequest(http.MethodGet, server.URL+"/v1/models", nil)
		require.NoError(t, err)
		req.Header.Set("Authorization", "Bearer alice-token")
		resp, err := http.DefaultClient.Do(req)
		require.NoError(t, err)
		defer resp.Body.Close()
		require.Equal(t, http.StatusOK, resp.StatusCode)
	})
}

type tokenTransport struct {
	token string
	base  http.RoundTripper
}

func (t tokenTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.Header.Set("Authorization", "Bearer "+t.token)
	return t.base.RoundTrip(req)
}
