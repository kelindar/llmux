// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package fixtures

import (
	"net/http"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHarness(t *testing.T) {
	server := startServer(echoAgent("ping"), chat.Info{})
	defer server.Close()

	resp, err := http.Get(server.URL + "/v1/models")
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)
}
