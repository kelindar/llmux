// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
package anthropic

import (
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/execution"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEncodeResponse(t *testing.T) {
	body, err := Adapter{}.Response(chat.Request{Target: "m"}, execution.Result{
		Items:   []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("hi"))},
		Outcome: chat.Outcome{Status: chat.StatusCompleted},
	}, responseMeta{Response: chat.Response{ID: "id", Target: "m"}})
	require.NoError(t, err)
	assert.Equal(t, "message", body.(messageResponse).Type)
}
