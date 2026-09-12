// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package parse_test

import (
	"encoding/json/jsontext"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/parse"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResponses(t *testing.T) {
	tests := map[string]struct {
		body string
		role chat.Role
		text string
	}{
		"string content": {
			body: `{"model":"agent","input":[{"role":"user","content":"Help me"}]}`,
			role: chat.RoleUser,
			text: "Help me",
		},
		"input_text parts": {
			body: `{"model":"agent","input":[{"role":"user","content":[{"type":"input_text","text":"Help me"}]}]}`,
			role: chat.RoleUser,
			text: "Help me",
		},
		"preserves system order": {
			body: `{"model":"agent","instructions":"Be brief","input":[{"role":"system","content":"ctx"},{"role":"user","content":"hi"}]}`,
			role: chat.RoleSystem,
			text: "ctx",
		},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			parsed, err := parse.Responses(jsontext.Value(tc.body))
			require.NoError(t, err)
			require.NotEmpty(t, parsed.Request.Input)
			assert.Equal(t, tc.role, parsed.Request.Input[0].Role)
			require.NotEmpty(t, parsed.Request.Input[0].Content)
			assert.Equal(t, chat.PartText, parsed.Request.Input[0].Content[0].Type)
			assert.Equal(t, tc.text, parsed.Request.Input[0].Content[0].Text)
		})
	}

	t.Run("rejects non-object", func(t *testing.T) {
		_, err := parse.Responses(jsontext.Value(`[]`))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "JSON object")
	})
}
