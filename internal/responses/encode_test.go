package responses

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
	}, responseMeta{Response: chat.Response{ID: "id", Created: 1, Target: "m"}})
	require.NoError(t, err)
	assert.Equal(t, "response", body.(map[string]any)["object"])
}
