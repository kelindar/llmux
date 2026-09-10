package anthropic

import (
	"net/http/httptest"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStreamStart(t *testing.T) {
	rec := httptest.NewRecorder()
	meta := responseMeta{ID: "m1", Model: "m"}
	stream := Adapter{}.Stream(rec, parsedRequest{}, &meta, chat.DefaultLimits())
	require.NoError(t, stream.Event(chat.Text("hi")))
	require.NoError(t, stream.Complete(chat.Outcome{Status: chat.StatusCompleted}, []chat.Item{chat.MessageItem(chat.RoleAssistant, chat.TextPart("hi"))}))
	assert.Contains(t, rec.Body.String(), "message_start")
}
