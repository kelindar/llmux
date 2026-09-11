package mcp

import (
	"strings"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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
