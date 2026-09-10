package identity

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNew(t *testing.T) {
	const prefix = "fc_"
	seen := make(map[string]struct{}, 100)

	for range 100 {
		id := New(prefix)
		require.True(t, strings.HasPrefix(id, prefix), "id %q missing prefix %q", id, prefix)
		suffix := strings.TrimPrefix(id, prefix)
		require.NotEmpty(t, suffix)
		_, dup := seen[id]
		assert.False(t, dup, "duplicate id %q", id)
		seen[id] = struct{}{}
	}

	other := New("msg_")
	assert.True(t, strings.HasPrefix(other, "msg_"))
	assert.NotContains(t, seen, other)
}
