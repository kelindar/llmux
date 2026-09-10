package chat

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestModelFields(t *testing.T) {
	model := Model{ID: "agent/basic", Created: 1, OwnedBy: "test"}
	assert.Equal(t, "agent/basic", model.ID)
	assert.Equal(t, int64(1), model.Created)
	assert.Equal(t, "test", model.OwnedBy)
}
