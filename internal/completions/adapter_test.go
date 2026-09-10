package completions

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAdapter(t *testing.T) {
	assert.NotNil(t, NewAdapter())
}
