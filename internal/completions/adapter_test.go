// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package completions

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAdapter(t *testing.T) {
	assert.NotNil(t, NewAdapter())
}
