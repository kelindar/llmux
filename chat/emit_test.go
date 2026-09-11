// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
package chat

import (
	"encoding/json/jsontext"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEmitMethods(t *testing.T) {
	var events []Event
	emit := Emit(func(ev Event) error {
		events = append(events, ev)
		return nil
	})

	require.NoError(t, emit.Text("hello"))
	require.NoError(t, emit.Delta("hel"))
	require.NoError(t, emit.Tool("c1", "search", `{}`))
	require.NoError(t, emit.Activity("progress", jsontext.Value(`{"n":1}`)))

	require.Len(t, events, 4)
	assert.Equal(t, Text("hello"), events[0])
	assert.Equal(t, TextDelta("hel"), events[1])
	assert.Equal(t, EventItem, events[2].Type)
	assert.Equal(t, ItemFunctionCall, events[2].Item.Type)
	assert.Equal(t, "c1", events[2].Item.CallID)
	assert.Equal(t, "search", events[2].Item.Name)
	assert.Equal(t, `{}`, events[2].Item.Arguments)
	assert.Equal(t, EventActivity, events[3].Type)
	assert.Equal(t, "progress", events[3].Name)
}
