// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
package chat

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEventConstructors(t *testing.T) {
	var events []Event
	emit := func(ev Event) error {
		events = append(events, ev)
		return nil
	}

	require.NoError(t, emit(Text("full")))
	require.NoError(t, emit(TextDelta("part")))
	require.NoError(t, emit(TextDone("msg_1")))
	require.NoError(t, emit(Tool("c1", "fn", `{}`)))
	require.NoError(t, emit(ToolStart("c1", "fn")))
	require.NoError(t, emit(ToolDelta("c1", `{"a"`)))
	require.NoError(t, emit(ToolDone("c1")))
	require.NoError(t, emit(MediaItem(ImagePart(InlineMedia("image/png", []byte{1})))))
	require.NoError(t, emit(OutputItem(MessageItem(RoleAssistant, TextPart("item")))))

	assert.Len(t, events, 9)
	assert.Equal(t, EventItem, events[0].Type)
	assert.Equal(t, EventTextDelta, events[1].Type)
	assert.Equal(t, EventTextDone, events[2].Type)
	assert.Equal(t, "msg_1", events[2].ItemID)
	assert.Equal(t, EventItem, events[3].Type)
	assert.Equal(t, EventToolCallStart, events[4].Type)
	assert.Equal(t, EventToolCallDelta, events[5].Type)
	assert.Equal(t, EventToolCallDone, events[6].Type)
	assert.Equal(t, EventItem, events[7].Type)
	assert.Equal(t, ItemMedia, events[7].Item.Type)
	assert.Equal(t, EventItem, events[8].Type)

	reasoning := Reasoning("brief")
	assert.Equal(t, EventItem, reasoning.Type)
	assert.Equal(t, ItemReasoning, reasoning.Item.Type)
}
