// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package agui

import (
	"encoding/json/jsontext"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/execution"
	internalprotocol "github.com/kelindar/llmux/internal/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateEvent(t *testing.T) {
	adapter := Adapter{}
	require.NoError(t, adapter.ValidateEvent(chat.Text("hello")))
	require.NoError(t, adapter.ValidateEvent(chat.Activity("progress", jsontext.Value(`{"step":1}`))))
	assert.Error(t, adapter.ValidateEvent(chat.Activity("", jsontext.Value(`{}`))))
	assert.Error(t, adapter.ValidateEvent(chat.Activity("progress", jsontext.Value(`[]`))))
	assert.Error(t, adapter.ValidateEvent(chat.Tool("call-1", "search", `{}`)))
	assert.Error(t, adapter.ValidateEvent(chat.OutputItem(chat.Item{Type: chat.ItemMessage, Role: chat.RoleAssistant, Content: []chat.Part{chat.ImagePart(chat.AssetMedia("image/png", "asset-1"))}})))
	_, err := adapter.Response(chat.Request{}, execution.Result{}, internalprotocol.Meta{})
	assert.Error(t, err)
}
