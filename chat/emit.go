// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package chat

import "encoding/json/jsontext"

// Text emits a complete assistant text message.
func (emit Emit) Text(text string) error { return emit(Text(text)) }

// Delta emits a streamed text fragment.
func (emit Emit) Delta(text string) error { return emit(TextDelta(text)) }

// Tool emits a completed function call.
func (emit Emit) Tool(callID, name, arguments string) error {
	return emit(Tool(callID, name, arguments))
}

// Activity emits a named JSON activity event.
func (emit Emit) Activity(name string, payload jsontext.Value) error {
	return emit(Activity(name, payload))
}
