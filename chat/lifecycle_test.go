// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package chat

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCapabilitiesNormalize(t *testing.T) {
	zero := Info{}.Normalize()
	assert.True(t, zero.InputModalities.Has(ModalityText))
	assert.True(t, zero.OutputModalities.Has(ModalityText))
	assert.True(t, zero.GenerationControls.Has(ControlMaxOutputTokens))
	assert.True(t, zero.GenerationControls.Has(ControlTemperature))

	withImage := Info{ImageGeneration: true}.Normalize()
	assert.True(t, withImage.OutputModalities.Has(ModalityImage))

	withTools := Info{Tools: true}.Normalize()
	assert.True(t, withTools.GenerationControls.Has(ControlParallelToolCalls))

	withReasoning := Info{ReasoningSummary: true}.Normalize()
	assert.True(t, withReasoning.GenerationControls.Has(ControlReasoning))

	withAudio := Info{OutputModalities: ModalityAudio}.Normalize()
	assert.True(t, withAudio.GenerationControls.Has(ControlAudio))
}

func TestModalityHas(t *testing.T) {
	m := ModalityText | ModalityImage
	assert.True(t, m.Has(ModalityText))
	assert.True(t, m.Has(ModalityImage))
	assert.False(t, m.Has(ModalityAudio))
	assert.True(t, Modality(0).Has(Modality(0)))
}
