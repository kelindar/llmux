package chat

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCapabilitiesNormalize(t *testing.T) {
	zero := Capabilities{}.Normalize()
	assert.True(t, zero.InputModalities.Has(ModalityText))
	assert.True(t, zero.OutputModalities.Has(ModalityText))
	assert.True(t, zero.GenerationControls.Has(ControlMaxOutputTokens))
	assert.True(t, zero.GenerationControls.Has(ControlTemperature))

	withImage := Capabilities{ImageGeneration: true}.Normalize()
	assert.True(t, withImage.OutputModalities.Has(ModalityImage))

	withTools := Capabilities{Tools: true}.Normalize()
	assert.True(t, withTools.GenerationControls.Has(ControlParallelToolCalls))

	withReasoning := Capabilities{ReasoningSummary: true}.Normalize()
	assert.True(t, withReasoning.GenerationControls.Has(ControlReasoning))

	withAudio := Capabilities{OutputModalities: ModalityAudio}.Normalize()
	assert.True(t, withAudio.GenerationControls.Has(ControlAudio))
}

func TestModalityHas(t *testing.T) {
	m := ModalityText | ModalityImage
	assert.True(t, m.Has(ModalityText))
	assert.True(t, m.Has(ModalityImage))
	assert.False(t, m.Has(ModalityAudio))
	assert.True(t, Modality(0).Has(Modality(0)))
}
