package llmux

import (
	"testing"

	"github.com/kelindar/llmux/audio"
	"github.com/stretchr/testify/assert"
)

func TestAudioTypeAliases(t *testing.T) {
	var (
		_ Transcriber          = audio.TranscriberFunc(nil)
		_ Speaker              = audio.SpeakerFunc(nil)
		_ TranscriptionRequest = audio.TranscriptionRequest{}
		_ Transcription        = audio.Transcription{}
		_ TranscriptSegment    = audio.TranscriptSegment{}
		_ SpeechRequest        = audio.SpeechRequest{}
		_ Speech               = audio.Speech{}
	)
	assert.True(t, true)
}
