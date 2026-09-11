// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
package chat

import (
	"encoding/json/jsontext"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolValidate(t *testing.T) {
	cases := map[string]struct {
		tool    FunctionTool
		wantErr string
	}{
		"minimalOk": {tool: FunctionTool{Name: "search"}},
		"paramsOk":  {tool: FunctionTool{Name: "search", Parameters: jsontext.Value(`{"type":"object"}`)}},
		"missingName": {
			tool:    FunctionTool{},
			wantErr: "name is required",
		},
		"badParams": {
			tool:    FunctionTool{Name: "search", Parameters: jsontext.Value(`[]`)},
			wantErr: "must be a JSON object",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := tc.tool.Validate()
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestToolChoiceValidate(t *testing.T) {
	modes := []string{"auto", "none", "required", "any"}
	for _, mode := range modes {
		t.Run(mode, func(t *testing.T) {
			require.NoError(t, ToolChoice{Mode: mode}.Validate())
		})
	}

	cases := map[string]struct {
		choice  ToolChoice
		wantErr string
	}{
		"autoWithName": {
			choice:  ToolChoice{Mode: "auto", Name: "x"},
			wantErr: "cannot have a name",
		},
		"functionOk": {
			choice: ToolChoice{Mode: "function", Name: "search"},
		},
		"functionMissingName": {
			choice:  ToolChoice{Mode: "function", Name: "  "},
			wantErr: "requires a name",
		},
		"unsupportedMode": {
			choice:  ToolChoice{Mode: "weird"},
			wantErr: "unsupported tool choice mode",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := tc.choice.Validate()
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestOutputFormatValidate(t *testing.T) {
	cases := map[string]struct {
		out     OutputFormat
		wantErr string
	}{
		"ok": {
			out: OutputFormat{Kind: FormatJSONSchema, Name: "out", Schema: jsontext.Value(`{"type":"object"}`)},
		},
		"missingName": {
			out:     OutputFormat{Kind: FormatJSONSchema, Schema: jsontext.Value(`{}`)},
			wantErr: "requires a name",
		},
		"invalidSchema": {
			out:     OutputFormat{Kind: FormatJSONSchema, Name: "out", Schema: jsontext.Value(`{`)},
			wantErr: "requires a name",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := tc.out.Validate()
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestReasoningValidate(t *testing.T) {
	assert.NoError(t, ReasoningControl{Effort: "high"}.Validate())
	assert.NoError(t, ReasoningControl{Summary: true}.Validate())
	err := ReasoningControl{}.Validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "effort or summary")
}

func TestAudioControlsValidate(t *testing.T) {
	formats := []string{"wav", "aac", "mp3", "flac", "opus", "pcm16", "WAV"}
	for _, format := range formats {
		t.Run(format, func(t *testing.T) {
			require.NoError(t, AudioControls{Voice: "alloy", Format: format}.Validate())
		})
	}

	cases := map[string]struct {
		ctrl    AudioControls
		wantErr string
	}{
		"missingVoice": {
			ctrl:    AudioControls{Format: "mp3"},
			wantErr: "voice is required",
		},
		"badFormat": {
			ctrl:    AudioControls{Voice: "alloy", Format: "wma"},
			wantErr: "unsupported Chat Completions audio format",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := tc.ctrl.Validate()
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestValidRoleStatus(t *testing.T) {
	assert.True(t, ValidRole(RoleUser))
	assert.False(t, ValidRole(Role("guest")))
	assert.True(t, validStatus(StatusCompleted))
	assert.False(t, validStatus(Status("pending")))
}

func TestGenerationControlHas(t *testing.T) {
	c := ControlMaxOutputTokens | ControlStop
	assert.True(t, c.Has(ControlStop))
	assert.False(t, c.Has(ControlAudio))
}
