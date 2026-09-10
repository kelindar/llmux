package chat

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOutcomeValidate(t *testing.T) {
	cases := map[string]struct {
		outcome Outcome
		wantErr string
	}{
		"emptyOk": {},
		"completedOk": {
			outcome: Outcome{Status: StatusCompleted, StopReason: StopStop},
		},
		"usageOk": {
			outcome: Outcome{
				Status: StatusCompleted,
				Usage:  &Usage{Input: 1, Output: 2, Total: 3},
			},
		},
		"invalidStatus": {
			outcome: Outcome{Status: "bad"},
			wantErr: "invalid outcome status",
		},
		"inProgress": {
			outcome: Outcome{Status: StatusInProgress},
			wantErr: "cannot be in_progress",
		},
		"negativeUsage": {
			outcome: Outcome{Status: StatusCompleted, Usage: &Usage{Input: -1}},
			wantErr: "negative values",
		},
		"invalidStopReason": {
			outcome: Outcome{Status: StatusCompleted, StopReason: "nope"},
			wantErr: "invalid outcome stop reason",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			err := tc.outcome.Validate()
			if tc.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

func TestAPIError(t *testing.T) {
	var nilErr *Error
	assert.Equal(t, "", nilErr.Error())

	withMessage := NewAPIError(400, "invalid_request_error", "invalid_request", "field", "bad field")
	assert.Equal(t, "bad field", withMessage.Error())
	assert.Equal(t, 400, withMessage.Status)

	wrapped := &Error{Err: errors.New("underlying")}
	assert.Equal(t, "underlying", wrapped.Error())
	assert.Equal(t, errors.New("underlying"), wrapped.Unwrap())

	defaultMsg := &Error{}
	assert.Equal(t, "llmux API error", defaultMsg.Error())

	invalid := Invalid("p", "msg")
	assert.Equal(t, 400, invalid.Status)
	assert.Equal(t, "invalid_request", invalid.Code)

	unsupported := Unsupported("p", "msg")
	assert.Equal(t, "unsupported", unsupported.Code)
}

func TestLimitsNormalize(t *testing.T) {
	def := DefaultLimits()
	assert.Equal(t, int64(8<<20), def.MaxRequestBytes)
	assert.Equal(t, int64(16<<20), def.MaxMediaBytes)
	assert.Equal(t, 32, def.MaxAssets)

	norm := Limits{}.Normalize()
	assert.Equal(t, def, norm)

	partial := Limits{MaxAssets: 10}.Normalize()
	assert.Equal(t, 10, partial.MaxAssets)
	assert.Equal(t, def.MaxRequestBytes, partial.MaxRequestBytes)
}
