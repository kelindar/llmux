package chat

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAgentFunc(t *testing.T) {
	var nilAgent AgentFunc
	_, err := nilAgent.Run(context.Background(), &Request{}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nil agent function")

	var captured Event
	agent := AgentFunc(func(_ context.Context, _ *Request, emit Emit) (Outcome, error) {
		require.NoError(t, emit(Text("hi")))
		return Outcome{Status: StatusCompleted}, nil
	})
	out, err := agent.Run(context.Background(), &Request{}, func(ev Event) error {
		captured = ev
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, StatusCompleted, out.Status)
	assert.Equal(t, EventItem, captured.Type)
}

type namedResolver struct{}

func (namedResolver) Resolve(_ context.Context, _ string) (Agent, Capabilities, error) {
	return AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
		return Outcome{}, nil
	}), Capabilities{Tools: true}, nil
}

func TestResolver(t *testing.T) {
	var nilResolver Resolver
	assert.Nil(t, nilResolver)

	resolver := Resolver(func(_ context.Context, target string) (Agent, Capabilities, error) {
		assert.Equal(t, "my-agent", target)
		return AgentFunc(func(context.Context, *Request, Emit) (Outcome, error) {
			return Outcome{}, nil
		}), Capabilities{Tools: true}, nil
	})
	gotAgent, gotCaps, err := resolver(context.Background(), "my-agent")
	require.NoError(t, err)
	require.NotNil(t, gotAgent)
	assert.Equal(t, Capabilities{Tools: true}, gotCaps)

	method := Resolver(namedResolver{}.Resolve)
	gotAgent, gotCaps, err = method(context.Background(), "x")
	require.NoError(t, err)
	require.NotNil(t, gotAgent)
	assert.True(t, gotCaps.Tools)
}

type namedAssets struct{}

func (namedAssets) Resolve(_ context.Context, media Media, max int64) (Media, error) {
	if media.Ref != "ref-1" || max != 100 {
		return Media{}, errors.New("unexpected media")
	}
	return InlineMedia("text/plain", []byte("data")), nil
}

func TestAssetResolver(t *testing.T) {
	var nilResolver AssetResolver
	assert.Nil(t, nilResolver)

	resolver := AssetResolver(func(_ context.Context, media Media, max int64) (Media, error) {
		assert.Equal(t, "ref-1", media.Ref)
		assert.Equal(t, int64(100), max)
		return InlineMedia("text/plain", []byte("data")), nil
	})
	got, err := resolver(context.Background(), AssetMedia("text/plain", "ref-1"), 100)
	require.NoError(t, err)
	assert.Equal(t, []byte("data"), got.Data)

	method := AssetResolver(namedAssets{}.Resolve)
	got, err = method(context.Background(), AssetMedia("text/plain", "ref-1"), 100)
	require.NoError(t, err)
	assert.Equal(t, []byte("data"), got.Data)
}
