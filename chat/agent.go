// Package chat contains the protocol-neutral agent, request, event, and
// capability types shared by llmux and application-owned agents.
package chat

import (
	"context"
	"errors"
)

// Agent is the application-owned execution seam.
type Agent interface {
	// Run executes once for an accepted HTTP request and streams events through emit.
	// It must stop when emit returns an error.
	Run(context.Context, *Request, Emit) (Outcome, error)
}

// AgentFunc adapts a function to Agent.
type AgentFunc func(context.Context, *Request, Emit) (Outcome, error)

// Run calls f, or returns an error if f is nil.
func (f AgentFunc) Run(ctx context.Context, req *Request, emit Emit) (Outcome, error) {
	if f == nil {
		return Outcome{}, errors.New("llmux: nil agent function")
	}
	return f(ctx, req, emit)
}

// Resolver selects an agent and its declared capabilities for a target name.
// Applications with resolver structs should pass a method value.
type Resolver func(context.Context, string) (Agent, Capabilities, error)

// Emit is the serial event function passed to Agent.Run.
type Emit func(Event) error

var (
	// ErrConcurrentEmit is returned when Emit is invoked concurrently.
	ErrConcurrentEmit = errors.New("llmux: concurrent Emit calls are not supported")
	// ErrEmitClosed is returned when Emit is called after the stream is closed.
	ErrEmitClosed = errors.New("llmux: emission is closed")
	// ErrDelivery is returned when a client write fails. With durable
	// acceptance, the handler stops delivery but does not cancel execution.
	ErrDelivery = errors.New("llmux: client delivery failed")
)

// AssetResolver is opt-in. It receives the original media descriptor and a
// hard byte ceiling. It must authorize the reference using the request
// context and return inline data or another bounded representation.
// Applications with resolver structs should pass a method value.
type AssetResolver func(context.Context, Media, int64) (Media, error)
