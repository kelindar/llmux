// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
package chat

import (
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"strings"
)

// Modality is a bit set used for input and output declarations.
type Modality uint8

const (
	ModalityText Modality = 1 << iota
	ModalityImage
	ModalityAudio
	ModalityFile
)

// Has reports whether m includes all bits in other.
func (m Modality) Has(other Modality) bool { return m&other == other }

// FormatKind identifies text output formatting for a request.
type FormatKind string

const (
	FormatText       FormatKind = "text"
	FormatJSONObject FormatKind = "json_object"
	FormatJSONSchema FormatKind = "json_schema"
)

// OutputFormat is text formatting. Zero value = plain text.
type OutputFormat struct {
	Kind        FormatKind
	Name        string
	Description string
	Schema      jsontext.Value
	Strict      bool
}

// IsStructured reports whether the format requests JSON object output.
func (f OutputFormat) IsStructured() bool {
	return f.Kind == FormatJSONObject || f.Kind == FormatJSONSchema
}

// Validate checks a declared output format.
func (f OutputFormat) Validate() error {
	switch f.Kind {
	case "", FormatText:
		return nil
	case FormatJSONObject:
		return nil
	case FormatJSONSchema:
		if f.Name == "" || len(f.Schema) == 0 || !f.Schema.IsValid() {
			return errors.New("structured output requires a name and valid schema")
		}
		return nil
	default:
		return fmt.Errorf("unknown output format kind %q", f.Kind)
	}
}

// OutputSpec declares expected response modalities and formatting.
type OutputSpec struct {
	Modalities Modality
	Format     OutputFormat
}

// FunctionTool declares a function tool available to the agent.
type FunctionTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  jsontext.Value `json:"parameters,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
}

// Validate checks a function tool declaration.
func (t FunctionTool) Validate() error {
	switch {
	case t.Name == "":
		return errors.New("tool name is required")
	case len(t.Parameters) != 0:
		var parameters map[string]jsontext.Value
		if err := json.Unmarshal(t.Parameters, &parameters); err != nil || parameters == nil {
			return fmt.Errorf("tool %q parameters must be a JSON object", t.Name)
		}
	}
	return nil
}

// ToolChoice selects how the model may use declared tools.
type ToolChoice struct {
	Mode string
	Name string
}

// Validate checks a tool-choice declaration.
func (t ToolChoice) Validate() error {
	switch t.Mode {
	case "auto", "none", "required", "any":
		if t.Name != "" {
			return errors.New("tool choice mode cannot have a name")
		}
	case "function":
		if strings.TrimSpace(t.Name) == "" {
			return errors.New("function tool choice requires a name")
		}
	default:
		return fmt.Errorf("unsupported tool choice mode %q", t.Mode)
	}
	return nil
}

// ReasoningControl configures optional reasoning effort and summary output.
type ReasoningControl struct {
	Effort  string
	Summary bool
}

// Validate checks reasoning controls.
func (r ReasoningControl) Validate() error {
	if strings.TrimSpace(r.Effort) == "" && !r.Summary {
		return errors.New("reasoning requires effort or summary")
	}
	return nil
}

// AudioControls configures Chat Completions audio output.
type AudioControls struct {
	Voice  string
	Format string
}

// Validate checks Chat Completions audio controls.
func (a AudioControls) Validate() error {
	if strings.TrimSpace(a.Voice) == "" {
		return errors.New("audio voice is required")
	}
	switch strings.ToLower(a.Format) {
	case "wav", "aac", "mp3", "flac", "opus", "pcm16":
		return nil
	default:
		return fmt.Errorf("unsupported Chat Completions audio format %q", a.Format)
	}
}

// Controls holds generation settings for a request.
type Controls struct {
	MaxOutputTokens  *int                      // Maximum tokens to generate.
	Temperature      *float64                  // Sampling temperature.
	TopP             *float64                  // Nucleus sampling threshold.
	Stop             []string                  // Stop sequences.
	Tools            []FunctionTool            // Function tools available to the model.
	ToolChoice       *ToolChoice               // Tool selection policy.
	ParallelToolCall *bool                     // Whether parallel tool calls are allowed.
	Reasoning        *ReasoningControl         // Reasoning effort and summary controls.
	Audio            *AudioControls            // Chat Completions audio output controls.
	ImageGeneration  bool                      // Whether image generation is requested.
	Extensions       map[string]jsontext.Value // Application extension payloads keyed by name.
}

// Request is the protocol-neutral input passed to Agent.Run.
// It contains only execution fields. Persistence, continuation, and
// idempotency live on TurnRequest for Lifecycle.Accept.
type Request struct {
	Target       string     // Agent target name selected by the resolver.
	Instructions string     // System or developer instructions for the run.
	Input        []Item     // Effective conversation for Agent.Run (history+turn).
	Controls     Controls   // Generation settings.
	Output       OutputSpec // Declared output modalities and format.
}

// Limits bound request, media, event, and accumulated response memory.
type Limits struct {
	MaxRequestBytes   int64
	MaxMediaBytes     int64
	MaxAssets         int
	MaxOutputBytes    int64
	MaxEventBytes     int64
	MaxMultipartBytes int64
}

// DefaultLimits returns the conservative limits used by a Handler when an option leaves a field at zero.
func DefaultLimits() Limits {
	return Limits{
		MaxRequestBytes:   8 << 20,
		MaxMediaBytes:     16 << 20,
		MaxAssets:         32,
		MaxOutputBytes:    8 << 20,
		MaxEventBytes:     1 << 20,
		MaxMultipartBytes: 32 << 20,
	}
}

// Normalize replaces non-positive fields with their documented defaults.
func (l Limits) Normalize() Limits {
	d := DefaultLimits()
	if l.MaxRequestBytes <= 0 {
		l.MaxRequestBytes = d.MaxRequestBytes
	}
	if l.MaxMediaBytes <= 0 {
		l.MaxMediaBytes = d.MaxMediaBytes
	}
	if l.MaxAssets <= 0 {
		l.MaxAssets = d.MaxAssets
	}
	if l.MaxOutputBytes <= 0 {
		l.MaxOutputBytes = d.MaxOutputBytes
	}
	if l.MaxEventBytes <= 0 {
		l.MaxEventBytes = d.MaxEventBytes
	}
	if l.MaxMultipartBytes <= 0 {
		l.MaxMultipartBytes = d.MaxMultipartBytes
	}
	return l
}

// Info describes the selected agent for clients: a human-readable
// description plus what the agent can actually execute. It is also the
// unified catalog entry: applications return caller-visible targets mapped
// to Info, and llmux projects that one catalog into GET /models and MCP.
type Info struct {
	InputModalities    Modality
	OutputModalities   Modality
	GenerationControls GenerationControl
	Extensions         map[string]bool
	Tools              bool
	ClientTools        bool
	StructuredOutput   bool
	ReasoningSummary   bool
	ImageGeneration    bool
	Continuation       bool

	// Description is a human-readable summary of the agent. Resolver
	// implementations should set it; llmux surfaces it as the MCP tool
	// description when the agent is exposed through /mcp.
	Description string

	// Tool is the stable public MCP tool name for this agent. Empty means
	// the agent is not exposed through MCP; nonempty exposes it, with the
	// name owned by the application (1-128 characters from [A-Za-z0-9._-],
	// unique per catalog). It is never derived from the resolver target.
	Tool string

	// Created is the Unix creation time in seconds shown in the model
	// catalog. Zero is rendered as the zero time value.
	Created int64
	// OwnedBy is the owner label shown in the model catalog.
	OwnedBy string
}

// GenerationControl identifies a generation control understood by an agent.
type GenerationControl uint16

const (
	ControlMaxOutputTokens GenerationControl = 1 << iota
	ControlTemperature
	ControlTopP
	ControlStop
	ControlParallelToolCalls
	ControlReasoning
	ControlAudio
)

// Has reports whether c includes all bits in other.
func (c GenerationControl) Has(other GenerationControl) bool { return c&other == other }

// Normalize fills the defaults implied by a zero Info value.
func (c Info) Normalize() Info {
	if c.InputModalities == 0 {
		c.InputModalities = ModalityText
	}
	if c.OutputModalities == 0 {
		c.OutputModalities = ModalityText
	}
	if c.ImageGeneration {
		c.OutputModalities |= ModalityImage
	}
	if c.GenerationControls == 0 {
		c.GenerationControls = ControlMaxOutputTokens | ControlTemperature | ControlTopP | ControlStop
	}
	if c.Tools {
		c.GenerationControls |= ControlParallelToolCalls
	}
	if c.ReasoningSummary {
		c.GenerationControls |= ControlReasoning
	}
	if c.OutputModalities.Has(ModalityAudio) {
		c.GenerationControls |= ControlAudio
	}
	return c
}
