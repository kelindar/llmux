// Package llmux exposes application-owned agent logic through standard HTTP
// AI protocols.
package llmux

import (
	"context"
	"encoding/json/jsontext"
	"time"

	"github.com/kelindar/llmux/audio"
	"github.com/kelindar/llmux/contract"
	"github.com/kelindar/llmux/internal/identity"
)

// Agent is the application-owned execution seam passed to Handler. Run is
// called once for each accepted HTTP request and must stop when emit returns
// an error.
type Agent = contract.Agent

// AgentFunc adapts a function to Agent.
type AgentFunc = contract.AgentFunc

// Resolver selects an agent and its declared capabilities for a target name.
type Resolver = contract.Resolver

// ResolverFunc adapts a function to Resolver.
type ResolverFunc = contract.ResolverFunc

// Role is the speaker role on a message item.
type Role = contract.Role

// ItemType identifies the tagged shape of an Item.
type ItemType = contract.ItemType

// PartType identifies the tagged shape of a Part.
type PartType = contract.PartType

// Status is the lifecycle state of an item or outcome.
type Status = contract.Status

// StopReason explains why generation ended.
type StopReason = contract.StopReason

// Modality is a bit set used for input and output declarations.
type Modality = contract.Modality

// Media identifies bytes, a remote URL, or an application-owned asset.
type Media = contract.Media

// Part is a typed content part within an item.
type Part = contract.Part

// Item is the tagged union used for request input and agent output.
type Item = contract.Item

// Tool is a function tool declaration on a request.
type Tool = contract.Tool

// ToolChoice selects how tools may be invoked.
type ToolChoice = contract.ToolChoice

// StructuredOutput requests schema-constrained text output.
type StructuredOutput = contract.StructuredOutput

// Reasoning carries reasoning-effort and summary controls.
type Reasoning = contract.Reasoning

// AudioControls carries Chat Completions audio output settings.
type AudioControls = contract.AudioControls

// Controls holds generation, tool, and protocol extension settings.
type Controls = contract.Controls

// OutputSpec describes the modalities and format requested from an agent.
type OutputSpec = contract.OutputSpec

// Request is the normalized input passed to Agent.Run.
type Request = contract.Request

// Limits bound request, media, event, and accumulated response memory.
type Limits = contract.Limits

// Usage reports token accounting for a completed run.
type Usage = contract.Usage

// Outcome is the terminal status returned from Agent.Run.
type Outcome = contract.Outcome

// Capabilities describes what the selected agent can execute.
type Capabilities = contract.Capabilities

// GenerationControl identifies a generation control understood by an agent.
type GenerationControl = contract.GenerationControl

// EventType identifies the kind of streaming or final output event.
type EventType = contract.EventType

// Event is the output unit emitted during Agent.Run.
type Event = contract.Event

// Emit is the serial event callback passed to Agent.Run.
type Emit = contract.Emit

// APIError is a sanitized protocol error returned to HTTP clients.
type APIError = contract.APIError

// AssetResolver resolves application-owned media references to inline bytes.
type AssetResolver = contract.AssetResolver

// AssetResolverFunc adapts a function to AssetResolver.
type AssetResolverFunc = contract.AssetResolverFunc

// ContinuationStore loads prior-turn history for previous_response_id.
type ContinuationStore = contract.ContinuationStore

// Lifecycle is the application-owned response identity and persistence seam.
type Lifecycle = contract.Lifecycle

// TurnRequest is the input to Lifecycle.Accept.
type TurnRequest = contract.TurnRequest

// Acceptance is the per-request result of Lifecycle.Accept.
type Acceptance = contract.Acceptance

// ResponseState is the client-visible response payload for creation, replay,
// finalization, and retrieval.
type ResponseState = contract.ResponseState

// TurnResult is the input to Lifecycle.Finalize.
type TurnResult = contract.TurnResult

// CleanupContext returns a bounded finalization context that keeps parent
// values but not parent cancellation.
func CleanupContext(parent context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	return contract.CleanupContext(parent, timeout)
}

// CloneMetadata returns a shallow copy of metadata.
func CloneMetadata(src map[string]string) map[string]string {
	return contract.CloneMetadata(src)
}

// Transcriber is the optional service behind POST /v1/audio/transcriptions.
type Transcriber = audio.Transcriber

// TranscriberFunc adapts a function to Transcriber.
type TranscriberFunc = audio.TranscriberFunc

// TranscriptionRequest is the normalized multipart input for a Transcriber.
type TranscriptionRequest = audio.TranscriptionRequest

// Transcription is the normalized result of a transcription service.
type Transcription = audio.Transcription

// TranscriptSegment is a timed portion of a verbose transcription.
type TranscriptSegment = audio.TranscriptSegment

// Speaker is the optional service behind POST /v1/audio/speech.
type Speaker = audio.Speaker

// SpeakerFunc adapts a function to Speaker.
type SpeakerFunc = audio.SpeakerFunc

// SpeechRequest is the normalized JSON input for a Speaker.
type SpeechRequest = audio.SpeechRequest

// Speech is the complete audio returned by a Speaker.
type Speech = audio.Speech

const (
	// RoleUser is the end-user message role.
	RoleUser = contract.RoleUser
	// RoleAssistant is the model message role.
	RoleAssistant = contract.RoleAssistant
	// RoleSystem is the system instruction role.
	RoleSystem = contract.RoleSystem
	// RoleDeveloper is the developer instruction role.
	RoleDeveloper = contract.RoleDeveloper
	// RoleTool is the tool result role.
	RoleTool = contract.RoleTool

	// ItemMessage is a conversational message item.
	ItemMessage = contract.ItemMessage
	// ItemFunctionCall is a client-visible tool call item.
	ItemFunctionCall = contract.ItemFunctionCall
	// ItemFunctionCallOutput is a tool result item.
	ItemFunctionCallOutput = contract.ItemFunctionCallOutput
	// ItemReasoning is a reasoning summary item.
	ItemReasoning = contract.ItemReasoning
	// ItemMedia is a standalone media output item.
	ItemMedia = contract.ItemMedia
	// ItemExtension is a protocol extension payload item.
	ItemExtension = contract.ItemExtension

	// PartText is a plain-text content part.
	PartText = contract.PartText
	// PartImage is an image media part.
	PartImage = contract.PartImage
	// PartAudio is an audio media part.
	PartAudio = contract.PartAudio
	// PartFile is a file media part.
	PartFile = contract.PartFile
	// PartJSON is a JSON content part.
	PartJSON = contract.PartJSON
	// PartReasoningSummary is a reasoning summary text part.
	PartReasoningSummary = contract.PartReasoningSummary

	// StatusInProgress means generation is still running.
	StatusInProgress = contract.StatusInProgress
	// StatusCompleted means generation finished successfully.
	StatusCompleted = contract.StatusCompleted
	// StatusIncomplete means generation ended before completion.
	StatusIncomplete = contract.StatusIncomplete
	// StatusFailed means generation failed.
	StatusFailed = contract.StatusFailed
	// StatusCancelled means generation was cancelled.
	StatusCancelled = contract.StatusCancelled

	// StopStop means generation ended normally.
	StopStop = contract.StopStop
	// StopToolCall means generation paused for a tool call.
	StopToolCall = contract.StopToolCall
	// StopLength means generation hit a length limit.
	StopLength = contract.StopLength
	// StopError means generation stopped due to an error.
	StopError = contract.StopError
	// StopCancelled means generation was cancelled.
	StopCancelled = contract.StopCancelled

	// ModalityText is the text input or output modality bit.
	ModalityText = contract.ModalityText
	// ModalityImage is the image input or output modality bit.
	ModalityImage = contract.ModalityImage
	// ModalityAudio is the audio input or output modality bit.
	ModalityAudio = contract.ModalityAudio
	// ModalityFile is the file input or output modality bit.
	ModalityFile = contract.ModalityFile

	// ControlMaxOutputTokens is the max-output-tokens generation control bit.
	ControlMaxOutputTokens = contract.ControlMaxOutputTokens
	// ControlTemperature is the temperature generation control bit.
	ControlTemperature = contract.ControlTemperature
	// ControlTopP is the top_p generation control bit.
	ControlTopP = contract.ControlTopP
	// ControlStop is the stop-sequence generation control bit.
	ControlStop = contract.ControlStop
	// ControlParallelToolCalls is the parallel tool call control bit.
	ControlParallelToolCalls = contract.ControlParallelToolCalls
	// ControlReasoning is the reasoning control bit.
	ControlReasoning = contract.ControlReasoning
	// ControlAudio is the audio output control bit.
	ControlAudio = contract.ControlAudio

	// EventMessage is a complete assistant message event.
	EventMessage = contract.EventMessage
	// EventTextDelta is an incremental text delta event.
	EventTextDelta = contract.EventTextDelta
	// EventTextDone marks the end of a text stream.
	EventTextDone = contract.EventTextDone
	// EventToolCall is a complete tool call event.
	EventToolCall = contract.EventToolCall
	// EventToolCallStart begins a streamed tool call.
	EventToolCallStart = contract.EventToolCallStart
	// EventToolCallDelta carries streamed tool call argument bytes.
	EventToolCallDelta = contract.EventToolCallDelta
	// EventToolCallDone completes a streamed tool call.
	EventToolCallDone = contract.EventToolCallDone
	// EventMedia is a standalone media output event.
	EventMedia = contract.EventMedia
	// EventReasoning is a reasoning summary event.
	EventReasoning = contract.EventReasoning
	// EventItem is a structured output item event.
	EventItem = contract.EventItem
	// EventActivity is application-specific streaming activity (Responses only).
	EventActivity = contract.EventActivity
)

var (
	// ErrConcurrentEmit is returned when emit is called concurrently.
	ErrConcurrentEmit = contract.ErrConcurrentEmit
	// ErrEmitClosed is returned when emit is called after the handler closes it.
	ErrEmitClosed = contract.ErrEmitClosed
	// ErrDelivery is returned when a client write fails under durable acceptance.
	ErrDelivery = contract.ErrDelivery
)

// InlineMedia builds media from inline bytes.
func InlineMedia(mime string, data []byte) Media { return contract.InlineMedia(mime, data) }

// RemoteMedia builds media from a remote URL.
func RemoteMedia(mime, url string) Media { return contract.RemoteMedia(mime, url) }

// AssetMedia builds media from an application-owned asset reference.
func AssetMedia(mime, ref string) Media { return contract.AssetMedia(mime, ref) }

// TextPart builds a text content part.
func TextPart(text string) Part { return contract.TextPart(text) }

// ImagePart builds an image content part.
func ImagePart(media Media) Part { return contract.ImagePart(media) }

// AudioPart builds an audio content part.
func AudioPart(media Media) Part { return contract.AudioPart(media) }

// FilePart builds a file content part.
func FilePart(media Media) Part { return contract.FilePart(media) }

// JSONPart builds a JSON content part.
func JSONPart(data jsontext.Value) Part { return contract.JSONPart(data) }

// SummaryPart builds a reasoning summary content part.
func SummaryPart(text string) Part { return contract.SummaryPart(text) }

// MessageItem builds a message item with the given role and content.
func MessageItem(role Role, content ...Part) Item {
	return contract.MessageItem(role, content...)
}

// FunctionCallItem builds a completed function call item.
func FunctionCallItem(callID, name, arguments string) Item {
	return contract.FunctionCallItem(callID, name, arguments)
}

// FunctionCallOutputItem builds a function call output item.
func FunctionCallOutputItem(callID string, output ...Part) Item {
	return contract.FunctionCallOutputItem(callID, output...)
}

// CompleteText builds a final assistant message event.
func CompleteText(text string) Event { return contract.CompleteText(text) }

// TextDelta builds an incremental text delta event.
func TextDelta(text string) Event { return contract.TextDelta(text) }

// ToolCall builds a complete tool call event.
func ToolCall(callID, name, arguments string) Event {
	return contract.ToolCall(callID, name, arguments)
}

// ToolCallStart builds the first event of a streamed tool call.
func ToolCallStart(callID, name string) Event { return contract.ToolCallStart(callID, name) }

// ToolCallDelta builds a streamed tool call argument delta event.
func ToolCallDelta(callID, delta string) Event { return contract.ToolCallDelta(callID, delta) }

// ToolCallDone builds the terminal event of a streamed tool call.
func ToolCallDone(callID string) Event { return contract.ToolCallDone(callID) }

// MediaOutput builds a standalone media output event.
func MediaOutput(part Part) Event { return contract.MediaOutput(part) }

// ReasoningSummary builds a reasoning summary event.
func ReasoningSummary(text string) Event { return contract.ReasoningSummary(text) }

// OutputItem builds a structured output item event.
func OutputItem(item Item) Event { return contract.OutputItem(item) }

// ActivityEvent builds an EventActivity with a structured JSON payload.
func ActivityEvent(name string, data jsontext.Value) Event {
	return contract.ActivityEvent(name, data)
}

// EmitText emits a complete assistant message.
func EmitText(emit Emit, text string) error { return contract.EmitText(emit, text) }

// EmitTextDelta emits an incremental text delta.
func EmitTextDelta(emit Emit, text string) error { return contract.EmitTextDelta(emit, text) }

// EmitToolCall emits a complete tool call.
func EmitToolCall(emit Emit, id, name, args string) error {
	return contract.EmitToolCall(emit, id, name, args)
}

// EmitToolCallStart begins a streamed tool call.
func EmitToolCallStart(emit Emit, id, name string) error {
	return contract.EmitToolCallStart(emit, id, name)
}

// EmitToolCallDelta emits streamed tool call argument bytes.
func EmitToolCallDelta(emit Emit, id, delta string) error {
	return contract.EmitToolCallDelta(emit, id, delta)
}

// EmitToolCallDone completes a streamed tool call.
func EmitToolCallDone(emit Emit, id string) error { return contract.EmitToolCallDone(emit, id) }

// EmitMedia emits a standalone media output event.
func EmitMedia(emit Emit, part Part) error { return contract.EmitMedia(emit, part) }

// EmitItem emits a structured output item event.
func EmitItem(emit Emit, item Item) error { return contract.EmitItem(emit, item) }

// NewAPIError builds a sanitized protocol error for HTTP clients.
func NewAPIError(status int, typ, code, param, message string) *APIError {
	return contract.NewAPIError(status, typ, code, param, message)
}

// Invalid builds a 400 invalid_request_error for param.
func Invalid(param, message string) *APIError { return contract.Invalid(param, message) }

// Unsupported builds a 400 unsupported feature error for param.
func Unsupported(param, message string) *APIError { return contract.Unsupported(param, message) }

// DefaultLimits returns the conservative limits used by Handler.
func DefaultLimits() Limits { return contract.DefaultLimits() }

func validRole(role Role) bool { return contract.ValidRole(role) }

// newID remains private to the root package for the handler's compatibility
// bridge; application code should not depend on its process-local format.
func newID(prefix string) string { return identity.New(prefix) }
