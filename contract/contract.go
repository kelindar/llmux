// Package contract contains the protocol-neutral agent, request, event, and
// capability types shared by llmux and application-owned agents.
package contract

import (
	"context"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/kelindar/llmux/internal/identity"
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
type Resolver interface {
	// Resolve returns the agent and capabilities for target.
	Resolve(context.Context, string) (Agent, Capabilities, error)
}

// ResolverFunc adapts a function to Resolver.
type ResolverFunc func(context.Context, string) (Agent, Capabilities, error)

// Resolve calls f, or returns an error if f is nil.
func (f ResolverFunc) Resolve(ctx context.Context, target string) (Agent, Capabilities, error) {
	if f == nil {
		return nil, Capabilities{}, errors.New("llmux: nil resolver function")
	}
	return f(ctx, target)
}

// Role identifies who produced a message item.
type Role string

const (
	// RoleUser is the end-user message role.
	RoleUser Role = "user"
	// RoleAssistant is the model or agent message role.
	RoleAssistant Role = "assistant"
	// RoleSystem is the system instruction message role.
	RoleSystem Role = "system"
	// RoleDeveloper is the developer instruction message role.
	RoleDeveloper Role = "developer"
	// RoleTool is the tool result message role.
	RoleTool Role = "tool"
)

// ItemType tags the shape of an input or output item.
type ItemType string

const (
	// ItemMessage is a conversational message with role and content parts.
	ItemMessage ItemType = "message"
	// ItemFunctionCall is a completed tool invocation with name and arguments.
	ItemFunctionCall ItemType = "function_call"
	// ItemFunctionCallOutput is the output returned for a tool call.
	ItemFunctionCallOutput ItemType = "function_call_output"
	// ItemReasoning is a reasoning trace with optional summary parts.
	ItemReasoning ItemType = "reasoning"
	// ItemMedia is a standalone image or audio output item.
	ItemMedia ItemType = "media"
	// ItemExtension carries application-defined JSON in Data.
	ItemExtension ItemType = "extension"
)

// PartType tags a content part within an item.
type PartType string

const (
	// PartText is plain text content.
	PartText PartType = "text"
	// PartImage is image media content.
	PartImage PartType = "image"
	// PartAudio is audio media content.
	PartAudio PartType = "audio"
	// PartFile is file media content.
	PartFile PartType = "file"
	// PartJSON is structured JSON content.
	PartJSON PartType = "json"
	// PartReasoningSummary is a reasoning summary text part.
	PartReasoningSummary PartType = "reasoning_summary"
)

// Status reports lifecycle state for items and outcomes.
type Status string

const (
	// StatusInProgress means the item or run is still active.
	StatusInProgress Status = "in_progress"
	// StatusCompleted means the item or run finished successfully.
	StatusCompleted Status = "completed"
	// StatusIncomplete means the item or run ended before completion.
	StatusIncomplete Status = "incomplete"
	// StatusFailed means the item or run failed.
	StatusFailed Status = "failed"
	// StatusCancelled means the item or run was cancelled.
	StatusCancelled Status = "cancelled"
)

// StopReason explains why generation ended.
type StopReason string

const (
	// StopStop means generation finished normally.
	StopStop StopReason = "stop"
	// StopToolCall means generation stopped to invoke tools.
	StopToolCall StopReason = "tool_call"
	// StopLength means generation stopped at a length limit.
	StopLength StopReason = "length"
	// StopError means generation stopped due to an error.
	StopError StopReason = "error"
	// StopCancelled means generation was cancelled.
	StopCancelled StopReason = "cancelled"
)

// Modality is a bit set used for input and output declarations.
type Modality uint8

const (
	// ModalityText selects text input or output.
	ModalityText Modality = 1 << iota
	// ModalityImage selects image input or output.
	ModalityImage
	// ModalityAudio selects audio input or output.
	ModalityAudio
	// ModalityFile selects file input or output.
	ModalityFile
)

// Has reports whether m includes all bits in other.
func (m Modality) Has(other Modality) bool { return m&other == other }

// Media identifies bytes, a remote URL, or an application-owned asset. The
// handler never fetches URLs unless an AssetResolver was explicitly supplied.
type Media struct {
	MIMEType string `json:"mime_type,omitempty"` // MIME type for inline or referenced bytes.
	Format   string `json:"format,omitempty"`    // Audio encoding format when applicable.
	Filename string `json:"filename,omitempty"`  // Suggested filename for the media.
	URL      string `json:"url,omitempty"`       // Absolute HTTP or HTTPS URL source.
	Ref      string `json:"ref,omitempty"`       // Application-owned asset reference.
	Data     []byte `json:"data,omitempty"`      // Inline byte payload.
}

// InlineMedia returns media backed by a copied byte slice.
func InlineMedia(mime string, data []byte) Media {
	return Media{MIMEType: mime, Data: append([]byte(nil), data...)}
}

// RemoteMedia returns media referenced by an absolute HTTP or HTTPS URL.
func RemoteMedia(mime, url string) Media { return Media{MIMEType: mime, URL: url} }

// AssetMedia returns media referenced by an application-owned asset ID.
func AssetMedia(mime, ref string) Media { return Media{MIMEType: mime, Ref: ref} }

// Clone returns independent byte ownership for media that has inline data.
func (m Media) Clone() Media {
	m.Data = append([]byte(nil), m.Data...)
	return m
}

func (m Media) sourceCount() int {
	n := 0
	if len(m.Data) > 0 {
		n++
	}
	if m.URL != "" {
		n++
	}
	if m.Ref != "" {
		n++
	}
	return n
}

func (m Media) validate(maxBytes int64) error {
	switch {
	case m.sourceCount() != 1:
		return errors.New("media must have exactly one of data, url, or ref")
	case maxBytes > 0 && int64(len(m.Data)) > maxBytes:
		return fmt.Errorf("media exceeds %d byte limit", maxBytes)
	case len(m.Data) > 0 && strings.TrimSpace(m.MIMEType) == "":
		return errors.New("inline media MIME type is required")
	}
	if m.URL != "" {
		parsed, err := url.ParseRequestURI(m.URL)
		if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
			return errors.New("media url must be an absolute HTTP or HTTPS URL")
		}
	}
	return nil
}

// Part is a typed content part. Text and JSON are held directly; media keeps
// its source and encoding metadata instead of flattening it into text.
type Part struct {
	Type     PartType       `json:"type"`               // Discriminant for the part shape.
	Text     string         `json:"text,omitempty"`     // Text payload for text and summary parts.
	Media    *Media         `json:"media,omitempty"`    // Media payload for image, audio, and file parts.
	Data     jsontext.Value `json:"data,omitempty"`     // JSON payload for JSON parts.
	Detail   string         `json:"detail,omitempty"`   // Optional rendering detail hint.
	Filename string         `json:"filename,omitempty"` // Optional filename override.
}

// TextPart returns a text content part.
func TextPart(text string) Part { return Part{Type: PartText, Text: text} }

// ImagePart returns an image content part backed by media.
func ImagePart(media Media) Part { return Part{Type: PartImage, Media: &media} }

// AudioPart returns an audio content part backed by media.
func AudioPart(media Media) Part { return Part{Type: PartAudio, Media: &media} }

// FilePart returns a file content part backed by media.
func FilePart(media Media) Part { return Part{Type: PartFile, Media: &media} }

// JSONPart returns a JSON content part with a copied payload.
func JSONPart(data jsontext.Value) Part {
	return Part{Type: PartJSON, Data: append(jsontext.Value(nil), data...)}
}

// SummaryPart returns a reasoning summary text part.
func SummaryPart(text string) Part { return Part{Type: PartReasoningSummary, Text: text} }

// Clone returns a deep copy of the part and any nested media or JSON data.
func (p Part) Clone() Part {
	if p.Media != nil {
		media := p.Media.Clone()
		p.Media = &media
	}
	p.Data = append(jsontext.Value(nil), p.Data...)
	return p
}

func (p Part) validate(maxMediaBytes int64) error {
	switch p.Type {
	case PartText, PartReasoningSummary:
		if p.Media != nil || len(p.Data) != 0 {
			return fmt.Errorf("%s part cannot contain media or JSON data", p.Type)
		}
	case PartImage, PartAudio, PartFile:
		if p.Media == nil {
			return fmt.Errorf("%s part is missing media", p.Type)
		}
		if err := p.Media.validate(maxMediaBytes); err != nil {
			return err
		}
		if p.Type == PartAudio && strings.TrimSpace(p.Media.Format) == "" {
			return errors.New("audio part format is required")
		}
	case PartJSON:
		if !p.Data.IsValid() {
			return errors.New("JSON part is not valid JSON")
		}
		if p.Media != nil {
			return errors.New("JSON part cannot contain media")
		}
	default:
		return fmt.Errorf("unknown content part type %q", p.Type)
	}
	return nil
}

// Validate checks the tagged shape of a content part and its inline media
// size. It is intended for adapters and agents that build requests directly.
func (p Part) Validate(maxMediaBytes int64) error { return p.validate(maxMediaBytes) }

// Item is the library-owned tagged union used for both request input and
// agent output. The fields used depend on Type; Validate enforces that shape.
type Item struct {
	Type             ItemType       `json:"type"`                        // Discriminant for the item shape.
	ID               string         `json:"id,omitempty"`                // Stable item identifier when present.
	Status           Status         `json:"status,omitempty"`            // Lifecycle status for output items.
	Role             Role           `json:"role,omitempty"`              // Message author role.
	Content          []Part         `json:"content,omitempty"`           // Message or media content parts.
	CallID           string         `json:"call_id,omitempty"`           // Tool call identifier.
	Name             string         `json:"name,omitempty"`              // Tool or schema name.
	Arguments        string         `json:"arguments,omitempty"`         // JSON-encoded tool arguments.
	Output           []Part         `json:"output,omitempty"`            // Tool output content parts.
	Summary          []Part         `json:"summary,omitempty"`           // Reasoning summary parts.
	EncryptedContent jsontext.Value `json:"encrypted_content,omitempty"` // Opaque encrypted reasoning payload.
	Data             jsontext.Value `json:"data,omitempty"`              // Extension JSON payload.
}

// MessageItem returns a message item with role and content parts.
func MessageItem(role Role, content ...Part) Item {
	return Item{Type: ItemMessage, Role: role, Content: append([]Part(nil), content...)}
}

// FunctionCallItem returns a completed function call item.
func FunctionCallItem(callID, name, arguments string) Item {
	return Item{Type: ItemFunctionCall, ID: identity.New("fc_"), Status: StatusCompleted, CallID: callID, Name: name, Arguments: arguments}
}

// FunctionCallOutputItem returns a function call output item.
func FunctionCallOutputItem(callID string, output ...Part) Item {
	return Item{Type: ItemFunctionCallOutput, CallID: callID, Output: append([]Part(nil), output...)}
}

// Clone returns a deep copy of the item and nested parts or JSON values.
func (i Item) Clone() Item {
	i.Content = cloneParts(i.Content)
	i.Output = cloneParts(i.Output)
	i.Summary = cloneParts(i.Summary)
	i.EncryptedContent = append(jsontext.Value(nil), i.EncryptedContent...)
	i.Data = append(jsontext.Value(nil), i.Data...)
	return i
}

func cloneParts(parts []Part) []Part {
	if parts == nil {
		return nil
	}
	out := make([]Part, len(parts))
	for n, part := range parts {
		out[n] = part.Clone()
	}
	return out
}

func (i Item) validate(maxMediaBytes int64, output bool) error {
	if i.Type == "" {
		return errors.New("item type is required")
	}
	if i.Status != "" && !validStatus(i.Status) {
		return fmt.Errorf("invalid item status %q", i.Status)
	}
	switch i.Type {
	case ItemMessage:
		if len(i.Output) > 0 || i.CallID != "" || i.Name != "" || i.Arguments != "" || len(i.Summary) > 0 || len(i.EncryptedContent) > 0 || len(i.Data) > 0 {
			return errors.New("message item contains fields from another item type")
		}
		if i.Role == "" {
			return errors.New("message role is required")
		}
		if !validRole(i.Role) {
			return fmt.Errorf("invalid message role %q", i.Role)
		}
		for _, part := range i.Content {
			if err := part.validate(maxMediaBytes); err != nil {
				return err
			}
		}
		if output && i.Role != RoleAssistant {
			return fmt.Errorf("agent message role must be %q", RoleAssistant)
		}
	case ItemFunctionCall:
		if len(i.Content) > 0 || len(i.Output) > 0 || len(i.Summary) > 0 || len(i.EncryptedContent) > 0 || len(i.Data) > 0 || i.Role != "" {
			return errors.New("function call item contains fields from another item type")
		}
		if i.CallID == "" || i.Name == "" {
			return errors.New("function call requires call_id and name")
		}
		if i.Arguments == "" || !jsontext.Value(i.Arguments).IsValid() {
			return errors.New("function call arguments must be valid JSON")
		}
	case ItemFunctionCallOutput:
		if len(i.Content) > 0 || i.Role != "" || i.Name != "" || i.Arguments != "" || len(i.Summary) > 0 || len(i.EncryptedContent) > 0 || len(i.Data) > 0 {
			return errors.New("function call output item contains fields from another item type")
		}
		if i.CallID == "" {
			return errors.New("function call output requires call_id")
		}
		for _, part := range i.Output {
			if err := part.validate(maxMediaBytes); err != nil {
				return err
			}
		}
	case ItemReasoning:
		if len(i.Content) > 0 || len(i.Output) > 0 || i.Role != "" || i.CallID != "" || i.Name != "" || i.Arguments != "" {
			return errors.New("reasoning item contains fields from another item type")
		}
		for _, part := range i.Summary {
			if part.Type != PartReasoningSummary {
				return errors.New("reasoning summary must contain reasoning_summary parts")
			}
			if err := part.validate(maxMediaBytes); err != nil {
				return err
			}
		}
	case ItemMedia:
		if len(i.Output) > 0 || len(i.Summary) > 0 || i.CallID != "" || i.Name != "" || i.Arguments != "" || len(i.EncryptedContent) > 0 || len(i.Data) > 0 {
			return errors.New("media item contains fields from another item type")
		}
		if len(i.Content) != 1 {
			return errors.New("media item requires exactly one content part")
		}
		if i.Content[0].Type != PartImage && i.Content[0].Type != PartAudio {
			return errors.New("media item must contain image or audio")
		}
		if err := i.Content[0].validate(maxMediaBytes); err != nil {
			return err
		}
	case ItemExtension:
		if len(i.Content) > 0 || len(i.Output) > 0 || i.Role != "" || i.CallID != "" || i.Name != "" || i.Arguments != "" || len(i.Summary) > 0 || len(i.EncryptedContent) > 0 {
			return errors.New("extension item contains fields from another item type")
		}
		if len(i.Data) == 0 || !i.Data.IsValid() {
			return errors.New("extension item requires valid JSON data")
		}
	default:
		return fmt.Errorf("unknown item type %q", i.Type)
	}
	return nil
}

// Validate checks the tagged shape of an input or output item.
func (i Item) Validate(maxMediaBytes int64, output bool) error {
	return i.validate(maxMediaBytes, output)
}

func validRole(role Role) bool {
	switch role {
	case RoleUser, RoleAssistant, RoleSystem, RoleDeveloper, RoleTool:
		return true
	default:
		return false
	}
}

// ValidRole reports whether role is one of the canonical message roles.
func ValidRole(role Role) bool { return validRole(role) }

func validStatus(status Status) bool {
	switch status {
	case StatusInProgress, StatusCompleted, StatusIncomplete, StatusFailed, StatusCancelled:
		return true
	default:
		return false
	}
}

// ValidStatus reports whether status is one of the canonical lifecycle
// statuses.
func ValidStatus(status Status) bool { return validStatus(status) }

// Tool declares a function tool available to the agent.
type Tool struct {
	Name        string         `json:"name"`                  // Function name exposed to the model.
	Description string         `json:"description,omitempty"` // Human-readable tool description.
	Parameters  jsontext.Value `json:"parameters,omitempty"`  // JSON Schema object for arguments.
	Strict      *bool          `json:"strict,omitempty"`      // Whether strict schema validation is required.
}

func (t Tool) validate() error {
	if t.Name == "" {
		return errors.New("tool name is required")
	}
	if len(t.Parameters) != 0 {
		var parameters map[string]jsontext.Value
		if err := json.Unmarshal(t.Parameters, &parameters); err != nil || parameters == nil {
			return fmt.Errorf("tool %q parameters must be a JSON object", t.Name)
		}
	}
	return nil
}

// Validate checks a function tool declaration.
func (t Tool) Validate() error { return t.validate() }

// ToolChoice selects how the model may use declared tools.
type ToolChoice struct {
	Mode string // Selection mode: auto, none, required, any, or function.
	Name string // Function name when Mode is function.
}

func (t ToolChoice) validate() error {
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

// Validate checks a tool-choice declaration.
func (t ToolChoice) Validate() error { return t.validate() }

// StructuredOutput declares a named JSON schema for model output.
type StructuredOutput struct {
	Name        string         // Schema name returned with the output.
	Description string         // Human-readable schema description.
	Schema      jsontext.Value // JSON Schema for the output object.
	Strict      bool           // Whether strict schema validation is required.
}

func (s StructuredOutput) validate() error {
	if s.Name == "" || len(s.Schema) == 0 || !s.Schema.IsValid() {
		return errors.New("structured output requires a name and valid schema")
	}
	return nil
}

// Validate checks a structured-output declaration.
func (s StructuredOutput) Validate() error { return s.validate() }

// Reasoning configures optional reasoning effort and summary output.
type Reasoning struct {
	Effort  string // Requested reasoning effort level.
	Summary bool   // Whether to emit reasoning summaries.
}

func (r Reasoning) validate() error {
	if strings.TrimSpace(r.Effort) == "" && !r.Summary {
		return errors.New("reasoning requires effort or summary")
	}
	return nil
}

// Validate checks reasoning controls.
func (r Reasoning) Validate() error { return r.validate() }

// AudioControls configures Chat Completions audio output.
type AudioControls struct {
	Voice  string // Voice identifier for synthesized audio.
	Format string // Output encoding format.
}

func (a AudioControls) validate() error {
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

// Validate checks Chat Completions audio controls.
func (a AudioControls) Validate() error { return a.validate() }

// Controls holds generation and protocol options for a request.
type Controls struct {
	MaxOutputTokens    *int                      // Maximum tokens to generate.
	Temperature        *float64                  // Sampling temperature.
	TopP               *float64                  // Nucleus sampling threshold.
	Stop               []string                  // Stop sequences.
	Tools              []Tool                    // Function tools available to the model.
	ToolChoice         *ToolChoice               // Tool selection policy.
	ParallelToolCall   *bool                     // Whether parallel tool calls are allowed.
	IncludeUsage       bool                      // Whether to include token usage in the response.
	Structured         *StructuredOutput         // Structured JSON output schema.
	PreviousResponseID *string                   // Prior response ID for continuation.
	Store              *bool                     // Wire store flag; nil when omitted. Effective policy is Request.Retain.
	Reasoning          *Reasoning                // Reasoning effort and summary controls.
	Audio              *AudioControls            // Chat Completions audio output controls.
	ImageGeneration    bool                      // Whether image generation is requested.
	Metadata           map[string]string         // Request metadata; cloned into ResponseState at acceptance.
	Extensions         map[string]jsontext.Value // Application extension payloads keyed by name.
}

// OutputSpec declares expected response modalities and formatting.
type OutputSpec struct {
	Modalities Modality       // Bit set of output modalities.
	Format     string         // Response format identifier.
	Schema     jsontext.Value // JSON Schema when structured text output is requested.
}

// Request is the protocol-neutral input passed to Agent.Run.
//
// Turn vs Input:
//   - Turn is the items submitted in this HTTP request only.
//   - Input is the effective conversation for the agent: loaded history
//     followed by Turn, exactly once and in order.
//   - Controls.PreviousResponseID names the prior response whose history
//     was loaded; the application authorizes that load.
//
// Retain is the effective content-retention policy after applying the
// handler's StoreDefault. Controls.Store remains the wire value (nil when
// omitted). The handler sets Retain before Accept; applications must treat
// the request as read-only afterward.
//
// Applications that persist turns should retain Turn and ResponseState.Output
// rather than re-saving the full Input conversation on every response.
type Request struct {
	Target       string     // Agent target name selected by the resolver.
	Instructions string     // System or developer instructions for the run.
	Turn         []Item     // Items submitted in this request only.
	Input        []Item     // Effective history+turn for Agent.Run.
	Controls     Controls   // Generation and protocol controls.
	Output       OutputSpec // Declared output modalities and format.
	Retain       bool       // Effective content retention; set by the handler.
}

// Limits bound request, media, event, and accumulated response memory. Zero
// fields use the documented defaults.
type Limits struct {
	MaxRequestBytes   int64 // Maximum decoded request body size.
	MaxMediaBytes     int64 // Maximum inline media payload size.
	MaxAssets         int   // Maximum asset references resolved per request.
	MaxOutputBytes    int64 // Maximum accumulated agent output size.
	MaxEventBytes     int64 // Maximum serialized event payload size.
	MaxMultipartBytes int64 // Maximum multipart upload size.
}

// DefaultLimits returns the conservative limits used by a Handler when an
// option leaves a field at zero.
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

// Usage reports token accounting for a completed run.
type Usage struct {
	InputTokens     int // Prompt tokens charged for the request.
	OutputTokens    int // Completion tokens generated for the response.
	TotalTokens     int // Combined input and output token count.
	CachedTokens    int // Prompt tokens served from cache.
	ReasoningTokens int // Tokens attributed to reasoning output.
}

// Outcome is the final agent result returned from Agent.Run.
// Outcome cannot be in_progress; use ResponseState for retrieval of
// non-terminal stored responses.
type Outcome struct {
	Status     Status     // Final lifecycle status for the run.
	StopReason StopReason // Reason generation stopped, when applicable.
	Usage      *Usage     // Token usage for the run, when reported.
}

// ResponseState is the client-visible response payload shared by creation,
// finalization, replay, and retrieval. Response identity (ID) and creation
// timestamp remain application-controlled and are supplied separately.
//
// Error is a sanitized public error only. Operational Go errors stay on
// TurnResult.Err and are never copied into Error.Message automatically.
//
// Metadata is owned by this value after Clone; callers must Clone before
// retaining across requests.
type ResponseState struct {
	Status      Status            // completed, failed, incomplete, cancelled, or in_progress
	Output      []Item            // Output items for this response turn
	Usage       *Usage            // Token usage when known
	Error       *APIError         // Sanitized public error when status is failed
	Incomplete  string            // incomplete_details.reason when status is incomplete
	CompletedAt int64             // Unix completion time; zero while in_progress
	Metadata    map[string]string // Response metadata captured at acceptance
	Store       bool              // Effective content-retention policy
}

// Clone returns a deep copy safe for independent retention.
func (s ResponseState) Clone() ResponseState {
	out := s
	if s.Output != nil {
		out.Output = make([]Item, len(s.Output))
		for i, item := range s.Output {
			out.Output[i] = item.Clone()
		}
	}
	if s.Usage != nil {
		u := *s.Usage
		out.Usage = &u
	}
	if s.Error != nil {
		e := *s.Error
		e.Err = nil // never retain operational cause on cloned public errors
		out.Error = &e
	}
	out.Metadata = CloneMetadata(s.Metadata)
	return out
}

// CloneMetadata returns a shallow copy of metadata, or nil when empty.
func CloneMetadata(src map[string]string) map[string]string {
	if len(src) == 0 {
		return nil
	}
	out := make(map[string]string, len(src))
	for k, v := range src {
		out[k] = v
	}
	return out
}

// CleanupContext returns a bounded context that keeps parent values but not
// parent cancellation. The caller owns cancel and must call it after Finalize.
// timeout must be positive.
func CleanupContext(parent context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		timeout = time.Second
	}
	return context.WithTimeout(context.WithoutCancel(parent), timeout)
}

func (o Outcome) validate() error {
	switch {
	case o.Status != "" && !validStatus(o.Status):
		return fmt.Errorf("invalid outcome status %q", o.Status)
	case o.Status == StatusInProgress:
		return errors.New("outcome status cannot be in_progress")
	case o.Usage != nil && (o.Usage.InputTokens < 0 || o.Usage.OutputTokens < 0 || o.Usage.TotalTokens < 0 || o.Usage.CachedTokens < 0 || o.Usage.ReasoningTokens < 0):
		return errors.New("outcome usage cannot contain negative values")
	}
	if o.StopReason != "" {
		switch o.StopReason {
		case StopStop, StopToolCall, StopLength, StopError, StopCancelled:
		default:
			return fmt.Errorf("invalid outcome stop reason %q", o.StopReason)
		}
	}
	return nil
}

// Validate checks an agent outcome.
func (o Outcome) Validate() error { return o.validate() }

// Capabilities describes what the selected agent can actually execute. A
// zero Capabilities value means text in, text out, with ordinary generation
// controls. Extra modalities and behaviors must be declared explicitly.
type Capabilities struct {
	InputModalities    Modality          // Accepted request input modalities.
	OutputModalities   Modality          // Supported response output modalities.
	GenerationControls GenerationControl // Supported generation control flags.
	Tools              bool              // Whether function tools are supported.
	ClientTools        bool              // Whether client-executed tools are supported.
	StructuredOutput   bool              // Whether structured JSON output is supported.
	ReasoningSummary   bool              // Whether reasoning summaries may be emitted.
	ImageGeneration    bool              // Whether image generation is supported.
	Continuation       bool              // Whether response continuation is supported.
	Extensions         map[string]bool   // Supported extension names.
}

// GenerationControl identifies a generation control understood by an agent.
// The zero Capabilities value accepts the ordinary text-generation controls.
type GenerationControl uint16

const (
	// ControlMaxOutputTokens allows MaxOutputTokens in Controls.
	ControlMaxOutputTokens GenerationControl = 1 << iota
	// ControlTemperature allows Temperature in Controls.
	ControlTemperature
	// ControlTopP allows TopP in Controls.
	ControlTopP
	// ControlStop allows Stop sequences in Controls.
	ControlStop
	// ControlParallelToolCalls allows ParallelToolCall in Controls.
	ControlParallelToolCalls
	// ControlReasoning allows Reasoning in Controls.
	ControlReasoning
	// ControlAudio allows Audio in Controls.
	ControlAudio
)

// Has reports whether c includes all bits in other.
func (c GenerationControl) Has(other GenerationControl) bool { return c&other == other }

func (c Capabilities) normalized() Capabilities {
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

// Normalize fills the defaults implied by a zero Capabilities value.
func (c Capabilities) Normalize() Capabilities { return c.normalized() }

// EventType tags an event emitted during Agent.Run.
type EventType string

const (
	// EventMessage is a complete assistant message item.
	EventMessage EventType = "message"
	// EventTextDelta is an incremental text fragment.
	EventTextDelta EventType = "text_delta"
	// EventTextDone marks the end of a text stream for an item.
	EventTextDone EventType = "text_done"
	// EventToolCall is a complete tool call with final arguments.
	EventToolCall EventType = "tool_call"
	// EventToolCallStart opens a streaming tool call.
	EventToolCallStart EventType = "tool_call_start"
	// EventToolCallDelta carries incremental tool call arguments.
	EventToolCallDelta EventType = "tool_call_delta"
	// EventToolCallDone closes a streaming tool call.
	EventToolCallDone EventType = "tool_call_done"
	// EventMedia is a media output part.
	EventMedia EventType = "media"
	// EventReasoning is a reasoning summary item.
	EventReasoning EventType = "reasoning"
	// EventItem is a generic output item.
	EventItem EventType = "item"
	// EventActivity is application-specific streaming activity (Responses only).
	EventActivity EventType = "activity"
)

// Event is the only output path from an Agent. Delta arguments may be
// incomplete JSON until EventToolCallDone or Run return closes the call.
type Event struct {
	Type      EventType      // Discriminant for the event shape.
	ItemID    string         // Target item identifier for incremental updates.
	CallID    string         // Tool call identifier for tool events.
	Name      string         // Tool, schema, or activity name when applicable.
	Delta     string         // Incremental text or arguments fragment.
	Arguments string         // Final JSON tool arguments for complete tool calls.
	Text      string         // Final text payload when used without Item.
	Item      Item           // Complete output item for message, reasoning, and item events.
	Part      Part           // Media part for media output events.
	Data      jsontext.Value // Structured JSON payload for EventActivity.
}

// CompleteText returns an EventMessage carrying a full assistant text message.
func CompleteText(text string) Event {
	return Event{Type: EventMessage, Item: MessageItem(RoleAssistant, TextPart(text))}
}

// TextDelta returns an EventTextDelta with a text fragment.
func TextDelta(text string) Event { return Event{Type: EventTextDelta, Delta: text} }

// ToolCall returns an EventToolCall with final arguments.
func ToolCall(callID, name, arguments string) Event {
	return Event{Type: EventToolCall, CallID: callID, Name: name, Arguments: arguments}
}

// ToolCallStart returns an EventToolCallStart that opens a streaming tool call.
func ToolCallStart(callID, name string) Event {
	return Event{Type: EventToolCallStart, CallID: callID, Name: name}
}

// ToolCallDelta returns an EventToolCallDelta with an arguments fragment.
func ToolCallDelta(callID, argumentsDelta string) Event {
	return Event{Type: EventToolCallDelta, CallID: callID, Delta: argumentsDelta}
}

// ToolCallDone returns an EventToolCallDone that closes a streaming tool call.
func ToolCallDone(callID string) Event { return Event{Type: EventToolCallDone, CallID: callID} }

// MediaOutput returns an EventMedia carrying a media part.
func MediaOutput(part Part) Event { return Event{Type: EventMedia, Part: part} }

// ReasoningSummary returns an EventReasoning with a summary item.
func ReasoningSummary(text string) Event {
	return Event{Type: EventReasoning, Item: Item{Type: ItemReasoning, Summary: []Part{SummaryPart(text)}}}
}

// OutputItem returns an EventItem carrying an output item.
func OutputItem(item Item) Event { return Event{Type: EventItem, Item: item} }

// Emit is the serial event function passed to Agent.Run.
type Emit func(Event) error

// EmitText emits a complete assistant text message through emit.
func EmitText(emit Emit, text string) error { return emit(CompleteText(text)) }

// EmitTextDelta emits a text fragment through emit.
func EmitTextDelta(emit Emit, text string) error { return emit(TextDelta(text)) }

// EmitToolCall emits a complete tool call through emit.
func EmitToolCall(emit Emit, callID, name, arguments string) error {
	return emit(ToolCall(callID, name, arguments))
}

// EmitToolCallStart opens a streaming tool call through emit.
func EmitToolCallStart(emit Emit, callID, name string) error {
	return emit(ToolCallStart(callID, name))
}

// EmitToolCallDelta emits a streaming tool call arguments fragment through emit.
func EmitToolCallDelta(emit Emit, callID, argumentsDelta string) error {
	return emit(ToolCallDelta(callID, argumentsDelta))
}

// EmitToolCallDone closes a streaming tool call through emit.
func EmitToolCallDone(emit Emit, callID string) error { return emit(ToolCallDone(callID)) }

// EmitMedia emits a media output part through emit.
func EmitMedia(emit Emit, part Part) error { return emit(MediaOutput(part)) }

// EmitItem emits an output item through emit.
func EmitItem(emit Emit, item Item) error { return emit(OutputItem(item)) }

var (
	// ErrConcurrentEmit is returned when Emit is invoked concurrently.
	ErrConcurrentEmit = errors.New("llmux: concurrent Emit calls are not supported")
	// ErrEmitClosed is returned when Emit is called after the stream is closed.
	ErrEmitClosed = errors.New("llmux: emission is closed")
	// ErrDelivery is returned when a client write fails. With durable
	// acceptance, the handler stops delivery but does not cancel execution.
	ErrDelivery = errors.New("llmux: client delivery failed")
)

// ActivityEvent builds an EventActivity with a structured JSON payload.
// Name must be a non-empty token; Responses emits it as response.activity.<name>.
// Activity is never assistant output and is omitted from continuation history.
func ActivityEvent(name string, data jsontext.Value) Event {
	return Event{Type: EventActivity, Name: name, Data: append(jsontext.Value(nil), data...)}
}

// APIError is a sanitized protocol error that an application may return from
// a resolver or agent. The handler never exposes an arbitrary underlying error
// message unless the application placed it in Message explicitly.
type APIError struct {
	Status  int    // HTTP status code for the error response.
	Type    string // Protocol error type string.
	Code    string // Machine-readable error code.
	Param   string // Request field associated with the error.
	Message string // Sanitized error message exposed to clients.
	Err     error  // Underlying error retained for logging and Unwrap.
}

// Error returns Message, the wrapped error text, or a default API error string.
func (e *APIError) Error() string {
	switch {
	case e == nil:
		return ""
	case e.Message != "":
		return e.Message
	case e.Err != nil:
		return e.Err.Error()
	default:
		return "llmux API error"
	}
}

// Unwrap returns the underlying error.
func (e *APIError) Unwrap() error { return e.Err }

// NewAPIError constructs an APIError with the given response fields.
func NewAPIError(status int, typ, code, param, message string) *APIError {
	return &APIError{Status: status, Type: typ, Code: code, Param: param, Message: message}
}

// Invalid constructs a 400 invalid_request_error for param.
func Invalid(param, message string) *APIError {
	return NewAPIError(400, "invalid_request_error", "invalid_request", param, message)
}

// Unsupported constructs a 400 unsupported invalid_request_error for param.
func Unsupported(param, message string) *APIError {
	return NewAPIError(400, "invalid_request_error", "unsupported", param, message)
}

// AssetResolver is opt-in. Resolve receives the original media descriptor and
// a hard byte ceiling. It must authorize the reference using the request
// context and return inline data or another bounded representation.
type AssetResolver interface {
	// Resolve materializes media.Ref into bounded inline or remote media.
	Resolve(context.Context, Media, int64) (Media, error)
}

// AssetResolverFunc adapts a function to AssetResolver.
type AssetResolverFunc func(context.Context, Media, int64) (Media, error)

// Resolve calls f, or returns an error if f is nil.
func (f AssetResolverFunc) Resolve(ctx context.Context, media Media, maxBytes int64) (Media, error) {
	if f == nil {
		return Media{}, errors.New("llmux: nil asset resolver function")
	}
	return f(ctx, media, maxBytes)
}

// ContinuationStore loads prior-turn history for previous_response_id.
// The handler passes the authenticated request context. Persistence of new
// turns is owned by Lifecycle, not this interface.
type ContinuationStore interface {
	// Load returns the history items that should precede the current Turn
	// when continuing from responseID. The application authorizes access and
	// validates that the prior response belongs to the selected target.
	Load(context.Context, string) ([]Item, error)
}

// Lifecycle is the optional application-owned response identity and
// persistence seam.
//
// Ordering for a configured Lifecycle:
//  1. Validate request, resolve effective store policy, load continuation.
//  2. Accept — may reserve identity, reject conflicts, or return a Replay.
//  3. Agent.Run (skipped on Replay) using Accept.Context when Durable.
//  4. Finalize — exactly once for every accepted execution (success, failure,
//     or cancellation). Uses Accept.Finalize when set, otherwise the
//     execution context. Not called for Accept errors or completed Replays.
//  5. Advertise success to the client only after Finalize succeeds.
//
// Durable detaches client disconnect from execution cancellation; it does
// not create a job system. Production applications must bound execution
// (Accept.Context) and finalization (Accept.Finalize / CleanupContext).
//
// store:false (explicit or effective) means do not retain response content
// for later retrieval or continuation. Finalize still runs so bookkeeping
// can clear; ResponseState.Store is false. Applications that cannot honor
// store:false should reject in Accept.
//
// Activity: set Acceptance.Activity to allow EventActivity frames on
// Responses streams. Emit with ActivityEvent(name, json). Clients consume
// response.activity.<name>. Keep application-specific response fields
// outside the standard envelope.
type Lifecycle interface {
	// Accept runs after validation and history loading, before Agent.Run and
	// before any successful streaming headers or events.
	Accept(context.Context, *TurnRequest) (Acceptance, error)
	// Finalize persists the terminal result. A non-nil error must not be
	// reported to the client as successful completion.
	Finalize(context.Context, *TurnResult) error
}

// TurnRequest is the input to Lifecycle.Accept.
type TurnRequest struct {
	Request        *Request // Populated Turn, Input, and Retain; treat as read-only.
	Stream         bool     // Whether the client requested streaming.
	IdempotencyKey string   // Idempotency-Key header value, if any.
	Store          *bool    // Wire store value; nil when the field was omitted.
	Retain         bool     // Effective retention after StoreDefault.
}

// Acceptance is the per-request result of Lifecycle.Accept.
type Acceptance struct {
	ID       string          // Response ID for envelopes; empty uses a library default.
	Created  int64           // Unix created time; zero uses a library default.
	Replay   *ResponseState  // When set, skip Agent.Run and encode this result.
	Durable  bool            // When true, client disconnect does not cancel execution.
	Context  context.Context // Optional bounded run context when Durable.
	Finalize context.Context // Optional bounded cleanup context for Finalize.
	Activity bool            // When true, Responses may emit EventActivity frames.
}

// TurnResult is the input to Lifecycle.Finalize.
type TurnResult struct {
	ID      string        // Response ID that was accepted.
	Created int64         // Creation timestamp used in envelopes.
	Request *Request      // Same request; Turn is the submitted input.
	State   ResponseState // Client-visible terminal (or cancelled) state.
	Err     error         // Operational execution error, if any.
	Stream  bool          // Whether the client requested streaming.
}
