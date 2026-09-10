package chat

import (
	"encoding/json/jsontext"
	"errors"
	"fmt"
	"net/url"
	"strings"

	"github.com/rs/xid"
)

// Role identifies who produced a message item.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
	RoleTool      Role = "tool"
)

// ItemType tags the shape of an input or output item.
type ItemType string

const (
	ItemMessage            ItemType = "message"
	ItemFunctionCall       ItemType = "function_call"
	ItemFunctionCallOutput ItemType = "function_call_output"
	ItemReasoning          ItemType = "reasoning"
	ItemMedia              ItemType = "media"
	ItemExtension          ItemType = "extension"
)

// PartType tags a content part within an item.
type PartType string

const (
	PartText             PartType = "text"
	PartImage            PartType = "image"
	PartAudio            PartType = "audio"
	PartFile             PartType = "file"
	PartJSON             PartType = "json"
	PartReasoningSummary PartType = "reasoning_summary"
)

// Status reports lifecycle state for items and outcomes.
type Status string

const (
	StatusInProgress Status = "in_progress"
	StatusCompleted  Status = "completed"
	StatusIncomplete Status = "incomplete"
	StatusFailed     Status = "failed"
	StatusCancelled  Status = "cancelled"
)

// Media identifies bytes, a remote URL, or an application-owned asset.
type Media struct {
	MIMEType string `json:"mime_type,omitempty"`
	Format   string `json:"format,omitempty"`
	Filename string `json:"filename,omitempty"`
	URL      string `json:"url,omitempty"`
	Ref      string `json:"ref,omitempty"`
	Data     []byte `json:"data,omitempty"`
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

// Part is a typed content part.
type Part struct {
	Type     PartType       `json:"type"`
	Text     string         `json:"text,omitempty"`
	Media    *Media         `json:"media,omitempty"`
	Data     jsontext.Value `json:"data,omitempty"`
	Detail   string         `json:"detail,omitempty"`
	Filename string         `json:"filename,omitempty"`
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

// Validate checks the tagged shape of a content part and its inline media size.
func (p Part) Validate(maxMediaBytes int64) error {
	switch p.Type {
	case PartText, PartReasoningSummary:
		if p.Media != nil || len(p.Data) != 0 {
			return fmt.Errorf("%s part cannot contain media or JSON data", p.Type)
		}
	case PartImage, PartAudio, PartFile:
		if p.Media == nil {
			return fmt.Errorf("%s part is missing media", p.Type)
		}
		if err := validateMedia(*p.Media, maxMediaBytes); err != nil {
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

func validateMedia(m Media, maxBytes int64) error {
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

// Item is the library-owned tagged union used for both request input and agent output.
type Item struct {
	Type             ItemType       `json:"type"`
	ID               string         `json:"id,omitempty"`
	Status           Status         `json:"status,omitempty"`
	Role             Role           `json:"role,omitempty"`
	Content          []Part         `json:"content,omitempty"`
	CallID           string         `json:"call_id,omitempty"`
	Name             string         `json:"name,omitempty"`
	Arguments        string         `json:"arguments,omitempty"`
	Output           []Part         `json:"output,omitempty"`
	Summary          []Part         `json:"summary,omitempty"`
	EncryptedContent jsontext.Value `json:"encrypted_content,omitempty"`
	Data             jsontext.Value `json:"data,omitempty"`
}

// MessageItem returns a message item with role and content parts.
func MessageItem(role Role, content ...Part) Item {
	return Item{Type: ItemMessage, Role: role, Content: append([]Part(nil), content...)}
}

// FunctionCallItem returns a completed function call item.
func FunctionCallItem(callID, name, arguments string) Item {
	return Item{Type: ItemFunctionCall, ID: xid.New().String(), Status: StatusCompleted, CallID: callID, Name: name, Arguments: arguments}
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

// Validate checks the tagged shape of an input or output item.
func (i Item) Validate(maxMediaBytes int64, output bool) error {
	if i.Type == "" {
		return errors.New("item type is required")
	}
	if i.Status != "" && !ValidStatus(i.Status) {
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
		if !ValidRole(i.Role) {
			return fmt.Errorf("invalid message role %q", i.Role)
		}
		for _, part := range i.Content {
			if err := part.Validate(maxMediaBytes); err != nil {
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
			if err := part.Validate(maxMediaBytes); err != nil {
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
			if err := part.Validate(maxMediaBytes); err != nil {
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
		if err := i.Content[0].Validate(maxMediaBytes); err != nil {
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

// ValidRole reports whether role is one of the canonical message roles.
func ValidRole(role Role) bool {
	switch role {
	case RoleUser, RoleAssistant, RoleSystem, RoleDeveloper, RoleTool:
		return true
	default:
		return false
	}
}

// ValidStatus reports whether status is one of the canonical lifecycle statuses.
func ValidStatus(status Status) bool {
	switch status {
	case StatusInProgress, StatusCompleted, StatusIncomplete, StatusFailed, StatusCancelled:
		return true
	default:
		return false
	}
}
