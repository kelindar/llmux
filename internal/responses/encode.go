package responses

import (
	"encoding/base64"
	"encoding/json/jsontext"
	"errors"

	"github.com/kelindar/llmux/chat"
	"github.com/kelindar/llmux/internal/execution"
)

// wireResponse is the OpenAI Responses JSON envelope.
type wireResponse struct {
	ID                 string            `json:"id"`
	Object             string            `json:"object"`
	CreatedAt          int64             `json:"created_at"`
	Status             string            `json:"status"`
	Error              any               `json:"error"`
	IncompleteDetails  any               `json:"incomplete_details"`
	Instructions       any               `json:"instructions"`
	Model              string            `json:"model"`
	Output             []any             `json:"output"`
	ParallelToolCalls  bool              `json:"parallel_tool_calls"`
	PreviousResponseID any               `json:"previous_response_id"`
	Reasoning          wireReasoning     `json:"reasoning"`
	Store              bool              `json:"store"`
	Temperature        any               `json:"temperature"`
	Text               wireText          `json:"text"`
	ToolChoice         any               `json:"tool_choice"`
	Tools              []any             `json:"tools"`
	TopP               any               `json:"top_p"`
	Metadata           map[string]string `json:"metadata"`
	MaxOutputTokens    any               `json:"max_output_tokens"`
	CompletedAt        int64             `json:"completed_at,omitzero"`
	Usage              *wireUsage        `json:"usage,omitzero"`
}

type wireReasoning struct {
	Effort  any `json:"effort"`
	Summary any `json:"summary"`
}

type wireText struct {
	Format wireTextFormat `json:"format"`
}

type wireTextFormat struct {
	Type        string         `json:"type"`
	Name        string         `json:"name,omitzero"`
	Description string         `json:"description,omitzero"`
	Schema      jsontext.Value `json:"schema,omitzero"`
	Strict      bool           `json:"strict,omitzero"`
}

type wireUsage struct {
	InputTokens         int             `json:"input_tokens"`
	OutputTokens        int             `json:"output_tokens"`
	TotalTokens         int             `json:"total_tokens"`
	InputTokensDetails  wireTokenDetail `json:"input_tokens_details"`
	OutputTokensDetails wireTokenDetail `json:"output_tokens_details"`
}

type wireTokenDetail struct {
	CachedTokens    int `json:"cached_tokens,omitzero"`
	ReasoningTokens int `json:"reasoning_tokens,omitzero"`
}

type wireOutputText struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	Annotations []any  `json:"annotations"`
}

type wireMessageItem struct {
	ID      string           `json:"id"`
	Type    string           `json:"type"`
	Status  chat.Status      `json:"status"`
	Role    chat.Role        `json:"role"`
	Content []wireOutputText `json:"content"`
}

type wireFunctionCallItem struct {
	ID        string      `json:"id"`
	Type      string      `json:"type"`
	Status    chat.Status `json:"status"`
	CallID    string      `json:"call_id"`
	Name      string      `json:"name"`
	Arguments string      `json:"arguments"`
}

type wireFunctionOutputItem struct {
	ID     string      `json:"id"`
	Type   string      `json:"type"`
	Status chat.Status `json:"status"`
	CallID string      `json:"call_id"`
	Output any         `json:"output"`
}

type wireReasoningItem struct {
	ID               string            `json:"id"`
	Type             string            `json:"type"`
	Status           chat.Status       `json:"status"`
	Summary          []wireSummaryPart `json:"summary"`
	EncryptedContent jsontext.Value    `json:"encrypted_content,omitzero"`
}

type wireSummaryPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type wireImageItem struct {
	ID     string      `json:"id"`
	Type   string      `json:"type"`
	Status chat.Status `json:"status"`
	Result string      `json:"result,omitzero"`
}

type wireFunctionTool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  jsontext.Value `json:"parameters"`
	Strict      bool           `json:"strict"`
}

type wireErrorBody struct {
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

type wireIncomplete struct {
	Reason string `json:"reason"`
}

// Response builds a non-streaming Responses API response body.
func (Adapter) Response(req chat.Request, result execution.Result, meta responseMeta) (any, error) {
	output := make([]any, 0, len(result.Items))
	for _, item := range result.Items {
		value, err := responseItem(item)
		if err != nil {
			return nil, err
		}
		output = append(output, value)
	}
	resp := meta.Response
	if len(resp.Output) == 0 {
		resp.Output = result.Items
	}
	if resp.Status == "" {
		resp.Status = result.Outcome.Status
	}
	if resp.Usage == nil {
		resp.Usage = result.Outcome.Usage
	}
	if resp.Target == "" {
		resp.Target = req.Target
	}
	if resp.Instructions == "" {
		resp.Instructions = req.Instructions
	}
	return responseObject(req, resp, output), nil
}

// Render builds the same Responses envelope used for creation, replay, and
// application-owned GET retrieval from one Response value.
func Render(resp chat.Response) (any, error) {
	output := make([]any, 0, len(resp.Output))
	for _, item := range resp.Output {
		value, err := responseItem(item)
		if err != nil {
			return nil, err
		}
		output = append(output, value)
	}
	req := chat.Request{Target: resp.Target, Instructions: resp.Instructions}
	return responseObject(req, resp, output), nil
}

func responseStatus(resp chat.Response, outcome chat.Outcome) string {
	switch {
	case resp.Status != "":
		return string(resp.Status)
	case outcome.Status != "":
		return string(outcome.Status)
	default:
		return string(chat.StatusCompleted)
	}
}

func responseObject(req chat.Request, resp chat.Response, output []any) wireResponse {
	status := responseStatus(resp, chat.Outcome{})
	model := resp.Target
	if model == "" {
		model = req.Target
	}
	value := wireResponse{
		ID:                 resp.ID,
		Object:             "response",
		CreatedAt:          resp.Created,
		Status:             status,
		Error:              nil,
		IncompleteDetails:  nil,
		Instructions:       nil,
		Model:              model,
		Output:             output,
		ParallelToolCalls:  true,
		PreviousResponseID: nil,
		Reasoning:          wireReasoning{},
		Store:              resp.Store,
		Temperature:        nil,
		Text:               wireText{Format: wireTextFormat{Type: "text"}},
		ToolChoice:         "auto",
		Tools:              []any{},
		TopP:               nil,
		Metadata:           map[string]string{},
		MaxOutputTokens:    nil,
	}
	if resp.CompletedAt > 0 {
		value.CompletedAt = resp.CompletedAt
	}
	if resp.Error != nil {
		value.Error = wireErrorBody{Type: resp.Error.Code, Code: resp.Error.Code, Message: resp.Error.Message}
	}
	if resp.Incomplete != "" {
		value.IncompleteDetails = wireIncomplete{Reason: resp.Incomplete}
	}
	if len(resp.Metadata) > 0 {
		value.Metadata = resp.Metadata
	}
	instructions := resp.Instructions
	if instructions == "" {
		instructions = req.Instructions
	}
	if instructions != "" {
		value.Instructions = instructions
	}
	if req.Controls.MaxOutputTokens != nil {
		value.MaxOutputTokens = *req.Controls.MaxOutputTokens
	}
	if req.Controls.Temperature != nil {
		value.Temperature = *req.Controls.Temperature
	}
	if req.Controls.TopP != nil {
		value.TopP = *req.Controls.TopP
	}
	if req.Controls.ParallelToolCall != nil {
		value.ParallelToolCalls = *req.Controls.ParallelToolCall
	}
	if resp.Previous != nil {
		value.PreviousResponseID = *resp.Previous
	}
	switch req.Output.Format.Kind {
	case chat.FormatJSONSchema:
		value.Text = wireText{Format: wireTextFormat{
			Type:        "json_schema",
			Name:        req.Output.Format.Name,
			Description: req.Output.Format.Description,
			Schema:      jsontext.Value(req.Output.Format.Schema),
			Strict:      req.Output.Format.Strict,
		}}
	case chat.FormatJSONObject:
		value.Text = wireText{Format: wireTextFormat{Type: "json_object"}}
	}
	if len(req.Controls.Tools) > 0 {
		tools := make([]any, 0, len(req.Controls.Tools))
		for _, tool := range req.Controls.Tools {
			parameters := jsontext.Value(tool.Parameters)
			if len(parameters) == 0 {
				parameters = jsontext.Value(`{"type":"object"}`)
			}
			tools = append(tools, wireFunctionTool{
				Type:        "function",
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  parameters,
				Strict:      boolPointerValue(tool.Strict),
			})
		}
		value.Tools = tools
	}
	if req.Controls.ImageGeneration {
		value.Tools = append(value.Tools, map[string]any{"type": "image_generation"})
	}
	if req.Controls.ToolChoice != nil {
		if req.Controls.ToolChoice.Mode == "function" {
			value.ToolChoice = map[string]any{"type": "function", "name": req.Controls.ToolChoice.Name}
		} else {
			value.ToolChoice = req.Controls.ToolChoice.Mode
		}
	}
	if resp.Usage != nil {
		u := responseUsage(resp.Usage)
		value.Usage = &u
	}
	return value
}

func boolPointerValue(value *bool) bool { return value != nil && *value }

func responseUsage(usage *chat.Usage) wireUsage {
	return wireUsage{
		InputTokens:         usage.Input,
		OutputTokens:        usage.Output,
		TotalTokens:         usage.Total,
		InputTokensDetails:  wireTokenDetail{CachedTokens: usage.Cached},
		OutputTokensDetails: wireTokenDetail{ReasoningTokens: usage.Reasoning},
	}
}

func responseItem(item chat.Item) (any, error) {
	status := item.Status
	if status == "" {
		status = chat.StatusCompleted
	}
	switch item.Type {
	case chat.ItemMessage:
		for _, part := range item.Content {
			if part.Type != chat.PartText {
				return nil, chat.Unsupported("output", "Responses message output supports text parts only")
			}
		}
		return wireMessageItem{
			ID: item.ID, Type: "message", Status: status, Role: item.Role,
			Content: outputTextParts(item.Content),
		}, nil
	case chat.ItemFunctionCall:
		return wireFunctionCallItem{
			ID: item.ID, Type: "function_call", Status: status,
			CallID: item.CallID, Name: item.Name, Arguments: item.Arguments,
		}, nil
	case chat.ItemFunctionCallOutput:
		output, err := responseFunctionOutput(item.Output)
		if err != nil {
			return nil, err
		}
		return wireFunctionOutputItem{
			ID: item.ID, Type: "function_call_output", Status: status,
			CallID: item.CallID, Output: output,
		}, nil
	case chat.ItemReasoning:
		if len(item.Data) > 0 {
			return nil, chat.Unsupported("output", "Responses reasoning output cannot carry opaque data")
		}
		summary := make([]wireSummaryPart, 0, len(item.Summary))
		for _, part := range item.Summary {
			summary = append(summary, wireSummaryPart{Type: "summary_text", Text: part.Text})
		}
		value := wireReasoningItem{ID: item.ID, Type: "reasoning", Status: status, Summary: summary}
		if len(item.EncryptedContent) > 0 {
			value.EncryptedContent = jsontext.Value(item.EncryptedContent)
		}
		return value, nil
	case chat.ItemMedia:
		if len(item.Content) != 1 || item.Content[0].Type != chat.PartImage || item.Content[0].Media == nil {
			return nil, chat.Unsupported("output", "Responses image output requires one image media part")
		}
		if len(item.Content[0].Media.Data) == 0 {
			return nil, chat.Unsupported("output", "Responses image output requires inline image data")
		}
		return wireImageItem{
			ID: item.ID, Type: "image_generation_call", Status: status,
			Result: base64.StdEncoding.EncodeToString(item.Content[0].Media.Data),
		}, nil
	default:
		return nil, chat.Unsupported("output", "unsupported Responses output item")
	}
}

func responseFunctionOutput(parts []chat.Part) (any, error) {
	if len(parts) == 1 && parts[0].Type == chat.PartText {
		return parts[0].Text, nil
	}
	values := make([]any, 0, len(parts))
	for _, part := range parts {
		switch part.Type {
		case chat.PartText:
			values = append(values, map[string]any{"type": "input_text", "text": part.Text})
		case chat.PartImage, chat.PartFile:
			if part.Media == nil {
				return nil, errors.New("function output media is missing")
			}
			if len(part.Media.Data) > 0 {
				data, err := mediaDataURL(*part.Media)
				if err != nil {
					return nil, err
				}
				if part.Type == chat.PartImage {
					values = append(values, map[string]any{"type": "input_image", "image_url": data, "detail": "auto"})
				} else {
					values = append(values, map[string]any{"type": "input_file", "file_data": base64.StdEncoding.EncodeToString(part.Media.Data), "filename": part.Media.Filename})
				}
			} else {
				return nil, chat.Unsupported("output", "function output media must be inline")
			}
		default:
			return nil, chat.Unsupported("output", "unsupported function output part")
		}
	}
	return values, nil
}
