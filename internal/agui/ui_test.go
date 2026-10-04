// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package agui

import (
	"encoding/json/jsontext"
	"strings"
	"testing"

	"github.com/kelindar/llmux/chat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const validUIData = `{"kind":"ui","format":"a2ui","version":"0.9.1","catalogId":"catalog-1","payload":[{"version":"v0.9.1","createSurface":{"surfaceId":"surface-1","catalogId":"catalog-1"}},{"version":"v0.9.1","updateComponents":{"surfaceId":"surface-1","components":[{"id":"root","component":{"Text":{"text":"hello"}}}]}},{"version":"v0.9.1","updateDataModel":{"surfaceId":"surface-1","path":"/","value":{"title":"hello"}}}]}`

func TestValidateUI(t *testing.T) {
	item := chat.Item{Type: chat.ItemExtension, Data: jsontext.Value(validUIData)}
	require.NoError(t, (Adapter{}).ValidateEvent(chat.Event{Type: chat.EventItem, Item: item}))
	_, err := parseUIItem(item)
	require.NoError(t, err)
}

func TestValidateUIRejects(t *testing.T) {
	components := `{"version":"v0.9.1","updateComponents":{"surfaceId":"surface-1","components":[{}]}}`
	model := `{"version":"v0.9.1","updateDataModel":{"surfaceId":"surface-1","value":{}}}`
	create := `{"version":"v0.9.1","createSurface":{"surfaceId":"surface-1","catalogId":"catalog-1"}}`
	cases := map[string]string{
		"wrongKind":       `{"kind":"ui_action","format":"a2ui","version":"0.9.1","catalogId":"catalog-1","payload":[]}`,
		"wrongVersion":    `{"kind":"ui","format":"a2ui","version":"1.0","catalogId":"catalog-1","payload":[]}`,
		"wrongCatalog":    `{"kind":"ui","format":"a2ui","version":"0.9.1","catalogId":"catalog-1","payload":[{"version":"v0.9.1","createSurface":{"surfaceId":"surface-1","catalogId":"other"}},{"version":"v0.9.1","updateComponents":{"surfaceId":"surface-1","components":[{}]}},{"version":"v0.9.1","updateDataModel":{"surfaceId":"surface-1","path":"/","value":{}}}]}`,
		"noComponents":    `{"kind":"ui","format":"a2ui","version":"0.9.1","catalogId":"catalog-1","payload":[{"version":"v0.9.1","createSurface":{"surfaceId":"surface-1","catalogId":"catalog-1"}},{"version":"v0.9.1","updateDataModel":{"surfaceId":"surface-1","path":"/","value":{}}}]}`,
		"noModelValue":    `{"kind":"ui","format":"a2ui","version":"0.9.1","catalogId":"catalog-1","payload":[{"version":"v0.9.1","createSurface":{"surfaceId":"surface-1","catalogId":"catalog-1"}},{"version":"v0.9.1","updateComponents":{"surfaceId":"surface-1","components":[{}]}},{"version":"v0.9.1","updateDataModel":{"surfaceId":"surface-1","path":"/"}}]}`,
		"duplicateKey":    `{"kind":"ui","kind":"ui","format":"a2ui","version":"0.9.1","catalogId":"catalog-1","payload":[]}`,
		"unknownOutput":   `{"kind":"ui","format":"a2ui","version":"0.9.1","catalogId":"catalog-1","other":1,"payload":[]}`,
		"payloadNotArray": `{"kind":"ui","format":"a2ui","version":"0.9.1","catalogId":"catalog-1","payload":{}}`,
		"noCreate":        uiData(components, model),
		"wrongOrder":      uiData(components, create, model),
		"deleteSurface":   uiData(create, `{"version":"v0.9.1","deleteSurface":{"surfaceId":"surface-1"}}`, components, model),
		"secondCreate":    uiData(create, components, create, model),
		"wrongSurface":    uiData(create, strings.ReplaceAll(components, "surface-1", "other"), model),
		"sendDataModel":   uiData(strings.Replace(create, `"catalogId":"catalog-1"`, `"catalogId":"catalog-1","sendDataModel":true`, 1), components, model),
		"sendDataType":    uiData(strings.Replace(create, `"catalogId":"catalog-1"`, `"catalogId":"catalog-1","sendDataModel":"yes"`, 1), components, model),
		"emptyComponents": uiData(create, `{"version":"v0.9.1","updateComponents":{"surfaceId":"surface-1","components":[]}}`, model),
		"badPath":         uiData(create, components, strings.Replace(model, `"value":{}`, `"path":1,"value":{}`, 1)),
		"missingSurface":  uiData(`{"version":"v0.9.1","createSurface":{"catalogId":"catalog-1"}}`, components, model),
		"badEnvelope":     uiData(`1`, components, model),
		"badVersion":      uiData(strings.Replace(create, `"version":"v0.9.1"`, `"version":1`, 1), components, model),
		"unknownEnvelope": uiData(strings.Replace(create, `,"createSurface":`, `,"other":true,"createSurface":`, 1), components, model),
		"twoMessages":     uiData(strings.Replace(create, `"catalogId":"catalog-1"`, `"catalogId":"catalog-1"},"updateComponents":{"surfaceId":"surface-1","components":[{}]`, 1), components, model),
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			item := chat.Item{Type: chat.ItemExtension, ID: "ui-1", Data: jsontext.Value(data)}
			assert.Error(t, (Adapter{}).ValidateEvent(chat.Event{Type: chat.EventItem, Item: item}))
		})
	}
}

func uiData(payload ...string) string {
	return `{"kind":"ui","format":"a2ui","version":"0.9.1","catalogId":"catalog-1","payload":[` + strings.Join(payload, ",") + `]}`
}

func TestMatchesSupport(t *testing.T) {
	item := chat.Item{Type: chat.ItemExtension, ID: "ui-1", Data: jsontext.Value(validUIData)}
	good := jsontext.Value(`{"format":"a2ui","version":"0.9.1","catalogId":"catalog-1"}`)
	bad := jsontext.Value(`{"format":"a2ui","version":"0.9.1","catalogId":"other"}`)
	assert.NoError(t, MatchesSupport(item, good))
	assert.Error(t, MatchesSupport(item, bad))
}
