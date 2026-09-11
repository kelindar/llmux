// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package fixtures

import (
	"net/http/httptest"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
)

func anthropicClient(server *httptest.Server) anthropic.Client {
	return anthropic.NewClient(
		anthropicoption.WithoutEnvironmentDefaults(),
		anthropicoption.WithAPIKey("test"),
		anthropicoption.WithBaseURL(server.URL+"/"),
		anthropicoption.WithMaxRetries(0),
	)
}

func lookupToolAnthropic() anthropic.ToolUnionParam {
	return anthropic.ToolUnionParamOfTool(anthropic.ToolInputSchemaParam{Type: "object"}, "lookup")
}
