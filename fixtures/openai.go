// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package fixtures

import (
	"net/http/httptest"

	"github.com/openai/openai-go/v3"
	openaioption "github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
)

func openaiClient(server *httptest.Server) openai.Client {
	return openai.NewClient(
		openaioption.WithAPIKey("test"),
		openaioption.WithBaseURL(server.URL+"/v1/"),
		openaioption.WithMaxRetries(0),
	)
}

func lookupToolOpenAI() openai.ChatCompletionToolUnionParam {
	return openai.ChatCompletionToolUnionParam{
		OfFunction: &openai.ChatCompletionFunctionToolParam{
			Function: shared.FunctionDefinitionParam{
				Name:       "lookup",
				Parameters: shared.FunctionParameters{"type": "object"},
			},
		},
	}
}
