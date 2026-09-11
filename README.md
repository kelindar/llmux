<p align="center">
    <img width="300" height="100" src=".github/logo.png" border="0" alt="kelindar/llmux">
    <br>
    <img src="https://img.shields.io/github/go-mod/go-version/kelindar/llmux" alt="Go Version">
    <a href="https://pkg.go.dev/github.com/kelindar/llmux"><img src="https://pkg.go.dev/badge/github.com/kelindar/llmux" alt="PkgGoDev"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
    <a href="https://coveralls.io/github/kelindar/llmux"><img src="https://coveralls.io/repos/github/kelindar/llmux/badge.svg" alt="Coverage"></a>
</p>

`llmux` is a small, embeddable Go HTTP handler for exposing application-owned
agent logic through standard AI client protocols. It is a protocol server, not
an LLM proxy: your `Agent` runs in-process and does not need an upstream HTTP
API.

## Quick start

```sh
go get github.com/kelindar/llmux
```

```go
import (
	"context"
	"net/http"

	"github.com/kelindar/llmux"
	"github.com/kelindar/llmux/chat"
)

type agents struct {
	echo chat.Agent
}

func (a *agents) List(context.Context) (map[string]chat.Info, error) {
	return map[string]chat.Info{"echo": {}}, nil
}

func (a *agents) Load(_ context.Context, target string) (chat.Agent, chat.Info, error) {
	if target != "echo" {
		return nil, chat.Info{}, chat.NotFound()
	}
	return a.echo, chat.Info{}, nil
}

mux := http.NewServeMux()
mux.Handle("/v1/", http.StripPrefix("/v1", llmux.New(&agents{
	echo: chat.AgentFunc(func(ctx context.Context, req *chat.Request, emit chat.Emit) (chat.Outcome, error) {
		return chat.Outcome{}, emit.Text("hello")
	}),
})))
http.ListenAndServe(":8080", mux)
```

Mount `llmux.New(catalog)` under your existing `net/http` server so auth
middleware and request context stay yours. The constructor does not open a
listener.

`Agent.Run` runs once per accepted request. All output goes through the serial
`Emit` callback (`chat.ErrConcurrentEmit` on concurrent calls). Stop when
`Emit` returns an error. Use `emit.Text`, `emit.Tool`, `emit.Delta`, and the
related helpers in `github.com/kelindar/llmux/chat`. Audio backends plug in
with `WithTranscriber` / `WithSpeaker` from `github.com/kelindar/llmux/audio`.

See [`examples/basic`](examples/basic).

## Endpoints

Paths are exact. Mount under a prefix with `http.StripPrefix` (for example
`/v1`).

| Path | Protocol | Notes |
| --- | --- | --- |
| `POST /chat/completions` | OpenAI Chat Completions | Text, media input, text/audio output, client tools, structured output, streaming |
| `POST /responses` | OpenAI Responses compatibility profile | Documented subset: text/media/tools/reasoning, generated images when requested |
| `POST /messages` | Anthropic Messages | Text/image/document input, tool use, streaming |
| `GET /models` | OpenAI models list | `Catalog.List` keys; empty when Catalog is nil |
| `POST /audio/transcriptions` | OpenAI transcription | Requires `WithTranscriber` |
| `POST /audio/speech` | OpenAI speech | Requires `WithSpeaker` |
| `POST /mcp` | MCP Streamable HTTP | Requires `WithMCP` |

Unsupported recognized features return protocol errors instead of being
ignored. Responses is a compatibility subset, not full OpenAI parity.

## Catalog and capabilities

`Catalog.List` feeds `/models` and MCP discovery. `Catalog.Load` obtains the
agent for each call. Listing visibility never replaces Load authorization. A
nil Catalog projects empty listings and fails Load clearly.

```go
chat.Info{
	InputModalities:  chat.ModalityText | chat.ModalityImage,
	OutputModalities: chat.ModalityText,
	Tools:            true,
	ClientTools:      true,
	Description:      "Draws charts from tabular input.",
	Tool:             "draw_chart", // nonempty exposes this agent on /mcp
	Created:          1735689600,
	OwnedBy:          "acme-agents",
}
```

Zero `Info` means text in/out with ordinary generation controls. Set
`ImageGeneration: true` for Responses image output (request must include the
`image_generation` tool). `ClientTools` is required before function calls are
handed to a client. Internal agent tools never become client tool calls on
their own.

## Persistence and lifecycle

Without a `Store`, llmux assigns response IDs and timestamps. With
`WithStore`, your app owns continuation (`previous_response_id`), identity,
replay, and `Finish`. Retention defaults stay off until `WithStoreDefault` or
an explicit `store` field says otherwise. Chat Completions and Anthropic have
no continuation mapping here.

```go
llmux.New(catalog,
	llmux.WithStore(store),
	llmux.WithStoreDefault(true),
)
```

Use `ResponsesBody` when you serve GET retrieval so create, replay, and
retrieve share one encoder. For Accept / Finish / `RunTimeout` details, see
[`examples/lifecycle`](examples/lifecycle) and
[`examples/lifecycle-full`](examples/lifecycle-full), plus the `llmux.Store` and
`chat.Acceptance` docs.

## MCP

`WithMCP` enables `POST /mcp` (stateless Streamable HTTP, revision
`2026-07-28`). Catalog entries with a nonempty `Info.Tool` become tools.
Each tool takes `{"message":"<string>"}` and runs through the same Load,
validation, limits, Store, and Finish path as the chat endpoints. Put your
auth middleware in front of `/mcp`; llmux does not implement OAuth.

See [`examples/mcp`](examples/mcp).

## Media and audio

`Media` carries inline bytes, an HTTP(S) URL, or an app-owned asset ref. URLs
are not fetched during decode; use `WithAssetResolver` for authorized
resolution. Protocol mappings differ: Chat Completions supports audio I/O,
Responses supports generated images (not generic audio items), Anthropic has
no audio mapping here.

See [`examples/multimodal`](examples/multimodal).

## Limits and errors

Zero `Limits` default to:

- request JSON: 8 MiB
- each inline media value: 16 MiB
- resolved assets per request: 32
- accumulated agent output: 8 MiB
- one SSE event: 1 MiB
- multipart request: 32 MiB

Override with `WithLimits`. Client-facing errors are sanitized;
`WithErrorLog` observes operational failures without leaking them into
responses. Streaming failures before the first event are ordinary error
bodies; after headers are committed, codecs emit their protocol stream error.
