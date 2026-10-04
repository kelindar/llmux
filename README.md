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
| `POST /ag-ui` | AG-UI stored-turn profile | Requires `WithAGUI` and `WithStore` |

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

## AG-UI

AG-UI connects agents to interactive clients over HTTP/SSE. Enable the exact
`POST /ag-ui` route with a Store:

```go
llmux.New(catalog, llmux.WithAGUI(), llmux.WithStore(store))
```

The route is disabled by default. This stored-turn profile supports one new
user message or one UI action per request, and is tested with the official
`@ag-ui/client` and `@ag-ui/core` **1.0.1** SDKs. Authentication remains in your
middleware; Store and Agent implementations own application behavior.

### Sending a turn

```json
{
  "threadId": "client-creation-correlation",
  "runId": "request-identity",
  "messages": [{"id":"message-1","role":"user","content":"Make a form"}],
  "tools": [],
  "context": [],
  "state": {},
  "forwardedProps": {
    "llmux": {
      "target": "agent/example",
      "ui": {"format":"a2ui","version":"0.9.1","catalogId":"example/v1"}
    }
  }
}
```

`threadId`, `runId`, and target are required. `threadId` becomes
`TurnRequest.Thread`; the Store authorizes it and resolves initial creation
correlations. Return the canonical thread in
`Acceptance.Response.Metadata["thread_id"]`, which clients use on later turns.
Unknown existing threads must not create replacements. `runId` becomes
`TurnRequest.IdempotencyKey`; an HTTP `Idempotency-Key` header, if supplied,
must agree. Add `forwardedProps.llmux.previousResponseId` for an exact parent.

Text accepts a string or typed `{type:"text",text}` parts. Images use AG-UI's
`source` union for inline base64 data, URLs, or application file references,
then follow existing modality, size, and authorized asset-resolution checks.
The decoder does not fetch URLs or accept provider-specific file handles.

Clients must trim SDK display history before sending. Histories, privileged
messages, nonempty tools/context/state, nested runs, and interrupt/resume
options are rejected before Store acceptance. Empty SDK defaults are allowed.

### Submitting a UI action

Send `messages: []`, the exact `previousResponseId`, and this action inside
`forwardedProps.llmux`:

```json
{
  "action": {
    "name": "submit",
    "surfaceId": "form",
    "sourceComponentId": "submit",
    "context": {"name":"Alice","channel":"email"},
    "sourceResponseId": "response-1",
    "sourceItemId": "item-1"
  }
}
```

The adapter creates one `ItemExtension` whose Data has `kind:"ui_action"` and
the same fields. `sourceResponseId` must equal the exact parent. Store.Accept
must authorize the stored source widget, validate fields and choices against
its declared action, and reject stale or concurrent submissions. Deduplicate
request identities under the acceptance lock before checking stale parents;
changed requests with a reused identity must conflict. Actions are ordinary
new turns. Approval decisions remain application-owned operations.

### Returning complete UI

Declare `Info.Continuation=true`. UI-enabled agents also declare
`Info.Extensions["x-ui"]=true`; client catalog support arrives in
`Request.Controls.Extensions["x-ui"]`. Other adapters reject these agents
before execution. Text-only agents can use AG-UI without UI catalog support.

Emit an `ItemExtension` with this Data shape:

```json
{
  "kind": "ui",
  "format": "a2ui",
  "version": "0.9.1",
  "catalogId": "example/v1",
  "payload": [
    {"version":"v0.9.1","createSurface":{"surfaceId":"form","catalogId":"example/v1"}},
    {"version":"v0.9.1","updateComponents":{"surfaceId":"form","components":[{"id":"root","component":"Text","text":"Hello"}]}},
    {"version":"v0.9.1","updateDataModel":{"surfaceId":"form","value":{}}}
  ]
}
```

Each item carries one complete A2UI 0.9.1 surface, including components and a
data model. Cross-turn patches, surface deletion, and client data-model
synchronization are outside this profile. llmux checks transport shape and
the requested catalog before retention. Applications validate component
schemas, roots, references, bindings, cycles, count/depth, and declared
actions before emission. Catalog schemas remain source-managed application
artifacts. UI is bounded to 256 KiB per item; actions to 64 KiB, in addition
to configured request, output, and event limits.

### Persistence and events

Every new acceptance must supply `Finish`. Rich items publish only after it
successfully persists the finalized response. `Acceptance.Agent` may own
effective stored history; otherwise the ordinary Store.Load merge applies.
Returning terminal `Replay` skips Agent.Run and Finish, preserving saved item
IDs. Active Replay returns HTTP 409 `run_in_progress` with the response ID and
authorized metadata. Store errors may supply the same references through
`chat.Error.Metadata`; other protocols omit that field.

| Event | Meaning |
| --- | --- |
| `RUN_STARTED` | Request run ID and authoritative thread ID |
| `TEXT_MESSAGE_START` / `TEXT_MESSAGE_CONTENT` / `TEXT_MESSAGE_END` | Normalized assistant text with stable item IDs |
| `ACTIVITY_SNAPSHOT` | Authorized activity, enabled by `Acceptance.Activity`; sanitize content in the application |
| `CUSTOM`, name `llmux.item` | Persisted UI value `{id,format,version,catalogId,payload}` |
| `RUN_FINISHED` | Completed work, with `result:{responseId,metadata}` |
| `RUN_ERROR` | Failed, cancelled, or incomplete work, including persistence errors |

Clients replace a repeated rich item by ID and enable its controls only after
successful `RUN_FINISHED`. Invalid UI never enters saved partial output.
There is no `[DONE]` marker. Positive `Acceptance.RunTimeout` allows bounded
execution to continue after delivery disconnects. Pending approvals remain
pending, and explicit Stop stays application-owned. Metadata and retrieval
endpoints expose only application-authorized references; IDs do not grant
access.

The [form/action example](examples/agui) includes a small catalog, a Store,
and an official-client proof covering replay, source validation, stale actions,
and subsequent turns. Run `npm ci` and `npm test` there. Measure transport
cost with `go -C bench test -run '^$' -bench '^BenchmarkTransport$' -benchmem`.
See the [AG-UI 1.0 specification](https://docs.ag-ui.com/spec/1.0/index.md) and
[A2UI 0.9.1 protocol](https://github.com/a2ui-project/a2ui/blob/main/specification/v0_9_1/docs/a2ui_protocol.md)
for their full protocols.

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
