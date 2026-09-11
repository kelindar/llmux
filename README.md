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
an LLM proxy: your `Agent` runs directly in the process and does not need an
upstream HTTP API.

The root package is the convenient application entry point. The canonical
agent contract is also available as `github.com/kelindar/llmux/chat`; protocol
code and execution machinery are kept under `internal/`.

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

Mount `llmux.New(catalog)` in an existing `net/http` server to retain the
application's authentication middleware and request context. The constructor
does not open a listener.

`Agent.Run` is called once per accepted request. All output is emitted through
the ordered, backpressure-aware `Emit` callback; returning from `Run` closes
emission. `Emit` is serial by contract, and a concurrent call returns
`chat.ErrConcurrentEmit`. Agents should stop when `Emit` returns an error.

Complete output is emitted with `emit.Text`, `emit.Tool`, and related helpers,
or by constructing events with `chat.Text` / `chat.Tool` and calling `emit(event)`.
Streaming uses `emit.Delta` plus `chat.TextDone(itemID)`, and
`chat.ToolStart` / `chat.ToolDelta` / `chat.ToolDone(callID)`.
Done events are closure signals; final text and tool arguments come from
execution's accumulated state.

```go
type Agent interface {
	Run(context.Context, *Request, Emit) (Outcome, error)
}

type Emit func(Event) error
```

These types live in `github.com/kelindar/llmux/chat`. The root `llmux`
package provides the HTTP handler, options, and routing. Audio services are
configured with `WithTranscriber` / `WithSpeaker` using types from
`github.com/kelindar/llmux/audio`.

The handler owns protocol indexes, event completion, and response envelopes.
By default it also assigns response IDs. With a `Store`, the application
supplies identity and persistence; see below.

## Endpoints

The handler matches exact paths. Mount it under a prefix with
`http.StripPrefix` (for example `/v1` or `/api/v1`).

| Path | Protocol | Supported surface |
| --- | --- | --- |
| `POST /chat/completions` | OpenAI Chat Completions | Text, image/file/audio input, text/audio output, client function calls, structured output, serial streaming |
| `POST /responses` | Open Responses with an OpenAI Responses compatibility profile | Text/image/file/function/reasoning input, text/function/reasoning/generated-image output, typed OpenAI-style streaming |
| `POST /messages` | Anthropic Messages | Text/image/document/file input, tool use/results, Anthropic message streaming |
| `GET /models` | OpenAI model catalog shape | Projects `Catalog.List` keys as model IDs; an empty list when Catalog is nil or List is empty |
| `POST /audio/transcriptions` | OpenAI-compatible audio transcription | Enabled only with `WithTranscriber` |
| `POST /audio/speech` | OpenAI-compatible speech | Enabled only with `WithSpeaker`; binary audio or typed audio SSE |
| `POST /mcp` | MCP stateless Streamable HTTP | Enabled only with `WithMCP`; see "MCP endpoint" below |

Typical mount for OpenAI-compatible clients:

```go
mux.Handle("/v1/", http.StripPrefix("/v1", llmux.New(resolver)))
```

Unsupported recognized features return a protocol error instead of being
silently ignored. The Responses route is intentionally a documented subset,
not a claim of complete OpenAI Responses compatibility. Background execution,
hosted tools, log probabilities, WebSocket/WebRTC realtime audio, and generic
Responses audio are rejected.

The supported stream terminators differ by protocol: Chat Completions and
Responses use `data: [DONE]`; Anthropic uses `message_stop`; speech SSE uses
`speech.audio.done` and does not append `[DONE]`.

## Info and tools

The resolver returns both an agent and its declarations:

```go
chat.Info{
	InputModalities:  chat.ModalityText | chat.ModalityImage,
	OutputModalities: chat.ModalityText | chat.ModalityImage,
	Tools:            true,
	ClientTools:      true,
	Description:      "Draws charts from tabular input.",
	Tool:             "draw_chart", // nonempty exposes this agent through /mcp
	Created:          1735689600,
	OwnedBy:          "acme-agents",
}
```

An image-producing Responses agent also sets `ImageGeneration: true`; the
request must carry the compatibility profile's `image_generation` tool.

The zero Info value means text in/text out with ordinary generation
controls. `Description` is the single human-readable summary: llmux surfaces
it as the MCP tool description when `Tool` is nonempty. `Tool` is the stable
public MCP tool name (1–128 characters from `[A-Za-z0-9._-]`); empty means
the agent is listed only in `/models`, never derived by sanitizing the
Load target. `Created` and `OwnedBy` flow into the `/models` envelope.
Info checks cover wire support, the selected agent, and configured services.
`ClientTools` is required before a canonical function call can be handed to
a client. Tools used internally by an agent never become client tool calls
automatically.

Continuation is application-owned. Configure `WithStore` so `Store.Load`
returns history for Responses `previous_response_id` and `Store.Accept`
handles identity, replay, and `Finish`. Merely supplying a Store does not
change retention defaults — use `WithStoreDefault` when needed. Chat
Completions and Anthropic continuation fields are rejected because they have
no equivalent mapping in this compatibility profile.

## Application-owned response lifecycle

Applications with their own durable execution and response storage take
control through `llmux.Store`:

```go
llmux.WithStore(store)

type Store interface {
	Load(context.Context, string) ([]chat.Item, error)
	Accept(context.Context, *chat.TurnRequest) (chat.Acceptance, error)
}

type Acceptance struct {
	Response   Response  // identity for new work; ignored when Replay is set
	Replay     *Response // when set, skip Agent.Run and encode this result
	RunTimeout time.Duration // 0=request ctx; >0=detach+bound; <0=rejected
	Activity   bool
	Finish     func(context.Context, *Response, error) error
}
```

Ordering when a Store is configured:

1. Validate the request, resolve effective store policy, load continuation.
2. `Accept` — reserve identity, reject conflicts, or return an idempotent
   `*Response` replay. Capture request-local resources (idempotency
   key) on `Acceptance.Finish` instead of reconnecting through global maps.
3. `Agent.Run` (skipped on replay). `RunTimeout` of zero follows the HTTP
   request context. A positive duration detaches client cancellation,
   preserves context values, and bounds execution; llmux owns cancel and
   releases it on every exit. Delivery failures after disconnect do not
   cancel detached execution. A negative duration is rejected and Finish
   still runs so reserved resources cannot leak.
4. `Finish` — exactly once for every accepted execution, with the
   execution context. That context may already be cancelled. Finish owns
   any detached, bounded cleanup work, for example
   `context.WithTimeout(context.WithoutCancel(ctx), timeout)`. Not called
   for Accept errors, completed replays, or when Finish is nil.
   The `Response` passed to Finish shares nested data with the response
   encoded after Finish returns: treat it as read-only and call
   `Response.Clone()` before retaining or modifying it. Operational Go
   errors are passed as the Finish error argument and are never copied
   into public error text automatically.
5. Advertise successful completion only after Finish succeeds.

`Response` is the shared client-visible payload for creation,
finalization, replay, and `ResponsesBody` retrieval: identity, timestamps,
status, output, usage, sanitized public error, incomplete reason,
metadata, effective store, and retrieval fields such as target and
previous response ID.

Without a Store, llmux keeps generating response IDs and timestamps
internally. Effective retention (`TurnRequest.Retain`) requires a Store.
`WithStoreDefault` sets the policy when `store` is omitted (default false).
Explicit `store:true` / `store:false` always win. Applications that cannot
honor `store:false` should reject in Accept.

`TurnRequest.Turn` is this request's input. `Request.Input` is the effective
conversation for the agent. Persist `Turn` plus `Response.Output` with a
parent reference — do not rewrite full history each turn or retain the
accumulated `Input` slice.

Metadata from the accepted request is cloned into `Response` and reused
for replay/retrieval; a retry's metadata does not replace the original.

Mount under an application prefix with `http.StripPrefix` (for example
`mux.Handle("/api/v1/", http.StripPrefix("/api/v1", handler))`). Use
`ResponsesBody` for GET retrieval so creation, replay, and retrieval share one
encoder.

### Activity events

Enable per request with `Acceptance.Activity` after inspecting application
headers or extensions. Emit with `emit.Activity(name, json)` — Responses
streams `response.activity.<name>` with a bounded JSON payload. Activity is
never assistant output or continuation history, and is not translated to Chat
or Anthropic. Keep application-specific response fields outside the standard
envelope rather than mutating protocol objects.

See [`examples/lifecycle`](examples/lifecycle) for a minimal Accept / Finish
sketch, and [`examples/lifecycle-full`](examples/lifecycle-full) for
idempotency, continuation, retrieval, and `RunTimeout` durable execution.
Cleanup after cancellation starts inside Finish via
`context.WithTimeout(context.WithoutCancel(ctx), …)`.

## Unified catalog and MCP endpoint

`Catalog` is required construction: `List` feeds `GET /models` and MCP
discovery; `Load` obtains agents for invocation. Keys are Load targets.
Listing visibility never replaces Load authorization. llmux reads List maps
and never mutates them. A nil Catalog projects empty listings and fails Load
clearly. Execution never calls List merely to obtain capabilities.

```go
handler := llmux.New(catalog,
	llmux.WithStore(store),          // optional persistence
	llmux.WithStoreDefault(true),    // optional retention default
	llmux.WithMCP(),                 // optional; nil/empty List → no tools
)
```

`WithMCP` enables the exact path `/mcp` over **stateless Streamable HTTP**
(protocol revision **2026-07-28**, official Go SDK
`github.com/modelcontextprotocol/go-sdk`). Option ordering does not matter:
the transport is built after all options apply. Without `WithMCP`, chat-only
applications are unchanged. Applications own any prefix mounting.

Supported MCP methods:

- `server/discover` — handled by the SDK; advertises the 2026-07-28 revision,
  server identity, and capabilities.
- `tools/list` — `Catalog.List` entries whose `Info.Tool` is nonempty.
- `tools/call` — maps the tool name to its catalog target, then `Catalog.Load`.

`Transport` notes: no session storage and no sticky sessions —
`Mcp-Session-Id` is neither read nor issued, `GET`/`DELETE` return 405, and
tool results are returned as one `application/json` body. Requests
identifying older revisions are rejected; legacy `initialize` handshakes
cannot proceed against this endpoint.

MCP projection uses `Info.Tool` as the public name and `Info.Description` as
its description. Tool names must be unique and valid
(`[A-Za-z0-9._-]`, 1–128 characters) — duplicates and invalid names are
rejected (logged, HTTP 500), never silently overwritten. Listings are
returned in sorted name order (`/models` by ID; MCP by tool name). Every
cacheable MCP result (`server/discover`, `tools/list`) is marked
`cacheScope: "private"` with a zero TTL so caller-specific catalogs cannot
leak across identities.

Fixed tool input: every agent tool accepts exactly one JSON object argument,
validated strictly against an internal schema (the schema is not
configurable, and no JSON-to-prompt conversion exists):

```json
{
	"type": "object",
	"properties": { "message": { "type": "string" } },
	"required": ["message"],
	"additionalProperties": false
}
```

The message becomes one canonical user-message item, and the invocation runs
through the same machinery as the chat endpoints: authenticated context,
Catalog.Load authorization, Info validation, request/output limits, cancellation
and bounded durable execution (`Acceptance.RunTimeout`), Store.Accept,
exactly-once `Finish`, and persistence before a successful tool result.
Replays returned by Store.Accept skip execution and map the stored response
like any other. JSON-RPC request IDs are never treated as idempotency keys;
MCP requests carry no `Idempotency-Key`, so replay behavior is owned entirely
by the application Store. A listed tool is not an authorization grant:
invocation independently loads and executes the target.

Output mapping (completed tool results; ordering preserved):

| Agent output | MCP result |
| --- | --- |
| Assistant text (message/media items, streamed or complete) | `TextContent`, in emission order |
| Image, audio, or file output | Rejected explicitly as a tool error (execution already rejects non-text modalities for MCP requests) |
| Internal agent function calls | Rejected explicitly; never surfaced as client tool requests |
| Reasoning summaries, activity events | Rejected explicitly (not enabled/mappable for MCP requests) |
| Truncated output (`incomplete`) | Partial text plus an explicit incompleteness notice, `IsError: true` |
| Execution failures, cancellations, Finish failures | Tool-execution errors (`IsError: true`) with sanitized messages; raw operational errors go to `WithErrorLog` |

Malformed requests — unknown tools, invalid arguments, oversized inputs — are
protocol-level JSON-RPC errors (`-32602`), not tool results. Tasks, sampling,
elicitation, approval-resumption flows, and streaming tool results are not
implemented or advertised in this version.

### OAuth responsibilities of the host

llmux implements no OAuth server, token store, or token-validation policy.
Stateless MCP is designed to sit behind the host application's middleware:

- Wrap `/mcp` (or the mounted prefix) with authentication middleware; llmux
  preserves request context through listing, resolution, acceptance,
  execution, and Finish, and preserves application-generated `401` responses
  and `WWW-Authenticate` challenges.
- The host owns token audience/resource validation, scope enforcement,
  authorization-server discovery, and RFC 9728 protected-resource metadata
  routes. Do not add permissive CORS; the transport's Origin/dns-rebinding
  validation stays enabled.
- Listing and invocation each perform their own authorization checks under
  the same authenticated identity.

See [`examples/mcp`](examples/mcp) for an agent, unified catalog,
authenticated listing, optional MCP enablement, prefix mounting, and an
illustrative (deliberately not production) authentication wrapper.

## Media and audio

`Media` preserves one of three sources: inline bytes, an HTTP(S) URL, or an
application-owned asset reference. URL descriptors are never fetched during
decoding. Supply `WithAssetResolver` when the application wants authorized,
bounded resolution of URLs or asset references.

The canonical contract keeps MIME type, filename, format, and bytes separate.
It does not describe images or transcribe audio. Native mappings are explicit:

- Chat Completions accepts image/file/audio input and audio output.
- Responses accepts image/file input and generated-image output represented as
  `image_generation_call` only when the request includes the explicitly
  supported `{"type":"image_generation"}` tool and the resolver declares
  `ImageGeneration: true`; it does not invent a generic audio item.
- Anthropic accepts image/document/file input but has no audio mapping in this
  library.
- Transcription uses bounded multipart input and `json`, `text`, or
  `verbose_json` output.
- Speech uses JSON input and binary output, or typed
  `speech.audio.delta`/`speech.audio.done` SSE.

The public audio contracts are in `github.com/kelindar/llmux/audio`, with root
aliases for convenience. See [`examples/multimodal`](examples/multimodal) for
a deterministic media-aware agent.

## Limits and errors

Zero-valued `Limits` use these defaults:

- request JSON: 8 MiB
- each inline media value: 16 MiB
- resolved assets per request: 32
- accumulated agent output: 8 MiB
- one SSE event: 1 MiB
- multipart request: 32 MiB

Set limits with `WithLimits`. Requests are bounded before decoding, output is
bounded while events are accumulated, and cancellation propagates through the
agent and asset resolver. Client errors are sanitized; `WithErrorLog` can
observe operational failures without putting raw backend errors in responses.

Streaming validates the request and capabilities before committing headers.
Failures before the first event are ordinary protocol error responses. After
headers are committed, codecs emit their protocol-specific stream error and
terminal lifecycle event rather than appending a JSON body.
