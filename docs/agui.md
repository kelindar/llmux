# AG-UI stored-turn profile

Enable the exact `POST /ag-ui` route with `llmux.WithAGUI()` and
`llmux.WithStore(store)`. Authentication middleware remains application-owned.
The route is disabled by default. This profile uses JSON over HTTP/SSE and is
tested with `@ag-ui/client` and `@ag-ui/core` **1.0.1**. It supports the subset
below, not every AG-UI feature.

## Request binding

```json
{
  "threadId": "client-creation-correlation",
  "runId": "request-identity",
  "messages": [{"id": "message-1", "role": "user", "content": "Make a form"}],
  "tools": [],
  "context": [],
  "state": {},
  "forwardedProps": {
    "llmux": {
      "target": "agent/example",
      "ui": {"format": "a2ui", "version": "0.9.1", "catalogId": "example/v1"}
    }
  }
}
```

`threadId`, `runId`, and target are required. `previousResponseId` inside the
llmux section selects an exact parent for subsequent turns. `runId` becomes
`TurnRequest.IdempotencyKey`. If the HTTP `Idempotency-Key` header is present,
it must agree. `threadId` becomes `TurnRequest.Thread`, not an execution
permission. These fields never override the authenticated context.

One request carries one new user message, or one action with an empty messages
array. Text uses a string or `{type:"text",text}` parts. Image parts use the
AG-UI 1.0 `source` union for inline base64 data, URLs, or application file refs.
They become canonical media and follow existing modality, byte-limit, and
authorized asset-resolution checks. The adapter never fetches remote images.

Nonempty frontend tools, context, writable state, histories, privileged
messages, branching, and interrupt/resume options are rejected before Store
acceptance. Empty SDK defaults for tools, context, and state are allowed.
Client adapters must trim SDK display history before sending the request.

For an action, send no messages and include the exact parent plus:

```json
{
  "action": {
    "name": "submit",
    "surfaceId": "form",
    "sourceComponentId": "submit",
    "context": {"name": "Alice", "channel": "email"},
    "sourceResponseId": "response-1",
    "sourceItemId": "item-1"
  }
}
```

The action becomes one `ItemExtension` with `Data.kind="ui_action"` and the
same fields. The adapter checks shape and the 64 KiB action ceiling. Store
acceptance must authorize the source, validate fields and choices against the
stored widget, and reject stale or concurrent submissions. An action is an
ordinary new turn. It cannot resolve an approval or invoke an arbitrary URL.

## Application responsibilities

Declare `Info.Continuation=true`. A UI-enabled agent declares
`Info.Extensions["x-ui"]=true`; the adapter passes client UI support through
`Request.Controls.Extensions["x-ui"]`. Other adapters reject that agent before
execution because they cannot represent its UI. Agents without generated UI
can still use AG-UI for text and authorized activity.

The Store resolves initial thread correlations within the authenticated scope
and returns the canonical thread in `Acceptance.Response.Metadata["thread_id"]`.
Clients use that returned ID on later turns. Existing IDs and exact parents
must be authorized; unknown threads must not silently create replacements.
The Store owns deduplication under the same lock as turn acceptance and checks
identity before stale-parent rejection. The example demonstrates these rules
in memory; llmux adds no thread registry or transcript store.

Each new acceptance must provide `Finish`. It persists the outcome and is
called once. `Acceptance.Agent` can own effective stored history, bypassing
ordinary Store.Load history merging. Returning `Replay` skips Agent.Run and
Finish. An in-progress Replay returns HTTP 409 with `run_in_progress`, the
response ID, and authorized response metadata before SSE starts. Store errors
can supply `chat.Error.Metadata` for the same purpose. Changed requests with a
reused identity must conflict at Store acceptance.

Metadata is returned only from application acceptance or saved responses.
Use `thread_id`, `execution_id`, and other application-authorized references;
the terminal result also contains `responseId`. Saved state and retrieval
remain application endpoints. Resource identifiers never grant access.

## Output binding

Text maps to `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, and
`TEXT_MESSAGE_END` using stable canonical item IDs. Authorized activity maps
to `ACTIVITY_SNAPSHOT`; enable it with `Acceptance.Activity`. Applications
must sanitize activity content, including any trusted approval descriptor.
Approval decisions use the application's existing authorized operation and
continue the waiting execution. This adapter exposes no approval tokens.

Generate complete UI using `ItemExtension` with the following Data shape:

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

The profile carries a complete surface, not cross-turn patches or progressive
UI generation. UI is bounded to 256 KiB per item, additionally subject to the
configured output/event ceilings. llmux validates transport shape and declared
client format/catalog. Applications validate component schemas, references,
bindings, declared actions, roots, cycles, component count, and depth before
emission. Catalog schemas remain application-owned, source-managed artifacts.

Rich items are emitted after successful persistence as `CUSTOM` events named
`llmux.item`, with value `{id,format,version,catalogId,payload}`. Replayed
delivery uses the same ID; clients replace the item rather than appending
controls or firing actions. Invalid UI does not enter saved partial output.
Controls are enabled only after successful `RUN_FINISHED`.

Every delivery begins with `RUN_STARTED` carrying the request run ID and
authoritative thread ID. Completed work ends with `RUN_FINISHED` and
`result:{responseId,metadata}`. Failure, cancellation, or incomplete outcomes
produce `RUN_ERROR`, preserving their outcome rather than announcing success.
Persistence failures do not publish rich items. There is no OpenAI `[DONE]`
marker. Before SSE, errors use the canonical JSON error envelope; only AG-UI
includes optional error metadata.

With a positive `Acceptance.RunTimeout`, disconnect stops delivery while work
continues within the existing bounded execution context. Waiting approvals
remain pending; they do not produce success. Explicit Stop is an application
control. Terminal duplicate requests replay their saved outcomes without
executing tools again.

## Transport measurements

`go -C bench test -run '^$' -bench '^BenchmarkTransport$' -benchmem -count=5`
measures a deterministic one-turn workload. These are five-run medians on
Windows/amd64, Intel i7-13700K, Go 1.27.0. The AG-UI store is a no-op so the
figures isolate decoding, execution, and SSE delivery; application storage,
model calls, and rendering are outside the measurement.

| Workload | Before ns/op | After ns/op | Before B/op | After B/op | Before/after allocs/op |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing Responses text | 37,285 | 33,491 | 29,310 | 29,377 | 137 / 137 |
| Canonical UI execution | 5,109 | 3,964 | 3,606 | 3,606 | 22 / 22 |
| New AG-UI text | — | 14,634 | — | 17,356 | — / 116 |
| New AG-UI text + UI | — | 25,843 | — | 23,665 | — / 189 |

Timing varies with machine load. Existing allocation counts are unchanged;
the text transport's additional bytes include the larger parsed-request value.
Canonical UI execution and AG-UI HTTP delivery measure different scopes.

## References

- [AG-UI 1.0 specification](https://docs.ag-ui.com/spec/1.0/index.md)
- [AG-UI SDK](https://github.com/ag-ui-protocol/ag-ui)
- [A2UI 0.9.1 protocol](https://github.com/a2ui-project/a2ui/blob/main/specification/v0_9_1/docs/a2ui_protocol.md)
