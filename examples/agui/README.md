# AG-UI contact form example

This example serves a deterministic contact-form agent at `POST /ag-ui`. Its
in-memory store issues an authoritative thread ID, persists each accepted turn,
checks submitted actions against the latest stored A2UI widget, and replays an
identical `runId` without running the agent again.

Run `npm ci` and then `npm test` to build the TypeScript client and execute the
official `@ag-ui/client` round-trip proof. It starts and cleans up its own
loopback server. Run `go run .` separately to inspect the server at
`http://127.0.0.1:8080`.

The component catalog and complete form/result A2UI 0.9.1 payload live in
[`catalog.json`](catalog.json) and [`testdata/ui.json`](testdata/ui.json).
The catalog's child links use the protocol's shared reference types from
[`common_types.json`](common_types.json). Both schema files are served by the
demo and can be resolved locally without fetching a catalog URL.
This is a single-process illustration: it has no authentication, tenant
isolation, or durable storage.
