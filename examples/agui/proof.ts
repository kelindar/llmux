import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import type { Message } from "@ag-ui/client";
import { TurnAgent } from "./client.js";

const catalogId = "https://example.invalid/llmux/agui/catalog/v1";
const fixture = JSON.parse(readFileSync("testdata/ui.json", "utf8")) as {
  form: UIItem;
  result: UIItem;
};

interface UIItem {
  id?: string;
  kind: string;
  format: string;
  version: string;
  catalogId: string;
  payload: Array<Record<string, any>>;
}

interface Action {
  name: string;
  surfaceId: string;
  sourceComponentId: string;
  context: Record<string, string>;
  sourceResponseId: string;
  sourceItemId: string;
}

interface ForwardedProps {
  llmux: {
    target: string;
    previousResponseId?: string;
    ui: { format: "a2ui"; version: "0.9.1"; catalogId: string };
    action?: Action;
  };
}

interface RunSnapshot {
  threadId: string;
  responseId: string;
  metadata: Record<string, string>;
  items: UIItem[];
  events: unknown[];
}

interface State {
  agentRuns: number;
  actionEffects: number;
  storedResponses: number;
}

async function main(): Promise<void> {
  assert.equal(fixture.form.catalogId, catalogId);
  assert.equal(fixture.result.catalogId, catalogId);

  const port = await freePort();
  const server = spawn("go", ["run", "."], {
    cwd: process.cwd(),
    env: { ...process.env, GOCACHE: process.env.GOCACHE ?? join(tmpdir(), "llmux-agui-go-cache"), PORT: String(port) },
    windowsHide: true,
    detached: process.platform !== "win32",
    stdio: ["ignore", "ignore", "pipe"],
  });
  let spawnError: Error | undefined;
  let serverLog = "";
  server.on("error", (err) => {
    spawnError = err;
  });
  server.stderr?.on("data", (chunk: Buffer) => {
    serverLog = `${serverLog}${chunk.toString()}`.slice(-4_000);
  });
  const exited = new Promise<void>((resolve) => server.once("exit", () => resolve()));
  const baseURL = `http://127.0.0.1:${port}`;

  try {
    await waitReady(server, baseURL, () => spawnError, () => serverLog);
    await runProof(baseURL);
    process.stdout.write("AG-UI SDK proof passed: form, action, replay, validation, and stale-parent checks.\n");
  } finally {
  await stopServer(server, baseURL, exited);
  }
}

async function runProof(baseURL: string): Promise<void> {
  const correlationId = "client-create-correlation";
  const agent = new TurnAgent({ url: `${baseURL}/ag-ui`, threadId: correlationId });
  const firstMessage: Message = {
    id: "message-1",
    role: "user",
    content: "Help me send a contact request.",
  };

  agent.addMessage(firstMessage);
  agent.setTurn([firstMessage]);
  const formRunId = "form-run-1";
  const form = await runTurn(agent, formRunId, props());
  assert.notEqual(form.threadId, correlationId, "RUN_STARTED must return the store's authoritative thread ID");
  assert.equal(form.metadata.thread_id, form.threadId);
  assert.equal(form.items.length, 1);
  assert.equal(form.items[0].format, "a2ui");
  assert.equal(form.items[0].version, "0.9.1");
  assert.deepEqual(form.items[0].payload, fixture.form.payload);
  agent.threadId = form.threadId;

  const stateAfterForm = await readState(baseURL);
  assert.deepEqual(stateAfterForm, { agentRuns: 1, actionEffects: 0, storedResponses: 1 });

  const formMessage = form.items[0];
  const componentUpdate = formMessage.payload.find((message) => message.updateComponents)?.updateComponents;
  assert.ok(componentUpdate, "form has an updateComponents message");
  const submit = componentUpdate.components.find((component: any) => component.id === "submit");
  assert.ok(submit?.action?.event, "form has the submit action");
  const action: Action = {
    name: submit.action.event.name,
    surfaceId: componentUpdate.surfaceId,
    sourceComponentId: submit.id,
    context: { name: "Ada Lovelace", email: "ada@example.test" },
    sourceResponseId: form.responseId,
    sourceItemId: formMessage.id ?? "",
  };
  assert.equal(action.name, "submit_contact");
  assert.equal(action.sourceComponentId, "submit");
  assert.equal(action.sourceResponseId, form.responseId);
  assert.ok(action.sourceItemId);

  const forged = { ...action, sourceComponentId: "title" };
  await expectError(baseURL, request(form.threadId, "forged-run-1", [], form.responseId, forged), 400, "invalid_request");
  assert.deepEqual(await readState(baseURL), stateAfterForm, "rejected widget actions do not execute or persist");

  const actionRunId = "action-run-1";
  const actionProps = props(form.responseId, action);
  agent.setTurn([]);
  const result = await runTurn(agent, actionRunId, actionProps);
  assert.equal(result.threadId, form.threadId);
  assert.equal(result.metadata.thread_id, form.threadId);
  assert.equal(result.items.length, 1);
  assert.deepEqual(result.items[0].payload, fixture.result.payload);

  const stateAfterAction = await readState(baseURL);
  assert.deepEqual(stateAfterAction, { agentRuns: 2, actionEffects: 1, storedResponses: 2 });

  // The exact same SDK request replays its saved Response after the parent has advanced.
  agent.setTurn([]);
  const replay = await runTurn(agent, actionRunId, actionProps);
  assert.equal(replay.responseId, result.responseId);
  assert.equal(replay.items[0].id, result.items[0].id);
  assert.deepEqual(await readState(baseURL), stateAfterAction, "replay performs no agent call or action effect");

  const changed = { ...action, context: { ...action.context, email: "changed@example.test" } };
  await expectError(baseURL, request(form.threadId, actionRunId, [], form.responseId, changed), 409, "idempotency_conflict");
  await expectError(baseURL, request(form.threadId, "stale-run-1", [], form.responseId, action), 409, "stale_parent");
  assert.deepEqual(await readState(baseURL), stateAfterAction, "conflicts perform no agent call or action effect");
}

async function runTurn(agent: TurnAgent, runId: string, forwardedProps: ForwardedProps): Promise<RunSnapshot> {
  const snapshot: RunSnapshot = { threadId: "", responseId: "", metadata: {}, items: [], events: [] };
  await agent.runAgent({ runId, forwardedProps }, {
    onEvent({ event }) {
      snapshot.events.push(event);
    },
    onRunStartedEvent({ event }) {
      snapshot.threadId = event.threadId;
    },
    onCustomEvent({ event }) {
      if (event.name === "llmux.item") snapshot.items.push(event.value as UIItem);
    },
    onRunFinishedEvent(params) {
      if (params.outcome !== "success" || !params.result || typeof params.result !== "object") return;
      const response = params.result as { responseId?: string; metadata?: Record<string, string> };
      snapshot.responseId = response.responseId ?? "";
      snapshot.metadata = response.metadata ?? {};
    },
  });
  assert.ok(snapshot.threadId, "RUN_STARTED includes threadId");
  assert.ok(snapshot.responseId, `RUN_FINISHED result includes responseId: ${JSON.stringify(snapshot.events)}`);
  return snapshot;
}

function props(previousResponseId?: string, action?: Action): ForwardedProps {
  return {
    llmux: {
      target: "contact",
      ...(previousResponseId ? { previousResponseId } : {}),
      ui: { format: "a2ui", version: "0.9.1", catalogId },
      ...(action ? { action } : {}),
    },
  };
}

function request(
  threadId: string,
  runId: string,
  messages: Message[],
  previousResponseId: string,
  action: Action,
): Record<string, unknown> {
  return {
    threadId,
    runId,
    messages,
    forwardedProps: props(previousResponseId, action),
  };
}

async function expectError(
  baseURL: string,
  body: Record<string, unknown>,
  status: number,
  code: string,
): Promise<void> {
  const runId = body.runId as string;
  const response = await fetch(`${baseURL}/ag-ui`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Idempotency-Key": runId },
    body: JSON.stringify(body),
  });
  const bodyText = await response.text();
  assert.equal(response.status, status, bodyText);
  const error = JSON.parse(bodyText) as { error?: { code?: string } };
  assert.equal(error.error?.code, code);
}

async function readState(baseURL: string): Promise<State> {
  const response = await fetch(`${baseURL}/debug/state`);
  assert.equal(response.status, 200);
  return await response.json() as State;
}

async function freePort(): Promise<number> {
  const listener = createServer();
  await new Promise<void>((resolve, reject) => {
    listener.once("error", reject);
    listener.listen(0, "127.0.0.1", resolve);
  });
  const address = listener.address();
  assert.ok(address && typeof address === "object");
  const port = address.port;
  await new Promise<void>((resolve, reject) => listener.close((err) => err ? reject(err) : resolve()));
  return port;
}

async function waitReady(
  server: ReturnType<typeof spawn>,
  baseURL: string,
  spawnFailure: () => Error | undefined,
  serverOutput: () => string,
): Promise<void> {
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    const error = spawnFailure();
    if (error) throw error;
    if (server.exitCode !== null) throw new Error(`example server exited with code ${server.exitCode}: ${serverOutput()}`);
    try {
      if ((await fetch(`${baseURL}/healthz`)).status === 204) return;
    } catch {
      // The listener may still be binding while go run compiles the example.
    }
    await delay(75);
  }
  throw new Error(`example server did not become ready within 120 seconds: ${serverOutput()}`);
}

async function stopServer(server: ReturnType<typeof spawn>, baseURL: string, exited: Promise<void>): Promise<void> {
  if (server.exitCode !== null || server.signalCode !== null || !server.pid) return;

  try {
    await fetch(`${baseURL}/debug/shutdown`, { method: "POST" });
  } catch {
    // Forceful cleanup below handles startup failures before the route is ready.
  }

  const gracefullyClosed = await Promise.race([exited.then(() => true), delay(3_000).then(() => false)]);
  if (gracefullyClosed) return;

  if (process.platform === "win32") {
    const killer = spawn("taskkill", ["/PID", String(server.pid), "/T", "/F"], {
      windowsHide: true,
      stdio: "ignore",
    });
    await new Promise<void>((resolve) => {
      killer.once("error", () => resolve());
      killer.once("exit", () => resolve());
    });
  } else {
    try {
      process.kill(-server.pid, "SIGTERM");
    } catch {
      server.kill("SIGTERM");
    }
  }

  const closed = await Promise.race([exited.then(() => true), delay(3_000).then(() => false)]);
  if (!closed) {
    try {
      process.kill(process.platform === "win32" ? server.pid : -server.pid, "SIGKILL");
    } catch {
      server.kill("SIGKILL");
    }
    await exited;
  }
}

void main().catch((err: unknown) => {
  process.stderr.write(`${err instanceof Error ? err.stack : String(err)}\n`);
  process.exitCode = 1;
});
