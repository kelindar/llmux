import { HttpAgent, type Message, type RunAgentInput } from "@ag-ui/client";

// TurnAgent leaves the SDK's conversation history available for display while
// sending only the items from the current request to llmux.
export class TurnAgent extends HttpAgent {
  private turn: Message[] | undefined;

  setTurn(messages: Message[]): void {
    this.turn = messages;
  }

  protected override requestInit(input: RunAgentInput): RequestInit {
    if (!this.turn) throw new Error("setTurn must be called before each run");
    const request = super.requestInit({ ...input, messages: this.turn });
    const headers = new Headers(request.headers);
    headers.set("Idempotency-Key", input.runId);
    this.turn = undefined;
    return { ...request, headers };
  }
}
