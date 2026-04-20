// Agent Feature - AI assistant chat panel
// Exports the public API for the agent feature

export { AgentPanel } from "./components/AgentPanel";
export { ChatMessage } from "./components/ChatMessage";

// Re-export types from Python models if needed
export interface AgentMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  tools?: AgentToolCall[];
}

export interface AgentToolCall {
  name: string;
  arguments: Record<string, unknown>;
  result?: unknown;
}
