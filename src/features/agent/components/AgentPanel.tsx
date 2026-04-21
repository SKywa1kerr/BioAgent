import { FormEvent, KeyboardEvent, useState } from "react";
import type { Sample, AgentMessage } from "../../../shared/types";
import "./AgentPanel.css";

const { invoke } = window.electronAPI;

interface AgentPanelProps {
  samples: Sample[];
  selectedSampleId: string | null;
}

export function AgentPanel({ samples, selectedSampleId }: AgentPanelProps) {
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [isRunning, setIsRunning] = useState(false);

  const handleSubmit = async (event?: FormEvent<HTMLFormElement>) => {
    event?.preventDefault();
    const prompt = draft.trim();
    if (!prompt || isRunning) return;

    const userMessage: AgentMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: prompt,
      timestamp: Date.now(),
    };

    const conversationHistory = [...messages, userMessage];
    setDraft("");
    setMessages(conversationHistory);
    setIsRunning(true);

    try {
      // Placeholder for agent chat - to be implemented
      const response = await invoke("agent-chat", {
        message: prompt,
        context: {
          samples: samples.map(s => ({
            id: s.id,
            name: s.name || s.clone || s.id,
            status: s.status,
          })),
          selectedSampleId,
        },
      });

      const agentMessage: AgentMessage = {
        id: `agent-${Date.now()}`,
        role: "assistant",
        content: typeof response === "string" ? response : JSON.stringify(response),
        timestamp: Date.now(),
      };

      setMessages([...conversationHistory, agentMessage]);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      const agentMessage: AgentMessage = {
        id: `agent-${Date.now()}`,
        role: "assistant",
        content: `Error: ${errorMessage}`,
        timestamp: Date.now(),
      };
      setMessages([...conversationHistory, agentMessage]);
    } finally {
      setIsRunning(false);
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSubmit();
    }
  };

  return (
    <div className="agent-panel">
      <header className="agent-panel-header">
        <h3>AI Agent</h3>
        <span className={`agent-status ${isRunning ? "busy" : "idle"}`}>
          {isRunning ? "Running..." : "Ready"}
        </span>
      </header>

      <div className="agent-message-list">
        {messages.length === 0 ? (
          <div className="agent-empty-state">
            <h4>Ask me anything</h4>
            <p>I can help analyze your sequencing results, query samples, and more.</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`agent-message ${message.role}`}
            >
              <div className="message-content">{message.content}</div>
              <span className="message-time">
                {new Date(message.timestamp).toLocaleTimeString()}
              </span>
            </div>
          ))
        )}
      </div>

      <form className="agent-composer" onSubmit={handleSubmit}>
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          rows={3}
          disabled={isRunning}
        />
        <div className="agent-composer-footer">
          <span>Press Enter to send, Shift+Enter for new line</span>
          <button
            type="submit"
            disabled={isRunning || draft.trim().length === 0}
            className="btn-primary"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
