import type { AppLanguage, ChatMessage } from "../types";
import { t } from "../i18n";
import "./ChatMessage.css";

interface ChatMessageCardProps {
  language: AppLanguage;
  message: ChatMessage;
}

function formatToolLabel(toolName: string) {
  return toolName.replace(/_/g, " ");
}

function getToolStatusLabel(language: AppLanguage, status: "running" | "done" | "failed") {
  if (status === "running") return t(language, "agent.toolRunning");
  if (status === "done") return t(language, "agent.toolDone");
  return t(language, "agent.toolFailed");
}

export function ChatMessageCard({ language, message }: ChatMessageCardProps) {
  if (message.type === "tool_status") {
    return (
      <article className={`chat-message tool-status ${message.status}`}>
        <div className="message-row">
          <span className="message-label">{formatToolLabel(message.toolName)}</span>
          <span className={`tool-pill ${message.status}`}>
            {getToolStatusLabel(language, message.status)}
          </span>
        </div>
        <div className="message-body">{message.content}</div>
      </article>
    );
  }

  return (
    <article className={`chat-message ${message.type}`}>
      <div className="message-body">{message.content}</div>
      {message.type === "agent" && (message.stopReason || message.usage) ? (
        <div className="message-meta">
          {message.stopReason ? (
            <span>
              {t(language, "agent.stop")}: {message.stopReason}
            </span>
          ) : null}
          {message.usage ? (
            <span>
              {t(language, "agent.tokens")}:{" "}
              {message.usage.total ?? message.usage.input + message.usage.output}
            </span>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}
