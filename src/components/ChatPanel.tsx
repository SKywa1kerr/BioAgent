import { useRef, useEffect, useState, type KeyboardEvent, type ReactNode } from "react";
import type { AppLanguage } from "../i18n";
import { t } from "../i18n";

interface ChatMessage {
  role: string;
  content: string;
}

interface ChatPanelProps {
  messages: ChatMessage[];
  isRunning: boolean;
  progress: { phase: string; progress: number; label: string };
  language: AppLanguage;
  initialized: boolean;
  onSend: (text: string) => void;
  onExportDebug: () => void;
  onToggleLanguage: () => void;
  onToggleTheme: () => void;
  onOpenSettings: () => void;
  onClear: () => void;
  theme: "light" | "dark";
  prefillText?: string | null;
  onPrefillConsumed?: () => void;
  inputRef?: React.RefObject<HTMLTextAreaElement>;
  onOpenPalette?: () => void;
}

function renderInlineRichText(text: string): ReactNode[] {
  return text.split("**").map((chunk, idx) => {
    const withCode = chunk.split("`").map((part, codeIdx) => {
      if (codeIdx % 2 === 1) return <code key={`code-${idx}-${codeIdx}`}>{part}</code>;
      return <span key={`txt-${idx}-${codeIdx}`}>{part}</span>;
    });
    if (idx % 2 === 1) return <strong key={`strong-${idx}`}>{withCode}</strong>;
    return <span key={`span-${idx}`}>{withCode}</span>;
  });
}

function renderStructuredMessage(content: string): ReactNode[] {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i].trim();
    if (!line) {
      i += 1;
      continue;
    }

    if (line.startsWith("### ") || line.startsWith("## ")) {
      const heading = line.replace(/^#{2,3}\s+/, "");
      blocks.push(<h4 key={`h-${i}`}>{renderInlineRichText(heading)}</h4>);
      i += 1;
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      const items: ReactNode[] = [];
      let j = i;
      while (j < lines.length && /^\d+\.\s+/.test(lines[j].trim())) {
        const itemText = lines[j].trim().replace(/^\d+\.\s+/, "");
        items.push(<li key={`ol-${j}`}>{renderInlineRichText(itemText)}</li>);
        j += 1;
      }
      blocks.push(<ol key={`ol-block-${i}`}>{items}</ol>);
      i = j;
      continue;
    }

    if (line.startsWith("- ") || line.startsWith("* ")) {
      const items: ReactNode[] = [];
      let j = i;
      while (j < lines.length) {
        const cur = lines[j].trim();
        if (!(cur.startsWith("- ") || cur.startsWith("* "))) break;
        items.push(<li key={`ul-${j}`}>{renderInlineRichText(cur.slice(2).trim())}</li>);
        j += 1;
      }
      blocks.push(<ul key={`ul-block-${i}`}>{items}</ul>);
      i = j;
      continue;
    }

    const paragraph: string[] = [];
    let j = i;
    while (j < lines.length) {
      const cur = lines[j].trim();
      if (!cur || cur.startsWith("## ") || cur.startsWith("### ") || /^\d+\.\s+/.test(cur) || cur.startsWith("- ") || cur.startsWith("* ")) break;
      paragraph.push(cur);
      j += 1;
    }
    blocks.push(<p key={`p-${i}`}>{renderInlineRichText(paragraph.join(" "))}</p>);
    i = j;
  }

  if (blocks.length === 0) {
    return [<p key="plain-empty">{content}</p>];
  }
  return blocks;
}

function isLongAssistantMessage(content: string): boolean {
  const normalized = content.replace(/\r\n/g, "\n");
  const lineCount = normalized.split("\n").length;
  return normalized.length > 420 || lineCount > 9;
}

function formatTime(ts: number): string {
  const d = new Date(ts);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

export function ChatPanel({
  messages, isRunning, progress, language, initialized,
  onSend, onExportDebug, onToggleLanguage, onToggleTheme, onOpenSettings, onClear, theme,
  prefillText, onPrefillConsumed, inputRef, onOpenPalette,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [expandedMessageKeys, setExpandedMessageKeys] = useState<Set<string>>(new Set());
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messageListRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (prefillText != null) {
      setInput(prefillText);
      inputRef?.current?.focus();
      onPrefillConsumed?.();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefillText]);

  // Stable message IDs: counter increments for each new message
  const idCounterRef = useRef(0);
  const stableIdsRef = useRef<string[]>([]);

  // Timestamps keyed by stable ID
  const timestampsRef = useRef<Map<string, number>>(new Map());

  // Reconcile stable IDs with current messages array
  if (stableIdsRef.current.length > messages.length) {
    // Messages were cleared or trimmed — reset
    stableIdsRef.current = [];
    timestampsRef.current.clear();
  }
  while (stableIdsRef.current.length < messages.length) {
    idCounterRef.current += 1;
    const newId = `msg-${idCounterRef.current}`;
    stableIdsRef.current.push(newId);
    timestampsRef.current.set(newId, Date.now());
  }

  useEffect(() => {
    const node = messageListRef.current;
    if (!node) return;
    const distanceToBottom = node.scrollHeight - node.scrollTop - node.clientHeight;
    if (distanceToBottom < 140 || isRunning) {
      node.scrollTop = node.scrollHeight;
    }
  }, [messages, isRunning]);

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      const content = input.trim();
      if (content && !isRunning) {
        onSend(content);
        setInput("");
      }
    }
  }

  function handleSendClick() {
    const content = input.trim();
    if (content && !isRunning) {
      onSend(content);
      setInput("");
    }
  }

  function handleCopy(stableId: string, content: string) {
    void navigator.clipboard.writeText(content).then(() => {
      setCopiedId(stableId);
      setTimeout(() => setCopiedId((current) => (current === stableId ? null : current)), 1500);
    });
  }

  const showProgress = isRunning || (["run", "thinking", "tool_calls", "tool_call", "tool_result"].includes(progress.phase) && progress.progress < 100);

  return (
    <aside className="chat-panel" aria-label="Chat">
      <div className="panel-title panel-title-row">
        <span>{t(language, "app.title")}</span>
        <div className="panel-action-group">
          <button className="theme-toggle action-danger" onClick={onClear} title={t(language, "chat.clear")} aria-label={t(language, "chat.clear")}>
            {t(language, "chat.clear")}
          </button>
          <button className="theme-toggle action-settings" onClick={onOpenSettings} title={t(language, "settings.title")} aria-label={t(language, "settings.title")}>
            {"\u2699"}
          </button>
          <button className="theme-toggle" onClick={onToggleLanguage} aria-label={t(language, "app.lang")}>
            {t(language, "app.lang")}
          </button>
          <button className="theme-toggle action-secondary" onClick={onExportDebug} aria-label={t(language, "app.action.exportDebug")}>
            {t(language, "app.action.exportDebug")}
          </button>
          <button className="theme-toggle" onClick={onToggleTheme} aria-label={theme === "dark" ? t(language, "app.theme.light") : t(language, "app.theme.dark")}>
            {theme === "dark" ? t(language, "app.theme.light") : t(language, "app.theme.dark")}
          </button>
        </div>
      </div>

      <div className="message-list" ref={messageListRef} role="log" aria-live="polite" aria-label="Messages">
        {messages.map((message, index) => {
          const stableId = stableIdsRef.current[index];
          const ts = timestampsRef.current.get(stableId);
          const isAssistant = message.role === "assistant";
          const isLong = isAssistant && isLongAssistantMessage(message.content);
          const expanded = isLong && expandedMessageKeys.has(stableId);

          return (
            <div key={stableId} className={`message message-${message.role}`}>
              {ts != null && (
                <div className="message-timestamp">{formatTime(ts)}</div>
              )}
              {isAssistant ? (
                <>
                  <div className={`message-content${isLong && !expanded ? " message-content-collapsed" : ""}`}>
                    {renderStructuredMessage(message.content)}
                  </div>
                  <div className="message-actions">
                    {isLong ? (
                      <button
                        type="button"
                        className="message-expand-button"
                        onClick={() => {
                          setExpandedMessageKeys((prev) => {
                            const next = new Set(prev);
                            if (next.has(stableId)) next.delete(stableId);
                            else next.add(stableId);
                            return next;
                          });
                        }}
                      >
                        {expanded ? t(language, "app.message.collapse") : t(language, "app.message.expand")}
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className="message-copy-button"
                      onClick={() => handleCopy(stableId, message.content)}
                    >
                      {copiedId === stableId ? t(language, "chat.copied") : t(language, "chat.copy")}
                    </button>
                  </div>
                </>
              ) : message.content}
            </div>
          );
        })}
        {isRunning ? (
          <div className="message message-assistant message-pending">
            <span>{t(language, "app.assistant.pending")}</span>
            <span className="typing-dots" aria-hidden="true"><i /><i /><i /></span>
          </div>
        ) : null}
      </div>

      {showProgress ? (
        <div className="chat-progress-inline" aria-live="polite">
          <div className="chat-progress-inline-track">
            <div className="chat-progress-inline-fill" style={{ width: `${Math.max(10, progress.progress)}%` }} />
          </div>
        </div>
      ) : null}

      <div className="composer" role="form" aria-label="Message composer">
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={t(language, "app.input.placeholder")}
          aria-label={t(language, "app.input.placeholder")}
          title={t(language, "shortcut.focusChat")}
          disabled={!initialized || isRunning}
        />
        <button onClick={handleSendClick} disabled={!initialized || isRunning}>
          {isRunning ? t(language, "app.action.sending") : t(language, "app.action.send")}
        </button>
      </div>
    </aside>
  );
}
