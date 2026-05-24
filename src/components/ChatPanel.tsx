import { useRef, useEffect, useState, type KeyboardEvent, type ReactNode } from "react";
import { ArrowDown, ArrowUp, Bot, Download, Loader2, Paperclip, Pencil, RotateCcw, Square, User } from "lucide-react";
import { ModelPicker } from "./ModelPicker";
import type { AppLanguage } from "../i18n";
import { t } from "../i18n";
import { Icon } from "./ui/Icon";

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
  /** Cancel the in-flight runTurn. Optional — when omitted the Stop
   *  button is hidden and the send button just stays in spinner mode. */
  onCancel?: () => void;
  /** Resend a previous user message verbatim. */
  onResend?: (text: string) => void;
  /** Pull a previous user message back into the composer for editing.
   *  Host should also trim subsequent assistant messages so the edit
   *  is the new "tail" of the conversation. */
  onEdit?: (index: number, text: string) => void;
  onExportDebug: () => void;
  onToggleLanguage: () => void;
  onToggleTheme: () => void;
  onOpenSettings: () => void;
  onClear: () => void;
  /** Optional: export current chat as Markdown. */
  onExportMarkdown?: () => void;
  theme: "light" | "dark";
  prefillText?: string | null;
  onPrefillConsumed?: () => void;
  inputRef?: React.RefObject<HTMLTextAreaElement>;
  onOpenPalette?: () => void;
  /** Optional paperclip-attach handler. Invoked when the user clicks the
   *  📎 button in the input area; the host (App) opens the file/folder
   *  picker and routes the chosen paths through the same import flow as
   *  drag-and-drop. Hide the button when this is undefined. */
  onAttach?: () => void;
  /** Model picker (OpenWebUI-style top selector). Optional — if omitted
   *  the chat title is shown instead of the picker. */
  modelPicker?: {
    currentModel: string;
    availableModels: readonly string[];
    providerLabel: string;
    onChange: (model: string) => void;
  };
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

/** Markdown table helpers — detect `| a | b |` header rows and `| --- |`
 *  separator rows. Permissive: we accept rows with or without leading/
 *  trailing pipes and trim whitespace per cell. */
function isMdTableHeader(line: string): boolean {
  if (!line.includes("|")) return false;
  // Reject lines that are obviously not table-like (e.g. just "|").
  const stripped = line.replace(/^\||\|$/g, "").trim();
  if (!stripped) return false;
  // Must have at least one pipe in the stripped middle.
  return stripped.includes("|");
}

function isMdTableSeparator(line: string): boolean {
  if (!line.includes("|")) return false;
  const cells = line.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  if (cells.length === 0) return false;
  return cells.every((c) => /^:?-{3,}:?$/.test(c));
}

function splitMdRow(line: string): string[] {
  return line.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
}

function parseMdTableAlignments(separator: string, columnCount: number): Array<"left" | "right" | "center" | undefined> {
  const cells = separator.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  const out: Array<"left" | "right" | "center" | undefined> = [];
  for (let i = 0; i < columnCount; i += 1) {
    const c = cells[i] || "";
    if (c.startsWith(":") && c.endsWith(":")) out.push("center");
    else if (c.endsWith(":")) out.push("right");
    else if (c.startsWith(":")) out.push("left");
    else out.push(undefined);
  }
  return out;
}

/** Fenced code block: ```lang\n...\n``` */
function CodeBlock({ lang, code }: { lang: string; code: string }): JSX.Element {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    void navigator.clipboard?.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <pre className="md-codeblock">
      <div className="md-codeblock-header">
        <span className="md-codeblock-lang">{lang || "code"}</span>
        <button
          type="button"
          className="md-codeblock-copy"
          onClick={handleCopy}
          aria-label="Copy code"
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <code>{code}</code>
    </pre>
  );
}

function renderStructuredMessage(content: string): ReactNode[] {
  const lines = content.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i] ?? "";
    const trimmed = line.trim();

    // Fenced code block: ```optional-lang ... ```
    if (trimmed.startsWith("```")) {
      const lang = trimmed.replace(/^```\s*/, "").trim();
      const codeLines: string[] = [];
      let j = i + 1;
      while (j < lines.length && !(lines[j] ?? "").trim().startsWith("```")) {
        codeLines.push(lines[j] ?? "");
        j += 1;
      }
      blocks.push(<CodeBlock key={`code-${i}`} lang={lang} code={codeLines.join("\n")} />);
      i = j + 1; // skip closing fence
      continue;
    }

    if (!trimmed) {
      i += 1;
      continue;
    }

    if (trimmed.startsWith("### ") || trimmed.startsWith("## ")) {
      const heading = trimmed.replace(/^#{2,3}\s+/, "");
      blocks.push(<h4 key={`h-${i}`}>{renderInlineRichText(heading)}</h4>);
      i += 1;
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      const items: ReactNode[] = [];
      let j = i;
      while (j < lines.length && /^\d+\.\s+/.test((lines[j] ?? "").trim())) {
        const itemText = (lines[j] ?? "").trim().replace(/^\d+\.\s+/, "");
        items.push(<li key={`ol-${j}`}>{renderInlineRichText(itemText)}</li>);
        j += 1;
      }
      blocks.push(<ol key={`ol-block-${i}`}>{items}</ol>);
      i = j;
      continue;
    }

    // Markdown table detection: a header row of cells separated by `|`
    // followed by a separator row whose cells are dashes (with optional
    // colons for alignment). Anything else falls through to the
    // paragraph/list branches below.
    if (isMdTableHeader(trimmed) && isMdTableSeparator((lines[i + 1] ?? "").trim())) {
      const headerCells = splitMdRow(trimmed);
      const aligns = parseMdTableAlignments((lines[i + 1] ?? "").trim(), headerCells.length);
      const bodyRows: string[][] = [];
      let j = i + 2;
      while (j < lines.length) {
        const cur = (lines[j] ?? "").trim();
        if (!isMdTableHeader(cur)) break;
        bodyRows.push(splitMdRow(cur));
        j += 1;
      }
      blocks.push(
        <div className="md-table-wrap" key={`tbl-${i}`}>
          <table className="md-table">
            <thead>
              <tr>
                {headerCells.map((cell, idx) => (
                  <th key={`th-${idx}`} style={aligns[idx] ? { textAlign: aligns[idx] } : undefined}>
                    {renderInlineRichText(cell)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {bodyRows.map((row, ri) => (
                <tr key={`tr-${ri}`}>
                  {row.map((cell, ci) => (
                    <td key={`td-${ri}-${ci}`} style={aligns[ci] ? { textAlign: aligns[ci] } : undefined}>
                      {renderInlineRichText(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      i = j;
      continue;
    }

    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      const items: ReactNode[] = [];
      let j = i;
      while (j < lines.length) {
        const cur = (lines[j] ?? "").trim();
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
      const cur = (lines[j] ?? "").trim();
      if (!cur || cur.startsWith("## ") || cur.startsWith("### ") || cur.startsWith("```") || /^\d+\.\s+/.test(cur) || cur.startsWith("- ") || cur.startsWith("* ") || (isMdTableHeader(cur) && isMdTableSeparator((lines[j + 1] ?? "").trim()))) break;
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
  onSend, onCancel, onResend, onEdit, onExportDebug, onExportMarkdown, onToggleLanguage, onToggleTheme, onOpenSettings, onClear, theme,
  prefillText, onPrefillConsumed, inputRef, onOpenPalette, onAttach, modelPicker,
}: ChatPanelProps) {
  const [input, setInput] = useState(() => {
    try {
      return window.localStorage?.getItem("bioagent.draftInput") || "";
    } catch {
      return "";
    }
  });
  // Auto-save composer draft to localStorage on every change so a refresh
  // (or an accidental window close) doesn't drop the unsent message.
  useEffect(() => {
    try {
      if (input) window.localStorage?.setItem("bioagent.draftInput", input);
      else window.localStorage?.removeItem("bioagent.draftInput");
    } catch {
      // ignore — storage may be disabled
    }
  }, [input]);
  const [expandedMessageKeys, setExpandedMessageKeys] = useState<Set<string>>(new Set());
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messageListRef = useRef<HTMLDivElement | null>(null);
  // Tracks whether the user is scrolled away from the bottom so we can show
  // a floating "↓ jump to latest" button.
  const [scrolledAway, setScrolledAway] = useState(false);

  useEffect(() => {
    const node = messageListRef.current;
    if (!node) return;
    const onScroll = () => {
      const distance = node.scrollHeight - node.scrollTop - node.clientHeight;
      setScrolledAway(distance > 120);
    };
    node.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => node.removeEventListener("scroll", onScroll);
  }, []);

  const scrollToBottom = () => {
    const node = messageListRef.current;
    if (!node) return;
    node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
  };

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
        {modelPicker ? (
          <ModelPicker
            currentModel={modelPicker.currentModel}
            availableModels={modelPicker.availableModels}
            providerLabel={modelPicker.providerLabel}
            onChange={modelPicker.onChange}
            onOpenSettings={onOpenSettings}
            labels={{
              current: t(language, "modelPicker.current"),
              chooseLabel: t(language, "modelPicker.choose"),
              settingsCTA: t(language, "modelPicker.settingsCTA"),
            }}
          />
        ) : (
          <span>{t(language, "app.title")}</span>
        )}
        <div className="panel-action-group">
          <button className="icon-button" onClick={onClear} title={t(language, "chat.clear")} aria-label={t(language, "chat.clear")}>
            <Icon name="trash" size={16} />
          </button>
          <button className="icon-button" onClick={onOpenSettings} title={t(language, "settings.title")} aria-label={t(language, "settings.title")}>
            <Icon name="settings" size={16} />
          </button>
          <button className="icon-button" onClick={onToggleLanguage} aria-label={t(language, "app.lang")} title={t(language, "app.lang")}>
            <Icon name="language" size={16} />
          </button>
          <button className="icon-button" onClick={onExportDebug} aria-label={t(language, "app.action.exportDebug")} title={t(language, "app.action.exportDebug")}>
            <Icon name="download" size={16} />
          </button>
          {onExportMarkdown ? (
            <button
              className="icon-button"
              onClick={onExportMarkdown}
              aria-label={t(language, "chat.exportMarkdown")}
              title={t(language, "chat.exportMarkdown")}
              disabled={messages.length === 0}
            >
              <Download size={16} strokeWidth={1.7} aria-hidden="true" />
            </button>
          ) : null}
          <button className="icon-button" onClick={onToggleTheme} aria-label={theme === "dark" ? t(language, "app.theme.light") : t(language, "app.theme.dark")} title={theme === "dark" ? t(language, "app.theme.light") : t(language, "app.theme.dark")}>
            <Icon name={theme === "dark" ? "sun" : "moon"} size={16} />
          </button>
        </div>
      </div>

      <div className="message-list" ref={messageListRef} role="log" aria-live="polite" aria-label="Messages">
        {messages.length === 0 && !isRunning ? (
          <div className="chat-empty-hero" aria-hidden="true">
            <div className="chat-empty-greeting">
              {language === "zh" ? "今天分析点什么？" : "What shall we analyze today?"}
            </div>
            <div className="chat-empty-hint">
              {language === "zh"
                ? "拖入文件夹、点 📎 添加文件，或直接输入指令开始。"
                : "Drop a folder, click 📎 to attach files, or just type to begin."}
            </div>
          </div>
        ) : null}
        {messages.map((message, index) => {
          const stableId = stableIdsRef.current[index] ?? `msg-fallback-${index}`;
          const ts = timestampsRef.current.get(stableId);
          const isAssistant = message.role === "assistant";
          const isLong = isAssistant && isLongAssistantMessage(message.content);
          const expanded = isLong && expandedMessageKeys.has(stableId);

          return (
            <div key={stableId} className={`message message-${message.role}`}>
              {/* Avatar — small circle to the left (assistant) or right (user). */}
              <div className={`message-avatar message-avatar-${message.role}`} aria-hidden="true">
                {isAssistant ? <Bot size={14} strokeWidth={1.8} /> : <User size={14} strokeWidth={1.8} />}
              </div>
              <div className="message-body">
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
                        className="message-icon-button"
                        onClick={() => handleCopy(stableId, message.content)}
                        title={copiedId === stableId ? t(language, "chat.copied") : t(language, "chat.copy")}
                        aria-label={t(language, "chat.copy")}
                      >
                        {copiedId === stableId ? "✓" : t(language, "chat.copy")}
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="message-content message-content-user">{message.content}</div>
                    <div className="message-actions message-actions-user">
                      <button
                        type="button"
                        className="message-icon-button"
                        onClick={() => handleCopy(stableId, message.content)}
                        title={copiedId === stableId ? t(language, "chat.copied") : t(language, "chat.copy")}
                        aria-label={t(language, "chat.copy")}
                      >
                        {copiedId === stableId ? "✓" : t(language, "chat.copy")}
                      </button>
                      {onEdit ? (
                        <button
                          type="button"
                          className="message-icon-button"
                          onClick={() => onEdit(index, message.content)}
                          title={t(language, "chat.edit")}
                          aria-label={t(language, "chat.edit")}
                        >
                          <Pencil size={12} strokeWidth={1.8} aria-hidden="true" />
                        </button>
                      ) : null}
                      {onResend ? (
                        <button
                          type="button"
                          className="message-icon-button"
                          onClick={() => onResend(message.content)}
                          title={t(language, "chat.resend")}
                          aria-label={t(language, "chat.resend")}
                          disabled={isRunning}
                        >
                          <RotateCcw size={12} strokeWidth={1.8} aria-hidden="true" />
                        </button>
                      ) : null}
                    </div>
                  </>
                )}
              </div>
            </div>
          );
        })}
        {isRunning ? (
          <div className="message message-assistant message-pending">
            <div className="message-avatar message-avatar-assistant" aria-hidden="true">
              <Bot size={14} strokeWidth={1.8} />
            </div>
            <div className="message-body">
              <Loader2 size={14} strokeWidth={1.8} className="message-pending-spinner" aria-hidden="true" />
              <span>{t(language, "app.assistant.pending")}</span>
            </div>
          </div>
        ) : null}
      </div>

      {scrolledAway && messages.length > 0 ? (
        <button
          type="button"
          className="chat-scroll-bottom"
          onClick={scrollToBottom}
          aria-label={t(language, "chat.scrollToBottom")}
          title={t(language, "chat.scrollToBottom")}
        >
          <ArrowDown size={16} strokeWidth={2} aria-hidden="true" />
        </button>
      ) : null}

      <div className="composer" role="form" aria-label="Message composer">
        {onAttach ? (
          <button
            type="button"
            className="composer-attach"
            onClick={onAttach}
            disabled={!initialized || isRunning}
            aria-label={t(language, "composer.attach")}
            title={t(language, "composer.attach")}
          >
            <Paperclip size={16} strokeWidth={1.7} aria-hidden="true" />
          </button>
        ) : null}
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
        <button
          className="composer-send"
          onClick={isRunning && onCancel ? onCancel : handleSendClick}
          disabled={!initialized || (!isRunning && input.trim().length === 0)}
          aria-label={isRunning ? t(language, "app.action.cancel") : t(language, "app.action.send")}
          title={isRunning ? t(language, "app.action.cancel") : t(language, "app.action.send")}
        >
          {isRunning ? (
            <Square size={14} strokeWidth={2.4} fill="currentColor" aria-hidden="true" />
          ) : (
            <ArrowUp size={16} strokeWidth={2.2} aria-hidden="true" />
          )}
        </button>
      </div>
    </aside>
  );
}
