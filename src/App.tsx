import { useEffect, useMemo, useState } from "react";
import { SmartCanvas, type PanelType } from "./components/SmartCanvas";
import { AnalysisPanel } from "./components/panels/AnalysisPanel";
import { MutationTrendPanel } from "./components/panels/MutationTrendPanel";
import { LabSuggestionPanel } from "./components/panels/LabSuggestionPanel";
import { ConfirmationDialog } from "./components/panels/ConfirmationDialog";

declare global {
  interface Window {
    electronAPI: {
      invoke: (channel: string, ...args: unknown[]) => Promise<any>;
      onAgentEvent: (callback: (payload: any) => void) => () => void;
    };
  }
}

const MAX_MESSAGES = 60;
const MAX_EVENTS = 80;

function resolvePanelFromEvent(payload: any): { panelType: PanelType; panelPayload: any; confirmMessage?: string } | null {
  if (payload.type === "confirm") {
    return {
      panelType: "confirmation",
      panelPayload: null,
      confirmMessage: payload.message || "请确认执行该操作",
    };
  }

  if (payload.type === "tool_result") {
    if (payload.tool === "detect_mutation_trends") return { panelType: "trends", panelPayload: payload.result };
    if (payload.tool === "generate_lab_suggestions") return { panelType: "suggestions", panelPayload: payload.result };
    if (payload.tool === "analyze_sequences") return { panelType: "analysis", panelPayload: payload.result };
  }

  if (payload.type === "reply") {
    if (payload.uiAction === "show_trends") return { panelType: "trends", panelPayload: payload.result || payload };
    if (payload.uiAction === "show_suggestions") return { panelType: "suggestions", panelPayload: payload.result || payload };
    if (payload.uiAction === "show_analysis") return { panelType: "analysis", panelPayload: payload.result || payload };
    return { panelType: "text", panelPayload: payload };
  }

  return null;
}

export function App() {
  const [messages, setMessages] = useState<Array<{ role: string; content: string }>>([]);
  const [input, setInput] = useState("");
  const [events, setEvents] = useState<any[]>([]);
  const [initialized, setInitialized] = useState(false);
  const [panelType, setPanelType] = useState<PanelType>("text");
  const [panelPayload, setPanelPayload] = useState<any>(null);
  const [confirmMessage, setConfirmMessage] = useState("");
  const [llmApiKey, setLlmApiKey] = useState("");
  const [llmBaseUrl, setLlmBaseUrl] = useState("https://models.sjtu.edu.cn/api/v1");
  const [llmModel, setLlmModel] = useState("deepseek-chat");
  const [statusMessage, setStatusMessage] = useState("请先初始化智能体");

  useEffect(() => {
    const unsubscribe = window.electronAPI.onAgentEvent((payload) => {
      setEvents((current) => [...current, payload].slice(-MAX_EVENTS));

      if (payload.type === "reply") {
        setMessages((current) => [...current, { role: "assistant", content: payload.content }].slice(-MAX_MESSAGES));
      }

      const resolved = resolvePanelFromEvent(payload);
      if (resolved) {
        setPanelType(resolved.panelType);
        setPanelPayload(resolved.panelPayload);
        if (resolved.confirmMessage) {
          setConfirmMessage(resolved.confirmMessage);
        }
      }
    });
    return () => unsubscribe?.();
  }, []);

  const latestEvent = useMemo(() => events[events.length - 1], [events]);

  async function initializeHarness() {
    try {
      setStatusMessage("正在初始化智能体...");
      await window.electronAPI.invoke("agent-harness-shutdown");
      await window.electronAPI.invoke("agent-harness-init", {
        llmApiKey,
        llmBaseUrl,
        llmModel,
      });
      setInitialized(true);
      setStatusMessage("智能体已就绪");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatusMessage(`初始化失败：${message}`);
    }
  }

  async function handleSend() {
    const content = input.trim();
    if (!content) return;
    if (!initialized) {
      setStatusMessage("请先初始化智能体");
      return;
    }
    setMessages((current) => [...current, { role: "user", content }].slice(-MAX_MESSAGES));
    setInput("");
    try {
      await window.electronAPI.invoke("agent-harness-run", content, {
        llmApiKey,
        llmBaseUrl,
        llmModel,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setMessages((current) => [...current, { role: "assistant", content: `调用失败：${message}` }]);
    }
  }

  function renderPanel() {
    if (!initialized) {
      return (
        <div className="detail-card">
          <h3>智能体配置</h3>
          <div className="settings-form">
            <label>
              <span>LLM API Key</span>
              <input type="password" value={llmApiKey} onChange={(e) => setLlmApiKey(e.target.value)} placeholder="请输入 API Key" />
            </label>
            <label>
              <span>Base URL</span>
              <input value={llmBaseUrl} onChange={(e) => setLlmBaseUrl(e.target.value)} placeholder="https://models.sjtu.edu.cn/api/v1" />
            </label>
            <label>
              <span>Model</span>
              <input value={llmModel} onChange={(e) => setLlmModel(e.target.value)} placeholder="deepseek-chat" />
            </label>
            <div className="settings-actions">
              <button className="primary-button" onClick={() => void initializeHarness()}>初始化智能体</button>
            </div>
            <div className="status-line">{statusMessage}</div>
          </div>
        </div>
      );
    }

    if (panelType === "analysis") return <AnalysisPanel result={panelPayload} />;
    if (panelType === "trends") return <MutationTrendPanel result={panelPayload} />;
    if (panelType === "suggestions") return <LabSuggestionPanel result={panelPayload} />;
    if (panelType === "confirmation") {
      return (
        <ConfirmationDialog
          message={confirmMessage}
          onConfirm={() => setPanelType("text")}
          onCancel={() => setPanelType("text")}
        />
      );
    }
    return latestEvent ? <pre>{JSON.stringify(latestEvent, null, 2)}</pre> : "等待智能体结果...";
  }

  return (
    <div className="app-shell">
      <aside className="chat-panel">
        <div className="panel-title">Ultimate BioAgent</div>
        <div className="message-list">
          {messages.map((message, index) => (
            <div key={index} className={`message message-${message.role}`}>
              {message.content}
            </div>
          ))}
        </div>
        <div className="composer">
          <textarea value={input} onChange={(e) => setInput(e.target.value)} placeholder="请输入中文指令，比如：分析这批数据并给出实验建议" />
          <button onClick={handleSend}>发送</button>
        </div>
      </aside>
      <main className="canvas-panel">
        <SmartCanvas title="Smart Canvas" panelType={panelType}>
          {renderPanel()}
        </SmartCanvas>
      </main>
    </div>
  );
}
