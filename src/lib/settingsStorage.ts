// src/lib/settingsStorage.ts
const STORAGE_KEY = "bioagent-settings";

export interface AgentSettings {
  llmApiKey: string;
  llmBaseUrl: string;
  llmModel: string;
}

const DEFAULTS: AgentSettings = {
  llmApiKey: "",
  llmBaseUrl: "https://models.sjtu.edu.cn/api/v1",
  llmModel: "deepseek-chat",
};

export function loadSettings(): AgentSettings {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw);
    return {
      llmApiKey: typeof parsed.k === "string" ? atob(parsed.k) : DEFAULTS.llmApiKey,
      llmBaseUrl: typeof parsed.u === "string" ? parsed.u : DEFAULTS.llmBaseUrl,
      llmModel: typeof parsed.m === "string" ? parsed.m : DEFAULTS.llmModel,
    };
  } catch {
    return { ...DEFAULTS };
  }
}

export function saveSettings(settings: AgentSettings): void {
  try {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        k: btoa(settings.llmApiKey),
        u: settings.llmBaseUrl,
        m: settings.llmModel,
      }),
    );
  } catch {
    // storage full or blocked — silently ignore
  }
}

export function clearSettings(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
