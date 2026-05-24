// src/lib/settingsStorage.ts
import { inferProviderFromBaseUrl, type ProviderId } from "./providers";

const STORAGE_KEY = "bioagent-settings";

export interface AgentSettings {
  provider: ProviderId;
  llmApiKey: string;
  llmBaseUrl: string;
  llmModel: string;
  maxTokens: number;
  /** Optional CSS hex like "#d97757" to override the accent color.
   *  Empty string = use the theme default. */
  accentColor?: string;
  /** Whether to show per-message token usage in the chat. Default off. */
  showTokenUsage?: boolean;
}

const DEFAULTS: AgentSettings = {
  provider: "custom",
  llmApiKey: "",
  llmBaseUrl: "",
  llmModel: "deepseek-chat",
  maxTokens: 2400,
  accentColor: "",
  showTokenUsage: false,
};

const ALLOWED_PROVIDERS: readonly ProviderId[] = [
  "openai",
  "anthropic",
  "deepseek",
  "ollama",
  "custom",
];

function coerceProvider(raw: unknown, baseUrl: string): ProviderId {
  if (typeof raw === "string" && (ALLOWED_PROVIDERS as readonly string[]).includes(raw)) {
    return raw as ProviderId;
  }
  return inferProviderFromBaseUrl(baseUrl);
}

export function loadSettings(): AgentSettings {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw);
    const llmBaseUrl = typeof parsed.u === "string" ? parsed.u : DEFAULTS.llmBaseUrl;
    return {
      provider: coerceProvider(parsed.p, llmBaseUrl),
      llmApiKey: typeof parsed.k === "string" ? atob(parsed.k) : DEFAULTS.llmApiKey,
      llmBaseUrl,
      llmModel: typeof parsed.m === "string" ? parsed.m : DEFAULTS.llmModel,
      maxTokens: typeof parsed.t === "number" ? parsed.t : DEFAULTS.maxTokens,
      accentColor: typeof parsed.a === "string" ? parsed.a : DEFAULTS.accentColor,
      showTokenUsage: typeof parsed.tu === "boolean" ? parsed.tu : DEFAULTS.showTokenUsage,
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
        p: settings.provider,
        k: btoa(settings.llmApiKey),
        u: settings.llmBaseUrl,
        m: settings.llmModel,
        t: settings.maxTokens,
        a: settings.accentColor || "",
        tu: !!settings.showTokenUsage,
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
