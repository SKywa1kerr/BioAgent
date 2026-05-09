// Multi-provider preset module. Pure data + pure functions, no DOM/storage.

export const PROVIDERS = [
  {
    id: "openai",
    label: "OpenAI",
    defaultBaseUrl: "https://api.openai.com/v1",
    suggestedModels: ["gpt-4o-mini", "gpt-4o"],
    requiresApiKey: true,
  },
  {
    id: "anthropic",
    label: "Anthropic",
    defaultBaseUrl: "",
    suggestedModels: ["claude-sonnet-4-6", "claude-haiku-4-5"],
    requiresApiKey: true,
    noteI18nKey: "provider.anthropic.proxyRequired",
  },
  {
    id: "deepseek",
    label: "DeepSeek",
    defaultBaseUrl: "https://api.deepseek.com/v1",
    suggestedModels: ["deepseek-chat", "deepseek-coder"],
    requiresApiKey: true,
  },
  {
    id: "sjtu",
    label: "SJTU AI Lab",
    defaultBaseUrl: "https://models.sjtu.edu.cn/api/v1",
    suggestedModels: ["deepseek-chat"],
    requiresApiKey: true,
  },
  {
    id: "ollama",
    label: "Ollama (Local)",
    defaultBaseUrl: "http://localhost:11434/v1",
    suggestedModels: ["llama3.2", "qwen2.5"],
    requiresApiKey: false,
  },
  {
    id: "custom",
    label: "Custom (OpenAI-compatible)",
    defaultBaseUrl: "",
    suggestedModels: [],
    requiresApiKey: true,
  },
];

export function getProvider(id) {
  const found = PROVIDERS.find((p) => p.id === id);
  if (found) return found;
  // last entry is the "custom" fallback
  return PROVIDERS[PROVIDERS.length - 1];
}

/** Compute next baseUrl/model on provider switch.
 *  Preserves user-customized values (those not equal to the previous preset's defaults). */
export function applyProviderSwitch(args) {
  const from = getProvider(args.fromId);
  const to = getProvider(args.toId);
  const currentBaseUrl = (args.currentBaseUrl ?? "").trim();
  const currentModel = (args.currentModel ?? "").trim();
  const baseUrlIsCustom = currentBaseUrl !== "" && currentBaseUrl !== from.defaultBaseUrl;
  const modelIsCustom = currentModel !== "" && !from.suggestedModels.includes(currentModel);
  return {
    baseUrl: baseUrlIsCustom ? currentBaseUrl : to.defaultBaseUrl,
    model: modelIsCustom ? currentModel : (to.suggestedModels[0] ?? ""),
    baseUrlIsCustom,
    modelIsCustom,
  };
}

/** Best-effort guess for the provider id given an existing baseURL.
 *  Used when migrating settings that pre-date the provider field. */
export function inferProviderFromBaseUrl(baseUrl) {
  const u = (baseUrl ?? "").trim();
  if (!u) return "custom";
  if (u.includes("openai.com")) return "openai";
  if (u.includes("anthropic.com")) return "anthropic";
  if (u.includes("deepseek.com")) return "deepseek";
  if (u.includes("sjtu.edu.cn")) return "sjtu";
  if (u.includes("localhost:11434") || u.includes("127.0.0.1:11434")) return "ollama";
  return "custom";
}
