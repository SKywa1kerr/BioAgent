export type ProviderId =
  | "openai"
  | "anthropic"
  | "deepseek"
  | "sjtu"
  | "ollama"
  | "custom";

export interface ProviderPreset {
  id: ProviderId;
  label: string;
  defaultBaseUrl: string;
  suggestedModels: readonly string[];
  requiresApiKey: boolean;
  noteI18nKey?: string;
}

export const PROVIDERS: readonly ProviderPreset[];

export function getProvider(id: string): ProviderPreset;

export interface ApplyProviderSwitchArgs {
  fromId: ProviderId;
  toId: ProviderId;
  currentBaseUrl: string;
  currentModel: string;
}

export interface ApplyProviderSwitchResult {
  baseUrl: string;
  model: string;
  baseUrlIsCustom: boolean;
  modelIsCustom: boolean;
}

export function applyProviderSwitch(
  args: ApplyProviderSwitchArgs,
): ApplyProviderSwitchResult;

export function inferProviderFromBaseUrl(baseUrl: string): ProviderId;
