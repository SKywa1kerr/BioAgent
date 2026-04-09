import type { AppSettings } from "../types";

export const DEFAULT_ANALYSIS_DECISION_MODE = "rules" as const;

export function isAiReviewEnabled(settings: Pick<AppSettings, "analysisDecisionMode">) {
  return (settings.analysisDecisionMode ?? DEFAULT_ANALYSIS_DECISION_MODE) === "hybrid";
}

export function validateAiReviewSettings(
  settings: Pick<AppSettings, "analysisDecisionMode" | "llmApiKey" | "llmBaseUrl" | "llmModel">
):
  | { ok: true }
  | { ok: false; reason: "missing_api_key" | "missing_base_url" | "missing_model" } {
  if (!isAiReviewEnabled(settings)) {
    return { ok: true };
  }

  if (!settings.llmApiKey?.trim()) {
    return { ok: false, reason: "missing_api_key" };
  }

  if (!settings.llmBaseUrl?.trim()) {
    return { ok: false, reason: "missing_base_url" };
  }

  if (!settings.llmModel?.trim()) {
    return { ok: false, reason: "missing_model" };
  }

  return { ok: true };
}


