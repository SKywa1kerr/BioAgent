import type { AppLanguage } from "../i18n";

export type SummaryEvent =
  | { type: "summary_pending"; analysis_id?: string; dataset?: string }
  | { type: "summary_ready"; analysis_id?: string; dataset?: string; content?: string }
  | { type: "summary_failed"; analysis_id?: string; dataset?: string; message?: string };

export type SummaryReducerResult =
  | { kind: "pending"; analysisId?: string; assistantText: string }
  | { kind: "ready"; analysisId?: string; assistantText: string | null }
  | { kind: "failed"; analysisId?: string; assistantText: string };

export function reduceSummaryEvent(
  event: SummaryEvent | { type: string; [key: string]: unknown },
  opts: { lang: AppLanguage },
): SummaryReducerResult | null;
