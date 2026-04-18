import type { WorkbenchMutation, WorkbenchSample } from "./types";
import { t, type AppLanguage } from "../../i18n";

function toMutation(item: any): WorkbenchMutation {
  return {
    position: item?.position ?? item?.ref_pos,
    refBase: item?.refBase ?? item?.ref_base,
    queryBase: item?.queryBase ?? item?.qry_base,
    type: item?.type,
    effect: item?.effect,
  };
}

function deriveStatus(item: any, mutationCount: number): "ok" | "wrong" | "uncertain" {
  const identity = typeof item?.identity === "number" ? item.identity : 0;
  const coverage = typeof item?.cds_coverage === "number" ? item.cds_coverage : (typeof item?.coverage === "number" ? item.coverage : 0);
  if (item?.frameshift) return "wrong";
  if (mutationCount > 0) return "wrong";
  if (identity >= 0.99 && coverage >= 0.8) return "ok";
  return "uncertain";
}

function deriveReason(item: any, mutationCount: number, language: AppLanguage): string {
  const identity = typeof item?.identity === "number" ? item.identity : 0;
  const coverage = typeof item?.cds_coverage === "number" ? item.cds_coverage : (typeof item?.coverage === "number" ? item.coverage : 0);
  if (item?.error) return String(item.error);
  if (item?.frameshift) return t(language, "analysis.reason.frameshift");
  if (mutationCount > 0) return t(language, "analysis.reason.detectedMut", { count: mutationCount });
  if (identity >= 0.99 && coverage >= 0.8) return t(language, "analysis.reason.highQuality");
  return t(language, "analysis.reason.review");
}

export function normalizeSamples(result: any, language: AppLanguage): WorkbenchSample[] {
  const direct = Array.isArray(result?.samples) ? result.samples : [];
  const detailSamples = Array.isArray(result?.detail?.samples) ? result.detail.samples : [];
  const payload = direct.length > 0 ? direct : detailSamples;

  return payload
    .filter((item: any) => item && typeof item === "object")
    .map((item: any, idx: number) => {
      const id = item.id || item.sid || item.name || `sample-${idx + 1}`;
      const mutations = Array.isArray(item?.mutations) ? item.mutations.map(toMutation) : [];
      const mutationCount =
        (item.sub_count ?? item.sub ?? 0) +
        (item.ins_count ?? item.ins ?? 0) +
        (item.del_count ?? item.dele ?? item.del ?? 0) ||
        mutations.length;
      const status = (item.status as "ok" | "wrong" | "uncertain" | undefined) || deriveStatus(item, mutationCount);
      const reason = item.reason || item.review_reason || item.llm_reason || item.auto_reason || deriveReason(item, mutationCount, language);

      return {
        id: String(id),
        name: item.name,
        clone: item.clone,
        status,
        reason,
        review_reason: item.review_reason,
        llm_reason: item.llm_reason,
        auto_reason: item.auto_reason,
        error: item.error,
        identity: item.identity,
        coverage: item.coverage,
        cds_coverage: item.cds_coverage,
        sub_count: item.sub_count,
        ins_count: item.ins_count,
        del_count: item.del_count,
        sub: item.sub,
        ins: item.ins,
        dele: item.dele ?? item.del,
        aa_changes: item.aa_changes,
        aa_changes_n: item.aa_changes_n,
        avg_qry_quality: item.avg_qry_quality,
        avg_quality: item.avg_quality,
        orientation: item.orientation,
        frameshift: item.frameshift,
        mutations,
        ref_sequence: item.ref_sequence,
        query_sequence: item.query_sequence,
        aligned_ref_g: item.aligned_ref_g,
        aligned_query_g: item.aligned_query_g,
        aligned_query: item.aligned_query,
        matches: item.matches,
        cds_start: item.cds_start,
        cds_end: item.cds_end,
        traces_a: item.traces_a,
        traces_t: item.traces_t,
        traces_g: item.traces_g,
        traces_c: item.traces_c,
        quality: item.quality,
        base_locations: item.base_locations,
        mixed_peaks: item.mixed_peaks,
      };
    });
}
