import type { ChromatogramData, WorkbenchMutation, WorkbenchSample } from "./types";
import { t, type AppLanguage } from "../../i18n";

function firstDefined<T>(...values: T[]): T | undefined {
  return values.find((value) => value !== undefined && value !== null);
}

function toNumber(value: any): number | undefined {
  if (typeof value === "number") return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function toArray<T = any>(value: T[] | undefined | null): T[] | undefined {
  return Array.isArray(value) ? value : undefined;
}

function getIdentity(item: any): number {
  return toNumber(item?.identity) ?? 0;
}

function getCoverage(item: any): number {
  return toNumber(firstDefined(item?.cds_coverage, item?.cdsCoverage, item?.coverage)) ?? 0;
}

export function normalizeMutation(item: any): WorkbenchMutation {
  return {
    position: toNumber(firstDefined(item?.position, item?.ref_pos, item?.refPos)),
    refBase: firstDefined(item?.refBase, item?.ref_base),
    queryBase: firstDefined(item?.queryBase, item?.qry_base, item?.query_base),
    type: item?.type,
    effect: item?.effect,
  };
}

function deriveStatus(item: any, mutationCount: number): "ok" | "wrong" | "uncertain" {
  const identity = getIdentity(item);
  const coverage = getCoverage(item);
  if (item?.frameshift) return "wrong";
  if (mutationCount > 0) return "wrong";
  if (identity >= 0.99 && coverage >= 0.8) return "ok";
  return "uncertain";
}

function deriveReason(item: any, mutationCount: number, language: AppLanguage): string {
  const identity = getIdentity(item);
  const coverage = getCoverage(item);
  if (item?.error) return String(item.error);
  if (item?.frameshift) return t(language, "analysis.reason.frameshift");
  if (mutationCount > 0) return t(language, "analysis.reason.detectedMut", { count: mutationCount });
  if (identity >= 0.99 && coverage >= 0.8) return t(language, "analysis.reason.highQuality");
  return t(language, "analysis.reason.review");
}

export function normalizeSample(item: any, idx: number, language: AppLanguage): WorkbenchSample {
  const id = firstDefined(item?.id, item?.sid, item?.name, `sample-${idx + 1}`);
  const mutations = Array.isArray(item?.mutations) ? item.mutations.map(normalizeMutation) : [];
  const mutationCount =
    (toNumber(firstDefined(item?.sub_count, item?.subCount, item?.sub)) ?? 0) +
    (toNumber(firstDefined(item?.ins_count, item?.insCount, item?.ins)) ?? 0) +
    (toNumber(firstDefined(item?.del_count, item?.delCount, item?.dele, item?.del)) ?? 0) ||
    mutations.length;
  const status = (item?.status as "ok" | "wrong" | "uncertain" | undefined) || deriveStatus(item, mutationCount);
  const reason = firstDefined(item?.reason, item?.review_reason, item?.reviewReason, item?.llm_reason, item?.llmReason, item?.auto_reason, item?.autoReason) || deriveReason(item, mutationCount, language);

  return {
    id: String(id),
    name: item?.name,
    clone: item?.clone,
    status,
    reason,
    review_reason: firstDefined(item?.review_reason, item?.reviewReason),
    llm_reason: firstDefined(item?.llm_reason, item?.llmReason),
    auto_reason: firstDefined(item?.auto_reason, item?.autoReason),
    error: item?.error,
    identity: toNumber(item?.identity),
    coverage: toNumber(item?.coverage),
    cds_coverage: toNumber(firstDefined(item?.cds_coverage, item?.cdsCoverage)),
    sub_count: toNumber(firstDefined(item?.sub_count, item?.subCount)),
    ins_count: toNumber(firstDefined(item?.ins_count, item?.insCount)),
    del_count: toNumber(firstDefined(item?.del_count, item?.delCount)),
    sub: toNumber(item?.sub),
    ins: toNumber(item?.ins),
    dele: toNumber(firstDefined(item?.dele, item?.del)),
    aa_changes: firstDefined(item?.aa_changes, item?.aaChanges),
    aa_changes_n: toNumber(firstDefined(item?.aa_changes_n, item?.aaChangesN)),
    avg_qry_quality: toNumber(firstDefined(item?.avg_qry_quality, item?.avgQryQuality)),
    avg_quality: toNumber(firstDefined(item?.avg_quality, item?.avgQuality)),
    orientation: item?.orientation,
    frameshift: item?.frameshift,
    mutations,
    ref_sequence: firstDefined(item?.ref_sequence, item?.refSequence),
    query_sequence: firstDefined(item?.query_sequence, item?.querySequence),
    aligned_ref_g: firstDefined(item?.aligned_ref_g, item?.alignedRefG),
    aligned_query_g: firstDefined(item?.aligned_query_g, item?.alignedQueryG),
    aligned_query: firstDefined(item?.aligned_query, item?.alignedQuery),
    matches: toArray<boolean>(firstDefined(item?.matches, item?.matchMask)),
    cds_start: toNumber(firstDefined(item?.cds_start, item?.cdsStart)),
    cds_end: toNumber(firstDefined(item?.cds_end, item?.cdsEnd)),
    traces_a: toArray<number>(firstDefined(item?.traces_a, item?.tracesA)),
    traces_t: toArray<number>(firstDefined(item?.traces_t, item?.tracesT)),
    traces_g: toArray<number>(firstDefined(item?.traces_g, item?.tracesG)),
    traces_c: toArray<number>(firstDefined(item?.traces_c, item?.tracesC)),
    quality: toArray<number>(item?.quality),
    base_locations: toArray<number>(firstDefined(item?.base_locations, item?.baseLocations)),
    mixed_peaks: toArray<number>(firstDefined(item?.mixed_peaks, item?.mixedPeaks)),
  };
}

export function buildChromatogramData(sample: Partial<WorkbenchSample> | any): ChromatogramData | null {
  const querySequence = firstDefined(sample?.query_sequence, sample?.querySequence);
  const tracesA = toArray<number>(firstDefined(sample?.traces_a, sample?.tracesA));
  const tracesT = toArray<number>(firstDefined(sample?.traces_t, sample?.tracesT));
  const tracesG = toArray<number>(firstDefined(sample?.traces_g, sample?.tracesG));
  const tracesC = toArray<number>(firstDefined(sample?.traces_c, sample?.tracesC));

  if (!querySequence || !tracesA || !tracesT || !tracesG || !tracesC) {
    return null;
  }

  return {
    traces: { A: tracesA, T: tracesT, G: tracesG, C: tracesC },
    quality: toArray<number>(sample?.quality) ?? [],
    baseCalls: String(querySequence),
    base_locations: toArray<number>(firstDefined(sample?.base_locations, sample?.baseLocations)) ?? [],
    mixed_peaks: toArray<number>(firstDefined(sample?.mixed_peaks, sample?.mixedPeaks)) ?? [],
  };
}

export function normalizeSamples(result: any, language: AppLanguage): WorkbenchSample[] {
  const direct = Array.isArray(result?.samples) ? result.samples : [];
  const detailSamples = Array.isArray(result?.detail?.samples) ? result.detail.samples : [];
  const payload = direct.length > 0 ? direct : detailSamples;

  return payload
    .filter((item: any) => item && typeof item === "object")
    .map((item: any, idx: number) => normalizeSample(item, idx, language));
}
