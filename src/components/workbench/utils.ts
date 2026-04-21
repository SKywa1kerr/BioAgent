import type { WorkbenchSample, WorkbenchStatus } from "./types";

export type ResultsStatusFilter = "all" | WorkbenchStatus;
export type ResultsSortKey = "status" | "sample" | "identity" | "coverage" | "mutations";

const UNTESTED_REASON = "???";

const STATUS_PRIORITY: Record<WorkbenchStatus, number> = {
  wrong: 0,
  uncertain: 1,
  untested: 2,
  ok: 3,
};

function normalizePercent(value?: number) {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0;
  return value <= 1.5 ? value * 100 : value;
}

function compareText(a: string, b: string) {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

export function bucketSampleStatus(sample: WorkbenchSample): WorkbenchStatus {
  if (sample.bucket === "ok" || sample.bucket === "wrong"
      || sample.bucket === "uncertain" || sample.bucket === "untested") {
    return sample.bucket;
  }
  if (sample.reason === UNTESTED_REASON) return "untested";
  if (sample.status === "ok" || sample.status === "wrong") return sample.status;
  return "uncertain";
}

export function countSampleMutations(sample: WorkbenchSample) {
  const explicit =
    (sample.sub_count ?? sample.sub ?? 0) +
    (sample.ins_count ?? sample.ins ?? 0) +
    (sample.del_count ?? sample.dele ?? 0);
  if (explicit > 0) return explicit;
  return Array.isArray(sample.mutations) ? sample.mutations.length : 0;
}

function matchesSearch(sample: WorkbenchSample, searchQuery: string) {
  if (!searchQuery.trim()) return true;
  const q = searchQuery.trim().toLowerCase();
  const haystack = [
    sample.id,
    sample.name,
    sample.clone,
    sample.reason,
    sample.review_reason,
    sample.llm_reason,
    sample.auto_reason,
    sample.error,
  ]
    .filter((v): v is string => typeof v === "string" && v.trim().length > 0)
    .join(" ")
    .toLowerCase();
  return haystack.includes(q);
}

function compareByStatus(a: WorkbenchSample, b: WorkbenchSample) {
  const d = STATUS_PRIORITY[bucketSampleStatus(a)] - STATUS_PRIORITY[bucketSampleStatus(b)];
  if (d !== 0) return d;
  return compareText(a.id, b.id);
}

export function buildResultsView(
  samples: WorkbenchSample[],
  options: { statusFilter: ResultsStatusFilter; searchQuery: string; sortKey: ResultsSortKey },
) {
  const filtered = samples.filter((sample) => {
    const status = bucketSampleStatus(sample);
    const matchesStatus = options.statusFilter === "all" || status === options.statusFilter;
    return matchesStatus && matchesSearch(sample, options.searchQuery);
  });

  return filtered.sort((left, right) => {
    switch (options.sortKey) {
      case "identity": {
        const d = normalizePercent(right.identity) - normalizePercent(left.identity);
        return d !== 0 ? d : compareText(left.id, right.id);
      }
      case "coverage": {
        const d = normalizePercent(right.cds_coverage ?? right.coverage) - normalizePercent(left.cds_coverage ?? left.coverage);
        return d !== 0 ? d : compareText(left.id, right.id);
      }
      case "mutations": {
        const d = countSampleMutations(right) - countSampleMutations(left);
        return d !== 0 ? d : compareByStatus(left, right);
      }
      case "sample":
        return compareText(left.id, right.id);
      case "status":
      default:
        return compareByStatus(left, right);
    }
  });
}

export function formatPercent(value?: number) {
  const normalized = normalizePercent(value);
  return `${normalized.toFixed(1)}%`;
}
