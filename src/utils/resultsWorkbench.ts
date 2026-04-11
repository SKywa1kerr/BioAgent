import type { ResultWorkbenchStatus, Sample } from "../types";

export type ResultsStatusFilter = "all" | ResultWorkbenchStatus;
export type ResultsSortKey = "status" | "sample" | "identity" | "coverage" | "mutations";

const UNTESTED_REASON = "未测通";

const STATUS_PRIORITY: Record<ResultWorkbenchStatus, number> = {
  wrong: 0,
  uncertain: 1,
  untested: 2,
  ok: 3,
};

function compareText(a: string, b: string) {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

export function bucketSampleStatus(sample: Sample): ResultWorkbenchStatus {
  if (sample.reason === UNTESTED_REASON) {
    return "untested";
  }

  if (sample.status === "ok" || sample.status === "wrong") {
    return sample.status;
  }

  return "uncertain";
}

export function countSampleMutations(sample: Sample) {
  const explicitCount =
    (sample.sub_count ?? sample.sub ?? 0) +
    (sample.ins_count ?? sample.ins ?? 0) +
    (sample.del_count ?? sample.dele ?? 0);

  if (explicitCount > 0) {
    return explicitCount;
  }

  return Array.isArray(sample.mutations) ? sample.mutations.length : 0;
}

function matchesSearch(sample: Sample, searchQuery: string) {
  if (!searchQuery.trim()) {
    return true;
  }

  const normalizedQuery = searchQuery.trim().toLowerCase();
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
    .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
    .join(" ")
    .toLowerCase();

  return haystack.includes(normalizedQuery);
}

function compareSamplesByStatus(a: Sample, b: Sample) {
  const toneDelta = STATUS_PRIORITY[bucketSampleStatus(a)] - STATUS_PRIORITY[bucketSampleStatus(b)];
  if (toneDelta !== 0) {
    return toneDelta;
  }

  return compareText(a.id, b.id);
}

export function buildResultsView(
  samples: Sample[],
  options: {
    statusFilter: ResultsStatusFilter;
    searchQuery: string;
    sortKey: ResultsSortKey;
  }
) {
  const filtered = samples.filter((sample) => {
    const sampleStatus = bucketSampleStatus(sample);
    const matchesStatus = options.statusFilter === "all" || sampleStatus === options.statusFilter;
    return matchesStatus && matchesSearch(sample, options.searchQuery);
  });

  return filtered.sort((left, right) => {
    switch (options.sortKey) {
      case "identity": {
        const delta = (right.identity || 0) - (left.identity || 0);
        return delta !== 0 ? delta : compareText(left.id, right.id);
      }
      case "coverage": {
        const delta = (right.coverage || 0) - (left.coverage || 0);
        return delta !== 0 ? delta : compareText(left.id, right.id);
      }
      case "mutations": {
        const delta = countSampleMutations(right) - countSampleMutations(left);
        return delta !== 0 ? delta : compareSamplesByStatus(left, right);
      }
      case "sample":
        return compareText(left.id, right.id);
      case "status":
      default:
        return compareSamplesByStatus(left, right);
    }
  });
}
