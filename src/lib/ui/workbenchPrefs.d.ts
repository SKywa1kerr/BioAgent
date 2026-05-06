export type WorkbenchPrefSortKey =
  | "status"
  | "sample"
  | "identity"
  | "coverage"
  | "mutations";

export type WorkbenchPrefSortDir = "asc" | "desc";

export type WorkbenchPrefDensity = "compact" | "detailed";

export type WorkbenchPrefStatusFilter =
  | "all"
  | "ok"
  | "wrong"
  | "uncertain"
  | "untested";

export type WorkbenchPrefSummaryScope = "filtered" | "all";

export interface WorkbenchPrefs {
  sortKey: WorkbenchPrefSortKey;
  sortDir: WorkbenchPrefSortDir;
  density: WorkbenchPrefDensity;
  statusFilter: WorkbenchPrefStatusFilter;
  summaryScope: WorkbenchPrefSummaryScope;
}

interface Storage {
  getItem(k: string): string | null;
  setItem(k: string, v: string): void;
}

export const defaultWorkbenchPrefs: Readonly<WorkbenchPrefs>;

export function loadWorkbenchPrefs(store?: Storage): WorkbenchPrefs;
export function saveWorkbenchPrefs(prefs: WorkbenchPrefs, store?: Storage): void;
