export const CONTROLS_STORAGE_KEY = "bioagent-workbench-controls-v1";

export const DEFAULT_CONTROLS = Object.freeze({
  statusFilter: "all",
  searchQuery: "",
  sortKey: "status",
  summaryScope: "filtered",
});

const VALID_STATUS = new Set(["all", "ok", "wrong", "uncertain", "untested"]);
const VALID_SORT = new Set(["status", "sample", "identity", "coverage", "mutations"]);
const VALID_SCOPE = new Set(["filtered", "all"]);

export function validateControls(value) {
  if (!value || typeof value !== "object") return null;
  const { statusFilter, searchQuery, sortKey, summaryScope } = value;
  if (!VALID_STATUS.has(statusFilter)) return null;
  if (typeof searchQuery !== "string") return null;
  if (!VALID_SORT.has(sortKey)) return null;
  if (!VALID_SCOPE.has(summaryScope)) return null;
  return { statusFilter, searchQuery, sortKey, summaryScope };
}

export function readControls(storage) {
  try {
    const raw = storage.getItem(CONTROLS_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_CONTROLS };
    const parsed = JSON.parse(raw);
    const valid = validateControls(parsed);
    return valid ?? { ...DEFAULT_CONTROLS };
  } catch {
    return { ...DEFAULT_CONTROLS };
  }
}

export function writeControls(storage, value) {
  try {
    const valid = validateControls(value);
    if (!valid) return;
    storage.setItem(CONTROLS_STORAGE_KEY, JSON.stringify(valid));
  } catch {
    /* ignore quota / privacy-mode errors */
  }
}
