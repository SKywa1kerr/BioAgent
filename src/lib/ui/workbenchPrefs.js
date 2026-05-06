const STORAGE_KEY = "bioagent.workbench.prefs.v1";

const VALID_SORT_KEYS = ["status", "sample", "identity", "coverage", "mutations"];
const VALID_SORT_DIRS = ["asc", "desc"];
const VALID_DENSITY = ["compact", "detailed"];
const VALID_STATUS_FILTERS = ["all", "ok", "wrong", "uncertain", "untested"];
const VALID_SUMMARY_SCOPES = ["filtered", "all"];

export const defaultWorkbenchPrefs = Object.freeze({
  sortKey: "status",
  sortDir: "desc",
  density: "compact",
  statusFilter: "all",
  summaryScope: "filtered",
});

const noopStore = { getItem: () => null, setItem: () => {} };

function resolveStore(store) {
  if (store) return store;
  if (typeof localStorage !== "undefined") return localStorage;
  return noopStore;
}

function clamp(value, allowed, fallback) {
  return allowed.includes(value) ? value : fallback;
}

function sanitize(input) {
  if (!input || typeof input !== "object") return null;
  return {
    sortKey: clamp(input.sortKey, VALID_SORT_KEYS, defaultWorkbenchPrefs.sortKey),
    sortDir: clamp(input.sortDir, VALID_SORT_DIRS, defaultWorkbenchPrefs.sortDir),
    density: clamp(input.density, VALID_DENSITY, defaultWorkbenchPrefs.density),
    statusFilter: clamp(
      input.statusFilter,
      VALID_STATUS_FILTERS,
      defaultWorkbenchPrefs.statusFilter,
    ),
    summaryScope: clamp(
      input.summaryScope,
      VALID_SUMMARY_SCOPES,
      defaultWorkbenchPrefs.summaryScope,
    ),
  };
}

export function loadWorkbenchPrefs(store) {
  const s = resolveStore(store);
  try {
    const raw = s.getItem(STORAGE_KEY);
    if (!raw) return { ...defaultWorkbenchPrefs };
    const parsed = JSON.parse(raw);
    const valid = sanitize(parsed);
    return valid ?? { ...defaultWorkbenchPrefs };
  } catch {
    return { ...defaultWorkbenchPrefs };
  }
}

export function saveWorkbenchPrefs(prefs, store) {
  const s = resolveStore(store);
  const valid = sanitize(prefs);
  if (!valid) return;
  try {
    s.setItem(STORAGE_KEY, JSON.stringify(valid));
  } catch {
    // ignore quota / disabled storage
  }
}
