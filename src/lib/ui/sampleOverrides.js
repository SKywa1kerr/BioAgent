const STORAGE_KEY = "bioagent.sample.overrides.v1";

const VALID_STATUS = ["ok", "wrong", "uncertain"];

const noopStore = { getItem: () => null, setItem: () => {} };

function resolveStore(store) {
  if (store) return store;
  if (typeof localStorage !== "undefined") return localStorage;
  return noopStore;
}

export function getOverrideKey(analysisId, sampleId) {
  return `${String(analysisId)}::${String(sampleId)}`;
}

function sanitizeEntry(entry) {
  if (!entry || typeof entry !== "object") return null;
  if (!VALID_STATUS.includes(entry.status)) return null;
  const reason = typeof entry.reason === "string" ? entry.reason : "";
  const updatedAt = typeof entry.updatedAt === "string" && entry.updatedAt
    ? entry.updatedAt
    : new Date(0).toISOString();
  return { status: entry.status, reason, updatedAt };
}

function sanitizeMap(input) {
  if (!input || typeof input !== "object") return {};
  const out = {};
  for (const key of Object.keys(input)) {
    if (typeof key !== "string" || !key.includes("::")) continue;
    const [analysisId, sampleId] = key.split("::");
    if (!analysisId || !sampleId) continue;
    const entry = sanitizeEntry(input[key]);
    if (entry) out[key] = entry;
  }
  return out;
}

export function loadSampleOverrides(store) {
  const s = resolveStore(store);
  try {
    const raw = s.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return sanitizeMap(parsed);
  } catch {
    return {};
  }
}

export function saveSampleOverrides(map, store) {
  const s = resolveStore(store);
  const valid = sanitizeMap(map);
  try {
    s.setItem(STORAGE_KEY, JSON.stringify(valid));
  } catch {
    // ignore quota / disabled storage
  }
}
