const STORAGE_KEY = "bioagent.chatRail.v1";
const ORDER = ["wide", "narrow", "hidden"];

const noopStore = { getItem: () => null, setItem: () => {} };

function resolveStore(store) {
  if (store) return store;
  if (typeof localStorage !== "undefined") return localStorage;
  return noopStore;
}

export function nextRailState(state) {
  const i = ORDER.indexOf(state);
  if (i < 0) return "wide";
  return ORDER[(i + 1) % ORDER.length];
}

export function loadRailState(store) {
  const s = resolveStore(store);
  const raw = s.getItem(STORAGE_KEY);
  return ORDER.includes(raw) ? raw : "wide";
}

export function saveRailState(state, store) {
  const s = resolveStore(store);
  s.setItem(STORAGE_KEY, state);
}
