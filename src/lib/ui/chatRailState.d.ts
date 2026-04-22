export type ChatRailState = "wide" | "narrow" | "hidden";

interface Storage {
  getItem(k: string): string | null;
  setItem(k: string, v: string): void;
}

export function nextRailState(state: ChatRailState): ChatRailState;
export function loadRailState(store?: Storage): ChatRailState;
export function saveRailState(state: ChatRailState, store?: Storage): void;
