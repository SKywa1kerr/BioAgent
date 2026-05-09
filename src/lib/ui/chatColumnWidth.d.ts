interface Storage {
  getItem(k: string): string | null;
  setItem(k: string, v: string): void;
}

export interface ChatWidthState {
  width: number;
  collapsed: boolean;
  lastExpandedWidth: number;
}

export interface ComputeNextWidthResult {
  width: number;
  collapsed: boolean;
}

export const SPLITTER_CONSTANTS: {
  readonly DEFAULT_WIDTH: number;
  readonly COLLAPSE_THRESHOLD: number;
  readonly RAIL_WIDTH: number;
};

export function computeNextWidth(
  desiredWidth: number,
  containerWidth: number,
  canvasMin: number,
): ComputeNextWidthResult;

export function loadChatWidthState(store?: Storage): ChatWidthState;
export function saveChatWidthState(state: ChatWidthState, store?: Storage): void;
