// Pure helper for the chat-column splitter state machine.
//
// Width state lives in localStorage as JSON: { width, collapsed, lastExpandedWidth }.
// The pure `computeNextWidth` function is exported separately so it can be unit
// tested without React or the DOM.

const STORAGE_KEY = "bioagent-chat-width";

const DEFAULT_WIDTH = 320;
const COLLAPSE_THRESHOLD = 120;
const RAIL_WIDTH = 32;

const noopStore = { getItem: () => null, setItem: () => {} };

function resolveStore(store) {
  if (store) return store;
  if (typeof localStorage !== "undefined") return localStorage;
  return noopStore;
}

/** Pure: given a desired width and constraints, return clamped width and
 *  collapse flag. Caller is responsible for handing in the correct
 *  `containerWidth` and `canvasMin`.
 *
 *  Rules:
 *  - If desiredWidth < COLLAPSE_THRESHOLD -> collapse (return RAIL_WIDTH).
 *  - Otherwise clamp to [COLLAPSE_THRESHOLD, containerWidth - canvasMin]. */
export function computeNextWidth(desiredWidth, containerWidth, canvasMin) {
  if (!Number.isFinite(desiredWidth)) {
    return { width: DEFAULT_WIDTH, collapsed: false };
  }
  if (desiredWidth < COLLAPSE_THRESHOLD) {
    return { width: RAIL_WIDTH, collapsed: true };
  }
  const upper = Math.max(COLLAPSE_THRESHOLD, containerWidth - canvasMin);
  if (desiredWidth > upper) {
    return { width: upper, collapsed: false };
  }
  return { width: desiredWidth, collapsed: false };
}

export const SPLITTER_CONSTANTS = Object.freeze({
  DEFAULT_WIDTH,
  COLLAPSE_THRESHOLD,
  RAIL_WIDTH,
});

export function loadChatWidthState(store) {
  const s = resolveStore(store);
  try {
    const raw = s.getItem(STORAGE_KEY);
    if (!raw) return { width: DEFAULT_WIDTH, collapsed: false, lastExpandedWidth: DEFAULT_WIDTH };
    const parsed = JSON.parse(raw);
    const width = Number.isFinite(parsed.width) ? parsed.width : DEFAULT_WIDTH;
    const lastExpandedWidth = Number.isFinite(parsed.lastExpandedWidth)
      ? parsed.lastExpandedWidth
      : DEFAULT_WIDTH;
    return {
      width,
      collapsed: parsed.collapsed === true,
      lastExpandedWidth,
    };
  } catch {
    return { width: DEFAULT_WIDTH, collapsed: false, lastExpandedWidth: DEFAULT_WIDTH };
  }
}

export function saveChatWidthState(state, store) {
  const s = resolveStore(store);
  try {
    s.setItem(
      STORAGE_KEY,
      JSON.stringify({
        width: state.width,
        collapsed: state.collapsed,
        lastExpandedWidth: state.lastExpandedWidth,
      }),
    );
  } catch {
    // ignore quota / disabled storage
  }
}
