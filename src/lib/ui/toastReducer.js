export const MAX_VISIBLE = 5;
export const DEFAULT_DURATION_MS = 5000;

export const initialToastState = { toasts: [], nextId: 1 };

export function pushToast(payload) {
  return { type: "push", payload };
}

export function dismissToast(id) {
  return { type: "dismiss", id };
}

export function clearAllToasts() {
  return { type: "clearAll" };
}

export function toastReducer(state, action) {
  switch (action.type) {
    case "push": {
      const { payload } = action;
      const durationMs = payload.durationMs === undefined ? DEFAULT_DURATION_MS : payload.durationMs;
      const toast = {
        id: `toast-${state.nextId}`,
        kind: payload.kind,
        title: payload.title,
        durationMs,
      };
      if (payload.description !== undefined) toast.description = payload.description;
      if (payload.action !== undefined) toast.action = payload.action;
      const next = state.toasts.concat(toast);
      const trimmed = next.length > MAX_VISIBLE ? next.slice(next.length - MAX_VISIBLE) : next;
      return { toasts: trimmed, nextId: state.nextId + 1 };
    }
    case "dismiss": {
      const filtered = state.toasts.filter((t) => t.id !== action.id);
      if (filtered.length === state.toasts.length) return state;
      return { toasts: filtered, nextId: state.nextId };
    }
    case "clearAll": {
      if (state.toasts.length === 0) return state;
      return { toasts: [], nextId: state.nextId };
    }
    default:
      return state;
  }
}
