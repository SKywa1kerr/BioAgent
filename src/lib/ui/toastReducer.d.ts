export type ToastKind = "info" | "error" | "success" | "warning";

export interface ToastAction {
  label: string;
  actionId: string;
}

export interface Toast {
  id: string;
  kind: ToastKind;
  title: string;
  description?: string;
  durationMs: number;
  action?: ToastAction;
}

export interface ToastState {
  toasts: Toast[];
  nextId: number;
}

export interface ToastPushPayload {
  kind: ToastKind;
  title: string;
  description?: string;
  durationMs?: number;
  action?: ToastAction;
}

export type ToastReducerAction =
  | { type: "push"; payload: ToastPushPayload }
  | { type: "dismiss"; id: string }
  | { type: "clearAll" };

export const initialToastState: ToastState;
export const MAX_VISIBLE: number;
export const DEFAULT_DURATION_MS: number;

export function pushToast(payload: ToastPushPayload): ToastReducerAction;
export function dismissToast(id: string): ToastReducerAction;
export function clearAllToasts(): ToastReducerAction;

export function toastReducer(state: ToastState, action: ToastReducerAction): ToastState;
