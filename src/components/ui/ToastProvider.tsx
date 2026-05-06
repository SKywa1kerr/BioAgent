import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from "react";
import {
  toastReducer,
  initialToastState,
  pushToast as pushToastAction,
  dismissToast as dismissToastAction,
  clearAllToasts as clearAllToastsAction,
} from "../../lib/ui/toastReducer.js";
import type { ToastPushPayload, ToastState } from "../../lib/ui/toastReducer";
import { ToastContainer } from "./ToastContainer";
import "./Toast.css";

export interface ToastApi {
  pushToast: (payload: ToastPushPayload) => string;
  dismissToast: (id: string) => void;
  clearAllToasts: () => void;
  registerActionHandler: (
    actionId: string,
    handler: (toastId: string) => void,
  ) => () => void;
}

type ActionDispatcher = (actionId: string, toastId: string) => void;

const ToastApiContext = createContext<ToastApi | null>(null);
const ToastStateContext = createContext<ToastState | null>(null);
const ToastActionDispatchContext = createContext<ActionDispatcher | null>(null);

export function ToastProvider({ children }: { children: ReactNode }): JSX.Element {
  const [state, dispatch] = useReducer(toastReducer, initialToastState);

  const nextIdRef = useRef<number>(initialToastState.nextId);
  const timersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const handlersRef = useRef<Map<string, (toastId: string) => void>>(new Map());

  const clearTimerFor = useCallback((id: string) => {
    const timers = timersRef.current;
    const existing = timers.get(id);
    if (existing !== undefined) {
      clearTimeout(existing);
      timers.delete(id);
    }
  }, []);

  const dismissToast = useCallback(
    (id: string) => {
      clearTimerFor(id);
      dispatch(dismissToastAction(id));
    },
    [clearTimerFor],
  );

  const clearAllToasts = useCallback(() => {
    const timers = timersRef.current;
    timers.forEach((handle) => clearTimeout(handle));
    timers.clear();
    dispatch(clearAllToastsAction());
  }, []);

  const pushToast = useCallback(
    (payload: ToastPushPayload): string => {
      const assignedId = `toast-${nextIdRef.current}`;
      nextIdRef.current += 1;
      dispatch(pushToastAction(payload));
      const durationMs = payload.durationMs === undefined ? 5000 : payload.durationMs;
      if (durationMs > 0) {
        const handle = setTimeout(() => {
          timersRef.current.delete(assignedId);
          dispatch(dismissToastAction(assignedId));
        }, durationMs);
        timersRef.current.set(assignedId, handle);
      }
      return assignedId;
    },
    [],
  );

  // Reconcile timers with state: if reducer dropped a toast (FIFO cap), clear its timer.
  useEffect(() => {
    const live = new Set(state.toasts.map((t) => t.id));
    const timers = timersRef.current;
    timers.forEach((handle, id) => {
      if (!live.has(id)) {
        clearTimeout(handle);
        timers.delete(id);
      }
    });
  }, [state.toasts]);

  // Cleanup all timers when provider unmounts.
  useEffect(() => {
    const timers = timersRef.current;
    return () => {
      timers.forEach((handle) => clearTimeout(handle));
      timers.clear();
    };
  }, []);

  const registerActionHandler = useCallback(
    (actionId: string, handler: (toastId: string) => void) => {
      handlersRef.current.set(actionId, handler);
      return () => {
        const current = handlersRef.current.get(actionId);
        if (current === handler) handlersRef.current.delete(actionId);
      };
    },
    [],
  );

  const dispatchAction = useCallback<ActionDispatcher>((actionId, toastId) => {
    const handler = handlersRef.current.get(actionId);
    if (handler) handler(toastId);
  }, []);

  const api = useMemo<ToastApi>(
    () => ({ pushToast, dismissToast, clearAllToasts, registerActionHandler }),
    [pushToast, dismissToast, clearAllToasts, registerActionHandler],
  );

  return (
    <ToastApiContext.Provider value={api}>
      <ToastStateContext.Provider value={state}>
        <ToastActionDispatchContext.Provider value={dispatchAction}>
          {children}
          <ToastContainer />
        </ToastActionDispatchContext.Provider>
      </ToastStateContext.Provider>
    </ToastApiContext.Provider>
  );
}

export function useToasts(): ToastApi {
  const value = useContext(ToastApiContext);
  if (!value) {
    throw new Error("useToasts must be used within a ToastProvider");
  }
  return value;
}

export function useToastsState(): ToastState {
  const value = useContext(ToastStateContext);
  if (!value) {
    throw new Error("useToastsState must be used within a ToastProvider");
  }
  return value;
}

export function useToastActionDispatcher(): ActionDispatcher {
  const value = useContext(ToastActionDispatchContext);
  if (!value) {
    throw new Error("useToastActionDispatcher must be used within a ToastProvider");
  }
  return value;
}
