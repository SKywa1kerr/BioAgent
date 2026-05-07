import { useCallback, useEffect, useState } from "react";

export type UpdaterPhase =
  | "idle"
  | "checking"
  | "available"
  | "downloading"
  | "ready"
  | "up-to-date"
  | "error";

export interface UpdaterState {
  phase: UpdaterPhase;
  version?: string;
  percent?: number;
  message?: string;
}

interface UpdaterStatePayload {
  phase: UpdaterPhase;
  version?: string;
  percent?: number;
  bytesPerSecond?: number;
  message?: string;
}

interface ElectronUpdaterBridge {
  quitAndInstall: () => Promise<{ ok: boolean; error?: string }>;
  onState: (callback: (payload: UpdaterStatePayload) => void) => () => void;
}

declare global {
  interface Window {
    electronUpdater?: ElectronUpdaterBridge;
  }
}

const INITIAL_STATE: UpdaterState = { phase: "idle" };

export function useUpdater(): { state: UpdaterState; install: () => void } {
  const [state, setState] = useState<UpdaterState>(INITIAL_STATE);

  useEffect(() => {
    const bridge = typeof window !== "undefined" ? window.electronUpdater : undefined;
    if (!bridge) return;

    const off = bridge.onState((payload) => {
      setState({
        phase: payload.phase,
        version: payload.version,
        percent: payload.percent,
        message: payload.message,
      });
    });
    return () => {
      try { off?.(); } catch { /* ignore */ }
    };
  }, []);

  const install = useCallback(() => {
    const bridge = typeof window !== "undefined" ? window.electronUpdater : undefined;
    if (!bridge) return;
    void bridge.quitAndInstall().catch(() => { /* ignore */ });
  }, []);

  return { state, install };
}
