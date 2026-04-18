import { useCallback, useEffect, useRef, useState } from "react";
import {
  CONTROLS_STORAGE_KEY,
  DEFAULT_CONTROLS,
  readControls,
  writeControls,
  validateControls,
  type PersistedControls,
  type SummaryScope,
} from "./workbenchControlsStore.js";

export { CONTROLS_STORAGE_KEY, DEFAULT_CONTROLS, validateControls };
export type { PersistedControls, SummaryScope };
export type WorkbenchControls = PersistedControls;

const WRITE_DEBOUNCE_MS = 300;

function getSafeStorage(): Storage | null {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}

export function useWorkbenchControls() {
  const storageRef = useRef<Storage | null>(null);
  if (storageRef.current === null) storageRef.current = getSafeStorage();

  const [controls, setControls] = useState<PersistedControls>(() => {
    const store = storageRef.current;
    return store ? readControls(store) : { ...DEFAULT_CONTROLS };
  });

  const writeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const store = storageRef.current;
    if (!store) return;
    if (writeTimerRef.current) clearTimeout(writeTimerRef.current);
    writeTimerRef.current = setTimeout(() => writeControls(store, controls), WRITE_DEBOUNCE_MS);
    return () => {
      if (writeTimerRef.current) clearTimeout(writeTimerRef.current);
    };
  }, [controls]);

  const setStatusFilter = useCallback((statusFilter: PersistedControls["statusFilter"]) => {
    setControls((prev) => ({ ...prev, statusFilter }));
  }, []);
  const setSearchQuery = useCallback((searchQuery: string) => {
    setControls((prev) => ({ ...prev, searchQuery }));
  }, []);
  const setSortKey = useCallback((sortKey: PersistedControls["sortKey"]) => {
    setControls((prev) => ({ ...prev, sortKey }));
  }, []);
  const setSummaryScope = useCallback((summaryScope: SummaryScope) => {
    setControls((prev) => ({ ...prev, summaryScope }));
  }, []);
  const reset = useCallback(() => setControls({ ...DEFAULT_CONTROLS }), []);

  return { controls, setStatusFilter, setSearchQuery, setSortKey, setSummaryScope, reset };
}
