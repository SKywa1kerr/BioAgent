import { useCallback, useEffect, useRef, useState } from "react";
import {
  defaultWorkbenchPrefs,
  loadWorkbenchPrefs,
  saveWorkbenchPrefs,
  type WorkbenchPrefDensity,
  type WorkbenchPrefSortDir,
  type WorkbenchPrefSortKey,
  type WorkbenchPrefStatusFilter,
  type WorkbenchPrefSummaryScope,
  type WorkbenchPrefs,
} from "../lib/ui/workbenchPrefs.js";

export type SummaryScope = WorkbenchPrefSummaryScope;

export interface PersistedControls {
  statusFilter: WorkbenchPrefStatusFilter;
  searchQuery: string;
  sortKey: WorkbenchPrefSortKey;
  sortDir: WorkbenchPrefSortDir;
  density: WorkbenchPrefDensity;
  summaryScope: SummaryScope;
}

export type WorkbenchControls = PersistedControls;

export const DEFAULT_CONTROLS: Readonly<PersistedControls> = Object.freeze({
  statusFilter: defaultWorkbenchPrefs.statusFilter,
  searchQuery: "",
  sortKey: defaultWorkbenchPrefs.sortKey,
  sortDir: defaultWorkbenchPrefs.sortDir,
  density: defaultWorkbenchPrefs.density,
  summaryScope: defaultWorkbenchPrefs.summaryScope,
});

const WRITE_DEBOUNCE_MS = 300;

function getSafeStorage(): Storage | null {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}

function controlsToPrefs(controls: PersistedControls): WorkbenchPrefs {
  return {
    sortKey: controls.sortKey,
    sortDir: controls.sortDir,
    density: controls.density,
    statusFilter: controls.statusFilter,
    summaryScope: controls.summaryScope,
  };
}

export function useWorkbenchControls() {
  const storageRef = useRef<Storage | null>(null);
  if (storageRef.current === null) storageRef.current = getSafeStorage();

  const [controls, setControls] = useState<PersistedControls>(() => {
    const store = storageRef.current;
    const prefs = store ? loadWorkbenchPrefs(store) : { ...defaultWorkbenchPrefs };
    return {
      ...DEFAULT_CONTROLS,
      statusFilter: prefs.statusFilter,
      sortKey: prefs.sortKey,
      sortDir: prefs.sortDir,
      density: prefs.density,
      summaryScope: prefs.summaryScope,
    };
  });

  const writeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const store = storageRef.current;
    if (!store) return;
    if (writeTimerRef.current) clearTimeout(writeTimerRef.current);
    writeTimerRef.current = setTimeout(
      () => saveWorkbenchPrefs(controlsToPrefs(controls), store),
      WRITE_DEBOUNCE_MS,
    );
    return () => {
      if (writeTimerRef.current) clearTimeout(writeTimerRef.current);
    };
  }, [
    controls.statusFilter,
    controls.sortKey,
    controls.sortDir,
    controls.density,
    controls.summaryScope,
  ]);

  const setStatusFilter = useCallback((statusFilter: PersistedControls["statusFilter"]) => {
    setControls((prev) => ({ ...prev, statusFilter }));
  }, []);
  const setSearchQuery = useCallback((searchQuery: string) => {
    setControls((prev) => ({ ...prev, searchQuery }));
  }, []);
  const setSortKey = useCallback((sortKey: PersistedControls["sortKey"]) => {
    setControls((prev) => ({ ...prev, sortKey }));
  }, []);
  const setSortDir = useCallback((sortDir: PersistedControls["sortDir"]) => {
    setControls((prev) => ({ ...prev, sortDir }));
  }, []);
  const setDensity = useCallback((density: PersistedControls["density"]) => {
    setControls((prev) => ({ ...prev, density }));
  }, []);
  const setSummaryScope = useCallback((summaryScope: SummaryScope) => {
    setControls((prev) => ({ ...prev, summaryScope }));
  }, []);
  const reset = useCallback(() => setControls({ ...DEFAULT_CONTROLS }), []);

  return {
    controls,
    setStatusFilter,
    setSearchQuery,
    setSortKey,
    setSortDir,
    setDensity,
    setSummaryScope,
    reset,
  };
}
