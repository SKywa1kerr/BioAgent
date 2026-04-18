import type { ResultsSortKey, ResultsStatusFilter } from "../components/workbench/utils";

export const CONTROLS_STORAGE_KEY: string;

export type SummaryScope = "filtered" | "all";

export interface PersistedControls {
  statusFilter: ResultsStatusFilter;
  searchQuery: string;
  sortKey: ResultsSortKey;
  summaryScope: SummaryScope;
}

export const DEFAULT_CONTROLS: Readonly<PersistedControls>;

export function validateControls(value: unknown): PersistedControls | null;

export interface ControlsStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function readControls(storage: ControlsStorage): PersistedControls;
export function writeControls(storage: ControlsStorage, value: PersistedControls): void;
