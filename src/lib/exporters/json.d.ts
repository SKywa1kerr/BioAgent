import type { WorkbenchSample } from "../../components/workbench/types";

export interface ExportFilters {
  statusFilter: string;
  searchQuery: string;
  sortKey: string;
}

export interface SamplesToJsonOptions {
  filters?: ExportFilters;
  date?: Date;
}

export function samplesToJson(samples: WorkbenchSample[], options?: SamplesToJsonOptions): string;
