import type { WorkbenchSample } from "../../components/workbench/types";

export interface BuildDocArgs {
  samples: WorkbenchSample[];
  filters?: { statusFilter: string; searchQuery: string; sortKey: string } | null;
  dataset?: string;
  detailMode: boolean;
  stringsFn?: (key: string, params?: Record<string, string | number>) => string;
  date?: Date;
}

export function buildDocDefinition(args: BuildDocArgs): unknown;

export const PDF_LIMITS: Readonly<{
  MAX_DETAIL_SAMPLES: number;
  MAX_REASON_CHARS: number;
}>;
