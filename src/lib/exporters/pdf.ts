import type { WorkbenchSample } from "../../components/workbench/types";
import type { AppLanguage } from "../../i18n";

export interface ExportPdfArgs {
  samples: WorkbenchSample[];
  filters: { statusFilter: string; searchQuery: string; sortKey: string };
  dataset?: string;
  language: AppLanguage;
  onWarn?: (message: string) => void;
}

export async function exportPdf(_args: ExportPdfArgs): Promise<void> {
  throw new Error("PDF export not yet wired up (Phase 3).");
}
