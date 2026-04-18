import type { WorkbenchSample } from "../../components/workbench/types";
import type { AppLanguage } from "../../i18n";
import { samplesToCsv } from "./csv";
import { samplesToJson } from "./json";
import { buildExportFilename } from "./filename";
import { saveFile } from "./saveFile";

export type ExportFormat = "csv" | "json" | "pdf";

export interface RunExportArgs {
  samples: WorkbenchSample[];
  filters: { statusFilter: string; searchQuery: string; sortKey: string };
  dataset?: string;
  language: AppLanguage;
  onWarn?: (message: string) => void;
}

export async function runExport(fmt: ExportFormat, args: RunExportArgs): Promise<void> {
  if (fmt === "csv") {
    await saveFile({
      filename: buildExportFilename({ dataset: args.dataset, ext: "csv" }),
      mime: "text/csv;charset=utf-8",
      data: samplesToCsv(args.samples),
    });
    return;
  }
  if (fmt === "json") {
    await saveFile({
      filename: buildExportFilename({ dataset: args.dataset, ext: "json" }),
      mime: "application/json;charset=utf-8",
      data: samplesToJson(args.samples, { filters: args.filters }),
    });
    return;
  }
  if (fmt === "pdf") {
    const { exportPdf } = await import("./pdf");
    await exportPdf(args);
    return;
  }
  throw new Error(`Unknown export format: ${fmt}`);
}
