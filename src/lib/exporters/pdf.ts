import type { WorkbenchSample } from "../../components/workbench/types";
import type { AppLanguage } from "../../i18n";
import { t } from "../../i18n";
import { buildDocDefinition, PDF_LIMITS } from "./pdfDoc.js";
import { buildExportFilename } from "./filename";
import { saveFile } from "./saveFile";

export interface ExportPdfArgs {
  samples: WorkbenchSample[];
  filters: { statusFilter: string; searchQuery: string; sortKey: string };
  dataset?: string;
  language: AppLanguage;
  onWarn?: (message: string) => void;
}

let pdfMakeRef: any = null;
let fontsLoaded = false;

async function ensurePdfMake(): Promise<any> {
  if (pdfMakeRef && fontsLoaded) return pdfMakeRef;
  const mod: any = await import("pdfmake/build/pdfmake");
  const pdfMake = mod.default ?? mod;
  if (!pdfMake?.createPdf) throw new Error("pdfmake module did not load correctly.");

  if (!fontsLoaded) {
    const [regular, bold] = await Promise.all([
      fetchAsBase64("/fonts/NotoSansSC-Regular-subset.otf"),
      fetchAsBase64("/fonts/NotoSansSC-Bold-subset.otf"),
    ]);
    pdfMake.vfs = {
      ...(pdfMake.vfs ?? {}),
      "NotoSansSC-Regular.otf": regular,
      "NotoSansSC-Bold.otf": bold,
    };
    pdfMake.fonts = {
      ...(pdfMake.fonts ?? {}),
      NotoSansSC: {
        normal: "NotoSansSC-Regular.otf",
        bold: "NotoSansSC-Bold.otf",
        italics: "NotoSansSC-Regular.otf",
        bolditalics: "NotoSansSC-Bold.otf",
      },
    };
    fontsLoaded = true;
  }

  pdfMakeRef = pdfMake;
  return pdfMake;
}

async function fetchAsBase64(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Font fetch failed: ${url} (${res.status})`);
  const buf = await res.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(buf);
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + chunk)));
  }
  return btoa(binary);
}

export async function exportPdf(args: ExportPdfArgs): Promise<void> {
  const detailMode = args.samples.length <= PDF_LIMITS.MAX_DETAIL_SAMPLES;
  if (!detailMode && args.onWarn) {
    args.onWarn(t(args.language, "export.warn.bigBatch", { count: args.samples.length }));
  }

  const pdfMake = await ensurePdfMake();
  const doc = buildDocDefinition({
    samples: args.samples,
    filters: args.filters,
    dataset: args.dataset,
    detailMode,
    stringsFn: (key, params) => t(args.language, key, params),
  });
  const filename = buildExportFilename({ dataset: args.dataset, ext: "pdf" });

  await new Promise<void>((resolve, reject) => {
    try {
      pdfMake.createPdf(doc).getBase64(async (base64: string) => {
        try {
          await saveFile({
            filename,
            mime: "application/pdf",
            data: base64,
            encoding: "base64",
          });
          resolve();
        } catch (err) {
          reject(err);
        }
      });
    } catch (err) {
      reject(err);
    }
  });
}
