export type SaveEncoding = "utf8" | "base64";

export interface SaveFileArgs {
  filename: string;
  mime: string;
  data: string;
  encoding?: SaveEncoding;
}

export interface SaveFileResult {
  filePath?: string;
  canceled: boolean;
  error?: string;
}

export async function saveFile(args: SaveFileArgs): Promise<SaveFileResult> {
  const api = typeof window !== "undefined" ? window.electronAPI : undefined;
  if (api?.invoke) {
    try {
      const filters = filtersForExt(args.filename);
      const result = (await api.invoke("export-save-file", {
        defaultPath: args.filename,
        filters,
        data: args.data,
        encoding: args.encoding ?? "utf8",
      })) as SaveFileResult | undefined;
      if (result && typeof result === "object") return result;
    } catch (err) {
      console.warn("Electron save failed, falling back to Blob:", err);
    }
  }
  downloadAsBlob(args);
  return { canceled: false };
}

function filtersForExt(filename: string) {
  const ext = (filename.split(".").pop() ?? "").toLowerCase();
  const labels: Record<string, string> = { csv: "CSV", json: "JSON", pdf: "PDF" };
  return [{ name: labels[ext] ?? ext.toUpperCase(), extensions: [ext] }];
}

function downloadAsBlob({ filename, mime, data, encoding }: SaveFileArgs) {
  const part: BlobPart =
    encoding === "base64" ? base64ToBuffer(data) : data;
  const blob = new Blob([part], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function base64ToBuffer(b64: string): ArrayBuffer {
  const binary = atob(b64);
  const buffer = new ArrayBuffer(binary.length);
  const view = new Uint8Array(buffer);
  for (let i = 0; i < binary.length; i++) view[i] = binary.charCodeAt(i);
  return buffer;
}
