export function formatStamp(date?: Date): string;
export function sanitizeSegment(s: string | undefined | null): string;

export interface BuildExportFilenameArgs {
  dataset?: string;
  ext: string;
  date?: Date;
}
export function buildExportFilename(args: BuildExportFilenameArgs): string;
