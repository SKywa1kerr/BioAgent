import type { DatasetImportState } from "../types";

type DatasetFolderName = DatasetImportState["missing"][number];

function normalizePath(input: string) {
  return input.replace(/\\/g, "/").replace(/\/+$/, "");
}

function joinDatasetPath(baseDir: string, folder: DatasetFolderName) {
  return `${baseDir}/${folder}`;
}

export function resolveDatasetPaths(
  datasetDir: string,
  existingDirs: Set<string> = new Set<string>()
): DatasetImportState {
  const normalizedDatasetDir = normalizePath(datasetDir);
  const knownDirs = new Set(Array.from(existingDirs, normalizePath));
  const ab1Dir = joinDatasetPath(normalizedDatasetDir, "ab1");
  const gbDir = joinDatasetPath(normalizedDatasetDir, "gb");

  const resolvedAb1 = knownDirs.has(ab1Dir) ? ab1Dir : null;
  const resolvedGb = knownDirs.has(gbDir) ? gbDir : null;
  const missing: DatasetImportState["missing"] = [];

  if (!resolvedAb1) {
    missing.push("ab1");
  }
  if (!resolvedGb) {
    missing.push("gb");
  }

  return {
    datasetDir: normalizedDatasetDir,
    datasetName: normalizedDatasetDir.split(/[/\\]/).filter(Boolean).pop() ?? normalizedDatasetDir,
    ab1Dir: resolvedAb1,
    gbDir: resolvedGb,
    missing,
    valid: Boolean(resolvedAb1 || resolvedGb),
  };
}
