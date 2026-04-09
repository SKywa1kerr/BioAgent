import assert from "node:assert/strict";

const { resolveDatasetPaths } = await import("../src/utils/datasetImport.ts");

function setOf(paths) {
  return new Set(paths);
}

function testResolveDatasetPathsDetectsBothFolders() {
  const datasetDir = "D:/datasets/base";
  const result = resolveDatasetPaths(datasetDir, setOf([`${datasetDir}/ab1`, `${datasetDir}/gb`]));

  assert.deepEqual(result, {
    datasetDir,
    datasetName: "base",
    ab1Dir: `${datasetDir}/ab1`,
    gbDir: `${datasetDir}/gb`,
    missing: [],
    valid: true,
  });
}

function testResolveDatasetPathsReportsMissingGb() {
  const datasetDir = "D:/datasets/base";
  const result = resolveDatasetPaths(datasetDir, setOf([`${datasetDir}/ab1`]));

  assert.equal(result.ab1Dir, `${datasetDir}/ab1`);
  assert.equal(result.gbDir, null);
  assert.deepEqual(result.missing, ["gb"]);
  assert.equal(result.valid, true);
}

function testResolveDatasetPathsAcceptsMissingSecondArgument() {
  const result = resolveDatasetPaths("D:/datasets/empty");

  assert.deepEqual(result, {
    datasetDir: "D:/datasets/empty",
    datasetName: "empty",
    ab1Dir: null,
    gbDir: null,
    missing: ["ab1", "gb"],
    valid: false,
  });
}

testResolveDatasetPathsDetectsBothFolders();
testResolveDatasetPathsReportsMissingGb();
testResolveDatasetPathsAcceptsMissingSecondArgument();
