import test from "node:test";
import assert from "node:assert/strict";
import {
  buildChromatogramData,
  normalizeMutation,
  normalizeSample,
  normalizeSamples,
} from "../src/components/workbench/normalize.js";

test("normalizeMutation accepts snake_case and camelCase aliases", () => {
  assert.deepEqual(
    normalizeMutation({ position: 7, ref_base: "A", qry_base: "G", type: "substitution", effect: "synonymous" }),
    { position: 7, refBase: "A", queryBase: "G", type: "substitution", effect: "synonymous" },
  );
  assert.deepEqual(
    normalizeMutation({ refPos: 9, refBase: "T", queryBase: "C" }),
    { position: 9, refBase: "T", queryBase: "C", type: undefined, effect: undefined },
  );
});

test("normalizeSample preserves current workbench shape while accepting camelCase aliases", () => {
  const sample = normalizeSample({
    sid: "C366-3",
    alignedRefG: "ATG-C",
    alignedQueryG: "ATGTC",
    querySequence: "ATGTC",
    cdsStart: 1,
    cdsEnd: 5,
    tracesA: [0, 10],
    tracesT: [0, 8],
    tracesG: [0, 7],
    tracesC: [0, 5],
    baseLocations: [0, 1],
    mixedPeaks: [1],
    mutations: [{ refPos: 4, refBase: "-", queryBase: "T", type: "insertion" }],
  }, 0, "en");

  assert.equal(sample.id, "C366-3");
  assert.equal(sample.aligned_ref_g, "ATG-C");
  assert.equal(sample.aligned_query_g, "ATGTC");
  assert.equal(sample.query_sequence, "ATGTC");
  assert.equal(sample.cds_start, 1);
  assert.equal(sample.cds_end, 5);
  assert.deepEqual(sample.traces_a, [0, 10]);
  assert.deepEqual(sample.base_locations, [0, 1]);
  assert.deepEqual(sample.mutations, [{ position: 4, refBase: "-", queryBase: "T", type: "insertion", effect: undefined }]);
});

test("normalizeSamples reads detail samples when direct samples are absent", () => {
  const samples = normalizeSamples({ detail: { samples: [{ id: "S1", identity: 1, coverage: 1 }] } }, "en");
  assert.equal(samples.length, 1);
  assert.equal(samples[0].id, "S1");
  assert.equal(samples[0].status, "ok");
});

test("normalizeSample derives status from camelCase coverage alias", () => {
  const sample = normalizeSample({ id: "S2", identity: 1, cdsCoverage: 1 }, 0, "en");
  assert.equal(sample.status, "ok");
  assert.equal(sample.reason, "High-quality match");
  assert.equal(sample.cds_coverage, 1);
});

test("buildChromatogramData returns null when required traces or query sequence are missing", () => {
  assert.equal(buildChromatogramData({ id: "S1", query_sequence: "AT" }), null);
});

test("buildChromatogramData creates current ChromatogramData shape", () => {
  const chrom = buildChromatogramData({
    id: "S1",
    query_sequence: "AT",
    traces_a: [1, 2],
    traces_t: [3, 4],
    traces_g: [5, 6],
    traces_c: [7, 8],
    quality: [40, 35],
    base_locations: [0, 1],
    mixed_peaks: [1],
  });

  assert.deepEqual(chrom, {
    traces: { A: [1, 2], T: [3, 4], G: [5, 6], C: [7, 8] },
    quality: [40, 35],
    baseCalls: "AT",
    base_locations: [0, 1],
    mixed_peaks: [1],
  });
});
